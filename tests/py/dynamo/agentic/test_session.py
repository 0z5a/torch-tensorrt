# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import json
from pathlib import Path

import pytest

from torch_tensorrt.agentic import (
    Budget,
    Constraints,
    Goal,
    OptimizationSession,
    Planner,
    Proposal,
    SessionError,
    Workload,
)
from torch_tensorrt.agentic import __main__ as cli
from torch_tensorrt.agentic import (
    _trial,
)

FACTORY = f"{Path(__file__).with_name('_cases.py')}:create_fake"


@pytest.fixture
def jobs(monkeypatch):
    """Run worker jobs in this process and record them.

    Set ``jobs.slow_best_confirmations`` to make the best arm look slower in
    the final confirmation.
    """

    class Jobs(list):
        slow_best_confirmations = False

    recorded = Jobs()

    def execute(self, request, directory):
        recorded.append((directory.relative_to(self.directory).as_posix(), request))
        try:
            result = _trial.run(request)
        except Exception as error:  # noqa: BLE001 - like the worker's own handler
            result = {"status": "failed", "stage": "setup", "error": str(error)}
        if (
            recorded.slow_best_confirmations
            and directory.parent.name == "confirm"
            and directory.name.endswith("best")
            and result.get("status") == "ok"
        ):
            result["latency_ms"] = {k: v * 3 for k, v in result["latency_ms"].items()}
            result["throughput"] /= 3
        _trial.write_json(Path(request["result_path"]), result)
        return result

    monkeypatch.setattr(OptimizationSession, "_execute", execute)
    return recorded


def session(directory, kwargs=None, **options):
    # The fake workload sleeps, so its timings jitter by a few percent on a
    # busy machine; scripted speedups are 2x or more, noise never 20%.
    options.setdefault("goal", Goal(min_improvement=0.2))
    options.setdefault("budget", Budget(max_trials=3))
    options.setdefault("confirm_repeats", 1)
    return OptimizationSession(
        Workload(FACTORY, kwargs or {}), directory=directory, **options
    )


def decisions(directory):
    return [
        json.loads(path.read_text())["decision"]
        for path in sorted(Path(directory).glob("trials/*/decision.json"))
    ]


def test_session_promotes_measured_gains_and_keeps_the_rest(tmp_path, jobs):
    result = session(
        tmp_path / "s", {"speedups": {"runtime.cudagraphs=True": 0.5}}
    ).run()
    # precision (signal) first, then runtime: only the CUDA-graph change pays.
    assert decisions(tmp_path / "s") == ["baseline", "rejected", "promoted", "rejected"]
    assert result.decision == "promoted"
    assert result.recipe == {"runtime.cudagraphs": True}
    assert result.stop_reason == "trial budget of 3 used"
    assert result.confirmation["confirmed"]
    assert result.confirmation["median_gain"] == pytest.approx(0.5, abs=0.1)
    assert sorted(path.name for path in result.artifact.iterdir()) == [
        "fake.txt",
        "recipe.json",
    ]
    assert json.loads((result.artifact / "recipe.json").read_text()) == result.recipe
    report = result.report()
    for section in ("## Before / after", "## Recipe", "## Trials", "## Profiles"):
        assert section in report
    assert "promoted trial 0002" in report
    assert [name for name, _ in jobs] == [
        "reference",
        "trials/0000",
        "trials/0001",
        "trials/0002",
        "trials/0003",
        "confirm/0-baseline",
        "confirm/0-best",
        "export",
    ]
    evidence = json.loads(
        (tmp_path / "s" / "trials" / "0003" / "evidence.json").read_text()
    )
    assert evidence["incumbent"]["trial"] == "0002"
    assert [entry["decision"] for entry in evidence["history"]] == [
        "rejected",
        "promoted",
    ]


def test_inaccurate_and_broken_candidates_are_rejected_with_reasons(tmp_path, jobs):
    session(
        tmp_path / "s",
        {
            "speedups": {"engine.precision=fp16": 0.3, "runtime.cudagraphs=True": 0.4},
            "inaccurate": ["engine.precision=fp16"],
            "broken": ["runtime.cudagraphs=True"],
        },
        budget=Budget(max_trials=2),
    ).run()
    first = json.loads((tmp_path / "s/trials/0001/decision.json").read_text())
    second = json.loads((tmp_path / "s/trials/0002/decision.json").read_text())
    result = json.loads((tmp_path / "s/trials/0002/result.json").read_text())
    assert first["decision"] == "rejected"
    assert first["reason"].startswith("accuracy: max_abs_error=0.5")
    assert second == {
        "decision": "rejected",
        "reason": "build failed: RuntimeError: cannot build ['runtime.cudagraphs=True']",
    }
    assert result["status"] == "failed" and result["stage"] == "build"


def test_session_stops_at_the_target(tmp_path, jobs):
    result = session(
        tmp_path / "s",
        {"speedups": {"engine.precision=fp16": 0.25}},
        goal=Goal(target=1.0, min_improvement=0.2),
        budget=Budget(max_trials=5),
    ).run()
    assert result.stop_reason == "goal target reached"
    assert decisions(tmp_path / "s") == ["baseline", "promoted"]


def test_session_stops_after_patience_trials_without_promotion(tmp_path, jobs):
    result = session(tmp_path / "s", budget=Budget(max_trials=5, patience=2)).run()
    assert result.stop_reason == "no promotion in the last 2 trials"
    assert result.decision == "baseline retained"
    assert result.confirmation is None
    assert decisions(tmp_path / "s") == ["baseline", "rejected", "rejected"]
    assert "baseline retained (no candidate was promoted)" in result.report()


class Scripted(Planner):
    name = "scripted"

    def __init__(self, proposals):
        self.proposals = list(proposals)

    def propose(self, evidence, workdir):
        return (
            self.proposals.pop(0)
            if self.proposals
            else Proposal(stop=True, rationale="done")
        )


def test_invalid_proposals_are_recorded_and_end_the_search(tmp_path, jobs):
    planner = Scripted(
        [
            Proposal(changes={"engine.turbo": True}),
            Proposal(changes={"engine.optimization_level": 4}),
            Proposal(changes={"workload.variant": "a"}),
        ]
    )
    result = session(tmp_path / "s", planner=planner, budget=Budget(max_trials=5)).run()
    assert result.stop_reason == "the planner made 3 invalid proposals in a row"
    reasons = [
        json.loads(path.read_text())["reason"]
        for path in sorted((tmp_path / "s/trials").glob("*/decision.json"))[1:]
    ]
    assert reasons[0].startswith("proposal rejected: Unknown tool 'engine.turbo'")
    assert reasons[1].startswith(
        "proposal rejected: Tool engine.optimization_level does not allow 4"
    )
    assert reasons[2] == "proposal rejected: The proposal changes nothing" or reasons[
        2
    ].startswith("proposal rejected: this recipe was already evaluated")
    # Invalid proposals spend no evaluation.
    assert [name for name, _ in jobs if name.startswith("trials/")] == ["trials/0000"]


def test_planner_stop_ends_the_search(tmp_path, jobs):
    result = session(tmp_path / "s", planner=Scripted([])).run()
    assert result.stop_reason == "the planner stopped: done"
    assert result.decision == "baseline retained"


def test_unconfirmed_best_keeps_the_baseline(tmp_path, jobs):
    jobs.slow_best_confirmations = True
    result = session(
        tmp_path / "s",
        {"speedups": {"engine.precision=fp16": 0.5}},
        budget=Budget(max_trials=1),
    ).run()
    assert decisions(tmp_path / "s") == ["baseline", "promoted"]
    assert not result.confirmation["confirmed"]
    assert result.decision == "baseline retained"
    assert result.recipe == {}
    assert "NOT confirmed" in result.report()


def test_resume_reuses_recorded_trials_and_extends_the_budget(tmp_path, jobs):
    kwargs = {"speedups": {"engine.precision=fp16": 0.5}}
    session(tmp_path / "s", kwargs, budget=Budget(max_trials=1)).run()
    first = [name for name, _ in jobs]
    jobs.clear()
    result = session(tmp_path / "s", kwargs, budget=Budget(max_trials=3)).run()
    # The case is described again, but recorded trials are not re-measured.
    assert first[:2] == ["reference", "trials/0000"]
    assert [name for name, _ in jobs][:3] == ["describe", "trials/0002", "trials/0003"]
    assert decisions(tmp_path / "s") == ["baseline", "promoted", "rejected", "rejected"]
    assert result.recipe == {"engine.precision": "fp16"}
    resumed = OptimizationSession.resume(tmp_path / "s")
    assert resumed.workload.kwargs == kwargs
    assert resumed.budget == Budget(max_trials=3)


def test_an_interrupted_trial_is_evaluated_again(tmp_path, jobs):
    session(tmp_path / "s", budget=Budget(max_trials=1)).run()
    (tmp_path / "s/trials/0001/decision.json").unlink()
    jobs.clear()
    session(tmp_path / "s", budget=Budget(max_trials=1)).run()
    assert [name for name, _ in jobs][:2] == ["describe", "trials/0001"]


def test_a_directory_holds_one_session(tmp_path, jobs):
    session(tmp_path / "s", budget=Budget(max_trials=0)).run()
    with pytest.raises(SessionError, match="differs in goal"):
        session(tmp_path / "s", goal=Goal(min_improvement=0.1)).run()
    with pytest.raises(SessionError, match="differs in workload"):
        session(tmp_path / "s", {"milliseconds": 1.0}).run()


def test_invalid_baselines_and_references_end_the_session(tmp_path, jobs):
    with pytest.raises(SessionError, match="baseline is not valid.*build failed"):
        session(
            tmp_path / "a",
            {
                "broken": ["workload.variant=b"],
                "baseline_changes": {"workload.variant": "b"},
            },
        ).run()
    with pytest.raises(SessionError, match="no reference today"):
        session(tmp_path / "b", {"reference_fails": True}).run()
    with pytest.raises(SessionError, match="accuracy"):
        session(
            tmp_path / "c",
            {"inaccurate": ["workload.variant=c"]},
            baseline={"workload.variant": "c"},
        ).run()
    # Unknown memory (no CUDA) cannot satisfy a limit either.
    with pytest.raises(SessionError, match="memory .* MB exceeds max_memory_mb=1"):
        session(tmp_path / "d", constraints=Constraints(max_memory_mb=1.0)).run()


def test_judge_needs_a_gain_beyond_the_quartiles(tmp_path):
    judge = session(tmp_path / "s", goal=Goal()).judge
    bounds = {"top1": (">=", 0.9)}

    def result(median, spread, top1=1.0):
        latency = {"median": median, "p25": median - spread, "p75": median + spread}
        return {
            "status": "ok",
            "accuracy": {"top1": top1},
            "accuracy_after_timing": {"top1": top1},
            "latency_ms": latency,
            "throughput": 1000 / median,
            "items": 1,
        }

    incumbent = result(10.0, 0.1)
    assert judge(result(9.0, 0.1), incumbent, bounds)[0]
    # A 10% gain with overlapping quartiles is noise.
    assert not judge(result(9.0, 1.5), incumbent, bounds)[0]
    # Separated, but below min_improvement (2%).
    assert not judge(result(9.9, 0.01), result(10.0, 0.01), bounds)[0]
    accepted, reason = judge(result(5.0, 0.1, top1=0.5), incumbent, bounds)
    assert (
        not accepted
        and reason
        == "accuracy: top1=0.5 (needs >= 0.9); after timing: top1=0.5 (needs >= 0.9)"
    )


def test_cli_reports_status_and_the_report(tmp_path, jobs, capsys):
    session(
        tmp_path / "s",
        {"speedups": {"engine.precision=fp16": 0.5}},
        budget=Budget(max_trials=1),
    ).run()
    assert cli.main(["status", str(tmp_path / "s")]) == 0
    status = capsys.readouterr().out
    assert (
        "0001   ok" in status
        and "promoted" in status
        and '{"engine.precision": "fp16"}' in status
    )
    assert cli.main(["report", str(tmp_path / "s")]) == 0
    assert "# Optimization session report" in capsys.readouterr().out
    assert cli.main(["status", str(tmp_path)]) == 1
