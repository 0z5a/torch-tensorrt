# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import json
from pathlib import Path

import pytest
from torch_tensorrt.agentic import (
    Budget,
    Constraints,
    Goal,
    Measurement,
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
# The fake workload sleeps, so its samples spread widely on a loaded machine;
# these tests script decisions, not stability, which has its own tests below.
SINGLE_RUN = Measurement(calibration_runs=0, race_pairs=0, quiet_gpu_s=0, max_spread=10)


@pytest.fixture
def jobs(monkeypatch):
    """Run worker jobs in this process and record them.

    Set ``jobs.slow_best_confirmations`` to make the best arm look slower in
    the final confirmation.
    """

    class Jobs(list):
        slow_best_confirmations = False
        slow_race_candidates = False
        unstable: set = set()

    recorded = Jobs()

    def execute(self, request, directory):
        name = directory.relative_to(self.directory).as_posix()
        recorded.append((name, request))
        try:
            result = _trial.run(request)
        except Exception as error:  # noqa: BLE001 - like the worker's own handler
            result = {"status": "failed", "stage": "setup", "error": str(error)}
        slow = (
            recorded.slow_best_confirmations
            and directory.parent.name == "confirm"
            and directory.name.endswith("best")
        ) or (
            recorded.slow_race_candidates
            and directory.parent.name == "race"
            and directory.name.endswith("candidate")
        )
        if slow and result.get("status") == "ok":
            result["latency_ms"] = {k: v * 3 for k, v in result["latency_ms"].items()}
            result["throughput"] /= 3
        if name in recorded.unstable and result.get("status") == "ok":
            result["measurement"] = {
                "attempts": 3,
                "spreads": [0.2, 0.2, 0.2],
                "gpu_wait_s": 0.0,
                "gpu_quiet": False,
                "stable": False,
            }
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
    # These tests script the worker sequence one run per trial; calibration and
    # racing are covered on their own below.
    options.setdefault("measurement", SINGLE_RUN)
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


def calibrated(directory, kwargs=None, **options):
    options.setdefault(
        "measurement",
        Measurement(calibration_runs=2, race_pairs=2, quiet_gpu_s=0, max_spread=10),
    )
    return session(directory, kwargs, **options)


def test_calibration_sets_the_threshold_and_racing_confirms_gains(tmp_path, jobs):
    result = calibrated(
        tmp_path / "s",
        {"speedups": {"runtime.cudagraphs=True": 0.5}},
        budget=Budget(max_trials=2),
    ).run()
    assert [name for name, _ in jobs] == [
        "reference",
        "trials/0000",
        "calibration/0",
        "calibration/1",
        "trials/0001",
        "trials/0002",
        # Interleaved pairs, alternating which side runs first.
        "trials/0002/race/0-incumbent",
        "trials/0002/race/0-candidate",
        "trials/0002/race/1-candidate",
        "trials/0002/race/1-incumbent",
        "confirm/0-baseline",
        "confirm/0-best",
        "export",
    ]
    calibration = json.loads(
        (tmp_path / "s" / "calibration" / "summary.json").read_text()
    )
    assert len(calibration["metrics"]) == 3
    # The fake workload sleeps, so its spread depends on the machine's load;
    # the threshold is the goal's 20% or twice that spread, whichever is larger.
    spread = calibration["spread"]
    assert spread == pytest.approx(
        (max(calibration["metrics"]) - min(calibration["metrics"]))
        / sorted(calibration["metrics"])[1]
    )
    assert calibration["threshold"] == pytest.approx(max(0.2, 2 * spread))
    assert result.calibration["threshold"] == calibration["threshold"]
    decision = json.loads(
        (tmp_path / "s" / "trials" / "0002" / "decision.json").read_text()
    )
    assert decision["decision"] == "promoted"
    assert decision["reason"].startswith("raced: median paired gain")
    assert len(decision["race"]["gains"]) == 2
    assert min(decision["race"]["gains"]) > 0.3
    assert result.decision == "promoted"
    ledger = (tmp_path / "s" / "ledger.tsv").read_text().splitlines()
    assert ledger[0].split("\t")[:4] == ["trial", "parent", "decision", "metric"]
    assert [row.split("\t")[2] for row in ledger[1:]] == [
        "baseline",
        "rejected",
        "promoted",
    ]
    paired_gain = float(ledger[3].split("\t")[4].rstrip("%")) / 100
    assert paired_gain == pytest.approx(0.5, abs=0.1)


def test_racing_rejects_a_gain_that_does_not_hold_in_paired_runs(tmp_path, jobs):
    jobs.slow_race_candidates = True
    result = calibrated(
        tmp_path / "s",
        {"speedups": {"runtime.cudagraphs=True": 0.5}},
        budget=Budget(max_trials=2),
    ).run()
    decision = json.loads(
        (tmp_path / "s" / "trials" / "0002" / "decision.json").read_text()
    )
    assert decision["decision"] == "rejected"
    assert decision["reason"].startswith("not confirmed in paired runs")
    assert decision["screen"].startswith("latency_ms improved")
    assert result.decision == "baseline retained"


def test_unstable_measurements_are_never_promoted(tmp_path, jobs):
    jobs.unstable = {"trials/0002"}
    calibrated(
        tmp_path / "s",
        {"speedups": {"runtime.cudagraphs=True": 0.5}},
        budget=Budget(max_trials=2),
    ).run()
    decision = json.loads(
        (tmp_path / "s" / "trials" / "0002" / "decision.json").read_text()
    )
    assert decision["decision"] == "rejected"
    assert decision["reason"].startswith("unstable measurement: interquartile spread")
    assert "the GPU was busy" in decision["reason"]


def test_resume_reuses_the_calibration(tmp_path, jobs):
    calibrated(tmp_path / "s", budget=Budget(max_trials=1)).run()
    first = [name for name, _ in jobs]
    assert first.count("calibration/0") == 1
    jobs.clear()
    calibrated(tmp_path / "s", budget=Budget(max_trials=2)).run()
    assert not any(name.startswith("calibration/") for name, _ in jobs)


def test_the_evaluator_retimes_unstable_samples():
    from torch_tensorrt.agentic._evaluator import evaluate
    from torch_tensorrt.agentic._spec import Timing
    from torch_tensorrt.agentic._tools import catalog, resolve

    from ._cases import FAKE_TOOLS, FakeCase

    values = resolve({}, catalog(FAKE_TOOLS))
    timing = Timing(warmup=0, rounds=5, calls=1, profile_calls=0)
    settings = {"quiet_gpu_s": 0, "max_spread": 0.1}
    # The accuracy check takes call 0; calls 1-5 (the first attempt) wobble.
    # 20 ms calls keep sleep jitter on a busy machine well inside the 10% limit.
    case = FakeCase(milliseconds=20.0, wobble_calls=6)
    retimed = evaluate(
        case, values, case.reference(), timing, measurement={**settings, "retries": 2}
    )
    assert retimed["measurement"]["attempts"] == 2
    assert retimed["measurement"]["stable"]
    assert retimed["measurement"]["spreads"][0] > 0.1
    case = FakeCase(milliseconds=20.0, wobble_calls=6)
    once = evaluate(
        case, values, case.reference(), timing, measurement={**settings, "retries": 0}
    )
    assert once["measurement"]["attempts"] == 1
    assert not once["measurement"]["stable"]


class PatchScript(Planner):
    """Return the given proposals in order, then stop."""

    name = "patch-script"

    def __init__(self, proposals):
        self.proposals = list(proposals)
        self.evidence = []

    def propose(self, evidence, workdir):
        self.evidence.append(evidence)
        if not self.proposals:
            return Proposal(stop=True, rationale="script done")
        return self.proposals.pop(0)


def test_code_patches_are_validated_stored_measured_and_promoted(tmp_path, jobs):
    planner = PatchScript(
        [
            Proposal(
                patches={
                    "speed": [{"search": "SPEED = 1.0", "replace": "SPEED = 0.4"}]
                },
                rationale="halve the time",
            ),
            # Outside the fence: refused before anything runs.
            Proposal(
                patches={"speed": [{"search": "fenced factor", "replace": "factor"}]}
            ),
            Proposal(patches={"nope": [{"search": "x", "replace": "y"}]}),
            Proposal(
                patches={"speed": [{"search": "SPEED = 0.4", "replace": "SPEED = ("}]}
            ),
        ]
    )
    result = session(
        tmp_path / "s",
        {"slot": True},
        planner=planner,
        budget=Budget(max_trials=4),
        brief="Try code before knobs.",
    ).run()
    root = tmp_path / "s"
    assert decisions(root) == ["baseline", "promoted", "invalid", "invalid", "invalid"]
    reasons = [
        json.loads((root / "trials" / f"{n:04d}" / "decision.json").read_text())[
            "reason"
        ]
        for n in (2, 3, 4)
    ]
    assert "not inside an EVOLVE-BLOCK region" in reasons[0]
    assert "Unknown patch slot 'nope'" in reasons[1]
    assert "does not compile" in reasons[2]
    sha = result.recipe["patch.speed"]
    assert (root / "patches" / f"{sha}.py").read_text().count("SPEED = 0.4") == 1
    assert result.decision == "promoted"
    assert result.confirmation["median_gain"] == pytest.approx(0.6, abs=0.1)
    # The planner saw the slot, then the incumbent's patched source, and the brief.
    first, second = planner.evidence[0], planner.evidence[1]
    assert first["patch_slots"][0]["editable_blocks"] == ["SPEED = 1.0\n"]
    assert second["patch_slots"][0]["editable_blocks"] == ["SPEED = 0.4\n"]
    assert second["patch_slots"][0]["patched"] == sha
    assert first["brief"] == "Try code before knobs."


def test_a_retry_counts_once_the_gpu_went_quiet(monkeypatch):
    from torch_tensorrt.agentic import _evaluator
    from torch_tensorrt.agentic._spec import Timing
    from torch_tensorrt.agentic._tools import catalog, resolve

    from ._cases import FAKE_TOOLS, FakeCase

    gates = []

    def gate(max_wait_s):
        gates.append(max_wait_s)
        if len(gates) == 1:
            return {"waited_s": 7.0, "quiet": False}
        return {"waited_s": 1.0, "quiet": True}

    monkeypatch.setattr(_evaluator, "wait_for_quiet_gpu", gate)
    values = resolve({}, catalog(FAKE_TOOLS))
    timing = Timing(warmup=0, rounds=5, calls=1, profile_calls=0)
    settings = {"quiet_gpu_s": 10, "max_spread": 10}
    case = FakeCase(milliseconds=2.0)
    result = _evaluator.evaluate(
        case, values, case.reference(), timing, measurement={**settings, "retries": 2}
    )
    measurement = result["measurement"]
    assert measurement["attempts"] == 2
    assert measurement["gpu_quiet"] is True and measurement["stable"]
    # quiet_gpu_s bounds the whole wait; once spent, an attempt still looks briefly.
    assert gates == [10, 3.0]
    gates.clear()
    case = FakeCase(milliseconds=2.0)
    once = _evaluator.evaluate(
        case, values, case.reference(), timing, measurement={**settings, "retries": 0}
    )
    assert once["measurement"]["gpu_quiet"] is False
    assert not once["measurement"]["stable"]


def test_a_calibration_is_redone_when_its_settings_change(tmp_path, jobs):
    session(tmp_path / "s", budget=Budget(max_trials=1)).run()
    assert not any(name.startswith("calibration/") for name, _ in jobs)
    jobs.clear()
    calibrated(tmp_path / "s", budget=Budget(max_trials=1)).run()
    names = [name for name, _ in jobs]
    assert names.count("calibration/0") == names.count("calibration/1") == 1
    summary = json.loads((tmp_path / "s/calibration/summary.json").read_text())
    assert summary["requested"] == 2 and summary["spread"] is not None


def test_patch_changes_must_name_a_declared_slot_and_a_stored_patch(tmp_path, jobs):
    planner = PatchScript(
        [
            Proposal(changes={"patch.speed": "0" * 64}),
            Proposal(changes={"patch.nope": ""}),
        ]
    )
    session(tmp_path / "s", {"slot": True}, planner=planner).run()
    root = tmp_path / "s"
    assert decisions(root) == ["baseline", "invalid", "invalid"]
    reasons = [
        json.loads((root / "trials" / f"{n:04d}" / "decision.json").read_text())[
            "reason"
        ]
        for n in (1, 2)
    ]
    assert "names no stored patch" in reasons[0]
    assert "Unknown patch slot 'patch.nope'" in reasons[1]


def test_a_patch_that_restores_the_original_is_the_original(tmp_path, jobs):
    planner = PatchScript(
        [
            Proposal(
                patches={"speed": [{"search": "SPEED = 1.0", "replace": "SPEED = 0.4"}]}
            ),
            Proposal(
                patches={"speed": [{"search": "SPEED = 0.4", "replace": "SPEED = 1.0"}]}
            ),
        ]
    )
    session(tmp_path / "s", {"slot": True}, planner=planner).run()
    root = tmp_path / "s"
    assert decisions(root) == ["baseline", "promoted", "invalid"]
    reason = json.loads((root / "trials/0002/decision.json").read_text())["reason"]
    assert "already evaluated" in reason


def test_a_resumed_session_forgets_earlier_invalid_proposals(tmp_path, jobs):
    bad = [Proposal(changes={"engine.turbo": True}) for _ in range(3)]
    first = session(tmp_path / "s", planner=Scripted(bad)).run()
    assert first.stop_reason == "the planner made 3 invalid proposals in a row"
    asked = PatchScript([])
    second = session(tmp_path / "s", planner=asked).run()
    assert len(asked.evidence) == 1
    assert second.stop_reason == "the planner stopped: script done"


def test_an_edited_program_md_survives_a_resume(tmp_path, jobs):
    planner = Scripted([])
    session(tmp_path / "s", planner=planner, brief="Try A.").run()
    program = tmp_path / "s" / "program.md"
    assert program.read_text() == "Try A."
    program.write_text("Try B instead.")
    session(tmp_path / "s", planner=Scripted([]), brief="Try A.").run()
    assert program.read_text() == "Try B instead."
    session(tmp_path / "s", planner=Scripted([]), brief="Try C.").run()
    assert program.read_text() == "Try C."
