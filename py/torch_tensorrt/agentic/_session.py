# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""The session controller: budgets, checkpoints, acceptance and the final artifact."""

from __future__ import annotations

import json
import logging
import os
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from torch_tensorrt.agentic._planner import (
    HeuristicPlanner,
    Planner,
    planner_from_json,
    validate_changes,
)
from torch_tensorrt.agentic._report import write_report
from torch_tensorrt.agentic._spec import (
    Budget,
    Constraints,
    Goal,
    Recipe,
    Timing,
    Workload,
    digest,
    violations,
)
from torch_tensorrt.agentic._tools import Tool, apply, catalog, difference
from torch_tensorrt.agentic._trial import write_json

logger = logging.getLogger(__name__)

_VERSION = 1
# Consecutive invalid proposals (unknown tools, disallowed values, repeats)
# after which the session stops asking the planner.
_MAX_INVALID_PROPOSALS = 3


class SessionError(RuntimeError):
    """The session cannot continue, for example because the baseline failed."""


@dataclass(frozen=True)
class SessionResult:
    """The outcome of a session.

    ``decision`` is ``"promoted"`` when a candidate beat the baseline in the
    final confirmation, otherwise ``"baseline retained"``. ``recipe`` is the
    winner's recipe and ``artifact`` the directory it was exported to (``None``
    if the workload cannot export). Memory is the process's GPU memory from
    NVML where available, otherwise PyTorch's peak allocation, which does not
    include TensorRT's own allocations.
    """

    directory: Path
    decision: str
    stop_reason: str
    recipe: Recipe
    baseline: dict[str, Any]
    best: dict[str, Any]
    confirmation: dict[str, Any] | None
    artifact: Path | None
    report_path: Path

    def report(self) -> str:
        return self.report_path.read_text()


def metric_value(result: Mapping[str, Any], goal: Goal) -> float:
    if goal.metric == "throughput":
        return float(result["throughput"])
    return float(result["latency_ms"]["median"])


def _separated(candidate: Mapping[str, Any], incumbent: Mapping[str, Any]) -> bool:
    """Whether the candidate's latency quartiles lie entirely below the incumbent's.

    With equal items per unit this also decides throughput; otherwise compare
    the throughput implied by the quartiles.
    """
    c, i = candidate["latency_ms"], incumbent["latency_ms"]
    c_items, i_items = candidate.get("items", 1), incumbent.get("items", 1)
    if c_items == i_items:
        return bool(c["p75"] < i["p25"])
    return bool(c_items / c["p75"] > i_items / i["p25"])


def memory_mb(result: Mapping[str, Any]) -> float | None:
    memory = result.get("memory") or {}
    value = memory.get("process_gpu_mb")
    return value if value is not None else memory.get("torch_peak_mb")


class OptimizationSession:
    """Optimize one workload toward a goal within constraints and a budget.

    The session records a validated baseline (the workload's baseline recipe,
    updated by ``baseline``), then repeats: the planner reads the evidence and
    proposes changes to the best recipe so far, the session validates them
    against the workload's tools, and a fresh worker process builds, checks,
    measures and profiles the candidate. A candidate is promoted only if it
    passes the accuracy and memory constraints and improves the goal metric by
    more than ``goal.min_improvement`` with non-overlapping quartiles. The
    search stops at the goal's target, at the end of the budget, after
    ``budget.patience`` trials without a promotion, or when the planner stops.

    Finally the best candidate is confirmed against the baseline in
    ``confirm_repeats`` fresh baseline/best worker pairs run back to back; it
    is kept only if the median paired gain still exceeds
    ``goal.min_improvement``. The winner is exported to ``directory/artifact``
    with its recipe and the report.

    Every step is recorded under ``directory``. Running a session again in the
    same directory resumes it: completed trials are reused, an interrupted
    trial is evaluated again, and the final confirmation always measures anew.
    """

    def __init__(
        self,
        workload: Workload,
        *,
        directory: str | os.PathLike[str],
        goal: Goal | None = None,
        constraints: Constraints | None = None,
        budget: Budget | None = None,
        planner: Planner | None = None,
        baseline: Recipe | None = None,
        timing: Timing | None = None,
        confirm_repeats: int = 2,
        export: bool = True,
        python: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(workload, Workload):
            raise TypeError(
                "OptimizationSession needs a torch_tensorrt.agentic.Workload"
            )
        if not isinstance(confirm_repeats, int) or confirm_repeats < 1:
            raise ValueError("confirm_repeats must be a positive integer")
        self.workload = workload
        self.directory = Path(directory).expanduser().resolve()
        self.goal = goal or Goal()
        self.constraints = constraints or Constraints()
        self.budget = budget or Budget()
        self.planner = planner or HeuristicPlanner()
        self.baseline_changes = dict(baseline or {})
        self.timing = timing
        self.confirm_repeats = confirm_repeats
        self.export = export
        self.python = python or sys.executable
        self.env = dict(env or {})
        self.cwd = os.getcwd()

    # ----------------------------------------------------------------- setup

    @classmethod
    def resume(
        cls,
        directory: str | os.PathLike[str],
        *,
        planner: Planner | None = None,
        budget: Budget | None = None,
    ) -> "OptimizationSession":
        """Rebuild the session recorded in ``directory``, optionally with a new budget."""
        directory = Path(directory).expanduser().resolve()
        config = json.loads((directory / "session.json").read_text())
        constraints = config["constraints"]
        session = cls(
            Workload(**config["workload"]),
            directory=directory,
            goal=Goal(**config["goal"]),
            constraints=Constraints(
                accuracy={
                    name: tuple(bound)
                    for name, bound in constraints["accuracy"].items()
                },
                max_memory_mb=constraints["max_memory_mb"],
            ),
            budget=budget or Budget(**config["budget"]),
            planner=planner or planner_from_json(config["planner"]),
            baseline=config["baseline_changes"],
            timing=Timing(**config["timing"]) if config.get("timing") else None,
            confirm_repeats=config["confirm_repeats"],
            export=config["export"],
            python=config.get("python"),
        )
        session.cwd = config.get("cwd", session.cwd)
        return session

    def _identity(self) -> dict[str, Any]:
        # What makes two runs the same session; the budget, planner and
        # Python executable may change between resumes.
        return {
            "version": _VERSION,
            "workload": self.workload.to_json(),
            "goal": asdict(self.goal),
            "constraints": self.constraints.to_json(),
            "baseline_changes": self.baseline_changes,
            "timing": asdict(self.timing) if self.timing else None,
        }

    def _open(self) -> dict[str, Any]:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "session.json"
        identity = self._identity()
        config: dict[str, Any]
        if path.is_file():
            recorded: dict[str, Any] = json.loads(path.read_text())
            mismatched = [
                name for name in identity if recorded.get(name) != identity[name]
            ]
            if mismatched:
                raise SessionError(
                    f"{self.directory} holds a different session (it differs in "
                    f"{', '.join(mismatched)}); use another directory"
                )
            config = recorded
        else:
            config = {**identity, "elapsed_s": 0.0, "created": time.time()}
        config.update(
            budget=asdict(self.budget),
            planner=self.planner.to_json(),
            confirm_repeats=self.confirm_repeats,
            export=self.export,
            python=self.python,
            cwd=self.cwd,
        )
        write_json(path, config)
        return config

    def _event(self, event: str, **fields: Any) -> None:
        line = json.dumps({"time": time.time(), "event": event, **fields}, default=str)
        with open(self.directory / "events.jsonl", "a") as stream:
            stream.write(line + "\n")

    # --------------------------------------------------------------- workers

    def _run_worker(self, request: dict[str, Any], directory: Path) -> dict[str, Any]:
        """Run one job in a fresh worker process; return its result."""
        directory.mkdir(parents=True, exist_ok=True)
        result_path = directory / "result.json"
        result_path.unlink(missing_ok=True)
        request = {
            **request,
            "workload": self.workload.to_json(),
            "reference_path": str(self.directory / "reference.pt"),
            "result_path": str(result_path),
        }
        write_json(directory / "request.json", request)
        return self._execute(request, directory)

    def _execute(self, request: dict[str, Any], directory: Path) -> dict[str, Any]:
        """Run ``directory/request.json`` in a fresh Python process."""
        result_path = Path(request["result_path"])
        command = [
            self.python,
            "-P",
            "-m",
            "torch_tensorrt.agentic._trial",
            str(directory / "request.json"),
        ]
        environment = {**os.environ, **self.env}
        timeout = self.budget.trial_timeout_s
        with open(directory / "worker.log", "wb") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=self.cwd,
                env=environment,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                with_suppressed(os.killpg, process.pid, signal.SIGKILL)
                process.wait()
                return {
                    "status": "failed",
                    "stage": "timeout",
                    "error": f"The worker exceeded trial_timeout_s={timeout:g}",
                }
        if result_path.is_file():
            result: dict[str, Any] = json.loads(result_path.read_text())
            return result
        tail = (directory / "worker.log").read_text(errors="replace")[-1500:]
        return {
            "status": "failed",
            "stage": "crash",
            "error": f"The worker exited with status {returncode} without a result: {tail}",
        }

    def _prepare(self) -> dict[str, Any]:
        """Describe the case, computing its reference outputs once per session.

        A resumed session describes the case again, so a changed workload
        definition or environment is detected instead of mixed with old trials.
        """
        path = self.directory / "reference" / "result.json"
        have_reference = (
            path.is_file()
            and (self.directory / "reference.pt").is_file()
            and json.loads(path.read_text()).get("status") == "ok"
        )
        kind = "describe" if have_reference else "reference"
        result = self._run_worker({"kind": kind}, self.directory / kind)
        if result.get("status") != "ok":
            raise SessionError(
                f"Preparing the workload ({kind}) failed "
                f"({result.get('stage')}): {result.get('error')}"
            )
        self._event(kind, case=result["case"]["name"])
        return result

    # ---------------------------------------------------------------- trials

    def _trial_dir(self, number: int) -> Path:
        return self.directory / "trials" / f"{number:04d}"

    def _load_trials(self) -> list[dict[str, Any]]:
        trials = []
        root = self.directory / "trials"
        for directory in sorted(root.glob("[0-9][0-9][0-9][0-9]")):
            record_path = directory / "trial.json"
            if not record_path.is_file():
                continue
            record = json.loads(record_path.read_text())
            result_path = directory / "result.json"
            decision_path = directory / "decision.json"
            if decision_path.is_file() and result_path.is_file():
                record["result"] = json.loads(result_path.read_text())
                record["decision"] = json.loads(decision_path.read_text())
            elif decision_path.is_file():
                record["decision"] = json.loads(decision_path.read_text())
            trials.append(record)
        return trials

    def _evaluate(
        self, number: int, record: dict[str, Any], timing: Timing
    ) -> dict[str, Any]:
        directory = self._trial_dir(number)
        write_json(directory / "trial.json", record)
        self._event("trial", trial=record["id"], changes=record.get("changes"))
        return self._run_worker(
            {
                "kind": "trial",
                "recipe": record["recipe"],
                "timing": asdict(timing),
                "profile": True,
            },
            directory,
        )

    def judge(
        self,
        result: Mapping[str, Any],
        incumbent: Mapping[str, Any] | None,
        bounds: Mapping[str, tuple[str, float]],
    ) -> tuple[bool, str]:
        """Whether ``result`` qualifies, and replaces ``incumbent`` if one is given."""
        if result.get("status") != "ok":
            error = str(result.get("error", "")).strip().splitlines()
            return False, f"{result.get('stage')} failed: {error[-1] if error else '?'}"
        missed = violations(result["accuracy"], bounds) + [
            f"after timing: {miss}"
            for miss in violations(result["accuracy_after_timing"], bounds)
        ]
        if missed:
            return False, "accuracy: " + "; ".join(missed)
        memory = memory_mb(result)
        limit = self.constraints.max_memory_mb
        if limit is not None and (memory is None or memory > limit):
            return False, f"memory {memory} MB exceeds max_memory_mb={limit:g}"
        if incumbent is None:
            return True, "validated baseline"
        value = metric_value(result, self.goal)
        best = metric_value(incumbent, self.goal)
        gain = self.goal.gain(best, value)
        if gain > self.goal.min_improvement and _separated(result, incumbent):
            return (
                True,
                f"{self.goal.metric} improved {gain:+.1%} ({best:.4g} -> {value:.4g})",
            )
        return False, (
            f"no gain beyond noise: {self.goal.metric} {gain:+.1%} ({best:.4g} -> "
            f"{value:.4g}); needs more than {self.goal.min_improvement:.1%} with "
            "non-overlapping quartiles"
        )

    # -------------------------------------------------------------- evidence

    @staticmethod
    def summary(record: Mapping[str, Any], goal: Goal) -> dict[str, Any]:
        result = record.get("result") or {}
        summary: dict[str, Any] = {
            "trial": record["id"],
            "recipe": record["recipe"],
            "status": result.get("status"),
        }
        if result.get("status") == "ok":
            summary.update(
                metric=metric_value(result, goal),
                latency_ms={
                    key: result["latency_ms"][key] for key in ("median", "p25", "p75")
                },
                throughput=result["throughput"],
                accuracy=result["accuracy"],
                memory=result.get("memory"),
                build_s=result.get("build_s"),
            )
        else:
            summary.update(stage=result.get("stage"), error=result.get("error"))
        for key in ("structure", "profile", "signals"):
            if key in result:
                summary[key] = result[key]
        return summary

    def _evidence(
        self,
        tools: Mapping[str, Tool],
        trials: list[dict[str, Any]],
        incumbent: dict[str, Any],
        bounds: Mapping[str, tuple[str, float]],
        used: int,
        elapsed_s: float,
    ) -> dict[str, Any]:
        history = []
        for record in trials[1:]:
            result = record.get("result") or {}
            entry = {
                "trial": record["id"],
                "parent": record.get("parent"),
                "changes": record.get("changes", {}),
                "rationale": record.get("proposal", {}).get("rationale", ""),
                "status": result.get("status", "invalid"),
                "decision": record.get("decision", {}).get("decision"),
                "reason": record.get("decision", {}).get("reason"),
            }
            if result.get("status") == "ok":
                entry["metric"] = metric_value(result, self.goal)
                entry["signals"] = [s["signal"] for s in result.get("signals", ())]
            elif result:
                entry["stage"] = result.get("stage")
                entry["error"] = str(result.get("error", ""))[-500:]
            history.append(entry)
        current = incumbent["recipe"]
        return {
            "goal": asdict(self.goal),
            "constraints": {
                "accuracy": {name: list(bound) for name, bound in bounds.items()},
                "max_memory_mb": self.constraints.max_memory_mb,
            },
            "budget": {
                "trials_used": used,
                "trials_left": self.budget.max_trials - used,
                "time_left_s": (
                    None
                    if self.budget.max_time_s is None
                    else max(0.0, self.budget.max_time_s - elapsed_s)
                ),
            },
            "baseline": self.summary(trials[0], self.goal),
            "incumbent": {
                **self.summary(incumbent, self.goal),
                "signals": (incumbent.get("result") or {}).get("signals", []),
            },
            "history": history,
            "tools": [
                {**tool.to_json(), "current": current.get(name, tool.default)}
                for name, tool in tools.items()
            ],
            "tried": sorted(record["digest"] for record in trials),
        }

    # ------------------------------------------------------------------- run

    def run(self) -> SessionResult:
        """Run (or resume) the session and return its result."""
        config = self._open()
        started = time.monotonic()
        elapsed_before = float(config.get("elapsed_s", 0.0))

        def elapsed() -> float:
            return elapsed_before + time.monotonic() - started

        def checkpoint() -> None:
            config["elapsed_s"] = elapsed()
            write_json(self.directory / "session.json", config)

        prepared = self._prepare()
        case = prepared["case"]
        tools = catalog([Tool(**_tool_args(tool)) for tool in case["tools"]])
        for key, what in (
            ("case", "workload's tools, baseline or accuracy defaults"),
            ("environment", "software or GPU"),
        ):
            if key in config and config[key] != prepared[key]:
                raise SessionError(
                    f"The {what} changed since {self.directory} was created, so its "
                    f"recorded measurements no longer apply; start a new session "
                    f"directory ({key} was {config[key]}, now {prepared[key]})"
                )
            config[key] = prepared[key]
        timing = self.timing or Timing(**case["timing"])
        bounds = self.constraints.bounds(
            {name: tuple(bound) for name, bound in case["accuracy_defaults"].items()}
        )
        checkpoint()

        trials = self._load_trials()
        baseline_recipe = apply(case["baseline"], self.baseline_changes, tools)
        if not trials or trials[0].get("result") is None:
            record: dict[str, Any] = {
                "id": "0000",
                "parent": None,
                "recipe": baseline_recipe,
                "changes": {},
                "digest": digest(baseline_recipe),
            }
            record["result"] = self._evaluate(0, record, timing)
            accepted, reason = self.judge(record["result"], None, bounds)
            record["decision"] = {
                "decision": "baseline" if accepted else "failed",
                "reason": reason,
            }
            write_json(self._trial_dir(0) / "decision.json", record["decision"])
            trials = [record] + trials[1:]
            checkpoint()
        baseline = trials[0]
        if baseline["decision"]["decision"] != "baseline":
            raise SessionError(
                f"The baseline is not valid, so nothing can be compared with it: "
                f"{baseline['decision']['reason']}. See {self._trial_dir(0)}/worker.log"
            )

        # Replay recorded decisions; evaluate an interrupted trial again.
        incumbent = baseline
        used = invalid = stale = 0
        for number, record in enumerate(trials[1:], start=1):
            if "decision" not in record:
                record["result"] = self._evaluate(number, record, timing)
                accepted, reason = self.judge(
                    record["result"], incumbent["result"], bounds
                )
                record["decision"] = {
                    "decision": "promoted" if accepted else "rejected",
                    "reason": reason,
                }
                write_json(
                    self._trial_dir(number) / "decision.json", record["decision"]
                )
            decision = record["decision"]["decision"]
            if decision == "invalid":
                invalid += 1
                continue
            invalid = 0
            used += 1
            if decision == "promoted":
                incumbent, stale = record, 0
            else:
                stale += 1

        stop_reason = ""
        while not stop_reason:
            if self.goal.reached(metric_value(incumbent["result"], self.goal)):
                stop_reason = "goal target reached"
            elif used >= self.budget.max_trials:
                stop_reason = f"trial budget of {self.budget.max_trials} used"
            elif (
                self.budget.max_time_s is not None
                and elapsed() >= self.budget.max_time_s
            ):
                stop_reason = f"time budget of {self.budget.max_time_s:g} s used"
            elif self.budget.patience is not None and stale >= self.budget.patience:
                stop_reason = f"no promotion in the last {stale} trials"
            elif invalid >= _MAX_INVALID_PROPOSALS:
                stop_reason = f"the planner made {invalid} invalid proposals in a row"
            if stop_reason:
                break

            number = len(trials)
            directory = self._trial_dir(number)
            evidence = self._evidence(tools, trials, incumbent, bounds, used, elapsed())
            write_json(directory / "evidence.json", evidence)
            proposal = self.planner.propose(evidence, directory / "planner")
            proposal_json = {
                "changes": dict(proposal.changes),
                "rationale": proposal.rationale,
                "stop": proposal.stop,
                "planner": proposal.planner or self.planner.name,
                "note": proposal.note,
            }
            write_json(directory / "proposal.json", proposal_json)
            if proposal.stop:
                stop_reason = f"the planner stopped: {proposal.rationale}"
                self._event("stop", reason=stop_reason)
                break
            record = {
                "id": f"{number:04d}",
                "parent": incumbent["id"],
                "proposal": proposal_json,
            }
            try:
                recipe = validate_changes(proposal, incumbent["recipe"], tools)
                if digest(recipe) in {r["digest"] for r in trials}:
                    raise ValueError("this recipe was already evaluated")
            except ValueError as error:
                record.update(
                    recipe=incumbent["recipe"],
                    changes=dict(proposal.changes),
                    digest=f"invalid-{number}",
                    decision={
                        "decision": "invalid",
                        "reason": f"proposal rejected: {error}",
                    },
                )
                write_json(directory / "trial.json", record)
                write_json(directory / "decision.json", record["decision"])
                trials.append(record)
                invalid += 1
                checkpoint()
                continue
            invalid = 0
            record.update(
                recipe=recipe,
                changes=difference(incumbent["recipe"], recipe, tools),
                digest=digest(recipe),
            )
            record["result"] = self._evaluate(number, record, timing)
            accepted, reason = self.judge(record["result"], incumbent["result"], bounds)
            record["decision"] = {
                "decision": "promoted" if accepted else "rejected",
                "reason": reason,
            }
            write_json(directory / "decision.json", record["decision"])
            self._event("decision", trial=record["id"], **record["decision"])
            trials.append(record)
            used += 1
            if accepted:
                incumbent, stale = record, 0
            else:
                stale += 1
            checkpoint()

        confirmation = self._confirm(baseline, incumbent, timing, bounds)
        winner = incumbent if confirmation and confirmation["confirmed"] else baseline
        decision = "promoted" if winner is not baseline else "baseline retained"
        artifact = self._export(winner, timing) if self.export else None
        checkpoint()
        report_path = write_report(
            self.directory,
            goal=self.goal,
            bounds=bounds,
            tools=tools,
            trials=trials,
            baseline=baseline,
            winner=winner,
            decision=decision,
            stop_reason=stop_reason,
            confirmation=confirmation,
            artifact=artifact,
            environment=config.get("environment"),
            planner=self.planner.to_json(),
            elapsed_s=elapsed(),
        )
        self._event("finished", decision=decision, winner=winner["id"])
        return SessionResult(
            directory=self.directory,
            decision=decision,
            stop_reason=stop_reason,
            recipe=winner["recipe"],
            baseline=self.summary(baseline, self.goal),
            best=self.summary(winner, self.goal),
            confirmation=confirmation,
            artifact=artifact,
            report_path=report_path,
        )

    # ------------------------------------------------------------- finishing

    def _confirm(
        self,
        baseline: dict[str, Any],
        best: dict[str, Any],
        timing: Timing,
        bounds: Mapping[str, tuple[str, float]],
    ) -> dict[str, Any] | None:
        """Measure baseline and best again in fresh, back-to-back worker pairs."""
        if best is baseline:
            return None
        pairs = []
        for repeat in range(self.confirm_repeats):
            pair: dict[str, Any] = {}
            for arm, record in (("baseline", baseline), ("best", best)):
                directory = self.directory / "confirm" / f"{repeat}-{arm}"
                pair[arm] = self._run_worker(
                    {
                        "kind": "trial",
                        "recipe": record["recipe"],
                        "timing": asdict(timing),
                        "profile": False,
                    },
                    directory,
                )
            pairs.append(pair)
        gains, problems = [], []
        for index, pair in enumerate(pairs):
            for arm in ("baseline", "best"):
                accepted, reason = self.judge(pair[arm], None, bounds)
                if not accepted:
                    problems.append(f"pair {index} {arm}: {reason}")
            if (
                pair["baseline"].get("status") == "ok"
                and pair["best"].get("status") == "ok"
            ):
                gains.append(
                    self.goal.gain(
                        metric_value(pair["baseline"], self.goal),
                        metric_value(pair["best"], self.goal),
                    )
                )
        gain = statistics.median(gains) if gains else None
        confirmed = (
            not problems and gain is not None and gain > self.goal.min_improvement
        )
        confirmation = {
            "confirmed": confirmed,
            "median_gain": gain,
            "gains": gains,
            "problems": problems,
            "pairs": [
                {arm: _short(result, self.goal) for arm, result in pair.items()}
                for pair in pairs
            ],
        }
        write_json(self.directory / "confirm" / "summary.json", confirmation)
        self._event("confirmation", confirmed=confirmed, median_gain=gain)
        return confirmation

    def _export(self, winner: dict[str, Any], timing: Timing) -> Path | None:
        directory = self.directory / "artifact"
        write_json(directory / "recipe.json", winner["recipe"])
        result = self._run_worker(
            {
                "kind": "trial",
                "recipe": winner["recipe"],
                "timing": asdict(Timing(warmup=1, rounds=3, calls=1, profile_calls=0)),
                "profile": False,
                "export_dir": str(directory),
            },
            self.directory / "export",
        )
        if result.get("status") != "ok":
            logger.warning(
                "Exporting the %s recipe failed (%s): %s; its recipe is in %s",
                winner["id"],
                result.get("stage"),
                result.get("error"),
                directory / "recipe.json",
            )
            return None
        return directory


def _short(result: Mapping[str, Any], goal: Goal) -> dict[str, Any]:
    if result.get("status") != "ok":
        return {"status": result.get("status"), "stage": result.get("stage")}
    return {
        "status": "ok",
        "metric": metric_value(result, goal),
        "latency_ms": result["latency_ms"],
        "accuracy": result["accuracy"],
    }


def _tool_args(tool: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": tool["name"],
        "values": tuple(tool["values"]),
        "default": tool["default"],
        "description": tool["description"],
        "signals": tuple(tool["signals"]),
    }


def with_suppressed(function: Any, *args: Any) -> None:
    try:
        function(*args)
    except OSError:
        pass
