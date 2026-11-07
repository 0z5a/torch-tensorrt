# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""The session report: before/after comparison, trials and the reproducible recipe."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from torch_tensorrt.agentic._tools import Tool, difference

if TYPE_CHECKING:
    from torch_tensorrt.agentic._spec import Goal


def _number(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-" if value is None else str(value)
    return f"{value:.{digits}g}"


def _latency(result: Mapping[str, Any]) -> str:
    latency = result.get("latency_ms")
    if not latency:
        return "-"
    return (
        f"{_number(latency['median'])} [{_number(latency['p25'])}, "
        f"{_number(latency['p75'])}]"
    )


def _change(before: Any, after: Any, lower_is_better: bool) -> str:
    if not all(isinstance(v, (int, float)) for v in (before, after)) or not before:
        return ""
    relative = (after - before) / before
    better = relative < 0 if lower_is_better else relative > 0
    return f"{relative:+.1%}" + (" (better)" if better and abs(relative) > 0 else "")


def _profile_lines(title: str, result: Mapping[str, Any]) -> list[str]:
    profile = result.get("profile")
    lines = [f"### {title}", ""]
    if not profile:
        return lines + ["No profile was recorded.", ""]
    lines += [
        f"- {_number(profile['wall_ms'])} ms per unit wall (untraced), "
        f"{_number(profile['device_ms'])} ms GPU work: the GPU waits for the host "
        f"{profile['host_overhead_fraction']:.0%} of the time",
        f"- {profile['kernels']:.0f} kernel launches and "
        f"{profile['host_syncs']:.0f} device-to-host copies per unit",
    ]
    for signal in result.get("signals", ()):
        lines.append(f"- signal `{signal['signal']}`: {signal['detail']}")
    lines += [
        "",
        "| GPU time (ms/unit) | Share | Calls/unit | Kernel |",
        "|---|---|---|---|",
    ]
    for kernel in profile.get("top_kernels", ()):
        lines.append(
            f"| {_number(kernel['ms'])} | {kernel['share']:.0%} | "
            f"{_number(kernel['calls'], 3)} | `{kernel['name']}` |"
        )
    return lines + [""]


def write_report(
    directory: Path,
    *,
    goal: Goal,
    bounds: Mapping[str, tuple[str, float]],
    tools: Mapping[str, Tool],
    trials: list[dict[str, Any]],
    baseline: dict[str, Any],
    winner: dict[str, Any],
    decision: str,
    stop_reason: str,
    confirmation: dict[str, Any] | None,
    artifact: Path | None,
    environment: Mapping[str, Any] | None,
    planner: Mapping[str, Any],
    elapsed_s: float,
) -> Path:
    """Write ``report.md`` and ``report.json`` in ``directory``; return the Markdown path."""
    before, after = baseline["result"], winner["result"]
    lower = goal.lower_is_better
    evaluated = [t for t in trials[1:] if t["decision"]["decision"] != "invalid"]
    invalid = len(trials) - 1 - len(evaluated)
    headline = (
        f"promoted trial {winner['id']}"
        if decision == "promoted"
        else (
            "baseline retained (no candidate beat it in confirmation)"
            if confirmation
            else "baseline retained (no candidate was promoted)"
        )
    )
    lines = [
        f"# Optimization session report: {directory.name}",
        "",
        f"**Decision: {headline}.** Search stopped: {stop_reason}.",
        "",
        f"- Goal: {'minimize' if lower else 'maximize'} `{goal.metric}`"
        + (f", target {goal.target:g}" if goal.target else "")
        + f"; a candidate must improve it by more than {goal.min_improvement:.1%} "
        "with non-overlapping quartiles",
        "- Accuracy bounds: "
        + ", ".join(f"`{name} {op} {value:g}`" for name, (op, value) in bounds.items()),
        f"- Planner: `{json.dumps(dict(planner), sort_keys=True)}`",
        f"- Trials: {len(evaluated)} evaluated, {invalid} invalid proposals; "
        f"elapsed {elapsed_s:.0f} s",
    ]
    if environment:
        lines.append(
            "- Environment: "
            + ", ".join(f"{key} {value}" for key, value in environment.items())
        )
    lines += [
        "",
        "## Before / after",
        "",
        f"| | Baseline ({baseline['id']}) | Selected ({winner['id']}) | Change |",
        "|---|---|---|---|",
        f"| latency ms/unit, median [p25, p75] | {_latency(before)} | {_latency(after)} | "
        f"{_change(before['latency_ms']['median'], after['latency_ms']['median'], True)} |",
        f"| throughput items/s | {_number(before.get('throughput'))} | "
        f"{_number(after.get('throughput'))} | "
        f"{_change(before.get('throughput'), after.get('throughput'), False)} |",
    ]
    for name in bounds:
        lines.append(
            f"| accuracy `{name}` | {_number(before['accuracy'].get(name), 6)} | "
            f"{_number(after['accuracy'].get(name), 6)} | |"
        )
    for key, label in (
        ("process_gpu_mb", "GPU memory MB (process)"),
        ("torch_peak_mb", "PyTorch peak MB"),
    ):
        b = (before.get("memory") or {}).get(key)
        a = (after.get("memory") or {}).get(key)
        lines.append(
            f"| {label} | {_number(b)} | {_number(a)} | {_change(b, a, True)} |"
        )
    structure_b, structure_a = before.get("structure", {}), after.get("structure", {})
    lines.append(
        "| TRT engines / PyTorch partitions / plugins | "
        + " / ".join(
            str(structure_b.get(k, "-"))
            for k in ("trt_engines", "torch_partitions", "plugin_regions")
        )
        + " | "
        + " / ".join(
            str(structure_a.get(k, "-"))
            for k in ("trt_engines", "torch_partitions", "plugin_regions")
        )
        + " | |"
    )
    lines.append(
        f"| build s | {_number(before.get('build_s'))} | {_number(after.get('build_s'))} | |"
    )
    lines.append("")
    if confirmation:
        gains = ", ".join(f"{gain:+.1%}" for gain in confirmation["gains"])
        median = confirmation["median_gain"]
        lines.append(
            "Confirmation in fresh baseline/best worker pairs: median gain "
            + (f"{median:+.1%}" if median is not None else "unavailable")
            + f" (per pair: {gains or 'none'}); "
            + ("confirmed." if confirmation["confirmed"] else "NOT confirmed.")
        )
        for problem in confirmation["problems"]:
            lines.append(f"- {problem}")
        lines.append("")
    changes = difference(baseline["recipe"], winner["recipe"], tools)
    lines += [
        "## Recipe",
        "",
        "Changes from the baseline: "
        + (f"`{json.dumps(changes, sort_keys=True)}`" if changes else "none"),
        "",
        f"Full recipe: `{json.dumps(winner['recipe'], sort_keys=True)}`"
        + (f" (also in `{artifact / 'recipe.json'}`)" if artifact else ""),
        "",
        "Artifact: " + (f"`{artifact}`" if artifact else "not exported"),
        "",
        "## Trials",
        "",
        "| Trial | Parent | Changes | Status | latency ms/unit | Decision | Reason | Planner rationale |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for trial in trials:
        result = trial.get("result") or {}
        status = result.get("status", "-")
        if status == "failed":
            status = f"failed ({result.get('stage')})"
        rationale = (trial.get("proposal") or {}).get("rationale", "")
        note = (trial.get("proposal") or {}).get("note")
        if note:
            rationale = f"{rationale} [{note}]"
        lines.append(
            "| {id} | {parent} | `{changes}` | {status} | {latency} | {decision} | {reason} | {rationale} |".format(
                id=trial["id"],
                parent=trial.get("parent") or "-",
                changes=json.dumps(trial.get("changes") or {}, sort_keys=True),
                status=status,
                latency=_latency(result),
                decision=trial["decision"]["decision"],
                reason=str(trial["decision"]["reason"]).replace("|", "\\|"),
                rationale=rationale.replace("|", "\\|").replace("\n", " "),
            )
        )
    lines += ["", "## Profiles", ""]
    lines += _profile_lines(f"Baseline ({baseline['id']})", before)
    if winner is not baseline:
        lines += _profile_lines(f"Selected ({winner['id']})", after)
    report_path = directory / "report.md"
    report_path.write_text("\n".join(lines) + "\n")
    summary = {
        "decision": decision,
        "stop_reason": stop_reason,
        "goal": asdict(goal),
        "accuracy_bounds": {name: list(bound) for name, bound in bounds.items()},
        "baseline": {
            "trial": baseline["id"],
            "recipe": baseline["recipe"],
            "result": before,
        },
        "selected": {
            "trial": winner["id"],
            "recipe": winner["recipe"],
            "result": after,
        },
        "changes": changes,
        "confirmation": confirmation,
        "artifact": str(artifact) if artifact else None,
        "environment": dict(environment or {}),
        "planner": dict(planner),
        "elapsed_s": elapsed_s,
        "trials": [
            {
                "id": trial["id"],
                "parent": trial.get("parent"),
                "changes": trial.get("changes"),
                "recipe": trial.get("recipe"),
                "status": (trial.get("result") or {}).get("status"),
                "decision": trial["decision"],
                "proposal": trial.get("proposal"),
                "latency_ms": (trial.get("result") or {}).get("latency_ms"),
            }
            for trial in trials
        ],
    }
    (directory / "report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
    )
    return report_path
