# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect, report on and resume optimization sessions.

    python -m torch_tensorrt.agentic status SESSION_DIR
    python -m torch_tensorrt.agentic report SESSION_DIR
    python -m torch_tensorrt.agentic resume SESSION_DIR [--max-trials N] ...

``status`` and ``report`` only read the session's records. ``resume``
continues the recorded session, optionally with a larger budget.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any


def _status(directory: Path) -> int:
    config = json.loads((directory / "session.json").read_text())
    workload = config["workload"]
    print(f"Session {directory}")
    print(f"  workload: {workload['factory']} {json.dumps(workload['kwargs'])}")
    print(f"  goal: {json.dumps(config['goal'])}")
    print(
        f"  planner: {json.dumps(config['planner'])}; elapsed {config.get('elapsed_s', 0):.0f} s"
    )
    rows = []
    for trial in sorted((directory / "trials").glob("[0-9][0-9][0-9][0-9]")):
        record = _read(trial / "trial.json") or {}
        result = _read(trial / "result.json") or {}
        decision = _read(trial / "decision.json") or {}
        latency = (result.get("latency_ms") or {}).get("median")
        rows.append(
            (
                trial.name,
                json.dumps(record.get("changes") or {}, sort_keys=True),
                result.get("status", "running" if record else "-"),
                "-" if latency is None else f"{latency:.4g}",
                decision.get("decision", "pending"),
                decision.get("reason", ""),
            )
        )
    print(
        f"  {'trial':5}  {'status':8}  {'ms/unit':>8}  {'decision':9}  changes / reason"
    )
    for name, changes, status, latency, decision, reason in rows:
        print(f"  {name:5}  {status:8}  {latency:>8}  {decision:9}  {changes}")
        if reason:
            print(f"  {'':35}{reason}")
    report = directory / "report.md"
    if report.is_file():
        print(f"  finished; report: {report}")
    return 0


def _read(path: Path) -> Any:
    return json.loads(path.read_text()) if path.is_file() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m torch_tensorrt.agentic")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "report"):
        commands.add_parser(name).add_argument("directory", type=Path)
    resume = commands.add_parser("resume")
    resume.add_argument("directory", type=Path)
    resume.add_argument("--max-trials", type=int)
    resume.add_argument("--max-time-s", type=float)
    resume.add_argument("--patience", type=int)
    args = parser.parse_args(argv)
    directory = args.directory.expanduser().resolve()
    if not (directory / "session.json").is_file():
        print(f"{directory} is not an optimization session directory", file=sys.stderr)
        return 1
    if args.command == "status":
        return _status(directory)
    if args.command == "report":
        report = directory / "report.md"
        if not report.is_file():
            print(
                "The session has not finished; run `status` to see its progress",
                file=sys.stderr,
            )
            return 1
        print(report.read_text())
        return 0

    from torch_tensorrt.agentic._session import OptimizationSession

    session = OptimizationSession.resume(directory)
    changes = {
        key: value
        for key, value in (
            ("max_trials", args.max_trials),
            ("max_time_s", args.max_time_s),
            ("patience", args.patience),
        )
        if value is not None
    }
    if changes:
        session.budget = replace(session.budget, **changes)
    result = session.run()
    print(result.report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
