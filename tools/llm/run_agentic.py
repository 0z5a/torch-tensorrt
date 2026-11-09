# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
.. _run_agentic:

Running an agentic compiler session on any workload
===================================================

Runs a ``torch_tensorrt.agentic`` optimization session on a workload factory,
``path/to/file.py:create_case`` called with JSON keyword arguments. It serves
the prefill, encoder and hybrid-model workloads in this directory;
``run_agentic_decode.py`` keeps the decode workload's own options. Rerunning
the command resumes the session.

.. code-block:: bash

    python tools/llm/run_agentic.py --session /tmp/gemma3-prefill-2k \
        --workload tools/llm/agentic_prefill.py:create_case \
        --args '{"model": "unsloth/gemma-3-1b-it", "length": 2048}' \
        --accuracy top1_agreement '>=' 0.97 --max-trials 10

Accuracy limits not given with ``--accuracy`` use the workload's defaults.
``--planner agent`` asks the logged-in ``claude`` (or ``--provider codex``) CLI
to choose the experiments, with ``--brief FILE`` as its research brief.
"""

import argparse
import json
from pathlib import Path

from torch_tensorrt.agentic import (
    AgentPlanner,
    Budget,
    Constraints,
    Goal,
    HeuristicPlanner,
    Measurement,
    OptimizationSession,
    Workload,
)


def _factory(spec: str) -> str:
    """Make a ``file.py:function`` factory absolute: workers may start elsewhere."""
    module, sep, attribute = spec.rpartition(":")
    if sep and module.endswith(".py"):
        return f"{Path(module).resolve()}:{attribute}"
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument(
        "--workload",
        required=True,
        help="workload factory: path/to/file.py:function or package.module:function",
    )
    parser.add_argument(
        "--args", default="{}", help="JSON keyword arguments of the factory"
    )
    parser.add_argument("--session", required=True, help="session directory")
    parser.add_argument(
        "--planner", choices=("heuristic", "agent"), default="heuristic"
    )
    parser.add_argument(
        "--provider", default="claude", help="agent CLI or module:attribute"
    )
    parser.add_argument("--agent-model", help="model name for the agent CLI")
    parser.add_argument(
        "--metric", choices=("latency_ms", "throughput"), default="latency_ms"
    )
    parser.add_argument("--max-trials", type=int, default=10)
    parser.add_argument("--max-time-s", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument(
        "--trial-timeout-s",
        type=float,
        default=1800.0,
        help="kill a worker that builds or measures for longer",
    )
    parser.add_argument("--min-improvement", type=float, default=0.02)
    parser.add_argument(
        "--target", type=float, help="stop once the metric reaches this value"
    )
    parser.add_argument(
        "--accuracy",
        nargs=3,
        action="append",
        default=[],
        metavar=("METRIC", "OP", "VALUE"),
        help="accuracy limit, e.g. --accuracy top1_agreement '>=' 0.97 (repeatable)",
    )
    parser.add_argument("--max-memory-mb", type=float)
    parser.add_argument("--brief", help="Markdown research brief for the agent")
    parser.add_argument(
        "--baseline", help="JSON recipe to start from instead of the defaults"
    )
    parser.add_argument("--calibration-runs", type=int, default=2)
    parser.add_argument("--race-pairs", type=int, default=2)
    parser.add_argument("--confirm-repeats", type=int, default=2)
    parser.add_argument(
        "--quiet-gpu-s",
        type=float,
        default=600.0,
        help="seconds each worker may wait for an idle GPU; 0 turns the wait off",
    )
    parser.add_argument(
        "--max-spread",
        type=float,
        default=0.03,
        help="largest interquartile range, relative to the median, of a stable run",
    )
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    planner = (
        AgentPlanner(args.provider, args.agent_model)
        if args.planner == "agent"
        else HeuristicPlanner()
    )
    session = OptimizationSession(
        Workload(_factory(args.workload), json.loads(args.args)),
        directory=args.session,
        goal=Goal(
            metric=args.metric,
            target=args.target,
            min_improvement=args.min_improvement,
        ),
        constraints=Constraints(
            accuracy={name: (op, float(value)) for name, op, value in args.accuracy},
            max_memory_mb=args.max_memory_mb,
        ),
        budget=Budget(
            max_trials=args.max_trials,
            max_time_s=args.max_time_s,
            patience=args.patience,
            trial_timeout_s=args.trial_timeout_s,
        ),
        planner=planner,
        baseline=json.loads(args.baseline) if args.baseline else None,
        confirm_repeats=args.confirm_repeats,
        measurement=Measurement(
            calibration_runs=args.calibration_runs,
            race_pairs=args.race_pairs,
            quiet_gpu_s=args.quiet_gpu_s,
            retries=args.retries,
            max_spread=args.max_spread,
        ),
        brief=Path(args.brief).read_text() if args.brief else None,
    )
    result = session.run()
    print(result.report())


if __name__ == "__main__":
    main()
