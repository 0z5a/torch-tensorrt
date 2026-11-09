# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
.. _run_agentic_decode:

Optimizing LLM decode with an agentic compiler session
=======================================================

Runs a ``torch_tensorrt.agentic`` optimization session on the batch-1 decode of
a Hugging Face causal LM (default ``Qwen/Qwen3-0.6B``): it measures the
documented ``tools/llm`` baseline, lets a planner propose one change at a time
(KV cache lowering, precision, builder options, CUDA graphs, ...), evaluates
every candidate in a fresh process against an FP32 reference, and keeps only
validated gains. The session directory holds every trial, the report and the
exported decode engine; rerunning the command resumes it.

.. code-block:: bash

    python tools/llm/run_agentic_decode.py --session /tmp/qwen3-decode --max-trials 10
    python -m torch_tensorrt.agentic status /tmp/qwen3-decode

``--planner agent`` asks the logged-in ``claude`` (or ``--provider codex``) CLI to
choose the experiments; the heuristic planner needs no agent. ``--brief FILE``
gives the agent a Markdown research brief (``program.md`` in the session);
``--calibration-runs`` and ``--race-pairs`` set how the session separates
gains from run-to-run noise; ``--quiet-gpu-s 0`` and a looser ``--max-spread``
run it on a GPU that is never idle. ``--prompt`` and ``--prefill`` choose the
token layout and the static_v3 prefill (see ``agentic_decode.py``); their
``auto`` defaults suit both short and long contexts:

.. code-block:: bash

    python tools/llm/run_agentic_decode.py --session /tmp/qwen3-decode-8k \
        --isl 8192 --osl 64 --max-trials 8
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--isl", type=int, default=128, help="prompt tokens")
    parser.add_argument("--osl", type=int, default=64, help="decode tokens per call")
    parser.add_argument(
        "--prompt",
        choices=("auto", "text", "recall"),
        default="auto",
        help="token layout; 'recall' makes decode depend on the whole context",
    )
    parser.add_argument(
        "--prefill",
        choices=("auto", "engine", "eager"),
        default="auto",
        help="static_v3 prefill: through the engine one token at a time, or eager",
    )
    parser.add_argument("--session", required=True, help="session directory")
    parser.add_argument(
        "--planner", choices=("heuristic", "agent"), default="heuristic"
    )
    parser.add_argument(
        "--provider", default="claude", help="agent CLI or module:attribute"
    )
    parser.add_argument("--agent-model", help="model name for the agent CLI")
    parser.add_argument("--max-trials", type=int, default=10)
    parser.add_argument("--max-time-s", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--min-improvement", type=float, default=0.02)
    parser.add_argument("--target-ms", type=float, help="stop at this ms/token")
    parser.add_argument("--min-top1", type=float, help="minimum top-1 agreement")
    parser.add_argument("--max-kl", type=float, help="maximum mean KL divergence")
    parser.add_argument("--brief", help="Markdown research brief for the agent")
    parser.add_argument(
        "--baseline",
        help='JSON recipe to start from, e.g. \'{"workload.cache": "static_v3"}\'',
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

    accuracy = {}
    if args.min_top1 is not None:
        accuracy["top1_agreement"] = (">=", args.min_top1)
    if args.max_kl is not None:
        accuracy["mean_kl"] = ("<=", args.max_kl)
    planner = (
        AgentPlanner(args.provider, args.agent_model)
        if args.planner == "agent"
        else HeuristicPlanner()
    )
    workload_args = {"model": args.model, "isl": args.isl, "osl": args.osl}
    # Recorded only when set, so earlier sessions resume with the same workload.
    if args.prompt != "auto":
        workload_args["prompt"] = args.prompt
    if args.prefill != "auto":
        workload_args["prefill"] = args.prefill
    session = OptimizationSession(
        Workload(
            f"{Path(__file__).with_name('agentic_decode.py')}:create_case",
            workload_args,
        ),
        directory=args.session,
        goal=Goal(target=args.target_ms, min_improvement=args.min_improvement),
        constraints=Constraints(accuracy=accuracy),
        budget=Budget(
            max_trials=args.max_trials,
            max_time_s=args.max_time_s,
            patience=args.patience,
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
