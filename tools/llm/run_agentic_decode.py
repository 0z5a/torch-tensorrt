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
choose the experiments; the heuristic planner needs no agent.
"""

import argparse
from pathlib import Path

from torch_tensorrt.agentic import (
    AgentPlanner,
    Budget,
    Constraints,
    Goal,
    HeuristicPlanner,
    OptimizationSession,
    Workload,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--isl", type=int, default=128, help="prompt tokens")
    parser.add_argument("--osl", type=int, default=64, help="decode tokens per call")
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
    session = OptimizationSession(
        Workload(
            f"{Path(__file__).with_name('agentic_decode.py')}:create_case",
            {"model": args.model, "isl": args.isl, "osl": args.osl},
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
    )
    result = session.run()
    print(result.report())


if __name__ == "__main__":
    main()
