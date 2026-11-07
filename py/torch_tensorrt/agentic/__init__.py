# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental: agent-guided optimization sessions for Torch-TensorRT.

The user provides a workload (a model and representative inputs), a goal, the
accuracy and memory limits and a budget. An :class:`OptimizationSession`
establishes a validated baseline, profiles it, and lets a planner propose one
experiment at a time across four layers: graph and regions, TensorRT engine,
kernel generation and runtime execution. A fresh worker process builds,
checks, measures and profiles every candidate, and the session promotes a
candidate only when that independent evaluation shows a gain beyond noise
that keeps accuracy and memory within limits. The user receives the best
validated artifact, a before/after comparison and a reproducible recipe; the
baseline is retained if nothing beats it. Inference runs the artifact and
needs no agent.
"""

from torch_tensorrt.agentic._case import (
    Candidate,
    Case,
    ModuleCase,
    compare_outputs,
    describe_module,
)
from torch_tensorrt.agentic._planner import (
    AgentPlanner,
    HeuristicPlanner,
    Planner,
    Proposal,
)
from torch_tensorrt.agentic._session import (
    OptimizationSession,
    SessionError,
    SessionResult,
)
from torch_tensorrt.agentic._spec import Budget, Constraints, Goal, Timing, Workload
from torch_tensorrt.agentic._tools import GENERIC_TOOLS, Tool

__all__ = [
    "AgentPlanner",
    "Budget",
    "Candidate",
    "Case",
    "Constraints",
    "GENERIC_TOOLS",
    "Goal",
    "HeuristicPlanner",
    "ModuleCase",
    "OptimizationSession",
    "Planner",
    "Proposal",
    "SessionError",
    "SessionResult",
    "Timing",
    "Tool",
    "Workload",
    "compare_outputs",
    "describe_module",
]
