# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Workloads for the agentic session tests.

``create_fake`` builds a CPU-only case whose latency, accuracy and failures
are scripted per tool value, so controller behavior is deterministic.
``create_mlp`` is a small real module for GPU tests.
"""

import time
from typing import Any, Mapping

import torch

from torch_tensorrt.agentic import Candidate, Case, ModuleCase, Timing, Tool

FAKE_TOOLS = (
    Tool(
        "runtime.cudagraphs",
        (False, True),
        False,
        "replay a graph",
        ("launch_overhead",),
    ),
    Tool("engine.precision", ("fp32", "fp16"), "fp32", "precision", ("precision",)),
    Tool("engine.optimization_level", (3, 5), 3, "builder level", ("dominant_kernel",)),
    Tool("workload.variant", ("a", "b", "c"), "a", "workload option"),
)


class FakeCandidate(Candidate):
    def __init__(self, milliseconds: float, output: torch.Tensor) -> None:
        self.milliseconds = milliseconds
        self.output = output
        self.units = 2

    def __call__(self) -> torch.Tensor:
        time.sleep(self.milliseconds / 1000)
        return self.output.clone()

    def describe(self) -> dict[str, Any]:
        return {"trt_engines": 1, "torch_partitions": 0, "plugin_regions": 0}

    def export(self, directory):
        (directory / "fake.txt").write_text(f"{self.milliseconds}\n")
        return ["fake.txt"]


class FakeCase(Case):
    """``speedups`` maps "tool=value" to a latency factor; ``broken`` lists
    options whose build fails; ``inaccurate`` lists options that change the
    output; ``baseline_changes`` is the case's own baseline recipe."""

    name = "fake"

    def __init__(
        self,
        milliseconds: float = 6.0,
        speedups: Mapping[str, float] | None = None,
        broken: tuple = (),
        inaccurate: tuple = (),
        baseline_changes: Mapping[str, Any] | None = None,
        reference_fails: bool = False,
    ) -> None:
        self.milliseconds = milliseconds
        self.speedups = dict(speedups or {})
        self.broken = set(broken)
        self.inaccurate = set(inaccurate)
        self.baseline_changes = dict(baseline_changes or {})
        self.reference_fails = reference_fails

    def tools(self):
        return FAKE_TOOLS

    def baseline(self):
        return dict(self.baseline_changes)

    def timing(self):
        return Timing(warmup=1, rounds=5, calls=2, profile_calls=0)

    def accuracy_defaults(self):
        return {"max_abs_error": ("<=", 1e-3)}

    def reference(self):
        if self.reference_fails:
            raise RuntimeError("no reference today")
        return torch.ones(4)

    def build(self, values: Mapping[str, Any]) -> Candidate:
        options = {f"{name}={value}" for name, value in values.items()}
        if options & self.broken:
            raise RuntimeError(f"cannot build {sorted(options & self.broken)}")
        milliseconds = self.milliseconds
        for option, factor in self.speedups.items():
            if option in options:
                milliseconds *= factor
        output = torch.ones(4) * (1.5 if options & self.inaccurate else 1.0)
        return FakeCandidate(milliseconds, output)


def create_fake(**kwargs: Any) -> FakeCase:
    return FakeCase(**kwargs)


def create_mlp(width: int = 256, depth: int = 4, batch: int = 8) -> ModuleCase:
    torch.manual_seed(0)
    layers: list[torch.nn.Module] = []
    for _ in range(depth):
        layers += [torch.nn.Linear(width, width), torch.nn.GELU()]
    model = torch.nn.Sequential(*layers).cuda()
    inputs = [torch.randn(batch, width, device="cuda")]
    return ModuleCase(
        model,
        inputs,
        name="mlp",
        compile_options={"min_block_size": 1},
        timing=Timing(warmup=2, rounds=5, calls=4, profile_calls=2),
    )
