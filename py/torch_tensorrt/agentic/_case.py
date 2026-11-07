# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""The workload contract: how to build candidates and what correct output is."""

from __future__ import annotations

import contextlib
import copy
import math
from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.utils._pytree as pytree

from torch_tensorrt.agentic._spec import Recipe, Timing
from torch_tensorrt.agentic._tools import (
    GENERIC_TOOLS,
    Tool,
    autocast_settings,
    compile_settings,
)


class Candidate:
    """One built configuration of a workload, ready to check and measure.

    The evaluator calls :meth:`setup` once (untimed), then calls the candidate:
    each call performs ``units`` work units of ``items`` items each (for example
    ``units`` decoded tokens of a batch of ``items`` sequences) and returns the
    outputs that :meth:`Case.compare` checks.
    """

    units: int = 1
    items: int = 1

    def setup(self) -> None:
        """Prepare state before the first call, for example a prefill."""

    def __call__(self) -> Any:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        """Structural evidence: engines, PyTorch partitions, plugins and so on."""
        return {}

    def export(self, directory: Path) -> list[str]:
        """Save this candidate as a deployable artifact; return the files written."""
        raise NotImplementedError(f"{type(self).__name__} cannot export an artifact")

    def close(self) -> None:
        """Release runtime state; the evaluator calls it after measuring."""


class Case(ABC):
    """A user workload.

    A case declares the tools it honors and its baseline recipe, builds a
    :class:`Candidate` for resolved tool values (every declared tool present),
    and defines the reference outputs and the accuracy metrics that candidates
    are checked with. Cases are rebuilt in every worker process from a
    :class:`~torch_tensorrt.agentic.Workload` factory.
    """

    name: str = "workload"

    def tools(self) -> Sequence[Tool]:
        return GENERIC_TOOLS

    def baseline(self) -> Recipe:
        """The configuration the session starts from and keeps if nothing beats it."""
        return {}

    @abstractmethod
    def build(self, values: Mapping[str, Any]) -> Candidate: ...

    @abstractmethod
    def reference(self) -> Any:
        """Outputs that a correct candidate reproduces, computed once per session."""

    def compare(self, outputs: Any, reference: Any) -> dict[str, float]:
        return compare_outputs(outputs, reference)

    def accuracy_defaults(self) -> dict[str, tuple[str, float]]:
        return {"cosine_similarity": (">=", 0.9999), "rel_l2_error": ("<=", 1e-2)}

    def timing(self) -> Timing:
        return Timing()


def compare_outputs(outputs: Any, reference: Any) -> dict[str, float]:
    """Compare two output trees leaf by leaf.

    Floating leaves give ``max_abs_error``, ``rel_l2_error`` (the L2 norm of the
    difference over the reference's) and ``cosine_similarity``; integer and
    Boolean leaves give ``match_fraction``. Each metric reports the worst leaf.
    """
    leaves, spec = pytree.tree_flatten(outputs)
    expected, expected_spec = pytree.tree_flatten(reference)
    if spec != expected_spec:
        raise ValueError("Candidate outputs do not have the reference's structure")
    metrics: dict[str, float] = {}

    def worst(name: str, value: float, larger_is_worse: bool) -> None:
        if name not in metrics or not math.isfinite(value):
            metrics[name] = value
        elif math.isfinite(metrics[name]):
            pick = max if larger_is_worse else min
            metrics[name] = pick(metrics[name], value)

    for actual, wanted in zip(leaves, expected):
        if not isinstance(wanted, torch.Tensor):
            match = float(actual == wanted)
            worst("match_fraction", match, larger_is_worse=False)
            continue
        if not isinstance(actual, torch.Tensor) or actual.shape != wanted.shape:
            raise ValueError(
                f"Candidate output shape {getattr(actual, 'shape', None)} does not "
                f"match the reference shape {tuple(wanted.shape)}"
            )
        actual = actual.detach().to(device="cpu")
        wanted = wanted.detach().to(device="cpu")
        if wanted.is_floating_point() or wanted.is_complex():
            a, b = actual.double().flatten(), wanted.double().flatten()
            difference = (a - b).abs()
            norm = b.norm().item()
            finite = bool(torch.isfinite(a).all())
            worst(
                "max_abs_error",
                difference.max().item() if finite and a.numel() else math.inf,
                larger_is_worse=True,
            )
            worst(
                "rel_l2_error",
                (a - b).norm().item() / norm if finite and norm else math.inf,
                larger_is_worse=True,
            )
            cosine = (
                torch.nn.functional.cosine_similarity(a, b, dim=0).item()
                if finite and a.numel()
                else -1.0
            )
            worst("cosine_similarity", cosine, larger_is_worse=False)
        else:
            match = (actual == wanted).double().mean().item() if wanted.numel() else 1.0
            worst("match_fraction", match, larger_is_worse=False)
    return metrics


def describe_module(module: torch.nn.Module) -> dict[str, Any]:
    """Count the TensorRT engines, PyTorch partitions and plugin regions of ``module``."""
    from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

    engines = [m for m in module.modules() if isinstance(m, TorchTensorRTModule)]
    partitions = [
        name
        for name, child in module.named_children()
        if not isinstance(child, TorchTensorRTModule)
        and isinstance(child, torch.fx.GraphModule)
    ]
    layers = 0
    for engine in engines:
        cuda_engine = getattr(getattr(engine, "engine", None), "cuda_engine", None)
        layers += int(getattr(cuda_engine, "num_layers", 0) or 0)
    plugins = 0
    with contextlib.suppress(Exception):
        from torch_tensorrt.dynamo.region_compilation._types import get_region_records

        plugins = sum(
            1 for record in get_region_records(module) if record.route == "plugin"
        )
    return {
        "trt_engines": len(engines),
        "torch_partitions": len(partitions),
        "plugin_regions": plugins,
        "engine_layers": layers,
    }


class ModuleCase(Case):
    """A module with fixed example inputs, compiled by ``torch_tensorrt.compile``.

    It honors the generic tools: precision through Torch-TensorRT autocast,
    builder and partitioning options, and CUDA-graph and pre-allocated-output
    runtime modes. ``compile_options`` are passed to every compilation;
    ``reference_module`` defaults to ``module`` run eagerly.

    With a ``generator`` (a :class:`~torch_tensorrt.region.KernelGenerator`),
    every entry of ``regions``, a name mapped to a submodule path, adds a
    kernel-generation tool ``kernels.<name>``: ``"generated"`` compiles that
    submodule as a ``fused_kernel(generator=...)`` region, whose kernel the
    generator writes, validates and caches. Region capture needs strict export,
    so such candidates compile with ``strict=True``.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        inputs: Sequence[torch.Tensor],
        *,
        name: str = "module",
        compile_options: Mapping[str, Any] | None = None,
        reference_module: torch.nn.Module | None = None,
        baseline: Recipe | None = None,
        timing: Timing | None = None,
        tools: Sequence[Tool] = GENERIC_TOOLS,
        regions: Mapping[str, str] | None = None,
        generator: Any = None,
    ) -> None:
        self.name = name
        self.module = module.eval()
        self.inputs = tuple(inputs)
        self.compile_options = dict(compile_options or {})
        self.reference_module = (reference_module or module).eval()
        self._baseline = dict(baseline or {})
        self._timing = timing or Timing()
        self.regions = dict(regions or {})
        self.generator = generator
        if self.regions and generator is None:
            raise ValueError("ModuleCase regions need a KernelGenerator generator")
        for path in self.regions.values():
            module.get_submodule(path)
        self._tools = _defaults_from(tools, self.compile_options) + tuple(
            Tool(
                f"kernels.{region}",
                ("native", "generated"),
                "native",
                f"Compile submodule {path!r} as a region whose kernel the "
                "KernelGenerator writes, validates and caches, instead of TensorRT's "
                "own layers.",
                ("dominant_kernel",),
            )
            for region, path in self.regions.items()
        )

    def tools(self) -> Sequence[Tool]:
        return self._tools

    def baseline(self) -> Recipe:
        return dict(self._baseline)

    def timing(self) -> Timing:
        return self._timing

    def build(self, values: Mapping[str, Any]) -> Candidate:
        import torch_tensorrt

        settings = {
            **self.compile_options,
            **compile_settings(values),
            **autocast_settings(values.get("engine.precision", "fp32")),
        }
        module = self.module
        generated = [
            region
            for region in self.regions
            if values.get(f"kernels.{region}") == "generated"
        ]
        if generated:
            module = copy.deepcopy(self.module)
            for region in generated:
                _generate_region(
                    module.get_submodule(self.regions[region]), region, self.generator
                )
            settings["strict"] = True
        compiled = torch_tensorrt.compile(
            module, ir="dynamo", arg_inputs=list(self.inputs), **settings
        )
        return _ModuleCandidate(
            compiled,
            self.inputs,
            cudagraphs=bool(values.get("runtime.cudagraphs", False)),
            pre_allocated_outputs=bool(
                values.get("runtime.pre_allocated_outputs", False)
            ),
        )

    def reference(self) -> Any:
        with torch.inference_mode():
            return _detached(self.reference_module(*self.inputs))


# Compile options that a generic tool controls, by tool name.
_COMPILE_OPTION_TOOLS = {
    "graph.min_block_size": "min_block_size",
    "engine.optimization_level": "optimization_level",
    "engine.tiling_optimization_level": "tiling_optimization_level",
}


def _defaults_from(
    tools: Sequence[Tool], compile_options: Mapping[str, Any]
) -> tuple[Tool, ...]:
    """Make ``compile_options`` the defaults of the tools that control them.

    A recipe that leaves a tool alone then compiles with the user's option,
    and the catalog the planner reads shows what actually runs.
    """
    adjusted = []
    for tool in tools:
        option = _COMPILE_OPTION_TOOLS.get(tool.name)
        if option is None or option not in compile_options:
            adjusted.append(tool)
            continue
        value = compile_options[option]
        values = tool.values if value in tool.values else (value,) + tool.values
        adjusted.append(replace(tool, values=values, default=value))
    return tuple(adjusted)


def _generate_region(submodule: torch.nn.Module, name: str, generator: Any) -> None:
    """Make ``submodule`` a ``fused_kernel(generator=...)`` region named ``name``."""
    from torch_tensorrt.region import fused_kernel

    body = submodule.forward

    def forward(*args: Any, **kwargs: Any) -> Any:
        with fused_kernel(generator=generator, name=name):
            return body(*args, **kwargs)

    submodule.forward = forward


def _detached(tree: Any) -> Any:
    return pytree.tree_map(
        lambda value: (
            value.detach().clone() if isinstance(value, torch.Tensor) else value
        ),
        tree,
    )


class _ModuleCandidate(Candidate):
    def __init__(
        self,
        compiled: torch.nn.Module,
        inputs: Sequence[torch.Tensor],
        *,
        cudagraphs: bool,
        pre_allocated_outputs: bool,
    ) -> None:
        self.compiled = compiled
        self.inputs = tuple(inputs)
        self.cudagraphs = cudagraphs
        self.pre_allocated_outputs = pre_allocated_outputs
        self._stack = contextlib.ExitStack()
        self._run: Any = compiled

    def setup(self) -> None:
        import torch_tensorrt

        if self.pre_allocated_outputs:
            self._stack.enter_context(
                torch_tensorrt.runtime.enable_pre_allocated_outputs(self.compiled)
            )
        if self.cudagraphs:
            self._run = self._stack.enter_context(
                torch_tensorrt.runtime.enable_cudagraphs(self.compiled)
            )

    def __call__(self) -> Any:
        with torch.inference_mode():
            return _detached(self._run(*self.inputs))

    def describe(self) -> dict[str, Any]:
        return describe_module(self.compiled)

    def export(self, directory: Path) -> list[str]:
        import torch_tensorrt

        path = Path(directory) / "model.ep"
        torch_tensorrt.save(self.compiled, str(path), arg_inputs=list(self.inputs))
        return [path.name]

    def close(self) -> None:
        self._stack.close()
        self._run = self.compiled
