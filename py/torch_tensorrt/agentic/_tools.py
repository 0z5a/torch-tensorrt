# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compiler tools: the controlled changes a planner may request."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from torch_tensorrt.agentic._spec import Recipe

# The four optimization layers of a session, plus options a workload defines.
LAYERS = ("graph", "engine", "kernels", "runtime", "workload")
# Bottleneck signals the profiler reports (see _profile.signals).
SIGNALS = ("launch_overhead", "host_syncs", "fallback", "dominant_kernel", "precision")


@dataclass(frozen=True)
class Tool:
    """A named compiler option with the finite values a planner may choose.

    ``name`` is ``"<layer>.<option>"`` with a layer from ``graph``, ``engine``,
    ``kernels``, ``runtime`` and ``workload``. ``values`` lists the allowed
    values (JSON scalars) in the order a heuristic search tries them;
    ``default`` is the value a recipe that omits the tool uses. ``signals``
    names the profiler bottleneck signals the option can relieve.
    """

    name: str
    values: tuple[Any, ...]
    default: Any
    description: str
    signals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        layer, sep, option = self.name.partition(".")
        if not sep or layer not in LAYERS or not option:
            raise ValueError(
                f"Tool name must be '<layer>.<option>' with a layer in {LAYERS}; "
                f"got {self.name!r}"
            )
        values = tuple(self.values)
        if not values or any(
            not isinstance(value, (str, int, float, bool)) and value is not None
            for value in values
        ):
            raise ValueError(f"Tool {self.name} values must be JSON scalars")
        if len({_key(value) for value in values}) != len(values):
            raise ValueError(f"Tool {self.name} lists a value twice")
        if not is_member(self.default, values):
            raise ValueError(f"Tool {self.name} default must be one of its values")
        unknown = set(self.signals) - set(SIGNALS)
        if unknown:
            raise ValueError(
                f"Tool {self.name} names unknown signals {sorted(unknown)}"
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "signals", tuple(self.signals))

    @property
    def layer(self) -> str:
        return self.name.partition(".")[0]

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "layer": self.layer,
            "values": list(self.values),
            "default": self.default,
            "description": self.description,
            "signals": list(self.signals),
        }


def _key(value: Any) -> tuple[str, Any]:
    # bool is an int and 1.0 == 1 in Python; recipes must not conflate them.
    return type(value).__name__, value


def is_member(value: Any, values: Iterable[Any]) -> bool:
    return any(_key(value) == _key(allowed) for allowed in values)


GENERIC_TOOLS: tuple[Tool, ...] = (
    Tool(
        "runtime.cudagraphs",
        (False, True),
        False,
        "Replay the compiled module from a CUDA graph, removing per-call launch "
        "and host overhead; needs stream-capturable engines and static shapes.",
        ("launch_overhead", "host_syncs"),
    ),
    Tool(
        "runtime.pre_allocated_outputs",
        (False, True),
        False,
        "Reuse output buffers between calls instead of allocating them each call.",
        ("launch_overhead",),
    ),
    Tool(
        "engine.precision",
        ("fp32", "fp16", "bf16"),
        "fp32",
        "Compute precision of TensorRT layers. Lower precision halves memory "
        "traffic and enables tensor cores, within the accuracy constraints.",
        ("precision", "dominant_kernel"),
    ),
    Tool(
        "engine.optimization_level",
        (3, 5, 4),
        3,
        "TensorRT builder optimization level; higher levels search more tactics "
        "and take longer to build.",
        ("dominant_kernel",),
    ),
    Tool(
        "engine.tiling_optimization_level",
        ("none", "moderate", "fast", "full"),
        "none",
        "TensorRT tiling optimization: searches cross-kernel tiled schedules for "
        "on-chip cache reuse, at extra build time.",
        ("dominant_kernel",),
    ),
    Tool(
        "graph.min_block_size",
        (5, 1, 3, 10),
        5,
        "Minimum number of operators in a TensorRT engine. Lower values place "
        "more of the graph in TensorRT; higher values avoid many small engines.",
        ("fallback",),
    ),
)


def catalog(tools: Sequence[Tool]) -> dict[str, Tool]:
    """Index ``tools`` by name, rejecting duplicates."""
    indexed: dict[str, Tool] = {}
    for tool in tools:
        if not isinstance(tool, Tool):
            raise TypeError(f"Expected a Tool; got {type(tool).__name__}")
        if tool.name in indexed:
            raise ValueError(f"Tool {tool.name} is defined twice")
        indexed[tool.name] = tool
    return indexed


def apply(
    recipe: Mapping[str, Any], changes: Mapping[str, Any], tools: Mapping[str, Tool]
) -> Recipe:
    """Return ``recipe`` with ``changes`` applied, validated and normalized.

    Raises ``ValueError`` naming the first unknown tool or disallowed value.
    Options set to their default are dropped, so equal configurations compare
    equal.
    """
    merged = dict(recipe)
    for name, value in changes.items():
        tool = tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool {name!r}; choose one of {sorted(tools)}")
        if not is_member(value, tool.values):
            raise ValueError(
                f"Tool {name} does not allow {value!r}; allowed: {list(tool.values)}"
            )
        merged[name] = value
    for name, value in list(merged.items()):
        tool = tools.get(name)
        if tool is None:
            raise ValueError(f"Recipe sets unknown tool {name!r}")
        if not is_member(value, tool.values):
            raise ValueError(
                f"Recipe sets {name}={value!r}; allowed: {list(tool.values)}"
            )
        if _key(value) == _key(tool.default):
            del merged[name]
    return dict(sorted(merged.items()))


def resolve(recipe: Mapping[str, Any], tools: Mapping[str, Tool]) -> dict[str, Any]:
    """Every tool's value under ``recipe``, defaults included."""
    normalized = apply(recipe, {}, tools)
    return {name: normalized.get(name, tool.default) for name, tool in tools.items()}


def difference(
    before: Mapping[str, Any], after: Mapping[str, Any], tools: Mapping[str, Tool]
) -> dict[str, Any]:
    """The options whose value ``after`` changes, with their new values."""
    old, new = resolve(before, tools), resolve(after, tools)
    return {name: new[name] for name in tools if _key(old[name]) != _key(new[name])}


def compile_settings(values: Mapping[str, Any]) -> dict[str, Any]:
    """Map resolved generic engine and graph tools to ``torch_tensorrt.compile`` options.

    Precision is left to the case: modules may cast weights and inputs, or use
    Torch-TensorRT autocast (:func:`autocast_settings`).
    """
    settings: dict[str, Any] = {}
    if "graph.min_block_size" in values:
        settings["min_block_size"] = values["graph.min_block_size"]
    if "engine.optimization_level" in values:
        settings["optimization_level"] = values["engine.optimization_level"]
    if "engine.tiling_optimization_level" in values:
        settings["tiling_optimization_level"] = values[
            "engine.tiling_optimization_level"
        ]
    return settings


def autocast_settings(precision: str) -> dict[str, Any]:
    """Torch-TensorRT autocast options for ``engine.precision``."""
    import torch

    if precision == "fp32":
        return {}
    low = {"fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    return {"enable_autocast": True, "autocast_low_precision_type": low}
