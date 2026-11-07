# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import math

import pytest
import torch

from torch_tensorrt.agentic import (
    Budget,
    Constraints,
    Goal,
    Timing,
    Tool,
    Workload,
    compare_outputs,
)
from torch_tensorrt.agentic._spec import digest, violations
from torch_tensorrt.agentic._tools import apply, catalog, difference, is_member, resolve


def test_workload_factories_are_module_functions(tmp_path):
    with pytest.raises(ValueError, match="module:function"):
        Workload("no_factory")
    with pytest.raises(ValueError, match="module:function"):
        Workload("package.module:not-an-identifier")
    with pytest.raises(FileNotFoundError):
        Workload(f"{tmp_path / 'missing.py'}:create")
    with pytest.raises(TypeError, match="JSON data"):
        Workload("package.module:create", {"model": object()})


def test_file_factories_resolve_to_absolute_paths(tmp_path, monkeypatch):
    (tmp_path / "case.py").write_text("def create():\n    return 3\n")
    monkeypatch.chdir(tmp_path)
    workload = Workload("case.py:create")
    assert workload.factory == f"{tmp_path / 'case.py'}:create"
    with pytest.raises(TypeError, match="not a torch_tensorrt.agentic.Case"):
        workload.load()


def test_goal_measures_gains_in_the_metric_direction():
    latency = Goal(target=1.0)
    assert latency.gain(2.0, 1.5) == pytest.approx(0.25)
    assert latency.reached(0.9) and not latency.reached(1.1)
    throughput = Goal(metric="throughput", target=100.0)
    assert throughput.gain(50.0, 75.0) == pytest.approx(0.5)
    assert throughput.reached(100.0) and not throughput.reached(99.0)
    assert not Goal().reached(0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"metric": "speed"},
        {"min_improvement": 1.0},
        {"min_improvement": -0.1},
        {"target": -1.0},
    ],
)
def test_goal_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        Goal(**kwargs)


def test_constraints_override_case_defaults_and_report_every_miss():
    constraints = Constraints(accuracy={"top1": (">=", 0.99)})
    bounds = constraints.bounds({"top1": (">=", 0.5), "kl": ("<=", 0.01)})
    assert bounds == {"top1": (">=", 0.99), "kl": ("<=", 0.01)}
    assert violations({"top1": 0.995, "kl": 0.001}, bounds) == []
    assert violations({"top1": 0.9, "kl": math.nan}, bounds) == [
        "kl is unavailable (needs <= 0.01)",
        "top1=0.9 (needs >= 0.99)",
    ]
    assert violations({}, {"top1": (">", 0.5)}) == ["top1 is unavailable (needs > 0.5)"]


@pytest.mark.parametrize(
    "bound", [("==", 1.0), (">=", True), (">=", math.inf), (">=",)]
)
def test_constraints_reject_malformed_bounds(bound):
    with pytest.raises(ValueError):
        Constraints(accuracy={"top1": bound})


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (Budget, {"max_trials": -1}),
        (Budget, {"patience": True}),
        (Budget, {"trial_timeout_s": 0}),
        (Budget, {"max_time_s": math.nan}),
        (Timing, {"rounds": 2}),
        (Timing, {"calls": 0}),
    ],
)
def test_budget_and_timing_reject_invalid_values(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)


@pytest.mark.parametrize(
    "args,match",
    [
        (("speed.option", (1,), 1, ""), "layer"),
        (("engine.option", (1, 1), 1, ""), "twice"),
        (("engine.option", (1, 2), 3, ""), "default"),
        (("engine.option", ([1],), [1], ""), "JSON scalars"),
        (("engine.option", (1,), 1, "", ("slow",)), "signals"),
    ],
)
def test_tools_validate_their_definition(args, match):
    with pytest.raises(ValueError, match=match):
        Tool(*args)


def test_booleans_and_numbers_are_different_values():
    assert not is_member(True, (0, 1))
    assert not is_member(1, (True, False))
    assert not is_member(1.0, (1, 2))
    assert is_member(None, (None, "x"))


def test_recipes_are_validated_and_normalized():
    tools = catalog(
        [
            Tool("runtime.cudagraphs", (False, True), False, ""),
            Tool("engine.level", (3, 5), 3, ""),
        ]
    )
    recipe = apply({}, {"runtime.cudagraphs": True}, tools)
    assert recipe == {"runtime.cudagraphs": True}
    # Setting an option back to its default drops it, so equal recipes match.
    assert apply(recipe, {"runtime.cudagraphs": False}, tools) == {}
    assert digest(apply({"engine.level": 3}, {}, tools)) == digest({})
    with pytest.raises(ValueError, match="Unknown tool"):
        apply({}, {"engine.speed": 1}, tools)
    with pytest.raises(ValueError, match="does not allow"):
        apply({}, {"engine.level": 4}, tools)
    with pytest.raises(ValueError, match="does not allow"):
        apply({}, {"engine.level": True}, tools)
    with pytest.raises(ValueError, match="unknown tool"):
        apply({"engine.speed": 1}, {}, tools)
    assert resolve({"engine.level": 5}, tools) == {
        "runtime.cudagraphs": False,
        "engine.level": 5,
    }
    assert difference({"engine.level": 5}, {"runtime.cudagraphs": True}, tools) == {
        "runtime.cudagraphs": True,
        "engine.level": 3,
    }
    with pytest.raises(ValueError, match="defined twice"):
        catalog([Tool("engine.level", (3,), 3, ""), Tool("engine.level", (5,), 5, "")])


def test_compare_outputs_reports_the_worst_leaf():
    reference = {"a": torch.ones(4), "ids": torch.tensor([1, 2, 3, 4])}
    same = compare_outputs(
        {"a": torch.ones(4), "ids": torch.tensor([1, 2, 3, 4])}, reference
    )
    assert same["max_abs_error"] == 0 and same["match_fraction"] == 1
    assert same["cosine_similarity"] == pytest.approx(1.0)
    off = compare_outputs(
        {"a": torch.tensor([1.0, 1.0, 1.0, 2.0]), "ids": torch.tensor([1, 2, 0, 0])},
        reference,
    )
    assert off["max_abs_error"] == pytest.approx(1.0)
    assert off["rel_l2_error"] == pytest.approx(0.5)
    assert off["match_fraction"] == pytest.approx(0.5)
    broken = compare_outputs(
        {"a": torch.full((4,), math.nan), "ids": reference["ids"]}, reference
    )
    assert broken["max_abs_error"] == math.inf and broken["cosine_similarity"] == -1.0
    with pytest.raises(ValueError, match="structure"):
        compare_outputs([torch.ones(4)], reference)
    with pytest.raises(ValueError, match="shape"):
        compare_outputs({"a": torch.ones(5), "ids": reference["ids"]}, reference)


def test_module_case_tools_default_to_its_compile_options():
    from torch_tensorrt.agentic import ModuleCase

    case = ModuleCase(
        torch.nn.Linear(2, 2),
        [torch.ones(1, 2)],
        compile_options={"min_block_size": 1, "optimization_level": 4},
    )
    tools = catalog(case.tools())
    assert tools["graph.min_block_size"].default == 1
    assert tools["engine.optimization_level"].default == 4
    assert tools["engine.optimization_level"].values == (3, 5, 4)
    unlisted = ModuleCase(
        torch.nn.Linear(2, 2), [torch.ones(1, 2)], compile_options={"min_block_size": 7}
    )
    assert catalog(unlisted.tools())["graph.min_block_size"].values == (7, 5, 1, 3, 10)
    assert resolve({}, tools)["graph.min_block_size"] == 1
    with pytest.raises(AttributeError):
        ModuleCase(
            torch.nn.Linear(2, 2), [], regions={"x": "missing"}, generator=object()
        )
