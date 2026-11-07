# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""What a session optimizes: the workload, the goal, the limits and the budget."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import operator
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

if TYPE_CHECKING:
    from torch_tensorrt.agentic._case import Case

# A recipe maps tool names to values; tools left at their default are omitted,
# so identical configurations have identical digests.
Recipe = dict[str, Any]

COMPARATORS: dict[str, Callable[[float, float], bool]] = {
    "<=": operator.le,
    ">=": operator.ge,
    "<": operator.lt,
    ">": operator.gt,
}
METRICS = ("latency_ms", "throughput")


def canonical(value: Any) -> str:
    """Return the stable JSON text identifying ``value``."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _positive(name: str, value: float | None) -> None:
    if value is not None and not (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    ):
        raise ValueError(f"{name} must be a positive number or None; got {value!r}")


@dataclass(frozen=True)
class Workload:
    """The workload every worker process rebuilds: ``factory(**kwargs)``.

    ``factory`` is ``"package.module:function"`` or ``"path/to/file.py:function"``;
    a file's directory is put on ``sys.path`` so its sibling imports work. The
    function returns a :class:`~torch_tensorrt.agentic.Case`. ``kwargs`` must be
    JSON data, because they are recorded with the session.
    """

    factory: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        module, sep, attribute = self.factory.rpartition(":")
        if not sep or not module or not attribute.isidentifier():
            raise ValueError(
                "Workload factory must be 'module:function' or "
                f"'path/to/file.py:function'; got {self.factory!r}"
            )
        if module.endswith(".py"):
            path = Path(module).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Workload factory file {path} does not exist")
            object.__setattr__(self, "factory", f"{path}:{attribute}")
        kwargs = dict(self.kwargs)
        try:
            canonical(kwargs)
        except (TypeError, ValueError) as error:
            raise TypeError(f"Workload kwargs must be JSON data: {error}") from None
        object.__setattr__(self, "kwargs", kwargs)

    def load(self) -> Case:
        """Import the factory and build the case (in the calling process)."""
        from torch_tensorrt.agentic._case import Case

        module_name, _, attribute = self.factory.rpartition(":")
        if module_name.endswith(".py"):
            path = Path(module_name)
            if str(path.parent) not in sys.path:
                sys.path.insert(0, str(path.parent))
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot import workload factory file {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[path.stem] = module
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_name)
        case = getattr(module, attribute)(**self.kwargs)
        if not isinstance(case, Case):
            raise TypeError(
                f"Workload factory {self.factory} returned {type(case).__name__}, "
                "not a torch_tensorrt.agentic.Case"
            )
        return case

    def to_json(self) -> dict[str, Any]:
        return {"factory": self.factory, "kwargs": dict(self.kwargs)}


@dataclass(frozen=True)
class Goal:
    """The objective of a session.

    ``metric`` is ``"latency_ms"`` (milliseconds per work unit, lower is better)
    or ``"throughput"`` (items per second, higher is better); the case defines
    its unit, for example one decoded token. A candidate replaces the best only
    when it improves the metric by more than ``min_improvement`` (relative) and
    beyond the measurement spread. The search stops early once the best reaches
    ``target``.
    """

    metric: str = "latency_ms"
    target: float | None = None
    min_improvement: float = 0.02

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ValueError(
                f"Goal metric must be one of {METRICS}; got {self.metric!r}"
            )
        _positive("Goal target", self.target)
        if not (
            isinstance(self.min_improvement, (int, float))
            and 0 <= self.min_improvement < 1
        ):
            raise ValueError("Goal min_improvement must be in [0, 1)")

    @property
    def lower_is_better(self) -> bool:
        return self.metric == "latency_ms"

    def gain(self, incumbent: float, candidate: float) -> float:
        """Relative improvement of ``candidate`` over ``incumbent``; positive is better."""
        change = (candidate - incumbent) / incumbent
        return -change if self.lower_is_better else change

    def reached(self, value: float) -> bool:
        if self.target is None:
            return False
        return value <= self.target if self.lower_is_better else value >= self.target


@dataclass(frozen=True)
class Constraints:
    """Limits every promoted candidate must satisfy.

    ``accuracy`` maps an accuracy metric reported by the case's ``compare`` to a
    bound, for example ``{"top1_agreement": (">=", 0.99)}``; metrics not listed
    use the case's defaults. ``max_memory_mb`` bounds the candidate's GPU memory
    (see :class:`~torch_tensorrt.agentic.SessionResult` for how it is measured).
    """

    accuracy: Mapping[str, tuple[str, float]] = field(default_factory=dict)
    max_memory_mb: float | None = None

    def __post_init__(self) -> None:
        bounds = {}
        for name, bound in dict(self.accuracy).items():
            comparator, value = _bound(name, bound)
            bounds[name] = (comparator, value)
        object.__setattr__(self, "accuracy", bounds)
        _positive("Constraints max_memory_mb", self.max_memory_mb)

    def bounds(
        self, defaults: Mapping[str, tuple[str, float]]
    ) -> dict[str, tuple[str, float]]:
        merged = {name: _bound(name, bound) for name, bound in defaults.items()}
        merged.update(self.accuracy)
        return merged

    def to_json(self) -> dict[str, Any]:
        return {
            "accuracy": {name: list(bound) for name, bound in self.accuracy.items()},
            "max_memory_mb": self.max_memory_mb,
        }


def _bound(name: str, bound: Any) -> tuple[str, float]:
    if (
        not isinstance(bound, (tuple, list))
        or len(bound) != 2
        or bound[0] not in COMPARATORS
        or isinstance(bound[1], bool)
        or not isinstance(bound[1], (int, float))
        or not math.isfinite(bound[1])
    ):
        raise ValueError(
            f"Accuracy bound for {name!r} must be (comparator, number) with a "
            f"comparator in {tuple(COMPARATORS)}; got {bound!r}"
        )
    return bound[0], float(bound[1])


def violations(
    metrics: Mapping[str, float], bounds: Mapping[str, tuple[str, float]]
) -> list[str]:
    """Describe every bound ``metrics`` miss; a missing or non-finite metric misses."""
    missed = []
    for name, (comparator, value) in sorted(bounds.items()):
        actual = metrics.get(name)
        if not isinstance(actual, (int, float)) or not math.isfinite(actual):
            missed.append(f"{name} is unavailable (needs {comparator} {value:g})")
        elif not COMPARATORS[comparator](actual, value):
            missed.append(f"{name}={actual:.6g} (needs {comparator} {value:g})")
    return missed


@dataclass(frozen=True)
class Budget:
    """How much the session may spend.

    ``max_trials`` counts candidate trials, not the baseline or the final
    confirmation. ``max_time_s`` is wall-clock time across resumes. ``patience``
    stops after that many consecutive trials without a promotion.
    ``trial_timeout_s`` kills a worker that builds or measures for longer.
    """

    max_trials: int = 8
    max_time_s: float | None = None
    patience: int | None = None
    trial_timeout_s: float = 1800.0

    def __post_init__(self) -> None:
        for name in ("max_trials", "patience"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"Budget {name} must be a non-negative integer")
        _positive("Budget max_time_s", self.max_time_s)
        _positive("Budget trial_timeout_s", self.trial_timeout_s)


@dataclass(frozen=True)
class Timing:
    """The evaluator's measurement protocol, the same for every candidate.

    After ``warmup`` untimed calls, each of ``rounds`` rounds times ``calls``
    back-to-back calls and synchronizes once; a round gives one sample of the
    per-unit latency. ``profile_calls`` are traced separately for diagnosis.
    """

    warmup: int = 5
    rounds: int = 15
    calls: int = 3
    profile_calls: int = 3

    def __post_init__(self) -> None:
        for name in ("warmup", "rounds", "calls", "profile_calls"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"Timing {name} must be a non-negative integer")
        if self.rounds < 3 or self.calls < 1:
            raise ValueError("Timing needs at least 3 rounds of at least 1 call")
