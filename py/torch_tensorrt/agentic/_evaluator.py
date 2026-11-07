# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""The independent evaluator: the same checks and measurements for every candidate.

It runs inside a worker process and never sees the planner. A candidate's
result is what this module measures, not what the planner expects.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import statistics
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import torch

from torch_tensorrt.agentic._case import Candidate, Case
from torch_tensorrt.agentic._profile import profile_calls, signals
from torch_tensorrt.agentic._spec import Timing


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
    return {
        "median": statistics.median(ordered),
        "p25": quartiles[0],
        "p75": quartiles[2],
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "count": len(ordered),
    }


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_rounds(candidate: Candidate, timing: Timing) -> list[float]:
    """Milliseconds per work unit, one sample per round of back-to-back calls."""
    units = max(int(candidate.units), 1)
    samples = []
    for _ in range(timing.rounds):
        synchronize()
        start = time.perf_counter()
        for _ in range(timing.calls):
            candidate()
        synchronize()
        samples.append((time.perf_counter() - start) * 1e3 / (timing.calls * units))
    return samples


def process_gpu_memory_mb() -> float | None:
    """GPU memory this process holds, from NVML via nvidia-smi; None if unknown.

    It includes TensorRT's own allocations, which PyTorch's allocator does not
    see. In containers whose process IDs differ from the host's, it is unknown.
    """
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    try:
        output = subprocess.run(
            [
                binary,
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in output.splitlines():
        pid, _, used = line.partition(",")
        if pid.strip() == str(os.getpid()):
            with contextlib.suppress(ValueError):
                return float(used)
    return None


def _failure(stage: str, error: BaseException) -> dict[str, Any]:
    lines = traceback.format_exception_only(type(error), error)
    message = "".join(lines).strip()
    return {
        "status": "failed",
        "stage": stage,
        "error": message if len(message) <= 2000 else message[:1997] + "...",
    }


def evaluate(
    case: Case,
    values: Mapping[str, Any],
    reference: Any,
    timing: Timing,
    *,
    profile: bool = True,
    export_dir: Path | None = None,
) -> dict[str, Any]:
    """Build, check, measure and (optionally) profile and export one candidate.

    Stages are ``build``, ``setup``, ``check``, ``time``, ``profile`` and
    ``export``; a failure records its stage and error and ends the evaluation.
    Accuracy is checked on the first call and again after timing, so outputs
    replayed from a captured CUDA graph are checked too.
    """
    result: dict[str, Any] = {"status": "ok"}
    candidate: Candidate | None = None
    stage = "build"
    try:
        start = time.perf_counter()
        candidate = case.build(values)
        result["build_s"] = time.perf_counter() - start
        result["structure"] = candidate.describe()
        result["units"] = int(candidate.units)
        result["items"] = int(candidate.items)
        stage = "setup"
        candidate.setup()
        stage = "check"
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        result["accuracy"] = case.compare(candidate(), reference)
        stage = "time"
        for _ in range(timing.warmup):
            candidate()
        samples = _time_rounds(candidate, timing)
        stage = "check"
        result["accuracy_after_timing"] = case.compare(candidate(), reference)
        latency = summarize(samples)
        result["latency_ms"] = latency
        result["throughput"] = result["items"] * 1e3 / latency["median"]
        result["samples_ms"] = samples
        result["memory"] = {
            "process_gpu_mb": process_gpu_memory_mb() if cuda else None,
            "torch_peak_mb": (
                torch.cuda.max_memory_allocated() / 2**20 if cuda else None
            ),
        }
        if profile and timing.profile_calls:
            stage = "profile"
            result["profile"] = profile_calls(
                candidate, timing.profile_calls, result["units"], latency["median"]
            )
        if export_dir is not None:
            stage = "export"
            export_dir.mkdir(parents=True, exist_ok=True)
            result["exported"] = candidate.export(export_dir)
    except Exception as error:  # noqa: BLE001 - every failure becomes evidence
        partial = {
            key: result[key]
            for key in ("build_s", "structure", "units", "items")
            if key in result
        }
        result = {**_failure(stage, error), **partial}
        traceback.print_exc()
    finally:
        if candidate is not None:
            with contextlib.suppress(Exception):
                candidate.close()
    result["signals"] = signals(
        result.get("profile"), result.get("structure", {}), values
    )
    return result
