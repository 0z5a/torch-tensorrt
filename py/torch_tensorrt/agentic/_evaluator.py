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


def gpu_utilization() -> float | None:
    """This process's GPU's utilization in percent from nvidia-smi; None if unknown."""
    binary = shutil.which("nvidia-smi")
    if binary is None or not torch.cuda.is_available():
        return None
    query = ["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
    with contextlib.suppress(Exception):
        uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
        query.insert(0, "--id=" + (uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"))
    try:
        output = subprocess.run(
            [binary, *query], capture_output=True, text=True, timeout=30, check=True
        ).stdout
        return float(output.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def wait_for_quiet_gpu(
    max_wait_s: float, *, threshold: float = 5.0, settle: int = 2
) -> dict[str, Any]:
    """Wait until the GPU is idle for ``settle`` consecutive one-second samples.

    Run between this process's own work, so any load it sees belongs to other
    jobs. ``quiet`` is None when utilization cannot be read.
    """
    record: dict[str, Any] = {"waited_s": 0.0, "quiet": None}
    if max_wait_s <= 0:
        return record
    synchronize()
    start = time.monotonic()
    quiet = 0
    while True:
        utilization = gpu_utilization()
        if utilization is None:
            return record
        quiet = quiet + 1 if utilization <= threshold else 0
        waited = time.monotonic() - start
        if quiet >= settle or waited >= max_wait_s:
            record.update(waited_s=waited, quiet=quiet >= settle)
            return record
        time.sleep(1.0)


# Once the quiet-GPU wait is spent, each further attempt still looks this long.
_GPU_LOOK_S = 3.0

_PEAK_READ_GBPS: float | None = None


def peak_read_gbps() -> float | None:
    """Achievable DRAM read bandwidth of this GPU in GB/s, measured once per process.

    A reduction over 512 MiB, more than any GPU's L2 cache, reads every byte
    from device memory once.
    """
    global _PEAK_READ_GBPS
    if _PEAK_READ_GBPS is None and torch.cuda.is_available():
        try:
            data = torch.ones(256 * 2**20, dtype=torch.float16, device="cuda")
        except RuntimeError:  # not enough free memory
            return None
        best_ms = float("inf")
        for _ in range(6):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            data.sum(dtype=torch.float32)
            end.record()
            end.synchronize()
            best_ms = min(best_ms, start.elapsed_time(end))
        _PEAK_READ_GBPS = data.numel() * data.element_size() / best_ms / 1e6
        del data
        torch.cuda.empty_cache()
    return _PEAK_READ_GBPS


def roofline(bytes_per_unit: float, latency_ms: float) -> dict[str, Any] | None:
    """The memory-bandwidth floor for ``bytes_per_unit`` and how close the latency gets."""
    peak = peak_read_gbps()
    if not peak or bytes_per_unit <= 0:
        return None
    floor_ms = bytes_per_unit / (peak * 1e9) * 1e3
    return {
        "bytes_per_unit": float(bytes_per_unit),
        "peak_read_gbps": peak,
        "floor_ms": floor_ms,
        "fraction": floor_ms / latency_ms if latency_ms > 0 else None,
    }


def _spread(samples: list[float]) -> float:
    """Interquartile range relative to the median."""
    summary = summarize(samples)
    return (summary["p75"] - summary["p25"]) / summary["median"]


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
    measurement: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build, check, measure and (optionally) profile and export one candidate.

    Stages are ``build``, ``setup``, ``check``, ``time``, ``profile`` and
    ``export``; a failure records its stage and error and ends the evaluation.
    Accuracy is checked on the first call and again after timing, so outputs
    replayed from a captured CUDA graph are checked too.

    ``measurement`` (``quiet_gpu_s``, ``retries``, ``max_spread``; see
    :class:`~torch_tensorrt.agentic.Measurement`) makes timing wait for an
    idle GPU (``quiet_gpu_s`` in total) and re-time when the samples spread
    too widely or the GPU was busy. A measurement is stable when its last
    attempt's spread is within ``max_spread`` and the GPU was not seen busy
    before it. A candidate with ``bytes_per_unit`` also gets its
    memory-bandwidth ``roofline``.
    """
    settings = dict(measurement or {})
    quiet_gpu_s = float(settings.get("quiet_gpu_s", 0.0))
    max_spread = float(settings.get("max_spread", float("inf")))
    retries = int(settings.get("retries", 0))
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
        spreads: list[float] = []
        waited_s, quiet = 0.0, None
        for _ in range(1 + retries):
            # quiet_gpu_s bounds the whole wait; once it is spent, a short look
            # still tells whether the GPU is busy for this attempt.
            gate = wait_for_quiet_gpu(
                max(quiet_gpu_s - waited_s, _GPU_LOOK_S) if quiet_gpu_s > 0 else 0.0
            )
            waited_s += gate["waited_s"]
            quiet = gate["quiet"]
            # Warm up right before timing: the wait may have let clocks drop.
            for _ in range(timing.warmup):
                candidate()
            samples = _time_rounds(candidate, timing)
            spreads.append(_spread(samples))
            if spreads[-1] <= max_spread and quiet is not False:
                break
        result["measurement"] = {
            "attempts": len(spreads),
            "spreads": spreads,
            "gpu_wait_s": waited_s,
            "gpu_quiet": quiet,
            "stable": spreads[-1] <= max_spread and quiet is not False,
        }
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
        # After the memory readings: the bandwidth probe allocates 512 MiB.
        bytes_per_unit = getattr(candidate, "bytes_per_unit", None)
        if bytes_per_unit:
            line = roofline(float(bytes_per_unit), latency["median"])
            if line is not None:
                result["roofline"] = line
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
        result.get("profile"),
        result.get("structure", {}),
        values,
        result.get("roofline"),
    )
    return result
