# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnose: where a candidate's time goes, and which bottleneck it shows."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Callable, Mapping

import torch

# Thresholds for reporting a bottleneck signal. They choose a starting point
# for the search; every change is still judged by whole-workload measurement.
LAUNCH_OVERHEAD_FRACTION = 0.2
DOMINANT_KERNEL_SHARE = 0.35
# Below this fraction of the memory-bandwidth floor a memory-bound workload
# has room for better kernels or fewer bytes.
ROOFLINE_FRACTION = 0.8
COPY_SHARE = 0.03
_TOP_KERNELS = 8


def profile_calls(
    call: Callable[[], Any], calls: int, units: int, wall_ms: float
) -> dict[str, Any]:
    """Trace ``calls`` calls of ``call`` and summarize them per work unit.

    ``device_ms`` sums GPU kernel and copy time. Tracing slows the host, so the
    host overhead compares it with ``wall_ms``, the untraced latency per unit:
    ``1 - device_ms / wall_ms`` estimates the share of wall time the GPU waits
    for host work. ``host_syncs`` counts device-to-host copies, which stall the
    host.
    """
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    from torch_tensorrt.agentic._evaluator import synchronize

    synchronize()
    with torch.profiler.profile(activities=activities) as profiler:
        start = time.perf_counter()
        for _ in range(calls):
            call()
        synchronize()
        traced_s = time.perf_counter() - start
    per_unit = max(calls * units, 1)
    kernels: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    device_us = copy_us = 0.0
    launches = syncs = 0
    for event in profiler.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        total = getattr(event, "device_time_total", None)
        if total is None:
            total = getattr(event, "cuda_time_total", 0.0)
        duration = float(total or 0.0)
        device_us += duration
        name = event.name
        if name.startswith("Memcpy") or name.startswith("Memset"):
            syncs += int("DtoH" in name)
            copy_us += duration
        else:
            launches += 1
        kernels[name][0] += duration
        kernels[name][1] += 1
    device_ms = device_us / 1e3 / per_unit
    top = sorted(kernels.items(), key=lambda item: -item[1][0])[:_TOP_KERNELS]
    return {
        "wall_ms": wall_ms,
        "traced_wall_ms": traced_s * 1e3 / per_unit,
        "device_ms": device_ms,
        "host_overhead_fraction": (
            max(0.0, 1.0 - device_ms / wall_ms) if wall_ms > 0 else 0.0
        ),
        "kernels": launches / per_unit,
        "host_syncs": syncs / per_unit,
        "copy_ms": copy_us / 1e3 / per_unit,
        "top_kernels": [
            {
                "name": name if len(name) <= 120 else name[:117] + "...",
                "ms": total / 1e3 / per_unit,
                "share": total / device_us if device_us else 0.0,
                "calls": count / per_unit,
            }
            for name, (total, count) in top
        ],
    }


def signals(
    profile: Mapping[str, Any] | None,
    structure: Mapping[str, Any],
    values: Mapping[str, Any],
    roofline: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Bottleneck signals, strongest first.

    Each names the evidence and a strength in [0, 1] used to order the search;
    the mapping from signal to layer follows the session's four layers.
    """
    found: list[dict[str, Any]] = []
    if roofline and roofline.get("fraction") is not None:
        fraction = float(roofline["fraction"])
        if fraction < ROOFLINE_FRACTION:
            found.append(
                {
                    "signal": "roofline_gap",
                    "strength": min(1.0, 1.0 - fraction),
                    "detail": (
                        f"it runs at {fraction:.0%} of memory-bandwidth-bound speed: "
                        f"its latency is {1.0 / fraction:.2g}x the floor of "
                        f"{roofline['floor_ms']:.3g} ms "
                        f"({roofline['bytes_per_unit'] / 1e9:.3g} GB per unit at "
                        f"{roofline['peak_read_gbps']:.0f} GB/s)"
                    ),
                }
            )
    if profile:
        copies = float(profile.get("copy_ms", 0.0))
        device = float(profile.get("device_ms", 0.0))
        if device > 0 and copies / device >= COPY_SHARE:
            found.append(
                {
                    "signal": "copy_overhead",
                    "strength": min(1.0, copies / device * 5),
                    "detail": (
                        f"{copies / device:.0%} of GPU time is memory copies "
                        f"({copies:.3g} ms per unit)"
                    ),
                }
            )
        overhead = float(profile.get("host_overhead_fraction", 0.0))
        if overhead >= LAUNCH_OVERHEAD_FRACTION:
            found.append(
                {
                    "signal": "launch_overhead",
                    "strength": overhead,
                    "detail": (
                        f"{overhead:.0%} of wall time is host work the GPU waits for "
                        f"({profile.get('kernels', 0):.0f} kernel launches per unit)"
                    ),
                }
            )
        host_syncs = float(profile.get("host_syncs", 0.0))
        if host_syncs >= 1:
            found.append(
                {
                    "signal": "host_syncs",
                    "strength": min(1.0, 0.5 + host_syncs / 100),
                    "detail": (
                        f"{host_syncs:.0f} device-to-host copies per unit stall the host, "
                        "prevent CPU/GPU overlap and usually block CUDA-graph capture"
                    ),
                }
            )
        top = (profile.get("top_kernels") or [{}])[0]
        if top.get("share", 0.0) >= DOMINANT_KERNEL_SHARE:
            found.append(
                {
                    "signal": "dominant_kernel",
                    "strength": float(top["share"]),
                    "detail": (
                        f"{top['share']:.0%} of GPU time is spent in one kernel "
                        f"({top['calls']:.3g} launches per unit): {top['name']}"
                    ),
                }
            )
    partitions = int(structure.get("torch_partitions", 0) or 0)
    engines = int(structure.get("trt_engines", 0) or 0)
    if partitions or engines > 1:
        found.append(
            {
                "signal": "fallback",
                "strength": min(1.0, 0.3 + 0.1 * (partitions + engines)),
                "detail": (
                    f"{engines} TensorRT engines and {partitions} PyTorch partitions; "
                    "every boundary costs a launch and blocks fusion"
                ),
            }
        )
    if values.get("engine.precision") == "fp32":
        found.append(
            {
                "signal": "precision",
                "strength": 0.4,
                "detail": "TensorRT layers run in FP32",
            }
        )
    return sorted(found, key=lambda signal: -signal["strength"])
