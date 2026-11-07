# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import json
from pathlib import Path

import pytest
import torch

import torch_tensorrt
from torch_tensorrt.agentic import Budget, OptimizationSession, Timing, Workload
from torch_tensorrt.agentic._evaluator import evaluate
from torch_tensorrt.agentic._tools import catalog, resolve

from ._cases import create_mlp

CASES = Path(__file__).with_name("_cases.py")


def worker_session(tmp_path, factory="create_fake", kwargs=None, timeout=300.0):
    return OptimizationSession(
        Workload(f"{CASES}:{factory}", kwargs or {}),
        directory=tmp_path / "s",
        budget=Budget(max_trials=0, trial_timeout_s=timeout),
    )


def test_worker_processes_run_jobs_and_report_failures(tmp_path):
    session = worker_session(tmp_path)
    session.directory.mkdir(parents=True)
    reference = session._run_worker(
        {"kind": "reference"}, session.directory / "reference"
    )
    assert reference["status"] == "ok" and reference["case"]["name"] == "fake"
    assert reference["environment"]["torch"] == torch.__version__
    trial = session._run_worker(
        {
            "kind": "trial",
            "recipe": {},
            "timing": {"warmup": 0, "rounds": 3, "calls": 1, "profile_calls": 0},
        },
        session.directory / "trial",
    )
    assert trial["status"] == "ok" and trial["accuracy"]["max_abs_error"] == 0
    assert (session.directory / "trial" / "worker.log").is_file()

    broken = worker_session(tmp_path / "broken", factory="missing_factory")
    failed = broken._run_worker({"kind": "describe"}, tmp_path / "broken" / "describe")
    assert failed["status"] == "failed" and failed["stage"] == "setup"
    assert "missing_factory" in failed["error"]


def test_a_worker_past_its_timeout_is_killed(tmp_path):
    session = worker_session(tmp_path, kwargs={"milliseconds": 60_000}, timeout=20.0)
    session.directory.mkdir(parents=True)
    session._run_worker({"kind": "reference"}, session.directory / "reference")
    result = session._run_worker(
        {
            "kind": "trial",
            "recipe": {},
            "timing": {"warmup": 0, "rounds": 3, "calls": 1, "profile_calls": 0},
        },
        session.directory / "slow",
    )
    assert result == {
        "status": "failed",
        "stage": "timeout",
        "error": "The worker exceeded trial_timeout_s=20",
    }


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@requires_cuda
@pytest.mark.parametrize("cudagraphs", [False, True])
def test_module_case_is_checked_measured_and_profiled(cudagraphs):
    case = create_mlp()
    tools = catalog(case.tools())
    result = evaluate(
        case,
        resolve({"runtime.cudagraphs": cudagraphs}, tools),
        case.reference(),
        Timing(warmup=1, rounds=3, calls=2, profile_calls=1),
    )
    assert result["status"] == "ok", result
    assert result["accuracy"]["cosine_similarity"] > 0.9999
    assert result["accuracy_after_timing"]["cosine_similarity"] > 0.9999
    assert result["structure"]["trt_engines"] == 1
    assert result["structure"]["torch_partitions"] == 0
    assert result["latency_ms"]["count"] == 3 and result["latency_ms"]["median"] > 0
    profile = result["profile"]
    assert profile["device_ms"] > 0 and profile["kernels"] >= 1
    assert profile["wall_ms"] == result["latency_ms"]["median"]
    assert 0 <= profile["host_overhead_fraction"] <= 1
    assert {s["signal"] for s in result["signals"]} >= {"precision"}


@requires_cuda
def test_a_session_runs_end_to_end_in_worker_processes(tmp_path):
    session = OptimizationSession(
        Workload(f"{CASES}:create_mlp"),
        directory=tmp_path / "mlp",
        budget=Budget(max_trials=2, trial_timeout_s=900),
        confirm_repeats=1,
    )
    result = session.run()
    trials = sorted((tmp_path / "mlp" / "trials").iterdir())
    assert len(trials) == 3
    for trial in trials:
        assert json.loads((trial / "result.json").read_text())["status"] == "ok"
    assert (tmp_path / "mlp" / "report.md").read_text() == result.report()
    assert result.artifact is not None
    loaded = torch_tensorrt.load(str(result.artifact / "model.ep")).module()
    case = create_mlp()
    torch.testing.assert_close(
        loaded(*case.inputs), case.reference(), rtol=1e-2, atol=1e-2
    )


TRITON_RELU = """import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice

@triton.jit
def generated(arg0, out0, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(arg0 + offsets, offsets < N, other=0.0)
    shifted = libdevice.add_rn(libdevice.mul_rn(values, 0.75), -0.25)
    tl.store(out0 + offsets, tl.where(shifted < 0, 0.0, shifted), offsets < N)
"""


class Activation(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x * 0.75 - 0.25)


class RegionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.act = Activation()

    def forward(self, x):
        return self.act(x + 1) - 0.5


@requires_cuda
@pytest.mark.skipif(
    not torch_tensorrt.ENABLED_FEATURES.qdp_plugin, reason="needs TensorRT QDP"
)
def test_region_tools_generate_kernels_through_the_kernel_generator(
    tmp_path, monkeypatch
):
    pytest.importorskip("triton")
    from torch_tensorrt.agentic import ModuleCase
    from torch_tensorrt.dynamo.region_compilation._generation import _providers
    from torch_tensorrt.region import KernelGenerator

    requests = []

    def author(**request):
        requests.append(request)
        candidate = {
            "source": TRITON_RELU,
            "kernel_name": "generated",
            "constants": [{"name": "N", "value": 771}, {"name": "BLOCK", "value": 128}],
            "grid": [7],
            "num_warps": 4,
            "num_stages": None,
        }
        return {
            "action": "propose",
            "candidates": [candidate],
            "resource": None,
            "offset": None,
        }

    monkeypatch.setattr(_providers, "propose", author)
    generator = KernelGenerator(
        cache_dir=tmp_path,
        max_attempts=1,
        timeout_s=180,
        candidate_timeout_s=90,
        num_tests=2,
        benchmark_iters=3,
    )
    with pytest.raises(ValueError, match="need a KernelGenerator"):
        ModuleCase(RegionModel(), [], regions={"act": "act"})
    case = ModuleCase(
        RegionModel().cuda(),
        [torch.linspace(-2, 2, 771, device="cuda").reshape(3, 257)],
        regions={"act": "act"},
        generator=generator,
        compile_options={"min_block_size": 1},
    )
    tools = catalog(case.tools())
    assert tools["kernels.act"].values == ("native", "generated")
    timing = Timing(warmup=1, rounds=3, calls=2, profile_calls=0)
    native = evaluate(case, resolve({}, tools), case.reference(), timing)
    assert native["status"] == "ok" and native["structure"]["plugin_regions"] == 0
    assert not requests
    generated = evaluate(
        case, resolve({"kernels.act": "generated"}, tools), case.reference(), timing
    )
    assert generated["status"] == "ok", generated
    assert generated["structure"]["plugin_regions"] == 1
    assert generated["accuracy"]["max_abs_error"] < 1e-5
    assert len(requests) == 1 and "aten.relu" in requests[0]["prompt"]
    # The user's module is not rewritten.
    assert "forward" not in vars(case.module.act)
