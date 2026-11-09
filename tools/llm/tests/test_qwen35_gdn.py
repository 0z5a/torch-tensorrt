# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""qwen35_gdn and qwen35_prefill: the functional Gated DeltaNet and the workload logic.

Run with ``python -m pytest tools/llm/tests/test_qwen35_gdn.py``. Everything
runs on the CPU with tiny random models and no download: the Triton scan
kernel runs in Triton's interpreter and is compiled (not run) for the RTX
5090 to check its shared memory. Only the test that runs the kernel with its
production launch settings needs a GPU.
"""

import math
import os
import sys
import types

import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import qwen35_gdn  # noqa: E402
import qwen35_prefill  # noqa: E402
from torch_tensorrt.agentic._tools import apply, catalog, resolve  # noqa: E402

hf = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")


def _rule_inputs(length, heads=2, dim=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    shape = (1, length, heads, dim)
    q, k, v = (torch.randn(shape, generator=generator) for _ in range(3))
    g = -2 * torch.rand(1, length, heads, generator=generator)
    beta = torch.rand(1, length, heads, generator=generator)
    return q, k, v, g, beta


def _scan_inputs(rows, chunks, c, dk=128, dv=128, device="cpu", seed=0):
    generator = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, chunks, c, dk, generator=generator) * 0.05
    u = torch.randn(rows, chunks, c, dv, generator=generator)
    k = torch.randn(rows, chunks, c, dk, generator=generator) * 0.05
    gamma = torch.rand(rows, chunks, generator=generator)
    return tuple(x.to(device) for x in (w, u, k, gamma))


def _run_scan_kernel(kernel, w, u, k, gamma, launch, tf32, **options):
    """Launch ``gdn_scan_kernel`` as the plugin does; returns (states, values)."""
    rows, chunks, c, dk = w.shape
    dv = u.shape[-1]
    states = torch.empty(rows, chunks, dk, dv, device=w.device)
    values = torch.empty(rows, chunks, c, dv, device=w.device)
    kernel[(dv // launch["block_v"], rows)](
        w,
        u,
        k,
        gamma,
        states,
        values,
        **qwen35_gdn._scan_constexprs(chunks, c, dk, dv, launch, tf32),
        **options,
    )
    return states, values


@pytest.mark.parametrize("size", [2, 3, 16, 64, 100])
def test_unit_lower_inverse_matches_a_triangular_solve(size):
    lower = 0.3 * torch.randn(3, size, size, dtype=torch.float64).tril(-1)
    eye = torch.eye(size, dtype=torch.float64)
    expected = torch.linalg.solve_triangular(
        eye + lower, eye.expand(3, size, size), upper=False
    )
    torch.testing.assert_close(qwen35_gdn.unit_lower_inverse(lower), expected)


def test_unit_lower_inverse_stays_finite_where_the_neumann_product_overflows():
    # Positively correlated unit keys with beta ~ 1, as in real layers: the
    # inverse is bounded, but N^32 of the Neumann product reaches ~1e17.
    generator = torch.Generator().manual_seed(0)
    keys = torch.randn(64, 8, generator=generator, dtype=torch.float64) + 3
    keys = keys / keys.norm(dim=-1, keepdim=True)
    lower = (keys @ keys.T).tril(-1) * 0.99
    expected = torch.linalg.inv(torch.eye(64, dtype=torch.float64) + lower)
    actual = qwen35_gdn.unit_lower_inverse(lower.float()).double()
    assert torch.isfinite(actual).all()
    assert (actual - expected).abs().max() < 1e-5


@pytest.mark.parametrize(
    "length, chunk", [(100, 64), (128, 32), (257, 64), (64, 128), (5, 2)]
)
def test_chunk_rule_matches_hugging_face(length, chunk):
    q, k, v, g, beta = _rule_inputs(length)
    expected, _ = hf.torch_chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True
    )
    recurrent, _ = hf.torch_recurrent_gated_delta_rule(
        q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True
    )
    actual = qwen35_gdn.chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=chunk)
    assert actual.shape == expected.shape and actual.dtype == q.dtype
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(actual, recurrent, atol=2e-6, rtol=1e-5)


def test_chunk_rule_returns_the_input_dtype():
    q, k, v, g, beta = (x.bfloat16() for x in _rule_inputs(64))
    out = qwen35_gdn.chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=32)
    assert out.dtype == torch.bfloat16


def test_triton_scan_runs_its_reference_body_outside_compilation(monkeypatch):
    q, k, v, g, beta = _rule_inputs(96)
    loop = qwen35_gdn.chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=32)
    region = qwen35_gdn.chunk_gated_delta_rule(
        q, k, v, g, beta, chunk_size=32, scan="triton"
    )
    torch.testing.assert_close(region, loop)
    launch = {"block_v": 32, "block_rows": 64, "num_warps": 4, "num_stages": 1}
    monkeypatch.setattr(qwen35_gdn, "SCAN_LAUNCH", {"ieee": launch, "tf32": launch})
    with pytest.raises(ValueError, match="block_v"):
        qwen35_gdn.chunk_gated_delta_rule(q, k, v, g, beta, scan="triton")


def test_scan_launch_blocks_long_chunks_into_rows():
    for tf32 in (False, True):
        assert qwen35_gdn.scan_launch(128, 128, tf32)["rows"] == 64
        assert qwen35_gdn.scan_launch(64, 128, tf32)["rows"] == 64
        assert qwen35_gdn.scan_launch(32, 128, tf32)["rows"] == 32
        with pytest.raises(ValueError, match="at least 16"):
            qwen35_gdn.scan_launch(8, 128, tf32)


# Runs in a fresh process after the helpers above: Triton's interpreter needs
# TRITON_INTERPRET=1 before triton.language is first imported (its tl.zeros
# is itself a @triton.jit function).
_INTERPRETER_CHECK = """
kernel = qwen35_gdn.gdn_scan_kernel_factory()
for chunk, block_rows in ((128, 64), (64, 16), (32, 64)):
    w, u, k, gamma = _scan_inputs(rows=2, chunks=3, c=chunk)
    launch = {"block_v": 16, "rows": min(chunk, block_rows)}
    states, values = _run_scan_kernel(kernel, w, u, k, gamma, launch, tf32=False)
    expected = qwen35_gdn.scan_states(
        w[None], u[None], k[None], gamma[None].unsqueeze(-1)
    )
    torch.testing.assert_close(states, expected[0][0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(values, expected[1][0], atol=1e-5, rtol=1e-5)
    print("ok", chunk, block_rows)
"""


def test_triton_scan_row_blocks_match_the_loop_in_the_interpreter():
    import inspect
    import subprocess

    llm = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = "\n".join(
        (
            f"import sys\nsys.path.insert(0, {llm!r})",
            "import torch\nimport qwen35_gdn",
            inspect.getsource(_scan_inputs),
            inspect.getsource(_run_scan_kernel),
            _INTERPRETER_CHECK,
        )
    )
    env = {**os.environ, "TRITON_INTERPRET": "1", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.count("ok") == 3


@pytest.mark.parametrize("chunk", qwen35_gdn.CHUNK_SIZES)
@pytest.mark.parametrize("tf32", [False, True])
def test_scan_kernel_fits_the_plugin_launch_on_an_rtx_5090(chunk, tf32):
    from triton.backends.compiler import GPUTarget

    resources = qwen35_gdn.check_scan_kernel(
        2048 // chunk, chunk, 128, 128, tf32, GPUTarget("cuda", 120, 32)
    )
    assert 0 < resources["shared"] <= qwen35_gdn.SHARED_MEMORY_LIMIT
    assert resources["global_scratch_size"] == resources["profile_scratch_size"] == 0


def test_check_scan_kernel_refuses_launches_over_48_kb(monkeypatch):
    from triton.backends.compiler import GPUTarget

    # Two pipeline stages need about 77 KB at chunk 64 (faster in isolation,
    # but the TensorRT 11.4 plugin launch fails at enqueue).
    launch = {"block_v": 16, "block_rows": 64, "num_warps": 8, "num_stages": 2}
    monkeypatch.setitem(qwen35_gdn.SCAN_LAUNCH, "ieee", launch)
    with pytest.raises(ValueError, match="shared memory"):
        qwen35_gdn.check_scan_kernel(
            32, 64, 128, 128, False, GPUTarget("cuda", 120, 32)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("chunk", qwen35_gdn.CHUNK_SIZES)
@pytest.mark.parametrize("tf32", [False, True])
def test_triton_scan_kernel_matches_the_loop_with_its_launch_settings(chunk, tf32):
    w, u, k, gamma = _scan_inputs(rows=3, chunks=5, c=chunk, device="cuda")
    launch = qwen35_gdn.scan_launch(chunk, 128, tf32)
    kernel = qwen35_gdn.gdn_scan_kernel_factory()
    states, values = _run_scan_kernel(
        kernel,
        w,
        u,
        k,
        gamma,
        launch,
        tf32,
        num_warps=launch["num_warps"],
        num_stages=launch["num_stages"],
    )
    expected = qwen35_gdn.scan_states(
        w[None], u[None], k[None], gamma[None].unsqueeze(-1)
    )
    # TF32 products keep 10 mantissa bits: about 1e-3 relative per product.
    tolerance = 2e-3 if tf32 else 1e-5
    for actual, reference in zip((states, values), expected):
        error = (actual - reference[0]).norm() / reference[0].norm()
        assert error < tolerance, (chunk, tf32, error.item())


def test_conv_variants_match_hugging_face():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(1, 12, 37, generator=generator)
    weight = torch.randn(12, 4, generator=generator)
    expected = hf.causal_conv1d_fn(x, weight, activation="silu").transpose(1, 2)
    tokens = x.transpose(1, 2)  # token-major [B, T, D]
    for conv in qwen35_gdn.CONVS:
        torch.testing.assert_close(
            qwen35_gdn.causal_conv1d_silu(tokens, weight, conv), expected
        )
    with pytest.raises(ValueError):
        qwen35_gdn.causal_conv1d_silu(tokens, weight, "fft")


def _tiny_config(**overrides):
    from transformers import Qwen3_5TextConfig

    settings = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
    )
    settings.update(overrides)
    return Qwen3_5TextConfig(**settings)


@pytest.mark.parametrize(
    "settings",
    [
        qwen35_gdn.GDNSettings(),
        qwen35_gdn.GDNSettings(chunk_size=32, conv="shifts"),
        qwen35_gdn.GDNSettings(chunk_size=128),
    ],
)
def test_installed_model_matches_hugging_face(settings):
    from transformers import Qwen3_5ForCausalLM

    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(_tiny_config()).eval()
    ids = torch.randint(0, 128, (1, 75))
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
        assert qwen35_gdn.install(model, settings) == 3
        actual = model(ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    layer = qwen35_gdn.gdn_layers(model)[0]
    with pytest.raises(ValueError, match="cache-free"):
        layer(torch.zeros(1, 4, 64), cache_params=object())


def test_causal_attention_matches_the_materialized_mask():
    from transformers import Qwen3_5ForCausalLM

    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(_tiny_config()).eval()
    ids = torch.randint(0, 128, (1, 33))
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
        name = qwen35_prefill.attention_implementation("causal")
        model.config._attn_implementation = name
        actual = model(ids, use_cache=False).logits
    assert qwen35_prefill.attention_implementation("mask") == "sdpa"
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    with pytest.raises(ValueError):
        qwen35_prefill.attention_implementation("sliding")


def test_scan_values_select_kernel_settings():
    case = qwen35_prefill.create_case(length=128, positions=8)
    assert case.gdn_settings({}).scan == "loop"
    triton = case.gdn_settings({"kernels.gdn_scan": "triton"})
    assert (triton.scan, triton.scan_tf32) == ("triton", False)
    tf32 = case.gdn_settings(
        {"kernels.gdn_scan": "triton-tf32", "workload.chunk_size": 32}
    )
    assert (tf32.scan, tf32.scan_tf32, tf32.chunk_size) == ("triton", True, 32)
    for launch in qwen35_gdn.SCAN_LAUNCH.values():
        assert launch["block_v"] >= 16 and 128 % launch["block_v"] == 0


def test_patch_slot_fences_the_gdn_numerics_and_loads_patched_files(tmp_path):
    from torch_tensorrt.agentic._patches import editable_blocks

    case = qwen35_prefill.create_case(length=128, positions=8)
    (slot,) = case.patch_slots()
    assert slot.name == "gdn" and slot.path == os.path.realpath(qwen35_gdn.__file__)
    assert "scratch" in slot.description and "48 KB" in slot.description
    blocks = editable_blocks(open(slot.path).read())
    fenced = "".join(blocks)
    for name in (
        "SCAN_LAUNCH",
        "unit_lower_inverse",
        "scan_states(",
        "scan_kernel(",
        "causal_conv1d_silu(",
        "functional_forward(",
    ):
        assert name in fenced
    # The resource check stays outside the fences, so a patch cannot remove it.
    assert "def check_scan_kernel" not in fenced
    assert case.gdn_module({}) is qwen35_gdn
    patched = tmp_path / "gdn.py"
    source = open(slot.path).read()
    patched.write_text(source.replace('"num_warps": 8', '"num_warps": 4'))
    module = case.gdn_module({"patch.gdn": str(patched)})
    assert module.__name__ == "qwen35_gdn_patch"
    assert module.SCAN_LAUNCH["ieee"]["num_warps"] == 4
    settings = case.gdn_settings({"workload.conv": "shifts"}, module)
    assert type(settings) is module.GDNSettings and settings.conv == "shifts"


def test_prompt_never_repeats_a_paragraph():
    text = qwen35_prefill.prompt_text(20000)
    assert len(text) == 20000
    paragraphs = [p for p in text.split("\n\n") if p.strip()][:-1]
    assert len(paragraphs) == len(set(paragraphs)) > 10


def test_prefill_logits_select_positions():
    from transformers import Qwen3_5ForCausalLM

    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(_tiny_config()).eval()
    ids = torch.randint(0, 128, (1, 40))
    positions = qwen35_prefill.logit_positions(40, 4)
    with torch.no_grad():
        full = model(ids, use_cache=False).logits[0]
        selected = qwen35_prefill.PrefillLogits(model, positions)(ids)
    torch.testing.assert_close(selected, full[positions])


def test_gdn_settings_validate_their_options():
    with pytest.raises(ValueError):
        qwen35_gdn.GDNSettings(chunk_size=48)
    with pytest.raises(ValueError):
        qwen35_gdn.GDNSettings(scan="parallel")
    with pytest.raises(ValueError):
        qwen35_gdn.GDNSettings(conv="fft")


def test_logit_positions_are_evenly_spaced_and_end_at_the_last_token():
    positions = qwen35_prefill.logit_positions(2048, 64)
    assert len(positions) == 64 and positions[-1] == 2047 and positions[0] == 31
    assert {b - a for a, b in zip(positions, positions[1:])} == {32}
    assert qwen35_prefill.logit_positions(10, 3) == [2, 5, 9]
    with pytest.raises(ValueError):
        qwen35_prefill.logit_positions(4, 5)


def test_catalog_baseline_and_requirements():
    case = qwen35_prefill.create_case(length=128, positions=8)
    tools = catalog(case.tools())
    # The baseline exports Hugging Face's own Gated DeltaNet and attention.
    baseline = resolve(case.baseline(), tools)
    assert baseline["workload.gdn"] == "hf"
    assert baseline["workload.attention"] == "mask"
    assert baseline["engine.precision"] == "bf16"
    assert tools["kernels.gdn_scan"].values == ("loop", "triton", "triton-tf32")
    assert tools["kernels.gdn_scan"].requires == (
        ("runtime.cudagraphs", True),
        ("workload.gdn", "functional"),
    )
    with pytest.raises(ValueError, match="requires"):
        apply({}, {"kernels.gdn_scan": "triton", "runtime.cudagraphs": True}, tools)
    recipe = apply(
        {},
        {
            "kernels.gdn_scan": "triton",
            "runtime.cudagraphs": True,
            "workload.gdn": "functional",
            # Every chunk size combines with the scan (row-blocked kernel).
            "workload.chunk_size": 128,
        },
        tools,
    )
    assert recipe["kernels.gdn_scan"] == "triton"
    for name in ("workload.chunk_size", "workload.conv"):
        assert tools[name].requires == (("workload.gdn", "functional"),)
    with pytest.raises(ValueError, match="requires"):
        apply({}, {"workload.chunk_size": 32}, tools)
    assert tools["engine.use_fp32_acc"].requires == (("engine.precision", "fp16"),)
    with pytest.raises(ValueError, match="requires"):
        apply({}, {"engine.use_fp32_acc": False}, tools)
    # Generic builder and runtime tools are offered; the partitioning tool is not.
    assert {"runtime.cudagraphs", "engine.optimization_level"} <= set(tools)
    assert "graph.min_block_size" not in tools


def test_accuracy_defaults_bound_mean_and_largest_kl():
    limits = qwen35_prefill.create_case(length=128, positions=8).accuracy_defaults()
    assert limits == {
        "top1_agreement": (">=", 0.95),
        "mean_kl": ("<=", 0.002),
        "max_kl": ("<=", 0.01),
    }


def test_build_rejects_unknown_backends_before_loading_anything():
    case = qwen35_prefill.create_case(length=128, positions=8)
    with pytest.raises(ValueError, match="bench.backend"):
        case.build({"bench.backend": "onnx"})


def test_bench_settings_validate_keys_modes_and_backends():
    settings = qwen35_prefill.bench_settings(
        {"bench.backend": "inductor", "bench.mode": "max-autotune", "x.y": 1}
    )
    assert settings.cudagraphs and settings.attn is None
    assert not qwen35_prefill.bench_settings({"bench.backend": "eager"}).cudagraphs
    assert qwen35_prefill.bench_settings({}).backend == "tensorrt"
    for values, match in (
        ({"bench.gdn": "functional"}, "Unknown bench"),
        ({"bench.tf32": True}, "Unknown bench"),
        ({"bench.backend": "inductor", "bench.mode": "fast"}, "bench.mode"),
        ({"bench.backend": "eager", "bench.mode": "max-autotune"}, "do not apply"),
        ({"bench.attn": "eager"}, "do not apply"),
        ({"bench.backend": "inductor", "bench.inductor": [1]}, "mapping"),
    ):
        with pytest.raises(ValueError, match=match):
            qwen35_prefill.bench_settings(values)
    # The default mode may be spelled out for any backend.
    qwen35_prefill.bench_settings({"bench.backend": "eager", "bench.mode": "default"})


@pytest.mark.parametrize(
    "values, match",
    [
        ({"kernels.gdn_scan": "triton"}, "only apply to TensorRT"),
        ({"kernels.inductor": "attention"}, "only apply to TensorRT"),
        ({"engine.decompose_attention": True}, "only apply to TensorRT"),
        ({"runtime.cudagraphs": True}, "only apply to TensorRT"),
        ({"engine.optimization_level": 5}, "only apply to TensorRT"),
        ({"workload.attention": "causal", "bench.attn": "eager"}, "conflicts"),
    ],
)
def test_torch_backends_refuse_what_they_cannot_honor_before_loading(values, match):
    case = qwen35_prefill.create_case(length=128, positions=8)
    for backend in ("eager", "inductor"):
        with pytest.raises(ValueError, match=match):
            case.build({**values, "bench.backend": backend})


def test_torch_matmul_flags_are_restored():
    matmul, cudnn = torch.backends.cuda.matmul, torch.backends.cudnn
    before = (matmul.allow_tf32, cudnn.allow_tf32, matmul.allow_fp16_accumulation)
    with qwen35_prefill._torch_matmul_flags(tf32=True, fp16_accumulation=True):
        assert matmul.allow_tf32 and cudnn.allow_tf32
        assert matmul.allow_fp16_accumulation
    assert (
        matmul.allow_tf32,
        cudnn.allow_tf32,
        matmul.allow_fp16_accumulation,
    ) == before


def test_compare_reports_agreement_kl_and_non_finite_outputs():
    case = qwen35_prefill.create_case(length=128, positions=4)
    reference = torch.randn(4, 50)
    same = case.compare(reference.clone(), reference)
    assert same["top1_agreement"] == 1.0 and abs(same["mean_kl"]) < 1e-12
    worse = case.compare(reference + torch.randn(4, 50), reference)
    assert worse["mean_kl"] > 0 and worse["max_abs_logit_error"] > 0
    broken = reference.clone()
    broken[0, 0] = float("nan")
    metrics = case.compare(broken, reference)
    assert metrics["top1_agreement"] == 0.0 and math.isinf(metrics["mean_kl"])
    with pytest.raises(ValueError, match="shape"):
        case.compare(reference[:2], reference)


def test_prefill_flops_count_projections_attention_and_gdn():
    config = types.SimpleNamespace(
        hidden_size=1024,
        intermediate_size=3584,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        layer_types=["linear_attention"] * 18 + ["full_attention"] * 6,
        vocab_size=248320,
    )
    flops = qwen35_prefill.prefill_flops(config, 2048, 64, 64)
    # About 0.5B non-embedding parameters at 2 FLOPs per token each, plus
    # attention, Gated DeltaNet chunk math and 64 rows of the 254M-entry head.
    assert 2.0e12 < flops < 2.6e12
    assert qwen35_gdn.flops_per_layer(2048, 16, 128, 128, 64) > 0


# ------------------------------------------------- kernels.inductor (hybrid)
_HYBRID = {
    "workload.gdn": "functional",
    "workload.attention": "causal",
    "runtime.cudagraphs": True,
}


def test_inductor_tool_offers_the_hybrid_variants_with_their_requirements():
    case = qwen35_prefill.create_case(length=128, positions=8)
    tools = catalog(case.tools())
    tool = tools["kernels.inductor"]
    assert tool.layer == "kernels" and tool.default == "none"
    assert tool.values == (
        "none",
        "attention",
        "prologue",
        "prologue-decay",
        "attention+prologue",
        "attention+prologue-decay",
    )
    assert dict(tool.requires) == _HYBRID
    # Requirements, boundaries and costs are described; measured results are not.
    for text in (
        "12 engine boundaries",
        "36 boundaries",
        "48",
        "runtime.cudagraphs",
        "memory",
        "emulate_precision_casts",
        "patch.gdn",
        "not exported",
    ):
        assert text in tool.description
    assert " ms" not in tool.description and "%" not in tool.description
    assert " MB" not in tool.description.replace("120 MB per layer", "")
    # The default leaves every recipe as it was.
    assert resolve({}, tools)["kernels.inductor"] == "none"
    assert "kernels.inductor" not in apply({}, {"kernels.inductor": "none"}, tools)
    for missing in _HYBRID:
        partial = {k: v for k, v in _HYBRID.items() if k != missing}
        with pytest.raises(ValueError, match="requires"):
            apply({}, {**partial, "kernels.inductor": "attention"}, tools)
    for value in tool.values[1:]:
        recipe = apply({}, {**_HYBRID, "kernels.inductor": value}, tools)
        assert recipe["kernels.inductor"] == value


_INDUCTOR_REGIONS = {
    "none": (False, "tensorrt"),
    "attention": (True, "tensorrt"),
    "prologue": (False, "inductor"),
    "prologue-decay": (False, "inductor-decay"),
    "attention+prologue": (True, "inductor"),
    "attention+prologue-decay": (True, "inductor-decay"),
}


def test_inductor_regions_name_the_operations_each_value_moves():
    for value, expected in _INDUCTOR_REGIONS.items():
        assert qwen35_prefill.inductor_regions(value) == expected
    with pytest.raises(ValueError, match="kernels.inductor"):
        qwen35_prefill.inductor_regions("mlp")
    case = qwen35_prefill.create_case(length=128, positions=8)
    assert case.gdn_settings({}).prologue == "tensorrt"
    for value, (_, prologue) in _INDUCTOR_REGIONS.items():
        settings = case.gdn_settings({**_HYBRID, "kernels.inductor": value})
        assert settings.prologue == prologue
        assert prologue in qwen35_gdn.PROLOGUES
    with pytest.raises(ValueError, match="prologue"):
        qwen35_gdn.GDNSettings(prologue="eager")


@pytest.mark.parametrize(
    "values, match",
    [
        ({"kernels.inductor": "attention"}, "kernels.inductor needs"),
        (
            {**_HYBRID, "workload.attention": "mask", "kernels.inductor": "prologue"},
            "kernels.inductor needs",
        ),
        (
            {**_HYBRID, "runtime.cudagraphs": False, "kernels.inductor": "attention"},
            "kernels.inductor needs",
        ),
        ({**_HYBRID, "kernels.inductor": "mlp"}, "kernels.inductor="),
    ],
)
def test_hybrid_build_refuses_unmet_requirements_before_loading(values, match):
    case = qwen35_prefill.create_case(length=128, positions=8)
    with pytest.raises(ValueError, match=match):
        case.build(values)


def test_prologue_values_refuse_patches_they_would_not_run(tmp_path):
    case = qwen35_prefill.create_case(length=128, positions=8)
    conflicts = qwen35_prefill.prologue_patch_conflicts
    source = open(qwen35_gdn.__file__).read()

    def patched(name, old, new):
        assert source.count(old) == 1
        path = tmp_path / f"{name}.py"
        path.write_text(source.replace(old, new))
        return str(path), case.gdn_module({"patch.gdn": str(path)})

    for prologue in qwen35_gdn.PROLOGUES:
        assert conflicts(qwen35_gdn, prologue) == []
    # Edits inside functional_forward and chunk_gated_delta_rule are left out.
    forward_path, forward = patched(
        "forward",
        "self.conv1d.weight.squeeze(1), settings.conv)",
        "self.conv1d.weight.squeeze(1), settings.conv)  # edited",
    )
    rule_path, rule = patched(
        "rule",
        "    if use_qk_l2norm:\n        q, k = l2norm(q), l2norm(k)",
        "    if use_qk_l2norm:\n        q, k = l2norm(q, 1e-5), l2norm(k, 1e-5)",
    )
    for prologue in ("inductor", "inductor-decay"):
        assert conflicts(forward, prologue) == ["functional_forward"]
        assert conflicts(rule, prologue) == ["chunk_gated_delta_rule"]
    # The TensorRT prologue (values 'none' and 'attention') runs every edit.
    assert conflicts(forward, "tensorrt") == conflicts(rule, "tensorrt") == []
    # Edits that the prologue builds do run: the conv, launch settings.
    _, conv = patched(
        "conv",
        "taps = weight.to(torch.float32).t()  # [K, D]",
        "taps = weight.to(torch.float32).t()  # [K, D] edited",
    )
    _, launch = patched("launch", '"num_warps": 8', '"num_warps": 4')
    for module in (conv, launch):
        assert conflicts(module, "inductor-decay") == []
    _, old = patched(
        "old", "def prologue_region_forward(", "def _prologue_region_forward("
    )
    assert "predates" in conflicts(old, "inductor")[0]
    # The build refuses before loading anything.
    session = {**_HYBRID, "kernels.gdn_scan": "triton-tf32", "workload.conv": "shifts"}
    for value in ("prologue", "attention+prologue-decay"):
        for path in (forward_path, rule_path):
            with pytest.raises(ValueError, match="does not run the patch.gdn edits"):
                case.build({**session, "kernels.inductor": value, "patch.gdn": path})


def test_hybrid_candidates_are_not_exported(tmp_path):
    candidate = qwen35_prefill._TensorRTPrefill(
        torch.nn.Identity(),
        torch.zeros(1, 4, dtype=torch.int64),
        cudagraphs=True,
        pre_allocated_outputs=False,
        structure={"inductor_regions": "attention"},
    )
    with pytest.raises(RuntimeError, match="not exported"):
        candidate.export(tmp_path)
    assert not list(tmp_path.iterdir())


def test_matmul_policy_follows_the_recipe():
    policy = qwen35_prefill.matmul_policy
    assert policy({}) == {"tf32": False, "fp16_accumulation": False}
    session = {
        "engine.precision": "fp16",
        "engine.use_fp32_acc": False,
        "engine.disable_tf32": False,
    }
    assert policy(session) == {"tf32": True, "fp16_accumulation": True}
    # FP16 accumulation needs FP16 weights and use_fp32_acc=False.
    assert not policy({**session, "engine.precision": "bf16"})["fp16_accumulation"]
    assert not policy({**session, "engine.use_fp32_acc": True})["fp16_accumulation"]


def test_tensorrt_candidate_sets_region_flags_only_while_open():
    matmul = torch.backends.cuda.matmul
    before = (matmul.allow_tf32, matmul.allow_fp16_accumulation)
    flags = {"tf32": not before[0], "fp16_accumulation": not before[1]}
    candidate = qwen35_prefill._TensorRTPrefill(
        torch.nn.Identity(),
        torch.zeros(1, 4, dtype=torch.int64),
        cudagraphs=False,
        pre_allocated_outputs=False,
        structure={},
        region_flags=flags,
    )
    candidate.setup()
    try:
        assert (matmul.allow_tf32, matmul.allow_fp16_accumulation) == (
            flags["tf32"],
            flags["fp16_accumulation"],
        )
    finally:
        candidate.close()
    assert (matmul.allow_tf32, matmul.allow_fp16_accumulation) == before
    # Without Inductor regions the candidate leaves the flags alone.
    plain = qwen35_prefill._TensorRTPrefill(
        torch.nn.Identity(),
        torch.zeros(1, 4, dtype=torch.int64),
        cudagraphs=False,
        pre_allocated_outputs=False,
        structure={},
    )
    plain.setup()
    assert (matmul.allow_tf32, matmul.allow_fp16_accumulation) == before
    plain.close()


@pytest.mark.parametrize(
    "length, chunk, scan", [(100, 64, "loop"), (128, 32, "triton"), (5, 2, "loop")]
)
def test_chunk_rule_head_and_body_compute_the_rule(length, chunk, scan):
    q, k, v, g, beta = _rule_inputs(length, heads=3)
    expected = qwen35_gdn.chunk_gated_delta_rule(
        q, k, v, g, beta, chunk_size=chunk, scan=scan
    )
    # The prologue's layout: head-major, q and k L2-normalized in FP32.
    q_head, k_head = (qwen35_gdn.l2norm(x.transpose(1, 2)) for x in (q, k))
    head = qwen35_gdn.chunk_rule_head(
        q_head, k_head, v.transpose(1, 2), g, beta, chunk
    )
    actual = qwen35_gdn.chunk_rule_body(*head, length=length, scan=scan)
    assert actual.shape == expected.shape and actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("prologue", ["inductor", "inductor-decay"])
@pytest.mark.parametrize("conv", qwen35_gdn.CONVS)
def test_prologue_region_forward_matches_the_functional_layer(conv, prologue):
    from transformers import Qwen3_5ForCausalLM

    torch.manual_seed(0)
    # Four value heads per two key heads: the prologue repeats q and k.
    model = Qwen3_5ForCausalLM(_tiny_config()).eval()
    ids = torch.randint(0, 128, (1, 75))
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
        qwen35_gdn.install(model, qwen35_gdn.GDNSettings(chunk_size=32, conv=conv))
        functional = model(ids, use_cache=False).logits
        settings = qwen35_gdn.GDNSettings(chunk_size=32, conv=conv, prologue=prologue)
        assert qwen35_gdn.install(model, settings) == 3
        layer = qwen35_gdn.gdn_layers(model)[0]
        assert layer.forward.__func__ is qwen35_gdn.prologue_region_forward
        hybrid = model(ids, use_cache=False).logits
        # The prologue's outputs: contiguous head-major, q and k in FP32.
        hidden = torch.randn(1, 75, 64)
        q, k, v = qwen35_gdn.gdn_prologue(layer, hidden, conv)
    assert q.shape == k.shape == (1, 4, 75, 16) and v.shape == (1, 4, 75, 8)
    assert q.dtype == k.dtype == torch.float32 and v.dtype == hidden.dtype
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    torch.testing.assert_close(hybrid, functional, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(hybrid, expected, atol=1e-5, rtol=1e-4)
    with pytest.raises(ValueError, match="cache-free"):
        layer(torch.zeros(1, 4, 64), cache_params=object())


def test_inductor_attention_matches_causal_attention():
    from transformers import Qwen3_5ForCausalLM

    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(_tiny_config()).eval()
    ids = torch.randint(0, 128, (1, 33))
    with torch.no_grad():
        model.config._attn_implementation = qwen35_prefill.attention_implementation(
            "causal"
        )
        causal = model(ids, use_cache=False).logits
        name = qwen35_prefill.attention_implementation("causal", inductor=True)
        assert name != model.config._attn_implementation
        model.config._attn_implementation = name
        hybrid = model(ids, use_cache=False).logits
    torch.testing.assert_close(hybrid, causal, atol=0, rtol=0)
    with pytest.raises(ValueError, match="needs 'causal'"):
        qwen35_prefill.attention_implementation("mask", inductor=True)


def _capture_regions(module, *inputs):
    """Strict export with region capture, as torch_tensorrt.compile does it."""
    from torch_tensorrt.dynamo.region_compilation import (
        check_decomposed_regions,
        materialize_regions,
        normalize_region_scopes,
    )
    from torch_tensorrt.region._session import RegionCompilationSession

    with torch.no_grad(), RegionCompilationSession(strict=True):
        program = torch.export.export(module, inputs, strict=True)
        program = normalize_region_scopes(program)
        captured = program.graph_module
        program = program.run_decompositions()
        check_decomposed_regions(captured, program.graph_module)
        return materialize_regions(program.module(), captured)


@pytest.mark.parametrize(
    "value, names",
    [
        ("attention", ["attn_core"]),
        ("prologue", ["gdn_prologue"] * 3),
        ("prologue-decay", ["gdn_prologue"] * 3),
        ("attention+prologue", ["gdn_prologue"] * 3 + ["attn_core"]),
        ("attention+prologue-decay", ["gdn_prologue"] * 3 + ["attn_core"]),
    ],
)
def test_strict_export_captures_one_inductor_region_per_layer(value, names):
    from torch_tensorrt.dynamo.region_compilation import get_region_records
    from transformers import Qwen3_5ForCausalLM

    attention, prologue = qwen35_prefill.inductor_regions(value)
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(_tiny_config()).eval()
    settings = qwen35_gdn.GDNSettings(chunk_size=16, conv="shifts", prologue=prologue)
    qwen35_gdn.install(model, settings)
    model.config._attn_implementation = qwen35_prefill.attention_implementation(
        "causal", inductor=attention
    )
    wrapper = qwen35_prefill.PrefillLogits(model, [10, 39]).eval()
    ids = torch.randint(0, 128, (1, 40))
    graph = _capture_regions(wrapper, ids)
    records = get_region_records(graph)
    assert [r.name for r in records] == names
    assert {(r.backend, r.route) for r in records} == {("inductor", "torch")}
    for record in records:
        body = graph.get_submodule(record.child_target)
        targets = {str(n.target) for n in body.graph.nodes if n.op == "call_function"}
        if record.name == "attn_core":
            # Captured before Torch-TensorRT's decompositions: Inductor gets SDPA.
            assert "aten.scaled_dot_product_attention.default" in targets
        else:
            assert {"aten.linear.default", "aten.rsqrt.default"} <= targets
            # 'prologue' leaves the gates and the decays to TensorRT.
            decay = {"aten.sigmoid.default", "aten.cumsum.default"}
            if prologue == "inductor-decay":
                assert decay <= targets
            else:
                assert not decay & targets
    with torch.no_grad():
        torch.testing.assert_close(graph(ids), wrapper(ids), atol=1e-5, rtol=1e-5)
