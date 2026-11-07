# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hugging Face causal-LM decode workload for ``torch_tensorrt.agentic`` sessions.

The measured unit is one decoded token at batch 1. A candidate prefills a fixed
prompt once (untimed) and is then timed decoding ``osl`` tokens. Decoding is
teacher-forced on a fixed public-domain text, so one different token cannot
derail the rest: every step's logits are compared with an FP32 eager reference
at the same position (top-1 agreement, KL divergence and the largest logit
error).

Workload tools:

* ``workload.cache``: the KV cache lowering. ``static_v2`` and ``static_v1``
  (see ``static_cache_v2.py``) share one engine between prefill and decode, but
  slice the cache to the current length, which costs a host synchronization
  per layer and step. ``static_v3`` (``static_cache_v3.py``) compiles a
  shape-static decode engine instead, which has no host synchronizations and
  can be captured in a CUDA graph; its prefill runs one token at a time.
* ``engine.precision`` and ``engine.use_fp32_acc``: model precision, and FP32
  accumulation for FP16 matrix multiplications.

plus the generic builder and runtime tools. Every worker builds exactly one
candidate, so the process-global lowering passes of the cache variants never
mix.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch_tensorrt.agentic import GENERIC_TOOLS, Candidate, Case, Timing, Tool

# Opening of "A Tale of Two Cities" (Charles Dickens, 1859; public domain).
TEXT = (
    "It was the best of times, it was the worst of times, it was the age of wisdom, "
    "it was the age of foolishness, it was the epoch of belief, it was the epoch of "
    "incredulity, it was the season of Light, it was the season of Darkness, it was "
    "the spring of hope, it was the winter of despair, we had everything before us, "
    "we had nothing before us, we were all going direct to Heaven, we were all going "
    "direct the other way - in short, the period was so far like the present period, "
    "that some of its noisiest authorities insisted on its being received, for good "
    "or for evil, in the superlative degree of comparison only. There were a king "
    "with a large jaw and a queen with a plain face, on the throne of England; there "
    "were a king with a large jaw and a queen with a fair face, on the throne of "
    "France. In both countries it was clearer than crystal to the lords of the State "
    "preserves of loaves and fishes, that things in general were settled for ever. "
)

_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
_GENERIC = (
    "runtime.cudagraphs",
    "runtime.pre_allocated_outputs",
    "engine.optimization_level",
    "engine.tiling_optimization_level",
)


def create_case(
    model: str = "Qwen/Qwen3-0.6B",
    isl: int = 128,
    osl: int = 64,
    baseline: Mapping[str, Any] | None = None,
) -> "DecodeCase":
    """Workload factory: ``Workload("tools/llm/agentic_decode.py:create_case", {...})``."""
    return DecodeCase(model, isl=isl, osl=osl, baseline=baseline)


class DecodeCase(Case):
    """Batch-1 greedy decode of a Hugging Face causal LM with a static KV cache.

    The baseline is the documented ``tools/llm`` configuration: FP16 weights,
    ``static_v2`` cache, Python runtime, no CUDA graphs.
    """

    def __init__(
        self,
        model: str,
        *,
        isl: int,
        osl: int,
        baseline: Mapping[str, Any] | None = None,
    ) -> None:
        if isl < 1 or osl < 2:
            raise ValueError("DecodeCase needs isl >= 1 and osl >= 2")
        self.model_name = model
        self.name = f"{model.rsplit('/', 1)[-1]}-decode-isl{isl}-osl{osl}"
        self.isl, self.osl = isl, osl
        self.max_seq = isl + osl
        self._baseline = dict(baseline or {})
        self._tokens: torch.Tensor | None = None

    # ------------------------------------------------------------ contract

    def tools(self) -> Sequence[Tool]:
        return (
            Tool(
                "workload.cache",
                ("static_v2", "static_v3", "static_v1"),
                "static_v2",
                "KV cache lowering. static_v1/static_v2 share one dynamic-shape engine "
                "for prefill and decode and slice the cache to the current length, "
                "which forces a device-to-host synchronization per layer and step. "
                "static_v3 compiles a shape-static decode engine: in-place cache "
                "writes, masked full-length attention, no host synchronizations, "
                "CUDA-graph capturable.",
                ("host_syncs", "launch_overhead"),
            ),
            Tool(
                "engine.precision",
                ("fp16", "bf16", "fp32"),
                "fp16",
                "Model weight and activation precision (fp32 runs without TF32).",
                ("precision", "dominant_kernel"),
            ),
            Tool(
                "engine.use_fp32_acc",
                (True, False),
                True,
                "With fp16, accumulate matrix multiplications in FP32 by inserting "
                "casts; turning it off is faster and less accurate.",
                ("dominant_kernel",),
            ),
            *(tool for tool in GENERIC_TOOLS if tool.name in _GENERIC),
        )

    def baseline(self) -> dict[str, Any]:
        return dict(self._baseline)

    def timing(self) -> Timing:
        return Timing(warmup=2, rounds=10, calls=1, profile_calls=1)

    def accuracy_defaults(self) -> dict[str, tuple[str, float]]:
        return {"top1_agreement": (">=", 0.97), "mean_kl": ("<=", 0.005)}

    def tokens(self) -> torch.Tensor:
        """The fixed token sequence: ``isl`` prompt tokens, then ``osl`` decode inputs."""
        if self._tokens is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            ids = tokenizer(TEXT, return_tensors="pt")["input_ids"][0]
            repeats = -(-self.max_seq // ids.numel())
            self._tokens = ids.repeat(repeats)[: self.max_seq].clone()
        return self._tokens

    def reference(self) -> Any:
        """FP32 eager logits at every decode position, given the same tokens."""
        from transformers import AutoModelForCausalLM

        with torch.inference_mode():
            model = (
                AutoModelForCausalLM.from_pretrained(
                    self.model_name, attn_implementation="sdpa"
                )
                .eval()
                .cuda()
                .float()
            )
            tokens = self.tokens().cuda().unsqueeze(0)
            logits = model(tokens).logits[0, self.isl : self.max_seq].float().cpu()
        del model
        torch.cuda.empty_cache()
        return logits

    def compare(self, outputs: Any, reference: Any) -> dict[str, float]:
        actual = outputs.detach().double().cpu()
        expected = reference.double()
        if actual.shape != expected.shape:
            raise ValueError(
                f"Logits shape {tuple(actual.shape)} != {tuple(expected.shape)}"
            )
        kl = torch.nn.functional.kl_div(
            actual.log_softmax(-1),
            expected.log_softmax(-1),
            log_target=True,
            reduction="none",
        ).sum(-1)
        return {
            "top1_agreement": (actual.argmax(-1) == expected.argmax(-1))
            .double()
            .mean()
            .item(),
            "mean_kl": kl.mean().item(),
            "max_abs_logit_error": (actual - expected).abs().max().item(),
        }

    # -------------------------------------------------------------- build

    def build(self, values: Mapping[str, Any]) -> Candidate:
        import torch_tensorrt
        from transformers import AutoModelForCausalLM

        cache = values["workload.cache"]
        precision = values["engine.precision"]
        device = torch.device("cuda:0")
        with torch.inference_mode():
            model = (
                AutoModelForCausalLM.from_pretrained(
                    self.model_name, use_cache=False, attn_implementation="sdpa"
                )
                .eval()
                .to(device)
                .to(_DTYPES[precision])
            )
        settings = {
            "use_fp32_acc": precision == "fp16" and values["engine.use_fp32_acc"],
            "disable_tf32": True,
            "device": device,
            "use_python_runtime": True,
            "offload_module_to_cpu": True,
            "min_block_size": 1,
            "optimization_level": values["engine.optimization_level"],
            "tiling_optimization_level": values["engine.tiling_optimization_level"],
        }
        tokens = self.tokens().to(device)
        with torch.inference_mode():
            if cache == "static_v3":
                import static_cache_v3

                static_cache_v3.configure(self.max_seq)
                ids = tokens[:1].view(1, 1)
                positions = torch.zeros((1, 1), dtype=torch.int64, device=device)
                program = torch.export.export(
                    static_cache_v3.DecodeStep(model), (ids, positions), strict=False
                )
                module = torch_tensorrt.dynamo.compile(
                    program, arg_inputs=[ids, positions], **settings
                )
            else:
                from torchtrt_ext import register_sdpa
                from utils import export_llm

                register_sdpa.enable_sdpa_converter(self.model_name, model.config)
                if cache == "static_v2":
                    import static_cache_v2  # noqa: F401  (registers the lowering pass)
                else:
                    import static_cache_v1  # noqa: F401  (registers the lowering pass)
                prompt = tokens[: self.isl].unsqueeze(0)
                program = export_llm(model, prompt, max_seq_len=self.max_seq)
                positions = torch.arange(self.isl, device=device).unsqueeze(0)
                module = torch_tensorrt.dynamo.compile(
                    program, arg_inputs=[prompt, positions], **settings
                )
        del model
        torch.cuda.empty_cache()
        return _DecodeCandidate(
            module,
            cache=cache,
            tokens=tokens,
            isl=self.isl,
            osl=self.osl,
            cudagraphs=values["runtime.cudagraphs"],
            pre_allocated_outputs=values["runtime.pre_allocated_outputs"],
        )


class _DecodeCandidate(Candidate):
    def __init__(
        self,
        module: torch.nn.Module,
        *,
        cache: str,
        tokens: torch.Tensor,
        isl: int,
        osl: int,
        cudagraphs: bool,
        pre_allocated_outputs: bool,
    ) -> None:
        self.module = module
        self.cache = cache
        self.isl, self.osl = isl, osl
        self.units = osl
        self.items = 1
        self.cudagraphs = cudagraphs
        self.pre_allocated_outputs = pre_allocated_outputs
        device = tokens.device
        self.step_tokens = [tokens[isl + j].view(1, 1) for j in range(osl)]
        self.step_positions = [
            torch.full((1, 1), isl + j, dtype=torch.int64, device=device)
            for j in range(osl)
        ]
        self.prompt = tokens[:isl].view(1, isl)
        self._stack = contextlib.ExitStack()
        self._start: tuple[torch.Tensor, ...] = ()

    def _zero_caches(self) -> list[torch.Tensor]:
        placeholders = [n for n in self.module.graph.nodes if n.op == "placeholder"]
        caches = placeholders[2:] if self.cache == "static_v3" else placeholders[2:-2]
        return [
            torch.zeros(
                tuple(node.meta["val"].shape),
                dtype=node.meta["val"].dtype,
                device=self.prompt.device,
            )
            for node in caches
        ]

    def _step(
        self, token: torch.Tensor, position: torch.Tensor, index: int, caches: Any
    ) -> tuple[torch.Tensor, Any]:
        """Run one step; return its logits and the caches for the next step.

        A static v3 engine updates its cache inputs in place, so they stay the
        caches; the Python runtime returns clones of in-place outputs, which a
        CUDA graph must not be fed (it keeps the addresses it captured).
        """
        if self.cache == "static_v3":
            out = self.module(token, position, *caches)
            return out[0], caches
        out = self.module(token, position, *caches, index, index + 1)
        return out[0], out[1:]

    def setup(self) -> None:
        import torch_tensorrt

        if self.pre_allocated_outputs:
            self._stack.enter_context(
                torch_tensorrt.runtime.enable_pre_allocated_outputs(self.module)
            )
        if self.cudagraphs:
            torch_tensorrt.runtime.set_cudagraphs_mode(True)
            self._stack.callback(torch_tensorrt.runtime.set_cudagraphs_mode, False)
        with torch.inference_mode():
            caches: Any = self._zero_caches()
            if self.cache == "static_v3":
                for position in range(self.isl):
                    _, caches = self._step(
                        self.prompt[:, position : position + 1],
                        torch.full(
                            (1, 1),
                            position,
                            dtype=torch.int64,
                            device=self.prompt.device,
                        ),
                        position,
                        caches,
                    )
            else:
                positions = torch.arange(self.isl, device=self.prompt.device).unsqueeze(
                    0
                )
                out = self.module(self.prompt, positions, *caches, 0, self.isl)
                caches = out[1:]
            # Every timed call starts from this prefilled state. Static v3 writes
            # its caches in place, only at decode positions and identically on
            # every call, so it keeps these very tensors: a captured CUDA graph
            # binds their addresses. The other caches return new tensors.
            if self.cache == "static_v3":
                self._start = tuple(caches)
            else:
                self._start = tuple(tensor.clone() for tensor in caches)

    def __call__(self) -> torch.Tensor:
        with torch.inference_mode():
            caches: Any = self._start
            logits = []
            for j in range(self.osl):
                step_logits, caches = self._step(
                    self.step_tokens[j], self.step_positions[j], self.isl + j, caches
                )
                logits.append(step_logits[0, -1].to(torch.float32, copy=True))
            return torch.stack(logits)

    def describe(self) -> dict[str, Any]:
        from torch_tensorrt.agentic import describe_module

        return {**describe_module(self.module), "cache": self.cache}

    def export(self, directory: Path) -> list[str]:
        import torch_tensorrt

        path = Path(directory) / "decode.ep"
        torch_tensorrt.save(self.module, str(path), retrace=False)
        return [path.name]

    def close(self) -> None:
        self._stack.close()
