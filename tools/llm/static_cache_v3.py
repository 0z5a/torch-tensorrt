# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Shape-static KV cache for single-token decode.

``static_cache_v1`` and ``static_cache_v2`` slice the cache to the current
length, so attention runs over a key length that changes every step. TensorRT
resolves those data-dependent shapes on the host: every layer and step costs a
device-to-host copy and a stream synchronization, CPU and GPU work cannot
overlap, and the engine cannot be captured in a CUDA graph.

This pass keeps every shape static for decode (batch 1, one query token):

* each layer's key and value caches are graph inputs of shape
  ``[1, kv_heads, max_seq_len, head_dim]``;
* the new key and value are written at ``position_ids`` with ``index_copy``,
  which TensorRT lowers to an in-place ``IKVCacheUpdateLayer``;
* attention runs over the whole cache with the mask
  ``arange(max_seq_len) <= position_ids`` and native grouped-query attention,
  instead of Hugging Face's repeated key and value heads.

Export :class:`DecodeStep` (the model with positional ``input_ids`` and
``position_ids``). The compiled module, and the program
``torch_tensorrt.save`` writes for it, take ``(input_ids, position_ids,
key_cache_0, value_cache_0, ...)`` and return ``(logits, key_cache_0,
value_cache_0, ...)``; the engine updates the caches in place, so keep passing
the same cache tensors. Prefill runs one token at a time through the same
engine. Import this module and call :func:`configure` before compiling;
importing registers the lowering pass for every compilation in the process.
"""

import logging
from typing import Any, List, Optional, Tuple

import torch
import torch.utils._pytree as pytree
from cache_utils import set_output_spec
from torch._export.utils import _detect_fake_mode_from_gm
from torch.fx import Node
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass import (
    _aten_lowering_pass,
)
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)

SDPA = torch.ops.aten.scaled_dot_product_attention.default
_aten = torch.ops.aten
# Ops through which Hugging Face's repeat_kv expands key and value heads.
_REPEAT_KV_OPS = {
    _aten.expand.default,
    _aten.unsqueeze.default,
    _aten.reshape.default,
    _aten._reshape_copy.default,
    _aten.view.default,
    _aten._unsafe_view.default,
    _aten.clone.default,
    _aten.slice.Tensor,
}

_max_seq_len: Optional[int] = None


class DecodeStep(torch.nn.Module):
    """A causal LM as ``(input_ids, position_ids) -> logits``.

    Exporting Hugging Face's ``forward`` with ``position_ids`` as a keyword
    would leave the cache inputs this pass appends out of order with it in a
    saved program's call signature; positional inputs keep them in order.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> Any:
        return self.model(input_ids, position_ids=position_ids).logits


def configure(max_seq_len: int) -> None:
    """Set the cache length that the next compilations allocate."""
    global _max_seq_len
    if not isinstance(max_seq_len, int) or max_seq_len < 1:
        raise ValueError("max_seq_len must be a positive integer")
    _max_seq_len = max_seq_len


def _grouped_source(node: Node, heads: int) -> Optional[Node]:
    """The key or value before Hugging Face's repeat_kv, if ``node`` comes from it."""
    current: Any = node
    for _ in range(8):
        if not isinstance(current, Node) or current.op != "call_function":
            return None
        if current.target not in _REPEAT_KV_OPS:
            return None
        current = current.args[0]
        value = current.meta.get("val") if isinstance(current, Node) else None
        if (
            isinstance(value, torch.Tensor)
            and value.dim() == 4
            and value.shape[1] < heads
            and heads % value.shape[1] == 0
        ):
            return current
    return None


def _add_cache_input(gm: torch.fx.GraphModule, name: str, value: torch.Tensor) -> Node:
    from cache_utils import _add_graph_input

    return _add_graph_input(gm, name, value)


def _call(
    gm: torch.fx.GraphModule, fake_mode: Any, target: Any, args: Tuple, kwargs: dict
) -> Node:
    node = gm.graph.call_function(target, args, kwargs)

    def fake(argument: Any) -> Any:
        return argument.meta["val"] if isinstance(argument, Node) else argument

    with fake_mode:
        node.meta["val"] = target(
            *pytree.tree_map(fake, args), **pytree.tree_map(fake, kwargs)
        )
    return node


@_aten_lowering_pass
def insert_static_cache_v3(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Give every attention a static KV cache input and output (see the module docstring)."""
    if _max_seq_len is None:
        raise RuntimeError(
            "Call static_cache_v3.configure(max_seq_len) before compiling"
        )
    sdpa_nodes = [
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target == SDPA
    ]
    if not sdpa_nodes:
        return gm
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    position_ids = next(
        (node for node in placeholders if node.name == "position_ids"), None
    )
    if position_ids is None:
        raise ValueError("static_cache_v3 needs a 'position_ids' graph input")
    query = sdpa_nodes[0].args[0].meta["val"]
    if query.shape[0] != 1 or query.shape[2] != 1:
        raise ValueError(
            "static_cache_v3 compiles single-token decode for batch 1; got a query "
            f"of shape {tuple(query.shape)}"
        )
    fake_mode = _detect_fake_mode_from_gm(gm)
    max_seq_len = _max_seq_len

    caches: List[Tuple[Node, Node, Node, Node, bool]] = []
    for index, sdpa in enumerate(sdpa_nodes):
        q, k, v = sdpa.args[:3]
        heads = q.meta["val"].shape[1]
        k_source, v_source = _grouped_source(k, heads), _grouped_source(v, heads)
        grouped = k_source is not None and v_source is not None
        if not grouped:
            k_source, v_source = k, v
        value = k_source.meta["val"]
        shape = (value.shape[0], value.shape[1], max_seq_len, value.shape[3])
        k_cache = _add_cache_input(
            gm,
            f"key_cache_{index}",
            torch.zeros(shape, dtype=value.dtype, device=value.device),
        )
        v_cache = _add_cache_input(
            gm,
            f"value_cache_{index}",
            torch.zeros(shape, dtype=value.dtype, device=value.device),
        )
        caches.append((k_cache, v_cache, k_source, v_source, grouped))

    # Lowering can leave weight get_attr nodes ahead of the placeholders, so
    # anchor on the node after the last placeholder, not the first non-placeholder.
    last_input = [node for node in gm.graph.nodes if node.op == "placeholder"][-1]
    with gm.graph.inserting_before(last_input.next):
        write_index = _call(
            gm, fake_mode, _aten.reshape.default, (position_ids, [-1]), {}
        )
        key_positions = _call(
            gm,
            fake_mode,
            _aten.arange.start_step,
            (0, max_seq_len),
            {"dtype": torch.int64, "device": query.device},
        )
        last_position = _call(
            gm, fake_mode, _aten.reshape.default, (position_ids, [-1, 1, 1, 1]), {}
        )
        mask = _call(gm, fake_mode, _aten.le.Tensor, (key_positions, last_position), {})

    new_outputs = []
    for sdpa, (k_cache, v_cache, k_source, v_source, grouped) in zip(
        sdpa_nodes, caches
    ):
        with gm.graph.inserting_before(sdpa):
            keys = _call(
                gm,
                fake_mode,
                _aten.index_copy.default,
                (k_cache, 2, write_index, k_source),
                {},
            )
            values = _call(
                gm,
                fake_mode,
                _aten.index_copy.default,
                (v_cache, 2, write_index, v_source),
                {},
            )
            attention = _call(
                gm,
                fake_mode,
                SDPA,
                (sdpa.args[0], keys, values, mask, 0.0, False),
                {"scale": sdpa.kwargs.get("scale"), "enable_gqa": grouped},
            )
        sdpa.replace_all_uses_with(attention)
        gm.graph.erase_node(sdpa)
        new_outputs += [keys, values]

    output = next(node for node in gm.graph.nodes if node.op == "output")
    outputs = output.args[0]
    outputs = tuple(outputs) if isinstance(outputs, (tuple, list)) else (outputs,)
    gm.graph.output(outputs + tuple(new_outputs))
    gm.graph.erase_node(output)
    gm = clean_up_graph_after_modifications(gm)
    examples = [
        node.meta["val"] if isinstance(node, Node) else node
        for node in outputs + tuple(new_outputs)
    ]
    set_output_spec(gm, pytree.tree_flatten(tuple(examples))[1])
    logger.debug("After inserting the static v3 KV cache: %s", gm.graph)
    return gm
