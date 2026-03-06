"""HF-side monkey patch entry for the SGLang Triton attention bridge."""

import types
import torch


_dumper = None


def _maybe_dump(name: str, value: torch.Tensor) -> None:
    """Lazily import sglang dumper and dump a tensor."""
    global _dumper
    if _dumper is False:
        return
    if _dumper is None:
        try:
            _dumper = __import__(
                "sglang.srt.debug_utils.dumper",
                fromlist=["dumper"],
            ).dumper
        except Exception:
            _dumper = False
            return
    _dumper.dump(name, value)


def _is_patchable_attention(module) -> bool:
    return (
        hasattr(module, "q_proj")
        and hasattr(module, "k_proj")
        and hasattr(module, "v_proj")
        and hasattr(module, "o_proj")
    )


def run_unified_extend(
    q_varlen: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    batch: int,
    seq_len: int,
) -> torch.Tensor:
    """Execute extend_attention_fwd_unified in teacher-forcing prefill mode."""
    from sglang.srt.layers.attention.triton_ops.extend_attention import (
        extend_attention_fwd_unified,
    )

    device = q_varlen.device
    o = torch.empty_like(q_varlen)
    qo_indptr = torch.arange(0, batch + 1, device=device, dtype=torch.int32) * seq_len
    kv_indptr = qo_indptr.clone()
    kv_indices = torch.arange(batch * seq_len, device=device, dtype=torch.int64)
    prefix_lens = torch.zeros(batch, device=device, dtype=torch.int32)

    extend_attention_fwd_unified(
        q_varlen,
        o,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        prefix_lens,
        max_len_extend=seq_len,
        is_causal=True,
    )
    return o


def apply_sglang_triton_attention_patch(model, enable_dump=True):
    """Patch HF attention modules to use the SGLang Triton unified-extend path.

    Sets dump attributes on layer 0 and last layer so that the patched forward
    can emit debug tensors matching SGLang's dump names.
    """
    from .models.qwen3 import qwen3_triton_forward

    # Determine num_hidden_layers from config
    num_hidden_layers = None
    config = getattr(model, "config", None)
    if config is not None:
        num_hidden_layers = getattr(config, "num_hidden_layers", None)

    patched = 0
    for _name, module in model.named_modules():
        if not _is_patchable_attention(module):
            continue
        if getattr(module, "_sglang_triton_patched", False):
            continue

        module.forward = types.MethodType(qwen3_triton_forward, module)
        module._sglang_triton_patched = True

        # Set dump attributes for layer 0 and last layer
        if enable_dump:
            layer_id = getattr(module, "layer_idx", None)
            if layer_id is not None:
                is_layer0 = layer_id == 0
                is_last = (
                    num_hidden_layers is not None
                    and layer_id == num_hidden_layers - 1
                )
                if is_layer0 or is_last:
                    module._dump_layer_id = layer_id
                    module._dump_is_last_layer = is_last

        patched += 1

    return patched
