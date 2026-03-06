"""Qwen3 minimal attention patch for the SGLang Triton bridge.

Only replaces the attention kernel (SDPA -> SGLang Triton extend_attention_fwd_unified).
All other operations (projections, norms, rope) use HF's original implementations.

Dump format: [total_tokens, dim] to match SGLang's varlen dump format, where
  dim = num_heads * head_dim for q, and num_kv_heads * head_dim for k/v.
"""

import torch

from ..hf_sglang_triton_patch import run_unified_extend, _maybe_dump


def _resolve_rotary():
    try:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        return apply_rotary_pos_emb
    except Exception:
        try:
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

            return apply_rotary_pos_emb
        except Exception:
            return None


APPLY_ROTARY_POS_EMB = _resolve_rotary()


def _dump_flat(name, tensor):
    """Dump tensor in [total_tokens, dim] format, collapsing all leading dims."""
    _maybe_dump(name, tensor.reshape(-1, tensor.shape[-1]))


def qwen3_triton_forward(
    self,
    hidden_states,
    position_embeddings=None,
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    """Minimal Qwen3 attention patch — only replaces the attention kernel.

    Dump points mirror SGLang's qwen3.py exactly (all in [total_tokens, dim]):
      - layer 0:    layer0_{q,k,v}_pre_norm, layer0_{q,k}_post_norm,
                    layer0_{q,k}_post_rope, layer0_attn_context_before_o_proj,
                    layer0_attn_out_after_o_proj
      - last layer: {q,k,v}_pre_norm, {q,k}_post_norm, {q,k}_post_rope,
                    attn_context_before_o_proj, attn_out_last_layer
    """
    input_shape = hidden_states.shape[:-1]  # [batch, seq_len]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Dump control attributes (set during patching)
    layer_id = getattr(self, "_dump_layer_id", None)
    is_last = getattr(self, "_dump_is_last_layer", False)

    # --- HF original: Q/K/V projections ---
    # Output shape: [batch, seq_len, num_heads * head_dim]
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)

    # Dump pre_norm in [total_tokens, num_heads * head_dim] — before view
    if layer_id == 0:
        _dump_flat("layer0_q_pre_norm", q)
        _dump_flat("layer0_k_pre_norm", k)
        _dump_flat("layer0_v_pre_norm", v)
    if is_last:
        _dump_flat("q_pre_norm", q)
        _dump_flat("k_pre_norm", k)
        _dump_flat("v_pre_norm", v)

    # Reshape to [batch, seq_len, num_heads, head_dim]
    q = q.view(hidden_shape)
    k = k.view(hidden_shape)
    v = v.view(hidden_shape)

    # --- HF original: Q/K norms ---
    q = self.q_norm(q)
    k = self.k_norm(k)

    # Dump post_norm: [B, seq, num_heads, head_dim] -> [total_tokens, num_heads * head_dim]
    if layer_id == 0:
        _maybe_dump("layer0_q_post_norm", q.reshape(q.shape[0] * q.shape[1], -1))
        _maybe_dump("layer0_k_post_norm", k.reshape(k.shape[0] * k.shape[1], -1))
    if is_last:
        _maybe_dump("q_post_norm", q.reshape(q.shape[0] * q.shape[1], -1))
        _maybe_dump("k_post_norm", k.reshape(k.shape[0] * k.shape[1], -1))

    q = q.transpose(1, 2)  # [B, num_heads, seq, head_dim]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # --- HF original: RoPE ---
    if position_embeddings is not None and APPLY_ROTARY_POS_EMB is not None:
        cos, sin = position_embeddings
        q, k = APPLY_ROTARY_POS_EMB(q, k, cos, sin)

    # Dump post_rope: [B, num_heads, seq, head_dim] -> [total_tokens, num_heads * head_dim]
    if layer_id == 0:
        _maybe_dump(
            "layer0_q_post_rope",
            q.permute(0, 2, 1, 3).reshape(q.shape[0] * q.shape[2], -1),
        )
        _maybe_dump(
            "layer0_k_post_rope",
            k.permute(0, 2, 1, 3).reshape(k.shape[0] * k.shape[2], -1),
        )
    if is_last:
        _maybe_dump(
            "q_post_rope",
            q.permute(0, 2, 1, 3).reshape(q.shape[0] * q.shape[2], -1),
        )
        _maybe_dump(
            "k_post_rope",
            k.permute(0, 2, 1, 3).reshape(k.shape[0] * k.shape[2], -1),
        )

    # === ONLY CHANGE: replace SDPA with SGLang Triton kernel ===
    batch, num_heads, seq_len, head_dim = q.shape
    num_kv_heads = k.shape[1]
    total_tokens = batch * seq_len

    q_varlen = (
        q.to(torch.bfloat16)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(total_tokens, num_heads, head_dim)
    )
    k_buffer = (
        k.to(torch.bfloat16)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(total_tokens, num_kv_heads, head_dim)
    )
    v_buffer = (
        v.to(torch.bfloat16)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(total_tokens, num_kv_heads, head_dim)
    )

    o = run_unified_extend(q_varlen, k_buffer, v_buffer, batch, seq_len)

    # o shape: [total_tokens, num_heads, head_dim]
    # Reshape to [batch, seq, num_heads * head_dim] for dump and o_proj
    attn_output = o.view(batch, seq_len, num_heads * head_dim)

    if layer_id == 0:
        _dump_flat("layer0_attn_context_before_o_proj", attn_output)
    if is_last:
        _dump_flat("attn_context_before_o_proj", attn_output)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if layer_id == 0:
        _dump_flat("layer0_attn_out_after_o_proj", attn_output)
    if is_last:
        _dump_flat("attn_out_last_layer", attn_output)

    return attn_output, None
