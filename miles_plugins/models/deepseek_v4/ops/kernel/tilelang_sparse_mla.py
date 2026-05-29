import torch

from miles_plugins.models.deepseek_v4.ops.kernel import tilelang_sparse_mla_bwd as sparse_mla_bwd
from miles_plugins.models.deepseek_v4.ops.kernel import tilelang_sparse_mla_fwd as sparse_mla_fwd


class DeepSeekV4SparseAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, kv, attn_sink, topk_idxs, sm_scale=None):
        o, lse = sparse_mla_fwd.sparse_mqa_fwd_interface(q, kv, attn_sink, topk_idxs, sm_scale=sm_scale)

        ctx.save_for_backward(q, kv, attn_sink, topk_idxs, o.clone(), lse)
        ctx.sm_scale = sm_scale

        return o

    @staticmethod
    def backward(ctx, do):
        q, kv, attn_sink, topk_idxs, o, lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale

        dq, dkv, d_attn_sink = sparse_mla_bwd.sparse_mqa_bwd_interface(
            q, kv, attn_sink, o, do, topk_idxs, lse, sm_scale=sm_scale
        )

        return dq, dkv, d_attn_sink, None, None


def _dense_mla_reference(q, kv, attn_sink, topk_idxs, sm_scale=None):
    """Pure-torch fp32 reference for the sparse-MQA forward.

    Diagnostic-1 swap (MILES_DSV4_DENSE_ATTN=1). Computes the SAME math the
    tilelang kernel intends, but in plain torch fp32 so we can isolate whether
    the tilelang fwd kernel itself diverges. We honor the exact same sparse set
    (topk_idxs, -1 == masked) rather than attending to all keys, so this tests
    kernel-math, not selection.

    Shapes (matching sparse_mqa_fwd_interface):
        q:         [B, S, H, D]
        kv:        [B, S_kv, D]   (single KV head, MQA)
        attn_sink: [H] fp32       (pre-scaled logit, added to softmax denom)
        topk_idxs: [B, S, topk]   int (-1 = masked)
    Returns o: [B, S, H, D] in q.dtype.
    """
    B, S, H, D = q.shape
    _, S_kv, _ = kv.shape
    topk = topk_idxs.shape[-1]
    if sm_scale is None:
        sm_scale = D**-0.5

    qf = q.float()
    kvf = kv.float()

    idx = topk_idxs.long()
    valid = idx != -1
    gather_idx = idx.clamp(min=0)  # [B, S, topk]

    # Gather selected KV vectors: [B, S, topk, D]
    gathered = torch.gather(
        kvf.unsqueeze(1).expand(B, S, S_kv, D),
        2,
        gather_idx.unsqueeze(-1).expand(B, S, topk, D),
    )

    # scores: [B, S, H, topk] = q . k
    scores = torch.einsum("bshd,bskd->bshk", qf, gathered) * sm_scale
    # mask invalid (-1) entries
    mask = valid.unsqueeze(2)  # [B, S, 1, topk]
    scores = scores.masked_fill(~mask, float("-inf"))

    # softmax with attn_sink term added to the denominator.
    # attn_sink[h] is a pre-scaled logit (same space as scores). Online softmax
    # in the kernel uses base-2; algebraically equivalent to natural-log softmax
    # done here, plus the sink term in the denominator.
    m = scores.amax(dim=-1, keepdim=True)  # [B, S, H, 1]
    m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    exp_scores = torch.exp(scores - m)  # [B, S, H, topk]
    denom = exp_scores.sum(dim=-1)  # [B, S, H]
    sink = attn_sink.float().view(1, 1, H)  # [1,1,H]
    denom = denom + torch.exp(sink - m.squeeze(-1))  # add sink to denominator
    probs = exp_scores / denom.unsqueeze(-1)  # [B, S, H, topk]

    o = torch.einsum("bshk,bskd->bshd", probs, gathered)
    return o.to(q.dtype)


def sparse_attn_tilelang(q, kv, attn_sink, topk_idxs, sm_scale=None):
    import os

    if os.environ.get("MILES_DSV4_DENSE_ATTN") == "1":
        return _dense_mla_reference(q, kv, attn_sink, topk_idxs, sm_scale)
    return DeepSeekV4SparseAttention.apply(q, kv, attn_sink, topk_idxs, sm_scale)
