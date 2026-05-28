"""Validate that aiter.fmoe_fp8_blockscale_g1u1 can serve a Megatron-style permuted-token
MoE forward via the "topk=1 identity routing" trick.

This is the missing puzzle piece for wiring fmoe into Megatron's TEGroupedMLP. The
Megatron MoE flow, by the time TEGroupedMLP.forward runs, hands the experts a
*pre-permuted* token tensor [T_local, K] (already sorted by local expert) plus a
``tokens_per_expert`` histogram and ``permuted_probs`` [T_local].

But fmoe_fp8_blockscale_g1u1 expects an UN-routed [T, K] tensor + topk_ids and does
its OWN routing internally. To bridge them we feed fmoe:

    input          = permuted_local_hidden_states   # [T_local, K]
    sorted_token_ids = arange(T_local)              # identity: slot i reads input row i
    sorted_expert_ids[block_i] = which_expert handles block i (derived from tokens_per_expert)
    num_valid_ids  = T_local
    sorted_weights = permuted_probs                 # [T_local], post-multiplied by kernel
    topk           = 1                              # 1 write per slot

The kernel then computes ``out[sorted_token_ids[i] // topk] = sorted_weights[i] * mlp_e(input[i])``,
which with the identity ids and topk=1 reduces to ``out[i] = permuted_probs[i] * mlp_e(input[i])``
-- exactly what Megatron's TEGroupedMLP returns BEFORE its quantization_unpadding step.

This test builds that exact setup and compares to the per-expert reference.

Run:
    docker exec miles-hai2 bash -lc 'cd /data/data/hai/miles-dsv4 && \
        python tests/rocm/test_fmoe_megatron_layout.py'
"""
import torch
import torch.nn.functional as F


def _e4m3():
    from aiter.ops.triton.utils.types import get_fp8_dtypes
    _, e4m3 = get_fp8_dtypes()
    return e4m3, float(torch.finfo(e4m3).max)


def _q_128x128_per_expert(W):
    e4m3, fmax = _e4m3()
    E, N, K = W.shape
    wv = W.float().view(E, N // 128, 128, K // 128, 128)
    scale = wv.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-12) / fmax
    wq = (wv / scale).clamp(-fmax, fmax).to(e4m3).view(E, N, K)
    ws = scale.view(E, N // 128, K // 128).float().contiguous()
    return wq, ws


def _q_1x128_act(x):
    e4m3, fmax = _e4m3()
    M, K = x.shape
    xv = x.float().view(M, K // 128, 128)
    scale = xv.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / fmax
    xq = (xv / scale).clamp(-fmax, fmax).to(e4m3).view(M, K)
    return xq, scale.squeeze(-1).contiguous()


def _per_expert_ref(permuted_input, tokens_per_expert, permuted_probs, w1, w2):
    """Reference: do the MLP per-expert via the SAME per-expert aiter blockscale path.

    permuted_input: [T_local, K]; already grouped by expert.
    tokens_per_expert: list[int], len == E_local.
    permuted_probs: [T_local], post-mul scaling per token slot.
    w1: [E_local, 2*I, K] gate-first; w2: [E_local, K, I].
    Returns [T_local, K] = probs * mlp_e(input) per slot.
    """
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale

    T_local, K = permuted_input.shape
    E, twoI, _ = w1.shape
    I = twoI // 2
    out = torch.zeros((T_local, K), device=permuted_input.device, dtype=torch.float32)
    offsets = [0]
    for c in tokens_per_expert:
        offsets.append(offsets[-1] + c)

    def _q_w128(w):
        e4m3, fmax = _e4m3()
        N, K_ = w.shape
        wv = w.float().view(N // 128, 128, K_ // 128, 128)
        scale = wv.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12) / fmax
        wq = (wv / scale).clamp(-fmax, fmax).to(e4m3).view(N, K_)
        return wq, scale.view(N // 128, K_ // 128).float().contiguous()

    for e in range(E):
        lo, hi = offsets[e], offsets[e + 1]
        if hi == lo:
            continue
        x_e = permuted_input[lo:hi].contiguous()
        probs_e = permuted_probs[lo:hi].contiguous().float()
        x_q, x_s = _q_1x128_act(x_e)
        w_gu_q, w_gu_s = _q_w128(w1[e])
        gu = gemm_a8w8_blockscale(x_q, w_gu_q, x_s, w_gu_s, dtype=torch.bfloat16)
        g, u = gu[:, :I], gu[:, I:]
        h = (F.silu(g.float()) * u.float()).to(torch.bfloat16)
        h_q, h_s = _q_1x128_act(h)
        w_d_q, w_d_s = _q_w128(w2[e])
        y_e = gemm_a8w8_blockscale(h_q, w_d_q, h_s, w_d_s, dtype=torch.bfloat16).float()
        out[lo:hi] = probs_e.view(-1, 1) * y_e
    return out.to(torch.bfloat16)


def _via_fmoe_identity_routing(permuted_input, tokens_per_expert, permuted_probs, w1, w2):
    """Drive fmoe_fp8_blockscale_g1u1 with the identity-routing trick.

    Because hand-constructing sorted_token_ids in the kernel's exact (padded-with-magic-values)
    format is error-prone, we instead build a SYNTHETIC topk_ids tensor with topk=1 -- where
    each "token" is one row of the already-permuted input -- and run aiter.moe_sorting on it.

    Since the input is ALREADY sorted by expert, moe_sorting produces an essentially identity
    sorted_token_ids: the slot for permuted row i references row i. fmoe then computes
    ``out[i] = topk_w[i] * mlp_{expert(i)}(input[i])`` -- which is EXACTLY Megatron's
    TEGroupedMLP return value (probs applied between FC1+SwiGLU and FC2 commutes with the
    final FC2 linear).
    """
    import aiter
    from aiter import ActivationType
    from aiter.fused_moe import moe_sorting
    from aiter.ops.shuffle import shuffle_weight

    device = permuted_input.device
    T_local, K = permuted_input.shape
    E = w1.shape[0]

    # Quantize input 1x128 along K
    a_q, a_s = _q_1x128_act(permuted_input)
    a_s_t = a_s.t().contiguous()  # [K/128, T_local] per asm_moe_test convention

    # Quantize and shuffle stacked weights
    w1q, w1s = _q_128x128_per_expert(w1)
    w2q, w2s = _q_128x128_per_expert(w2)
    w1q_s = shuffle_weight(w1q, (16, 16))
    w2q_s = shuffle_weight(w2q, (16, 16))

    # Build synthetic topk_ids=[T_local, 1] where row i's "expert" is the expert assigned to
    # the i-th permuted slot. This is the inverse of `tokens_per_expert.cumsum`:
    expert_for_slot = torch.zeros(T_local, dtype=torch.int32, device=device)
    cursor = 0
    for e, c in enumerate(tokens_per_expert):
        if c == 0:
            continue
        expert_for_slot[cursor:cursor + c] = e
        cursor += c
    topk_ids = expert_for_slot.view(T_local, 1)
    topk_w = permuted_probs.view(T_local, 1).float().contiguous()

    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_w, E, K, torch.bfloat16
    )

    aiter.fmoe_fp8_blockscale_g1u1(
        moe_buf,
        a_q,
        w1q_s,
        w2q_s,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        1,        # topk = 1
        a_s_t,
        w1s,
        w2s,
        "",
        128,
        128,
        None,
        ActivationType.Silu,
    )
    return moe_buf


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12)).item()


def main():
    torch.manual_seed(0)
    device = "cuda"

    # DSv4-flavored shapes (small for test speed):
    K = 512        # hidden / model_dim
    I = 256        # moe_ffn_hidden_size (must be /128)
    E = 4          # num local experts

    # Tokens per expert (each multiple of 128 for clean blockwise quant of inputs)
    tokens_per_expert = [128, 256, 128, 256]
    T_local = sum(tokens_per_expert)

    permuted_input = (torch.randn(T_local, K, device=device, dtype=torch.bfloat16) * 0.5)
    permuted_probs = torch.rand(T_local, device=device, dtype=torch.float32) * 0.5 + 0.25
    w1 = (torch.randn(E, 2 * I, K, device=device, dtype=torch.bfloat16) * 0.1)
    w2 = (torch.randn(E, K, I, device=device, dtype=torch.bfloat16) * 0.1)

    out_ref = _per_expert_ref(permuted_input, tokens_per_expert, permuted_probs, w1, w2)
    out_fmoe = _via_fmoe_identity_routing(permuted_input, tokens_per_expert, permuted_probs, w1, w2)

    err = _rel(out_fmoe, out_ref)
    print(f"fmoe (identity-routing) vs per-expert aiter loop: rel err = {err:.4f}")
    print(f"out_fmoe stats: mean={out_fmoe.float().mean().item():.4g} max={out_fmoe.float().abs().max().item():.4g}")
    print(f"out_ref  stats: mean={out_ref.float().mean().item():.4g} max={out_ref.float().abs().max().item():.4g}")
    assert torch.isfinite(out_fmoe).all(), "fmoe output not finite"
    assert err < 0.10, f"fmoe identity-routing diverges from per-expert ref: {err:.4f}"
    print("PASS: fmoe with identity routing matches the per-expert path on Megatron-style inputs")


if __name__ == "__main__":
    main()
