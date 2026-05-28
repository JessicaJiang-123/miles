"""Compare aiter fmoe_fp8_blockscale_g1u1 (one launch) vs the per-expert aiter dense
gemm_a8w8_blockscale loop currently used by miles' TEGroupedLinear patch.

Both consume the SAME inputs (bf16 tokens [T, K], routing, per-expert MLP weights):
  - fmoe: stack gate+up into w13 [E, 2*I, K], call fused_moe(...)
  - per-expert: route tokens to experts by topk_ids, for each expert run
      x_e @ wg^T -> g; x_e @ wu^T -> u; silu(g)*u @ wd^T -> y_e; scale by topk_w; combine.
    Each per-expert GEMM is via aiter.gemm_a8w8_blockscale exactly as the current TE
    patch does it.

If both FP8 paths agree to ~1% (both quantize the same bf16 weights identically), then
fmoe is a drop-in numerical match for the existing patched per-expert loop, and the
ONLY differences will be (a) one launch instead of 2E launches, (b) internal routing
in the kernel vs Python-side routing.

Run:
    docker exec miles-hai2 bash -lc 'cd /data/data/hai/miles-dsv4 && \
        python tests/rocm/test_fmoe_vs_per_expert.py'
"""
import torch
import torch.nn.functional as F


def _e4m3():
    from aiter.ops.triton.utils.types import get_fp8_dtypes
    _, e4m3 = get_fp8_dtypes()
    return e4m3, float(torch.finfo(e4m3).max)


def _q_1x128_act(x):
    e4m3, fmax = _e4m3()
    M, K = x.shape
    xv = x.float().view(M, K // 128, 128)
    scale = xv.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / fmax
    xq = (xv / scale).clamp(-fmax, fmax).to(e4m3).view(M, K)
    return xq, scale.squeeze(-1).contiguous()  # xs: [M, K/128]


def _q_128x128_w(w):
    """w: [N, K] -> wq [N, K] e4m3, ws [N/128, K/128] fp32."""
    e4m3, fmax = _e4m3()
    N, K = w.shape
    wv = w.float().view(N // 128, 128, K // 128, 128)
    scale = wv.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12) / fmax
    wq = (wv / scale).clamp(-fmax, fmax).to(e4m3).view(N, K)
    return wq, scale.view(N // 128, K // 128).float().contiguous()


def _q_128x128_per_expert(W):
    """W: [E, N, K] -> wq [E, N, K] e4m3, ws [E, N/128, K/128] fp32."""
    e4m3, fmax = _e4m3()
    E, N, K = W.shape
    wv = W.float().view(E, N // 128, 128, K // 128, 128)
    scale = wv.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-12) / fmax
    wq = (wv / scale).clamp(-fmax, fmax).to(e4m3).view(E, N, K)
    ws = scale.view(E, N // 128, K // 128).float().contiguous()
    return wq, ws


def _per_expert_loop(x_bf16, w1, w2, topk_w, topk_ids):
    """Reference: same per-expert aiter gemm_a8w8_blockscale that the TE patch uses.

    x_bf16: [T, K]; w1: [E, 2I, K] gate-first; w2: [E, K, I];
    topk_w: [T, topk] fp32; topk_ids: [T, topk] int.
    Returns [T, K] bf16.
    """
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale

    T, K = x_bf16.shape
    E, twoI, _ = w1.shape
    I = twoI // 2
    out = torch.zeros((T, K), device=x_bf16.device, dtype=torch.float32)

    for e in range(E):
        # gather tokens routed to expert e (across all topk slots)
        mask = (topk_ids == e)
        if not mask.any():
            continue
        tok_idx, slot = mask.nonzero(as_tuple=True)
        x_e = x_bf16[tok_idx].contiguous()  # [m_e, K]
        w_e_gu = w1[e]      # [2I, K]
        w_e_d  = w2[e]      # [K, I]
        # FC1 gate+up: y_e = x_e @ w_e_gu^T -> [m_e, 2I]; split into [g, u].
        # The current TE patch quantizes x_e 1x128, w 128x128 -> aiter_gemm(x, w, xs, ws).
        x_q, x_s = _q_1x128_act(x_e)
        w_gu_q, w_gu_s = _q_128x128_w(w_e_gu)
        gu = gemm_a8w8_blockscale(x_q, w_gu_q, x_s, w_gu_s, dtype=torch.bfloat16)
        g, u = gu[:, :I], gu[:, I:]
        h = F.silu(g.float()) * u.float()  # bf16-ish
        h_bf = h.to(torch.bfloat16)
        # FC2: y_e = h @ w_e_d^T -> [m_e, K]
        h_q, h_s = _q_1x128_act(h_bf)
        w_d_q, w_d_s = _q_128x128_w(w_e_d)
        y_e = gemm_a8w8_blockscale(h_q, w_d_q, h_s, w_d_s, dtype=torch.bfloat16).float()
        # combine: out[tok_idx] += topk_w[tok_idx, slot] * y_e
        weights = topk_w[tok_idx, slot].view(-1, 1).float()
        out.index_add_(0, tok_idx, weights * y_e)
    return out.to(torch.bfloat16)


def _via_aiter_fmoe(x_bf16, w1, w2, topk_w, topk_ids):
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.shuffle import shuffle_weight

    w1q, w1s = _q_128x128_per_expert(w1)
    w2q, w2s = _q_128x128_per_expert(w2)

    w1q_s = shuffle_weight(w1q, (16, 16))
    w2q_s = shuffle_weight(w2q, (16, 16))

    return fused_moe(
        x_bf16,
        w1q_s,
        w2q_s,
        topk_w,
        topk_ids,
        quant_type=QuantType.per_128x128,
        w1_scale=w1s,
        w2_scale=w2s,
        activation=ActivationType.Silu,
    )


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12)).item()


def main():
    torch.manual_seed(0)
    device = "cuda"
    T = 256
    K = 512
    I = 256
    E = 8
    topk = 2

    x = (torch.randn(T, K, device=device, dtype=torch.bfloat16) * 0.5)
    w1 = (torch.randn(E, 2 * I, K, device=device, dtype=torch.bfloat16) * 0.1)
    w2 = (torch.randn(E, K, I, device=device, dtype=torch.bfloat16) * 0.1)

    scores = torch.randn(T, E, device=device, dtype=torch.float32)
    topk_w, topk_ids = torch.topk(scores, topk, dim=-1)
    topk_w = torch.softmax(topk_w, dim=-1).to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    out_loop = _per_expert_loop(x, w1, w2, topk_w, topk_ids)
    out_fmoe = _via_aiter_fmoe(x, w1, w2, topk_w, topk_ids)
    err = _rel(out_fmoe, out_loop)
    print(f"fmoe vs per-expert aiter loop: rel err = {err:.4f}")
    # both are FP8 paths quantizing the same bf16 weights -> should agree closely
    assert err < 0.05, f"divergent FP8 paths: {err:.4f}"
    print("PASS: fmoe matches per-expert aiter blockwise FP8 loop")


if __name__ == "__main__":
    main()
