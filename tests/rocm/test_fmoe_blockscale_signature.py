"""Probe aiter.fmoe_fp8_blockscale_g1u1 signature with the canonical sglang-style call.

Goal: confirm that calling fmoe via aiter.fused_moe.fused_moe with QuantType.per_128x128
and our random per-expert weights produces a result that matches a torch reference within
FP8 tolerance. This validates the *exact* signature sglang uses to serve DSv4.

Once this passes we know:
  - the kernel works on this MI355X build,
  - shuffle_weight((16,16)) is the correct weight layout,
  - the gate-and-up are STACKED into w13 of shape [E, 2*inter, K] (interleaved? or
    concatenated? we test both).

Run:
    docker exec miles-hai2 bash -lc 'cd /data/data/hai/miles-dsv4 && \
        python tests/rocm/test_fmoe_blockscale_signature.py'
"""
import torch
import torch.nn.functional as F


def _torch_ref(x, w1, w2, topk_w, topk_ids, *, gate_first=True):
    """Reference: per-token, per-topk MLP with SwiGLU.

    x:      [T, K]
    w1:     [E, 2*I, K] -- if gate_first=True, [:I,:]=gate, [I:,:]=up
                          else interleaved (gate, up, gate, up, ...) along the 2*I axis
    w2:     [E, K, I]
    topk_w: [T, topk]
    topk_ids: [T, topk]
    """
    T, K = x.shape
    E, twoI, _ = w1.shape
    I = twoI // 2
    out = torch.zeros((T, K), device=x.device, dtype=torch.float32)
    for e in range(E):
        mask = (topk_ids == e)  # [T, topk]
        if not mask.any():
            continue
        we = w1[e].float()  # [2I, K]
        if gate_first:
            wg = we[:I]
            wu = we[I:]
        else:
            wg = we[0::2]
            wu = we[1::2]
        wd = w2[e].float()  # [K, I]
        for t in range(T):
            for k in range(topk_w.shape[1]):
                if topk_ids[t, k].item() == e:
                    xt = x[t].float()
                    g = xt @ wg.t()
                    u = xt @ wu.t()
                    h = F.silu(g) * u
                    o = h @ wd.t()
                    out[t] += topk_w[t, k].item() * o
    return out.to(x.dtype)


def _quantize_128x128_per_expert(w):
    """w: [E, N, K] bf16 -> wq [E, N, K] e4m3-as-uint8, ws [E, N/128, K/128] fp32 scale."""
    from aiter.ops.triton.utils.types import get_fp8_dtypes
    _, e4m3 = get_fp8_dtypes()
    fmax = float(torch.finfo(e4m3).max)
    E, N, K = w.shape
    wv = w.float().view(E, N // 128, 128, K // 128, 128)
    scale = wv.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-12) / fmax
    wq = (wv / scale).clamp(-fmax, fmax).to(e4m3).view(E, N, K)
    ws = scale.view(E, N // 128, K // 128).float().contiguous()
    return wq, ws


def _quantize_1x128_act(x):
    """x: [T, K] bf16 -> xq [T, K] e4m3, xs [T, K/128] (column-major if needed)."""
    from aiter.ops.triton.utils.types import get_fp8_dtypes
    _, e4m3 = get_fp8_dtypes()
    fmax = float(torch.finfo(e4m3).max)
    T, K = x.shape
    xv = x.float().view(T, K // 128, 128)
    scale = xv.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / fmax
    xq = (xv / scale).clamp(-fmax, fmax).to(e4m3).view(T, K)
    xs = scale.squeeze(-1).contiguous()  # [T, K/128]
    return xq, xs


def _via_aiter_fused_moe(x_bf16, w1, w2, topk_w, topk_ids):
    """Use the high-level aiter.fused_moe (sglang-style) for FP8 blockwise.

    Returns the bf16 MoE output [T, K].
    """
    import aiter
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.shuffle import shuffle_weight

    # Quantize weights 128x128.
    w1q, w1s = _quantize_128x128_per_expert(w1)  # w1q: e4m3
    w2q, w2s = _quantize_128x128_per_expert(w2)

    # Shuffle for the asm kernel.
    w1q_s = shuffle_weight(w1q, (16, 16))
    w2q_s = shuffle_weight(w2q, (16, 16))

    out = fused_moe(
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
    return out


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12)).item()


def main():
    torch.manual_seed(0)
    device = "cuda"

    # Small-but-realistic DSv4-like shapes (scaled down for test speed).
    T = 128       # tokens
    K = 512       # hidden / model_dim (must be /128)
    I = 256       # inter_dim (must be /128)
    E = 8         # num experts
    topk = 2      # routed-experts per token

    x = (torch.randn(T, K, device=device, dtype=torch.bfloat16) * 0.5)
    # w1 stacked [E, 2*I, K] gate-first
    w1 = (torch.randn(E, 2 * I, K, device=device, dtype=torch.bfloat16) * 0.1)
    w2 = (torch.randn(E, K, I, device=device, dtype=torch.bfloat16) * 0.1)

    # Random routing: each token gets `topk` distinct experts; weights sum-to-one.
    scores = torch.randn(T, E, device=device, dtype=torch.float32)
    topk_w, topk_ids = torch.topk(scores, topk, dim=-1)
    topk_w = torch.softmax(topk_w, dim=-1).to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    # ----- reference (bf16) -----
    # try both gate-first and interleaved; whichever matches fmoe is the truth
    out_ref_gf = _torch_ref(x, w1, w2, topk_w, topk_ids, gate_first=True)
    out_ref_il = _torch_ref(x, w1, w2, topk_w, topk_ids, gate_first=False)

    # ----- aiter fmoe path -----
    out_fmoe = _via_aiter_fused_moe(x, w1, w2, topk_w, topk_ids)

    err_gf = _rel(out_fmoe, out_ref_gf)
    err_il = _rel(out_fmoe, out_ref_il)
    print(f"fmoe vs ref (gate-first concatenated [:I gate, I: up]) rel err: {err_gf:.4f}")
    print(f"fmoe vs ref (interleaved gate/up along 2*I axis) rel err:      {err_il:.4f}")

    # whichever is small (<10%) is the right interpretation
    best = min(err_gf, err_il)
    layout = "gate_first" if err_gf < err_il else "interleaved"
    print(f"=> matching layout: {layout}, err={best:.4f}")
    assert best < 0.10, f"fmoe diverges from ref in BOTH layouts: gf={err_gf:.4f} il={err_il:.4f}"
    print("PASS")


if __name__ == "__main__":
    main()
