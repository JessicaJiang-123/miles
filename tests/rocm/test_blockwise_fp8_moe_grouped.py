"""Validate MoE grouped GEMM blockwise FP8 via per-expert aiter loop.

Goal: prove that the same `gemm_a8w8_blockscale` we use for the dense Linear path
can be loop-applied per expert in TEGroupedLinear's three GEMM calls
(fprop "TN", dgrad "NN", wgrad "NT") and match a bf16 reference within ~few %.

Mirrors the operand layout TE's `general_grouped_gemm` actually hands us:
    fprop:  A=W_e[N,K] sd=2,  B=X_e[M,K] sd=1,  layout="TN", out=Y[M,N]   (Y = X @ W^T)
    dgrad:  A=W_e[N,K] sd=2,  B=dY_e[M,N] sd=1, layout="NN", out=dX[M,K]  (dX = dY @ W)
    wgrad:  A=dY_e[M,N] sd=1, B=X_e[M,K] sd=1,  layout="NT", out=dW[N,K]  (dW = dY^T @ X)

Run with aiter available:
    docker exec miles-hai2 bash -lc 'cd /data/data/hai/miles-dsv4 && python tests/rocm/test_blockwise_fp8_moe_grouped.py'
"""
import torch

BLK = 128


def _aiter_bits():
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale
    from aiter.ops.triton.utils.types import get_fp8_dtypes

    _, e4m3 = get_fp8_dtypes()
    return gemm_a8w8_blockscale, e4m3, float(torch.finfo(e4m3).max)


def quantize_1x128(x):
    _, e4m3, fmax = _aiter_bits()
    M, K = x.shape
    xv = x.float().view(M, K // BLK, BLK)
    scale = xv.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / fmax
    xq = (xv / scale).clamp(-fmax, fmax).to(e4m3).view(M, K)
    return xq, scale.squeeze(-1).contiguous()


def quantize_128x128(w):
    _, e4m3, fmax = _aiter_bits()
    N, K = w.shape
    wv = w.float().view(N // BLK, BLK, K // BLK, BLK)
    scale = wv.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12) / fmax
    wq = (wv / scale).clamp(-fmax, fmax).to(e4m3).view(N, K)
    return wq, scale.view(N // BLK, K // BLK).contiguous()


def dense_blockwise_matmul(x_bf16, w_bf16):
    """X[M,K] @ W^T[N,K] via aiter blockscale FP8 (1x128 act, 128x128 weight)."""
    aiter_gemm, _, _ = _aiter_bits()
    xq, xs = quantize_1x128(x_bf16)
    wq, ws = quantize_128x128(w_bf16)
    return aiter_gemm(xq, wq, xs, ws, dtype=torch.bfloat16)


def dense_blockwise_wgrad(dy_bf16, x_bf16):
    """dW[N,K] = dY^T[N,M] @ X[M,K] via aiter; both operands 1x128 along K.

    For wgrad we need both sides quantized along the contraction dim M.
    aiter_gemm wants x:[M,K] 1x128(K), w:[N,K] 128x128. Trick (same as dense path):
    treat dY as "activation" along M-as-K i.e. quantize dY along M.
    """
    aiter_gemm, _, _ = _aiter_bits()
    M, N = dy_bf16.shape
    M_, K = x_bf16.shape
    assert M == M_
    # We want dW = dY^T @ X. Use aiter_gemm(a, b, ...) computes a @ b^T.
    # Set a = dY^T[N, M] (so M-dim is contraction), b = X^T[K, M].
    # Both quantized 1x128 along the M-axis (the K of aiter).
    a = dy_bf16.t().contiguous()  # [N, M]
    b = x_bf16.t().contiguous()  # [K, M]
    aq, as_ = quantize_1x128(a)
    bq, bs = quantize_1x128(b)
    return aiter_gemm(aq, bq, as_, bs, dtype=torch.bfloat16)  # -> [N, K]


def grouped_fprop(x_list, w_list):
    """List of per-expert fprop: y_e = x_e @ w_e^T."""
    return [dense_blockwise_matmul(x, w) for x, w in zip(x_list, w_list)]


def grouped_dgrad(dy_list, w_list):
    """List of per-expert dgrad: dx_e = dy_e @ w_e."""
    # dX = dY @ W with X[M,K] = dY[M,N] and W[N,K] => aiter does a @ b^T with b=[K,N].
    # We want a @ b with a=dY[M,N], b=W[N,K]. Rephrase: c = dY @ W = (W^T @ dY^T)^T
    # Use aiter as: aiter_gemm(dY, W^T) but W^T is [K,N] — needs N divisible by BLK twice.
    # Easier: aiter_gemm(dY[M,N], W^T[K,N], ...) computes dY @ (W^T)^T = dY @ W. Right!
    # So: take w transposed shape [K, N], quantize 128x128, the contraction is N which
    # must be multiple of BLK.
    aiter_gemm, _, _ = _aiter_bits()
    out = []
    for dy, w in zip(dy_list, w_list):
        # dy [M, N], w [N, K]. Want [M, K].
        wT = w.t().contiguous()  # [K, N]
        dyq, dys = quantize_1x128(dy)
        wTq, wTs = quantize_128x128(wT)
        out.append(aiter_gemm(dyq, wTq, dys, wTs, dtype=torch.bfloat16))
    return out


def grouped_wgrad(dy_list, x_list):
    """List of per-expert wgrad: dw_e = dy_e^T @ x_e."""
    return [dense_blockwise_wgrad(dy, x) for dy, x in zip(dy_list, x_list)]


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12)).item()


def _ref_fprop(x_list, w_list):
    return [x @ w.t() for x, w in zip(x_list, w_list)]


def _ref_dgrad(dy_list, w_list):
    return [dy @ w for dy, w in zip(dy_list, w_list)]


def _ref_wgrad(dy_list, x_list):
    return [dy.t() @ x for dy, x in zip(dy_list, x_list)]


def _run():
    torch.manual_seed(0)
    device = "cuda"
    E = 4  # local experts
    H = 512  # hidden
    FFN = 768  # 1.5x hidden, divisible by 128
    # tokens per expert
    tpe = [128, 256, 384, 128]
    # all multiples of 128 to keep both 1x128 and 128x128 happy
    K = H
    N = FFN
    x_list = [(torch.randn(t, K, device=device, dtype=torch.bfloat16) * 0.1) for t in tpe]
    w_list = [(torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.1) for _ in range(E)]
    dy_list = [(torch.randn(t, N, device=device, dtype=torch.bfloat16) * 0.1) for t in tpe]

    # FPROP
    y_fp8 = grouped_fprop(x_list, w_list)
    y_ref = _ref_fprop(x_list, w_list)
    fwd_errs = [_rel(a, b) for a, b in zip(y_fp8, y_ref)]
    print(f"fprop rel errs: {[f'{e:.4f}' for e in fwd_errs]}")

    # DGRAD
    dx_fp8 = grouped_dgrad(dy_list, w_list)
    dx_ref = _ref_dgrad(dy_list, w_list)
    dx_errs = [_rel(a, b) for a, b in zip(dx_fp8, dx_ref)]
    print(f"dgrad rel errs: {[f'{e:.4f}' for e in dx_errs]}")

    # WGRAD
    dw_fp8 = grouped_wgrad(dy_list, x_list)
    dw_ref = _ref_wgrad(dy_list, x_list)
    dw_errs = [_rel(a, b) for a, b in zip(dw_fp8, dw_ref)]
    print(f"wgrad rel errs: {[f'{e:.4f}' for e in dw_errs]}")

    assert all(e < 0.06 for e in fwd_errs), f"fprop too noisy: {fwd_errs}"
    assert all(e < 0.10 for e in dx_errs), f"dgrad too noisy: {dx_errs}"
    assert all(e < 0.10 for e in dw_errs), f"wgrad too noisy: {dw_errs}"
    print("PASS: grouped FP8 fprop/dgrad/wgrad match bf16 reference within tolerance")


if __name__ == "__main__":
    _run()
