"""ROCm blockwise FP8 (DeepSeek-style Float8BlockScaling) enablement via aiter.

Why this exists
---------------
DeepSeek-V4 (and V3) train in FP8 with the *block-scaled* recipe: activations are
quantized in 1x128 groups and weights in 128x128 blocks, E4M3 with FP32 scales
(NVIDIA TE: ``Float8BlockScaling``, Megatron flag ``--fp8-recipe blockwise``).

On NVIDIA this "just works" -- NVIDIA/TransformerEngine shipped the blockwise FP8
GEMM/quant kernels back in the DeepSeek-V3 era, so yueming's DSv4 work only had to
touch miles + Megatron, never TE.

On AMD it does NOT work: ``ROCm/TransformerEngine`` hard-gates the recipe off
(``check_fp8_block_scaling_support`` returns False on HIP) and its HIP cast/GEMM
kernels don't implement the DeepSeek block-scaling mode. AMD's TE block-FP8 effort
went into MXFP8 / FP4, not DeepSeek's 1x128/128x128 scheme.

The actual *kernels* DO exist on AMD though -- in ``aiter`` (``gemm_a8w8_blockscale``),
which is exactly what sglang uses to serve DSv4 in FP8 on MI3xx. This module reuses
those aiter blockscale kernels to build a training-capable blockwise FP8 linear, so the
SAME kernel is used for train and inference (train/infer numerical consistency for free).

Status (validated on MI355X / gfx950) -- all three GEMMs are FP8:
-----------------------------------------------------------------
- fprop : aiter gemm_a8w8_blockscale  (X 1x128  x  W 128x128)        ~3.7% vs bf16
- dgrad : aiter gemm_a8w8_blockscale  (dY 1x128 x  W^T 128x128)      ~4.1% vs bf16
- wgrad : symmetric_blockscale_gemm   (dY^T 1x128 x X^T 1x128)       ~3.6% vs bf16

wgrad is now FAITHFUL to DeepSeek's recipe: ``dW = dY^T @ X`` contracts over the
token dim M, so BOTH operands are activation-like and both are quantized 1x128 along
M (not X-as-128x128 as in the v1 MVP). This is exposed as ``symmetric_blockscale_gemm``,
a both-operands-1D-scaled GEMM. It reuses the SAME aiter ``gemm_a8w8_blockscale``
kernel: passing a per-row B scale of shape ``[Q, C/128]`` makes the kernel's
``GROUP_N`` collapse to 1, so each B row carries its own 1x128 scale along the
contraction axis -- exactly the symmetric (both 1x128) case. No kernel fork needed.

Known simplifications (MVP): tensor dims must be multiples of 128 (no padding yet).

Next: wire this into TransformerEngine (lift the gate + route the blockwise
quantize/GEMM through here) so Megatron/miles pick it up unchanged.
"""
from __future__ import annotations

import torch

BLK = 128


def _is_rocm() -> bool:
    return getattr(torch.version, "hip", None) is not None


# aiter pieces are imported lazily so importing this module is cheap / safe off-ROCm.
def _aiter_bits():
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale
    from aiter.ops.triton.utils.types import get_fp8_dtypes

    _, e4m3 = get_fp8_dtypes()
    return gemm_a8w8_blockscale, e4m3, float(torch.finfo(e4m3).max)


def quantize_1x128(x: torch.Tensor):
    """Activation/gradient quant: [M, K] -> (e4m3 [M, K], fp32 scale [M, K/128]).

    Dequant convention matches aiter: ``x ~= xq.float() * scale`` (per 1x128 group).
    """
    _, e4m3, fmax = _aiter_bits()
    M, K = x.shape
    xv = x.float().view(M, K // BLK, BLK)
    scale = xv.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / fmax
    xq = (xv / scale).clamp(-fmax, fmax).to(e4m3).view(M, K)
    return xq, scale.squeeze(-1).contiguous()


def quantize_128x128(w: torch.Tensor):
    """Weight quant: [N, K] -> (e4m3 [N, K], fp32 scale [N/128, K/128])."""
    _, e4m3, fmax = _aiter_bits()
    N, K = w.shape
    wv = w.float().view(N // BLK, BLK, K // BLK, BLK)
    scale = wv.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12) / fmax
    wq = (wv / scale).clamp(-fmax, fmax).to(e4m3).view(N, K)
    return wq, scale.view(N // BLK, K // BLK).contiguous()


def symmetric_blockscale_gemm(
    a_q: torch.Tensor,
    b_q: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """SYMMETRIC block-scaled FP8 GEMM: ``out[P,Q] = A @ B^T``, contraction over C.

    BOTH operands carry 1D per-token-group (1x128) scales along the contraction dim C
    -- the faithful DeepSeek recipe for GEMMs where both inputs are activation-like
    (e.g. wgrad ``dW = dY^T @ X``, contracting the token dim).

    Args:
        a_q:     e4m3 [P, C]              (quantize_1x128 output)
        b_q:     e4m3 [Q, C]              (quantize_1x128 output)
        a_scale: fp32  [P, C // 128]      (1x128 scale along C, "A role")
        b_scale: fp32  [Q, C // 128]      (1x128 scale along C, "B role" -- per ROW)
    Returns:
        out:     dtype [P, Q] = A_deq @ B_deq^T

    Implementation note: this reuses aiter ``gemm_a8w8_blockscale`` unchanged. That
    kernel maps (x, w, x_scale, w_scale) -> x @ w^T with w_scale of shape
    [scale_n, scale_k], scale_n = ceil(N / GROUP_N). Passing a per-row b_scale of shape
    [Q, C/128] (scale_n == Q) drives the wrapper's GROUP_N to next_pow2(ceil(Q/Q)) = 1,
    so every B row gets its own 1x128 scale along C -- i.e. the symmetric case.
    """
    gemm, _, _ = _aiter_bits()
    return gemm(a_q, b_q, a_scale, b_scale, dtype=dtype)


class BlockwiseFP8Linear(torch.autograd.Function):
    """y = x @ w^T with DeepSeek blockwise FP8; all 3 GEMMs FP8 on aiter.

    fprop/dgrad use the asymmetric kernel (act 1x128 x weight 128x128); wgrad uses the
    symmetric kernel (both operands 1x128 along the token dim), faithful to DeepSeek.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        gemm, _, _ = _aiter_bits()
        xq, xs = quantize_1x128(x)
        wq, ws = quantize_128x128(w)
        y = gemm(xq, wq, xs, ws, dtype=torch.bfloat16)
        ctx.save_for_backward(x, w)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        gemm, _, _ = _aiter_bits()
        x, w = ctx.saved_tensors
        dy = dy.contiguous()
        # dgrad: dX = dY @ W = aiter(dY[1x128], W^T[128x128])
        dyq, dys = quantize_1x128(dy)
        wtq, wts = quantize_128x128(w.t().contiguous())
        dx = gemm(dyq, wtq, dys, wts, dtype=torch.bfloat16)
        # wgrad: dW = dY^T @ X, contraction over the token dim M. Both operands are
        # activation-like, so both are quantized 1x128 along M (faithful DeepSeek recipe)
        # and combined via the symmetric block-scaled GEMM.
        #   a = dY^T [N, M] 1x128 along M ; b = X^T [K, M] 1x128 along M ; out = a @ b^T = [N, K]
        dytq, dyts = quantize_1x128(dy.t().contiguous())
        xtq, xts = quantize_1x128(x.t().contiguous())
        dw = symmetric_blockscale_gemm(dytq, xtq, dyts, xts, dtype=torch.bfloat16).to(w.dtype)
        return dx, dw


def blockwise_fp8_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Functional blockwise FP8 linear (no bias). x:[M,K] w:[N,K] -> [M,N]."""
    return BlockwiseFP8Linear.apply(x, w)


def lift_te_gate() -> bool:
    """Monkeypatch ROCm/TransformerEngine so Float8BlockScaling passes recipe support.

    NOTE: lifting the gate alone is not sufficient -- TE's HIP quantize/GEMM kernels
    still don't implement the recipe; the quantize + GEMM routing (to this module's
    aiter path) must also be patched. That wiring is added separately.
    """
    if not _is_rocm():
        return False
    import transformer_engine.pytorch.fp8 as _tefp8

    _tefp8.check_fp8_block_scaling_support = lambda: (True, "")
    return True
