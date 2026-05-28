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

Status (v1, validated on MI355X / gfx950)
-----------------------------------------
- fprop : aiter gemm_a8w8_blockscale  (X 1x128  x  W 128x128)        ~3.7% vs bf16
- dgrad : aiter gemm_a8w8_blockscale  (dY 1x128 x  W^T 128x128)      ~4.1% vs bf16
- wgrad : bf16 fallback (both operands are 1x128; FP8 path is TODO -- needs an
          aiter "both per-token-group" blockscale GEMM)

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


class BlockwiseFP8Linear(torch.autograd.Function):
    """y = x @ w^T with DeepSeek blockwise FP8, fprop+dgrad on aiter, wgrad bf16 (v1)."""

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
        # wgrad: dW = dY^T @ X -- v1 bf16 fallback (both operands 1x128)
        dw = (dy.float().t() @ x.float()).to(w.dtype)
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
