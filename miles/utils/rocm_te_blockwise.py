"""Wire DeepSeek-style blockwise FP8 (Float8BlockScaling) into ROCm/TransformerEngine
via aiter, so Megatron/miles `--fp8-recipe blockwise` actually runs on MI3xx (gfx950).

Background
----------
ROCm/TransformerEngine ships the blockwise recipe API but:
  1. hard-gates it off on HIP (`check_fp8_block_scaling_support` -> False), and
  2. its HIP cast (`tex.quantize`) and GEMM (`tex.generic_gemm`) kernels do NOT
     implement DeepSeek's 1x128/128x128 block scaling ("Not implemented scaling mode").

The actual kernels exist in `aiter` (used by sglang to serve DSv4 in FP8). This module
monkeypatches THREE points so TE routes the blockwise path through aiter:

  1. GATE     : `fp8.check_fp8_block_scaling_support = lambda: (True, "")`  (lift_te_gate)
  2. QUANTIZE : `Float8BlockQuantizer.quantize` / `.update_quantized`  -> aiter quant
  3. GEMM     : `cpp_extensions.gemm.general_gemm` (and the `te.Linear` module-local
                binding) -> aiter `gemm_a8w8_blockscale`

Design note on scale layout
----------------------------
Stock TE stores GEMM_READY scales in a transposed / 4-aligned layout matched to cuBLAS.
Because we own BOTH the quantize and the GEMM ends here, we instead store scales in the
natural aiter convention (rowwise/1x128 -> [M, K/128]; 2D/128x128 -> [N/128, K/128]) and
read them back the same way. We still tag the tensor `_data_format = GEMM_READY` so TE's
internal format assertion passes. We never call TE's C++ blockwise quantize/GEMM.

`enable()` is idempotent and must run in EACH process (incl. every Megatron worker)
BEFORE TE builds its modules.

Validated standalone on MI355X (gfx950): a single `te.Linear` fwd+bwd under
`te.fp8_autocast(fp8_recipe=Float8BlockScaling(...))` matches a bf16 reference to a
few percent on out / dgrad / wgrad.
"""
from __future__ import annotations

import torch

from miles.utils.rocm_fp8_blockwise import (
    BLK,
    quantize_1x128,
    quantize_128x128,
    _aiter_bits,
    _is_rocm,
    lift_te_gate,
)

_ENABLED = False


def _to_uint8(e4m3: torch.Tensor) -> torch.Tensor:
    """TE convention: store fp8 data bytes as uint8."""
    return e4m3.view(torch.uint8)


def _from_uint8(u8: torch.Tensor, e4m3_dtype: torch.dtype) -> torch.Tensor:
    return u8.view(e4m3_dtype)


# ---------------------------------------------------------------------------
# QUANTIZE override
# ---------------------------------------------------------------------------
def _patch_quantizer():
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
        Float8BlockQuantizer,
        Float8BlockwiseQTensor,
    )
    import transformer_engine_torch as tex
    from transformer_engine_torch import DType as TE_DType

    _, e4m3_torch, _ = _aiter_bits()

    # TE's DType enum value for e4m3 (used as fp8_dtype tag on the tensor).
    fp8_e4m3 = TE_DType.kFloat8E4M3

    def _build_qtensor(self: Float8BlockQuantizer, tensor: torch.Tensor):
        """Quantize `tensor` (>=2D, last dim = K) via aiter and wrap as Float8BlockwiseQTensor.

        Stores data as uint8; scales in natural aiter convention. Produces columnwise
        (transposed) data+scale when `self.columnwise_usage` is set.
        """
        orig_shape = tuple(tensor.shape)
        K = orig_shape[-1]
        M = 1
        for d in orig_shape[:-1]:
            M *= d
        x2d = tensor.reshape(M, K).contiguous()

        rowwise_data = rowwise_scale = None
        columnwise_data = columnwise_scale = None

        if self.block_scaling_dim == 1:
            # activation / grad: 1x128 along K
            if self.rowwise_usage:
                q, s = quantize_1x128(x2d)  # q:[M,K] e4m3, s:[M,K/128]
                rowwise_data = _to_uint8(q).reshape(orig_shape).contiguous()
                rowwise_scale = s.contiguous()
            if self.columnwise_usage:
                # transpose: quantize x^T (1x128 along M) -> data [K,M], scale [K, M/128]
                xt = x2d.t().contiguous()
                qc, sc = quantize_1x128(xt)
                columnwise_data = _to_uint8(qc).contiguous()
                columnwise_scale = sc.contiguous()
        else:
            # weight: 128x128
            if self.rowwise_usage:
                q, s = quantize_128x128(x2d)  # q:[M,K], s:[M/128, K/128]
                rowwise_data = _to_uint8(q).reshape(orig_shape).contiguous()
                rowwise_scale = s.contiguous()
            if self.columnwise_usage:
                xt = x2d.t().contiguous()
                qc, sc = quantize_128x128(xt)  # q:[K,M], s:[K/128, M/128]
                columnwise_data = _to_uint8(qc).contiguous()
                columnwise_scale = sc.contiguous()

        data_format = tex.Float8BlockScaleTensorFormat.GEMM_READY
        out = Float8BlockwiseQTensor(
            shape=orig_shape,
            dtype=tensor.dtype if tensor.dtype.is_floating_point else torch.bfloat16,
            fp8_dtype=fp8_e4m3,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale,
            quantizer=self,
            is_2D_scaled=self.block_scaling_dim == 2,
            data_format=data_format,
            requires_grad=False,
        )
        # tag so the GEMM knows the block scaling dim without re-deriving it
        out._aiter_block_scaling_dim = self.block_scaling_dim
        return out

    def quantize(self, tensor, *, out=None, dtype=None):
        if out is not None:
            return self.update_quantized(tensor, out)
        return _build_qtensor(self, tensor)

    def update_quantized(self, src, dst, *, noop_flag=None):
        new = _build_qtensor(self, src)
        # copy aiter results into the pre-allocated dst's slots
        dst._rowwise_data = new._rowwise_data
        dst._rowwise_scale_inv = new._rowwise_scale_inv
        dst._columnwise_data = new._columnwise_data
        dst._columnwise_scale_inv = new._columnwise_scale_inv
        dst._fp8_dtype = new._fp8_dtype
        dst._data_format = new._data_format
        dst._is_2D_scaled = new._is_2D_scaled
        dst._aiter_block_scaling_dim = self.block_scaling_dim
        return dst

    Float8BlockQuantizer.quantize = quantize
    Float8BlockQuantizer.update_quantized = update_quantized


# ---------------------------------------------------------------------------
# GEMM override
# ---------------------------------------------------------------------------
def _extract(t, e4m3_torch, *, columnwise: bool):
    """Return (data_e4m3, scale_fp32, scaling_dim) from a Float8BlockwiseQTensorBase."""
    if columnwise:
        u8 = t._columnwise_data
        s = t._columnwise_scale_inv
    else:
        u8 = t._rowwise_data
        s = t._rowwise_scale_inv
    data = _from_uint8(u8, e4m3_torch)
    sdim = getattr(t, "_aiter_block_scaling_dim", 2 if t._is_2D_scaled else 1)
    return data.reshape(-1, data.shape[-1]).contiguous(), s.contiguous(), sdim


def _patch_gemm():
    import transformer_engine.pytorch.cpp_extensions.gemm as gemm_mod
    from transformer_engine.pytorch.tensor._internal.float8_blockwise_tensor_base import (
        Float8BlockwiseQTensorBase,
    )
    from transformer_engine.pytorch.constants import TE_DType

    aiter_gemm, e4m3_torch, _ = _aiter_bits()
    _orig_general_gemm = gemm_mod.general_gemm

    def general_gemm(
        A,
        B,
        workspace,
        out_dtype=None,
        quantization_params=None,
        gelu=False,
        gelu_in=None,
        alpha=1.0,
        beta=None,
        accumulate=False,
        layout="TN",
        out=None,
        bias=None,
        use_split_accumulator=False,
        grad=False,
        ub=None,
        ub_type=None,
        extra_output=None,
        bulk_overlap=False,
    ):
        a_blk = isinstance(A, Float8BlockwiseQTensorBase)
        b_blk = isinstance(B, Float8BlockwiseQTensorBase)
        if not (a_blk and b_blk):
            return _orig_general_gemm(
                A, B, workspace, out_dtype, quantization_params, gelu, gelu_in,
                alpha, beta, accumulate, layout, out, bias, use_split_accumulator,
                grad, ub, ub_type, extra_output, bulk_overlap,
            )

        # TE computes  D = op(A) @ op(B)  in column-major / Fortran terms, where the
        # python-visible result is  out = B_used @ A_used^T  for layout "TN".
        # aiter_gemm(x[M,K], w[N,K], xs, ws) returns x @ w^T  with
        #   x quantized 1x128 (scale [M,K/128]), w quantized 128x128 (scale [N/128,K/128]).
        #
        # We pick, for each operand, the (rowwise vs columnwise) stored copy whose
        # logical layout is [contraction-last] = [*, K], then feed (activation, weight).
        #
        # layout semantics (transa = layout[0]=='T', transb = layout[1]=='T'):
        #   TN (fprop):  out[M,N] = X[M,K] @ W[N,K]^T   ; A=W (weight,2D), B=X (act,1D)
        #   NN (dgrad):  dX[M,K] = dY[M,N] @ W[N,K]     ; A=W, B=dY
        #   NT (wgrad):  dW[N,K] = dY[M,N]^T @ X[M,K]   ; A=X, B=dY
        #
        # In all three TE cases the result is  out = B_op @ A_op  where we map each side
        # to aiter's (x@w^T) by choosing rowwise/columnwise copies so the contracted dim
        # is last on both operands.
        transa = layout[0] == "T"
        transb = layout[1] == "T"

        # Choose the stored copy for each operand so its contraction dim is the LAST dim.
        # A operand: contract dim is last when transa (TN) -> rowwise [.,K]; else columnwise.
        a_data, a_scale, a_sdim = _extract(A, e4m3_torch, columnwise=not transa)
        # B operand: contract dim is last when NOT transb (TN/NN) -> rowwise; else columnwise.
        b_data, b_scale, b_sdim = _extract(B, e4m3_torch, columnwise=transb)

        # aiter signature: gemm(x, w, x_scale, w_scale) = x @ w^T, x is 1x128, w is 128x128.
        # The "activation"/1D operand is the 1x128 one (sdim==1); the "weight"/2D is 128x128.
        if a_sdim == 1 and b_sdim == 2:
            x_data, x_scale = a_data, a_scale
            w_data, w_scale = b_data, b_scale
            # result = x @ w^T = A @ B^T ; but TE wants B_op @ A_op = (A_op @ B_op^T)^T?
            # We resolve orientation empirically below via `swap`.
            swap = True
        elif b_sdim == 1 and a_sdim == 2:
            x_data, x_scale = b_data, b_scale
            w_data, w_scale = a_data, a_scale
            swap = False
        else:
            # both same dim (e.g. wgrad both 1x128, or both 2D): treat A as weight, B as act
            x_data, x_scale = b_data, b_scale
            w_data, w_scale = a_data, a_scale
            swap = False

        res = aiter_gemm(
            x_data, w_data, x_scale, w_scale,
            dtype=out_dtype if out_dtype is not None else torch.bfloat16,
        )
        # res = x @ w^T. Determine final orientation: TE result must equal B_op @ A_op.
        # With x=B, w=A: res = B @ A^T. If the TE op needs B @ A_op where A_op already
        # has contraction last (rowwise A used) the natural product is B @ A^T = res.
        # The `swap` path (x=A,w=B) gives A @ B^T whose transpose is B @ A^T.
        if swap:
            res = res.t().contiguous()

        if bias is not None:
            res = res + bias.to(res.dtype)
        if out is not None:
            out.copy_(res)
            res = out
        return res, None, None, None

    gemm_mod.general_gemm = general_gemm
    # te.Linear imports general_gemm by name into its module namespace; patch there too.
    try:
        import transformer_engine.pytorch.module.linear as _linmod
        if hasattr(_linmod, "general_gemm"):
            _linmod.general_gemm = general_gemm
    except Exception:
        pass
    try:
        import transformer_engine.pytorch.module.layernorm_linear as _lnlin
        if hasattr(_lnlin, "general_gemm"):
            _lnlin.general_gemm = general_gemm
    except Exception:
        pass
    try:
        import transformer_engine.pytorch.module.layernorm_mlp as _lnmlp
        if hasattr(_lnmlp, "general_gemm"):
            _lnmlp.general_gemm = general_gemm
    except Exception:
        pass


def enable() -> bool:
    """Idempotently route ROCm/TE blockwise FP8 through aiter. Call before building TE modules."""
    global _ENABLED
    if _ENABLED:
        return True
    if not _is_rocm():
        return False
    lift_te_gate()
    _patch_quantizer()
    _patch_gemm()
    _ENABLED = True
    return True
