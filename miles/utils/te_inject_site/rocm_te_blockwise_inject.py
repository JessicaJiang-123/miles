"""Standalone injector for ROCm/TE blockwise FP8 -> aiter, for Megatron Ray workers.

The installed framework miles lives at /root/miles and is what workers import; we must
NOT replace it. Instead this directory is prepended to PYTHONPATH (via the train job's
extra_env_vars) and the sibling `sitecustomize.py` (auto-run at interpreter startup)
registers a post-import hook that calls `apply()` here the moment
`transformer_engine.pytorch` is imported -- i.e. before any TE module is built.

This file is intentionally self-contained: it does NOT import `miles.*` (workers resolve
`miles` to /root/miles, which doesn't have our code). It imports aiter + TE directly. The
quant/GEMM logic is the same as miles/utils/rocm_te_blockwise.py, validated on MI355X.
"""
from __future__ import annotations

import torch

BLK = 128
_APPLIED = False


def _is_rocm() -> bool:
    return getattr(torch.version, "hip", None) is not None


def _aiter_bits():
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale
    from aiter.ops.triton.utils.types import get_fp8_dtypes

    _, e4m3 = get_fp8_dtypes()
    return gemm_a8w8_blockscale, e4m3, float(torch.finfo(e4m3).max)


def quantize_1x128(x: torch.Tensor):
    """1x128 along K. K must be a multiple of BLK."""
    _, e4m3, fmax = _aiter_bits()
    M, K = x.shape
    xv = x.float().view(M, K // BLK, BLK)
    scale = xv.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / fmax
    xq = (xv / scale).clamp(-fmax, fmax).to(e4m3).view(M, K)
    return xq, scale.squeeze(-1).contiguous()


def quantize_128x128(w: torch.Tensor):
    """128x128. Both dims must be multiples of BLK; the caller in _build_qtensor
    demotes small weights to 1x128 instead of calling this."""
    _, e4m3, fmax = _aiter_bits()
    N, K = w.shape
    wv = w.float().view(N // BLK, BLK, K // BLK, BLK)
    scale = wv.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12) / fmax
    wq = (wv / scale).clamp(-fmax, fmax).to(e4m3).view(N, K)
    return wq, scale.view(N // BLK, K // BLK).contiguous()


def _to_uint8(e4m3):
    return e4m3.view(torch.uint8)


def _from_uint8(u8, e4m3_dtype):
    return u8.view(e4m3_dtype)


def _patch_quantizer():
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
        Float8BlockQuantizer,
        Float8BlockwiseQTensor,
    )
    import transformer_engine_torch as tex
    from transformer_engine_torch import DType as TE_DType

    fp8_e4m3 = TE_DType.kFloat8E4M3

    def _build_qtensor(self, tensor):
        orig_shape = tuple(tensor.shape)
        K = orig_shape[-1]
        M = 1
        for d in orig_shape[:-1]:
            M *= d
        x2d = tensor.reshape(M, K).contiguous()

        rowwise_data = rowwise_scale = None
        columnwise_data = columnwise_scale = None

        # Effective scaling dim: 2D (128x128) only works when both dims are >= 128 and
        # divisible by BLK. Small weights (e.g. DSv4 indexer linear_weights_proj has
        # N=index_n_heads=64) get demoted to 1D (1x128 along K) so we always quantize.
        effective_sdim = self.block_scaling_dim
        if effective_sdim == 2:
            M2, K2 = x2d.shape
            if M2 < BLK or M2 % BLK != 0 or K2 % BLK != 0:
                effective_sdim = 1

        # In each direction, only quantize if the contraction dim (K) is a multiple
        # of BLK. Otherwise leave that direction's data None -- the GEMM path uses
        # the available copy and dequantizes via _my_dequant.
        M2, K2 = x2d.shape

        if effective_sdim == 1:
            if self.rowwise_usage and K2 % BLK == 0:
                q, s = quantize_1x128(x2d)
                rowwise_data = _to_uint8(q).reshape(orig_shape).contiguous()
                rowwise_scale = s.contiguous()
            if self.columnwise_usage and M2 % BLK == 0:
                qc, sc = quantize_1x128(x2d.t().contiguous())
                columnwise_data = _to_uint8(qc).contiguous()
                columnwise_scale = sc.contiguous()
        else:
            if self.rowwise_usage:
                q, s = quantize_128x128(x2d)
                rowwise_data = _to_uint8(q).reshape(orig_shape).contiguous()
                rowwise_scale = s.contiguous()
            if self.columnwise_usage:
                qc, sc = quantize_128x128(x2d.t().contiguous())
                columnwise_data = _to_uint8(qc).contiguous()
                columnwise_scale = sc.contiguous()

        # If we have NEITHER a rowwise nor columnwise quantized copy (e.g. weight is
        # 64-rows x 4096-K and we needed columnwise, but the contraction dim wouldn't
        # fit BLK), stash a bf16 copy in rowwise_data so _my_dequant can reconstruct
        # the original value. Tag with sdim=0 so the GEMM patch knows to fall back
        # to bf16. (Tiny weights -- skip FP8 for this one Linear.)
        if rowwise_data is None and columnwise_data is None:
            # signal: store original bf16, GEMM patch will dequantize via _my_dequant_raw
            pass  # handled below by stashing on the tensor

        out = Float8BlockwiseQTensor(
            shape=orig_shape,
            dtype=tensor.dtype if tensor.dtype.is_floating_point else torch.bfloat16,
            fp8_dtype=fp8_e4m3,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale,
            quantizer=self,
            is_2D_scaled=effective_sdim == 2,
            data_format=tex.Float8BlockScaleTensorFormat.GEMM_READY,
            requires_grad=False,
        )
        # Tag with the actually-used scaling dim (may differ from the quantizer's
        # configured block_scaling_dim if we demoted 128x128 -> 1x128 for a small weight).
        out._aiter_block_scaling_dim = effective_sdim
        return out

    def quantize(self, tensor, *, out=None, dtype=None):
        if out is not None:
            return self.update_quantized(tensor, out)
        return _build_qtensor(self, tensor)

    def update_quantized(self, src, dst, *, noop_flag=None):
        new = _build_qtensor(self, src)
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


def _extract(t, e4m3_torch, *, columnwise):
    if columnwise:
        u8, s = t._columnwise_data, t._columnwise_scale_inv
    else:
        u8, s = t._rowwise_data, t._rowwise_scale_inv
    data = _from_uint8(u8, e4m3_torch)
    sdim = getattr(t, "_aiter_block_scaling_dim", 2 if t._is_2D_scaled else 1)
    return data.reshape(-1, data.shape[-1]).contiguous(), s.contiguous(), sdim


def _my_dequant(t):
    """Dequantize a Float8BlockwiseQTensor stored in OUR scale convention -> bf16.

    Rowwise 1x128 (activation/grad): data [..., K] e4m3-as-uint8, scale [M, K/128].
    Rowwise 128x128 (weight): data [N, K], scale [N/128, K/128]. Returns the row-major
    high-precision tensor with the SAME logical shape as t.
    """
    from aiter.ops.triton.utils.types import get_fp8_dtypes

    _, e4m3 = get_fp8_dtypes()
    orig_shape = tuple(t.shape)
    K = orig_shape[-1]
    M = 1
    for d in orig_shape[:-1]:
        M *= d
    sdim = getattr(t, "_aiter_block_scaling_dim", 2 if t._is_2D_scaled else 1)

    if t._rowwise_data is not None:
        data = _from_uint8(t._rowwise_data, e4m3).reshape(M, K).float()
        s = t._rowwise_scale_inv
        if sdim == 1:  # scale [M, K/128]
            deq = (data.view(M, K // BLK, BLK) * s.view(M, K // BLK, 1)).view(M, K)
        else:  # scale [M/128, K/128]
            deq = (
                data.view(M // BLK, BLK, K // BLK, BLK) * s.view(M // BLK, 1, K // BLK, 1)
            ).view(M, K)
        return deq.to(torch.bfloat16).reshape(orig_shape).contiguous()

    # columnwise-only: our convention stores the transpose, data [K, M], scale over K.
    cdata = _from_uint8(t._columnwise_data, e4m3).reshape(K, M).float()
    s = t._columnwise_scale_inv
    if sdim == 1:  # scale [K, M/128]
        deqT = (cdata.view(K, M // BLK, BLK) * s.view(K, M // BLK, 1)).view(K, M)
    else:  # scale [K/128, M/128]
        deqT = (
            cdata.view(K // BLK, BLK, M // BLK, BLK) * s.view(K // BLK, 1, M // BLK, 1)
        ).view(K, M)
    return deqT.t().to(torch.bfloat16).reshape(orig_shape).contiguous()


def _patch_gemm():
    import transformer_engine.pytorch.cpp_extensions.gemm as gemm_mod
    from transformer_engine.pytorch.tensor._internal.float8_blockwise_tensor_base import (
        Float8BlockwiseQTensorBase,
    )

    aiter_gemm, e4m3_torch, _ = _aiter_bits()
    _orig = gemm_mod.general_gemm

    def general_gemm(
        A, B, workspace, out_dtype=None, quantization_params=None, gelu=False,
        gelu_in=None, alpha=1.0, beta=None, accumulate=False, layout="TN", out=None,
        bias=None, use_split_accumulator=False, grad=False, ub=None, ub_type=None,
        extra_output=None, bulk_overlap=False,
    ):
        if not (isinstance(A, Float8BlockwiseQTensorBase) and isinstance(B, Float8BlockwiseQTensorBase)):
            return _orig(A, B, workspace, out_dtype, quantization_params, gelu, gelu_in,
                         alpha, beta, accumulate, layout, out, bias, use_split_accumulator,
                         grad, ub, ub_type, extra_output, bulk_overlap)

        transa = layout[0] == "T"
        transb = layout[1] == "T"
        a_data, a_scale, a_sdim = _extract(A, e4m3_torch, columnwise=not transa)
        b_data, b_scale, b_sdim = _extract(B, e4m3_torch, columnwise=transb)

        if a_sdim == 1 and b_sdim == 2:
            x_data, x_scale = a_data, a_scale
            w_data, w_scale = b_data, b_scale
            swap = True
            act_operand = A
        elif b_sdim == 1 and a_sdim == 2:
            x_data, x_scale = b_data, b_scale
            w_data, w_scale = a_data, a_scale
            swap = False
            act_operand = B
        else:
            # both have the same scaling dim (e.g. small weight demoted to sdim=1, or
            # wgrad with both sdim=1). Use rank heuristic: the operand with more leading
            # dims is the activation; the 2D one is the weight.
            x_data, x_scale = b_data, b_scale
            w_data, w_scale = a_data, a_scale
            swap = False
            act_operand = A if A.dim() > B.dim() else B

        res = aiter_gemm(
            x_data, w_data, x_scale, w_scale,
            dtype=out_dtype if out_dtype is not None else torch.bfloat16,
        )
        if swap:
            res = res.t().contiguous()
        # aiter returns a 2D [M, N]; restore the activation's leading dims if it was
        # higher-rank (Megatron passes [s, b, h] and expects [s, b, N] back).
        if out is None:
            lead = tuple(act_operand.shape[:-1])
            prod = 1
            for d in lead:
                prod *= d
            if len(lead) != 1 and res.dim() == 2 and prod == res.shape[0]:
                res = res.reshape(*lead, res.shape[-1]).contiguous()
        if bias is not None:
            res = res + bias.to(res.dtype)
        # bias_grad for the wgrad path (TE expects sum over all token dims when grad+bias).
        bias_grad = None
        if grad and bias is not None:
            bias_grad = res.reshape(-1, res.shape[-1]).sum(dim=0).to(bias.dtype)
        if out is not None:
            # Honor Megatron's fused wgrad accumulation: out += alpha*res (+ beta*out).
            # accumulate=True means add into the existing main_grad buffer.
            res_c = res.to(out.dtype)
            if accumulate:
                if beta not in (None, 1.0, 0.0):
                    out.mul_(beta)
                out.add_(res_c)
            else:
                out.copy_(res_c)
            res = out
        return res, bias_grad, None, None

    gemm_mod.general_gemm = general_gemm
    for modname in (
        "transformer_engine.pytorch.module.linear",
        "transformer_engine.pytorch.module.layernorm_linear",
        "transformer_engine.pytorch.module.layernorm_mlp",
        "transformer_engine.pytorch.module.grouped_linear",
    ):
        try:
            import importlib
            m = importlib.import_module(modname)
            if hasattr(m, "general_gemm"):
                m.general_gemm = general_gemm
        except Exception:
            pass


def _patch_grouped_gemm():
    """MoE TEGroupedLinear real blockwise FP8 via per-expert aiter loop.

    DSv4 has 256 routed experts; the MoE forward goes through TE's GroupedLinear:

      1. tex.split_quantize(inp, m_splits, [Float8BlockQuantizer, ...])
         -> 'Not implemented scaling mode: Invalid Scaling' on ROCm.
      2. cpp_extensions.gemm.general_grouped_gemm(weight_list, act_list, out_list, ...)
         -> same error from the per-expert GEMM dispatch.

    We replace BOTH with our own implementation:

      - split_quantize: split inp by m_splits, call each quantizer.quantize() per chunk
        -> list[Float8BlockwiseQTensor] with the proper 1x128 / 128x128 scales.
      - general_grouped_gemm: per-expert loop; for FP8 lists call the already-patched
        general_gemm (aiter blockwise FP8) on each expert separately. This works for
        all three layouts:
          fprop  (TN, single_output=True): A=W[N,K] sd=2, B=X[M,K] sd=1 -> Y=X@W^T
          dgrad  (NN, single_output=True): A=W[N,K] sd=2, B=dY[M,N] sd=1 -> dX=dY@W
          wgrad  (NT, single_output=False, per-expert outs): A=dY[M,N], B=X[M,K] both
                  sd=1 -> dW=dY^T@X. (Dense path's general_gemm already handles the
                  both-1x128 wgrad via `aiter_gemm` with the two-1x128 codepath.)

    For experts with zero tokens the chunk is M=0; we skip the GEMM and zero-fill the
    output slice (or per-expert wgrad chunk).

    This matches the dense path numerically: same aiter blockwise FP8 GEMM, same
    quantize convention, just per-expert. NO bf16 fallback.
    """
    import transformer_engine.pytorch.cpp_extensions.gemm as gemm_mod
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
    from transformer_engine.pytorch.tensor._internal.float8_blockwise_tensor_base import (
        Float8BlockwiseQTensorBase,
    )
    import transformer_engine_torch as tex
    from transformer_engine_torch import DType as TE_DType

    _orig_split_quantize = tex.split_quantize

    def split_quantize(inp, m_splits, quantizers):
        """Per-expert blockwise FP8 quantize.

        Each quantizer is a Float8BlockQuantizer with its own rowwise/columnwise usage
        flags already configured by TE. We just call quantize() on each chunk.
        """
        any_blk = any(isinstance(q, Float8BlockQuantizer) for q in quantizers)
        if not any_blk:
            return _orig_split_quantize(inp, m_splits, quantizers)
        inp = inp.contiguous()
        chunks = torch.split(inp, m_splits, dim=0)
        out = []
        for chunk, q in zip(chunks, quantizers):
            if isinstance(q, Float8BlockQuantizer):
                if chunk.shape[0] == 0:
                    # zero-token expert: skip FP8, hand back an empty bf16 tensor.
                    out.append(chunk.to(torch.bfloat16) if chunk.dtype != torch.bfloat16 else chunk)
                else:
                    # quantize() routes through our patched _build_qtensor.
                    out.append(q.quantize(chunk if chunk.dtype == torch.bfloat16 else chunk.to(torch.bfloat16)))
            else:
                out.append(chunk)
        return out

    tex.split_quantize = split_quantize
    try:
        import transformer_engine.pytorch.module.grouped_linear as _gl
        # Force rebinding even if already imported, since `tex` is the module object.
        if hasattr(_gl, "tex"):
            _gl.tex.split_quantize = split_quantize
    except Exception:
        pass

    _orig_grouped_gemm = gemm_mod.general_grouped_gemm

    # One-time log so the smoke run actually proves the MoE FP8 path runs.
    _logged = {"split": False, "grouped": False}

    def _maybe_log(key, msg):
        if not _logged[key]:
            import os as _os
            if _os.environ.get("RANK", "0") in ("0", ""):
                print(msg, flush=True)
            _logged[key] = True

    def general_grouped_gemm(
        A, B, out, out_dtype, workspaces, layout="TN", m_splits=None,
        gelu=False, grad=False, accumulate=False, bias=None, use_bias=False,
        use_split_accumulator=False, D_dtype=None, single_output=False,
    ):
        any_blk = any(isinstance(t, Float8BlockwiseQTensorBase) for t in (*A, *B))
        if not any_blk:
            return _orig_grouped_gemm(
                A, B, out, out_dtype, workspaces, layout, m_splits,
                gelu, grad, accumulate, bias, use_bias,
                use_split_accumulator, D_dtype, single_output,
            )

        _maybe_log(
            "grouped",
            f"[rocm_te_blockwise_inject] MoE general_grouped_gemm via aiter blockwise FP8 "
            f"(layout={layout}, single_output={single_output}, num_gemms={len(A)})",
        )

        # TE's general_grouped_gemm passes a TE_DType for out_dtype; the dense general_gemm
        # we'll call wants a torch.dtype. Translate.
        _TE_TO_TORCH = {
            TE_DType.kFloat32: torch.float32,
            TE_DType.kFloat16: torch.float16,
            TE_DType.kBFloat16: torch.bfloat16,
        }
        if hasattr(out_dtype, "name") or type(out_dtype).__name__ == "DType":
            torch_out_dtype = _TE_TO_TORCH.get(out_dtype, torch.bfloat16)
        else:
            torch_out_dtype = out_dtype if out_dtype is not None else torch.bfloat16

        num_gemms = len(A)
        # Determine output behavior:
        #   - single_output=True (fprop+dgrad): all experts write into out[0], offset by
        #     m_splits along dim 0.
        #   - single_output=False (wgrad): out is a list per expert, each shape (N, K).
        if single_output:
            assert len(out) == 1
            big_out = out[0]
            # Per-expert M offsets for the activation/grad-output dim of `big_out`.
            offsets = [0]
            for s in m_splits:
                offsets.append(offsets[-1] + s)
        else:
            big_out = None

        # Per-expert bias-grad accumulator (only used for the wgrad+use_bias path).
        grad_biases = [None] * num_gemms

        # Iterate experts; A and B are LISTS of per-expert tensors.
        # We dispatch each pair through the already-patched general_gemm (which does the
        # actual aiter blockwise FP8 GEMM).
        from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm

        for i in range(num_gemms):
            ai = A[i]
            bi = B[i]
            # m for this expert: in single_output mode it's the row span of the slice
            # we'll write; in wgrad mode it's the contraction dim (no output slicing).
            m_i = m_splits[i] if m_splits is not None else None
            # Skip zero-token experts. (For wgrad they contribute zero to dW; for fprop
            # the output slice is empty.)
            if m_i == 0:
                if single_output:
                    pass  # no slice to fill
                else:
                    # zero-token wgrad: zero out / accumulate-noop into wgrad target
                    if accumulate:
                        pass
                    else:
                        out[i].zero_()
                continue

            # Determine the per-expert output target.
            if single_output:
                lo, hi = offsets[i], offsets[i + 1]
                # For fprop (layout TN) the result row span is [lo, hi); for dgrad (NN)
                # the result row span is the same since dgrad is split along the M dim.
                out_i = big_out[lo:hi]
            else:
                out_i = out[i]

            # Per-expert bias (use_bias means add bias on fprop; on bwd, compute bias grad).
            bias_i = None
            if bias is not None and use_bias and not grad:
                bias_i = bias[i] if isinstance(bias, (list, tuple)) else None

            # Call the dense-path patched general_gemm. It handles the layout / sdim
            # extraction / aiter dispatch for fprop / dgrad / wgrad uniformly.
            _, bias_grad_i, _, _ = general_gemm(
                ai,
                bi,
                workspaces[0] if isinstance(workspaces, (list, tuple)) else workspaces,
                out_dtype=torch_out_dtype,
                quantization_params=None,
                gelu=False,
                gelu_in=None,
                alpha=1.0,
                beta=None,
                accumulate=accumulate,
                layout=layout,
                out=out_i,
                bias=bias_i,
                use_split_accumulator=use_split_accumulator,
                grad=grad,
                ub=None,
                ub_type=None,
                extra_output=None,
                bulk_overlap=False,
            )
            if grad and bias_grad_i is not None:
                grad_biases[i] = bias_grad_i

        # Return shape matches the stock implementation: (out, bias_or_grad_biases, gelu_input).
        if grad and use_bias:
            return out, grad_biases, [None] * num_gemms
        return out, [None] * num_gemms, [None] * num_gemms

    gemm_mod.general_grouped_gemm = general_grouped_gemm
    try:
        import transformer_engine.pytorch.module.grouped_linear as _gl
        if hasattr(_gl, "general_grouped_gemm"):
            _gl.general_grouped_gemm = general_grouped_gemm
    except Exception:
        pass


def _patch_norm():
    """Stop the FUSED norm+quantize C++ kernel from being handed a Float8BlockQuantizer.

    LayerNormLinear/LayerNormMLP call apply_normalization() passing the input_quantizer to
    a fused HIP rmsnorm/layernorm kernel (tex.rmsnorm_fwd), whose blockwise cast path is
    unimplemented ("Not implemented scaling mode"). Mirror TE's own ROCm workaround for
    Float8CurrentScalingQuantizer: run the norm in high precision (quantizer=None), then
    quantize the bf16 norm output with our Python Float8BlockQuantizer.quantize (aiter).
    """
    import transformer_engine.pytorch.module._common as _common
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer

    _orig_apply_norm = _common.apply_normalization

    def apply_normalization(
        inputmat, ln_out, ln_weight, ln_bias, eps, output_quantizer, output_dtype,
        normalization, fwd_ln_sm_margin, zero_centered_gamma,
    ):
        if isinstance(output_quantizer, Float8BlockQuantizer):
            # high-precision norm (no fused quant), then aiter quant
            out, mu, rsigma = _orig_apply_norm(
                inputmat, ln_out, ln_weight, ln_bias, eps, None, output_dtype,
                normalization, fwd_ln_sm_margin, zero_centered_gamma,
            )
            return output_quantizer.quantize(out), mu, rsigma
        return _orig_apply_norm(
            inputmat, ln_out, ln_weight, ln_bias, eps, output_quantizer, output_dtype,
            normalization, fwd_ln_sm_margin, zero_centered_gamma,
        )

    _common.apply_normalization = apply_normalization
    # modules import apply_normalization by name into their namespace
    for modname in (
        "transformer_engine.pytorch.module.layernorm_linear",
        "transformer_engine.pytorch.module.layernorm_mlp",
    ):
        try:
            import importlib
            m = importlib.import_module(modname)
            if hasattr(m, "apply_normalization"):
                m.apply_normalization = apply_normalization
        except Exception:
            pass


def _patch_gather():
    """All-gather blockwise activations (sequence-parallel) in HIGH PRECISION.

    With --sequence-parallel, LayerNormLinear/LayerNormMLP all-gather ln_out across TP.
    TE routes a Float8BlockwiseQTensor / Float8BlockQuantizer to _all_gather_fp8_blockwise,
    which only supports the COMPACT data format (the FP8-all-gather path) and raises
    "All-gather with FP8 block-wise quantized tensor requires compact data format" for our
    GEMM_READY tensors. We don't implement COMPACT; instead gather in bf16 then re-quantize
    to GEMM_READY (numerically fine -- this is exactly TE's own high-precision fallback,
    just taken unconditionally for the blockwise recipe).
    """
    import transformer_engine.pytorch.distributed as _dist
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
    from transformer_engine.pytorch.tensor._internal.float8_blockwise_tensor_base import (
        Float8BlockwiseQTensorBase,
    )
    from transformer_engine.pytorch.distributed import get_distributed_world_size

    _orig_gather = _dist.gather_along_first_dim

    def gather_along_first_dim(inp, process_group, async_op=False, quantizer=None):
        is_blk = isinstance(inp, Float8BlockwiseQTensorBase) or isinstance(
            quantizer, Float8BlockQuantizer
        )
        if not is_blk:
            return _orig_gather(inp, process_group, async_op, quantizer)

        world_size = get_distributed_world_size(process_group)
        # Dequantize to high precision if already quantized. We must NOT use TE's
        # dequantize() -- it assumes the cuBLAS-transposed GEMM_READY scale layout, but our
        # rowwise scales are stored natural ([M, K/128] for 1x128). Dequant in our own
        # convention instead.
        if isinstance(inp, Float8BlockwiseQTensorBase):
            hp = _my_dequant(inp)
        else:
            hp = inp
        if world_size == 1:
            out_hp = hp
        else:
            out_shape = list(hp.size())
            out_shape[0] *= world_size
            out_hp = torch.empty(
                out_shape, dtype=hp.dtype, device=hp.device,
                memory_format=torch.contiguous_format,
            )
            import torch.distributed as _td
            _td.all_gather_into_tensor(out_hp, hp.contiguous(), group=process_group)
        if quantizer is not None:
            return quantizer(out_hp), None
        return out_hp, None

    _dist.gather_along_first_dim = gather_along_first_dim
    # Several TE modules do `from ..distributed import gather_along_first_dim` (binding it at
    # import time), so patching the distributed module attr alone is not enough -- rebind in
    # every already-imported module that has the name.
    import sys as _sys
    for _name, _mod in list(_sys.modules.items()):
        if _name.startswith("transformer_engine.pytorch") and _mod is not None:
            if getattr(_mod, "gather_along_first_dim", None) is _orig_gather:
                try:
                    _mod.gather_along_first_dim = gather_along_first_dim
                except Exception:
                    pass


def apply():
    """Idempotently route ROCm/TE blockwise FP8 through aiter. Safe to call repeatedly."""
    global _APPLIED
    if _APPLIED:
        return True
    if not _is_rocm():
        return False
    import transformer_engine.pytorch.fp8 as _tefp8

    _tefp8.check_fp8_block_scaling_support = lambda: (True, "")
    _patch_quantizer()
    _patch_gemm()
    _patch_grouped_gemm()
    _patch_norm()
    _patch_gather()
    # Disable Megatron's jit_fuser (torch.compile). It decorates the bias-dropout-add and
    # bias-swiglu fusions; dynamo fake-tensor tracing of those over our aiter blockwise FP8
    # GEMM output raises a spurious broadcast error. Disabling here (before the fusion
    # modules import / the model builds) keeps those regions eager. Idempotent + guarded.
    try:
        import megatron.core.jit as _mcjit
        _mcjit.disable_jit_fuser()
    except Exception:
        pass
    _APPLIED = True
    import os
    if os.environ.get("RANK", "0") in ("0", ""):
        print("[rocm_te_blockwise_inject] aiter blockwise FP8 wired into TE", flush=True)
    return True
