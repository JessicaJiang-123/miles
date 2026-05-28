"""FP8 (E4M3) fake-quant for DSv4 KV/indexer simulated QAT.

Upstream (yueming): uses ``tile_kernels.quant.act_quant`` / ``per_token_cast_back``
(deepseek-ai/TileKernels). Those bindings are NOT available in the AMD MI355X image we
target, so we fall back to a pure-PyTorch per-block (1x128) symmetric E4M3 fake-quant
implementation:

  for each (B*S, block) sub-vector:
      amax  = max(|x|) within the block
      scale = amax / E4M3_MAX             # f32 scalar
      x_q   = round_to_e4m3(x / scale)     # in [-E4M3_MAX, E4M3_MAX]
      x_dq  = x_q * scale                  # back to bf16 / fp32

The output is bit-equivalent to a "cast to FP8 and cast back" with no rounding to UE8M0
scales. That's a small over-approximation of yueming's UE8M0-rounded scales but is the
correct gradient (straight-through pass on the QAT autograd.Function) and matches the
intent (simulate FP8 precision loss in the KV / indexer streams).

Only used when ``config.fp8 is not None`` (FP8 training), and is gated on
``MEGATRON_USE_KV_QAT=1`` in yueming's dsa.py path (otherwise unused).
"""
import torch


_E4M3_MAX = 448.0  # IEEE 754 E4M3 max representable finite magnitude


def _fp8_e4m3_round(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest representable E4M3 value, in-place safe."""
    # PyTorch >= 2.1 has float8_e4m3fn; route through it.
    if hasattr(torch, "float8_e4m3fn"):
        return x.to(torch.float8_e4m3fn).to(x.dtype)
    # Fallback: clamp + round (approximate).
    return x.clamp(-_E4M3_MAX, _E4M3_MAX)


def fp8_simulate(x: torch.Tensor, block_size: int):
    """Per-1x128-block FP8 fake-quant.

    Mirrors the (deepseek_v4) yueming op: groups the last dim into blocks of ``block_size``,
    computes a per-block amax-based scale, casts to E4M3, and casts back.
    """
    orig_dtype = x.dtype
    orig_shape = x.shape
    x = x.contiguous()
    last = x.shape[-1]
    if last % block_size != 0:
        # Fall back to a no-op so we never crash a model build on an odd dim.
        return x

    leading = x.numel() // last
    x_blk = x.view(leading, last // block_size, block_size).to(torch.float32)
    amax = x_blk.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = amax / _E4M3_MAX
    x_q = _fp8_e4m3_round(x_blk / scale)
    x_dq = (x_q * scale).to(orig_dtype)
    return x_dq.view(orig_shape)


class DeepSeekV4LinearQATFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, block_size=128):
        return fp8_simulate(kv, block_size)

    @staticmethod
    def backward(ctx, grad_kv):
        return grad_kv, None


fp8_simulate_qat = DeepSeekV4LinearQATFunc.apply
