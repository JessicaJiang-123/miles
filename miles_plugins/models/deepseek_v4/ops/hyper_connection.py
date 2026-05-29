"""DeepSeek V4 Hyper-Connection utility -- pure-PyTorch reference implementation.

Upstream (yueming): backs `hc_pre_raw`/`hc_post_raw`/`hc_head_raw` with the
``deepseek-ai/TileKernels`` CUDA kernels (``tile_kernels.modeling.mhc.ops``).
Those kernels are NOT available in the AMD MI355X image we target, so we
re-derive the math from sglang's reference torch implementation
(`sglang/srt/models/deepseek_v4.py::DeepseekV4HCBase.hc_pre/hc_post`) and the
TileLang sinkhorn kernel in `sglang/srt/layers/mhc.py::hc_split_sinkhorn_kernel`.

The math (per token of x of shape (B, S, hc_mult, hidden)):

  1. ``rms`` = rsqrt(mean(x_flat ** 2, dim=-1) + rms_eps)     # x_flat: (BS, hc_mult*hidden)
  2. ``mixes`` = F.linear(x_flat, hc_fn) * rms                # (BS, (2+hc_mult)*hc_mult)
  3. split mixes into [pre | post | comb]:
        pre[j]      = sigmoid(mixes[j]               * hc_scale[0] + hc_base[j])         + eps
        post[j]     = 2 * sigmoid(mixes[j + hc]      * hc_scale[1] + hc_base[j + hc])
        comb[j, k]  = mixes[j*hc + k + 2*hc] * hc_scale[2] + hc_base[j*hc + k + 2*hc]
  4. comb -> Sinkhorn-normalize: row softmax once, then alternate
     row-/col-normalize for ``sinkhorn_iters`` more iterations.
  5. layer_input = (pre.unsqueeze(-1) * x_flat).sum(dim=1)    # (BS, hidden)

Post-step (hc_post_raw):

      out = post.unsqueeze(-1) * x.unsqueeze(1)
            + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)

Heads (hc_head_raw) -- only the first hc_mult slots of `mixes` (the "pre" half)
are used; pre is mapped to a single hidden via the same weighted sum.

All ops are vanilla torch.{F.linear, sigmoid, sum, mean, rsqrt}. They are
gradient-correct (autograd handles backward). Numerical precision: inputs are
cast to fp32 inside the kernel (yueming's TileKernels do the same; the linear
GEMM is FP32 over BF16 inputs).
"""

import os

import einops
import torch
import torch.nn.functional as F
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor


# --- Diagnostic-2 dump for layer outputs + final hidden (MILES_DSV4_DUMP) ---
# hc_post_raw output == decoder-layer output (per layer); hc_head_raw output ==
# final hidden after all layers (pre final-layernorm/lm_head). Tagged by a global
# call counter so the 4 layers (8 hc_post calls: attn+ffn per layer) and the
# single hc_head call are distinguishable. fp32 CPU, rank-0-keyed, once per tag.
_DSV4_HC_DUMP_SEEN = set()
_DSV4_HC_POST_COUNT = 0


def _dsv4_hc_dump(tag, tensor):
    prefix = os.environ.get("MILES_DSV4_DUMP")
    if not prefix or tensor is None:
        return
    try:
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        rank = 0
    # Disk-frugal: rank-0 only (the hidden state is TP-replicated for these
    # layer-output/final-hidden tensors, so rank 0 is representative), and only
    # the first occurrence of each tag (first forward, layers in order).
    if rank != 0:
        return
    key = (tag, rank)
    if key in _DSV4_HC_DUMP_SEEN:
        return
    _DSV4_HC_DUMP_SEEN.add(key)
    try:
        torch.save(tensor.detach().float().cpu(), f"{prefix}.miles.{tag}.r{rank}.pt")
    except Exception as e:  # noqa: BLE001
        print(f"[dsv4-hc-dump] failed {tag}: {e}")


# DeepSeek-V4 post-layer mixer factor (matches sglang `hc_post_mult_value=2.0`).
_HC_POST_MULT_VALUE = 2.0


class HCHeadParams(MegatronModule):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        hc_mult = config.dsv4_hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_head_fn = torch.nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = torch.nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = torch.nn.Parameter(torch.empty(1, dtype=torch.float32))

        for p in [self.hc_head_fn, self.hc_head_base, self.hc_head_scale]:
            p._keep_fp32 = True

    def forward(self):
        raise NotImplementedError


def _sinkhorn_normalize(comb: Tensor, iters: int, eps: float) -> Tensor:
    """Sinkhorn normalize the last two dims of ``comb`` (..., hc, hc).

    Mirrors the TileLang kernel: row softmax once, then alternate
    row-/col-normalize for (iters - 1) more iterations.
    """
    # numerically stable row softmax
    row_max = comb.amax(dim=-1, keepdim=True)
    comb = torch.exp(comb - row_max)
    row_sum = comb.sum(dim=-1, keepdim=True)
    comb = comb / row_sum + eps

    col_sum = comb.sum(dim=-2, keepdim=True)
    comb = comb / (col_sum + eps)

    for _ in range(iters - 1):
        row_sum = comb.sum(dim=-1, keepdim=True)
        comb = comb / (row_sum + eps)
        col_sum = comb.sum(dim=-2, keepdim=True)
        comb = comb / (col_sum + eps)
    return comb


def _hc_pre_norm_mixes(x_flat: Tensor, hc_fn: Tensor, rms_eps: float) -> tuple[Tensor, Tensor]:
    """Compute (rms-normalized) mixes = F.linear(x_flat, hc_fn) / rms(x_flat).

    Inputs:
      x_flat: (n, hc_mult * hidden)  -- bf16 or fp32
      hc_fn:  (mix_hc, hc_mult * hidden)  -- fp32

    Returns:
      x_flat_fp32: fp32 copy of x_flat (so callers can re-use without recompute)
      mixes:       fp32 (n, mix_hc)
    """
    x_flat_fp32 = x_flat.float()
    rsqrt = torch.rsqrt(x_flat_fp32.square().mean(-1, keepdim=True) + rms_eps)
    mixes = F.linear(x_flat_fp32, hc_fn) * rsqrt
    return x_flat_fp32, mixes


def _hc_split_sinkhorn(
    mixes: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pure-torch port of sglang's hc_split_sinkhorn_kernel.

    Inputs are (B, S, mix_hc) where mix_hc = (2 + hc_mult) * hc_mult.
    Returns (pre, post, comb) with shapes (*, hc_mult), (*, hc_mult), (*, hc_mult, hc_mult).
    """
    hc = hc_mult
    # mixes: (..., mix_hc) -- index slots are [pre(hc) | post(hc) | comb(hc*hc)]
    pre_slot = mixes[..., :hc]
    post_slot = mixes[..., hc:2 * hc]
    comb_slot = mixes[..., 2 * hc:].reshape(*mixes.shape[:-1], hc, hc)

    pre_base = hc_base[:hc]
    post_base = hc_base[hc:2 * hc]
    comb_base = hc_base[2 * hc:].reshape(hc, hc)

    pre = torch.sigmoid(pre_slot * hc_scale[0] + pre_base) + eps
    post = _HC_POST_MULT_VALUE * torch.sigmoid(post_slot * hc_scale[1] + post_base)
    comb = comb_slot * hc_scale[2] + comb_base
    comb = _sinkhorn_normalize(comb, sinkhorn_iters, eps)
    return pre, post, comb


class DeepSeekV4HyperConnectionUtil:
    """Hyper-Connection helper (pure-PyTorch reference impl)."""

    def __init__(self, config: TransformerConfig):
        self.norm_eps = config.layernorm_epsilon
        self.hc_mult = config.dsv4_hc_mult
        self.hc_sinkhorn_iters = config.dsv4_hc_sinkhorn_iters
        self.hc_eps = config.dsv4_hc_eps

    def hc_pre_raw(
        self,
        x: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """``x`` is ``(B, S, hc_mult, hidden)``. Returns layer input + post/comb mixes."""
        assert hc_fn.dtype == torch.float32
        assert hc_scale.dtype == torch.float32
        assert hc_base.dtype == torch.float32
        dtype = x.dtype
        b, s, hc, hidden = x.shape
        # Flatten the (hc_mult, hidden) tail into one vector per token.
        x_flat = x.reshape(b * s, hc * hidden)
        x_flat_fp32, mixes = _hc_pre_norm_mixes(x_flat, hc_fn, self.norm_eps)
        pre, post, comb = _hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        # pre: (n, hc); x reshape: (n, hc, hidden). weighted sum across hc.
        x_per_token = x_flat_fp32.view(b * s, hc, hidden)
        layer_input = (pre.unsqueeze(-1) * x_per_token).sum(dim=1)  # (n, hidden)
        layer_input = layer_input.view(b, s, hidden).to(dtype)
        post = post.view(b, s, hc)
        comb = comb.view(b, s, hc, hc)
        return layer_input, post, comb

    def hc_post_raw(
        self,
        x: Tensor,
        residual: Tensor,
        post: Tensor,
        comb: Tensor,
    ) -> Tensor:
        """``x``: ``(B, S, hidden)``; ``residual``: ``(B, S, hc_mult, hidden)``.

        Returns ``(B, S, hc_mult, hidden)``: out[..., j, :] is
        ``post[..., j] * x + sum_k(comb[..., j, k] * residual[..., k, :])``.
        """
        dtype = x.dtype
        x_fp32 = x.float()
        residual_fp32 = residual.float()
        post_fp32 = post.float()
        comb_fp32 = comb.float()
        # post * x (B,S,hc,h): broadcast hc over hidden
        term_x = post_fp32.unsqueeze(-1) * x_fp32.unsqueeze(-2)  # (B,S,hc,hidden)
        # comb @ residual:  einsum('bsjk,bskh->bsjh', comb, residual)
        term_res = torch.einsum("bsjk,bskh->bsjh", comb_fp32, residual_fp32)
        out = (term_x + term_res).to(dtype)
        global _DSV4_HC_POST_COUNT
        _dsv4_hc_dump(f"hc_post{_DSV4_HC_POST_COUNT}", out)
        _DSV4_HC_POST_COUNT += 1
        return out

    def hc_head_raw(
        self,
        x: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ) -> Tensor:
        """``x``: ``(B, S, hc_mult, hidden)``. Returns ``(B, S, hidden)``.

        The head-mixer projects the hc_mult residual streams down to one. We
        reuse the same ``pre`` formula as in ``hc_pre_raw`` but pad/truncate
        the mixer FN to a single weighted-sum coefficient per stream.
        """
        assert hc_fn.dtype == torch.float32
        assert hc_scale.dtype == torch.float32
        assert hc_base.dtype == torch.float32
        dtype = x.dtype
        b, s, hc, hidden = x.shape
        x_flat = x.reshape(b * s, hc * hidden)
        x_flat_fp32 = x_flat.float()

        rsqrt = torch.rsqrt(x_flat_fp32.square().mean(-1, keepdim=True) + self.norm_eps)
        # hc_fn here is (hc_mult, hc_mult*hidden) (the head-mixer weights), so the
        # linear produces (n, hc_mult) directly: one scalar coefficient per residual stream.
        mixes = F.linear(x_flat_fp32, hc_fn) * rsqrt  # (n, hc)
        scale = hc_scale.reshape(-1)[0]  # single scalar
        pre = torch.sigmoid(mixes * scale + hc_base) + self.hc_eps  # (n, hc)
        x_per_token = x_flat_fp32.view(b * s, hc, hidden)
        layer_input = (pre.unsqueeze(-1) * x_per_token).sum(dim=1)  # (n, hidden)
        out = layer_input.view(b, s, hidden).to(dtype)
        _dsv4_hc_dump("final_hidden", out)  # post-all-layers, pre final-layernorm
        return out

    def layer_pre(
        self,
        hidden_states: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        assert hc_fn.dtype == torch.float32
        assert hc_scale.dtype == torch.float32
        assert hc_base.dtype == torch.float32

        x = einops.rearrange(hidden_states, "s b hc d -> b s hc d")
        x, post, comb = self.hc_pre_raw(x=x, hc_fn=hc_fn, hc_scale=hc_scale, hc_base=hc_base)
        hidden_states = einops.rearrange(x, "b s d -> s b d")
        return hidden_states, post, comb

    def layer_post(
        self,
        output_with_bias: Tensor | tuple[Tensor, Tensor | None],
        residual: Tensor,
        post: Tensor,
        comb: Tensor,
    ) -> Tensor:
        if isinstance(output_with_bias, tuple):
            out, bias = output_with_bias
            assert bias is None
        else:
            out = output_with_bias
        assert isinstance(out, torch.Tensor)

        out = einops.rearrange(out, "s b d -> b s d")
        residual_bshd = einops.rearrange(residual, "s b hc d -> b s hc d")
        hidden_states = self.hc_post_raw(x=out, residual=residual_bshd, post=post, comb=comb)
        return einops.rearrange(hidden_states, "b s hc d -> s b hc d")

    def block_expand(self, hidden_states: Tensor) -> Tensor:
        return einops.repeat(hidden_states, "s b d -> s b hc d", hc=self.hc_mult)

    def block_head(
        self,
        hidden_states: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ) -> Tensor:
        x = einops.rearrange(hidden_states, "s b hc d -> b s hc d")
        x = self.hc_head_raw(x=x, hc_fn=hc_fn, hc_scale=hc_scale, hc_base=hc_base)
        return einops.rearrange(x, "b s d -> s b d")
