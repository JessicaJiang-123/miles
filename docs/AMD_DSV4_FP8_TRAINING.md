# AMD/ROCm DeepSeek-V4 Blockwise FP8 Training — Engineering Summary

Branches: `amd-fp8-training`, `amd-fp8-wgrad-symmetric`, `amd-fp8-te-run`,
`amd-fp8-dsv4-smoke` (all `JessicaJiang-123/miles`).
Container: `rlsys/miles:MI350-355-latest` (8× MI355X gfx950).
TransformerEngine: `2.8.0+a365f2de` (ROCm fork).
Status as of doc commit: qwen3-4B dense blockwise FP8 end-to-end validated;
DSv4-Flash 4-layer 3-GRPO-step smoke completes with real MoE blockwise FP8.

---

## 1. Executive Summary

The goal was to make DeepSeek-V4 trainable end-to-end on AMD MI355X under the
DeepSeek-style Float8BlockScaling recipe (1×128 act / 128×128 weight, E4M3,
FP32 scales), so that the training side matches what sglang+aiter already does
for the rollout side on AMD. The blockwise recipe is what DSv4-Flash actually
needs — MXFP8 (1×32, E8M0) is a different scheme and an earlier MXFP8-train +
blockwise-rollout experiment showed both that the schemes disagree and that
ROCm's MXFP8 GEMM is broken on this image ("`rocRoller only supports F32 as
scale type not Half`").

Headline outcomes per branch:

- `amd-fp8-training` (commits `1d7c16d`, `70973c2`): standalone validated
  blockwise FP8 linear core in `miles/utils/rocm_fp8_blockwise.py`. All three
  GEMMs (fprop / dgrad / wgrad) route through aiter's `gemm_a8w8_blockscale`.
  Standalone MI355X: fwd ~3.7%, dgrad ~4.1%, wgrad ~5.0% rel-err vs bf16.
- `amd-fp8-wgrad-symmetric` (commit `8d66e84`): a faithful both-1×128 wgrad
  via `symmetric_blockscale_gemm` that reuses the same aiter kernel by
  collapsing GROUP_N to 1. Standalone: wgrad rel-err ~3.6% (vs 5.0% with the
  v1 X-as-128×128 approximation).
- `amd-fp8-te-run` (9 commits, `60a01a9`…`5f7f6b1`): full
  TransformerEngine-layer wiring via a sitecustomize MetaPathFinder + runtime
  monkeypatch (`miles/utils/rocm_te_blockwise.py`). qwen3-4B dense smoke
  completes 3 GRPO steps under `--fp8-recipe blockwise` on 8× MI355X with
  TP2/CP4/SP, train-vs-rollout logprob diff bounded by the bf16 baseline.
- `amd-fp8-dsv4-smoke` (31 commits, HEAD `a539171`): yueming's DSv4 PR
  cherry-picked + ROCm-ported, weight-conversion plumbing, real MoE blockwise
  FP8 via per-expert aiter loop, fake_rollout for sglang-free smoke; 4-layer
  DSv4-Flash-FP8 completes 3 GRPO training steps end-to-end with FP8
  numerics, finite gradients.

Unresolved gaps (in increasing order of importance for parity with NV):
seq-parallel gather is bf16-then-requant (~2× comm vs NV blockwise gather);
norm+quant is split (bf16 norm then aiter quant) vs NV's fused HIP kernel;
the DSv4 launcher inherits the te-run X-as-128×128 wgrad approximation
(symmetric wgrad lives only on `amd-fp8-wgrad-symmetric`, not yet merged
forward); the MoE fprop is a per-expert aiter loop, not the fused
`fmoe_fp8_blockscale_g1u1` (numerically equivalent within ~3.3%, perf only;
fusion proven viable, wiring in flight); and we have NOT proved
convergence/accuracy or multi-node or full 284B DSv4-Flash. The smoke uses a
synthetic rollout — real RL training would use sglang.

## 2. Strategic Context

**Why blockwise FP8, not MXFP8.** DeepSeek-V3/V4 ship as a *block-scaled* FP8
model: activations and gradients are quantized in 1×128 groups along the
contraction axis, weights in 128×128 blocks, with E4M3 data and FP32 scales.
This is NV's `Float8BlockScaling` recipe and Megatron's
`--fp8-recipe blockwise`. MXFP8 (the OCP spec) is 1×32 along the contraction
axis, with E8M0 (power-of-two) scales. The two are numerically distinct,
and they are NOT interchangeable: a 1×128/E4M3-FP32 quantized GEMM cannot be
reproduced by a 1×32/E8M0 GEMM. Train must therefore match rollout
(sglang serves DSv4 on AMD using aiter's blockwise FP8). An earlier
agent-led experiment that ran MXFP8 training with blockwise rollout
confirmed not just the math mismatch but also that ROCm/TE's MXFP8 GEMM is
itself broken on this image (`rocRoller only supports F32 as scale type not
Half`), so MXFP8 was not a viable shortcut.

**Why ROCm needs work that NV didn't.** NVIDIA/TransformerEngine shipped
blockwise FP8 (Float8BlockScaling) during the DeepSeek-V3 era; that's why
yueming's DSv4 PR `radixark/miles#1045` only touched miles + Megatron and
never TE. On AMD that whole layer is missing:
`ROCm/TransformerEngine` hard-gates the blockwise recipe off
(`transformer_engine/pytorch/quantization.py:103-106`:
`if IS_HIP_EXTENSION: return False, "FP8 block scaled gemm not yet supported
for ROCm"`), and its HIP cast / GEMM kernels don't implement the DeepSeek
mode. The hipBLASLt blockwise path that does exist
(`common/gemm/rocm_gemm.cu:1281-1287`) only supports `VEC32_UE8M0`
(i.e. MXFP8), and explicitly errors on any other block-scaling combination
("`Not implemented scaling modes: ... and ...`"). The actual 1×128 E4M3 + FP32
kernels exist on AMD only in `aiter` (`gemm_a8w8_blockscale`), which is what
sglang already uses for inference. The whole effort here is wiring TE to
route the blockwise recipe through aiter, in Python, monkeypatched at runtime.

**Two TE repos, one used.** The installed TE in the container is
`ROCm/TransformerEngine` (AMD fork). `NVIDIA/TransformerEngine` is
CUDA-only. The `/mnt/data/data/hai/TransformerEngine` checkout in this
worktree-set is the ROCm fork (the gate + the rocm_gemm we just cited live
there). All TE-side fixes here, when upstreamed, would PR against
`ROCm/TransformerEngine`.

## 3. Architecture Overview

```
miles  (training framework)
   |
   | uses --fp8-recipe blockwise -> Megatron's Float8BlockScaling autocast
   v
Megatron-Core (yueming-megatron/deepseek-v4 fork for DSv4)
   |
   | TE-backed layers: te.Linear, te.LayerNormLinear, te.LayerNormMLP,
   | te.GroupedLinear (MoE), Float8BlockQuantizer
   v
TransformerEngine (ROCm fork, installed at /root/...)
   |        ^
   |        | runtime monkeypatch (sitecustomize MetaPathFinder)
   |        +-- rocm_te_blockwise_inject.apply() rebinds:
   |               check_fp8_block_scaling_support  (gate)
   |               Float8BlockQuantizer.quantize    (quant)
   |               cpp_extensions.gemm.general_gemm (dense GEMM)
   |               cpp_extensions.gemm.general_grouped_gemm + tex.split_quantize (MoE)
   |               module._common.apply_normalization (norm+quant)
   |               distributed.gather_along_first_dim (SP gather)
   |               module.fp8_padding.Fp8Padding/Unpadding.align_size (-> 128)
   v                       v
hipBLASLt (non-blockwise) | aiter.gemm_a8w8_blockscale   <-- the actual blockwise FP8 GEMM
                          | (same kernel sglang uses for DSv4 FP8 serving)
```

Branch placement on the stack:

- `amd-fp8-training` / `amd-fp8-wgrad-symmetric` add the bottom-most box
  (the standalone aiter-backed blockwise FP8 linear + quantize). They DO
  call `lift_te_gate()` but do not patch the rest of TE; they're the
  kernel-correctness substrate.
- `amd-fp8-te-run` adds everything between the bottom-most box and TE
  (the monkeypatch surface — gate / quantizer / GEMM / norm / gather), plus
  the sitecustomize MetaPathFinder that runs it inside every Megatron Ray
  worker. After this branch, qwen3-4B (dense) is end-to-end.
- `amd-fp8-dsv4-smoke` adds the MoE-specific patches (real per-expert aiter
  loop for `split_quantize` + `general_grouped_gemm`, `Fp8Padding` align→128,
  always-emit-columnwise so `update_usage(columnwise=True)` succeeds,
  small-weight zero-pad), the DSv4 model spec / mbridge / weight
  conversion / launcher / synthetic-rollout / `arguments.py` shim /
  ROCm bf16×bf16→fp32 GEMM workaround.

**Injection mechanism.** `miles/utils/te_inject_site/sitecustomize.py` is
auto-run at interpreter startup whenever the dir is on `PYTHONPATH`. It
gates on `ROCM_TE_BLOCKWISE_INJECT=1`, installs a `MetaPathFinder` at the
front of `sys.meta_path`, intercepts the import of
`transformer_engine.pytorch`, wraps the spec loader, and calls
`rocm_te_blockwise_inject.apply()` immediately after `exec_module`
completes — i.e. before any TE module is built. The injector itself
(`miles/utils/te_inject_site/rocm_te_blockwise_inject.py`) is a
self-contained copy of the patch logic; it must NOT import `miles.*`
because Ray workers resolve `miles` to the editable install at
`/root/miles` which has none of these changes. (The "Sync MoE patches into
the te_inject_site injector" commit `1f181a2` exists exactly to enforce
this rule: edits to `miles/utils/rocm_te_blockwise.py` don't reach the
workers.) The DSv4 launcher additionally pins `train_script` to the
worktree's `train.py` so the script-dir entry on `sys.path[0]` carries the
worktree's `miles.utils.arguments` over the installed copy.

## 4. Branch-by-Branch Chronology (brief)

**`amd-fp8-training`** (1d7c16d, 70973c2). Bring up the blockwise FP8
substrate as a standalone, testable thing. `1d7c16d` lands fprop+dgrad in
FP8 and a bf16 wgrad fallback. `70973c2` upgrades wgrad to FP8 via an
asymmetric mapping `dW = aiter(a=dY^T 1×128, b=X^T 128×128)`, i.e. treats
X^T as a 128×128-block weight even though it's an activation along its
contraction axis. All three GEMMs are now FP8 but wgrad over-quantizes one
operand. Validated: fwd 3.7%, dgrad 4.1%, wgrad 5.0% vs bf16. Single file
(`miles/utils/rocm_fp8_blockwise.py`) + one test.

**`amd-fp8-wgrad-symmetric`** (8d66e84). Faithful wgrad: both operands are
activation-like (`dW = dY^T @ X`, contracting the token dim M), so both
are quantized 1×128 along M. Adds `symmetric_blockscale_gemm(A, B, A_s,
B_s) -> A @ B^T`. The neat observation is that aiter's
`gemm_a8w8_blockscale` already handles this case without a kernel fork:
passing a per-row B scale of shape `[Q, C/128]` makes the kernel's
`GROUP_N` collapse to 1, so each B row carries its own 1×128 scale along
C. Standalone: wgrad rel-err 0.0364 symmetric vs 0.0369 asymmetric vs the
exact-dequant ref (i.e. now down to the floor of FP8 quantization noise).
Lives as a branch off `amd-fp8-training`; NOT merged into `amd-fp8-te-run`
or `amd-fp8-dsv4-smoke` yet — those still use the v1 asymmetric wgrad.

**`amd-fp8-te-run`** (60a01a9, 74e50b4, 138c96b, db05cd0, 9acabc1,
1956f7b, 4ac4f2b, 36f45fa, 5f7f6b1). Wire blockwise FP8 into TE itself.
`60a01a9` is the bring-up: lift the gate, override `Float8BlockQuantizer.
quantize / update_quantized` to route through aiter (storing uint8 data
with FP32 scales in our natural convention but still tagged
`Float8BlockScaleTensorFormat.GEMM_READY` so TE's internal asserts pass),
override `cpp_extensions.gemm.general_gemm` to route to
`aiter.gemm_a8w8_blockscale`, plus the sitecustomize MetaPathFinder
injector. `74e50b4` adds the qwen3-4B blockwise launcher. The remaining
seven commits are the iterative bring-up to the first training step:
honor `accumulate=True` in fused wgrad accumulation (`138c96b`); replace
the unimplemented HIP fused norm+quant with a bf16 norm + aiter quant
path (`db05cd0`); replace the COMPACT-required seq-parallel FP8 gather
with a bf16 gather + re-quantize (`9acabc1`), with a bespoke `_my_dequant`
that uses our scale convention rather than TE's transposed cuBLAS one
(`1956f7b`); disable Megatron's `jit_fuser` (torch.compile of BDA /
bias-swiglu fusions tripped dynamo fake-tensor tracing over our aiter
output) — keep those eager via `disable_jit_fuser()` and
`--no-bias-{dropout,swiglu}-fusion` (`1956f7b`); preserve the activation's
3D leading dims `[s, b, h]` through the GEMM (aiter returns flat 2D)
(`36f45fa`); and rebind `gather_along_first_dim` in EVERY already-imported
`transformer_engine.pytorch.*` module, since several do
`from ..distributed import gather_along_first_dim` at import time
(`5f7f6b1`). End state: qwen3-4B dense end-to-end on 8× MI355X
TP2/CP4/SP.

**`amd-fp8-dsv4-smoke`** (31 commits as of `a539171`; HEAD-most listed
first, oldest last). Take the te-run blockwise wiring and stand up
DSv4-Flash-FP8-4layer end-to-end. The first ten commits land yueming's
DSv4 plugin / mbridge / megatron-to-hf / IndexerReplayManager
(`aae3bfe`), the DSv4 launcher + `transformers_patch` (`e0b27a3`), the
pure-PyTorch ports of TileKernels-dependent ops — `qat.py` `d1e8602`
(per-1×128 E4M3 fake-quant, replaces `tile_kernels.quant.act_quant`) and
`hyper_connection.py` `a35f87b` (replaces
`tile_kernels.modeling.mhc.{mhc_pre_norm_fn, ..., sinkhorn_normalize}`
with sglang's pure-torch reference impl + a Sinkhorn loop). Then the MoE
FP8 bring-up: `a5d595b` first lands a bf16 dequant fallback for
`general_grouped_gemm`; `b7417149` adds the matching `tex.split_quantize`
bf16 fallback (without which `GroupedLinear.forward` dies before the GEMM
is called); `1f181a2` syncs both into the te_inject_site injector. Then
the synthetic rollout (`b7859680`), the weight-conversion suite —
`fp8_cast_bf16.py` for DSv4's `.scale` naming (`87ef31f`) and
load-on-CPU-dequant-tensor-by-tensor-on-GPU to avoid OOM (`d44572e`),
`convert_hf_to_torch_dist.py` `with_transformers_patch()` wrap
(`0805742`), mbridge rope-fields shim + the Pinaster→HF safetensors key
renamer (`54f9fec`), `mlp.router→mlp.gate` mapping fix (`300e175`). Then
the iteration-to-first-step launcher fixes (`3611de1`): defaults,
`StrEnum` Py3.10 polyfill, `hf_validate_args` `intermediate_size` skip
for `deepseek_v3/v4/ref`, `--use-(rollout-)indexer-replay` CLI flags,
`fake_rollout` groups list-of-list shape. Then `4ea52e9` adds the
ROCm bf16×bf16→fp32 GEMM workaround in DSv4's compressor
(`linear_bf16_fp32`); `52bfc43` adds small-weight handling for the
indexer's `linear_weights_proj` (N=64 forces sdim demotion to 1×128,
plus a fix to the rank-heuristic when both operands have sdim==1).
`0ffc83f` graces empty-tensor log paths + fills `rollout_log_probs`;
`29b2179` bumps fake response length so per-rank SP slices are non-empty.
Then the MoE FP8 upgrade: `06447f0` proto + `d99f0ee` lands real
per-expert aiter blockwise FP8 (drops the bf16 fallback). Launcher pin
(`b34cd0c`), an SP-drop / SP-restore exploration (`7b1824d` /
`e9281088`), fake_rollout prompt prefix (`ca78be3`), loss_mask shape
(`5737eff`), MoE per-expert M align→128 in `Fp8Padding`+`Fp8Unpadding`
(`5ee71a7`, `c400c72`), small-weight columnwise quant via zero-padding
M to BLK (`0ca057a`), always-produce-both-rowwise-and-columnwise so
`update_usage(columnwise=True)` after fprop passes its assertion
(`0e3ce1e`). Two terminal in-flight commits — `6e1ba62` and `a539171` —
prove the fused `fmoe_fp8_blockscale_g1u1` is a drop-in numerical
replacement for the per-expert loop (~3.3% rel-err) and demonstrate the
"identity-routing" trick needed to feed it Megatron's pre-permuted MoE
inputs. The fmoe wiring itself is not yet on the branch.

## 5. Module-Level Fix Catalog

### 5.1 Blockwise FP8 algorithm core (`miles/utils/rocm_fp8_blockwise.py`)

**What.** The standalone, dependency-light blockwise FP8 linear used both
as the substrate for the TE wiring and as a self-contained validation
target. 168 LoC.

**Why it needed changes on ROCm.** The DeepSeek 1×128/128×128 E4M3 GEMM
isn't exposed on AMD anywhere except aiter. This module reuses
`aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale` (the same kernel sglang
uses to serve DSv4 in FP8) and wraps it into a `torch.autograd.Function`
suitable for training.

| Commit | Trigger / symptom | File:lines (fn) | Before | After | Why correct | NV-alignment |
|---|---|---|---|---|---|---|
| `1d7c16d` | TE Float8BlockScaling unusable on ROCm (gate false; HIP kernels unimplemented) | `rocm_fp8_blockwise.py` (`quantize_1x128`, `quantize_128x128`, `BlockwiseFP8Linear`, `lift_te_gate`) | n/a | Adds aiter-backed quant + linear; fprop/dgrad use `gemm_a8w8_blockscale`, wgrad uses a bf16 fallback | Quantize formulas are the standard per-1×128 amax → fp32 scale → E4M3 cast; aiter GEMM consumes them in their native convention | ⚠️ wgrad in bf16, not FP8 |
| `70973c2` | Wgrad still in bf16 (asymmetric on AMD) | same file, `BlockwiseFP8Linear.backward` | `dw` via bf16 matmul | `dw = gemm(a=dY^T quant_1×128, b=X^T quant_128×128)` — all three GEMMs now FP8 | The asymmetric aiter signature requires one 1D-scaled and one 2D-scaled operand; treating X^T as a 128×128 weight is an over-quantization but lets us reuse the existing kernel | ⚠️ X-as-weight is an approximation of the symmetric DeepSeek wgrad |
| `8d66e84` (`amd-fp8-wgrad-symmetric`) | The 5.0% wgrad rel-err is dominated by X-as-128×128 over-quantization | adds `symmetric_blockscale_gemm`; rewires `backward` | asymmetric aiter call | Both operands quantized 1×128 along M; aiter called with per-row B scale `[Q, C/128]` so `GROUP_N` collapses to 1 — symmetric case without a kernel fork | Math: passing B-scale rowwise makes each B row carry its own 1×128 scale along C, exactly the DeepSeek wgrad recipe | ✅ identical to NV's symmetric block-scaled wgrad |

**Overall status:** ✅ identical to NV on `amd-fp8-wgrad-symmetric`;
⚠️ wgrad-only deviation (X-as-128×128) on `amd-fp8-training` and downstream
`amd-fp8-te-run` / `amd-fp8-dsv4-smoke`.

### 5.2 TE blockwise gate

**What.** `transformer_engine.pytorch.fp8.check_fp8_block_scaling_support`
is the single function that decides whether `Float8BlockScaling` is a
permitted recipe in the autocast.

**Why.** ROCm/TE
(`transformer_engine/pytorch/quantization.py:103-106`):

```
@functools.lru_cache(maxsize=None)
def check_fp8_block_scaling_support() -> Tuple[bool, str]:
    if IS_HIP_EXTENSION:
        return False, "FP8 block scaled gemm not yet supported for ROCm"
    ...
```

| Commit | Trigger | File:lines (fn) | Before | After | Why correct | NV-alignment |
|---|---|---|---|---|---|---|
| `1d7c16d` | `te.fp8_autocast(Float8BlockScaling(...))` raises "FP8 block scaled gemm not yet supported for ROCm" | `rocm_fp8_blockwise.py:156` (`lift_te_gate`) | gate returns `False` on HIP | `tefp8.check_fp8_block_scaling_support = lambda: (True, "")` | Lifting the gate alone would normally segfault into unimplemented HIP kernels, but every downstream codepath has been re-routed to aiter (see 5.3–5.7) | ✅ identical (NV's gate just returns True on cc≥9.0 + CUDA ≥12.9) |

### 5.3 TE quantize override

**What.** `Float8BlockQuantizer.quantize` / `.update_quantized` in
`transformer_engine/pytorch/tensor/float8_blockwise_tensor.py`. TE
normally dispatches into a C++ HIP kernel (`tex.quantize` / fused
norm+quant) that raises `Not implemented scaling mode` for the blockwise
mode. We replace the method body in Python with an aiter-backed quant.

**Why.** The HIP cast kernel does not implement DeepSeek's blockwise
modes (`NVTE_ERROR("Not implemented scaling mode: " + to_string(...))` at
`transformer_engine/common/cast/fp8/quantize_fp8.cuh:555/583/723`).

| Commit | Trigger / symptom | File:lines (fn) | Before | After | Why correct | NV-alignment |
|---|---|---|---|---|---|---|
| `60a01a9` | `tex.quantize` raises on the blockwise scaling mode | `rocm_te_blockwise.py:62` `_patch_quantizer / _build_qtensor` (and the inject_site copy) | C++ kernel | quant_1×128 (acts/grads) or quant_128×128 (weights); store data as uint8; scales in our natural aiter convention; tag tensor `Float8BlockScaleTensorFormat.GEMM_READY` so TE's format assertion passes | We own both the quant and the GEMM ends, so the scale-layout convention is symmetric — `_my_dequant` reads them back the same way. The `GEMM_READY` tag is a façade for TE; we never call TE's C++ GEMM | ⚠️ equivalent math, different storage convention |
| `a5d595b` | MoE GroupedLinear `tex.split_quantize` not patched yet | (initial bf16 fallback) | n/a | — | (superseded by `b741714`+`1f181a2`+`d99f0ee`) | — |
| `b7417149` | `tex.split_quantize` raises `Not implemented scaling mode: Invalid Scaling` before `general_grouped_gemm` is called | `rocm_te_blockwise_inject.py:387` (`_patch_grouped_gemm.split_quantize`) | C++ kernel | per-segment dispatch: `torch.split` then `q.quantize()` for each `Float8BlockQuantizer` quantizer | Same per-chunk semantics as the C++ split_quantize, but Python-side; zero-token experts handed back as empty bf16 | ✅ semantically identical |
| `1f181a2` | Edits to `miles/utils/rocm_te_blockwise.py` don't reach Ray workers (they import via the inject_site copy) | `te_inject_site/rocm_te_blockwise_inject.py:_patch_grouped_gemm` | (out of sync) | sync the bf16 fallback into the injector | Workers PYTHONPATH the inject_site dir, not `miles.utils.*` | — |
| `5ee71a7` | MoE `update_usage(columnwise=True)` after fprop asserts because columnwise_data is `None` when per-expert M is not a multiple of BLK | `_patch_fp8_padding` (`Fp8Padding` align_size→128) | TE default `align_size=16` | Forced ≥128 in `__init__` and `forward` | DeepSeek blockwise needs M%128==0 to build a 1×128 columnwise block | ⚠️ workaround equivalent (NV uses 16, but its kernels don't depend on M%128 for the columnwise rebuild) |
| `c400c72` | `Fp8Unpadding` still using align_size=16 → mismatched shapes → autograd grad-size rejected | `_patch_fp8_padding` (`Fp8Unpadding`) | Only `Fp8Padding` patched | Both classes uniformly bumped | The two compute padded_m_splits independently; must match | ⚠️ workaround equivalent |
| `0ca057a` | DSv4 indexer `linear_weights_proj` (N=64) crashes `quantize_128x128` view `[0, 128, 32, 128]` invalid; columnwise direction also has K=64<BLK | `_build_qtensor` (small-weight handling) | demote to 1×128 + skip columnwise → fail later | Demote effective_sdim to 1 on small N, zero-pad M up to BLK for columnwise quant, tag `_aiter_columnwise_M` so `_extract`/`_my_dequant` trim back | Zero-padded rows contribute 0 to any GEMM; quantization is unchanged on the real rows; the trim restores logical shape | ⚠️ workaround (NV doesn't need the demotion because it has a proper small-N kernel) |
| `0e3ce1e` | `te.pytorch.module.linear:402` calls `weightmat.update_usage(columnwise=True)` post-quant; the 1D-scaled branch asserts both copies present | `_build_qtensor` | Respected initial `columnwise_usage` flag (could leave `_columnwise_data=None`) | Always produce both rowwise and columnwise (one extra `quantize` on the transpose) | TE expects both copies for the 1D-scaled tensor; recomputing one from the other is non-trivial under our convention | ⚠️ workaround equivalent (NV's quant produces both lazily too; our slack is one extra cast per weight) |

**Status:** ✅ equivalent math, ⚠️ different storage convention (we tag
`GEMM_READY` but use natural scale layout), several small-shape
workarounds that NV would not need.

### 5.4 TE dense GEMM override

**What.** `transformer_engine.pytorch.cpp_extensions.gemm.general_gemm`
(and the module-local rebindings in `module.linear`,
`module.layernorm_linear`, `module.layernorm_mlp`, and later
`module.grouped_linear`).

**Why.** Both TE's `tex.generic_gemm` C++ entrypoint and hipBLASLt's
blockwise path (which only supports `VEC32_UE8M0` =MXFP8) refuse the
DeepSeek mode (`rocm_gemm.cu:1281-1287` only branches on
`is_delayed_tensor_scaling` and the UE8M0 case; everything else throws
"`Not implemented scaling modes: ... and ...`").

| Commit | Trigger | File:lines (fn) | Before | After | Why correct | NV-alignment |
|---|---|---|---|---|---|---|
| `60a01a9` | "Not implemented scaling modes" from rocm_gemm.cu | `rocm_te_blockwise.py:205` (`_patch_gemm.general_gemm`) | C++/hipBLASLt path | aiter `gemm_a8w8_blockscale`; pick rowwise vs columnwise stored copy per layout so the contraction dim is last on both operands; handle TN/NN/NT (fprop/dgrad/wgrad) | `aiter_gemm(x,w,xs,ws)` returns `x@w^T`; selecting rowwise vs columnwise correctly puts the contraction dim last on each side; the `swap` flag flips orientation when the operand we mapped to "weight" was actually the A side. Tested across the three layouts | ⚠️ math identical; perf likely lower than NV's cuBLAS blockwise GEMM |
| `138c96b` | Megatron's `--accumulate-allreduce-grads-in-fp32` wgrad path calls with `out=weight.main_grad` (fp32) and `accumulate=True`. Original `out.copy_(res)` overwrites main_grad every microbatch | same fn, the `out is not None` block | `out.copy_(res)` | If `accumulate`: cast and `out.add_(res)` (also honor `beta`); else copy. Compute and return `bias_grad = res.sum(token-dims)` when `grad and bias is not None` | This matches the cuBLAS `D = alpha*A@B + beta*D` semantics that Megatron's fused wgrad expects | ✅ identical |
| `36f45fa` | "output [1152,2560] doesn't match broadcast [1152,1152,2560]" — Megatron passes 3D activations `[s,b,h]` to general_gemm and expects 3D back, aiter returns flat 2D `[M,N]` | same fn, post-call reshape | flat 2D | Reshape res to the activation's leading dims when row count matches their product (skip when an `out` buffer was supplied) | The attention `o_proj` returned 2D while the residual stayed 3D, breaking the residual add. This was the real bug under the "BDA broadcast" red herring | ✅ identical |
| `52bfc43` (subset) | When both operands have `sdim==1` (small-weight demotion or wgrad both-1×128), the rank-based heuristic arbitrarily picked A — could be the weight, dropping the activation's leading dims | same fn, dispatch heuristic | `if a_sdim==1 and b_sdim==2 ... elif ... else: A is weight` | Track `act_operand` explicitly as `A if A.dim() > B.dim() else B` | The activation always has more leading dims than the weight in Megatron; rank picks the right one | ✅ correct heuristic |

**Status:** ✅ numerically identical to NV across the three layouts;
**perf is the open question** (we go through Python + 1-2 extra reshapes
per GEMM vs NV's direct cuBLAS dispatch).

### 5.5 TE fused norm+quant

**What.** `transformer_engine.pytorch.module._common.apply_normalization`
plus the module-local rebindings in `layernorm_linear` / `layernorm_mlp`.
TE normally calls a fused HIP `rmsnorm_fwd` (`tex.rmsnorm_fwd`) that
internally quantizes the output in the same kernel.

**Why.** The fused HIP norm-and-quant kernel's blockwise cast path is
unimplemented; the first real training forward died at
`cast_kernels_hip.cuh:2257 Not implemented scaling mode: Invalid Scaling`
(from `db05cd0` commit body).

| Commit | Trigger | File:lines (fn) | Before | After | Why correct | NV-alignment |
|---|---|---|---|---|---|---|
| `db05cd0` | `cast_kernels_hip.cuh:2257 Not implemented scaling mode: Invalid Scaling` | `_patch_norm.apply_normalization` (and `layernorm_linear` / `layernorm_mlp` bindings) | Fused norm+quant in one HIP kernel | If `output_quantizer` is a `Float8BlockQuantizer`: run the norm in high precision (`quantizer=None`), then `output_quantizer.quantize(out)` via our aiter path | Mirrors TE's own existing ROCm workaround for `Float8CurrentScalingQuantizer`. Math is identical at the quantized output; perf cost is one extra kernel launch and one extra read/write of the bf16 norm output | ⚠️ math equivalent; perf delta = one extra round-trip through HBM |

**Status:** ⚠️ workaround equivalent. NV does it fused; we do norm in bf16
then aiter-quant. Restoring fusion needs a HIP kernel that knows about
DeepSeek blockwise scaling, which doesn't exist yet.

### 5.6 TE sequence-parallel gather

**What.** `transformer_engine.pytorch.distributed.gather_along_first_dim`,
called by `LayerNormLinear` / `LayerNormMLP` to gather `ln_out` across the
TP group under `--sequence-parallel`. TE normally dispatches this to
`_all_gather_fp8_blockwise`, which requires the COMPACT data format.

**Why.** Our stored copies are tagged `GEMM_READY` and laid out in the
aiter-native scale convention; the COMPACT path is unimplemented and
raises:
"`All-gather with FP8 block-wise quantized tensor requires compact data
format, but found ...GEMM_READY`".

| Commit | Trigger | File:lines (fn) | Before | After | Why correct | NV-alignment |
|---|---|---|---|---|---|---|
| `9acabc1` | The "requires compact data format" error above | `_patch_gather.gather_along_first_dim` in `distributed.py` + `layernorm_linear` / `layernorm_mlp` bindings | COMPACT-only FP8 gather | If input is blockwise FP8 (or quantizer is `Float8BlockQuantizer`): dequant to bf16 with `_my_dequant`, `torch.distributed.all_gather_into_tensor`, re-quantize to GEMM_READY via the quantizer | Exactly TE's own "high-precision fallback" path, taken unconditionally for blockwise. Math is equivalent at the destination tensors | ⚠️ ~2× communication cost (bf16 vs fp8) |
| `1956f7b` | The fallback called `inp.dequantize()` (TE's), which assumes the cuBLAS-transposed `GEMM_READY` scale layout — `reshape '[1152,1152,128]' invalid for 165888 entries` | `_my_dequant` in both `rocm_te_blockwise.py` and the inject_site copy | TE's `dequantize()` | Our own dequant using natural scale shapes ([M, K/128] rowwise 1×128 or [M/128, K/128] rowwise 128×128); also handle the columnwise-only branch (TE may pre-quantize `ln_out` with only columnwise data) | We own the scale convention end-to-end; TE's dequant is wrong under that convention | ✅ correct |
| `5f7f6b1` | Backward dgrad gather in `base.py.grad_output_preprocess` hit the same compact-format error because `base.py` does `from ..distributed import gather_along_first_dim` at import time and held a stale binding | `_patch_gather` (loop over `sys.modules` for `transformer_engine.pytorch.*` and rebind any that still hold `_orig_gather`) | Explicit per-module rebinds (`layernorm_linear`, `layernorm_mlp`, …) — missed `base` | Sweep every already-imported TE module | Catches all current and future TE submodules that imported the name | ✅ |

**Status:** ⚠️ math equivalent at destination, ~2× comm cost vs NV's
in-FP8 blockwise gather. Real perf parity requires implementing the
COMPACT format conversion and routing through TE's blockwise gather.

### 5.7 TE MoE grouped GEMM

**What.** `cpp_extensions.gemm.general_grouped_gemm` and
`tex.split_quantize` for `te.GroupedLinear` (the MoE path).

**Why.** Both entrypoints raise "Not implemented scaling mode: Invalid
Scaling" on the blockwise recipe.

| Commit | Trigger | File:lines (fn) | Before | After | Why correct | NV-alignment |
|---|---|---|---|---|---|---|
| `a5d595b` | `cpp_extensions.gemm.general_grouped_gemm` fails on a list of `Float8BlockwiseQTensor` | initial `_patch_grouped_gemm` (bf16 dequant fallback) | C++ kernel | Dequantize each blockwise FP8 input via `_my_dequant`, forward the bf16 list to the stock kernel | Correct but bf16 — ~2× slower than a proper FP8 MoE GEMM. Unblocks the smoke | ⚠️ math equivalent, perf slower |
| `b7417149` | `tex.split_quantize` dies BEFORE the GEMM is reached | adds `split_quantize` override | C++ kernel | If any quantizer is `Float8BlockQuantizer`, `torch.split(inp.bf16(), m_splits)` and skip the quant; downstream bf16 GEMM path consumes it | Pairs with `a5d595b` to keep MoE functional | ⚠️ bf16 |
| `1f181a2` | Workers loaded the inject_site copy, not `miles.utils.rocm_te_blockwise` | sync edits into `te_inject_site/rocm_te_blockwise_inject.py` | (out of sync) | sync `_patch_grouped_gemm` | required by the injection mechanism | — |
| `d99f0ee` | Replace the bf16 fallback with a real FP8 path | `_patch_grouped_gemm` in the inject_site | bf16 dequant + stock kernel | `split_quantize`: per-expert `q.quantize()` → list[Float8BlockwiseQTensor]. `general_grouped_gemm`: per-expert loop calling the already-patched `general_gemm` (aiter blockwise FP8). Handles `layout=TN/NN/NT` (fprop/dgrad/wgrad), `single_output=True` writes into `out[0]` at per-expert M offsets, `single_output=False` writes per-expert `out[i]`, accumulate=True for `main_grad` fusion, zero-token experts skip. Integration test: fwd 3.7%, dx 3.9% rel err vs bf16 ref | Same aiter blockwise FP8 GEMM as the dense path, just per-expert; numerically identical to dense Linear at the per-expert level | ⚠️ math identical, perf-only delta (one launch per expert vs NV's one fused launch) |
| `5ee71a7`, `c400c72`, `0ca057a`, `0e3ce1e` | Various follow-ups so per-expert M %128==0, both rowwise+columnwise present, small-weight columnwise via zero-pad | see 5.3 (cross-listed) | — | — | — | — |

In-flight (proven viable, not yet wired into the live MoE path):

| Commit | What it proves | Status |
|---|---|---|
| `6e1ba62` | `aiter.fused_moe.fused_moe` with `QuantType.per_128x128` runs on gfx950 and matches the per-expert aiter blockscale loop to ~3.3% rel-err (FP8 noise floor) on (E=4, K=512, I=256, 4-expert tokens). Establishes the exact operand layout: `w1=[E, 2*I, K]` (gate-first concatenated, NOT interleaved), `w2=[E, K, I]`, `shuffle_weight((16,16))` applied for the asm kernel, scales `[E, ?/128, ?/128]` fp32 | Isolated tests pass |
| `a539171` | The "identity-routing" trick to feed fmoe Megatron-style pre-permuted `[T_local, K]`: build synthetic `topk_ids=[T_local, 1]`, run `aiter.moe_sorting`, call fmoe with topk=1 — yields the same outputs (~3.3% rel-err to the per-expert loop) on Megatron's input layout | Isolated test passes; live wiring TBD |

**Status:** real FP8 MoE today (per-expert aiter loop, ⚠️ perf-only delta
vs NV's fused FP8 grouped GEMM). The fmoe one-launch path is proven on
isolated tests and a custom-autograd TEGroupedMLP wiring is in flight.

### 5.8 Launcher + injection plumbing

| File | What it does | Branch |
|---|---|---|
| `scripts/run_qwen3_4b_blockwise_te.py` | qwen3-4B blockwise FP8 launcher (TP2 / CP4 / 8 GPUs, rollout-fp8 = the Qwen3-4B-FP8 [128,128] weight_block_size checkpoint, `--num-rollout 3 --rollout-batch-size 8 --n-samples-per-prompt 4 --global-batch-size 32 --num-epoch 1`, `--fp8-recipe blockwise`, no `--fp8-param-gather`, `--no-bias-{dropout,swiglu}-fusion`). Sets `ROCM_TE_BLOCKWISE_INJECT=1`, `NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1`, prepends `te_inject_site` + megatron_path to PYTHONPATH | te-run |
| `scripts/run_deepseek_v4.py` | typer launcher with `prepare-download`/`prepare-single`/`prepare-spmd`/`train`/`full-train`. TP=8/EP=8/SP single-node, `--debug-train-only`, `--rollout-function-path miles.rollout.fake_rollout.fake_generate_rollout`, `--use-rollout-logprobs`, `--sglang-enable-dp-attention` (required when sglang_dp_size>1, even though we don't use sglang). Sets the same FP8 env. `_patch_4layer_model_type` rewrites `config.json` `model_type` to `deepseek_v3`. Pins `train_script` to the worktree's `train.py` so `sys.path[0]` is the worktree (commit `b34cd0c`) | dsv4-smoke |
| `sitecustomize.py` (in `te_inject_site`) | MetaPathFinder at front of `sys.meta_path` intercepts `transformer_engine.pytorch`, wraps the loader, runs `rocm_te_blockwise_inject.apply()` post-exec. Chains the original system sitecustomize first (commit `74e50b4`). Gated on `ROCM_TE_BLOCKWISE_INJECT=1` | te-run |
| `te_inject_site/rocm_te_blockwise_inject.py` | The ACTUAL file Megatron Ray workers load (NOT `miles/utils/rocm_te_blockwise.py`). Self-contained: only imports torch / aiter / transformer_engine. Edits to `miles/utils/rocm_te_blockwise.py` do NOT reach workers (`1f181a2` makes this rule explicit) | te-run + dsv4-smoke |
| `disable_jit_fuser()` call inside `apply()` | Disables Megatron's `torch.compile` jit_fuser (which decorates BDA / bias-swiglu). Dynamo fake-tensor tracing over our aiter blockwise FP8 GEMM output raised a spurious "output [1152,2560] doesn't match broadcast (1152,1152,2560)". Disabling here BEFORE the fusion modules import keeps those regions eager. The launcher additionally passes `--no-bias-dropout-fusion --no-bias-swiglu-fusion` belt-and-braces | te-run (`1956f7b`) |
| `--no-bias-*-fusion` launcher flags | Redundant with `disable_jit_fuser()` but defensive. qwen3 has no attn/MLP bias so the fusions buy little | te-run / dsv4-smoke |

### 5.9 Weight + checkpoint conversion (DSv4)

| File | Fix | Why |
|---|---|---|
| `tools/fp8_cast_bf16.py` | `87ef31f`: handle DSv4 `.scale` naming (e.g. `layers.3.ffn.experts.0.w1.scale`) in addition to DSv3's `_scale_inv` suffix; try both keys per FP8 weight, drop both from the rewritten index. `d44572e`: load shards on CPU; for each (weight, scale) pair move only those two tensors to GPU for the triton `weight_dequant`, write the bf16 back to CPU, free + empty_cache. Previous `cuda` load path OOM'd part-way through shard 4 of the 4-layer model | DSv4 changed the scale suffix vs DSv3; OOM on single-GPU full load of even the 4-layer (~27 GB FP8) |
| `tools/convert_hf_to_torch_dist.py` | `0805742`: move `bridge.load_weights` inside `with_transformers_patch()` so `AutoConfig.from_pretrained` keeps resolving DSv4 correctly across nested config lookups | mirrors yueming's fix for the same issue |
| `tools/rename_dsv4_safetensors_to_hf.py` | New file (`54f9fec`). Pinaster mirror saves under internal short names (`layers.0.attn.wq_a.weight`, `embed.weight`, `head.weight`, …); mbridge expects standard HF DeepseekV3 layout (`model.layers.0.self_attn.wq_a.weight`, `model.embed_tokens.weight`, `lm_head.weight`, …). Renames every key + rewrites index.json in-place. Handles attn / ffn / shared_experts (`w1/w2/w3 → gate/down/up`), norms (`attn_norm → input_layernorm`, `ffn_norm → post_attention_layernorm`), `hc_*` blocks, compressor + indexer subtrees, top-level `embed.weight / norm.weight / head.weight` | Without it, `AutoBridge.load_weights` KeyError's on every expected HF key |
| `miles_plugins/mbridge/deepseekv4.py` | `54f9fec`: rope-fields shim — transformers ≥4.57 stores `rope_theta` / `rope_scaling` under `rope_parameters` dict; inject top-level attrs onto a copy of `hf_config` via a context manager around `_build_config` / `_get_gptmodel_args`. `300e175`: `_MLP_MAPPING` `mlp.router.tid2eid → mlp.gate.tid2eid` (matches the HF-canonical gate name that our renamer writes) | newer transformers; key-name canonicalization |

### 5.10 DSv4 model spec / plugin (yueming-derived)

| File | Notes | Branch |
|---|---|---|
| `miles_plugins/models/deepseek_v4/deepseek_v4.py` | Cherry-picked from yueming-yuan/miles@deepseek-v4 in `aae3bfe`. Provides `DeepSeekV4Attention`, `V4Indexer`, `DeepSeekV4Compressor`, hyper-connection wiring, tilelang sparse MLA / indexer kernel hooks, the `dsv4` spec exposed via `--spec miles_plugins.models.deepseek_v4.deepseek_v4 get_dsv4_spec`. Enables TF32 for fp32 matmul to match TileKernels MHC precision | dsv4-smoke |
| `miles_plugins/models/deepseek_v4/ops/qat.py` | `d1e8602`: pure-PyTorch port of yueming's `tile_kernels.quant.act_quant` / `per_token_cast_back`. Per-1×128-block symmetric E4M3 fake-quant: `amax → scale=amax/E4M3_MAX → x_q=round(x/scale) → x_dq=x_q*scale`. Identity-grad `autograd.Function` (straight-through). Bit-equivalent to "cast to FP8 and back" **without** UE8M0 scale-rounding — a strict over-approximation of yueming's UE8M0-rounded scales (i.e. higher precision than yueming's path) | dsv4-smoke |
| `miles_plugins/models/deepseek_v4/ops/hyper_connection.py` | `a35f87b`: pure-PyTorch port of yueming's TileKernels MHC. Re-derives math from sglang's reference torch impl (`DeepseekV4HCBase.hc_pre/hc_post` + the TileLang `hc_split_sinkhorn_kernel`). RMS norm + `F.linear` → mixes; split into `[pre | post | comb]`; Sinkhorn-normalize comb (row softmax then alternating row/col norms for `sinkhorn_iters` iterations); `layer_input = (pre * x_flat).sum(hc)`; `post_layer = post*x + comb @ residual`. Backward via autograd, no asymmetric-impl drift. Public API (`HCHeadParams`, `DeepSeekV4HyperConnectionUtil`) unchanged so yueming's `transformer_layer.py` / `transformer_block.py` callsites work unmodified | dsv4-smoke |
| `miles/utils/replay_base.py` | `aae3bfe`: adds `IndexerReplayManager` (referenced by yueming's `dsa.py` V4-mode indexer replay). The two `--use-(rollout-)indexer-replay` CLI flags hang off it | dsv4-smoke |
| `miles/utils/transformers_patch.py` | `e0b27a3`: standalone `AutoConfig.from_pretrained` patch — when `model_type ∈ {deepseek_v4, deepseek_ref}`, write a sidecar `config.json` with `model_type=deepseek_v3` into a process-unique tmp dir and load that. Mirrors sglang's `_load_deepseek_temp_model` trick. The sglang in our image doesn't know DSv4, so we need this inline | dsv4-smoke |

### 5.11 Misc framework fixes

| Fix | File:line | Why |
|---|---|---|
| `hf_validate_args` skip `intermediate_size↔ffn_hidden_size` check for `deepseek_v3/v4/ref` (`3611de1`) | `miles/utils/arguments.py:2297-2310` | DSv4 is all-MoE: `hf_config.intermediate_size` is the dense FFN width; Megatron's `--ffn-hidden-size` is `moe_ffn_hidden_size` per expert. They never match by construction |
| Add `--use-indexer-replay` / `--use-rollout-indexer-replay` CLI flags (`3611de1`) | `arguments.py:1054-1065` | Without them `actor.init()` crashes on `getattr(args, f"use_{m.name}_replay")` for the `IndexerReplayManager` registered in `aae3bfe` |
| Python 3.10 `StrEnum` polyfill (`3611de1`) | `miles/utils/chat_template_utils/tito_tokenizer.py:26-31` | `from enum import StrEnum` is 3.11+; the container's python is 3.10 |
| Empty-tensor guard in `log_rollout_data` (`0ffc83f`) | `miles/backends/training_utils/log_utils.py:145` | `split_with_sizes` crashes on empty / mismatched-length log-prob tensors. Only matters for synthetic rollouts |
| ROCm bf16×bf16→fp32 GEMM workaround (`4ea52e9`) | `miles_plugins/models/deepseek_v4/ops/kernel/precision_aligned_ops.py:25-29` (`_BFloat16LinearFP32Func`) | `torch.mm(x_bf16, w_bf16.t(), out_dtype=torch.float32)` raises "`gemm input type at::BFloat16 and output type float is not supported for ROCm`". Workaround: upcast both inputs to fp32 first (the BF16 mantissa truncation is already baked in). Hits every DSv4 layer's compressor + v4_indexer.compressor forward |
| ROCmFileSystemWriterAsync (pre-existing) | `miles/utils/rocm_checkpoint_writer.py` | HIP compat shim for `FileSystemWriterAsync.preload_tensors` |
| `_patch_4layer_model_type` (`e0b27a3` / `3611de1`) | `scripts/run_deepseek_v4.py:89-108` | Rewrite `config.json` `model_type` to `deepseek_v3` directly (rather than yueming's `deepseek_ref`, which only sglang's DSv4 fork knows). HF transformers' `DeepseekV3Config` preserves the unknown DSv4 fields (`hc_mult`, `compress_ratios`, …) as plain attributes |

### 5.12 Synthetic rollout (smoke-only)

`miles/rollout/fake_rollout.py` (added `b7859680`, refined `0ffc83f`,
`29b2179`, `ca78be3`, `5737eff`). Pulls prompts from `data_source.get_samples`
(returns `list[list[Sample]]` — one inner list per prompt-group of
`n_samples_per_prompt`), fills each sample with `max(rollout_max_response_len/2, 64)`
random tokens, a single-token BOS-ish prompt prefix if the source had no
prompt (so `logits[start-1:end-1]` doesn't slice into a negative
position), response-length-only `loss_mask` (rollout.py:702 asserts equality),
a random reward, and a zero `rollout_log_probs` (consumed by the GRPO loss
when paired with `--use-rollout-logprobs`).

This is **SMOKE-ONLY** — losses are meaningless. Real RL training would use
sglang, but the sglang in this image doesn't know DSv4 (yueming's
sgl-project/sglang@deepseek_v4 fork is required and is out of scope here).

### 5.13 Megatron-LM side (`/mnt/data/data/hai/yueming-megatron`)

Yueming's Megatron-LM fork (`deepseek-v4` branch) is **cloned and added to
PYTHONPATH**, not merged. The launcher
(`run_deepseek_v4.py:43-46, 293`) sets
`PYTHONPATH={te_inject_site}:{worktree_root}:{yueming_megatron}` and passes
`megatron_path=YUEMING_MEGATRON` to `U.execute_train`.

Two in-place edits to that clone:

1. `megatron/core/distributed/finalize_model_grads.py:302-307` — skip
   `_update_router_expert_bias` for modules where `expert_bias is None` or
   `local_tokens_per_expert is None`. DSv4 hash-mode routers (layers ≤
   `dsv4_n_hash_layers`) leave both `None` and would otherwise crash the
   stacked-tensor allreduce.
2. `megatron/core/transformer/experimental_attention_variant/dsa.py:864`
   — fix `from miles_plugins.models.deepseek_v4.ops.ref_model import
   apply_rotary_emb` → `ops.rope` (no `ref_model` file exists;
   `apply_rotary_emb` lives in `ops/rope.py`).

## 6. NV-Alignment Delta

| Module | NV path | Our path | Status |
|---|---|---|---|
| Blockwise FP8 algorithm core (`amd-fp8-wgrad-symmetric`) | NV blockwise quant + cuBLAS blockwise GEMM (symmetric wgrad) | aiter quant + `gemm_a8w8_blockscale`; wgrad via `symmetric_blockscale_gemm` (per-row B-scale collapses GROUP_N=1 → both-1×128 case) | ✅ identical math |
| Blockwise FP8 algorithm core (`amd-fp8-training`, `amd-fp8-te-run`, `amd-fp8-dsv4-smoke`) | same | wgrad uses asymmetric aiter call with X treated as 128×128 weight | ⚠️ over-quantizes one wgrad operand; ~5.0% vs 3.6% rel-err on the standalone wgrad bench |
| TE blockwise gate | `True` on cc≥9.0 + CUDA≥12.9 | Forced `True` via `lift_te_gate` | ✅ identical |
| TE quantize override | C++ HIP kernel (CUDA path: cuBLAS-transposed `GEMM_READY` layout) | Python; aiter quant; uint8 data; natural scale layout tagged `GEMM_READY` (façade); always emit both rowwise+columnwise; small-weight zero-pad on M; sdim demotion for N<128 weights | ⚠️ equivalent math, different storage convention; small-shape workarounds NV would not need |
| TE dense GEMM override | C++ cuBLAS direct dispatch | Python; aiter `gemm_a8w8_blockscale`; rowwise/columnwise pick per layout; 2D→3D reshape; fused wgrad `accumulate` + bias_grad | ✅ math identical; ⚠️ perf delta likely (Python dispatch + reshapes vs single cuBLAS launch) |
| TE fused norm+quant | Fused HIP kernel | bf16 norm (`quantizer=None`) + aiter quant in two launches | ⚠️ equivalent at the quantized output; +1 HBM round-trip per norm |
| TE seq-parallel gather | In-FP8 blockwise gather (COMPACT format) | Dequant to bf16 → `all_gather_into_tensor` → re-quantize to GEMM_READY | ⚠️ math equivalent at destination; ~2× comm cost (bf16 payload vs FP8 with separate scales) |
| TE MoE grouped GEMM fprop+dgrad+wgrad | Fused FP8 grouped GEMM (one launch over experts) | Per-expert Python loop calling the patched dense `general_gemm` (aiter blockwise FP8 per expert) | ⚠️ math identical (per-expert FP8 is the same kernel as dense); perf delta = E launches vs 1 |
| MoE alignment (`Fp8Padding` / `Fp8Unpadding` align_size) | 16 (NV default) | 128 (forced) | ⚠️ workaround for our quantizer's M%128 requirement; nominally extra padding cost in the input-permute |
| `jit_fuser` (BDA / bias-swiglu) | Enabled (torch.compile) | Disabled (`disable_jit_fuser()` + `--no-bias-*-fusion`) | ⚠️ perf only |
| MoE per-expert fprop (in-flight) | NV fused FP8 grouped GEMM | `aiter.fused_moe.fused_moe` with `QuantType.per_128x128` (validated ~3.3% rel-err vs the per-expert loop, kernel `fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256` on `gfx950`) | ⚠️ perf-only delta, math identical within FP8 noise — wiring TBD |
| ROCm bf16×bf16→fp32 GEMM (DSv4 compressor) | cuBLAS supports it natively | Upcast both inputs to fp32 first | ⚠️ equivalent math; one extra fp32 buffer per layer's compressor input |
| DSv4 hyper-connection / QAT ops | TileKernels (deepseek-ai/TileKernels) fused kernels | Pure-PyTorch impl (sglang torch reference + a Sinkhorn loop) | ⚠️ math equivalent; perf much slower (no kernel fusion) |
| Smoke rollout (`fake_rollout`) | sglang real rollout | Synthetic random tokens + random reward | ❌ SMOKE-ONLY; not a real rollout |

## 7. What We Proved vs What We Did NOT Prove

**Proved on MI355X (8× gfx950).**

- Standalone aiter blockwise FP8 linear: fwd 3.7% / dgrad 4.1% / wgrad
  5.0% (asymmetric) or 3.6% (symmetric) rel-err vs bf16 across asymmetric
  shapes, all finite.
- te.Linear, te.LayerNormLinear, te.LayerNormMLP under
  `fp8_autocast(Float8BlockScaling(E4M3))` match bf16 to ~3.7% on
  out/dgrad/wgrad. te.GroupedLinear (MoE) integration: fwd 3.7%, dx 3.9%
  rel-err.
- qwen3-4B dense FP8 blockwise end-to-end training run on TP2/CP4/SP/8GPU
  completes 3 GRPO steps; train-vs-rollout logprob diff bounded by the
  bf16 baseline (~0.028 on the prior bf16 reference).
- DSv4-Flash-FP8 4-layer prune on TP=8/EP=8/SP/8GPU: model loads from
  torch_dist, 3 GRPO steps complete with finite gradients, MoE FP8 path
  engaged (the "MoE general_grouped_gemm via aiter blockwise FP8" log
  fires).
- The fmoe one-launch path runs on gfx950 (kernel
  `fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256`) and agrees with
  the per-expert loop to ~3.3% rel-err — i.e. it's a drop-in numerical
  replacement.
- Weight conversion pipeline (FP8 HF → BF16 HF via triton dequant on CPU →
  torch_dist via mbridge AutoBridge under `transformers_patch`) for the
  4-layer DSv4-Flash-FP8 model completes (~40 s FP8→BF16, ~5 min
  BF16→torch_dist).

**NOT proved.** Convergence / accuracy over a real training run (the
fake_rollout makes the loss meaningless). Multi-node. The full 284B
DSv4-Flash. Perf parity with NV (we have not run head-to-head; several
known perf gaps above). Real-sglang rollout integration on this image
(blocked on the installed sglang not knowing DSv4).

## 8. Open Follow-ups (recommended order)

1. **Merge `amd-fp8-wgrad-symmetric` forward into `amd-fp8-dsv4-smoke`.**
   The faithful both-1×128 wgrad already exists; the current
   `amd-fp8-dsv4-smoke` inherits the v1 asymmetric wgrad from the
   `amd-fp8-training` history. This is a real numerical improvement and
   should be the next thing in.
2. **Land the `fmoe_fp8_blockscale_g1u1` fprop fusion.** Numerical
   viability is proven (commits `6e1ba62`, `a539171`); the wiring is a
   custom-autograd `TEGroupedMLP` patch (fprop = fmoe one launch; bwd =
   the existing per-expert dgrad/wgrad loop). Real perf win.
3. **Drop the seq-parallel bf16-then-requant gather.** Implement the
   COMPACT format conversion (or contribute it upstream to
   `ROCm/TransformerEngine`) and route through TE's blockwise gather.
   Removes the ~2× comm cost.
4. **Restore fused norm+quant on ROCm.** Either contribute a HIP fused
   `rmsnorm+blockwise_quant` kernel upstream, or accept the bf16-norm
   path until one ships.
5. **Replace `fake_rollout` with a brief sglang DSv4 rollout** to prove
   the loop on real data. Blocked on sglang DSv4 being installed in the
   image (yueming runs his own fork).
6. **Revisit MoE wgrad.** The per-expert loop uses the dense
   `gemm_a8w8_blockscale` path which on `amd-fp8-dsv4-smoke` still has
   the asymmetric wgrad. After (1) lands, MoE wgrad inherits the
   symmetric kernel for free; verify per-expert numerics.
7. **Eventually rebase onto a future `radixark/miles` main that has
   yueming's DSv4 PR merged, and DROP our duplicated DSv4 plugin
   files** (`miles_plugins/models/deepseek_v4/*`,
   `miles_plugins/mbridge/deepseekv4.py`,
   `miles/backends/megatron_utils/megatron_to_hf/deepseekv4.py`,
   `miles/utils/replay_base.py::IndexerReplayManager`) in favour of
   upstream. The ROCm-specific ports (pure-PyTorch `qat.py`,
   `hyper_connection.py`, ROCm bf16xbf16→fp32 GEMM workaround) need to
   survive the rebase.
8. **Push the TE-side fixes to `ROCm/TransformerEngine`** as proper C++
   PRs. The python monkeypatch is good for bring-up; the long-term home
   is upstream TE so the gate stops needing to be forced, the HIP
   `quantize` / `general_gemm` / `apply_normalization` /
   `general_grouped_gemm` / `gather_along_first_dim` route to aiter
   natively, and we drop sitecustomize entirely.

## 9. Appendix

### 9.1 Branch URLs

- `amd-fp8-training` — https://github.com/JessicaJiang-123/miles/tree/amd-fp8-training
- `amd-fp8-wgrad-symmetric` — https://github.com/JessicaJiang-123/miles/tree/amd-fp8-wgrad-symmetric
- `amd-fp8-te-run` — https://github.com/JessicaJiang-123/miles/tree/amd-fp8-te-run
- `amd-fp8-dsv4-smoke` — https://github.com/JessicaJiang-123/miles/tree/amd-fp8-dsv4-smoke
- `amd-fp8-docs` (this doc) — https://github.com/JessicaJiang-123/miles/tree/amd-fp8-docs

### 9.2 Reproduce: qwen3-4B blockwise FP8 (8× MI355X)

```
# in container miles-hai2 with the worktree mounted at /data/data/hai/miles-te:
cd /data/data/hai/miles-te
python scripts/run_qwen3_4b_blockwise_te.py \
    --mode debug_minimal --hardware MI355X --rollout-fp8 --train-fp8
```

Sets `ROCM_TE_BLOCKWISE_INJECT=1`,
`NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1`, prepends
`/data/data/hai/miles-te/miles/utils/te_inject_site` to PYTHONPATH.

### 9.3 Reproduce: DSv4-Flash-FP8 4-layer smoke (8× MI355X)

```
# in container miles-hai2 with the worktree mounted at /data/data/hai/miles-dsv4:
cd /data/data/hai/miles-dsv4
HF_TOKEN=hf_... python scripts/run_deepseek_v4.py full-train \
    --model-name DeepSeek-V4-Flash-FP8-4layer \
    --num-nodes 1 --num-gpus-per-node 8
```

Does prepare-download → prepare-single (FP8→BF16, ~40 s) → prepare-spmd
(BF16→torch_dist, ~5 min) → train (3 GRPO steps under fake_rollout).
TP=8/EP=8/SP, `--debug-train-only`, blockwise FP8 via injector.

### 9.4 Environment

- Container: `rlsys/miles:MI350-355-latest`
- TransformerEngine: 2.8.0+a365f2de (ROCm fork)
- aiter: `/sgl-workspace/aiter` (kernels: `gemm_a8w8_blockscale`,
  `fused_moe`, `moe_sorting`; HSA: `/sgl-workspace/aiter/hsa/gfx950/`)
- GPUs: 8× MI355X (gfx950)
- Python: 3.10 (requires `StrEnum` polyfill, applied in
  `miles/utils/chat_template_utils/tito_tokenizer.py:26-31`)

### 9.5 Weights / checkpoints

- 4-layer source: `Pinaster/DeepSeek-V4-Flash-FP8-4layer` (HF)
- Inside container: `/root/models/DeepSeek-V4-Flash-FP8-4layer` (raw HF),
  `/root/models/DeepSeek-V4-Flash-FP8-4layer-bf16` (FP8→BF16),
  `/root/models/DeepSeek-V4-Flash-FP8-4layer_torch_dist` (Megatron
  torch_dist, `release` tracker written)
- Full DSv4-Flash (250 GB FP8) at
  `/opt/shared/hai/models/DeepSeek-V4-Flash-FP8/` — out of scope for the
  single-node smoke

### 9.6 Pointers to the source of truth

- Blockwise FP8 algorithm core (faithful wgrad):
  `/mnt/data/data/hai/miles-wgrad/miles/utils/rocm_fp8_blockwise.py`
- TE blockwise wiring (qwen3-4B reference impl):
  `/mnt/data/data/hai/miles-te/miles/utils/rocm_te_blockwise.py`
- TE blockwise wiring (DSv4 injector — what workers actually load):
  `/mnt/data/data/hai/miles-dsv4/miles/utils/te_inject_site/rocm_te_blockwise_inject.py`
- DSv4 launcher: `/mnt/data/data/hai/miles-dsv4/scripts/run_deepseek_v4.py`
- DSv4 model spec: `/mnt/data/data/hai/miles-dsv4/miles_plugins/models/deepseek_v4/`
- DSv4 conversion tools: `/mnt/data/data/hai/miles-dsv4/tools/{fp8_cast_bf16.py,convert_hf_to_torch_dist.py,rename_dsv4_safetensors_to_hf.py}`
- Megatron-LM fork: `/mnt/data/data/hai/yueming-megatron` (paths used at runtime via PYTHONPATH; in-place edits in `finalize_model_grads.py:302-307` and `experimental_attention_variant/dsa.py:864`)
- TE reference (ROCm fork, for the gate + the rocm_gemm we cite):
  `/mnt/data/data/hai/TransformerEngine`
