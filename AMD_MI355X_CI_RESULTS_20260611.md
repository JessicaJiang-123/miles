# miles CI Results on AMD MI355X (gfx950) — 2026-06-11

Run of the miles per-commit CI suites on AMD Instinct MI355X, with a per-test
result and the test's registration status on the `amd-cicd` branch.

## Environment
- Hardware: 8x AMD Instinct MI355X (gfx950, ROCm 7.x)
- Image: `rocm/sgl-dev:miles-rocm720-mi35x-20260610`
- Source: `amd-cicd` branch, run inside the container.
- Image Python: 3.10.

## Temporary local patches needed to run
These were applied to the working tree only, to get past environment blockers so
the real per-test behaviour could be observed. Each has a dedicated upstream-style
branch (see below).
1. ~~**Py3.10 StrEnum guard** — `miles/utils/chat_template_utils/tito_tokenizer.py`~~
   ~~(and `miles/utils/test_utils/session_verify_agent.py`) do a bare~~
   ~~`from enum import StrEnum`, which fails on Py3.10. Guarded with~~
   ~~`try: from enum import StrEnum / except ImportError: from backports.strenum import StrEnum`.~~
   ~~Both `tito_tokenizer.py` and `session_verify_agent.py` are covered by the `amd-fix-py3.11` branch.~~ ✅ fixed in 1339
2. **Ray HIP-visibility fix** — `miles/utils/external_utils/command_utils.py`:
   on ROCm, set `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1` /
   `RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1` and pass HIP_VISIBLE_DEVICES into
   the Ray job, otherwise Ray training actors fail with `No HIP GPUs are available`.
3. **sglang triton attention backend** — on ROCm, deterministic-inference sglang
   defaults to FA3 which is not in the ROCm `sgl_kernel`; pin `--attention-backend triton`.

## Standalone fix branches
- ~~`amd-fix-py3.11` — Py3.10 StrEnum guard (covers `tito_tokenizer.py` + `session_verify_agent.py`).~~ ✅ fixed in 1339
- `amd-ray-hip-visibility` — Ray HIP visible-device handling on ROCm.
- `amd-sglang-triton-backend` — pin sglang deterministic inference to triton on ROCm.

## Error-type legend
- **A** `RuntimeError: Unable to find any suitable algorithms` — Megatron training,
  TE-flash attention path has no suitable kernel/algorithm on gfx950.
- **B** `TypeError: cannot pickle 'ReloadableProcessGroup' object` — in
  `MegatronTrainRayActor.update_weights()` (colocate + megatron-to-hf bridge).
- **C** `ValueError: rollout_routed_experts is required in rollout_data for replay`
  — routing replay needs sglang to emit per-token routed experts; not produced.
- **D** `PatchApplicationError: match text not found in source` — dumper/source-patcher
  target text not present in this ROCm/sglang version.
- **E** `ModuleNotFoundError: transformer_engine.pytorch.quantized_tensor` — ROCm TE
  wheel lacks this module (checkpoint save/load).
- **F** sglang `/v1/chat/completions` does not return `meta_info.output_token_logprobs`
  even with `logprobs=True` -> session-verify 502 -> no driver events -> timeout
  (needs a modified sglang; see `miles/rollout/session/sessions.py`).
- **G** `aiter.ops.fused_qk_rmsnorm_group_quant` / `fused_qk_rmsnorm` missing — ROCm
  `aiter` lacks DeepSeek-V4 fused QK rmsnorm ops.
- **H** `ValueError: ... 'DeepseekV32ForCausalLM' is not a registered model` —
  ROCm sglang has no DeepseekV32 model class.
- **I** `ValueError: Unsupported TP style 'mla_kv_a_proj' for Transformers backend`
  — ROCm sglang Transformers backend cannot shard GLM-5 MLA.

---

## stage-b-2-gpu  (NVIDIA suite: stage-b-2-gpu-h200 / AMD suite: stage-b-2-gpu-mi35x)

Results are with the Py3.10 StrEnum guard applied (the guard is a prerequisite to
import; without it the StrEnum-dependent tests fail at the conftest import).

| # | Test path | Ran | Result | Error / notes | AMD registration (amd-cicd) |
|---|---|---|---|---|---|
| 1 | `tests/fast-gpu/test_nvfp4_quantizer.py` | yes | FAIL | `ModuleNotFoundError: transformer_engine.pytorch.custom_recipes` (ROCm TE wheel has no `NVFP4QuantizerRef`); fails at its own TE import, independent of StrEnum | Not registered (NV-only) |
| 2 | `tests/fast-gpu/test_quantizer_ci.py` | yes | 3 failed / 5 passed | the 3 fail because `fake_int4_quant_cuda` is None on ROCm (CUDA-only kernel not built); without the StrEnum guard it fails earlier at the conftest StrEnum import | Not registered (NV-only; previously-disabled AMD reg removed) |
| 3 | `tests/fast-gpu/test_run_megatron_worker_main.py` | yes | PASS (7 passed) | works once StrEnum is guarded; without the guard it fails at the conftest StrEnum import | Registered, ENABLED (added this session) |
| 4 | `tests/fast-gpu/test_mxfp8_quantizer.py` | yes | FAIL | `ImportError: cannot import name MXFP8Quantizer from transformer_engine.pytorch` (its own TE import, independent of StrEnum) | Not registered (NV-only; disabled on CUDA) |
| 5 | `tests/fast-gpu/test_semaphore.py` | yes | 1 failed / 1 passed | `no_limit` case `assert 2 <= 1` (shared HTTP client concurrency not reset; already FIXME-disabled on CUDA, not ROCm-specific); without the guard it fails at the StrEnum import | Not registered (NV-only; disabled on CUDA) |

---

## stage-c-2-gpu-h200  (AMD suite: stage-c-2-gpu-mi35x)

| # | Test path | Ran | Result | Error | AMD registration (amd-cicd) |
|---|---|---|---|---|---|
| 1 | `long/test_qwen2.5_0.5B_gsm8k.py` | yes | FAIL (104s) | **B** `cannot pickle 'ReloadableProcessGroup'` (update_weights) | Not registered (removed this session) |
| 2 | `long/test_qwen2.5_0.5B_gsm8k_async.py` | yes | FAIL (157s) | **A** `Unable to find any suitable algorithms` (train, TE-flash); the surface `503` is a downstream sglang symptom | Not registered (removed this session) |

---

## stage-c-4-gpu-h200  (AMD suite: stage-c-4-gpu-mi35x)

| # | Test path | Ran | Result | Error / notes | AMD registration (amd-cicd) |
|---|---|---|---|---|---|
| 1 | `precision/test_hf_attention_cp_relayout.py` | yes | **PASS** (23s) | only needs the StrEnum guard to import | Registered, ENABLED |
| 2 | `precision/test_qwen3_5_cp_correctness.py` | yes | **PASS** (101s) | same | Registered, ENABLED |
| 3 | `lora/test_lora_qwen2.5_0.5B.py` | yes | **PASS** (375s) | before/after ray fix: FAIL (`No HIP GPUs are available`) -> PASS | Registered, ENABLED |
| 4 | `megatron/test_mimo_7B_mtp_only_grad.py` | yes | FAIL (227s) | **A** `Unable to find any suitable algorithms` (with ray fix; flash-attn + EAGLE/MTP) | Not registered (removed this session) |
| 5 | `sglang/test_chat_input_ids_equivalence.py` | yes | **PASS** (27.65s) | fa3->triton: default fa3 FAILs (`Can not import FA3 in sgl_kernel`) -> `--attention-backend triton` PASS (also needs Qwen3-0.6B weights pre-staged) | Registered, ENABLED |
| 6 | `megatron/test_qwen3_30B_A3B/test_baseline.py` | yes | **PASS** (rc=0, 874s) | full 30B GRPO (rollout -> train step 2/3 -> update_weights 6.85s -> CPU offload), Job succeeded. First Megatron 30B training test passing on MI355X. Uses flash + colocate + bridge yet did NOT hit Error A/B -> A/B are config/model specific | Registered, ENABLED |
| 7 | `megatron/test_qwen3_30B_A3B/test_r3_baseline.py` | yes | FAIL (rc=1, 251s) | **C** `rollout_routed_experts is required in rollout_data for replay` (`--use-rollout-routing-replay` + alltoall dispatcher) | Not registered (removed this session) |
| 8 | `megatron/test_qwen3_30B_A3B/test_int4_rollout.py` | yes | FAIL (rc=1) | `AttributeError: 'NoneType' object has no attribute 'fake_int4_quant_cuda'` (`convert_hf_to_int4_direct.py:122`, INT4 conversion). Matches the documented disable reason. | Not registered (removed this session; was registered+disabled "INT4 quant kernel is CUDA-only") |
| 9 | `sglang/test_r3_router_equivalence.py` | yes | FAIL | (1) fa3 crash: ROCm has no FA3 kernel, chosen by default -> fixed by switching to triton; (2) numerical non-determinism: deterministic inference not bit-identical on gfx950 -> miles/sgl paths differ; main cause aten::linear not covered (falls to hipBLASLt) is fixed, one attention non-determinism remains; (3) functional gap: native Rust gateway (smg) drops `return_routed_experts` flag -> sgl side routed_experts=None -> fixed by switching to openai-protocol + rebuilding smg | Not registered (removed this session) |
| 10 | `sglang/.../test_glm47.py` | yes | FAIL | Error **F** (same session-verify logprobs issue) | Not registered (removed this session) |
| 11 | `sglang/.../test_minimax_m27.py` | no | — | MiniMax-M2 (~230 GB), too large to download/run | Not registered (removed this session) |
| 12 | `sglang/.../test_nemotron3.py` | yes | FAIL | `modelopt_fp8` rejected by the ROCm guard | Not registered (removed this session; was registered+disabled "modelopt_fp8 ... ROCm guard") |
| 13 | `sglang/.../test_qwen3.py` | yes | FAIL (timeout 2400s) | **F** ROCm sglang `/v1/chat/completions` does not return `meta_info.output_token_logprobs` (logprobs=True set) -> session server 502 -> no driver events -> timeout. FP8 model load + sglang serve both OK. (also needed StrEnum guard in `session_verify_agent.py`) | Not registered (removed this session) |
| 14 | `sglang/.../test_qwen35.py` | yes | FAIL | **F** (reproduced, identical to 13) | Not registered (removed this session) |

---

## stage-c-8-gpu-h100  (AMD suite: stage-c-8-gpu-mi35x)

| # | Test path | Ran | Result | Error / notes | AMD registration (amd-cicd) |
|---|---|---|---|---|---|
| 1 | `short/test_qwen2.5_0.5B_gsm8k_async_short.py` | yes | FAIL (112s) | **A** `Unable to find any suitable algorithms` (train, TE-flash) | Not registered (removed this session) |
| 2 | `sglang_config/test_sglang_config_mixed_offload.py` | yes | FAIL (177s) | **B** `cannot pickle 'ReloadableProcessGroup'` | Not registered (removed this session) |
| 3 | `short/test_qwen2.5_0.5B_gsm8k_short.py` | yes | FAIL (137s) | **B** | Not registered (removed this session) |
| 4 | `sglang_config/test_sglang_config.py` | yes | FAIL (137s) | **B** | Not registered (removed this session) |
| 5 | `sglang_config/test_sglang_config_mixed_offload_ft.py` | yes | FAIL (182s) | **B** | Not registered (removed this session) |
| 6 | `ckpt/test_qwen3_4B_ckpt.py` | yes | FAIL (588s) | **E** `ModuleNotFoundError: transformer_engine.pytorch.quantized_tensor` (ckpt save/load) | Not registered (NV-only) |
| 7 | `megatron/test_deepseek_v32_5layer_fp8.py` | yes | FAIL | **H** `'DeepseekV32ForCausalLM' is not a registered model` (ROCm sglang); plus ignored `fused_qk_rmsnorm` aiter import errors. Model: Pinaster/DeepSeek-V3.2-5layer (28.6 GB) | Not registered (NV-only) |
| 8 | `megatron/test_deepseek_v4_flash_4layer_ci.py` | yes | FAIL (rc=1, 2441s) | **G** `aiter.ops.fused_qk_rmsnorm_group_quant` / `fused_qk_rmsnorm` missing. Model: Pinaster/DeepSeek-V4-Flash-FP8-4layer (28.6 GB) | Not registered (NV-only) |
| 9 | `megatron/test_glm47_flash/test_r3_mtp.py` | no | — | GLM-4.7-Flash (62.5 GB full) too large | Not registered (removed this session) |
| 10 | `megatron/test_glm5_744b_a40b_4layer_ci.py` | yes | FAIL | **I** `Unsupported TP style 'mla_kv_a_proj' for Transformers backend` (ROCm sglang can't shard GLM-5 MLA). Model: Pinaster/GLM-5_4layer (26 GB, NOT the full 744B) | Not registered (removed this session) |
| 11 | `megatron/test_glm5_744b_a40b_4layer_r3.py` | yes | FAIL | **I** (same; crashes in sglang MLA before routing replay) | Not registered (NV-only) |
| 12 | `megatron/test_qwen3_30B_A3B/test_deepep_fp8.py` | yes | FAIL | Deep-EP is WIP on ROCm | Not registered (removed this session; was registered+disabled "Deep-EP is WIP on ROCm") |
| 13 | `megatron/test_qwen3_30B_A3B/test_deepep_fp8_bridge.py` | yes | FAIL | same (Deep-EP WIP on ROCm) | Not registered (removed this session; was registered+disabled) |
| 14 | `megatron/test_qwen3_30B_A3B/test_disagg_broadcast.py` | yes | **PASS** (rc=0, 620s) | cached Qwen3-30B-A3B + torch_dist; colocate=False / use_bridge=False; rollout (reward=1) -> train step 0/1 -> update_weights 11.5s -> Job succeeded. 2nd Megatron 30B training test passing on MI355X (no Error B -> B is colocate+bridge specific) | **Registered, ENABLED (added this session)** |
| 15 | `short/test_dumper.py` | yes | FAIL (rc=1, 127s) | **D** `PatchApplicationError: match text not found in source` (sglang source-patcher version mismatch). Model: full Qwen3-30B-A3B | Not registered (NV-only) |
| 16 | `short/test_run_megatron.py` | yes | FAIL (rc=1, 83s) | **D** same source-patcher mismatch. Model: fzyzcjy/Qwen3-30B-A3B-5layer (separate 7 GB repo) | Not registered (removed this session) |
