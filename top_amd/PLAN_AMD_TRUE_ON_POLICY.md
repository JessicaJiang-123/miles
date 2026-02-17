# AMD True On-Policy 分步实现计划

> 目标：在 **run-qwen3-4B-amd-fsdp.sh**（已验证可跑通）的基础上，**逐步** 加入 True On-Policy 相关改动，每步单独测试，快速定位问题。

## 原则
- **基线**：始终以 `run-qwen3-4B-amd-fsdp.sh` 为参考
- **单步修改**：一次只加一个改动，跑通后再加下一个
- **快速验证**：可用 `--num-rollout 2` 等缩短测试时间

---

## 阶段 0：确认基线可跑（必做）

```bash
cd /mnt/data/yuzhen1/top/miles && bash scripts/run-qwen3-4B-amd-fsdp.sh
```

若能完整跑完至少 1 个 rollout + 1 个 train step，再进行后续步骤。

---

## 阶段 1：仅加环境变量（最小改动）

**目的**：先验证 `SGLANG_USE_AITER=0`、`NCCL_ALGO=allreduce:tree` 对现有脚本有无影响。

**修改**：只改 `RUNTIME_ENV_JSON`，在 `run-qwen3-4B-amd-fsdp.sh` 的 `RUNTIME_ENV_JSON` 中加入：

```json
"NCCL_ALGO": "allreduce:tree",
"SGLANG_USE_AITER": "0"
```

**验证**：用修改后的脚本跑一次，看是否仍能正常完成 rollout + train。

- 若通过 → 进入阶段 2  
- 若不通过 → 问题在环境变量，需逐个排查（先去掉 `SGLANG_USE_AITER`，再单独测 `NCCL_ALGO`）

---

## 阶段 2：加入 SGLang 的 deterministic inference（不改 train）

**目的**：验证 SGLang 侧 `--enable-deterministic-inference` + `--attention-backend triton` 是否能稳定跑通。

**修改**：在 `run-qwen3-4B-amd-fsdp.sh` 中：

1. 在 `SGLANG_ARGS` 中加入：
   - `--sglang-enable-deterministic-inference`
   - `--sglang-attention-backend triton`
   - （可选，若 OOM）`--sglang-disable-cuda-graph`、`--sglang-mem-fraction-static 0.6`

2. 不改 `TRAIN_BACKEND_ARGS`，不改 `--deterministic-mode`、`--true-on-policy-mode`。

**验证**：跑通 rollout + train。

- 若通过 → 进入阶段 3  
- 若不通过 → 问题在 SGLang 配置，可尝试：
  - 仅加 `--sglang-enable-deterministic-inference`，不加 triton
  - 或先加 triton，不加 deterministic

---

## 阶段 3：加入 deterministic-mode（训练侧）

**目的**：验证 `torch.use_deterministic_algorithms` 等对 FSDP 训练的影响。

**修改**：在 `TRAIN_BACKEND_ARGS` 或单独加一行：

```
--deterministic-mode
```

**验证**：跑通 train step。若报 `deterministic algorithm` 相关错误，说明部分 op 在 ROCm 下不支持 deterministic，需单独查文档或降级到不启用。

---

## 阶段 4：加入 true-on-policy-mode 相关参数

**目的**：启用 `--true-on-policy-mode` 和 `--sglang-rl-on-policy-target fsdp`。

**修改**：加入：

```
--true-on-policy-mode
--sglang-rl-on-policy-target fsdp
```

**验证**：跑通 1 个 rollout + 1 个 train step，并在 wandb 中查看 `train/train_rollout_logprob_abs_diff` 是否接近 0。

---

## 阶段 5：可选优化（按需）

若阶段 4 已稳定：

- `--sglang-dtype bfloat16`
- `--sglang-disable-cuda-graph`（若之前因 deterministic 已关，可保持）
- `--sglang-mem-fraction-static` 调优（0.6 ~ 0.75）
- AMD 一般不需要 `NVTE_ALLOW_NONDETERMINISTIC_ALGO`、`CUBLAS_WORKSPACE_CONFIG`（NVIDIA 专用），可先不设

---

## 建议的“最小 True On-Policy”测试脚本

为避免 `run-qwen3-4B-amd-fsdp-true-on-policy.sh` 被改乱，建议新建 `run-qwen3-4B-amd-fsdp-true-on-policy-incremental.sh`：

- 复制 `run-qwen3-4B-amd-fsdp.sh` 为基础
- 另用目录保存（如 `Qwen3-4B_miles_fsdp_top_test`），避免覆盖原有 checkpoint
- 按上述阶段 1 → 2 → 3 → 4 逐步加入参数，每步成功后再继续

---

## 快速测试参数（可选）

为加快迭代，可在脚本中临时修改：

```bash
# 只跑 2 个 rollout
--num-rollout 2

# 或
--rollout-batch-size 2
--n-samples-per-prompt 2
--global-batch-size 8
```

等确认流程无误后，再恢复为完整训练参数。

---

## 常见错误与排查

| 现象 | 可能原因 | 建议 |
|------|----------|------|
| SGLang 启动失败 / OOM | `mem-fraction-static` 过高或 AITER 相关 | 先 `SGLANG_USE_AITER=0`，再调低 `mem-fraction-static` |
| `deterministic algorithm` 报错 | ROCm 下部分 op 不支持 | 先去掉 `--deterministic-mode`，或查 ROCm 文档 |
| `train_rollout_logprob_abs_diff` 不为 0 | attention/GEMM 未对齐 | 确认 `sglang-attention-backend triton` 与 `attn-implementation flash_attention_2` 一致 |
| NCCL 通信错误 | `NCCL_ALGO` 或多机配置 | 单机可先不设 `NCCL_ALGO`，或试 `allreduce:tree` |
