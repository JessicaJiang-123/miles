#!/bin/bash
# Qwen3-4B FSDP + True On-Policy 增量测试 - AMD GPU 版
# 用法：按 PLAN_AMD_TRUE_ON_POLICY.md 逐步开启 PHASE_1/2/3/4，每步测试通过后再进行下一步
# 基线：run-qwen3-4B-amd-fsdp.sh

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

### 增量阶段开关（按 PLANT_AMD_TRUE_ON_POLICY.md 逐步开启）###
# PHASE_1: 仅环境变量 NCCL_ALGO, SGLANG_USE_AITER
# PHASE_2: SGLang deterministic + triton
# PHASE_3: --deterministic-mode
# PHASE_4: --true-on-policy-mode + sglang-rl-on-policy-target
PHASE_1=0   # 0=关 1=开；通过后设 1，然后测 PHASE_2
PHASE_2=0
PHASE_3=0
PHASE_4=0
### 快速测试：仅 2 个 rollout，便于快速验证（通过后可改回 3000）###
QUICK_TEST=1   # 1=快速测试 0=完整训练
####################

### 路径与 AMD 环境 ###
BASE_FOLDER="${BASE_FOLDER:-${MODEL_DIR:-/root}}"
DATA_BASE="${DATA_BASE:-${DATA_DIR:-/root}}"
export BASE_FOLDER
export DATA_BASE

export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES:-"1"}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"0,1,2,3"}
####################

export PYTHONBUFFERED=16
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export no_proxy="127.0.0.1,${MASTER_ADDR}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${MILES_ROOT}"

# 用独立目录，不覆盖原有 checkpoint
SAVE_DIR="${BASE_FOLDER}/Qwen3-4B_miles_fsdp_top_incr"
mkdir -p "${SAVE_DIR}"

CKPT_ARGS=(
   --hf-checkpoint ${BASE_FOLDER}/Qwen3-4B
   --ref-load ${BASE_FOLDER}/Qwen3-4B
   --load ${SAVE_DIR}/
   --save ${SAVE_DIR}/
   --save-interval 20
)

# 快速测试时减少 rollout
if [ "${QUICK_TEST}" = "1" ]; then
ROLLOUT_ARGS=(
   --prompt-data ${DATA_BASE}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --rm-type deepscaler
   --num-rollout 2
   --rollout-batch-size 2
   --n-samples-per-prompt 4
   --rollout-max-response-len 256
   --rollout-temperature 1
   --global-batch-size 16
)
else
ROLLOUT_ARGS=(
   --prompt-data ${DATA_BASE}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 256
)
fi

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime ${DATA_BASE}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

if [ -z "${WANDB_API_KEY}" ]; then
   WANDB_ARGS=()
else
   WANDB_ARGS=(
      --use-wandb
      --wandb-project miles-dev
      --wandb-group qwen3-4B-amd-fsdp-top-incr
      --wandb-key "${WANDB_API_KEY}"
   )
fi

# 基线 SGLANG（与 run-qwen3-4B-amd-fsdp.sh 一致）
# PHASE_2 开启时加入：--sglang-enable-deterministic-inference --sglang-attention-backend triton
# 若 OOM 可加：--sglang-disable-cuda-graph --sglang-mem-fraction-static 0.6
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.75
   --sglang-decode-log-interval 1000
   --sglang-chunked-prefill-size 4096
   --sglang-disable-custom-all-reduce
)
if [ "${PHASE_2}" = "1" ]; then
   SGLANG_ARGS+=(--sglang-enable-deterministic-inference --sglang-attention-backend triton)
   # 可选：若 OOM 取消下面两行注释
   # SGLANG_ARGS+=(--sglang-disable-cuda-graph --sglang-mem-fraction-static 0.6)
fi

# 基线 TRAIN_BACKEND
# PHASE_3 开启时加入：--deterministic-mode
TRAIN_BACKEND_ARGS=(
   --train-backend fsdp
   --update-weight-buffer-size 536870912
   --gradient-checkpointing
   --attn-implementation flash_attention_2
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
)
if [ "${PHASE_3}" = "1" ]; then
   TRAIN_BACKEND_ARGS+=(--deterministic-mode)
fi

# PHASE_4 开启时加入 true-on-policy 参数
TRUE_ON_POLICY_ARGS=()
if [ "${PHASE_4}" = "1" ]; then
   TRUE_ON_POLICY_ARGS=(
      --true-on-policy-mode
      --sglang-rl-on-policy-target fsdp
   )
fi

NUM_GPUS=$(echo "${HIP_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)

MISC_ARGS=(
   --colocate
   --use-fault-tolerance
   --dump-details "${BASE_FOLDER}/dump_details_qwen3-4B-amd-fsdp-top-incr"
)

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# PHASE_1 开启时加入 NCCL_ALGO 和 SGLANG_USE_AITER
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MILES_ROOT}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\""
if [ "${PHASE_1}" = "1" ]; then
   RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON},
    \"NCCL_ALGO\": \"allreduce:tree\",
    \"SGLANG_USE_AITER\": \"0\""
fi
RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON}
  }
}"

# 构建 ray job 参数
RAY_JOB_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node "${NUM_GPUS}"
   "${CKPT_ARGS[@]}"
   "${ROLLOUT_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${TRAIN_BACKEND_ARGS[@]}"
   "${GRPO_ARGS[@]}"
   "${WANDB_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${EVAL_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
)

if [ "${#TRUE_ON_POLICY_ARGS[@]}" -gt 0 ]; then
   RAY_JOB_ARGS+=("${TRUE_ON_POLICY_ARGS[@]}")
fi

RAY_JOB_ARGS+=("${MISC_ARGS[@]}")

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py "${RAY_JOB_ARGS[@]}"
