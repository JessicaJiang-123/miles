#!/bin/bash
# Qwen3-4B FSDP 训练 - AMD GPU 版
# 参考 run-qwen3-next-80B-A3B-fsdp.sh 结构 + run-qwen3-4B-fsdp.sh 参数

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

### 路径与 AMD 环境 ###
# 兼容 80B 的 BASE_FOLDER；top_amd 用 MODEL_DIR/DATA_DIR
BASE_FOLDER="${BASE_FOLDER:-${MODEL_DIR:-/root}}"
DATA_BASE="${DATA_BASE:-${DATA_DIR:-/root}}"
export BASE_FOLDER
export DATA_BASE

export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES:-"1"}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"0,1,2,3"}
####################

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export no_proxy="127.0.0.1,${MASTER_ADDR}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${MILES_ROOT}"

mkdir -p "${BASE_FOLDER}/Qwen3-4B_miles_fsdp"

# FSDP 从 HF 读架构，不需 MODEL_ARGS（参考 run-qwen3-4B-fsdp.sh）
CKPT_ARGS=(
   --hf-checkpoint ${BASE_FOLDER}/Qwen3-4B
   --ref-load ${BASE_FOLDER}/Qwen3-4B
   --load ${BASE_FOLDER}/Qwen3-4B_miles_fsdp/
   --save ${BASE_FOLDER}/Qwen3-4B_miles_fsdp/
   --save-interval 20
)

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

# 参考 run-qwen3-4B-fsdp.sh 的 WANDB 条件
if [ -z "${WANDB_API_KEY}" ]; then
   WANDB_ARGS=()
else
   WANDB_ARGS=(
      --use-wandb
      --wandb-project miles-dev
      --wandb-group qwen3-4B-amd-fsdp
      --wandb-key "${WANDB_API_KEY}"
   )
fi

# 参考 run-qwen3-4B-fsdp.sh；AMD 加 --sglang-disable-custom-all-reduce
# AMD 需指定 triton + flash_attention_2，否则 SGLang 默认 fa3 会崩溃
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.75
   --sglang-decode-log-interval 1000
   --sglang-chunked-prefill-size 4096
   --sglang-disable-custom-all-reduce
   --sglang-attention-backend triton
   --attn-implementation flash_attention_2
   --sglang-disable-cuda-graph
)

# 参考 run-qwen3-4B-fsdp.sh + 80B；AMD 用 flash_attention_2
TRAIN_BACKEND_ARGS=(
   --train-backend fsdp
   --update-weight-buffer-size 536870912
   --gradient-checkpointing
   --attn-implementation flash_attention_2
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
)

NUM_GPUS=$(echo "${HIP_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)

# 参考 80B：actor 参数在 submit 行；参考 4B-fsdp：colocate, use-fault-tolerance, dump-details
MISC_ARGS=(
   --colocate
   --use-fault-tolerance
   --dump-details "${BASE_FOLDER}/dump_details_qwen3-4B-amd-fsdp"
)

# launch ray - 单节点（4B 不需多节点，80B 才有 ssh workers）
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# 参考 80B 的 RUNTIME_ENV_JSON；AMD 无 Megatron/NCCL_NVLS/CUDA_DEVICE_MAX_CONNECTIONS
# AMD+Triton 首次推理 kernel 编译慢，需延长 health check 超时
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MILES_ROOT}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"SGLANG_HEALTH_CHECK_TIMEOUT\": \"120\",
    \"MILES_SGLANG_HEALTH_TIMEOUT\": \"120\"
  }
}"

# 参考 80B 的 argument order：actor 参数显式在前，再各 ARGS
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${NUM_GPUS}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${TRAIN_BACKEND_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
