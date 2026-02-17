#!/bin/bash
# Qwen3-4B AMD - debug-rollout-only 模式
# 只起 SGLang + Router，不训练。用于验证 Miles 起的 SGLang 是否确定性推理。
# 用法：先跑此脚本，在另一终端运行 scripts/test_miles_sglang_determinism.py 打请求测试。
# 参考：run-qwen3-4B-amd-fsdp.sh（基线可跑）

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

# debug-rollout-only 不需 checkpoint，但仍需 hf-checkpoint 加载模型
CKPT_ARGS=(
   --hf-checkpoint ${BASE_FOLDER}/Qwen3-4B
   --ref-load ${BASE_FOLDER}/Qwen3-4B
   --load ${BASE_FOLDER}/Qwen3-4B
   --save ${BASE_FOLDER}/Qwen3-4B
)

# 简化：只跑 2 个 rollout，便于快速验证；batch 缩小
ROLLOUT_ARGS=(
   --prompt-data ${DATA_BASE}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --rm-type deepscaler
   --num-rollout 2
   --rollout-batch-size 4
   --n-samples-per-prompt 4
   --rollout-max-response-len 256
   --rollout-temperature 1
   --global-batch-size 16
)

PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu 2048
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

# 不启用 wandb
WANDB_ARGS=()

# SGLang 配置：与 run-qwen3-4B-amd-fsdp.sh 一致，暂不加入确定性推理
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.75
   --sglang-decode-log-interval 1000
   --sglang-chunked-prefill-size 4096
   --sglang-disable-custom-all-reduce
)

# debug-rollout-only 不跑 FSDP，仅占位
TRAIN_BACKEND_ARGS=(
   --train-backend fsdp
   --update-weight-buffer-size 536870912
   --gradient-checkpointing
   --attn-implementation flash_attention_2
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
)

NUM_GPUS=$(echo "${HIP_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)

MISC_ARGS=(
   --colocate
   --use-fault-tolerance
   --dump-details "${BASE_FOLDER}/dump_details_amd_debug_rollout"
   --debug-rollout-only
   --sglang-router-port 30000
)

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MILES_ROOT}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\"
  }
}"

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
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
