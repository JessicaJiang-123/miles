#!/bin/bash
# top_amd 4B 副本：修复 PYTHONPATH 以兼容 /app 容器布局
# 一键跑通方案，由 run.py 调用（自动下载模型/数据后启动）

# bash top_amd/run-qwen3-4B-amd.sh


####clear before training
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python


set -euxo pipefail


### AMD Support ###
MILES_DIR="${MILES_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export MILES_DIR

MODEL_DIR="${MODEL_DIR:-/data/cache/huggingface/models}"
export MODEL_DIR

DATA_DIR="${DATA_DIR:-/data/cache/huggingface/datasets}"
export DATA_DIR

# For AMD GPU（4 卡，可改 0,1,2,3 或 4,5,6,7）
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES:-"1"}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"0,1,2,3"}
####################


export PYTHONBUFFERED=16

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/scripts"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"

# 首次训练：Qwen3-4B_miles/ 为空时 load 自动用 ref_load(torch_dist)；续训：从 miles 加载
CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-4B
   --ref-load ${MODEL_DIR}/Qwen3-4B_torch_dist
   --load ${MODEL_DIR}/Qwen3-4B_miles/
   --save ${MODEL_DIR}/Qwen3-4B_miles/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
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

WANDB_ARGS=(
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
   # (AMD) 避免 AITER custom all-reduce 与 torch_memory_saver offload 冲突
   --sglang-disable-custom-all-reduce
)
####################


MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

NUM_GPUS=$(echo ${HIP_VISIBLE_DEVICES} | tr ',' '\n' | wc -l)
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


# 动态检测 Megatron-LM 路径（兼容 /app 或 /workspace 等不同容器布局）
MEGATRON_LM_PATH=$(python3 -c "import megatron; import os; print(os.path.dirname(os.path.dirname(megatron.__file__)))" 2>/dev/null || echo "/app/Megatron-LM")
echo "MEGATRON_LM_PATH=${MEGATRON_LM_PATH}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{
     \"env_vars\": {
        \"PYTHONPATH\": \"${MEGATRON_LM_PATH}/\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
     }
   }" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}


####clear after training
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
