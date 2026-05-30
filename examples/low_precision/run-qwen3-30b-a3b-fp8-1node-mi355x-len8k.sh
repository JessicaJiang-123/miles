#!/bin/bash
# Test-D variant of run-qwen3-30b-a3b-fp8-1node-mi355x.sh: real (non-smoke) rollout
# sizes so the RL signal is non-degenerate, used to validate that gradients actually
# flow through the ROCm blockwise-FP8 MoE training path end-to-end (the smoke script
# truncated every response at 1024 tokens -> reward 0 -> zero advantage -> grad_norm 0).
# Changes vs the smoke script:
#   - --rollout-max-response-len 1024 -> 4096 (Qwen3-30B-A3B thinking CoT can finish,
#     so some answers are correct and rewards vary -> non-zero GRPO advantage)
#   - --num-rollout 2 -> 15 (observe grad_norm>0 + multi-step stability + reward trend)
#   - wandb enabled (key read from $WANDB_API_KEY, never committed)
# Everything else (FP8 blockwise recipe, TE->aiter injector, EP8, cpu-backup) identical.

pkill -9 sglang; sleep 3
ray stop --force; pkill -9 ray; pkill -9 python; sleep 3
pkill -9 ray; pkill -9 python

set -ex
export PYTHONBUFFERED=16
export SGLANG_USE_AITER=0

HAS_NVLINK=0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-30B-A3B.sh"

MODEL_DIR="/data/data/hai/models"
DATA_DIR="/data/data/hai/datasets"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3-30B-A3B-FP8/"
   --ref-load "${MODEL_DIR}/Qwen3-30B-A3B_torch_dist/"
   --load "${MODEL_DIR}/Qwen3-30B-A3B_miles_D/"
   --save "${MODEL_DIR}/Qwen3-30B-A3B_miles_D/"
   --save-interval 100000
)

ROLLOUT_ARGS=(
   --prompt-data "${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 1000
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 128
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 100
   --eval-prompt-data aime "${DATA_DIR}/aime-2024/aime-2024.jsonl"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 16384
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480

   --moe-token-dispatcher-type alltoall

   --transformer-impl transformer_engine
   --bf16
   --fp8-format e4m3
   --fp8-recipe blockwise
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project miles-rocm-fp8
   --wandb-group qwen3-30b-a3b-fp8-ref8k
   --wandb-key ${WANDB_API_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.6
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-expert-parallel-size 8
   --use-miles-router
   --sglang-enable-weights-cpu-backup
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
mkdir -p /data/ray_len8k
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --temp-dir /data/ray_len8k

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:/data/data/hai/miles/miles/utils/te_inject_site\",
    \"ROCM_TE_BLOCKWISE_INJECT\": \"1\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NVTE_FP8_BLOCK_SCALING_FP32_SCALES\": \"1\",
    \"NCCL_TIMEOUT_MS\":\"36000000\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
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
