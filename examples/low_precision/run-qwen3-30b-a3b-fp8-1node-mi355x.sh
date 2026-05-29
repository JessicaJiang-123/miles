#!/bin/bash
# Single-node (8x MI355X / gfx950) adaptation of run-qwen3-30b-a3b-fp8-two-nodes.sh.
# Changes vs the 2-node NV script (all because MI355X has 288GiB/GPU, so one 8-GPU
# node has more total HBM than the original 2-node H200 target):
#   - 2 nodes -> 1 node (--actor-num-nodes 1, local ray head with --num-gpus 8)
#   - PP4/EP4 (16 GPUs) -> PP1/EP8 (8 GPUs); drop --sequence-parallel (needs TP>1)
#   - drop DeepEP (not present on this ROCm build, and unneeded intra-node):
#       --moe-enable-deepep removed; dispatcher flex -> alltoall
#   - paths moved off the 99%-full overlay (/root) onto /data (3.1T free)
#   - smoke sizes (small num-rollout / response len) for fast debug iteration
# Keeps the real FP8 training recipe: --fp8-recipe blockwise (needs the ROCm TE->aiter
# enablement; vanilla ROCm/TE gates blockwise off).

pkill -9 sglang; sleep 3
ray stop --force; pkill -9 ray; pkill -9 python; sleep 3
pkill -9 ray; pkill -9 python

set -ex
export PYTHONBUFFERED=16
# ROCm: aiter fused_moe blockscale MoE kernel faults (GPU memory access fault) on MI355X
# under colocate memory-saver pause/resume; use the Triton MoE path instead (matches the
# validated AMD path, cf. miles PR #727). aiter all-reduce/etc. unaffected by rollout.
export SGLANG_USE_AITER=0

# ROCm: no NVLink; HAS_NVLINK gates NCCL_NVLS which is NV-only.
HAS_NVLINK=0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-30B-A3B.sh"

MODEL_DIR="/data/data/hai/models"
DATA_DIR="/data/data/hai/datasets"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3-30B-A3B-FP8/"
   --ref-load "${MODEL_DIR}/Qwen3-30B-A3B_torch_dist/"
   --load "${MODEL_DIR}/Qwen3-30B-A3B_miles/"
   --save "${MODEL_DIR}/Qwen3-30B-A3B_miles/"
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data "${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 2
   --rollout-batch-size 8
   --n-samples-per-prompt 4
   --rollout-max-response-len 1024
   --rollout-temperature 1
   --global-batch-size 32
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime "${DATA_DIR}/aime-2024/aime-2024.jsonl"
   --n-samples-per-eval-prompt 16
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

   # single-node MoE all-to-all (no DeepEP on this ROCm build)
   --moe-token-dispatcher-type alltoall

   # fp8 (real blockwise training recipe)
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

WANDB_ARGS=()

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.6
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-expert-parallel-size 8
   --use-miles-router
   # ROCm: memory-saver resume corrupts fp8 weights without CPU backup -> NaN logits -> sampler hang
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
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NVTE_FP8_BLOCK_SCALING_FP32_SCALES\": \"1\",
    \"NCCL_TIMEOUT_MS\":\"36000000\"
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
