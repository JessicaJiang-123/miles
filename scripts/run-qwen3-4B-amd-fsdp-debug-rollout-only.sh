#!/bin/bash
# Qwen3-4B AMD - debug-rollout-only 模式
# 只起 SGLang + Router，不训练。SGLang 已配置确定性推理（triton + enable-deterministic-inference）。
#
# 用法：bash scripts/run-qwen3-4B-amd-fsdp-debug-rollout-only.sh
# 可选：MODEL_DIR=/root DATA_DIR=/root RUN_DETERMINISM_TEST=1 DETERMINISM_TEST_PROMPT="1+1等于几" bash scripts/...
#
# 环境变量：
#   RUN_DETERMINISM_TEST=1        Router 就绪后自动跑 test_miles_sglang_determinism.py（默认 1）
#   DETERMINISM_TEST_PROMPT       测试用的 prompt，默认 "给我介绍下sglang"
#   DETERMINISM_TEST_TEMP         测试 temperature，默认 0
#   DETERMINISM_TEST_TRIALS       单条请求次数，默认 15
#
# 原理见 scripts/DEBUG_ROLLOUT_ONLY_README.md
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
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"0"}
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

# 跑较多 rollout；batch 缩小加速单次
DEBUG_ROLLOUT_NUM=${DEBUG_ROLLOUT_NUM:-50}
ROLLOUT_ARGS=(
   --prompt-data ${DATA_BASE}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --rm-type deepscaler
   --num-rollout ${DEBUG_ROLLOUT_NUM}
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

NUM_GPUS=$(echo "${HIP_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)

# SGLang 配置：加入确定性推理（参考你单独起 SGLang 成功的参数）
#   --attention-backend triton --enable-deterministic-inference --mem-fraction-static 0.7 --disable-radix-cache
# 单 worker 模式：所有 GPU 给一个 engine，避免多 worker 各自 RNG 导致非确定性（参见 DEBUG_ROLLOUT_ONLY_README）
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${NUM_GPUS}"
   --sglang-attention-backend triton
   --sglang-enable-deterministic-inference
   --sglang-mem-fraction-static 0.7
   --sglang-disable-radix-cache
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

ROUTER_PORT=30000
MISC_ARGS=(
   --seed 42
   --rollout-seed 12345
   --colocate
   --use-fault-tolerance
   --dump-details "${BASE_FOLDER}/dump_details_amd_debug_rollout"
   --debug-rollout-only
   --sglang-router-port ${ROUTER_PORT}
)

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
echo ""
echo "=============================================="
echo "  提交 Ray 任务，等待 Router (port ${ROUTER_PORT}) 就绪..."
echo "=============================================="

# AMD 确定性推理：关闭 AITER（与 run-qwen3-4B-amd-fsdp-true-on-policy.sh 一致）
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MILES_ROOT}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"SGLANG_USE_AITER\": \"0\"
  }
}"

# 后台提交 Ray 任务
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
   "${MISC_ARGS[@]}" &
RAY_JOB_PID=$!

# 轮询 Router 端口，就绪后可选运行确定性测试
DETERMINISM_HOST="${DETERMINISM_TEST_HOST:-172.17.0.2}"
DETERMINISM_PROMPT="${DETERMINISM_TEST_PROMPT:-给我介绍下sglang}"
DETERMINISM_TEMP="${DETERMINISM_TEST_TEMP:-0}"
DETERMINISM_TRIALS="${DETERMINISM_TEST_TRIALS:-15}"
RUN_DETERMINISM_TEST="${RUN_DETERMINISM_TEST:-1}"

router_ready=0
for i in $(seq 1 90); do
   if python3 -c "import socket; socket.create_connection(('${DETERMINISM_HOST}',${ROUTER_PORT}), timeout=2)" 2>/dev/null; then
      echo ""
      echo ">>> Router port open at ${DETERMINISM_HOST}:${ROUTER_PORT}, waiting for SGLang worker..."
      router_ready=1
      break
   fi
   sleep 2
   printf "."
done

# 等待 Worker 就绪：Router 端口开放时 SGLang 引擎可能仍在加载模型，需轮询 /generate 直到 200
if [[ ${router_ready} -eq 1 ]] && [[ "${RUN_DETERMINISM_TEST}" == "1" ]]; then
   for j in $(seq 1 36); do
      if python3 -c "
import requests
try:
    r = requests.post('http://${DETERMINISM_HOST}:${ROUTER_PORT}/generate',
        json={'text': 'x', 'sampling_params': {'max_new_tokens': 1}},
        timeout=15)
    exit(0 if r.status_code == 200 else 1)
except Exception:
    exit(1)
" 2>/dev/null; then
         echo "Worker ready."
         break
      fi
      sleep 5
      printf "w"
      if [[ $j -eq 36 ]]; then
         echo ""
         echo ">>> 超时：Worker 未在 180 秒内就绪，跳过确定性测试"
         router_ready=0
      fi
   done
fi

if [[ ${router_ready} -eq 1 ]] && [[ "${RUN_DETERMINISM_TEST}" == "1" ]]; then
   echo ""
   echo ">>> 运行确定性测试 (host=${DETERMINISM_HOST}, port=${ROUTER_PORT}, prompt=\"${DETERMINISM_PROMPT}\", temp=${DETERMINISM_TEMP})"
   python3 "${MILES_ROOT}/scripts/test_miles_sglang_determinism.py" \
      --host "${DETERMINISM_HOST}" --port "${ROUTER_PORT}" \
      --test-mode single --prompt "${DETERMINISM_PROMPT}" \
      --temperature "${DETERMINISM_TEMP}" --n-trials "${DETERMINISM_TRIALS}" --quiet
   echo ""
fi

wait ${RAY_JOB_PID} 2>/dev/null || true
