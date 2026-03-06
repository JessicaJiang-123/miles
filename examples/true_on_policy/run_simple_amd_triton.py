import os
import sys
from pathlib import Path


# Ensure this script always imports the in-repo miles package under /app.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import miles.utils.external_utils.command_utils as U

MODEL_NAME = os.environ.get("MILES_SCRIPT_MODEL_NAME", "Qwen3-0.6B")
assert MODEL_NAME in {"Qwen3-0.6B", "Qwen3-4B"}

MODE = os.environ.get("MILES_SCRIPT_MODE", "normal")
assert MODE in {"normal", "debug_minimal", "debug_one_sample"}

NUM_GPUS = int(os.environ.get("MILES_SCRIPT_NUM_GPUS", "1"))

def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        # Shuffle disabled in debug modes to simplify dump pairing
        f"{'--rollout-shuffle ' if MODE == 'normal' else ''}"
        "--rm-type math "
        f"--num-rollout {2 if MODE == 'debug_one_sample' else 3000} "
        f"--rollout-batch-size {1 if MODE == 'debug_one_sample' else 32} "
        f"--n-samples-per-prompt {1 if MODE == 'debug_one_sample' else 8} "
        f"--rollout-max-response-len {2 if MODE == 'debug_one_sample' else 1024} "
        "--rollout-temperature 1 "
        "--rollout-top-p 1.0 "
        "--rollout-top-k 1 "
        # temp remove this to make test easier
        # "--over-sampling-batch-size 64 "
        # "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        f"--global-batch-size {1 if MODE == 'debug_one_sample' else 256} "
    )

    eval_args = ""
    if MODE == "normal":
        eval_args = (
            "--eval-interval 20 "
            "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
            "--n-samples-per-eval-prompt 1 "
            "--eval-max-response-len 1024 "
            "--eval-top-k 1 "
        )

    grpo_args = (
        "--advantage-estimator grpo "
        # "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    _is_debug = MODE != "normal"
    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-decode-log-interval 1000 "
        "--sglang-enable-metrics "
        f"--sglang-mem-fraction-static {0.2 if MODEL_NAME == 'Qwen3-4B' else 0.4} "
        # Debug: always disable cuda-graph and radix-cache, skip warmup for cleaner dumps
        f"{'--sglang-disable-cuda-graph --sglang-disable-radix-cache --sglang-warmups 0 ' if _is_debug else ''}"
    )

    fsdp_args = (
        # Set to true for FULL_STATE_DICT mode, false for SHARDED_STATE_DICT mode (default)
        # "--fsdp-full-params "  # Uncomment this line to enable full params mode
        # Set the bucket size for weight update
        "--update-weight-buffer-size 536870912 "  # 512MB
    )

    ci_args = ""

    misc_args = "--actor-num-nodes 1 " f"--actor-num-gpus-per-node {NUM_GPUS} " "--colocate " "--train-backend fsdp "

    if MODEL_NAME == "Qwen3-4B":
        misc_args += (
            "--use-dynamic-batch-size "
            # TODO pick a good value
            "--max-tokens-per-gpu 2048 "
        )

    # AMD/Triton: replace fa3 with triton on both inference and training sides.
    # NOTE: --attn-implementation triton activates the SGLang Triton attention bridge
    # (apply_sglang_triton_attention_patch), which uses extend_attention_fwd_unified --
    # the same kernel as the SGLang inference-side Triton backend.
    # NOTE: Triton bwd is not yet implemented; _compute_log_prob runs under
    # torch.no_grad() so log-prob comparison works end-to-end without bwd.
    true_on_policy_args = (
        "--sglang-enable-deterministic-inference "
        "--sglang-rl-on-policy-target fsdp "
        "--sglang-attention-backend triton "   # inference side: triton (AMD-compatible)
        "--attn-implementation triton "        # training side: SGLang Triton bridge
        "--deterministic-mode "
        "--true-on-policy-mode "
    )
    true_on_policy_envs = {
        # NCCL_ALGO kept as-is; allreduce:tree is not NVIDIA-specific
        "NCCL_ALGO": "allreduce:tree",
        # NVTE_ALLOW_NONDETERMINISTIC_ALGO and CUBLAS_WORKSPACE_CONFIG are NVIDIA/cuBLAS-only,
        # omitted on AMD/ROCm.
    }

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{sglang_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{eval_args} "
        f"{fsdp_args} "
        f"{ci_args} "
        f"{misc_args} "
        f"{true_on_policy_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=None,
        extra_env_vars={
            **true_on_policy_envs,
            "PYTHONPATH": "/data/true_on_policy/sglang/python:/data/true_on_policy/miles:/root/Megatron-LM",
            "SGLANG_DUMPER_ENABLE": "1" if MODE == "debug_one_sample" else "0",
            "SGLANG_DUMPER_DIR": "/data/true_on_policy/results/amd_triton_dumps",
            "SGLANG_TEMP_UTILS_ENABLE_DEBUG_PRINT": "1" if MODE == "debug_one_sample" else "0",
        },
    )


if __name__ == "__main__":
    prepare()
    execute()
