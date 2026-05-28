"""Launcher for qwen3-4B (dense) FP8 *blockwise* training on AMD MI355X (gfx950).

Based on scripts/run_qwen3_4b.py / the prior run_qwen3_4b_mxfp8.py template, but:
  - train recipe is **blockwise** (DeepSeek 1x128/128x128 Float8BlockScaling)
  - the ROCm/TE blockwise path is wired to aiter via our injector: every Megatron Ray
    worker prepends miles/utils/te_inject_site to PYTHONPATH (so sitecustomize auto-runs)
    and sets ROCM_TE_BLOCKWISE_INJECT=1, applying the patch before TE builds modules.
  - NO --fp8-param-gather (megatron asserts it needs --fp8-recipe delayed; also our
    blockwise quantizer doesn't support the fp8 all-gather/compact path).
  - rollout stays blockwise FP8 (the Qwen3-4B-FP8 checkpoint, weight_block_size [128,128]).

Run a SHORT debug_minimal training to capture train-vs-rollout logprob diff:
  python scripts/run_qwen3_4b_blockwise_te.py --mode debug_minimal --hardware MI355X ...
"""
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

# Container path to the injector site dir (this worktree, mounted in miles-hai2).
TE_INJECT_SITE = "/data/data/hai/miles-te/miles/utils/te_inject_site"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "debug_minimal"
    run_id: str = U.create_run_id()
    model_name: str = "Qwen3-4B"
    megatron_model_type: str | None = None
    num_gpus_per_node: int | None = None
    hardware: str = "H100"
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    enable_eval: bool = False
    rollout_fp8: bool = True
    train_fp8: bool = True

    def __post_init__(self):
        self.megatron_model_type = {
            "Qwen3-4B": "qwen3-4B",
            "Qwen3-4B-Base": "qwen3-4B",
        }[self.model_name]
        # MI355X has 8 GPUs; reuse H100's count if not specified.
        self.num_gpus_per_node = self.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE.get(
            self.hardware, 8
        )


def prepare(args: ScriptArgs):
    # Weights / datasets / torch_dist already exist in the container; skip download/convert.
    pass


def execute(args: ScriptArgs):
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"
    ref_load_path = f"{args.model_dir}/{args.model_name}_torch_dist"

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name}{'-FP8' if args.rollout_fp8 else ''} "
        f"--load {load_save_path} "
        f"--ref-load {ref_load_path} "
        f"--save {load_save_path} "
        f"--save-interval {2 if args.mode == 'debug_minimal' else 20} "
        f"--save-retain-interval {2 if args.mode == 'debug_minimal' else 20} "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {3 if args.mode == 'debug_minimal' else 3000} "
        f"--rollout-batch-size {8 if args.mode == 'debug_minimal' else 32} "
        f"--n-samples-per-prompt {4 if args.mode == 'debug_minimal' else 8} "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 8192} "
        "--rollout-temperature 1 "
        f"--global-batch-size {32 if args.mode == 'debug_minimal' else 256} "
        "--balance-data "
        f"{'--num-epoch 1 ' if args.mode == 'debug_minimal' else ''}"
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
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

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-chunked-prefill-size 4096 "
        "--sglang-mem-fraction-static 0.7 "
    )

    tp_size = 2 if args.num_gpus_per_node == 8 else 1
    cp_size = 4 if args.num_gpus_per_node == 8 else 1
    train_backend_args = (
        f"--tensor-model-parallel-size {tp_size} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        f"--context-parallel-size {cp_size} "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--train-memory-margin-bytes 3221225472 "
        # Keep the bias-dropout-add / bias-swiglu fusions eager (they are torch.compile'd
        # via Megatron jit_fuser). Not strictly required after the GEMM output-rank fix, but
        # avoids dynamo tracing over our aiter blockwise FP8 path. Injector also calls
        # disable_jit_fuser().
        "--no-bias-dropout-fusion "
        "--no-bias-swiglu-fusion "
    )

    perf_args = "--use-dynamic-batch-size --max-tokens-per-gpu 9216 "

    misc_args = (
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        f"--dump-details {args.output_dir}/{args.run_id}/dump_details "
    )

    # --- blockwise FP8 training via our aiter-backed TE patch ---
    misc_args += (
        "--transformer-impl transformer_engine "
        "--bf16 "
        "--fp8-format e4m3 "
        "--fp8-recipe blockwise "
        # deliberately NO --fp8-param-gather
    )
    # Prepend the injector site dir AND keep megatron_path on PYTHONPATH (extra_env_vars
    # overrides the framework's PYTHONPATH=megatron_path, so we must include it).
    misc_env_vars = {
        "ROCM_TE_BLOCKWISE_INJECT": "1",
        "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
        "PYTHONPATH": f"{TE_INJECT_SITE}:{args.megatron_path}",
    }

    eval_args = ""
    if args.mode != "debug_minimal" and args.enable_eval:
        eval_args = (
            f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 16 "
            "--eval-max-response-len 16384 "
            "--eval-top-p 1 "
            "--eval-interval 20 "
        )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{train_backend_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars=misc_env_vars,
        megatron_path=args.megatron_path,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
