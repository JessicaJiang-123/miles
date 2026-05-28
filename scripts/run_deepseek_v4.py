"""Launcher for DeepSeek-V4-Flash-FP8-4layer single-node smoke test on AMD MI355X (gfx950).

Adapted from yueming-yuan/miles@deepseek-v4 scripts/run_deepseek_v4.py:
  - Targets only the 4-layer prune of sgl-project/DeepSeek-V4-Flash-FP8 (Pinaster mirror),
    which is the documented single-node profile.
  - Wires the ROCm/TE *blockwise* FP8 path via our injector (set ROCM_TE_BLOCKWISE_INJECT=1
    and prepend miles/utils/te_inject_site to the worker PYTHONPATH so sitecustomize fires
    before TE imports).
  - Forces yueming's Megatron-LM fork onto the worker PYTHONPATH ahead of the installed
    /root/Megatron-LM (his branch adds DSv4 plumbing in MLA/dsa.py/transformer_block etc).
  - Forces --debug-train-only by default (no sglang rollout; the installed sglang in
    /sgl-workspace/sglang does NOT know DSv4 yet, and adding it is out of scope for the
    training-only smoke).

Usage (run inside container `miles-hai2`):
    cd /data/data/hai/miles-dsv4
    HF_TOKEN=hf_... python scripts/run_deepseek_v4.py full-train \
        --model-name DeepSeek-V4-Flash-FP8-4layer --num-nodes 1 --num-gpus-per-node 8
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_DEFAULT_MODEL_ORG = {
    "DeepSeek-V4-Flash-FP8-4layer": "Pinaster",
}

_MEGATRON_MODEL_TYPE = {
    "DeepSeek-V4-Flash-FP8-4layer": "deepseek-v4-flash-4layer",
}

# Container path to the injector site dir (this worktree, mounted in miles-hai2).
TE_INJECT_SITE = "/data/data/hai/miles-dsv4/miles/utils/te_inject_site"
# Yueming's Megatron-LM fork (deepseek-v4 branch) cloned in host /mnt/data/data/hai;
# container sees it at /data/data/hai.
YUEMING_MEGATRON = "/data/data/hai/yueming-megatron"
# Worktree root (so `miles_plugins.models.deepseek_v4` resolves to OUR copy).
WORKTREE_ROOT = "/data/data/hai/miles-dsv4"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "debug_minimal"
    run_id: str = U.create_run_id()
    model_org: str = ""
    model_name: Literal["DeepSeek-V4-Flash-FP8-4layer"] = "DeepSeek-V4-Flash-FP8-4layer"

    hf_checkpoint: str | None = None
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    save_dir: str = "/root/models"
    # Use yueming's Megatron-LM fork (has DSv4 plumbing); fallback /root/Megatron-LM otherwise.
    megatron_path: str = YUEMING_MEGATRON

    num_gpus_per_node: int = 8

    # debug configs
    skip_saving: bool = True
    extra_args: str = ""

    def __post_init__(self):
        if not self.model_org:
            self.model_org = _DEFAULT_MODEL_ORG[self.model_name]

    @property
    def megatron_model_type(self):
        return _MEGATRON_MODEL_TYPE[self.model_name]

    @property
    def torch_dist_name(self):
        return f"{self.model_name}_torch_dist"

    @property
    def bf16_name(self):
        return f"{self.model_name}-bf16"


def _hf_checkpoint_path(args: ScriptArgs) -> str:
    return args.hf_checkpoint or f"{args.model_dir}/{args.model_name}"


def _patch_4layer_model_type(args: ScriptArgs):
    """HF transformers doesn't know `deepseek_v4`; rewrite the 4-layer config's
    model_type directly to `deepseek_v3`. transformers' DeepseekV3Config preserves
    all unknown extra fields (hc_mult, compress_ratios, ...) as plain attributes,
    so our mbridge ``_patched_from_config`` keys on ``hasattr(hf_config, 'hc_mult')``
    and still picks DeepseekV4Bridge. Yueming's pipeline rewrites to `deepseek_ref`
    and then relies on sglang's `_load_deepseek_temp_model` to substitute the
    config at load time; we don't have that path (sglang in this image doesn't
    know DSv4), so we skip the intermediate.
    """
    cfg = Path(_hf_checkpoint_path(args)) / "config.json"
    if not cfg.exists():
        return
    text = cfg.read_text()
    for old in ('"model_type": "deepseek_v4"', '"model_type": "deepseek_ref"'):
        if old in text:
            text = text.replace(old, '"model_type": "deepseek_v3"')
            cfg.write_text(text)
            print(f"[patch] {cfg}: model_type -> deepseek_v3")
            break


def _prepare_download(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.model_dir} {args.data_dir}")
    if args.hf_checkpoint is None:
        dest = f"{args.model_dir}/{args.model_name}"
        U.exec_command(
            f"hf download {args.model_org}/{args.model_name} --local-dir {dest}"
        )
    _patch_4layer_model_type(args)
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)


@app.command()
@U.dataclass_cli
def prepare_download(args: ScriptArgs):
    _prepare_download(args)


def _prepare_single(args: ScriptArgs):
    src = _hf_checkpoint_path(args)
    U.fp8_cast_bf16(path_src=src, path_dst=f"{args.model_dir}/{args.bf16_name}/")


@app.command()
@U.dataclass_cli
def prepare_single(args: ScriptArgs):
    _prepare_single(args)


def _prepare_spmd(args: ScriptArgs):
    # 4-layer prune: single-node, no PP/EP sharding -- TP=1,PP=1,EP=1,CP=1.
    extra_args = (
        "--expert-tensor-parallel-size 1 "
        "--context-parallel-size 1 "
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--expert-model-parallel-size 1 "
    )
    num_gpus_for_convert = min(args.num_gpus_per_node, 4)
    U.convert_checkpoint(
        model_name=args.model_name,
        hf_checkpoint=f"{args.model_dir}/{args.bf16_name}",
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=num_gpus_for_convert,
        multinode=False,
        num_nodes=args.num_nodes,
        extra_args=extra_args,
        dir_dst=f"{args.model_dir}",
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def prepare_spmd(args: ScriptArgs):
    _prepare_spmd(args)


def _train(args: ScriptArgs):
    _patch_4layer_model_type(args)

    load_save_path = f"{args.save_dir}/{args.run_id}/checkpoints"
    hf_checkpoint = args.hf_checkpoint or f"{args.model_dir}/{args.model_name}"
    ckpt_args = (
        f"--hf-checkpoint {hf_checkpoint} "
        f"--ref-load {args.model_dir}/{args.torch_dist_name} "
    )
    if not args.skip_saving:
        ckpt_args += (
            f"--load {load_save_path} "
            f"--save {load_save_path} "
            "--save-interval 20 "
            "--save-retain-interval 20 "
        )

    # Rollout args are required by the parser. We override the real rollout function
    # with miles.rollout.fake_rollout.fake_generate_rollout (set in misc_args below).
    # No --apply-chat-template: the Pinaster DSv4 tokenizer doesn't ship one.
    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 256 "
        "--rollout-temperature 1 "
        "--global-batch-size 32 "
        "--balance-data "
        "--num-epoch 1 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
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

    # Single-node smoke: TP=8, EP=8, no SP, no PP, no CP.
    # Note: --sequence-parallel was removed -- under TP=8 SP, the model returns
    # per-rank sequence-sharded logits, but `get_responses` indexes into them as
    # if they were full-sequence, yielding empty log_probs on most ranks and the
    # `ppo_kl = old_log_probs(128) - log_probs(0)` shape mismatch. Without SP,
    # logits are full-sequence on every rank and the loss path works as-is.
    # For training-step efficiency this is wasteful at scale, but for the smoke
    # (4-layer model, small batch) it's fine.
    perf_args = (
        f"--tensor-model-parallel-size {args.num_gpus_per_node} "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {args.num_gpus_per_node} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--micro-batch-size 1 "
        "--max-tokens-per-gpu 2048 "
    )

    # Rollout/sglang args: must be present so argparse doesn't fail, but unused under
    # --debug-train-only.
    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.num_gpus_per_node} "
        f"--sglang-tp-size {args.num_gpus_per_node} "
        f"--sglang-dp-size {args.num_gpus_per_node} "
        f"--sglang-ep-size {args.num_gpus_per_node} "
        "--sglang-enable-dp-attention "  # sglang_dp_size > 1 requires this
        "--sglang-chunked-prefill-size 8192 "
        "--sglang-mem-fraction-static 0.7 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--train-memory-margin-bytes 3221225472 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--model-name deepseekv4 "  # for mbridge load
        "--qkv-format bshd "
        "--colocate "
        # ROUTINE: bypass rollout (sglang on this image doesn't know DSv4):
        # --debug-train-only skips sglang server init in rollout.py:371; the
        # custom rollout-function below returns synthetic dummy samples so
        # _get_rollout_data still produces a batch and the Megatron train loop
        # has data to train on. This is a TRAINING smoke -- responses are
        # meaningless, but every FP8 GEMM/quant path is exercised.
        "--debug-train-only "
        "--rollout-function-path miles.rollout.fake_rollout.fake_generate_rollout "
        # Use the rollout-logprobs (which our fake rollout fills) as the
        # 'old' logprobs in the GRPO loss so the importance-sampling term is
        # well-defined even when the training forward log-probs are at a
        # different shape than what the policy_loss expects from a real rollout.
        "--use-rollout-logprobs "
    )

    # Blockwise FP8 training via our aiter-backed TE patch.
    misc_args += (
        "--transformer-impl transformer_engine "
        "--bf16 "
        "--fp8-format e4m3 "
        "--fp8-recipe blockwise "
        # NO --fp8-param-gather (asserted to require delayed recipe).
    )

    extra_env_vars = {
        "ROCM_TE_BLOCKWISE_INJECT": "1",
        "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
        # PYTHONPATH chain: injector -> our worktree root -> yueming-megatron (DSv4)
        # (worktree FIRST so its miles/* shadows the editable install at /root/miles,
        # which is missing DSv4 patches; megatron_path AFTER so we still pick our
        # miles_plugins.models.deepseek_v4 over yueming's).
        "PYTHONPATH": f"{TE_INJECT_SITE}:{WORKTREE_ROOT}:{args.megatron_path}",
        "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
        "SGLANG_APPLY_CONFIG_BACKUP": "none",
    }

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars=extra_env_vars,
        megatron_path=args.megatron_path,
        # Use OUR worktree's train.py and miles/* (the installed /root/miles is missing
        # the DSv4 patches: hf_validate_args intermediate_size skip, --use-indexer-replay
        # flags, etc). Pinning train_script keeps everything self-consistent.
        train_script=f"{WORKTREE_ROOT}/train.py",
    )


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    _train(args)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs):
    _prepare_download(args)

    bf16_dir = Path(f"{args.model_dir}/{args.bf16_name}")
    bf16_sentinel = bf16_dir / "model.safetensors.index.json"
    if not bf16_sentinel.exists():
        _prepare_single(args)
    else:
        print(f"[full_train] Skipping FP8->BF16 cast: {bf16_sentinel} already exists.")

    torch_dist_dir = Path(f"{args.model_dir}/{args.torch_dist_name}")
    torch_dist_sentinel = torch_dist_dir / "latest_checkpointed_iteration.txt"
    if not torch_dist_sentinel.exists():
        _prepare_spmd(args)
    else:
        print(f"[full_train] Skipping BF16->torch_dist conversion: {torch_dist_sentinel} already exists.")

    if args.hf_checkpoint is None:
        args.hf_checkpoint = f"{args.model_dir}/{args.model_name}"

    _train(args)


if __name__ == "__main__":
    app()
