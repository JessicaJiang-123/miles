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

Usage (run from any worktree of this repo, host or inside `miles-hai2`):
    cd /path/to/<worktree>     # e.g. miles-dsv4, miles-faithful, ...
    HF_TOKEN=hf_... python scripts/run_deepseek_v4.py full-train \
        --model-name DeepSeek-V4-Flash-FP8-4layer --num-nodes 1 --num-gpus-per-node 8
The worktree root is auto-derived from this script's location, so any branch checkout works.
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


def _host_to_container(p: str) -> str:
    """Translate host /mnt/data/... path to container /data/... path.

    The launcher runs on the host but emits paths consumed by Ray workers inside
    `miles-hai2`, which mounts /mnt/data -> /data.
    """
    return "/data" + p[len("/mnt/data"):] if p.startswith("/mnt/data/") else p


# Worktree root (so `miles_plugins.models.deepseek_v4` and our TE injector resolve
# to whichever worktree this script lives in -- works for miles-dsv4, miles-faithful,
# or any future branch checkout without code edits).
_HOST_WORKTREE = str(Path(__file__).resolve().parent.parent)
WORKTREE_ROOT = _host_to_container(_HOST_WORKTREE)
TE_INJECT_SITE = f"{WORKTREE_ROOT}/miles/utils/te_inject_site"
# Yueming's Megatron-LM fork (deepseek-v4 branch); cloned outside any worktree at
# host /mnt/data/data/hai/yueming-megatron -> container /data/data/hai/yueming-megatron.
YUEMING_MEGATRON = "/data/data/hai/yueming-megatron"
# Jessica's sglang fork (miles-dsv4-fp8-blockwise branch, based on
# upstream/amd/deepseek_v4 + 9 amd shims). Cloned at host
# /mnt/data/data/hai/sglang -> container /data/data/hai/sglang.
# Python layer only; we keep miles-hai2's container-installed sgl_kernel/aiter as-is
# (AMD-native compiled extensions), and PYTHONPATH-prepend this Python so DSv4
# model + config + sites load.
YUEMING_SGLANG_PY = "/data/data/hai/sglang/python"


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
    # Flip to True to use REAL sglang DSv4 rollout (sglang fork prepended on PYTHONPATH),
    # instead of miles.rollout.fake_rollout.fake_generate_rollout. Required for any
    # convergence run, and for surfacing real-rollout-only bugs (Megatron->sglang weight
    # bridge, etc) that fake_rollout hid.
    real_rollout: bool = False

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

    # Rollout args are required by the parser. Under --debug-train-only we
    # override the real rollout function with fake_rollout (set in misc_args
    # below); under --real-rollout the sglang DSv4 server handles generation.
    #
    # dapo-math-17k.jsonl ships prompts as list[{"role": "user", "content": str}]
    # (chat format), so the rollout MUST run them through tokenizer.apply_chat_template
    # before calling tokenizer.encode() -- otherwise transformers raises
    # `ValueError: text input must be of type str ...`. The Pinaster DSv4
    # tokenizer doesn't ship a chat_template, so we point --chat-template-path
    # at a minimal jinja template (just joins each message's content) for the
    # smoke. NOT the official DSv4 chat format; only here so the rollout
    # produces *something* tokenize-able so the GRPO step can complete.
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
        "--apply-chat-template "
        # H1 test: use DSv3 chat template (deepseek-ai canonical). DSv4's tokenizer
        # ships <｜User｜>, <｜Assistant｜>, <｜begin▁of▁sentence｜> tokens (vocab IDs
        # 128803, 128804, 0) — same as DSv3 — so the v3 template tokenizes cleanly.
        # The minimal template produced gibberish generations because the model has
        # no idea this is a chat (it's a base model with no chat template). The v3
        # template gives explicit "<｜User｜>...<｜Assistant｜>" framing.
        f"--chat-template-path {WORKTREE_ROOT}/scripts/dsv3_chat_template.jinja "
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

    # Single-node smoke: TP=8, EP=8, SP on, no PP, no CP.
    perf_args = (
        f"--tensor-model-parallel-size {args.num_gpus_per_node} "
        "--sequence-parallel "
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
        # Dump per-token train+rollout logprobs to /tmp/dsv4-real/dump/ so we can
        # post-mortem the train_rollout_logprob_abs_diff (which is ~11.19 on the
        # baseline run, vs qwen3-4B ~0.028). See debug_dump.py and data_source.py
        # for the file layout (rollout_data/{rollout_id}.pt + train_data/{rollout_id}_{rank}.pt
        # + policy_loss_debug/rank_{rank}_call_{counter}.pt).
        "--dump-details /tmp/dsv4-real/dump "
    )
    if args.real_rollout:
        # H5 test: enable Rollout Routing Replay (R3) so training-side recompute
        # uses sglang's MoE expert choices (rather than re-running its own router
        # whose output may diverge due to numerical differences). This is the
        # standard mitigation for MoE train-rollout mismatch (see
        # https://arxiv.org/abs/2510.11370) and is enabled in every other miles
        # MoE launcher (glm4.7-flash, qwen3-30B-A3B, nemotron-3-nano-30b-a3b,
        # qwen3-5-35B). DSv4 has 256 experts + per-token routing, so this is
        # likely the dominant source of the train-rollout logprob diff.
        misc_args += "--use-rollout-routing-replay "
        # The Rust sglang_router (0.3.2) deserializes /generate JSON into a
        # fixed GenerateRequest struct that does NOT include
        # `return_routed_experts`, so it gets silently dropped on the way to
        # the sglang engine and the server's enable_return_routed_experts
        # capturer never fires per-request. Switch to MilesRouter (a Python
        # pass-through proxy in miles/router/router.py) which forwards the
        # request body verbatim, preserving unknown fields. Verified by
        # comparing R3-HTTP-DEBUG (server-side raw body) with R3-PAYLOAD-DEBUG
        # (miles-side payload keys).
        misc_args += "--use-miles-router "
    misc_args += " "

    if args.real_rollout:
        # REAL sglang rollout (via sglang fork on PYTHONPATH). The default
        # --rollout-function-path is miles.rollout.sglang_rollout.generate_rollout
        # so no explicit override needed.
        pass
    else:
        # ROUTINE / fallback: bypass sglang entirely.
        # --debug-train-only skips sglang server init in rollout.py:371; the
        # custom rollout-function below returns synthetic dummy samples so
        # _get_rollout_data still produces a batch and the Megatron train loop
        # has data to train on. This is a TRAINING smoke -- responses are
        # meaningless, but every FP8 GEMM/quant path is exercised.
        misc_args += (
            "--debug-train-only "
            "--rollout-function-path miles.rollout.fake_rollout.fake_generate_rollout "
            # Use the rollout-logprobs (which our fake rollout fills) as the
            # 'old' logprobs in the GRPO loss so the importance-sampling term is
            # well-defined even when the training forward log-probs are at a
            # different shape than what the policy_loss expects from a real rollout.
            "--use-rollout-logprobs "
        )

    # Blockwise FP8 training via our aiter-backed TE patch.
    # H3 test: when args.real_rollout, DISABLE FP8 training to isolate FP8 train
    # vs FP8 infer drift as the cause of train_rollout_logprob_abs_diff ~ 11.
    # The torch_dist checkpoint stores BF16 weights (no FP8 scales for activations
    # baked in), so a pure-BF16 training forward is well-defined; sglang still
    # serves FP8 inference. If the diff stays ~11 with BF16 training, FP8 is NOT
    # the dominant cause and the bug is in the Megatron DSv4 forward path itself.
    misc_args += (
        "--transformer-impl transformer_engine "
        "--bf16 "
    )
    if not args.real_rollout:
        misc_args += (
            "--fp8-format e4m3 "
            "--fp8-recipe blockwise "
            # NO --fp8-param-gather (asserted to require delayed recipe).
        )

    # PYTHONPATH chain: injector -> our worktree root -> yueming-megatron (DSv4)
    # -> [optionally] sglang fork (DSv4 model + configs) for real rollout.
    # (worktree FIRST so its miles/* shadows the editable install at /root/miles,
    # which is missing DSv4 patches; megatron_path AFTER so we still pick our
    # miles_plugins.models.deepseek_v4 over yueming's.)
    pythonpath = f"{TE_INJECT_SITE}:{WORKTREE_ROOT}:{args.megatron_path}"
    if args.real_rollout:
        # Prepend sglang fork AFTER the injector/worktree but BEFORE any system
        # site-packages so the DSv4 model + configs override the installed
        # /sgl-workspace/sglang (which lacks DSv4). Container-installed sgl_kernel,
        # aiter, transformers stay as-is (AMD-native compiled extensions).
        pythonpath = f"{TE_INJECT_SITE}:{WORKTREE_ROOT}:{args.megatron_path}:{YUEMING_SGLANG_PY}"

    extra_env_vars = {
        "ROCM_TE_BLOCKWISE_INJECT": "1",
        "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
        "PYTHONPATH": pythonpath,
        "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
        # sglang fork defaults to SGLANG_APPLY_CONFIG_BACKUP=auto, which loads a
        # hardcoded 43-layer config; that breaks our 4-layer prune. Force "none" so
        # sglang reads the on-disk config.json.
        "SGLANG_APPLY_CONFIG_BACKUP": "none",
    }
    if args.real_rollout:
        # Enable the transformers 5.x rope_parameters -> rope_theta backfill so
        # the sglang fork's deepseek_v4 model can read config.rope_theta.
        # (See miles/utils/te_inject_site/dsv4_transformers_shim.py.)
        extra_env_vars["MILES_DSV4_TRANSFORMERS_SHIM"] = "1"
        # DSv4 sglang serve env vars (mirrors sglang fork's python/run_dsv4.sh):
        # these route DSv4 attention kernels to the ROCm/aiter/Triton paths
        # instead of NV-only CUDA JITs (no deep_gemm, no cuda_runtime.h, etc).
        extra_env_vars.update(
            {
                "SGLANG_USE_AITER": "1",
                "SGLANG_USE_ROCM700A": "1",
                # Skip deep_gemm-based paged MQA logits metadata (HIP path
                # uses aiter's deepgemm_fp8_paged_mqa_logits instead, no
                # deep_gemm package needed).
                "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH": "1",
                # Force HIP-friendly attention backend & helper kernels.
                "SGLANG_HACK_FLASHMLA_BACKEND": "triton",
                # =1 to route through topk_transform_512_pytorch_vectorized
                # instead of torch.ops.sgl_kernel.deepseek_v4_topk_transform_512
                # (the installed sgl_kernel 0.4.1 on miles-hai2 doesn't expose
                # DSv4 ops). Flip back to "0" after sgl_kernel >= 0.4.2 is
                # installed (per hai-1's image), which would add this op
                # natively via the precompiled .so.
                "SGLANG_TOPK_TRANSFORM_512_TORCH": "1",
                "SGLANG_OPT_USE_FUSED_STORE_CACHE": "true",
                "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "false",
                "SGLANG_OPT_USE_FUSED_COMPRESS": "true",
                "SGLANG_OPT_USE_FUSED_HASH_TOPK": "true",
                "SGLANG_OPT_USE_FUSED_PAGED_COMPRESS": "true",
                "SGLANG_OPT_USE_OLD_COMPRESSOR": "false",
                "SGLANG_OPT_USE_TILELANG_INDEXER": "true",
                "SGLANG_OPT_USE_TILELANG_SWA_PREPARE": "false",
                "SGLANG_OPT_USE_TILELANG_MHC_PRE": "false",
                "SGLANG_OPT_USE_TILELANG_MHC_POST": "false",
                # NOTE: miles-hai2's amd-aiter 0.1.11 ships an mhc kernel that
                # GPU-faults during cuda graph capture (memory access fault inside
                # /sgl-workspace/aiter/aiter/jit/module_mhc.so:mhc_pre_big_fuse).
                # Disable the aiter MHC path and fall back to the torch impl
                # (hc_pre_torch_impl / hc_post_torch_impl) which is slower but
                # exercises the same forward semantics. Flip these back to
                # "true" once aiter is upgraded to >= 0.1.14.
                "SGLANG_OPT_USE_AITER_MHC_PRE": "false",
                "SGLANG_OPT_USE_AITER_MHC_POST": "false",
                # Bypass @maybe_torch_compile during cuda graph capture: with
                # AITER_MHC disabled, capture would otherwise run inductor
                # autotune on hc_pre_torch_impl / hc_post_torch_impl for every
                # batch size on every DP rank, adding ~30-60 min to startup.
                "SGLANG_DISABLE_MAYBE_TORCH_COMPILE": "1",
                "SGLANG_OPT_USE_TRITON_SWA_PREPARE": "true",
                "SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK": "false",
                "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "false",
                "SGLANG_OPT_DPSK_V4_RADIX": "1",
                # Disable the SGLang fused-wqa+wkv parameter ("wqkv_a"). With
                # FUSE_WQA_WKV=true SGLang's load_weights expects BOTH the wq_a
                # and wkv halves to arrive in the same load_weights() call
                # (cache_wqkv_a_weight is a local var) so the two halves can be
                # cat'd into the fused "wqkv_a.weight" parameter. miles' refit
                # path buckets parameters by byte budget, which means wq_a and
                # wkv for the same layer may straddle a bucket boundary -- the
                # second half lands in the next load_weights() call, the cache
                # is reset, and the assertion `len(cache_wqkv_a_weight) == 0`
                # fires (AssertionError: dict_keys(['model.layers.N.self_attn.
                # wqkv_a.weight'])). Disabling the fusion makes SGLang build
                # separate wq_a / wkv ReplicatedLinears (one matmul becomes
                # two; functionally equivalent, slightly slower) which match
                # what miles' bridge already emits. Re-enable once miles' refit
                # is taught to keep paired params in the same bucket OR the
                # SGLang side persists cache_wqkv_a_weight across calls.
                "SGLANG_OPT_FUSE_WQA_WKV": "false",
                "SGLANG_FORCE_TRITON_MOE_FP8": "0",
                "SGLANG_ENABLE_THINKING": "1",
                "SGLANG_REASONING_EFFORT": "max",
                "AITER_BF16_FP8_MOE_BOUND": "1",
            }
        )
    # Opt-in: forward MoE through aiter's fused fmoe_fp8_blockscale_g1u1 (one launch
    # for fc1+swiglu+fc2 + routing) instead of the per-expert grouped-GEMM loop.
    # Backward stays on the per-expert path. See miles/utils/te_inject_site/
    # rocm_te_blockwise_inject.py:_patch_megatron_te_grouped_mlp_fmoe for details.
    # Propagate the host env so users can flip it on without editing this file.
    import os as _os
    if _os.environ.get("ROCM_FMOE_FPROP", "0") == "1":
        extra_env_vars["ROCM_FMOE_FPROP"] = "1"

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
