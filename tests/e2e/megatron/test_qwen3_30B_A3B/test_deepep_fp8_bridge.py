import os

from tests.ci.ci_register import register_amd_ci, register_cuda_ci
from tests.e2e.megatron.test_qwen3_30B_A3B._common import CaseConfig, execute, prepare

# Limited by host memory
register_cuda_ci(est_time=900, suite="stage-c-8-gpu-h100", labels=["megatron"])
register_amd_ci(est_time=900, suite="stage-c-8-gpu-mi35x", labels=["megatron"], disabled="Deep-EP is WIP on ROCm")
register_amd_ci(est_time=900, suite="stage-c-8-gpu-mi30x", labels=["megatron"], disabled="Deep-EP is WIP on ROCm")

CASE = CaseConfig(
    use_deepep=True,
    use_fp8_rollout=True,
    use_int4_rollout=False,
    use_bridge=True,
    use_r3=False,
    num_gpus_per_node=8,
    cp_size=1,
    pp_size=2,
    sglang_ep_size=8,
    max_tokens_per_gpu=2048,
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE, need_fp8=CASE.use_fp8_rollout, need_int4=CASE.use_int4_rollout, all_bridge=CASE.use_bridge)
    execute(CASE, wandb_file=__file__)
