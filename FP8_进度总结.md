# Miles ROCm FP8 训练 — 进度总结（截至 2026-06-04）

> 本文档由 Claude 在 `miles-hai2fp8` 容器内，系统梳理你两个 fork
> (`JessicaJiang-123/miles`、`JessicaJiang-123/sglang`) 上所有 fp8 相关分支后写成。
> 目的是帮你恢复"这个任务做到哪了"的全局记忆。
>
> **一句话现状**：DeepSeek-style blockwise FP8（1×128 激活 / 128×128 权重，E4M3，FP32 scale）
> 在 ROCm/MI355X 上的训练通路**已经从 0 到 1 打通并验证**——
> qwen3-4B dense 端到端跑通、DSv4-Flash 4 层 smoke 跑通、qwen3-30B-A3B MoE 复现脚本就绪。
> 卡点在两处：**(a) DSv4 真实 sglang rollout 与训练侧 MoE 输出对不齐（abs_diff≈11，未解决）；
> (b) 几处 perf gap（wgrad 近似、MoE 逐专家循环、SP gather 走 bf16、norm+quant 未融合）尚未收口。**

---

## 0. 当前环境（这次新建的）

| 项 | 值 |
|---|---|
| 容器 | `miles-hai2fp8`（本次新建，运行中）|
| 镜像 | `rlsys/miles:merge-rocm700-mi35x` |
| 镜像来源 | **已核实**：就是你用 fork 的 `amd-rocm-docker-daily-build` 分支 `docker/Dockerfile.rocm`（默认 build-arg = gfx950/MI355X）build 出来的成品。ENV、构建步骤逐条吻合，2026-06-04 构建，63GB。所以**没有重 build**（重 build 要数小时，且 `/` 盘只剩 111GB 会撑爆）。 |
| 挂载 | `/mnt/data/data/hai` → `/sgl-workspace/hai` |
| 本次产物目录 | `/sgl-workspace/hai/fp8/`：`miles/`（fork 全量 clone）、`sglang/`（fork 全量 clone）、本总结、后续计划 |
| 镜像内 Megatron | `/root/Megatron-LM` = `radixark/Megatron-LM@miles-main`（**注意：不是 yueming 的 deepseek-v4 fork**）|
| 镜像内 sglang | `/sgl-workspace/sglang`（内置版，**不认识 DSv4**）|
| TransformerEngine | ROCm fork，已装在镜像里 |

> ⚠️ 旧的散装 worktree（`miles-te / miles-dsv4 / miles-wgrad / yueming-megatron / TransformerEngine`）
> 现在**都已不在**磁盘上。所有成果都只活在 fork 的 git 分支里（本次已重新 clone 到 `fp8/`）。

---

## 1. 战略判断：为什么是 blockwise，不是 MXFP8

DeepSeek-V3/V4 出厂就是 **block-scaled FP8** 模型：
- 激活/梯度：沿收缩轴按 **1×128** 分组量化
- 权重：**128×128** 块量化
- 数据 E4M3，scale 用 **FP32**

这正是 NV 的 `Float8BlockScaling` 配方 / Megatron 的 `--fp8-recipe blockwise`。

**MXFP8（OCP 规范）是另一套**：1×32 分组、E8M0（2 的幂）scale。两者数学上不等价、不可互换。
训练侧必须和 rollout 侧（sglang+aiter 在 AMD 上用 blockwise FP8 服务 DSv4）走同一套配方。

> 早先有过"MXFP8 训练 + blockwise rollout"的尝试，结论是**双重失败**：
> ①两套配方数学对不上；②ROCm/TE 的 MXFP8 GEMM 在本镜像上本身就坏的
> （`rocRoller only supports F32 as scale type not Half`）。所以 MXFP8 不是捷径，已放弃。

**为什么 ROCm 要补 NV 不用补的活**：NV/TE 在 DeepSeek-V3 时代就出厂支持 blockwise，
所以 yueming 的 DSv4 PR (`radixark/miles#1045`) 只动 miles+Megatron，没碰 TE。
但 AMD 这一层是空的：
- `ROCm/TransformerEngine` 把 blockwise recipe **硬关掉**
  （`quantization.py:103-106`：`if IS_HIP_EXTENSION: return False, "FP8 block scaled gemm not yet supported for ROCm"`）；
- 它的 HIP cast/GEMM kernel 不实现 DeepSeek blockwise 模式；
- hipBLASLt 里**唯一**存在的 block-scaling 路径只支持 `VEC32_UE8M0`（即 MXFP8）。
- **真正的 1×128 E4M3 + FP32 kernel 在 AMD 上只存在于 `aiter`**
  （`gemm_a8w8_blockscale`，sglang 推理就用它）。

> **所以整个任务的本质 = 在 Python 里、运行时 monkeypatch，把 TE 的 blockwise 通路全部改道到 aiter。**

---

## 2. 技术架构（注入机制）

```
miles  ──(--fp8-recipe blockwise)──>  Megatron Float8BlockScaling autocast
                                          │
                                          ▼
        TE 层 (te.Linear / LayerNormLinear / LayerNormMLP / GroupedLinear)
                                          │  ← 运行时 monkeypatch
                                          │     (sitecustomize MetaPathFinder)
                                          ▼
   rocm_te_blockwise_inject.apply() 改写以下挂载点：
     check_fp8_block_scaling_support      (解开 gate)
     Float8BlockQuantizer.quantize        (量化 → aiter 约定)
     cpp_extensions.gemm.general_gemm     (dense GEMM → aiter)
     general_grouped_gemm + split_quantize(MoE → 逐专家 aiter)
     module._common.apply_normalization   (norm+quant 拆成 bf16 norm + aiter quant)
     distributed.gather_along_first_dim   (SP gather 降级 bf16)
     Fp8Padding/Unpadding.align_size      (→128)
                                          │
                                          ▼
              aiter.gemm_a8w8_blockscale   ← 真正的 blockwise FP8 GEMM
                                            (与 sglang 服务 DSv4 同一个 kernel)
```

**注入怎么进到 Ray worker**：`te_inject_site/sitecustomize.py` 在解释器启动时自动跑
（只要其目录在 `PYTHONPATH`），靠 `ROCM_TE_BLOCKWISE_INJECT=1` 开关，
在 `sys.meta_path` 最前面装一个 `MetaPathFinder`，拦截 `transformer_engine.pytorch` 的 import，
在 TE 模块构建**之前**调用 `apply()`。

> ⚠️ **关键坑**：worker 实际加载的是 `te_inject_site/rocm_te_blockwise_inject.py` 这个**自包含副本**
> （不能 import `miles.*`，因为 worker 把 `miles` 解析到 `/root/miles` 那个没改过的 editable 安装）。
> 改 `miles/utils/rocm_te_blockwise.py` **不会**传到 worker——必须同步进 inject_site 副本。
> （commit `1f181a2` 就是为了强制这条规则。）

---

## 3. 两个 fork 的 fp8 分支全景

### 3.1 `JessicaJiang-123/miles` 分支谱系

由底层 kernel 往上层应用，逐层堆叠：

| 分支 | 角色 | 关键内容 | 状态 |
|---|---|---|---|
| `amd-fp8-training` | **kernel 基座** | `rocm_fp8_blockwise.py`：aiter 量化 + linear，3 个 GEMM（fprop/dgrad/wgrad）全走 aiter。wgrad 用**非对称近似**（把 X^T 当 128×128 权重）| ✅ 独立验证：fwd 3.7% / dgrad 4.1% / wgrad 5.0% rel-err vs bf16 |
| `amd-fp8-wgrad-symmetric` | **wgrad 修正** | 加 `symmetric_blockscale_gemm`：两操作数都按 1×128 量化，靠给 aiter 传 per-row B-scale 让 `GROUP_N` 塌缩到 1，无需改 kernel。**与 NV 对称 wgrad 数学完全一致** | ✅ wgrad rel-err 降到 3.6%。**但还没 merge 进下游分支** |
| `amd-fp8-te-run` | **dense TE 接线** | 全套 monkeypatch（gate/quantizer/GEMM/norm/gather）+ sitecustomize 注入器 + qwen3-4B 启动脚本 | ✅ **qwen3-4B dense 端到端跑通**（8×MI355X，TP2/CP4/SP，3 GRPO step，logprob diff 被 bf16 基线 bound 住）|
| `amd-fp8-dsv4-smoke` | **DSv4 + MoE** | yueming DSv4 plugin 移植 + MoE blockwise FP8（逐专家 aiter 循环）+ 权重转换工具链 + fake_rollout（无需 sglang）| ✅ **DSv4-Flash 4 层 smoke 跑通**（TP8/EP8/SP，3 GRPO step，FP8 数值，梯度有限）|
| `amd-fp8-dsv4-faithful` | smoke 之上 | 合入对称 wgrad + 探索 fused fmoe fprop（`fmoe_fp8_blockscale_g1u1`）+ "identity-routing" 喂 Megatron 预排布输入 | ⚠️ fmoe 在隔离测试通过（≈3.3% rel-err），**未接进 live MoE 路径** |
| `amd-fp8-dsv4-real-rollout` | **最前沿（5-29）** | 用**真实 sglang rollout** 跑 DSv4；一长串 Diag-1/2/3 诊断 train↔rollout 散度；Mode 2 统一 blockwise FP8 | ❌ **卡住**：MoE 输出在 miles 与 sglang 间发散（详见 §5）|
| `amd-fp8-docs` | **文档** | `docs/AMD_DSV4_FP8_TRAINING.md`（657 行，覆盖到 smoke 为止）| ✅ 极详尽，但**不含 real-rollout / repro 两个新分支** |
| `rocm-fp8-repro` | **干净复现（5-30，最新）** | qwen3-30B-A3B FP8 单机 MI355X 启动脚本 + 清理过的 dense+MoE 注入器 | ✅ 脚本就绪，瞄准**可交付的真实 MoE 模型** |
| `rocm-fp8-reproduce` | 废弃 | 0 ahead，旧基点 | — 可忽略 |

### 3.2 `JessicaJiang-123/sglang` 分支

| 分支 | 关键内容 | 状态 |
|---|---|---|
| `miles-dsv4-fp8-blockwise` | **real-rollout 的 sglang 搭档（5-29）**：DSv4 的 Diag 对照 dump（sglang 侧）+ 给路由专家施加 `routed_scaling_factor` 修正 MoE 量级 + `SGLANG_DSV4_BF16_MOE`（用 bf16 服务专家）| ⚠️ Diag-3 验证："routed_scaling_factor 修正是对的，但**没把 abs_diff 移动**" → 散度未解 |
| `amd-deepseek-v4` | DSv4 在 sglang/ROCm 上的集成（20/N…25/N 系列：compressor、fuse_wqkv、Triton kernel 等）| 推理侧集成，5-16 |
| `rocm-fp8-repro` | 跟随 upstream，含 DSV4 测试覆盖 | 5-23 |
| `amd-top-sglang-miles*` | sglang↔miles 权重更新对接基线 | — |

---

## 4. 已验证 ✅ vs 未验证 ❌

**在 MI355X（8×gfx950）上已证明：**
- 独立 aiter blockwise FP8 linear：fwd 3.7% / dgrad 4.1% / wgrad 5.0%(非对称) 或 3.6%(对称) rel-err，全有限。
- `te.Linear / LayerNormLinear / LayerNormMLP` 在 `Float8BlockScaling(E4M3)` autocast 下 ≈3.7%；`te.GroupedLinear`(MoE) fwd 3.7% / dx 3.9%。
- **qwen3-4B dense FP8 blockwise 端到端**：3 GRPO step 跑完，train-vs-rollout logprob diff 被 bf16 基线 bound。
- **DSv4-Flash 4 层 FP8**：torch_dist 加载、3 GRPO step、MoE FP8 路径点亮、梯度有限。
- fmoe 单 launch 路径（`fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256`）在 gfx950 上跑通，与逐专家循环 ≈3.3% rel-err（即数值上可直接替换）。
- DSv4 权重转换流水线（FP8 HF →(CPU triton dequant) BF16 HF →(mbridge) torch_dist）4 层模型跑通。

**尚未证明 ❌：**
- **收敛性/精度**（fake_rollout 让 loss 无意义）。
- **多机**。
- **完整 284B DSv4-Flash**。
- **与 NV 的 perf parity**（没做过 head-to-head；已知多处 perf gap）。
- **真实 sglang rollout 闭环**（DSv4 卡在 sglang 不认 DSv4 + MoE 散度）。

---

## 5. 你停在哪：DSv4 real-rollout 的 MoE 散度调查（最关键）

`amd-fp8-dsv4-real-rollout`（miles）+ `miles-dsv4-fp8-blockwise`（sglang）是同一场战斗的两侧，
目标是让 DSv4 真实 sglang rollout 的前向与训练侧对齐。一串二分诊断的结论链：

1. **Diag-1**：MLA 注意力 kernel **不是** bug（用纯 torch fp32 dense MLA 替换 tilelang sparse，diff 不变，11.175→11.181）。
2. **Diag-2**：layer-0 注意力子层对齐（q-norm 两引擎 22.615 完全一致；attn_sink 仅 bf16 舍入差）。
3. **Diag-3（核心发现）**：**散度发生在 MoE 内部**——
   ```
   component        miles max|x|   sglang max|x|
   mlp_input(HC-pre)    0.96           0.60      一致
   moe_output(MoE raw) 20.25           1.19      发散 ← 这里
   hc_post_output      22.88           0.63      把发散往下传
   ```
   miles 有几个 token 在 MoE 里炸到 20+，而 sglang 没有任何 token 超过 1.2。
   Hyper-connection / Sinkhorn 数学已验证两侧一致，**bug 在 MoE 路径**（router 缩放 / shared expert / expert 数学）。
4. **sglang 侧尝试**：给路由专家施加 `routed_scaling_factor`（对齐 train/rollout MoE 量级）。
   **结论：修正本身是对的，但没有移动 abs_diff** → 散度根因仍未定位。
5. **Mode 2**：把训练侧也切到统一 blockwise FP8（之前 H3 实验故意让训练 bf16 / sglang FP8 来隔离漂移）。
   切了之后这个 4/43 层裁剪模型 diff 仍≈11——因为模型本身近乎均匀输出（信号 << 引擎实现漂移），
   但**管线端到端无错跑通**，符合 DeepSeek Mode 2 语义。

> **要点**：这是一个**训练引擎 ↔ 推理引擎 MoE 数值对齐**的硬骨头，**尚未解决**。
> 而且它是在一个 4 层裁剪模型上调试的（信号弱），结论的说服力本身受限。

---

## 6. 与 NV 的差距清单（perf / 保真）

| 模块 | NV | 我们(ROCm) | 性质 |
|---|---|---|---|
| wgrad（下游分支）| 对称 1×128 cuBLAS | 非对称近似(X 当 128×128) | ⚠️ 数值：5.0% vs 3.6%。**对称版已实现但未 merge 下游** |
| dense GEMM | cuBLAS 直派 | Python + aiter + 1~2 次 reshape | ✅ 数值一致；⚠️ perf 可能更低 |
| norm+quant | 融合 HIP kernel | bf16 norm + aiter quant（两次 launch）| ⚠️ 每个 norm 多一次 HBM 往返 |
| SP gather | in-FP8 blockwise(COMPACT) | dequant→bf16→all_gather→requant | ⚠️ ≈2× 通信量 |
| MoE grouped GEMM | 单 launch fused FP8 | **逐专家 Python 循环** | ⚠️ 数值一致；E 次 launch vs 1 次 |
| MoE fprop(在途) | fused FP8 | `aiter.fused_moe`(per_128x128)，≈3.3% | ⚠️ 仅 perf，**未接 live** |
| `Fp8Padding` align | 16 | 128(强制) | ⚠️ 我们 quantizer 需 M%128 |
| jit_fuser(BDA/bias-swiglu)| 开 | 关 | ⚠️ 仅 perf |
| DSv4 HC / QAT ops | TileKernels 融合 | 纯 PyTorch 实现 | ⚠️ 数值等价，perf 慢很多 |
| rollout | sglang 真实 | fake_rollout 随机 token | ❌ smoke-only |

---

## 7. 关键文件索引（都在分支里 / 本次 clone 到 `fp8/miles`）

| 内容 | 路径 | 所在分支 |
|---|---|---|
| blockwise FP8 算法核(对称 wgrad) | `miles/utils/rocm_fp8_blockwise.py` | `amd-fp8-wgrad-symmetric` |
| TE 接线(qwen3-4B 参考实现) | `miles/utils/rocm_te_blockwise.py` | `amd-fp8-te-run` |
| TE 注入器(worker 真正加载的) | `miles/utils/te_inject_site/rocm_te_blockwise_inject.py` | `te-run` / `dsv4-smoke` / `rocm-fp8-repro` |
| qwen3-4B 启动 | `scripts/run_qwen3_4b_blockwise_te.py` | `te-run` |
| DSv4 启动 | `scripts/run_deepseek_v4.py` | `dsv4-smoke` / `real-rollout` |
| qwen3-30B-A3B 启动 | `examples/low_precision/run-qwen3-30b-a3b-fp8-1node-mi355x*.sh` | `rocm-fp8-repro` |
| DSv4 model spec / ops | `miles_plugins/models/deepseek_v4/` | `dsv4-smoke` |
| DSv4 权重转换 | `tools/{fp8_cast_bf16.py,convert_hf_to_torch_dist.py,rename_dsv4_safetensors_to_hf.py}` | `dsv4-smoke` |
| 完整设计文档(657 行) | `docs/AMD_DSV4_FP8_TRAINING.md` | `amd-fp8-docs` |

> **复现 DSv4 还需要**（镜像里没有，要额外准备）：yueming 的 Megatron-LM `deepseek-v4` 分支（加进 PYTHONPATH）、
> yueming 的 sglang DSv4 fork（真实 rollout 才需要）。镜像里是 `radixark/Megatron-LM@miles-main` + 内置 sglang。

---

## 8. 一句话给未来的你

> **dense blockwise FP8 训练（qwen3-4B）已是 done 状态；MoE blockwise FP8 也以"逐专家循环"形态跑通了
> （qwen3-30B-A3B 脚本就绪、DSv4 4 层 smoke 跑通）。
> 真正没收口的是：①对称 wgrad / fmoe fusion 等几个改进没合并；②DSv4 真实 rollout 的 MoE 数值对齐没解决。
> 下一步该走哪条路，见同目录 `FP8_后续计划.md`。**
