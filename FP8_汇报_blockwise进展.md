# Miles ROCm Blockwise FP8 训练 — 进展汇报

> 面向:组员 + 老板。日期:2026-06-05。
> 主题:在 miles 上支持 DeepSeek-style blockwise FP8 **训练**(ROCm / MI355X / gfx950)的进展、技术路径、调研结论、缺口与下一步。

---

## 一、目标与定位(一页讲清要做什么)

- **目标**:让 miles 的 `--fp8-recipe blockwise`(DeepSeek 风格 blockwise FP8)在 AMD ROCm 上能**训练**。
- **配方**:激活按 **1×128**、权重按 **128×128** 量化,数据 **E4M3**,scale 用 **FP32**。这是 NV 的 `Float8BlockScaling` / Megatron `--fp8-recipe blockwise`。
- **不是 MXFP8**(1×32 / E8M0,另一套,不兼容;且 ROCm 的 MXFP8 GEMM 在本镜像上是坏的)。
- **任务本质一句话**:NV 的 TransformerEngine(TE)出厂就支持 blockwise FP8,ROCm 的 TE **全关着**。所以我们要做的,就是**把 TE 的 blockwise 通路在运行时改道到 AMD 的 aiter kernel**。

---

## 二、整体调用链:miles → Megatron → TE → kernel

```
miles  ──(--fp8-recipe blockwise)──►  Megatron 的 Float8BlockScaling autocast
                                          │
                                          ▼
        TE 层(te.Linear / LayerNormLinear / LayerNormMLP / GroupedLinear)
                                          │   ← 运行时 monkeypatch 注入
                                          ▼
        真正算数的 kernel:aiter.gemm_a8w8_blockscale 等
```

- **NV 侧**:miles 调 Megatron,Megatron 调 TE,TE 原生用 CUDA(cuBLAS)算 blockwise FP8。**miles/Megatron 不碰 TE,白嫖。**
- **ROCm 侧**:TE 把 blockwise 整条路关掉了,我们在 TE 和 kernel 之间插一层 monkeypatch,把每个挂载点改道到 aiter。

---

## 三、TE 侧:需要 gate / 改写哪些东西

**只有 1 个真正的 "gate",其余是"开门之后必须铺通的功能改道"。** 注入器 `apply()` 一共改写 9 处:

| # | 挂载点 | 作用 | 性质 |
|---|---|---|---|
| 1 | `check_fp8_block_scaling_support` | **唯一的 gate**:解开"ROCm 不支持 blockwise"的硬拒绝 | 解门 |
| 2 | `Float8BlockQuantizer.quantize` | 量化改走 aiter(1×128 激活 / 128×128 权重)| 改道 |
| 3 | `general_gemm`(dense GEMM)| dense 三个 GEMM 全派到 `aiter.gemm_a8w8_blockscale` | 改道 |
| 4 | `Fp8Padding/Unpadding.align_size→128` | MoE 每专家 M 对齐 128 | 兼容 |
| 5 | `general_grouped_gemm` + `split_quantize` | MoE 路径:逐专家调 aiter | 改道 |
| 6 | `apply_normalization` | norm+quant 拆成 bf16 norm + aiter quant 两步 | 改道(workaround)|
| 7 | `gather_along_first_dim` | 序列并行 gather 降级 bf16 | 改道(workaround)|
| 8 | `TEGroupedMLP.forward` | MoE 前向走 aiter fmoe(在途)| 改道(改的是 Megatron)|
| 9 | `disable_jit_fuser()` | 关掉 torch.compile 融合(避免 dynamo 误报)| workaround |

> **要真正"扔掉 monkeypatch"**,长期归宿是给 **`ROCm/TransformerEngine` 提一组 PR**,把 #2–#7 在 HIP 侧原生接到 aiter,gate(#1)才能名正言顺返回 True。现在的 Python 版是证明"路走得通"的过渡形态。

---

## 四、Kernel 层:核心认知(给非 kernel 背景的同事)

1. **aiter ≠ triton**。aiter 是 AMD 的**kernel 库**;triton 是写 kernel 的一种**语言(DSL)**。aiter 里一部分 kernel 用 triton 写、一部分用汇编/CK(Composable Kernel,AMD 的 C++ 模板库,类比 CUTLASS)写。我们 dense 用的 `gemm_a8w8_blockscale` 是 aiter 里**triton 写的那个**。
2. **一个 GEMM kernel 几乎覆盖一切**。前向(fprop)、反向(dgrad、wgrad)本质都是矩阵乘,**一个 `x@wᵀ` 的 blockwise FP8 GEMM,配合不同的操作数映射,就能表达全部三个**。
3. **量化是 GEMM 的前置步骤,不是 GEMM 自带**。GEMM 吃的是"已量化好的 FP8 + scale";把 bf16 切块量化成 E4M3+FP32-scale 这步要单独做(miles 现在用纯 torch,aiter 也有现成量化 kernel)。
4. **"FP8 训练"= 混合精度,不是全 FP8**:只有**矩阵乘那一步**用 FP8 算(图快、省带宽);**权重真身、累加、优化器**仍是高精度(BF16/FP32)。

---

## 五、已跑通:Dense Blockwise FP8(✅ 这是当前最硬的成果)

**dense 通路端到端打通并验证**,前向直接复用 sglang 推理同款 aiter kernel,反向自己拼。

### 5.1 一个线性层 `Y = X @ Wᵀ` 的三个 GEMM,全走同一个 aiter kernel

| | 算什么 | 收缩维 | 激活类操作数 | 另一操作数 | 量化方式 |
|---|---|---|---|---|---|
| **fprop**(前向)| `Y = X @ Wᵀ` | K | X `[M,K]` 1×128 | W `[N,K]` 权重 128×128 | 非对称 |
| **dgrad**(对输入)| `dX = dY @ W` | N | dY `[M,N]` 1×128 | Wᵀ 权重 128×128 | 非对称 |
| **wgrad**(对权重)| `dW = dYᵀ @ X` | **M(token维)** | dY 1×128 | X 1×128(也是激活)| **对称** |

> **反向是怎么"变出来"的**:纯矩阵乘有个性质——**梯度还是矩阵乘,只是把操作数转置**(`dX = dY@Wᵀ`、`dW=Xᵀ@dY`)。所以**同一个 GEMM kernel,换转置后的输入就能算反向**,不需要新 kernel。wgrad 两边都是激活(都 1×128),靠给 aiter 传 per-row scale 让 `GROUP_N` 塌缩到 1,复用同 kernel 实现对称情形。

### 5.2 验证数值(MI355X / gfx950,vs BF16 参考)

| 层级 | 指标 | rel-err |
|---|---|---|
| 独立 kernel | fprop | **~3.7%** |
| 独立 kernel | dgrad | **~4.1%** |
| 独立 kernel | wgrad(对称,忠实)| **~3.6%**(早期非对称近似是 5.0%)|
| TE 层 | te.Linear / LayerNormLinear / LayerNormMLP | **~3.7%**(out/dgrad/wgrad)|
| TE MoE | te.GroupedLinear | fwd **3.7%** / dx **3.9%** |
| 端到端 | **qwen3-4B dense FP8**,8×MI355X,3 GRPO step | logprob diff 被 bf16 基线 bound ✅ |
| 端到端 | **DSv4-Flash 4 层**,TP8/EP8,3 GRPO step | 梯度有限、MoE FP8 路径点亮 ✅ |

> ~3–4% 是 **FP8 量化的"噪声地板"**,属预期,不是 bug。

---

## 六、MoE:三种做法,以及为什么 NV 训练 ≠ sglang 推理

MoE 的专家 = 一堆小 MLP(gate/up/down 三个线性 + SiLU)。算 MoE 有三种 kernel 路线:

| 方式 | 谁在用 | 一次 launch? | 激活(SiLU)在哪 | 有反向? | kernel |
|---|---|---|---|---|---|
| **逐专家 dense 循环** | **miles ROCm 今天** | ❌ E 次 | 框架单独做 | ✅ | `gemm_a8w8_blockscale` × E |
| **grouped GEMM** | **NV 训练(TE)** | ✅(fc1/fc2 各一发)| **两个 grouped GEMM 之间单独做** | ✅ 转置出 dgrad/wgrad | TE GroupedLinear → cuBLASLt grouped |
| **全融合 fmoe** | **sglang 推理** | ✅(fc1+SiLU+fc2 焊一发)| **融进 kernel** | ❌ **只前向** | aiter `fmoe_fp8_blockscale_g1u1`(汇编/CK)|

**两个关键结论(澄清一个常见误解):**
1. **NV 训练用的不是 fmoe,是 grouped GEMM**(代码确认:`TEGroupedMLP` = grouped GEMM → 单独 SiLU → grouped GEMM,反向靠转置)。**sglang 推理才用全融合 fmoe。** 二者都"一次 launch",但不是一个东西。
2. **fmoe 不能靠转置得到反向,grouped 可以**:因为 fmoe 把**非线性 SiLU + 路由**焊进了 kernel,反向需要链式法则 + 中间激活,不是"转置矩阵乘";而 grouped 是**纯矩阵乘**,转置即得反向。
   → **所以训练侧正确路线 = 拆成 2 个 grouped GEMM + 单独 SiLU + 转置反向(就是 NV 的做法),不需要让 fmoe 长出反向。**

---

## 七、训练 vs 推理的根本区别(解释"为什么权重又是 BF16")

| | 推理(sglang)| 训练(miles)|
|---|---|---|
| 权重 | **静态**,出厂 FP8 ckpt 直接读 | **每步被优化器更新**,必须用 **BF16 master**(FP8 装不下微小更新)|
| 权重量化 | ❌ 不在运行时做 | ✅ **每个 forward 从 BF16 重新量化成 FP8** |
| 激活量化 | ✅(per-1×128)| ✅ |
| 梯度量化 | ❌ 没有反向 | ✅ 反向也要量化 |
| FP8 角色 | 权重的存储格式 | 权重的**临时副本**,做 GEMM 时换上、算完即弃 |

> 所以 miles 训练:先把 FP8 ckpt **反量化成 BF16** 当 master(`fp8_cast_bf16` 的 triton weight_dequant),训练中**每步再把 BF16 重新量化成 FP8** 喂 GEMM。**"FP8 负责快,BF16 负责准。"**

---

## 八、调研结果:ROCm 社区现有什么、缺什么(本地源码 + 联网双重确证)

### 8.1 一句话结论
**ROCm 上没有现成的"带反向、能训练 blockwise FP8 MoE grouped GEMM"。** 现有 blockwise FP8 GEMM 全是前向;带反向的 grouped GEMM 只支持 BF16。

### 8.2 两个关键 kernel(各有一半,没合到一起)—— 这是缺口的核心

| kernel | grouped? | blockwise FP8? | 有反向? | 评价 |
|---|---|---|---|---|
| **aiter `gmm` / `tgmm`** | ✅ | ❌ **只 BF16/FP16** | ✅ **反向齐全**(dgrad/wgrad/bias-grad)| **反向骨架现成,缺 FP8** |
| **aiter `moe_gemm_a8w8_blockscale`** | ✅ | ✅ **blockwise(128)** | ❌ **只前向** | **FP8 现成,缺反向** |

> **我们要造的东西 = 把这两半合起来**(一个 grouped GEMM,既 blockwise FP8 又带反向)。**不是从零写。**

### 8.3 外部社区(联网确证)

| 项目 | 是什么 | blockwise FP8? | 反向/训练? | gfx950? | 备注 |
|---|---|---|---|---|---|
| **DeepGEMM**(deepseek-ai)| FP8 GEMM 库 | ✅(含 MoE grouped)| ✅ **有 wgrad**(2025-05 起)| ❌ **CUDA only** | **黄金参考**:可对照移植反向 |
| **ROCm/DeepEP** | EP 通信(AMD fork)| FP8 dispatch | ❌ | ❌ **仅 gfx942/MI300** | 实验性,gfx950 还没到 |
| **mori(MORI-EP)** | EP 通信(AMD)| FP8 dispatch | 无 autograd | ✅ | sglang 已用;**是扩规模才需要,与 FP8 计算正交** |
| **AMD Primus** | AMD 训练框架 | ❌ 公开只 BF16 | (BF16)| — | 暂无公开 blockwise FP8 MoE 训练 |
| **ROCm/TransformerEngine** | TE 的 ROCm 版 | blockwise **gate 关** | flash-attn 有 | 部分 | 我们就是绕开它 |

### 8.4 一个相关 issue:aiter#2421 "FP8 fused MoE precision"(gfx950)
- 测的是**前向 fmoe**(`aiter.fused_moe`,CK 2-stage)。topk=1 通过;topk=4 约 3.5% 元素超 0.03、max_diff ~0.09。
- **结论:不是 bug,是 FP8 固有噪声**——topk≥2 时多个专家的 BF16 结果相加,把舍入误差放大了。社区(vLLM)靠**放宽容差**接受。
- 对我们的意义:**前向 fmoe 可信**;**MoE 误差随 topk 放大是正常物理现象**,训练侧心里有数即可。

---

## 九、当前进展 & 缺口(给老板的一页)

**已完成 ✅**
- dense blockwise FP8 训练**端到端跑通并验证**(qwen3-4B,3–4% 噪声地板,符合预期)。
- TE 全套改道注入器打通(gate + quantize + GEMM + norm + gather + MoE)。
- MoE 以"逐专家 dense GEMM"形态跑通(DSv4 4 层 smoke、qwen3-30B 脚本就绪)。
- 完成 ROCm 社区全面调研,定位了可复用的现成件。

**核心缺口 ⚠️**
- **MoE 的"带反向 grouped blockwise FP8 GEMM"** —— ROCm 上没有现成的,需要自建(但有现成两半 + DeepGEMM 参考,非从零)。
- 几个 perf 项(对称 wgrad 合并下游、fmoe 前向融合、SP gather 走 FP8、norm+quant 融合)未收口。
- 真实收敛/多机/完整 284B DSv4 未验证;DSv4 真实 rollout 的 MoE 数值对齐(abs_diff≈11)未解。

---

## 十、下一步方向(只列方向,不涉及具体实现)

1. **收口 dense**:把对称 wgrad 合并进 worker 实际加载的注入器副本(纯数值收益)。
2. **真实模型证明**:在 qwen3-30B-A3B 上跑真实 RL,拿收敛曲线(最有说服力的交付)。
3. **攻 MoE 带反向 grouped GEMM**(核心工程):两条候选路——
   - 甲:扩 aiter `gmm/tgmm`(BF16→blockwise FP8),反向逻辑现成;
   - 乙:前向用 `moe_gemm_a8w8_blockscale`,反向参照 DeepGEMM 自写转置 grouped GEMM。
4. **mori / EP 通信**:扩到多节点专家并行时再做,**当前阶段不需要**(与 FP8 计算正交)。

> **总结一句话**:**dense blockwise FP8 训练在 ROCm 上已跑通并验证(噪声地板 ~3–4%,符合预期);MoE 已用"逐专家"形态跑通,真正待造的是"带反向的 grouped blockwise FP8 GEMM"——而 ROCm 已有现成两半 + NV 的 DeepGEMM 参考,是拼接/移植,不是从零。EP 通信(mori)是后续扩规模的事。**
