# Miles ROCm Blockwise FP8 训练 — 后续计划

> 目标（你定的）：**在 miles 侧支持 FP8 training，先聚焦 DeepSeek-style blockwise FP8。**
> 现状基线见同目录 `FP8_进度总结.md`。
> 核心认识：**dense 通路已 done，MoE 以"逐专家循环"跑通；剩下的是"收口改进" + "DSv4 真实 rollout 数值对齐"两类活。**

---

## 0. 总体路线判断

现在有两条可走的路，性质完全不同：

- **Track A（工程收口 + 可交付）**：把已验证的 blockwise FP8 enablement**收敛成一个干净、可复现、可 PR 的特性**，
  并在一个**真实 MoE 模型 + 真实 RL rollout**（qwen3-30B-A3B，upstream 配方，sglang 认识它）上跑出**收敛证据**。
  风险低、产出明确，直接满足"miles 在 ROCm 上支持 blockwise FP8 training"这个目标。
- **Track B（研究攻坚）**：解决 DSv4 真实 sglang rollout 的 MoE 数值散度（abs_diff≈11），打通 DSv4-Flash 真实 RL。
  风险高、依赖 yueming 的 Megatron/sglang fork、且 bug 根因未定位。

> **推荐顺序：先 A 后 B。** 理由：
> 1. 你的目标是"支持 fp8 training"，Track A 能拿到**真实收敛曲线**这个最强证据，而 DSv4 卡在数值 bug 上短期出不了这个证据。
> 2. Track A 的收口（对称 wgrad、fmoe fusion、注入器统一）**对 DSv4 也是必需的**，是公共底座。
> 3. DSv4 散度调试目前在 4 层裁剪模型上做（信号 << 漂移），结论本身不可靠；先把底座和真实模型立住，再回头打 DSv4 更稳。

---

## Track A — 收口 + 真实 MoE 模型证明（推荐先做）

### A1. 合并对称 wgrad 到主线 ⭐ 最高优先、最便宜
- **为什么**：对称 1×128 wgrad 已实现（`amd-fp8-wgrad-symmetric` 的 `symmetric_blockscale_gemm`），
  与 NV 数学完全一致（wgrad rel-err 5.0%→3.6%）。但下游所有分支（te-run / dsv4 / repro）仍在用非对称近似。这是白捡的数值改进。
- **起点**：`amd-fp8-wgrad-symmetric:miles/utils/rocm_fp8_blockwise.py`
- **动作**：把对称 wgrad 逻辑同步进
  ① `miles/utils/rocm_te_blockwise.py` 的 dense GEMM wgrad 分支；
  ② **`te_inject_site/rocm_te_blockwise_inject.py`（worker 真正加载的副本，别漏！）**；
  ③ MoE 逐专家循环的 wgrad（它复用 dense `general_gemm`，合并后自动继承）。
- **验收**：`tests/rocm/test_blockwise_fp8_linear.py` wgrad rel-err ≤ 3.6%；qwen3-4B smoke 仍跑通。
- **风险**：低。注意 inject_site 副本同步这条规则。

### A2. 在 qwen3-30B-A3B 上跑通真实 RL（不是 fake_rollout）⭐ 拿收敛证据
- **为什么**：这是**真实 MoE 模型 + upstream blockwise 配方 + sglang 真实 rollout**，
  是证明"miles 支持 blockwise FP8 training"最有说服力的 demo，且不依赖 DSv4 那套尚未上游的东西。
- **起点**：`rocm-fp8-repro:examples/low_precision/run-qwen3-30b-a3b-fp8-1node-mi355x.sh`（脚本已就绪）
- **动作**：
  1. 准备权重/数据：`models/Qwen3-30B-A3B-FP8/`、`Qwen3-30B-A3B_torch_dist/`、`datasets/dapo-math-17k/`（确认在 `/sgl-workspace/hai/models`、`/datasets` 下）。
  2. 先按脚本默认（`SGLANG_USE_AITER=0` 走 Triton MoE，避开 colocate memory-saver 下 aiter fused_moe 的显存 fault）跑 2 个 num-rollout 的 smoke。
  3. smoke 过后放大 num-rollout / response len，跑一段**真实训练**，记录 reward / loss / train-rollout logprob diff 曲线。
- **验收**：① 端到端无错跑完多个 rollout；② reward 曲线随训练上升（或 loss 下降），证明收敛；③ train↔rollout logprob diff 在合理范围。
- **风险**：中。已知坑：colocate 下 aiter fused_moe 显存 fault（用 Triton 绕过）；`/root` 盘 99% 满（脚本已把路径挪到 `/data`）。

### A3. 落地 fmoe fprop fusion（perf）
- **为什么**：MoE fprop 现在是逐专家循环（E 次 launch）。`aiter.fused_moe`（`QuantType.per_128x128`）单 launch 已验证数值等价（≈3.3%），是实打实的 perf 收益。
- **起点**：`amd-fp8-dsv4-faithful` 的 `6e1ba62`/`a539171`（隔离测试 + identity-routing 喂 Megatron 预排布输入的技巧已证明）。
- **动作**：写 custom-autograd `TEGroupedMLP` patch——fprop 走 fmoe 单 launch，bwd 仍走现有逐专家 dgrad/wgrad 循环。关键操作数布局：`w1=[E,2I,K]`(gate-first 拼接)、`w2=[E,K,I]`、`shuffle_weight((16,16))`、scale `[E,?/128,?/128]` fp32。
- **验收**：MoE 集成测试 fwd/dx rel-err 与逐专家循环一致(~3%)；qwen3-30B-A3B 单步 MoE wall-time 下降。
- **风险**：中。fprop/bwd 量化路径不一致需小心 autograd 正确性。

### A4. 收掉两个通信/计算 perf gap（可选，看是否要 perf parity）
- **SP gather**：现在 dequant→bf16→all_gather→requant，≈2× 通信。要 parity 需实现 COMPACT 格式转换并走 TE blockwise gather（或上游贡献给 ROCm/TE）。
- **norm+quant**：现在 bf16 norm + aiter quant 两次 launch。要 parity 需一个懂 DeepSeek blockwise 的 HIP fused `rmsnorm+quant` kernel（目前不存在）。
- **判断**：这两个是 perf-only，**功能不阻塞**。除非要对 NV 做 perf parity，否则可押后。

---

## Track B — DSv4 真实 rollout 攻坚（研究，A 之后）

### B1. 定位 MoE 散度根因（核心硬骨头）
- **现状**：Diag-3 已把散度锁定在 **MoE 内部**（miles moe_output max 20.25 vs sglang 1.19）；
  sglang 侧 `routed_scaling_factor` 修正"对但不动 abs_diff"。根因未定位。
- **下一步动作**（接 `825a143` 的 routed-only vs shared-only dump）：
  1. 跑 `825a143` 那个 dump，看是**路由专家**还是 **shared expert** 炸（shared expert 在 miles 侧因 `--no-activation-func-clamp-shared-expert` 未 clamp，是重点嫌疑）。
  2. 逐子项对齐：router 权重/归一化、expert 输入 permute、gate/up/down 的量化对象、activation clamp 策略。
  3. 在**更大/非退化**的模型切片上复现（4 层裁剪信号太弱，结论不可靠——考虑用更多层或真实输入分布）。
- **依赖**：yueming 的 Megatron `deepseek-v4` 分支 + sglang DSv4 fork（`miles-dsv4-fp8-blockwise` 已是 sglang 搭档分支）。
- **风险**：高。这是训练↔推理两套 MoE 实现的数值对齐，可能涉及量化点、clamp、scaling 多处累积差。

### B2. 让镜像里的 sglang 认识 DSv4
- **现状**：本镜像内置 sglang 不认 DSv4，真实 rollout 闭环卡在这。
- **动作**：把 `JessicaJiang-123/sglang:miles-dsv4-fp8-blockwise`（或 `amd-deepseek-v4`）装进运行环境（PYTHONPATH 或重装）。评估是否值得进 Dockerfile.rocm（注意：那会改动镜像，又回到磁盘问题）。

### B3. 真实收敛 / 多机 / 完整 284B
- 全部押后到 B1/B2 通了之后。fake_rollout 的 loss 无意义，必须先有真实 rollout 才能谈收敛。

---

## Track C — 上游化 / 长期（机会性推进）

- **C1. TE 改进推 `ROCm/TransformerEngine`**：把 gate、`quantize`、`general_gemm`、`apply_normalization`、`general_grouped_gemm`、`gather_along_first_dim` 的 aiter 改道做成正经 C++ PR，最终丢掉 sitecustomize monkeypatch。
- **C2. rebase 到含 yueming DSv4 PR 的 `radixark/miles` main**，删掉我们重复的 DSv4 plugin 文件，只保留 ROCm 专属移植（纯 torch `qat.py`/`hyper_connection.py`、bf16×bf16→fp32 GEMM workaround）。
- **C3. 把 `amd-fp8-docs` 文档补上 real-rollout / repro 两个新分支**（目前文档只到 smoke）。

---

## 立即可做的第一步（建议本周）

1. **A1 对称 wgrad 合并**（半天，纯收益，零依赖）——先把这个白捡的数值改进落到 inject_site 副本，跑 `test_blockwise_fp8_linear.py` 验证。
2. **A2 的环境核查**：确认 `models/Qwen3-30B-A3B-FP8`、`Qwen3-30B-A3B_torch_dist`、`datasets/dapo-math-17k` 是否就位；不在就先准备。
3. 跑 `rocm-fp8-repro` 的 qwen3-30B-A3B **2-rollout smoke**，确认在 `miles-hai2fp8` 新容器里能起来。

> 这三步做完，"miles 支持 blockwise FP8 training" 就有了一个真实 MoE 模型上可复现、可展示的落点，再决定要不要回头啃 DSv4。

---

## 需要你提供 / 拍板的

1. **优先级确认**：同意"先 Track A（qwen3-30B-A3B 真实 RL 拿收敛证据），后 Track B（DSv4 攻坚）"吗？还是你更想直接继续 DSv4 real-rollout 的 MoE 散度调查？
2. **权重/数据位置**：`Qwen3-30B-A3B-FP8`、对应 `torch_dist`、`dapo-math-17k` 现在在哪？（我能在容器里帮你核查 `/sgl-workspace/hai/models` 和 `/datasets`。）
3. **DSv4 依赖**：如果走 Track B，需要 yueming 的 Megatron `deepseek-v4` 分支和 sglang DSv4 fork 的确切位置/URL——这两个不在当前镜像里。
4. **是否追求 NV perf parity**：决定 A3/A4 这些 perf-only 项要不要做。
5. **HF_TOKEN**：DSv4 4 层模型从 HF 下载需要 token；真实跑 30B 也可能要拉权重。
