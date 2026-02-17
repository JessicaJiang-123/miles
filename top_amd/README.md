# Qwen3-4B AMD 一键训练

类似 `examples/true_on_policy/run_simple.py`，自动下载模型/数据并启动 AMD GPU 训练。

## 一键启动

```bash
cd /mnt/data/yuzhen1/top/miles
python top_amd/run.py
```

脚本会依次：
1. 创建 `/data/cache/huggingface/models` 和 `/data/cache/huggingface/datasets` 目录
2. 下载 **Qwen3-4B** (HuggingFace)
3. 下载 **Qwen3-4B_torch_dist** (AMD 预转换，来自 yushengsu)
4. 下载 **dapo-math-17k** 训练数据
5. 下载 **aime-2024** 评估数据
6. 启动 `top_amd/run-qwen3-4B-amd.sh` 训练

## 仅下载（不训练）

```bash
python top_amd/run.py --prepare-only
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TOP_AMD_CACHE` | `/data/cache/huggingface` | 大文件缓存目录 |
| `TOP_AMD_MODEL_DIR` | `{CACHE}/models` | 模型目录 |
| `TOP_AMD_DATA_DIR` | `{CACHE}/datasets` | 数据目录 |
| `HIP_VISIBLE_DEVICES` | `4,5,6,7` | 使用的 AMD GPU |
