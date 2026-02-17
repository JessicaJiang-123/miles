#!/usr/bin/env python3
"""
Qwen3-4B AMD 一键训练脚本
自动下载模型/数据并启动训练。默认 Megatron 后端，加 --fsdp 使用 FSDP 后端。

用法:
  cd /mnt/data/yuzhen1/top/miles && python top_amd/run.py           # Megatron
  cd /mnt/data/yuzhen1/top/miles && python top_amd/run.py --fsdp   # FSDP

环境变量(可选):
  TOP_AMD_CACHE      大文件缓存目录，默认 /data/cache/huggingface
  TOP_AMD_MODEL_DIR  模型目录，默认 {CACHE}/models
  TOP_AMD_DATA_DIR   数据目录，默认 {CACHE}/datasets
  HIP_VISIBLE_DEVICES  使用的 GPU，默认 4,5,6,7
"""

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MILES_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(MILES_ROOT))

from miles.utils.misc import exec_command


def get_cache_base() -> str:
    return os.environ.get("TOP_AMD_CACHE", "/data/cache/huggingface")


def get_model_dir() -> str:
    cache = get_cache_base()
    return os.environ.get("TOP_AMD_MODEL_DIR", f"{cache}/models")


def get_data_dir() -> str:
    cache = get_cache_base()
    return os.environ.get("TOP_AMD_DATA_DIR", f"{cache}/datasets")


def prepare(need_torch_dist: bool = True):
    """下载模型、转换权重、数据集。FSDP 模式不需要 torch_dist。"""
    model_dir = get_model_dir()
    data_dir = get_data_dir()

    print("=" * 60)
    print("Step 1: 创建目录")
    print("=" * 60)
    exec_command(f"mkdir -p {model_dir} {data_dir}")

    hf_path = f"{model_dir}/Qwen3-4B"
    if not Path(hf_path).exists() or not (Path(hf_path) / "config.json").exists():
        print("\n" + "=" * 60)
        print("Step 2: 下载 Qwen3-4B (HuggingFace)")
        print("=" * 60)
        exec_command(f"hf download Qwen/Qwen3-4B --local-dir {hf_path}")
    else:
        print(f"跳过: {hf_path} 已存在")

    if need_torch_dist:
        torch_dist_path = f"{model_dir}/Qwen3-4B_torch_dist"
        if not Path(torch_dist_path).exists():
            print("\n" + "=" * 60)
            print("Step 3: 下载 Qwen3-4B_torch_dist (AMD 预转换，Megatron 用)")
            print("=" * 60)
            exec_command(f"hf download yushengsu/Qwen3-4B-torch-dist --local-dir {torch_dist_path}")
        else:
            print(f"跳过: {torch_dist_path} 已存在")
    else:
        print("\n(FSDP 模式跳过 torch_dist 下载)")

    dapo_path = f"{data_dir}/dapo-math-17k"
    if not Path(dapo_path).exists():
        print("\n" + "=" * 60)
        print("Step 4: 下载 dapo-math-17k 数据集")
        print("=" * 60)
        exec_command(f"hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir {dapo_path}")
    else:
        print(f"跳过: {dapo_path} 已存在")

    aime_path = f"{data_dir}/aime-2024"
    if not Path(aime_path).exists():
        print("\n" + "=" * 60)
        print("Step 5: 下载 aime-2024 数据集")
        print("=" * 60)
        exec_command(f"hf download --repo-type dataset zhuzilin/aime-2024 --local-dir {aime_path}")
    else:
        print(f"跳过: {aime_path} 已存在")

    exec_command(f"mkdir -p {model_dir}/Qwen3-4B_miles {model_dir}/Qwen3-4B_miles_fsdp")
    print("\n" + "=" * 60)
    print("Prepare 完成!")
    print("=" * 60)


def execute():
    """启动 AMD 训练（使用 top_amd 内修复过的脚本）"""
    model_dir = get_model_dir()
    data_dir = get_data_dir()

    os.environ["MILES_DIR"] = str(MILES_ROOT)
    os.environ["MODEL_DIR"] = model_dir
    os.environ["DATA_DIR"] = data_dir
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES", "1")
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0,1,2,3")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")

    script_path = SCRIPT_DIR / "run-qwen3-4B-amd.sh"
    if not script_path.exists():
        raise FileNotFoundError(f"找不到 {script_path}")

    print("=" * 60)
    print("启动训练 (top_amd/run-qwen3-4B-amd.sh)")
    print(f"  MODEL_DIR={model_dir}")
    print(f"  DATA_DIR={data_dir}")
    print("=" * 60)

    subprocess.run(
        ["bash", str(script_path)],
        cwd=str(MILES_ROOT),
        env=os.environ,
    )


def execute_fsdp():
    """启动 AMD FSDP 训练（使用 scripts/run-qwen3-4B-amd-fsdp.sh）"""
    model_dir = get_model_dir()
    data_dir = get_data_dir()

    os.environ["MILES_DIR"] = str(MILES_ROOT)
    os.environ["MODEL_DIR"] = model_dir
    os.environ["DATA_DIR"] = data_dir
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES", "1")
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0,1,2,3")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")

    script_path = MILES_ROOT / "scripts" / "run-qwen3-4B-amd-fsdp.sh"
    if not script_path.exists():
        raise FileNotFoundError(f"找不到 {script_path}")

    print("=" * 60)
    print("启动 FSDP 训练 (scripts/run-qwen3-4B-amd-fsdp.sh)")
    print(f"  MODEL_DIR={model_dir}")
    print(f"  DATA_DIR={data_dir}")
    print("=" * 60)

    subprocess.run(
        ["bash", str(script_path)],
        cwd=str(MILES_ROOT),
        env=os.environ,
    )


if __name__ == "__main__":
    use_fsdp = "--fsdp" in sys.argv
    if "--prepare-only" in sys.argv:
        prepare(need_torch_dist=not use_fsdp)
        print("\n仅下载完成，跳过训练。直接运行可开始训练:")
        print("  python top_amd/run.py" + (" --fsdp" if use_fsdp else ""))
    else:
        prepare(need_torch_dist=not use_fsdp)
        execute_fsdp() if use_fsdp else execute()
