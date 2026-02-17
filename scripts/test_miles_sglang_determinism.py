#!/usr/bin/env python3
"""
向 Miles Router 打请求，测试 SGLang 确定性推理。
用法：Miles 用 --debug-rollout-only 启动后，在另一终端运行此脚本。

  python scripts/test_miles_sglang_determinism.py --host 172.17.0.2 --port 30000 --test-mode single
  # Docker 内 Router 通常监听 172.17.0.2，若 localhost 报 Connection refused 请用 172.17.0.2

参考：sglang.test.test_deterministic，仅保留「看有多少个唯一输出」的逻辑。
"""
import argparse
import random
import requests

PROMPT = "给我介绍下sglang"
# prefix 模式用的长 prompt 片段
LONG_PROMPT_PREFIX = "Tell me about Richard Feynman. " * 50  # ~800 chars


def send_generate(host, port, text, temperature=0.0, sampling_seed=42, max_new_tokens=100):
    """向 router /generate 发单条请求，text 为字符串（Miles Router 要求 text 为 string，非 list）"""
    json_data = {
        "text": text,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "sampling_seed": sampling_seed,
        },
    }
    resp = requests.post(f"http://{host}:{port}/generate", json=json_data, timeout=120)
    if resp.status_code != 200:
        print(f"Error: {resp.status_code} {resp.text[:200]}")
        return None
    ret = resp.json()
    if isinstance(ret, list):
        return ret[0]["text"] if ret else ""
    return ret.get("text", "")


def run_single_mode(host, port, n_trials, temperature=0.0):
    """
    Single 模式：同一 prompt 发送 n_trials 次，每次单条请求。
    确定性时，所有输出应完全相同，Unique=1。
    """
    print(f"\n=== Single 模式 ({n_trials} 次单条请求) ===")
    random.seed(42)
    texts = []
    for i in range(1, n_trials + 1):
        out = send_generate(host, port, PROMPT, temperature=temperature)
        if out is None:
            return None
        t = out.replace("\n", " ") if isinstance(out, str) else str(out)
        texts.append(t)
        print(f"  Trial {i}: {t[:60]}...")

    unique = list(dict.fromkeys(texts))
    n_unique = len(unique)
    print(f"\nTotal: {len(texts)}, Unique: {n_unique}")
    if n_unique == 1:
        print("  -> 通过（所有输出一致）")
    else:
        print(f"  -> 存在 {n_unique} 种不同输出")
        for j, u in enumerate(unique[:5]):
            indices = [k + 1 for k, t in enumerate(texts) if t == u]
            print(f"     [输出{j+1}] 出现在 trial: {indices[:10]}{'...' if len(indices)>10 else ''}")
    return n_unique


def run_prefix_mode(host, port, n_trials, n_start=1, temperature=0.0):
    """
    Prefix 模式：不同长度前缀的 prompt，每个各发若干次单条请求。
    确定性时，相同 prefix 的输出应完全相同。
    """
    print(f"\n=== Prefix 模式 ===")
    len_prefix = [1, 100, 256, 512]
    prompts = [LONG_PROMPT_PREFIX[:l] for l in len_prefix]
    outputs = {i: [] for i in range(len(prompts))}
    # 每种 prefix 各发 n_trials 次
    per_prefix = max(1, n_trials // len(prompts))

    random.seed(42)
    for i in range(len(prompts)):
        for _ in range(per_prefix):
            out = send_generate(host, port, prompts[i], temperature=temperature)
            if out is None:
                return None
            t = out.replace("\n", " ") if isinstance(out, str) else str(out)
            outputs[i].append(t)

    results = []
    for i in range(len(prompts)):
        arr = outputs[i]
        n_unique = len(set(arr))
        print(f"  Prefix len {len_prefix[i]}: Total={len(arr)}, Unique={n_unique}")
        results.append(n_unique)

    passed = all(r == 1 for r in results)
    print(f"\n  -> {'通过' if passed else '存在不同输出'}")
    return results


def main():
    p = argparse.ArgumentParser(description="Miles SGLang 确定性推理测试")
    p.add_argument("--host", default="172.17.0.2", help="Docker 内 Router 通常监听 172.17.0.2")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--test-mode", choices=["single", "prefix"], default="single")
    p.add_argument("--n-trials", type=int, default=15)
    p.add_argument("--n-start", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.5)
    args = p.parse_args()

    if args.test_mode == "single":
        run_single_mode(args.host, args.port, args.n_trials, args.temperature)
    else:
        run_prefix_mode(args.host, args.port, args.n_trials, args.n_start, args.temperature)


if __name__ == "__main__":
    main()
