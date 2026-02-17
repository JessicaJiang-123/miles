#!/usr/bin/env python3
"""
向 Miles Router 打请求，测试 SGLang 确定性推理。
用法：Miles 用 --debug-rollout-only 启动后，在另一终端运行此脚本。

  python scripts/test_miles_sglang_determinism.py --host localhost --port 30000 --test-mode single
  python scripts/test_miles_sglang_determinism.py --host localhost --port 30000 --test-mode prefix

参考：sglang.test.test_deterministic，仅保留「看有多少个唯一输出」的逻辑。
"""
import argparse
import random
import requests

PROMPT = "给我介绍下sglang"
# prefix 模式用的长 prompt 片段
LONG_PROMPT_PREFIX = "Tell me about Richard Feynman. " * 50  # ~800 chars


def send_generate(host, port, text_list, temperature=0.0, sampling_seed=42, max_new_tokens=100):
    """向 router /generate 发请求，text_list 为 prompt 列表"""
    json_data = {
        "text": text_list,
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
        return [r["text"] for r in ret]
    return [ret["text"]]


def run_single_mode(host, port, n_trials, temperature=0.0):
    """
    Single 模式：同一 prompt，batch_size 从 1 到 n_trials 各跑一次。
    确定性时，所有输出应完全相同，Unique=1。
    """
    print(f"\n=== Single 模式 (batch_size 1..{n_trials}) ===")
    random.seed(42)
    texts = []
    for bs in range(1, n_trials + 1):
        out = send_generate(host, port, [PROMPT] * bs, temperature=temperature)
        if out is None:
            return None
        t = out[0].replace("\n", " ") if out else ""
        texts.append(t)
        print(f"  Trial {bs} (batch_size={bs}): {t[:60]}...")

    unique = list(dict.fromkeys(texts))
    n_unique = len(unique)
    print(f"\nTotal: {len(texts)}, Unique: {n_unique}")
    if n_unique == 1:
        print("  -> 通过（所有输出一致）")
    else:
        print(f"  -> 存在 {n_unique} 种不同输出")
        for i, u in enumerate(unique[:5]):
            indices = [j + 1 for j, t in enumerate(texts) if t == u]
            print(f"     [输出{i+1}] 出现在 trial: {indices[:10]}{'...' if len(indices)>10 else ''}")
    return n_unique


def run_prefix_mode(host, port, n_trials, n_start=1, temperature=0.0):
    """
    Prefix 模式：不同长度前缀的 prompt，混在一个 batch 里多次发送。
    确定性时，相同 prefix 的输出应完全相同。
    """
    print(f"\n=== Prefix 模式 (batch_size {n_start}..{n_start+n_trials-1}) ===")
    len_prefix = [1, 100, 256, 512]
    prompts = [LONG_PROMPT_PREFIX[:l] for l in len_prefix]
    outputs = {i: [] for i in range(len(prompts))}

    random.seed(42)
    for i in range(n_start, n_start + n_trials):
        batch_size = i
        sampled = [random.randint(0, len(prompts) - 1) for _ in range(batch_size)]
        batch_prompts = [prompts[s] for s in sampled]

        out = send_generate(host, port, batch_prompts, temperature=temperature)
        if out is None:
            return None

        for idx, s in enumerate(sampled):
            outputs[s].append(out[idx])

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
    p.add_argument("--host", default="localhost")
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
