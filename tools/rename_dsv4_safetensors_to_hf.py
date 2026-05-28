"""Rename DSv4 (Pinaster/sgl-project repackage) safetensors keys to the standard HF
DeepseekV3 layout that mbridge's DeepseekV3Bridge / our DeepseekV4Bridge expects.

The Pinaster mirror saves the deepseek-ai/DeepSeek-V4-Flash weights under their internal
short names (e.g. ``layers.0.attn.wq_a.weight``, ``layers.0.ffn.experts.0.w1.weight``,
``embed.weight``, ``norm.weight``, ``head.weight``). mbridge expects the standard
DeepseekV3ForCausalLM layout (``model.layers.0.self_attn.wq_a.weight``,
``model.layers.0.mlp.experts.0.gate_proj.weight``, ``model.embed_tokens.weight``,
``model.norm.weight``, ``lm_head.weight``).

Renaming rules (applied to every key; multiple substitutions are independent):

  embed.weight              -> model.embed_tokens.weight
  norm.weight               -> model.norm.weight
  head.weight               -> lm_head.weight
  hc_head_{fn,base,scale}   -> model.hc_head_{fn,base,scale}
  layers.X.attn_norm.weight -> model.layers.X.input_layernorm.weight
  layers.X.ffn_norm.weight  -> model.layers.X.post_attention_layernorm.weight
  layers.X.hc_*             -> model.layers.X.hc_*
  layers.X.attn.q_norm.weight        -> model.layers.X.self_attn.q_norm.weight
  layers.X.attn.kv_norm.weight       -> model.layers.X.self_attn.kv_norm.weight
  layers.X.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.{weight,scale}
      -> model.layers.X.self_attn.<same>
  layers.X.attn.attn_sink   -> model.layers.X.self_attn.attn_sink
  layers.X.attn.compressor.*  -> model.layers.X.self_attn.compressor.*
  layers.X.attn.indexer.*     -> model.layers.X.self_attn.indexer.*
  layers.X.ffn.gate.weight  -> model.layers.X.mlp.gate.weight
  layers.X.ffn.gate.bias    -> model.layers.X.mlp.gate.e_score_correction_bias
  layers.X.ffn.gate.tid2eid -> model.layers.X.mlp.gate.tid2eid    (dropped pre-load by mbridge if unmapped)
  layers.X.ffn.experts.E.w1.weight -> model.layers.X.mlp.experts.E.gate_proj.weight
  layers.X.ffn.experts.E.w3.weight -> model.layers.X.mlp.experts.E.up_proj.weight
  layers.X.ffn.experts.E.w2.weight -> model.layers.X.mlp.experts.E.down_proj.weight
  layers.X.ffn.shared_experts.w1.weight -> model.layers.X.mlp.shared_experts.gate_proj.weight
  layers.X.ffn.shared_experts.w3.weight -> model.layers.X.mlp.shared_experts.up_proj.weight
  layers.X.ffn.shared_experts.w2.weight -> model.layers.X.mlp.shared_experts.down_proj.weight

Run in-place: re-saves each shard's safetensors with renamed keys and rewrites the
index json. If --out-dir is given, writes to a sibling tree instead (uses 2x disk).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from glob import glob

from safetensors.torch import load_file, save_file


_RE_LAYER = re.compile(r"^layers\.(\d+)\.(.*)$")


def _rename_key(k: str) -> str:
    # top-level renames
    if k == "embed.weight":
        return "model.embed_tokens.weight"
    if k == "norm.weight":
        return "model.norm.weight"
    if k == "head.weight":
        return "lm_head.weight"
    if k in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        return f"model.{k}"

    m = _RE_LAYER.match(k)
    if m is None:
        return k  # unknown top-level, leave alone
    layer = m.group(1)
    rest = m.group(2)

    prefix = f"model.layers.{layer}"

    # layer-level norms
    if rest == "attn_norm.weight":
        return f"{prefix}.input_layernorm.weight"
    if rest == "ffn_norm.weight":
        return f"{prefix}.post_attention_layernorm.weight"

    # hyper-connection per-layer parameters
    if rest.startswith("hc_"):
        return f"{prefix}.{rest}"

    # attention
    if rest.startswith("attn."):
        sub = rest[len("attn."):]
        # compressor and indexer subtrees retain their structure under self_attn.
        return f"{prefix}.self_attn.{sub}"

    # MoE / shared experts / router
    if rest.startswith("ffn."):
        sub = rest[len("ffn."):]
        if sub == "gate.weight":
            return f"{prefix}.mlp.gate.weight"
        if sub == "gate.bias":
            return f"{prefix}.mlp.gate.e_score_correction_bias"
        if sub == "gate.tid2eid":
            return f"{prefix}.mlp.gate.tid2eid"

        # experts.E.wN.{weight|scale}  ->  experts.E.{gate|down|up}_proj.{weight|scale}
        m2 = re.match(r"^experts\.(\d+)\.w([123])(\..*)?$", sub)
        if m2:
            eid, wn, tail = m2.group(1), m2.group(2), m2.group(3) or ""
            proj = {"1": "gate_proj", "2": "down_proj", "3": "up_proj"}[wn]
            return f"{prefix}.mlp.experts.{eid}.{proj}{tail}"
        m3 = re.match(r"^shared_experts\.w([123])(\..*)?$", sub)
        if m3:
            wn, tail = m3.group(1), m3.group(2) or ""
            proj = {"1": "gate_proj", "2": "down_proj", "3": "up_proj"}[wn]
            return f"{prefix}.mlp.shared_experts.{proj}{tail}"
        return f"{prefix}.mlp.{sub}"

    # default: keep the layer prefix.
    return f"{prefix}.{rest}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True)
    p.add_argument("--out-dir", default=None,
                   help="Where to write the renamed files. If omitted, rewrite in-place.")
    args = p.parse_args()

    src = args.in_dir
    dst = args.out_dir or src
    if dst != src:
        os.makedirs(dst, exist_ok=True)
        # copy non-safetensors metadata over.
        for f in os.listdir(src):
            if f.endswith(".safetensors"):
                continue
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))

    index_path = os.path.join(src, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    new_weight_map = {}
    for old_key, fname in index["weight_map"].items():
        new_weight_map[_rename_key(old_key)] = fname
    index["weight_map"] = new_weight_map
    out_index = os.path.join(dst, "model.safetensors.index.json")
    with open(out_index, "w") as f:
        json.dump(index, f, indent=2)
    print(f"[rename] wrote new index: {out_index}")

    shard_files = sorted(glob(os.path.join(src, "*.safetensors")))
    for sf in shard_files:
        fname = os.path.basename(sf)
        print(f"[rename] {fname}")
        state = load_file(sf, device="cpu")
        new_state = {_rename_key(k): v for k, v in state.items()}
        out_path = os.path.join(dst, fname)
        save_file(new_state, out_path)
        del state, new_state


if __name__ == "__main__":
    main()
