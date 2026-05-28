"""Fake rollout function for SMOKE-ONLY runs.

Bypasses sglang entirely: pulls prompts from the data source, fills in dummy
responses (a few random tokens each) and dummy rewards. This is enough to:

  - exercise the full Megatron training step (model build, forward, backward, optimizer)
  - exercise the data pipeline (tokens, masks, packed seq params)
  - exercise FP8 quant/GEMM/dequant paths

Do NOT use for convergence runs -- losses are meaningless with random "responses".

Wire via ``--rollout-function-path miles.rollout.fake_rollout.fake_generate_rollout``.
The custom rollout is invoked by ``rollout.py`` exactly like the real one
(same signature, same return type). Eval is a no-op.
"""

from __future__ import annotations

import random
from argparse import Namespace
from typing import Any

from miles.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.utils.types import Sample


def fake_generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    if evaluation:
        # No eval data; rollout.py.eval() already guards with debug_train_only.
        return RolloutFnEvalOutput(data={}, metrics={})

    rng = random.Random(rollout_id)

    # Pull a batch of prompts from the data source (this works without sglang).
    target_num = args.rollout_batch_size * args.n_samples_per_prompt
    samples = []
    while len(samples) < target_num:
        batch = data_source.get_samples(args.rollout_batch_size)
        for proto in batch:
            for _ in range(args.n_samples_per_prompt):
                # Copy the prompt fields, fake a short response.
                if hasattr(proto, "to_dict"):
                    s = Sample.from_dict(proto.to_dict())
                else:
                    # last resort: shallow copy
                    s = proto
                # Vocab id range: assume 0..127999 is safe for any DeepSeek tokenizer.
                resp_len = min(16, max(4, args.rollout_max_response_len // 4))
                fake_tokens = [rng.randint(1000, 10000) for _ in range(resp_len)]
                s.tokens = list(s.tokens) + fake_tokens
                s.response = "fake response"
                s.response_length = resp_len
                s.reward = float(rng.random())
                s.loss_mask = [0] * (len(s.tokens) - resp_len) + [1] * resp_len
                s.status = Sample.Status.COMPLETED
                # Ensure index/group_index are set so downstream don't crash.
                if s.index is None:
                    s.index = len(samples)
                if s.group_index is None:
                    s.group_index = len(samples) // args.n_samples_per_prompt
                samples.append(s)
                if len(samples) >= target_num:
                    break
            if len(samples) >= target_num:
                break

    metrics: dict[str, Any] = {
        "fake_rollout": True,
        "fake_rollout_id": rollout_id,
        "fake_rollout_size": len(samples),
    }
    return RolloutFnTrainOutput(samples=samples, metrics=metrics)
