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

    def _fake_one(proto: Sample) -> Sample:
        if hasattr(proto, "to_dict"):
            s = Sample.from_dict(proto.to_dict())
        else:
            s = proto
        # Need response_length >= TP-size * micro-batch-size so per-rank sequence-parallel
        # slice is non-empty. The qwen3 smoke uses 100, and DSv4 smoke launches with TP=8.
        resp_len = max(args.rollout_max_response_len // 2, 64)
        fake_tokens = [rng.randint(1000, 10000) for _ in range(resp_len)]
        s.tokens = list(s.tokens) + fake_tokens
        s.response = "fake response"
        s.response_length = resp_len
        s.reward = float(rng.random())
        s.loss_mask = [0] * (len(s.tokens) - resp_len) + [1] * resp_len
        s.status = Sample.Status.COMPLETED
        # Fake rollout log probs (one per response token). Needed by the GRPO loss path
        # (TIS / importance-sampling weight = exp(log_p_train - log_p_rollout)).
        # We don't have a real sampling distribution, so just zeros (i.e. "uniform" logprob).
        s.rollout_log_probs = [0.0] * resp_len
        return s

    # data_source.get_samples returns list[list[Sample]] (one inner list per prompt-group,
    # with n_samples_per_prompt entries each).
    target_groups = args.rollout_batch_size
    groups: list[list[Sample]] = []
    while len(groups) < target_groups:
        batch = data_source.get_samples(target_groups - len(groups))
        for group in batch:
            faked_group = [_fake_one(s) for s in group]
            for j, s in enumerate(faked_group):
                if s.group_index is None:
                    s.group_index = len(groups)
                if s.index is None:
                    s.index = j
            groups.append(faked_group)
            if len(groups) >= target_groups:
                break

    metrics: dict[str, Any] = {
        "fake_rollout": True,
        "fake_rollout_id": rollout_id,
        "fake_rollout_groups": len(groups),
        "fake_rollout_total_samples": sum(len(g) for g in groups),
    }
    return RolloutFnTrainOutput(samples=groups, metrics=metrics)
