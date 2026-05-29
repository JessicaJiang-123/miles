from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from miles.utils.replay_base import (
    BaseReplayManager,
    IndexerReplayManager,
    RoutingReplayManager,
)


def _register_replay_list_moe(replay_list, replay_data, models):
    layer_indices = []
    replay_idx = 0
    for vp_stage, model in enumerate(models):
        config = model.module.config
        num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        for layer_id in range(offset, offset + num_layers_to_build):
            if isinstance(config.moe_layer_freq, int):
                if layer_id % config.moe_layer_freq != 0:
                    continue
            elif isinstance(config.moe_layer_freq, list):
                assert len(config.moe_layer_freq) == config.num_layers
                if config.moe_layer_freq[layer_id] == 0:
                    continue
            layer_indices.append(layer_id)

    for replay_idx, layer_idx in enumerate(layer_indices):
        layer_data = replay_data[:, layer_idx]
        replay_list[replay_idx].record(layer_data)


def _register_replay_list_indexer(replay_list, replay_data, models):
    """Indexer-side analogue of _register_replay_list_moe.

    DSv4 instantiates V4Indexer on every layer whose dsv4_compress_ratios entry
    equals 4 (the C4 path). The rollout-side capturer (sglang's C4Indexer ->
    state_capturer.indexer_topk) writes one row per c4 layer in the SAME order
    the indexer modules were constructed (which is the natural layer iteration
    order). Mirror that ordering here so replay slot i feeds C4 layer i.
    """
    layer_indices = []
    for vp_stage, model in enumerate(models):
        config = model.module.config
        compress_ratios = getattr(config, "dsv4_compress_ratios", None)
        if compress_ratios is None:
            raise ValueError(
                "IndexerReplayManager requires config.dsv4_compress_ratios (DSv4 only)"
            )
        num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        for layer_id in range(offset, offset + num_layers_to_build):
            if compress_ratios[layer_id] == 4:
                layer_indices.append(layer_id)

    for replay_idx, layer_idx in enumerate(layer_indices):
        # replay_data shape after _fill_replay_data is (n_tokens, n_c4_layers, topk).
        # replay_idx already indexes into the C4-only axis, NOT the full layer
        # axis -- so we use replay_idx for the middle dim, not layer_idx.
        layer_data = replay_data[:, replay_idx]
        replay_list[replay_idx].record(layer_data)


def get_register_replay_list_func(manager: BaseReplayManager):
    if isinstance(manager, RoutingReplayManager):
        return _register_replay_list_moe
    if isinstance(manager, IndexerReplayManager):
        return _register_replay_list_indexer
    raise ValueError(f"Unsupported manager type: {type(manager)}")
