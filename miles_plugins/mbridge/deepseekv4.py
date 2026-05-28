import copy
from contextlib import contextmanager

from megatron.core.transformer.enums import AttnBackend

from mbridge.core import register_model
from mbridge.models import DeepseekV3Bridge


@register_model("deepseek_v4")
class DeepseekV4Bridge(DeepseekV3Bridge):
    _ATTENTION_MAPPING = DeepseekV3Bridge._ATTENTION_MAPPING.copy()

    _ATTENTION_MAPPING.pop("self_attention.linear_q_up_proj.layer_norm_weight", None)
    _ATTENTION_MAPPING.pop("self_attention.linear_kv_up_proj.layer_norm_weight", None)

    _ATTENTION_MAPPING.update(
        {
            "self_attention.wq_a.weight": ["model.layers.{layer_number}.self_attn.wq_a.weight"],
            "self_attention.q_norm.weight": ["model.layers.{layer_number}.self_attn.q_norm.weight"],
            "self_attention.wq_b.weight": ["model.layers.{layer_number}.self_attn.wq_b.weight"],
            "self_attention.wkv.weight": ["model.layers.{layer_number}.self_attn.wkv.weight"],
            "self_attention.kv_norm.weight": ["model.layers.{layer_number}.self_attn.kv_norm.weight"],
            "self_attention.wo_a.weight": ["model.layers.{layer_number}.self_attn.wo_a.weight"],
            "self_attention.wo_b.weight": ["model.layers.{layer_number}.self_attn.wo_b.weight"],
            "self_attention.attn_sink": ["model.layers.{layer_number}.self_attn.attn_sink"],
            "self_attention.compressor.ape": ["model.layers.{layer_number}.self_attn.compressor.ape"],
            "self_attention.compressor.wkv.weight": ["model.layers.{layer_number}.self_attn.compressor.wkv.weight"],
            "self_attention.compressor.wgate.weight": [
                "model.layers.{layer_number}.self_attn.compressor.wgate.weight"
            ],
            "self_attention.compressor.norm.weight": ["model.layers.{layer_number}.self_attn.compressor.norm.weight"],
            "self_attention.indexer.linear_wq_b.weight": ["model.layers.{layer_number}.self_attn.indexer.wq_b.weight"],
            "self_attention.indexer.linear_weights_proj.weight": [
                "model.layers.{layer_number}.self_attn.indexer.weights_proj.weight"
            ],
            "self_attention.indexer.compressor.ape": ["model.layers.{layer_number}.self_attn.indexer.compressor.ape"],
            "self_attention.indexer.compressor.wkv.weight": [
                "model.layers.{layer_number}.self_attn.indexer.compressor.wkv.weight"
            ],
            "self_attention.indexer.compressor.wgate.weight": [
                "model.layers.{layer_number}.self_attn.indexer.compressor.wgate.weight"
            ],
            "self_attention.indexer.compressor.norm.weight": [
                "model.layers.{layer_number}.self_attn.indexer.compressor.norm.weight"
            ],
        }
    )

    _OTHER_MAPPING = {
        "hc_attn_fn": ["model.layers.{layer_number}.hc_attn_fn"],
        "hc_attn_base": ["model.layers.{layer_number}.hc_attn_base"],
        "hc_attn_scale": ["model.layers.{layer_number}.hc_attn_scale"],
        "hc_ffn_fn": ["model.layers.{layer_number}.hc_ffn_fn"],
        "hc_ffn_base": ["model.layers.{layer_number}.hc_ffn_base"],
        "hc_ffn_scale": ["model.layers.{layer_number}.hc_ffn_scale"],
    }

    _MLP_MAPPING = DeepseekV3Bridge._MLP_MAPPING.copy()
    _MLP_MAPPING.update(
        {
            # Our Pinaster -> HF renamer writes 'mlp.gate.tid2eid' (mirroring the standard
            # HF DeepseekV3 gate naming). Yueming's pre-renamer convention was
            # 'mlp.topk.tid2eid'; we use the HF-canonical name here.
            "mlp.router.tid2eid": ["model.layers.{layer_number}.mlp.gate.tid2eid"],
        }
    )

    _DIRECT_MAPPING = DeepseekV3Bridge._DIRECT_MAPPING.copy()
    _DIRECT_MAPPING.update(
        {
            "decoder.hc_head_params.hc_head_fn": "model.hc_head_fn",
            "decoder.hc_head_params.hc_head_base": "model.hc_head_base",
            "decoder.hc_head_params.hc_head_scale": "model.hc_head_scale",
        }
    )

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        try:
            return super()._weight_name_mapping_mcore_to_hf(mcore_weights_name)
        except NotImplementedError:
            return self._weight_name_mapping_other(mcore_weights_name)

    # ------------------------------------------------------------------
    # Rope-fields shim.
    #
    # Newer transformers (>=4.57) for DeepSeek configs stores rope_theta /
    # rope_scaling NOT as top-level attributes but under ``rope_parameters``
    # (e.g. ``hf_config.rope_parameters["rope_theta"]``). Stock DeepseekV3Bridge
    # accesses ``hf_config.rope_theta`` and ``hf_config.rope_scaling`` directly,
    # which raises ``AttributeError: ... has no attribute 'rope_theta'``.
    #
    # Mirror yueming's DeepseekV32Bridge shim: temporarily inject the
    # top-level rope_theta / rope_scaling fields onto a copy of self.hf_config
    # while calling the parent _build_config / _get_gptmodel_args.
    # ------------------------------------------------------------------
    def _get_rope_theta(self):
        if hasattr(self.hf_config, "rope_parameters") and isinstance(
            self.hf_config.rope_parameters, dict
        ) and "rope_theta" in self.hf_config.rope_parameters:
            return self.hf_config.rope_parameters["rope_theta"]
        return getattr(self.hf_config, "rope_theta", 10000)

    def _normalize_rope_scaling(self, rope_scaling):
        if rope_scaling is None:
            return None
        rope_scaling = dict(rope_scaling)
        rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type")
        if rope_type == "default":
            return None
        if rope_type is not None:
            rope_scaling["type"] = rope_type
        return rope_scaling

    def _get_rope_scaling(self):
        scaling = getattr(self.hf_config, "rope_scaling", None)
        # rope_parameters carries both rope_theta and the scaling spec on
        # transformers>=4.57. If no top-level rope_scaling, fall back to
        # rope_parameters and strip the rope_theta entry.
        if scaling is None and hasattr(self.hf_config, "rope_parameters"):
            params = dict(self.hf_config.rope_parameters or {})
            params.pop("rope_theta", None)
            scaling = params if params else None
        return self._normalize_rope_scaling(scaling)

    def _hf_config_with_rope_fields(self):
        hf_config = copy.copy(self.hf_config)
        hf_config.rope_theta = self._get_rope_theta()
        hf_config.rope_scaling = self._get_rope_scaling()
        return hf_config

    @contextmanager
    def _using_hf_config_with_rope_fields(self):
        original_hf_config = self.hf_config
        self.hf_config = self._hf_config_with_rope_fields()
        try:
            yield
        finally:
            self.hf_config = original_hf_config

    def _build_config(self):
        with self._using_hf_config_with_rope_fields():
            return self._build_config_inner()

    def _get_gptmodel_args(self) -> dict:
        with self._using_hf_config_with_rope_fields():
            return super()._get_gptmodel_args()

    def _build_config_inner(self):
        config = super()._build_config()

        config.attention_backend = AttnBackend.auto

        config.experimental_attention_variant = "dsv4"
        config.dsa_indexer_n_heads = getattr(self.hf_config, "index_n_heads", 64)
        config.dsa_indexer_head_dim = getattr(self.hf_config, "index_head_dim", 128)
        config.dsa_indexer_topk = getattr(self.hf_config, "index_topk", 512)

        config.dsv4_hc_mult = getattr(self.hf_config, "hc_mult", 4)
        config.dsv4_hc_sinkhorn_iters = getattr(self.hf_config, "hc_sinkhorn_iters", 20)
        config.dsv4_hc_eps = getattr(self.hf_config, "hc_eps", 1e-6)

        config.dsv4_compress_ratios = getattr(self.hf_config, "compress_ratios", None)
        config.dsv4_compress_rope_theta = getattr(self.hf_config, "compress_rope_theta", 160000)

        config.dsv4_swiglu_limit = getattr(self.hf_config, "swiglu_limit", 0.0)
        if config.dsv4_swiglu_limit > 0:
            config.bias_activation_fusion = False
            config.activation_func_clamp_value = config.dsv4_swiglu_limit

        config.dsv4_o_groups = getattr(self.hf_config, "o_groups", 8)
        config.dsv4_o_lora_rank = getattr(self.hf_config, "o_lora_rank", 1024)
        config.dsv4_n_hash_layers = getattr(self.hf_config, "n_hash_layers", 3)
        config.dsv4_window_size = getattr(self.hf_config, "window_size", 128)

        return config
