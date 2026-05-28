"""Standalone transformers AutoConfig patch for DeepSeek-V4.

Yueming's miles@deepseek-v4 uses ``sglang.srt.utils.hf_transformers_utils._load_deepseek_temp_model``
to spoof DSv4 as a v3 config so HF transformers can ``AutoConfig.from_pretrained`` it. The sglang
installed in our image (``/sgl-workspace/sglang``) does NOT yet have DSv4 support, so we
re-implement the same trick inline -- write a tmp config.json with ``model_type=deepseek_v3``
and load that.

Used by ``tools/convert_hf_to_torch_dist.py`` (in yueming's branch). We also use it from the
DSv4 launcher's prepare path.
"""

import json
import logging
import os
import tempfile
from contextlib import contextmanager


logger = logging.getLogger(__name__)

_original_from_pretrained = None


def _load_deepseek_temp_model(model_path: str, architecture: str = "DeepseekV4ForCausalLM", **kwargs):
    """Re-load a deepseek_v4 / deepseek_ref checkpoint as a v3 AutoConfig.

    Writes a sidecar config.json with model_type=deepseek_v3 (which HF transformers does
    recognise) into a process-unique tmp dir and AutoConfig.from_pretrained that.
    """
    from transformers import AutoConfig

    config_file = os.path.join(model_path, "config.json")
    if not os.path.exists(config_file):
        raise RuntimeError(f"transformers_patch: config file missing at {config_file}.")

    with open(config_file) as f:
        config_json = json.load(f)

    config_json["architectures"] = [architecture]
    config_json["model_type"] = "deepseek_v3"

    tmp_root = os.path.join(tempfile.gettempdir(), "_miles_dsv4_temp_cfg")
    os.makedirs(tmp_root, exist_ok=True)
    unique_dir = os.path.join(tmp_root, f"dsv4_{os.getpid()}")
    os.makedirs(unique_dir, exist_ok=True)
    with open(os.path.join(unique_dir, "config.json"), "w") as f:
        json.dump(config_json, f)
    return AutoConfig.from_pretrained(unique_dir, **kwargs)


@contextmanager
def with_transformers_patch():
    apply_transformers_patch()
    try:
        yield
    finally:
        unapply_transformers_patch()


def apply_transformers_patch():
    global _original_from_pretrained
    if _original_from_pretrained is not None:
        return

    from transformers.models.auto.configuration_auto import AutoConfig

    _original_from_pretrained = AutoConfig.from_pretrained

    @classmethod
    def _patched_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers.configuration_utils import PretrainedConfig

        config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get("model_type") in ("deepseek_v4", "deepseek_ref"):
            return _load_deepseek_temp_model(
                pretrained_model_name_or_path,
                architecture="DeepseekV4ForCausalLM",
                **kwargs,
            )

        return _original_from_pretrained.__func__(cls, pretrained_model_name_or_path, **kwargs)

    AutoConfig.from_pretrained = _patched_from_pretrained


def unapply_transformers_patch():
    global _original_from_pretrained
    if _original_from_pretrained is None:
        return

    from transformers.models.auto.configuration_auto import AutoConfig

    AutoConfig.from_pretrained = _original_from_pretrained
    _original_from_pretrained = None
