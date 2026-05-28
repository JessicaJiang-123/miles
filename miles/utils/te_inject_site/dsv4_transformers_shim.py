"""Compat shim: backfill ``rope_theta`` onto HF configs loaded for DeepSeek v3/v4.

transformers 5.x reorganised RoPE config: instead of a flat ``rope_theta`` attribute, the
canonical home is ``config.rope_parameters["rope_theta"]`` (or, when ``rope_scaling`` is
present, ``config.rope_parameters["rope_theta"]`` plus the scaling kwargs). The DSv4-Flash-FP8
4-layer checkpoint we use is FROM HF transformers 4.57 -- it has ``rope_theta`` at the top
level. transformers 5.3 silently migrates that into ``rope_parameters`` and then *drops*
``rope_theta``, so ``cfg.rope_theta`` raises AttributeError.

yueming-sglang's ``deepseek_v4.py`` accesses ``config.rope_theta`` directly (and indexer
code accesses ``config.compress_rope_theta``); we paper over the migration by reattaching
``rope_theta`` from ``rope_parameters`` whenever AutoConfig.from_pretrained returns a
DeepSeek-family config.

Gated by env var ``MILES_DSV4_TRANSFORMERS_SHIM=1``.
"""
from __future__ import annotations

import os


_APPLIED = False


def apply():
    """Patch transformers.AutoConfig.from_pretrained to backfill ``rope_theta``."""
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    try:
        import transformers
        from transformers.models.auto.configuration_auto import AutoConfig
    except Exception as exc:  # pragma: no cover
        print(f"[dsv4-transformers-shim] transformers import failed, skipping: {exc}", flush=True)
        return

    _orig = AutoConfig.from_pretrained  # this is a classmethod; .__func__ is the raw fn

    def _backfill(cfg):
        """If cfg has rope_parameters but no rope_theta, copy it over."""
        if cfg is None:
            return cfg
        rope_parameters = getattr(cfg, "rope_parameters", None)
        if rope_parameters is None:
            return cfg
        rope_theta = None
        if isinstance(rope_parameters, dict):
            rope_theta = rope_parameters.get("rope_theta")
        else:
            rope_theta = getattr(rope_parameters, "rope_theta", None)
        if rope_theta is not None and not hasattr(cfg, "rope_theta"):
            try:
                setattr(cfg, "rope_theta", rope_theta)
            except Exception:
                pass
        # Recurse into common sub-configs (text_config etc) for completeness.
        for sub in ("text_config", "llm_config", "language_config", "thinker_config"):
            child = getattr(cfg, sub, None)
            if child is not None and not isinstance(child, (str, int, float, bool, list, dict, tuple)):
                _backfill(child)
        return cfg

    @classmethod
    def _patched(cls, pretrained_model_name_or_path, *args, **kwargs):
        cfg = _orig.__func__(cls, pretrained_model_name_or_path, *args, **kwargs)
        try:
            return _backfill(cfg)
        except Exception as exc:  # pragma: no cover
            print(f"[dsv4-transformers-shim] backfill failed for {pretrained_model_name_or_path}: {exc}", flush=True)
            return cfg

    AutoConfig.from_pretrained = _patched
    print("[dsv4-transformers-shim] AutoConfig.from_pretrained patched to backfill rope_theta", flush=True)


if os.environ.get("MILES_DSV4_TRANSFORMERS_SHIM", "0") == "1":
    try:
        apply()
    except Exception as _e:  # pragma: no cover
        print(f"[dsv4-transformers-shim] apply() failed: {_e}", flush=True)
