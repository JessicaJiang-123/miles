"""Compat shims for running yueming-sglang DSv4 against ROCm/AMD MI355X.

Two Python-layer fixes, both gated by ``MILES_DSV4_TRANSFORMERS_SHIM=1``:

(1) Backfill ``rope_theta`` on AutoConfig load.
    transformers 5.x reorganised RoPE config: instead of a flat ``rope_theta`` attribute,
    the canonical home is ``config.rope_parameters["rope_theta"]``. The DSv4-Flash-FP8
    4-layer checkpoint is FROM HF transformers 4.57 -- it has ``rope_theta`` at the top
    level. transformers 5.3 silently migrates that into ``rope_parameters`` and then
    *drops* ``rope_theta``, so ``cfg.rope_theta`` raises AttributeError.

    yueming-sglang's ``deepseek_v4.py`` accesses ``config.rope_theta`` directly (and
    indexer code accesses ``config.compress_rope_theta``); we paper over the migration
    by reattaching ``rope_theta`` from ``rope_parameters`` whenever
    AutoConfig.from_pretrained returns a DeepSeek-family config.

(2) Disable PDL (Programmatic Dependent Launch) on ROCm.
    ``sglang.jit_kernel.utils.is_arch_support_pdl()`` returns True when
    ``torch.cuda.get_device_capability(0)[0] >= 9``. On MI355X PyTorch reports (9, 5)
    via the HIP runtime, so PDL gets enabled. But PDL is an NVIDIA-only CUDA Graph
    feature -- the underlying HIP graph mutator fails:

      Capture cuda graph failed: Check failed:
      (!mutator.has_trigger_launch_ && !mutator.has_grid_sync_) is false: PDL is not supported

    Override to force False on ROCm (detected via ``torch.version.hip`` not None).

IMPORTANT: All status output goes to **stderr**, not stdout. The sitecustomize hook fires
whenever the Python interpreter starts -- including for child python scripts whose stdout
the parent captures (e.g. ``/opt/rocm/bin/rocm_agent_enumerator`` is a Python script, and
tvm_ffi's ROCm-arch detection parses its stdout line-by-line; a stray ``print()`` ended up
as ``--offload-arch=[dsv4-transformers-shim] AutoConfig.from_pretrained ...`` baked into
build.ninja, breaking the cuda graph capture).
"""
from __future__ import annotations

import os
import sys


_APPLIED = False
_PRINT_PREFIX = "[dsv4-transformers-shim]"


def _log(msg: str) -> None:
    # ALWAYS stderr -- never stdout, see module docstring.
    print(f"{_PRINT_PREFIX} {msg}", file=sys.stderr, flush=True)


def apply():
    """Patch transformers.AutoConfig.from_pretrained to backfill ``rope_theta``."""
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    try:
        from transformers.models.auto.configuration_auto import AutoConfig
    except Exception as exc:  # pragma: no cover
        _log(f"transformers import failed, skipping: {exc}")
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
            _log(f"backfill failed for {pretrained_model_name_or_path}: {exc}")
            return cfg

    AutoConfig.from_pretrained = _patched
    _log("AutoConfig.from_pretrained patched to backfill rope_theta")

    _arm_pdl_shim_on_rocm()


def _arm_pdl_shim_on_rocm():
    """Arm a meta-path hook that forces ``is_arch_support_pdl() == False`` on ROCm."""
    try:
        import torch
    except Exception:
        return
    if getattr(torch.version, "hip", None) is None:
        return  # CUDA build: leave PDL alone

    # Force TRTLLM_ENABLE_PDL=0 (consulted in sglang.srt.entrypoints.engine), works regardless.
    os.environ["TRTLLM_ENABLE_PDL"] = "0"

    import importlib.util as _u
    from importlib.abc import MetaPathFinder, Loader

    _TARGET = "sglang.jit_kernel.utils"

    def _patch(mod):
        try:
            mod.is_arch_support_pdl = lambda: False
            _log("sglang.jit_kernel.utils.is_arch_support_pdl forced to False (ROCm)")
        except Exception as exc:  # pragma: no cover
            _log(f"PDL patch failed: {exc}")

    if _TARGET in sys.modules:
        _patch(sys.modules[_TARGET])
        return

    class _WrapLoader(Loader):
        def __init__(self, inner):
            self._inner = inner

        def create_module(self, spec):
            return self._inner.create_module(spec)

        def exec_module(self, module):
            self._inner.exec_module(module)
            try:
                _patch(module)
            except Exception as e:  # pragma: no cover
                _log(f"PDL wrap failed: {e}")

    class _Finder(MetaPathFinder):
        _busy = False

        def find_spec(self, fullname, path=None, target=None):
            if fullname != _TARGET or self._busy:
                return None
            self._busy = True
            try:
                spec = _u.find_spec(fullname)
            finally:
                self._busy = False
            if spec is None or spec.loader is None:
                return None
            spec.loader = _WrapLoader(spec.loader)
            return spec

    sys.meta_path.insert(0, _Finder())


if os.environ.get("MILES_DSV4_TRANSFORMERS_SHIM", "0") == "1":
    try:
        apply()
    except Exception as _e:  # pragma: no cover
        _log(f"apply() failed: {_e}")
