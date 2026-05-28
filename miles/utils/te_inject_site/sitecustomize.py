"""Auto-run at interpreter startup (when this dir is on PYTHONPATH).

The instant `transformer_engine.pytorch` finishes importing, route DeepSeek blockwise FP8
(Float8BlockScaling) through aiter by calling rocm_te_blockwise_inject.apply(). This
guarantees the patch is live in every Megatron Ray worker before any TE module is built,
without touching the installed /root/miles.

Mechanism: a MetaPathFinder placed at the FRONT of sys.meta_path intercepts the import of
`transformer_engine.pytorch`, delegates to the normal finders to build the spec, wraps the
spec's loader so that apply() runs immediately after the module's exec_module completes.

Gated by env var ROCM_TE_BLOCKWISE_INJECT=1 (no-op otherwise).
"""
import os
import sys

if os.environ.get("ROCM_TE_BLOCKWISE_INJECT", "0") == "1":
    import importlib.util
    from importlib.abc import MetaPathFinder, Loader

    _DIR = os.path.dirname(os.path.abspath(__file__))
    _TARGET = "transformer_engine.pytorch"

    def _apply_once():
        if getattr(_apply_once, "done", False):
            return
        path = os.path.join(_DIR, "rocm_te_blockwise_inject.py")
        spec = importlib.util.spec_from_file_location("rocm_te_blockwise_inject", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.apply()
        _apply_once.done = True

    class _WrapLoader(Loader):
        def __init__(self, inner):
            self._inner = inner

        def create_module(self, spec):
            return self._inner.create_module(spec)

        def exec_module(self, module):
            self._inner.exec_module(module)
            try:
                _apply_once()
            except Exception as e:  # pragma: no cover
                print(f"[sitecustomize] rocm_te_blockwise inject failed: {e}", flush=True)

    class _Finder(MetaPathFinder):
        _busy = False

        def find_spec(self, fullname, path=None, target=None):
            if fullname != _TARGET or self._busy:
                return None
            # Avoid recursion: ask the OTHER finders for the real spec.
            self._busy = True
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                self._busy = False
            if spec is None or spec.loader is None:
                return None
            spec.loader = _WrapLoader(spec.loader)
            return spec

    # If TE.pytorch is somehow already imported, patch right away.
    if _TARGET in sys.modules:
        try:
            _apply_once()
        except Exception as e:
            print(f"[sitecustomize] rocm_te_blockwise inject (eager) failed: {e}", flush=True)
    else:
        sys.meta_path.insert(0, _Finder())
