"""Validate the ROCm TE->aiter blockwise FP8 *MoE* (GroupedLinear) path end-to-end.

Exercises the exact runtime path Megatron MoE FP8 training uses under the injector:
TE GroupedLinear + Float8BlockScaling -> split_quantize -> general_grouped_gemm ->
aiter blockwise GEMM, for forward / dgrad / wgrad. Targets the two MoE-backward
fixes: non-128-divisible per-expert token counts (quantize padding) and empty
experts (m_splits[i] == 0).

  A  GroupedLinear fp8 vs bf16 reference (fwd / dgrad / wgrad rel-err) for
     divisible / non-128 / empty-expert m_splits, both fc1- and fc2-shaped.
  B  isolated unit tests for the two fixes (quantize padding round-trip;
     empty-expert wgrad is exactly zero).
  C  gradient actually flows + weights update through the fp8 MoE.
  E  counter guard: the aiter fp8 GEMM path is genuinely taken (not a silent
     bf16 fallback).

Run on an MI355X box:
    python tests/rocm/test_moe_blockwise_fp8_injector.py
or:
    python -m pytest tests/rocm/test_moe_blockwise_fp8_injector.py -s
"""
import importlib
import os
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    getattr(torch.version, "hip", None) is None, reason="ROCm/HIP only"
)

# --- bring up the injector (the same module the sitecustomize hook loads) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_INJ_DIR = os.path.normpath(
    os.path.join(_HERE, "..", "..", "miles", "utils", "te_inject_site")
)
if _INJ_DIR not in sys.path:
    sys.path.insert(0, _INJ_DIR)

import rocm_te_blockwise_inject as _inj  # noqa: E402

assert _inj.apply(), "injector apply() returned False (not ROCm?)"

import transformer_engine.pytorch as te  # noqa: E402
import transformer_engine.pytorch.cpp_extensions.gemm as _gemm_mod  # noqa: E402
from transformer_engine.common.recipe import Float8BlockScaling  # noqa: E402
from transformer_engine.pytorch import GroupedLinear, fp8_autocast  # noqa: E402
from transformer_engine.pytorch.tensor._internal.float8_blockwise_tensor_base import (  # noqa: E402
    Float8BlockwiseQTensorBase,
)

# --- Test E counter: wrap the (already-patched) general_gemm to count fp8-path calls ---
_COUNTS = {"fp8_gemm": 0, "total_gemm": 0}


def _install_counter():
    real = _gemm_mod.general_gemm

    def counting(A, B, *a, **k):
        _COUNTS["total_gemm"] += 1
        if isinstance(A, Float8BlockwiseQTensorBase) and isinstance(
            B, Float8BlockwiseQTensorBase
        ):
            _COUNTS["fp8_gemm"] += 1
        return real(A, B, *a, **k)

    _gemm_mod.general_gemm = counting
    for modname in (
        "transformer_engine.pytorch.module.linear",
        "transformer_engine.pytorch.module.grouped_linear",
    ):
        m = importlib.import_module(modname)
        if getattr(m, "general_gemm", None) is real:
            m.general_gemm = counting


_install_counter()


def _rel(a, b):
    return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-12)).item()


def _build(num_gemms, in_f, out_f, seed):
    torch.manual_seed(seed)
    return GroupedLinear(
        num_gemms, in_f, out_f, bias=False, params_dtype=torch.bfloat16, device="cuda"
    )


def _run_case(num_gemms, in_f, out_f, m_splits, scale=0.1, seed=0):
    """Run identical weights+inputs through the fp8 path and a bf16 reference."""
    M = sum(m_splits)
    gl_fp8 = _build(num_gemms, in_f, out_f, seed)
    gl_ref = _build(num_gemms, in_f, out_f, seed + 1)
    gl_ref.load_state_dict(gl_fp8.state_dict())  # identical weights

    torch.manual_seed(seed + 777)
    x0 = torch.randn(M, in_f, device="cuda", dtype=torch.bfloat16) * scale
    x_fp8 = x0.clone().requires_grad_()
    x_ref = x0.clone().requires_grad_()

    # bf16 reference
    c0 = _COUNTS["fp8_gemm"]
    with fp8_autocast(enabled=False):
        y_ref = gl_ref(x_ref, m_splits)
    y_ref.float().pow(2).mean().backward()
    ref_fp8_calls = _COUNTS["fp8_gemm"] - c0

    # fp8 path
    c1 = _COUNTS["fp8_gemm"]
    with fp8_autocast(enabled=True, fp8_recipe=Float8BlockScaling()):
        y_fp8 = gl_fp8(x_fp8, m_splits)
    y_fp8.float().pow(2).mean().backward()
    fp8_calls = _COUNTS["fp8_gemm"] - c1

    return {
        "y_fp8": y_fp8, "y_ref": y_ref, "x_fp8": x_fp8, "x_ref": x_ref,
        "gl_fp8": gl_fp8, "gl_ref": gl_ref, "m_splits": m_splits,
        "fp8_calls": fp8_calls, "ref_fp8_calls": ref_fp8_calls,
    }


# tolerances for blockwise e4m3 (per-expert grouped GEMM)
TOL_FWD, TOL_DX, TOL_DW = 0.06, 0.12, 0.12

_CASES = [
    ("fc1/divisible", 4, 2048, 1536, [128, 128, 128, 128]),
    ("fc1/non-128",   4, 2048, 1536, [23, 105, 200, 184]),
    ("fc1/empty-mid", 4, 2048, 1536, [200, 0, 170, 142]),
    ("fc1/empty-1st", 4, 2048, 1536, [0, 200, 170, 142]),
    ("fc2/non-128",   4, 768,  2048, [37, 91, 256, 128]),
]


def _check_case(name, num_gemms, in_f, out_f, m_splits):
    r = _run_case(num_gemms, in_f, out_f, m_splits)
    assert torch.isfinite(r["y_fp8"]).all(), f"{name}: non-finite fwd"
    assert torch.isfinite(r["x_fp8"].grad).all(), f"{name}: non-finite dgrad"

    fwd = _rel(r["y_fp8"], r["y_ref"])
    dx = _rel(r["x_fp8"].grad, r["x_ref"].grad)

    dws = []
    for i in range(num_gemms):
        g_fp8 = getattr(r["gl_fp8"], f"weight{i}").grad
        g_ref = getattr(r["gl_ref"], f"weight{i}").grad
        assert torch.isfinite(g_fp8).all(), f"{name}: non-finite wgrad expert {i}"
        if m_splits[i] == 0:
            # empty expert: zero tokens -> wgrad must be exactly zero (both paths)
            assert g_fp8.abs().max().item() == 0.0, (
                f"{name}: empty expert {i} fp8 wgrad not zero "
                f"({g_fp8.abs().max().item()})"
            )
            assert g_ref.abs().max().item() == 0.0
            dws.append(0.0)
        else:
            dws.append(_rel(g_fp8, g_ref))
    dw = max(dws)

    print(
        f"[A] {name:16s} m={m_splits} fwd={fwd:.4f} dX={dx:.4f} "
        f"dW={dw:.4f} fp8_gemm_calls={r['fp8_calls']} (ref={r['ref_fp8_calls']})"
    )
    assert fwd < TOL_FWD, f"{name}: fwd rel-err {fwd:.4f} >= {TOL_FWD}"
    assert dx < TOL_DX, f"{name}: dgrad rel-err {dx:.4f} >= {TOL_DX}"
    assert dw < TOL_DW, f"{name}: wgrad rel-err {dw:.4f} >= {TOL_DW}"
    # E (guard): fp8 path genuinely taken; bf16 ref takes none.
    assert r["fp8_calls"] > 0, f"{name}: NO fp8 GEMM calls -> silent bf16 fallback!"
    assert r["ref_fp8_calls"] == 0, f"{name}: bf16 ref unexpectedly hit fp8 path"
    return r


@pytest.mark.parametrize("case", _CASES, ids=[c[0] for c in _CASES])
def test_A_grouped_linear_fp8_vs_bf16(case):
    _check_case(*case)


def test_B1_quantize_padding_roundtrip():
    """quantize_1x128 on a non-128-divisible K: real entries dequant within FP8 tol,
    scale gains exactly one padded block."""
    from rocm_te_blockwise_inject import BLK, _aiter_bits, _from_uint8, quantize_1x128

    _, e4m3, _ = _aiter_bits()
    M, K = 5, 200  # K not divisible by 128
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5
    xq, scale = quantize_1x128(x)
    import math
    Kp = math.ceil(K / BLK) * BLK
    assert xq.shape == (M, Kp), (xq.shape, (M, Kp))
    assert scale.shape == (M, Kp // BLK), scale.shape
    # dequantize the real (first K) columns and compare
    data = _from_uint8(xq, e4m3).float()  # [M, Kp]
    deq = (data.view(M, Kp // BLK, BLK) * scale.view(M, Kp // BLK, 1)).view(M, Kp)
    deq_real = deq[:, :K]
    rel = _rel(deq_real, x)
    # padded tail must be exactly zero
    pad_max = deq[:, K:].abs().max().item()
    print(f"[B1] padding roundtrip K={K} Kp={Kp} rel={rel:.4f} pad_max={pad_max}")
    assert rel < 0.06, rel
    assert pad_max == 0.0, pad_max


def test_B2_empty_expert_wgrad_zero():
    """An expert that receives 0 tokens must produce exactly-zero wgrad and not crash."""
    r = _run_case(4, 2048, 1536, [0, 200, 170, 142])
    g0 = getattr(r["gl_fp8"], "weight0").grad  # empty expert
    g1 = getattr(r["gl_fp8"], "weight1").grad  # non-empty
    print(
        f"[B2] empty wgrad max={g0.abs().max().item()} "
        f"non-empty wgrad max={g1.abs().max().item()}"
    )
    assert g0.abs().max().item() == 0.0
    assert g1.abs().max().item() > 0.0  # real experts DO get gradient


def test_C_gradients_flow_and_weights_update():
    """Full autograd through the fp8 MoE updates weights (grad_norm finite & > 0)."""
    num_gemms, in_f, out_f = 4, 2048, 1536
    m_splits = [128, 96, 64, 224]
    gl = _build(num_gemms, in_f, out_f, seed=3)
    torch.manual_seed(99)
    x = (torch.randn(sum(m_splits), in_f, device="cuda", dtype=torch.bfloat16) * 0.1)
    opt = torch.optim.SGD(gl.parameters(), lr=1e-2)

    w_before = [getattr(gl, f"weight{i}").detach().float().clone() for i in range(num_gemms)]
    opt.zero_grad()
    with fp8_autocast(enabled=True, fp8_recipe=Float8BlockScaling()):
        y = gl(x, m_splits)
    loss = y.float().pow(2).mean()
    loss.backward()

    sq = 0.0
    for i in range(num_gemms):
        g = getattr(gl, f"weight{i}").grad
        assert g is not None and torch.isfinite(g).all()
        sq += g.float().pow(2).sum().item()
    grad_norm = sq ** 0.5
    opt.step()

    deltas = [
        (getattr(gl, f"weight{i}").detach().float() - w_before[i]).norm().item()
        for i in range(num_gemms)
    ]
    print(f"[C] loss={loss.item():.6f} grad_norm={grad_norm:.6f} weight_deltas={['%.2e'%d for d in deltas]}")
    assert grad_norm > 0.0 and torch.isfinite(torch.tensor(grad_norm))
    assert all(d > 0.0 for d in deltas), f"some weights did not update: {deltas}"


def test_E_fp8_path_is_taken():
    """Sanity: the fp8 case routes through the aiter fp8 GEMM; bf16 does not."""
    r = _run_case(4, 2048, 1536, [128, 128, 128, 128])
    print(f"[E] fp8_gemm_calls={r['fp8_calls']} ref_fp8_calls={r['ref_fp8_calls']}")
    assert r["fp8_calls"] > 0
    assert r["ref_fp8_calls"] == 0


if __name__ == "__main__":
    print(f"injector applied; gate={te.fp8.check_fp8_block_scaling_support()}")
    print("== Test A: GroupedLinear fp8 vs bf16 ==")
    for c in _CASES:
        _check_case(*c)
    print("== Test B1: padding roundtrip =="); test_B1_quantize_padding_roundtrip()
    print("== Test B2: empty-expert wgrad zero =="); test_B2_empty_expert_wgrad_zero()
    print("== Test C: grads flow + weights update =="); test_C_gradients_flow_and_weights_update()
    print("== Test E: fp8 path taken =="); test_E_fp8_path_is_taken()
    print("\nRESULT: ALL PASS")
