"""Validate the ROCm blockwise FP8 linear (aiter) vs a bf16 reference on gfx950.

Run on an MI355X box with aiter available:
    python -m pytest tests/rocm/test_blockwise_fp8_linear.py -s
or standalone:
    python tests/rocm/test_blockwise_fp8_linear.py
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    getattr(torch.version, "hip", None) is None, reason="ROCm/HIP only"
)


def _run():
    from miles.utils.rocm_fp8_blockwise import blockwise_fp8_linear

    torch.manual_seed(0)
    M, N, K = 256, 2048, 2048
    x = (torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_()
    w = (torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_()

    xr = x.detach().clone().requires_grad_()
    wr = w.detach().clone().requires_grad_()
    (xr @ wr.t()).float().pow(2).mean().backward()

    y = blockwise_fp8_linear(x, w)
    y.float().pow(2).mean().backward()

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm()).item()

    fwd, dx, dw = rel(y, xr @ wr.t()), rel(x.grad, xr.grad), rel(w.grad, wr.grad)
    print(f"fwd={fwd:.4f} dX={dx:.4f} dW={dw:.4f}")
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
    assert fwd < 0.05 and dx < 0.10 and dw < 0.10
    return fwd, dx, dw


def test_blockwise_fp8_linear_matches_bf16():
    _run()


if __name__ == "__main__":
    print("RESULT:", "PASS", _run())
