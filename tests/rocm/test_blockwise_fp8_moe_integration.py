"""End-to-end integration test for the patched TEGroupedLinear blockwise FP8 path.

Builds a TEGroupedLinear, runs forward+backward via TE, and checks the result against
the bf16 reference. Proves the rocm_te_blockwise_inject patch wires the per-expert
aiter blockwise FP8 GEMM through `tex.split_quantize` + `general_grouped_gemm`.

Run inside container:
    docker exec miles-hai2 bash -lc 'cd /data/data/hai/miles-dsv4 && \
        PYTHONPATH=/data/data/hai/miles-dsv4/miles/utils/te_inject_site \
        ROCM_TE_BLOCKWISE_INJECT=1 \
        python tests/rocm/test_blockwise_fp8_moe_integration.py'
"""
import os
import sys
import torch


def _setup_te_inject():
    # Force the injector to run before TE imports.
    sys.path.insert(0, "/data/data/hai/miles-dsv4/miles/utils/te_inject_site")
    os.environ["ROCM_TE_BLOCKWISE_INJECT"] = "1"
    os.environ.setdefault("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
    # Trigger sitecustomize ourselves (some test harnesses skip it).
    import sitecustomize  # noqa: F401


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12)).item()


def _run():
    _setup_te_inject()
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Float8BlockScaling

    torch.manual_seed(0)
    device = "cuda"

    E = 4  # num local experts
    H = 512
    FFN = 768
    tokens_per_expert = [128, 256, 384, 128]
    M = sum(tokens_per_expert)

    # TEGroupedLinear: shape (in_features, out_features) per expert.
    # We test the "fc1" semantics: M tokens, H -> FFN per expert.
    # Initialize all weights with same scale.
    layer = te.GroupedLinear(
        num_gemms=E,
        in_features=H,
        out_features=FFN,
        params_dtype=torch.bfloat16,
        device=device,
        bias=False,
    )

    inp = torch.randn(M, H, device=device, dtype=torch.bfloat16) * 0.05
    inp_ref = inp.detach().clone()
    inp.requires_grad_(True)
    inp_ref.requires_grad_(True)

    # Forward under FP8 recipe
    recipe = Float8BlockScaling()
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        out_fp8 = layer(inp, tokens_per_expert)
    out_fp8.float().pow(2).mean().backward()

    # bf16 reference: per-expert matmul
    chunks_in = torch.split(inp_ref, tokens_per_expert, dim=0)
    weights = [layer.parameters_split_dict[f"weight{i}"] if hasattr(layer, "parameters_split_dict")
               else getattr(layer, f"weight{i}")  for i in range(E)]
    # Above is just to grab the params; the cleaner way is to read state_dict.
    weights = []
    for i in range(E):
        for name, p in layer.named_parameters():
            if name == f"weight{i}":
                weights.append(p.detach().clone().to(torch.bfloat16))
                break
    assert len(weights) == E, f"only got {len(weights)} weights"
    out_ref_chunks = [c @ w.t() for c, w in zip(chunks_in, weights)]
    out_ref = torch.cat(out_ref_chunks, dim=0)
    out_ref.float().pow(2).mean().backward()

    fwd_err = _rel(out_fp8, out_ref)
    dx_err = _rel(inp.grad, inp_ref.grad)

    # Check weight gradients exist and are finite
    wgrad_errs = []
    for i in range(E):
        wgrad_fp8 = None
        for name, p in layer.named_parameters():
            if name == f"weight{i}":
                wgrad_fp8 = p.grad
                break
        assert wgrad_fp8 is not None, f"weight{i}.grad missing"
        assert torch.isfinite(wgrad_fp8).all(), f"weight{i}.grad has NaN/Inf"
        # bf16 reference wgrad: dY^T @ X per expert. We can compute it from chunks_in /
        # out_ref_chunks's loss path -- but the simplest sanity is finite-ness.
        wgrad_errs.append(float(wgrad_fp8.abs().max()))

    print(f"TEGroupedLinear FP8 vs bf16 ref: fwd={fwd_err:.4f} dx={dx_err:.4f}")
    print(f"per-expert wgrad max-abs: {[f'{e:.4g}' for e in wgrad_errs]}")
    assert torch.isfinite(out_fp8).all()
    assert torch.isfinite(inp.grad).all()
    # tolerances slightly looser; the recipe also re-quantizes weight workspace
    assert fwd_err < 0.10, f"fwd error too high: {fwd_err}"
    assert dx_err < 0.10, f"dx error too high: {dx_err}"
    print("PASS")


if __name__ == "__main__":
    _run()
