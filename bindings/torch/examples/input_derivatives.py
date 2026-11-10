#!/usr/bin/env python3
"""Validate analytic derivatives and train through an input-gradient penalty.

    python examples/input_derivatives.py
    python examples/input_derivatives.py --dtype fp16 --jit

The supported second-order family differentiates dL/dinput with respect to
input, parameters and the upstream output adjoint. It does not differentiate
dL/dparameters. The example exercises all three supported destinations.
"""

import argparse
import math

import torch


def reference_frequency(x, frequencies):
    # Supplied frequency.h and target encodings/math.metal both order features
    # by input dimension, octave, then phase 0/pi/2. Use sin/cos independently.
    scales = torch.tensor([math.pi * 2**index for index in range(frequencies)], dtype=x.dtype, device=x.device)
    angles = x[..., None] * scales
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(start_dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    if not torch.backends.mps.is_available():
        raise SystemExit("This example requires real MPS hardware and the compiled extension; no CPU backend is substituted.")
    import tinymetalnn as tcnn

    if args.jit and not tcnn.supports_jit_fusion():
        raise SystemExit("Native JIT fusion is unavailable in this build/device.")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    model = tcnn.Encoding(2, {"otype": "Frequency", "n_frequencies": 3}, dtype=dtype)
    model.jit_fusion = args.jit
    values = torch.tensor([[0.13, 0.27], [0.41, 0.63], [0.72, 0.89]])
    x = values.to("mps").requires_grad_()
    weight_values = torch.linspace(-0.25, 0.35, 36).reshape(3, 12).to(dtype)
    weights = weight_values.to("mps").requires_grad_()
    tangent = torch.tensor([[0.1, -0.2], [0.2, 0.3], [-0.25, 0.15]], device="mps")
    y = model(x)
    dx = torch.autograd.grad(y, x, weights, create_graph=True)[0]
    hessian_vector, output_adjoint = torch.autograd.grad((dx * tangent).sum(), (x, weights))

    # This CPU calculation is an independent mathematical oracle; the values
    # being checked above are produced by the actual native MPS extension.
    reference_x = values.double().requires_grad_()
    reference_weights = weight_values.double().requires_grad_()
    reference_y = reference_frequency(reference_x, 3)
    reference_dx = torch.autograd.grad(reference_y, reference_x, reference_weights, create_graph=True)[0]
    reference_hessian, reference_adjoint = torch.autograd.grad(
        (reference_dx * tangent.cpu().double()).sum(), (reference_x, reference_weights)
    )
    tolerance = {"rtol": 2e-2, "atol": 6e-2} if dtype == torch.float16 else {"rtol": 2e-4, "atol": 1e-4}
    for name, actual, expected in [
        ("forward", y, reference_y.to(dtype)),
        ("input gradient", dx, reference_dx.float()),
        ("Hessian-vector product", hessian_vector, reference_hessian.float()),
        ("output-adjoint derivative", output_adjoint, reference_adjoint.to(dtype)),
    ]:
        torch.testing.assert_close(actual.detach().cpu(), expected.detach(), **tolerance)
        error = (actual.detach().cpu().float() - expected.detach().float()).abs().max().item()
        print(f"PASS {name}: maximum absolute error {error:.6g}")

    grid = tcnn.Encoding(2, {
        "otype": "HashGrid", "n_levels": 2, "n_features_per_level": 2,
        "log2_hashmap_size": 6, "base_resolution": 4,
        "per_level_scale": 1.5, "interpolation": "Smoothstep",
    }, dtype=dtype)
    grid.jit_fusion = args.jit
    with torch.no_grad():
        grid.params.copy_(torch.linspace(-0.1, 0.1, grid.params.numel(), device="mps"))
    optimizer = torch.optim.Adam(grid.parameters(), lr=2e-3)
    coordinates = torch.rand(19, 2, device="mps") * 0.6 + 0.2
    before = grid.params.detach().clone()
    initial_penalty = None
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        points = coordinates.detach().requires_grad_()
        encoded = grid(points)
        gradient = torch.autograd.grad(encoded.float().sum(), points, create_graph=True)[0]
        penalty = gradient.square().mean()
        if initial_penalty is None:
            initial_penalty = penalty.item()
        penalty.backward()  # mixed d(dL/dinput)/dparams reaches FP32 masters
        assert grid.params.grad is not None and torch.isfinite(grid.params.grad).all().item()
        optimizer.step()
    assert not torch.equal(before, grid.params)
    print(f"PASS train through input gradients: penalty {initial_penalty:.6g} -> {penalty.item():.6g}")

    # The flag avoids the second input derivative without disabling the mixed
    # parameter derivative used for gradient-domain training above.
    model.backward_backward_input_no_input_grad = True
    x = values.to("mps").requires_grad_()
    weights = weight_values.to("mps").detach().requires_grad_()
    dx = torch.autograd.grad(model(x), x, weights, create_graph=True)[0]
    suppressed, retained = torch.autograd.grad(dx.sum(), (x, weights), allow_unused=True)
    assert suppressed is None and retained is not None
    print("PASS optional second input-gradient suppression")
    if args.jit:
        assert model.jit_fusion and grid.jit_fusion, "JIT fell back; inspect native diagnostics"


if __name__ == "__main__":
    main()

