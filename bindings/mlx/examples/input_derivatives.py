#!/usr/bin/env python3
"""Check native derivatives, JVP, vmap, compile, and gradient-penalty training.

Frequency has an independent NumPy float64 oracle. A small trainable grid then
exercises parameter JVP and training through input gradients. Exact parameter
JVP uses native backward basis evaluations; its cost includes the padded batch,
so this example deliberately uses a small output width.
"""

import argparse
import json
import math


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp32")
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")

    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    if not mx.metal.is_available():
        raise SystemExit("This example requires MLX Metal and the native tinymetalnn_mlx extension.")
    import tinymetalnn_mlx as tcnn
    if args.jit and not tcnn.supports_jit_fusion():
        raise SystemExit("Native JIT fusion is unavailable in this build/device.")
    mx.set_default_device(mx.gpu)
    dtype = mx.float16 if args.dtype == "fp16" else mx.float32

    def check(actual, expected, rtol=3e-5, atol=3e-5):
        mx.eval(actual)
        if isinstance(expected, mx.array):
            mx.eval(expected)
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=rtol, atol=atol)

    frequencies = 3
    model = tcnn.Encoding(2, {"otype": "Frequency", "n_frequencies": frequencies}, dtype=dtype)
    model.jit_fusion = args.jit
    x = mx.array([[0.13, 0.29], [0.47, 0.81]])
    g = mx.linspace(-0.3, 0.5, 24).reshape(2, 12).astype(dtype)
    vector = mx.array([[0.2, -0.3], [-0.4, 0.1]])
    (y,), (gradient,) = mx.vjp(model, [x], [g])
    input_gradient = lambda z, c: mx.grad(lambda u: mx.sum(model(u) * c))(z)
    hessian_vector = mx.grad(lambda z: mx.sum(input_gradient(z, g) * vector))(x)
    adjoint_gradient = mx.grad(lambda c: mx.sum(input_gradient(x, c) * vector))(g)
    _, (jvp,) = mx.jvp(model, [x], [vector])

    host_x = np.asarray(x, dtype=np.float64)
    scales = math.pi * 2.0 ** np.arange(frequencies, dtype=np.float64)
    phase = host_x[:, :, None] * scales[None, None, :]
    oracle_y = np.stack((np.sin(phase), np.cos(phase)), axis=-1)
    first = np.stack((np.cos(phase), -np.sin(phase)), axis=-1) * scales[None, None, :, None]
    second = -oracle_y * scales[None, None, :, None] ** 2
    host_g = np.asarray(g, dtype=np.float64).reshape(2, 2, frequencies, 2)
    host_vector = np.asarray(vector, dtype=np.float64)
    oracle_gradient = np.sum(first * host_g, axis=(2, 3))
    oracle_hvp = np.sum(second * host_g, axis=(2, 3)) * host_vector
    oracle_jvp = (first * host_vector[:, :, None, None]).reshape(2, -1)
    output_tolerance = dict(rtol=2e-3, atol=2e-3) if dtype == mx.float16 else dict(rtol=3e-5, atol=8e-6)
    check(y, oracle_y.reshape(2, -1), **output_tolerance)
    check(gradient, oracle_gradient)
    check(hessian_vector, oracle_hvp, rtol=5e-5, atol=3e-5)
    check(jvp, oracle_jvp, **output_tolerance)
    check(adjoint_gradient, oracle_jvp, **output_tolerance)

    # These remain lazy graph transformations. mx.compile and native model JIT
    # are independent: either or both may be used.
    compiled_gradient = mx.compile(lambda z: input_gradient(z, g))
    check(compiled_gradient(x), gradient)
    batches = mx.stack((x, x + 0.02))
    vmapped = mx.vmap(model)(batches)
    check(vmapped, mx.stack((model(x), model(x + 0.02))), **output_tolerance)
    model.backward_backward_input_no_input_grad = True
    suppressed = mx.grad(lambda z: mx.sum(input_gradient(z, g) * vector))(x)
    check(suppressed, mx.zeros_like(x), rtol=0, atol=0)
    # The adjoint derivative remains available when input curvature is disabled.
    check(mx.grad(lambda c: mx.sum(input_gradient(x, c) * vector))(g), oracle_jvp, **output_tolerance)
    model.backward_backward_input_no_input_grad = False

    grid = tcnn.Encoding(2, {
        "otype": "HashGrid", "n_levels": 1, "n_features_per_level": 1,
        "base_resolution": 4, "log2_hashmap_size": 4, "interpolation": "Smoothstep",
    }, dtype=dtype)
    grid.jit_fusion = args.jit
    sample = mx.array([[0.23, 0.37], [0.61, 0.43]])
    direction = mx.sin(mx.arange(grid.n_params, dtype=mx.float32)) * 0.1
    _, (parameter_jvp,) = mx.jvp(lambda p: grid.forward_with_params(sample, p), [grid.params], [direction])
    # Grid interpolation is linear in the parameters.
    check(parameter_jvp, grid.forward_with_params(sample, direction), **output_tolerance)

    optimizer = optim.Adam(learning_rate=0.01)

    def penalty(m):
        derivative = mx.grad(lambda z: mx.sum(m(z).astype(mx.float32)))(sample)
        return mx.mean(mx.square(derivative - 0.2))

    value_and_grad = nn.value_and_grad(grid, penalty)
    before = penalty(grid).item()
    for _ in range(args.steps):
        value, gradients = value_and_grad(grid)
        assert gradients["params"].dtype == mx.float32
        optimizer.update(grid, gradients)
        mx.eval(value, grid.parameters(), optimizer.state)
        assert mx.all(mx.isfinite(gradients["params"])).item()
    after = penalty(grid).item()
    assert after < before, f"Gradient penalty did not decrease ({before} -> {after})"
    if args.jit:
        assert model.jit_fusion and grid.jit_fusion, "Native JIT fell back to AOT"
    tcnn.free_temporary_memory()
    print(json.dumps({
        "status": "passed on native MLX Metal", "dtype": args.dtype, "jit": args.jit,
        "checked": ["forward", "input VJP", "input Hessian-vector product", "adjoint derivative",
                    "input and parameter JVP", "vmap", "mx.compile", "gradient-penalty training"],
        "gradient_penalty_before": before, "gradient_penalty_after": after,
    }))


if __name__ == "__main__":
    main()
