"""Analytic gradients and composition of the actual MLX native primitives."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.metal

from helpers import GRID, NETWORK, assert_close, assert_finite, frequency_reference


@pytest.mark.parametrize("tangent_dtype", [mx.float16, mx.float32, mx.int32])
def test_jvp_accepts_mlx_compatible_tangent_dtypes(tcnn, tangent_dtype):
    model = tcnn.Encoding(2, {"otype": "Identity", "scale": 1.75}, dtype=mx.float32)
    x = mx.full((3, 2), 0.3, dtype=mx.float32)
    tangent = mx.ones(x.shape, dtype=tangent_dtype)
    _, (actual,) = mx.jvp(model, [x], [tangent])
    assert_close(actual, 1.75 * tangent.astype(mx.float32), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_frequency_vjp_jvp_and_input_double_backward(tcnn, dtype):
    frequencies = 3
    model = tcnn.Encoding(2, {"otype": "Frequency", "n_frequencies": frequencies}, dtype=dtype)
    x = mx.array([[0.13, 0.29], [0.47, 0.81]], dtype=mx.float32)
    cotangent = mx.linspace(-0.3, 0.5, 24).reshape(2, 12).astype(dtype)
    vector = mx.array([[0.2, -0.3], [-0.4, 0.1]])
    (y,), (gx,) = mx.vjp(model, [x], [cotangent])
    _, (jvp,) = mx.jvp(model, [x], [vector])
    gradient = lambda z, g: mx.grad(lambda u: mx.sum(model(u) * g))(z)
    hessian_vector = mx.grad(lambda z: mx.sum(gradient(z, cotangent) * vector))(x)
    adjoint_derivative = mx.grad(lambda g: mx.sum(gradient(x, g) * vector))(cotangent)
    # Forward-over-reverse is the same Hessian-vector product for this scalar
    # objective, and should use the native second-order path as well.
    _, (forward_hessian_vector,) = mx.jvp(lambda z: gradient(z, cotangent), [x], [vector])

    reference_y, first, second = frequency_reference(np.asarray(x), frequencies)
    g = np.asarray(cotangent, dtype=np.float64).reshape(2, 2, frequencies, 2)
    v = np.asarray(vector, dtype=np.float64)
    reference_gx = np.sum(first * g, axis=(2, 3))
    reference_jvp = (first * v[:, :, None, None]).reshape(2, 12)
    reference_hvp = np.sum(second * g, axis=(2, 3)) * v
    tol = dict(rtol=2e-3, atol=2e-3) if dtype == mx.float16 else dict(rtol=3e-5, atol=8e-6)
    assert_close(y, reference_y, **tol)
    assert_close(gx, reference_gx, rtol=3e-5, atol=8e-6)
    assert_close(jvp, reference_jvp, **tol)
    assert_close(adjoint_derivative, reference_jvp, **tol)
    assert_close(hessian_vector, reference_hvp, rtol=5e-5, atol=3e-5)
    assert_close(forward_hessian_vector, reference_hvp, rtol=5e-5, atol=3e-5)


def test_native_second_derivative_through_composite_product(tcnn):
    model = tcnn.Encoding(2, {
        "otype": "Composite", "reduction": "Product", "nested": [
            {"otype": "Identity", "n_dims_to_encode": 2, "dims_to_encode_begin": 0, "scale": 2.0},
            {"otype": "Identity", "n_dims_to_encode": 2, "dims_to_encode_begin": 0, "scale": 3.0},
        ],
    }, dtype=mx.float32)
    x = mx.array([[0.2, 0.4], [0.5, 0.8]])
    vector = mx.array([[0.4, -0.1], [0.2, 0.3]])
    first = mx.grad(lambda z: mx.sum(model(z)))(x)
    second = mx.grad(lambda z: mx.sum(mx.grad(lambda u: mx.sum(model(u)))(z) * vector))(x)
    assert_close(model(x), 6 * x * x)
    assert_close(first, 12 * x)
    assert_close(second, 12 * vector)


def test_parameter_jvp_and_combined_direction_are_exact(tcnn):
    # Interpolation is linear in grid parameters: f(x, vp) is an independent
    # exact parameter-JVP oracle, avoiding finite-difference quantization.
    model = tcnn.Encoding(2, GRID, dtype=mx.float32)
    x = mx.array([[0.23, 0.37], [0.61, 0.43]])
    p = mx.linspace(-0.1, 0.1, model.n_params)
    vp = mx.sin(mx.arange(model.n_params, dtype=mx.float32)) * 0.1
    vx = mx.array([[0.05, -0.08], [0.04, 0.03]])
    fun = lambda z, q: model.forward_with_params(z, q)
    _, (jp,) = mx.jvp(lambda q: fun(x, q), [p], [vp])
    _, (jx,) = mx.jvp(lambda z: fun(z, p), [x], [vx])
    _, (combined,) = mx.jvp(fun, [x, p], [vx, vp])
    assert_close(jp, fun(x, vp), rtol=3e-5, atol=3e-6)
    assert_close(combined, jx + jp, rtol=3e-5, atol=3e-6)
    g = mx.linspace(-0.3, 0.4, 2 * model.n_output_dims).reshape(2, -1)
    _, (gx, gp) = mx.vjp(fun, [x, p], [g])
    assert_close(mx.sum(combined * g), mx.sum(gx * vx) + mx.sum(gp * vp), rtol=5e-5, atol=5e-6)


def test_grid_mixed_derivatives_and_parameter_backward_adjoint(tcnn):
    model = tcnn.Encoding(2, GRID, dtype=mx.float32)
    x = mx.array([[0.23, 0.37], [0.61, 0.43]])
    p = mx.linspace(-0.1, 0.1, model.n_params)
    g = mx.linspace(-0.3, 0.4, 2 * model.n_output_dims).reshape(2, -1)
    vx = mx.array([[0.05, -0.08], [0.04, 0.03]])
    vp = mx.sin(mx.arange(model.n_params, dtype=mx.float32)) * 0.1
    scalar = lambda z, q, c: mx.sum(model.forward_with_params(z, q) * c)
    gradient_x = mx.grad(scalar, argnums=0)
    gradient_p = mx.grad(scalar, argnums=1)
    mixed_p = mx.grad(lambda q: mx.sum(gradient_x(x, q, g) * vx))(p)
    mixed_x = mx.grad(lambda z: mx.sum(gradient_p(z, p, g) * vp))(x)
    parameter_adjoint = mx.grad(lambda c: mx.sum(gradient_p(x, p, c) * vp))(g)
    epsilon = 1e-3
    expected_p = (gradient_p(x + epsilon * vx, p, g) - gradient_p(x - epsilon * vx, p, g)) / (2 * epsilon)
    expected_x = (gradient_x(x, p + epsilon * vp, g) - gradient_x(x, p - epsilon * vp, g)) / (2 * epsilon)
    assert_close(mixed_p, expected_p, rtol=4e-3, atol=8e-5)
    assert_close(mixed_x, expected_x, rtol=4e-3, atol=8e-5)
    assert_close(parameter_adjoint, model.forward_with_params(x, vp), rtol=3e-5, atol=3e-6)
    # Mixed partials transposed in either differentiation order agree.
    assert_close(mx.sum(mixed_p * vp), mx.sum(mixed_x * vx), rtol=4e-4, atol=2e-6)


def test_second_order_suppression_only_changes_requested_input_gradient(tcnn):
    model = tcnn.Encoding(2, {"otype": "Frequency", "n_frequencies": 2}, dtype=mx.float32)
    x = mx.full((2, 2), 0.3)
    g = mx.full((2, model.n_output_dims), 0.25)
    vector = mx.full_like(x, 0.2)
    gradient = lambda z, c: mx.grad(lambda u: mx.sum(model(u) * c))(z)
    before = gradient(x, g)
    before_adjoint = mx.grad(lambda c: mx.sum(gradient(x, c) * vector))(g)
    mx.eval(before, before_adjoint)
    model.backward_backward_input_no_input_grad = True
    after = gradient(x, g)
    suppressed = mx.grad(lambda z: mx.sum(gradient(z, g) * vector))(x)
    after_adjoint = mx.grad(lambda c: mx.sum(gradient(x, c) * vector))(g)
    assert_close(after, before, rtol=0, atol=0)
    assert_close(suppressed, mx.zeros_like(x), rtol=0, atol=0)
    assert_close(after_adjoint, before_adjoint, rtol=0, atol=0)


def test_parameter_hessian_has_explicit_narrow_error(networks):
    model = networks.Network(2, 1, NETWORK)
    x = mx.full((2, 2), 0.2)
    parameter_gradient = mx.grad(lambda p: mx.sum(model.forward_with_params(x, p)))
    with pytest.raises((RuntimeError, ValueError, NotImplementedError), match="[Pp]arameter|Hessian|[Ss]econd|Hpp"):
        mx.eval(mx.grad(lambda p: mx.sum(parameter_gradient(p)))(model.params))


def test_third_derivative_has_explicit_narrow_error(tcnn):
    model = tcnn.Encoding(1, {"otype": "Frequency", "n_frequencies": 2}, dtype=mx.float32)
    first = mx.grad(lambda z: mx.sum(model(z)))
    second = mx.grad(lambda z: mx.sum(first(z)))
    with pytest.raises((RuntimeError, ValueError, NotImplementedError), match="[Tt]hird|[Hh]igher|[Ss]econd|derivative"):
        mx.eval(mx.grad(lambda z: mx.sum(second(z)))(mx.full((2, 1), 0.3)))


@pytest.mark.parametrize("mapped", ["inputs", "parameters", "both", "nonleading"])
def test_vmap_native_values_and_gradients(tcnn, mapped):
    model = tcnn.Encoding(2, GRID, dtype=mx.float32)
    xs = mx.linspace(0.15, 0.75, 12).reshape(3, 2, 2)
    ps = mx.stack([model.params + shift for shift in (0.0, 0.1, -0.2)])
    fun = lambda x, p: model.forward_with_params(x, p)
    if mapped == "inputs":
        actual = mx.vmap(fun, in_axes=(0, None))(xs, model.params)
        expected = mx.stack([fun(xs[i], model.params) for i in range(3)])
    elif mapped == "parameters":
        actual = mx.vmap(fun, in_axes=(None, 0))(xs[0], ps)
        expected = mx.stack([fun(xs[0], ps[i]) for i in range(3)])
    elif mapped == "both":
        actual = mx.vmap(fun)(xs, ps)
        expected = mx.stack([fun(xs[i], ps[i]) for i in range(3)])
    else:
        actual = mx.vmap(fun, in_axes=(1, None), out_axes=1)(mx.moveaxis(xs, 0, 1), model.params)
        expected = mx.moveaxis(mx.stack([fun(xs[i], model.params) for i in range(3)]), 0, 1)
    assert_close(actual, expected)
    gradient = mx.grad(lambda x, p: mx.sum(fun(x, p)), argnums=(0, 1))
    actual_gx, actual_gp = mx.vmap(gradient, in_axes=(0, None))(xs, model.params)
    expected_pairs = [gradient(xs[i], model.params) for i in range(3)]
    assert_close(actual_gx, mx.stack([pair[0] for pair in expected_pairs]))
    assert_close(actual_gp, mx.stack([pair[1] for pair in expected_pairs]))


def test_nested_vmap_row_mapping_and_empty_axis(tcnn):
    model = tcnn.Encoding(2, {"otype": "Identity", "scale": 1.5}, dtype=mx.float32)
    xs = mx.arange(24, dtype=mx.float32).reshape(2, 3, 2, 2)
    assert_close(mx.vmap(mx.vmap(model))(xs), 1.5 * xs)
    rows = xs.reshape(-1, 2)
    assert_close(mx.vmap(lambda row: model(row[None])[0])(rows), 1.5 * rows)
    empty = mx.zeros((0, 3, 2))
    empty_result = mx.vmap(model)(empty)
    mx.eval(empty_result)
    assert empty_result.shape == (0, 3, 2)
    empty_gradient = mx.grad(lambda value: mx.sum(mx.vmap(model)(value)))(empty)
    assert empty_gradient.shape == empty.shape
    mx.eval(empty_gradient)


def test_compiled_gradients_jvp_and_vmap(tcnn):
    model = tcnn.Encoding(2, {"otype": "Frequency", "n_frequencies": 2}, dtype=mx.float32)
    x = mx.linspace(0.1, 0.7, 12).reshape(3, 2, 2)
    gradient = mx.grad(lambda z: mx.sum(model(z)))
    expected = mx.vmap(gradient)(x)
    assert_close(mx.compile(mx.vmap(gradient))(x), expected)
    jvp = lambda z, v: mx.jvp(model, [z], [v])[1][0]
    vector = mx.full_like(x[0], 0.2)
    assert_close(mx.compile(jvp)(x[0], vector), jvp(x[0], vector))
    second = mx.grad(lambda z: mx.sum(gradient(z)))
    assert_close(mx.compile(second)(x[0]), second(x[0]))
    assert_finite(expected)


@pytest.mark.parametrize("otype", ["Identity", "Empty"])
def test_zero_parameter_and_empty_output_transform_composition(tcnn, otype):
    model = tcnn.Encoding(2, {"otype": otype}, dtype=mx.float32)
    x = mx.full((3, 2), 0.3)
    parameters = model.params
    assert parameters.shape == (0,)
    fun = lambda z, p: model.forward_with_params(z, p)
    _, (input_jvp,) = mx.jvp(model, [x], [mx.ones_like(x)])
    _, (parameter_jvp,) = mx.jvp(lambda p: fun(x, p), [parameters], [mx.zeros_like(parameters)])
    gx, gp = mx.grad(lambda z, p: mx.sum(fun(z, p)), argnums=(0, 1))(x, parameters)
    assert input_jvp.shape == (3, model.n_output_dims)
    assert parameter_jvp.shape == input_jvp.shape
    assert_close(parameter_jvp, mx.zeros_like(parameter_jvp), rtol=0, atol=0)
    assert_close(gx, mx.ones_like(x) if otype == "Identity" else mx.zeros_like(x), rtol=0, atol=0)
    assert gp.shape == (0,)
    groups = mx.stack((x, x + 0.1))
    grouped = mx.vmap(model)(groups)
    group_gradient = mx.grad(lambda values: mx.sum(mx.vmap(model)(values)))(groups)
    mapped_parameters = mx.vmap(lambda p: fun(x, p))(mx.zeros((2, 0)))
    assert grouped.shape == mapped_parameters.shape == (2, 3, model.n_output_dims)
    assert_close(group_gradient, mx.ones_like(groups) if otype == "Identity" else mx.zeros_like(groups), rtol=0, atol=0)
    assert_finite(input_jvp, parameter_jvp, gx, gp, grouped, mapped_parameters)


@pytest.mark.parametrize("coherent", [False, True], ids=["uniform", "repeated-point"])
@pytest.mark.parametrize("batch", [257, 65536, (1 << 20) + 1], ids=["adaptive-small-tail", "adaptive-small", "adaptive-large-tail"])
def test_large_fp16_hashgrid_gradient_conservation(tcnn, coherent, batch):
    """Every bilinear corner contributes to exactly one level/feature total."""
    config = {"otype": "HashGrid", "n_levels": 16, "n_features_per_level": 2,
              "log2_hashmap_size": 19, "base_resolution": 16, "per_level_scale": 2.0}
    model = tcnn.Encoding(2, config, dtype=mx.float16)
    # Keep the coherent total representable before the FP16 gradient cast.
    model.loss_scale = 1.0
    values = (np.tile(np.array([[.371, .417]], dtype=np.float32), (batch, 1))
              if coherent else np.random.default_rng(1337).random((batch, 2), dtype=np.float32))
    x = mx.array(values)
    gradient = mx.grad(lambda p: mx.sum(model.forward_with_params(x, p).astype(mx.float32)) / 256)(model.params)
    mx.eval(gradient)
    actual = np.asarray(gradient).astype(np.float32)
    assert np.isfinite(actual).all()
    offset = 0
    for level in range(16):
        vertices = min((16 * 2**level)**2, 2**19)
        sums = actual[offset:offset + 2 * vertices].reshape(vertices, 2).sum(axis=0, dtype=np.float64)
        np.testing.assert_allclose(sums, batch / 256, atol=2e-3, rtol=2e-3)
        offset += 2 * vertices
    assert offset == model.n_params
