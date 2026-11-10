"""Independent analytic and finite-difference checks of supported derivatives."""

import math

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.mps


@pytest.mark.parametrize("coherent", [False, True], ids=["uniform", "repeated-point"])
@pytest.mark.parametrize("batch", [257, 65536, (1 << 20) + 1], ids=["adaptive-small-tail", "adaptive-small", "adaptive-large-tail"])
def test_large_fp16_hashgrid_gradient_conservation(tcnn, coherent, batch):
    """The public autograd path preserves every level/feature total, including padding."""
    config = {"otype": "HashGrid", "n_levels": 16, "n_features_per_level": 2,
              "log2_hashmap_size": 19, "base_resolution": 16, "per_level_scale": 2.0}
    model = tcnn.Encoding(2, config, dtype=torch.float16)
    # Avoid overflowing FP16 gradients for the repeated-point case.
    model.loss_scale = 1.0
    values = (torch.tensor([[.371, .417]]).expand(batch, 2).contiguous()
              if coherent else torch.rand((batch, 2), generator=torch.Generator().manual_seed(1337)))
    (model(values.to("mps")).float().sum() / 256).backward()
    actual = model.params.grad.detach().cpu().double()
    assert torch.isfinite(actual).all()
    offset = 0
    for level in range(16):
        vertices = min((16 * 2**level)**2, 2**19)
        sums = actual[offset:offset + 2 * vertices].reshape(vertices, 2).sum(dim=0)
        torch.testing.assert_close(sums, torch.full_like(sums, batch / 256), atol=2e-3, rtol=2e-3)
        offset += 2 * vertices
    assert offset == model.params.numel()


def frequency_reference(x, frequencies):
    # Confirmed against supplied encodings/frequency.h: j/(2*n_frequencies)
    # selects the dimension, (j/2)%n_frequencies selects the octave, and j%2
    # adds a pi/2 phase. Target kernels/encodings/math.metal uses the same order.
    # sin(angle + pi/2) is independently evaluated as cos(angle) here.
    scales = torch.tensor([math.pi * 2**i for i in range(frequencies)], dtype=x.dtype, device=x.device)
    angles = x[..., None] * scales
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(start_dim=1)


def first_and_second(model, x, weights, tangent):
    y = model(x)
    dx = torch.autograd.grad(y, x, grad_outputs=weights, create_graph=True)[0]
    targets = (x, weights, model.params) if model.params.numel() else (x, weights)
    derivatives = torch.autograd.grad((dx * tangent).sum(), targets, allow_unused=True)
    return y.detach(), dx.detach(), tuple(None if d is None else d.detach() for d in derivatives)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("scale", [1.0, 128.0])
def test_frequency_forward_first_and_second_derivatives(tcnn, dtype, scale):
    model = tcnn.Encoding(2, {"otype": "Frequency", "n_frequencies": 3}, dtype=dtype)
    model.loss_scale = scale
    x_values = torch.tensor([[0.13, 0.27], [0.41, 0.63], [0.72, 0.89]], dtype=torch.float32)
    x = x_values.to("mps").requires_grad_()
    # A noncontiguous adjoint is a normal PyTorch autograd input.
    w_values = torch.linspace(-0.25, 0.35, 36).reshape(12, 3).t().to(dtype)
    weights = w_values.to("mps").detach().requires_grad_()
    tangent = torch.tensor([[0.1, -0.2], [0.2, 0.3], [-0.25, 0.15]], device="mps")
    actual_y, actual_dx, (actual_hv, actual_jv) = first_and_second(model, x, weights, tangent)

    reference_x = x_values.double().requires_grad_()
    reference_weights = w_values.double().requires_grad_()
    reference_y = frequency_reference(reference_x, 3)
    reference_dx = torch.autograd.grad(reference_y, reference_x, reference_weights, create_graph=True)[0]
    reference_hv, reference_jv = torch.autograd.grad(
        (reference_dx * tangent.cpu().double()).sum(), (reference_x, reference_weights)
    )
    tolerance = {"rtol": 2e-2, "atol": 6e-2} if dtype == torch.float16 else {"rtol": 2e-4, "atol": 1e-4}
    torch.testing.assert_close(actual_y.cpu(), reference_y.detach().to(dtype), **tolerance)
    torch.testing.assert_close(actual_dx.cpu(), reference_dx.detach().float(), **tolerance)
    torch.testing.assert_close(actual_hv.cpu(), reference_hv.float(), **tolerance)
    torch.testing.assert_close(actual_jv.cpu(), reference_jv.to(dtype), **tolerance)


@pytest.mark.parametrize("reduction", ["Concatenation", "Sum", "Product"])
def test_composite_chain_rule(tcnn, reduction):
    config = {
        "otype": "Composite",
        "reduction": reduction,
        "nested": [
            {"otype": "Identity", "n_dims_to_encode": 2, "dims_to_encode_begin": 0, "scale": 2.0, "offset": 0.3},
            {"otype": "Identity", "n_dims_to_encode": 2, "dims_to_encode_begin": 0, "scale": -1.5, "offset": 0.4},
        ],
    }
    model = tcnn.Encoding(2, config, dtype=torch.float32)
    x_values = torch.tensor([[0.17, 0.36], [0.58, 0.72]], dtype=torch.float32)
    x = x_values.to("mps").requires_grad_()
    weights = torch.linspace(0.1, 0.3, 2 * model.n_output_dims, device="mps").reshape(2, -1).requires_grad_()
    tangent = torch.tensor([[0.1, -0.2], [0.2, 0.1]], device="mps")
    y, dx, (hv, jv) = first_and_second(model, x, weights, tangent)
    reference_x = x_values.double().requires_grad_()
    reference_weights = weights.detach().cpu().double().requires_grad_()
    a, b = 2 * reference_x + 0.3, -1.5 * reference_x + 0.4
    reference_y = torch.cat((a, b), dim=1) if reduction == "Concatenation" else a + b if reduction == "Sum" else a * b
    reference_dx = torch.autograd.grad(reference_y, reference_x, reference_weights, create_graph=True)[0]
    reference_jv = torch.autograd.grad((reference_dx * tangent.cpu().double()).sum(), reference_weights, retain_graph=True)[0]
    if reduction == "Product":
        reference_hv = torch.autograd.grad((reference_dx * tangent.cpu().double()).sum(), reference_x)[0]
    else:
        reference_hv = torch.zeros_like(reference_x)
    for actual, expected in [(y, reference_y), (dx, reference_dx), (hv, reference_hv), (jv, reference_jv)]:
        torch.testing.assert_close(actual.cpu(), expected.detach().float(), rtol=2e-5, atol=2e-6)


def test_trainable_grid_mixed_derivatives_against_finite_differences(tcnn):
    config = {
        "otype": "HashGrid", "n_levels": 2, "n_features_per_level": 2,
        "log2_hashmap_size": 4, "base_resolution": 4, "per_level_scale": 1.5,
        "interpolation": "Smoothstep",
    }
    model = tcnn.Encoding(2, config, dtype=torch.float32)
    with torch.no_grad():
        model.params.copy_(torch.linspace(-0.2, 0.3, model.params.numel(), device="mps"))
    base_params = model.params.detach().clone()
    x = torch.tensor([[0.19, 0.31], [0.47, 0.68]], device="mps", requires_grad=True)
    tangent = torch.tensor([[0.1, -0.2], [0.15, 0.25]], device="mps")
    weights = torch.linspace(0.2, 0.5, 2 * model.n_output_dims, device="mps").reshape(2, -1).requires_grad_()
    parameter_direction = torch.cos(torch.arange(model.params.numel(), device="mps", dtype=torch.float32)) * 0.2
    y, dx, (hv, jv, mixed) = first_and_second(model, x, weights, tangent)
    assert mixed is not None and mixed.shape == model.params.shape
    step_x, step_p = 2e-3, 1e-2

    def evaluate(x_value, parameter_offset=0.0):
        with torch.no_grad():
            model.params.copy_(base_params + parameter_offset * parameter_direction)
            return model(x_value).float()

    try:
        plus = evaluate(x.detach() + step_x * tangent)
        minus = evaluate(x.detach() - step_x * tangent)
        numeric_jv = (plus - minus) / (2 * step_x)
        torch.testing.assert_close(jv, numeric_jv, rtol=8e-3, atol=2e-4)
        torch.testing.assert_close((dx * tangent).sum(), (numeric_jv * weights.detach()).sum(), rtol=8e-3, atol=2e-4)
        pp = evaluate(x.detach() + step_x * tangent, step_p)
        mp = evaluate(x.detach() - step_x * tangent, step_p)
        pm = evaluate(x.detach() + step_x * tangent, -step_p)
        mm = evaluate(x.detach() - step_x * tangent, -step_p)
        numeric_mixed = ((pp - mp - pm + mm) * weights.detach()).sum() / (4 * step_x * step_p)
        torch.testing.assert_close((mixed * parameter_direction).sum(), numeric_mixed, rtol=2e-2, atol=3e-3)
        with torch.no_grad():
            model.params.copy_(base_params)
        shifted_gradients = []
        for sign in (1, -1):
            shifted = (x.detach() + sign * step_x * tangent).requires_grad_()
            shifted_gradients.append(torch.autograd.grad(model(shifted), shifted, weights.detach())[0])
        numeric_hv = (shifted_gradients[0] - shifted_gradients[1]) / (2 * step_x)
        torch.testing.assert_close(hv, numeric_hv, rtol=1e-2, atol=1e-3)
    finally:
        with torch.no_grad():
            model.params.copy_(base_params)


def test_suppress_second_input_gradient_only(tcnn):
    model = tcnn.Encoding(2, {"otype": "Frequency", "n_frequencies": 2}, dtype=torch.float32)
    model.backward_backward_input_no_input_grad = True
    x = torch.tensor([[0.13, 0.27]], device="mps", requires_grad=True)
    weights = torch.ones(1, model.n_output_dims, device="mps", requires_grad=True)
    dx = torch.autograd.grad(model(x), x, weights, create_graph=True)[0]
    hessian, output_adjoint = torch.autograd.grad(dx.sum(), (x, weights), allow_unused=True)
    assert hessian is None
    assert output_adjoint is not None
    assert torch.isfinite(output_adjoint).all().item()


def test_parameter_gradient_second_derivative_fails_explicitly(tcnn):
    model = tcnn.Encoding(2, {
        "otype": "HashGrid", "n_levels": 1, "n_features_per_level": 2,
        "log2_hashmap_size": 4, "base_resolution": 4,
    }, dtype=torch.float32)
    x = torch.tensor([[0.17, 0.31]], device="mps", requires_grad=True)
    parameter_gradient = torch.autograd.grad(model(x).sum(), model.params, create_graph=True)[0]
    with pytest.raises(RuntimeError, match="parameter gradients|parameter.*gradient"):
        parameter_gradient.sum().backward()


@pytest.mark.parametrize("kind,otype", [
    ("encoding", None), ("network", "GemmMLP"), ("combined", "GemmMLP"),
    ("network", "FullyFusedMLP"), ("combined", "FullyFusedMLP"),
])
def test_jit_forward_and_derivatives_match_aot(tcnn, native, kind, otype):
    if not tcnn.supports_jit_fusion():
        pytest.skip("Native runtime compilation/model fusion capability is unavailable in this build/device")
    if kind != "encoding" and not native.has_networks():
        pytest.skip("Encoding-only build")
    encoding = {"otype": "Frequency", "n_frequencies": 2}
    network = {"otype": otype, "n_neurons": 16, "n_hidden_layers": 1, "activation": "Tanh", "output_activation": "None"}
    if kind == "encoding":
        model = tcnn.Encoding(2, encoding, dtype=torch.float32)
    elif kind == "network":
        model = tcnn.Network(2, 3, network)
    else:
        model = tcnn.NetworkWithInputEncoding(2, 3, encoding, network)
    x_values = torch.tensor([[0.13, 0.27], [0.39, 0.58]], device="mps")
    results = []
    for enabled in (False, True):
        model.jit_fusion = enabled
        x = x_values.detach().clone().requires_grad_()
        weights = torch.full((2, model.n_output_dims), 0.25, device="mps", dtype=model.dtype, requires_grad=True)
        tangent = torch.full_like(x, 0.125)
        results.append(first_and_second(model, x, weights, tangent))
        assert model.jit_fusion == enabled, "A JIT compilation fallback is not evidence of JIT execution"
    expected, actual = results
    tolerance = {"rtol": 7e-2, "atol": 3e-2} if model.dtype == torch.float16 else {"rtol": 2e-3, "atol": 5e-4}
    torch.testing.assert_close(actual[0], expected[0], **tolerance)
    torch.testing.assert_close(actual[1], expected[1], **tolerance)
    for observed, reference in zip(actual[2], expected[2]):
        assert (observed is None) == (reference is None)
        if observed is not None:
            torch.testing.assert_close(observed, reference, **tolerance)

