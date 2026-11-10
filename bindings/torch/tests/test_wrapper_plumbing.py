"""CPU-only mathematics checks of the Python wrapper, NOT native GPU tests.

An explicit toy f(x, p) = p[0] * x**3 + p[1] * sin(x) supplies independently
written closed-form native-shaped methods. Only this test fixture redirects an
MPS transfer to CPU. It cannot validate Metal buffers, stream ordering, native
factory implementations, shader compilation, or the extension ABI.
"""

import importlib.util
import inspect
from enum import IntEnum
from pathlib import Path
import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytestmark = [
    pytest.mark.plumbing,
    pytest.mark.filterwarnings("ignore:input must be an MPS tensor:UserWarning"),
    pytest.mark.filterwarnings("ignore:doutput must be an MPS tensor:UserWarning"),
]


class Precision(IntEnum):
    Fp32 = 0
    Fp16 = 1


class PolynomialOracle:
    """Closed-form arithmetic oracle, deliberately not a production backend."""

    def __init__(self, precision=Precision.Fp32):
        self.precision = precision
        self.dtype = torch.float32 if precision == Precision.Fp32 else torch.float16
        self.jit_fusion = False
        self.backward_backward_input_no_input_grad = False
        self.calls = []

    def param_precision(self):
        return self.precision

    def output_precision(self):
        return self.precision

    def n_input_dims(self):
        return 2

    def n_output_dims(self):
        return 2

    def n_params(self):
        return 2

    def initial_params(self, seed):
        return torch.tensor([0.7, -0.4], dtype=torch.float32)

    def hyperparams(self):
        return {"otype": "TestPolynomialOracle"}

    def name(self):
        return "TestPolynomialOracle"

    def fwd(self, x, params):
        assert x.dtype == torch.float32 and params.dtype == self.dtype
        assert x.is_contiguous() and params.is_contiguous()
        self.calls.append(("fwd", x.shape, x.requires_grad, params.requires_grad))
        a, b = params.float()
        output = (a * x**3 + b * x.sin()).to(self.dtype)
        return ({"saved": True} if x.requires_grad or params.requires_grad else None), output

    def bwd(self, ctx, x, params, y, dy):
        assert ctx is not None and dy.is_contiguous()
        self.calls.append(("bwd", dy.shape, dy.is_contiguous()))
        a, b = params.float()
        dx = (dy.float() * (3 * a * x**2 + b * x.cos())) if x.requires_grad else None
        dp = torch.stack(((dy.float() * x**3).sum(), (dy.float() * x.sin()).sum())).to(self.dtype) if params.requires_grad else None
        return dx, dp

    def bwd_bwd_input(self, ctx, x, params, tangent, dy):
        assert ctx is not None and tangent.is_contiguous() and dy.is_contiguous()
        self.calls.append(("bwd_bwd_input", dy.requires_grad, tangent.is_contiguous(), dy.is_contiguous()))
        a, b = params.float()
        d_dy = (tangent * (3 * a * x**2 + b * x.cos())).to(self.dtype) if dy.requires_grad else None
        dp = torch.stack(((tangent * dy.float() * 3 * x**2).sum(), (tangent * dy.float() * x.cos()).sum())).to(self.dtype) if params.requires_grad else None
        dx = tangent * dy.float() * (6 * a * x - b * x.sin()) if x.requires_grad and not self.backward_backward_input_no_input_grad else None
        return d_dy, dp, dx


@pytest.fixture
def wrapper(monkeypatch, tmp_path):
    fake = types.ModuleType("tinymetalnn_bindings._C")
    fake.Precision = Precision
    fake.batch_size_granularity = lambda: 8
    fake.default_loss_scale = lambda precision: 128.0 if precision == Precision.Fp16 else 1.0
    fake.preferred_precision = lambda: Precision.Fp16
    fake.supports_jit_fusion = lambda device=-1: False
    fake.has_networks = lambda: True
    fake.free_temporary_memory = lambda: None
    fake.rtc_set_include_dir = lambda path: setattr(fake, "include_dir", path)
    fake.rtc_set_cache_dir = lambda path: setattr(fake, "cache_dir", path)
    fake.create_encoding = lambda dims, config, precision: PolynomialOracle(precision)
    fake.create_network = lambda inputs, outputs, config: PolynomialOracle()
    fake.create_network_with_input_encoding = lambda inputs, outputs, encoding, config: PolynomialOracle()
    parent = types.ModuleType("tinymetalnn_bindings")
    parent.__path__ = []
    parent._C = fake
    monkeypatch.setitem(sys.modules, "tinymetalnn_bindings", parent)
    monkeypatch.setitem(sys.modules, "tinymetalnn_bindings._C", fake)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setenv("TCNN_METAL_CACHE_DIR", str(tmp_path / "rtc-cache"))

    original_to = torch.Tensor.to

    def cpu_test_transfer(tensor, *args, **kwargs):
        positional_mps = args and isinstance(args[0], (str, torch.device)) and torch.device(args[0]).type == "mps"
        keyword_mps = kwargs.get("device") is not None and torch.device(kwargs["device"]).type == "mps"
        if (positional_mps or keyword_mps) and tensor.dtype == torch.float64:
            raise RuntimeError("The test MPS transfer rejects float64, matching the native device type restriction")
        if positional_mps:
            args = (torch.device("cpu"), *args[1:])
        if keyword_mps:
            kwargs["device"] = torch.device("cpu")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_test_transfer)
    source = Path(__file__).resolve().parents[1] / "tinymetalnn" / "modules.py"
    spec = importlib.util.spec_from_file_location("_tinymetalnn_plumbing_only", source)
    api = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, api)
    spec.loader.exec_module(api)

    class ToyModule(api.Module):
        def __init__(self, dtype=torch.float32, seed=1337):
            self.precision = Precision.Fp32 if dtype == torch.float32 else Precision.Fp16
            self.n_input_dims = self.n_output_dims = 2
            super().__init__(seed=seed)

        def _native_tcnn_module(self):
            return PolynomialOracle(self.precision)

    return api, ToyModule, fake


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("loss_scale", [1.0, 128.0])
def test_two_autograd_layers_match_independent_mathematics(wrapper, dtype, loss_scale):
    if dtype == torch.float16:
        try:
            torch.zeros(1).to(torch.float16)
        except RuntimeError as error:
            if "cpuinfo" not in str(error).lower():
                raise
            pytest.skip("This CPU-only Torch environment cannot initialize cpuinfo for FP16 conversion; Metal was not exercised")
    _, ToyModule, _ = wrapper
    model = ToyModule(dtype=dtype)
    model.loss_scale = loss_scale
    values = torch.tensor([[0.13, 0.27], [0.41, 0.63], [0.72, 0.89]])
    x = values.clone().requires_grad_()
    # Transposed output adjoints exercise the wrapper's contiguous conversion.
    weights = torch.linspace(-0.25, 0.35, 6).reshape(2, 3).t().to(dtype).detach().requires_grad_()
    tangent = torch.tensor([[0.1, -0.2], [0.2, 0.3], [-0.25, 0.15]])
    y = model(x)
    dx, dp = torch.autograd.grad(y, (x, model.params), weights, create_graph=True)
    hessian, mixed, output_adjoint = torch.autograd.grad((dx * tangent).sum(), (x, model.params, weights))

    reference_x = values.double().requires_grad_()
    # The native oracle receives the rounded execution parameters. The cast's
    # derivative into the FP32 master is the identity, as in PyTorch.
    reference_params = model.params.detach().to(dtype).double().requires_grad_()
    reference_weights = weights.detach().double().requires_grad_()
    a, b = reference_params
    reference_y = a * reference_x**3 + b * reference_x.sin()
    reference_dx, reference_dp = torch.autograd.grad(reference_y, (reference_x, reference_params), reference_weights, create_graph=True)
    reference_hessian, reference_mixed, reference_output_adjoint = torch.autograd.grad(
        (reference_dx * tangent.double()).sum(), (reference_x, reference_params, reference_weights)
    )
    tolerance = {"rtol": 3e-3, "atol": 3e-4} if dtype == torch.float16 else {"rtol": 2e-6, "atol": 2e-7}
    for actual, expected in [
        (y, reference_y.to(dtype)), (dx, reference_dx.float()), (dp, reference_dp.float()),
        (hessian, reference_hessian.float()), (mixed, reference_mixed.float()), (output_adjoint, reference_output_adjoint.to(dtype)),
    ]:
        torch.testing.assert_close(actual, expected.detach(), **tolerance)
    assert model.params.dtype == torch.float32
    assert dp.dtype == torch.float32
    assert any(call[0] == "bwd_bwd_input" and call[1] for call in model.native_tcnn_module.calls)


@pytest.mark.parametrize("batch", [0, 1, 7, 8, 9])
def test_batch_padding_and_output_slicing(wrapper, batch):
    _, ToyModule, _ = wrapper
    model = ToyModule()
    x = torch.linspace(0.1, 0.8, batch * 2).reshape(batch, 2).requires_grad_()
    y = model(x)
    assert y.shape == (batch, 2)
    call = model.native_tcnn_module.calls[-1]
    assert call[1] == (8 * ((batch + 7) // 8), 2)
    expected = model.params[0] * x**3 + model.params[1] * x.sin()
    torch.testing.assert_close(y, expected)
    y.sum().backward()
    torch.testing.assert_close(x.grad, 3 * model.params[0].detach() * x.detach()**2 + model.params[1].detach() * x.detach().cos())


def test_no_grad_selects_native_inference_and_frozen_params(wrapper):
    _, ToyModule, _ = wrapper
    model = ToyModule()
    x = torch.ones(3, 2, requires_grad=True)
    with torch.no_grad():
        y = model(x)
    assert not y.requires_grad
    assert model.native_tcnn_module.calls[-1][2:] == (False, False)
    model.params.requires_grad_(False)
    model(x).sum().backward()
    assert x.grad is not None
    assert model.params.grad is None


def test_backward_contiguity_and_noncontiguous_input(wrapper):
    _, ToyModule, _ = wrapper
    model = ToyModule()
    x = torch.linspace(0.1, 0.6, 10).reshape(2, 5).t().detach().requires_grad_()
    weights = torch.arange(10, dtype=torch.float32).reshape(2, 5).t()
    assert not x.is_contiguous() and not weights.is_contiguous()
    dx = torch.autograd.grad(model(x), x, weights, create_graph=True)[0]
    torch.autograd.grad(dx.sum(), x)
    assert any(call[0] == "bwd" and call[2] for call in model.native_tcnn_module.calls)


def test_cpu_double_input_casts_before_transfer_and_keeps_leaf_gradient(wrapper):
    _, ToyModule, _ = wrapper
    model = ToyModule()
    x = torch.tensor([[0.13, 0.27], [0.41, 0.63]], dtype=torch.float64, requires_grad=True)
    y = model(x)
    assert y.dtype == torch.float32
    y.sum().backward()
    assert x.grad.dtype == torch.float64
    expected = 3 * model.params[0].detach().double() * x.detach()**2 + model.params[1].detach().double() * x.detach().cos()
    torch.testing.assert_close(x.grad, expected, rtol=2e-6, atol=2e-7)


def test_parameter_second_derivative_is_an_explicit_error(wrapper):
    _, ToyModule, _ = wrapper
    model = ToyModule()
    x = torch.full((3, 2), 0.3, requires_grad=True)
    dp = torch.autograd.grad(model(x).sum(), model.params, create_graph=True)[0]
    with pytest.raises(RuntimeError, match="parameter gradients"):
        dp.sum().backward()


def test_suppression_flag_keeps_other_second_derivatives(wrapper):
    _, ToyModule, _ = wrapper
    model = ToyModule()
    model.backward_backward_input_no_input_grad = True
    x = torch.full((3, 2), 0.3, requires_grad=True)
    weights = torch.ones(3, 2, requires_grad=True)
    dx = torch.autograd.grad(model(x), x, weights, create_graph=True)[0]
    hessian, mixed, output_adjoint = torch.autograd.grad(dx.sum(), (x, model.params, weights), allow_unused=True)
    assert hessian is None
    assert mixed is not None and output_adjoint is not None


def test_serialization_protocol_rebuilds_native_state(wrapper):
    _, ToyModule, _ = wrapper
    original = ToyModule(seed=17)
    original.jit_fusion = True
    original.backward_backward_input_no_input_grad = True
    state = original.__getstate__()
    assert "native_tcnn_module" not in state
    restored = ToyModule.__new__(ToyModule)
    restored.__setstate__(state)
    assert restored.native_tcnn_module is not original.native_tcnn_module
    assert restored.jit_fusion and restored.backward_backward_input_no_input_grad
    assert restored.seed == 17
    torch.testing.assert_close(restored.params, original.params)
    with pytest.raises(AttributeError):
        del restored.jit_fusion
    with pytest.raises(AttributeError):
        del restored.backward_backward_input_no_input_grad


def test_constructor_signatures_and_encoding_dtype_validation(wrapper):
    api, _, _ = wrapper
    assert str(inspect.signature(api.Encoding)) == "(n_input_dims, encoding_config, seed=1337, dtype=None)"
    assert str(inspect.signature(api.Network)) == "(n_input_dims, n_output_dims, network_config, seed=1337)"
    assert str(inspect.signature(api.NetworkWithInputEncoding)) == "(n_input_dims, n_output_dims, encoding_config, network_config, seed=1337)"
    for dtype, expected in [(None, torch.float16), (torch.float16, torch.float16), (torch.float32, torch.float32)]:
        model = api.Encoding(2, {"otype": "TestPolynomialOracle"}, dtype=dtype)
        assert model.dtype == expected
        assert model.params.dtype == torch.float32
    with pytest.raises(ValueError):
        api.Encoding(2, {}, dtype=torch.bfloat16)


def test_encoding_only_build_preserves_classes_and_reports_capability(wrapper):
    api, _, fake = wrapper
    fake.has_networks = lambda: False
    with pytest.raises(RuntimeError, match="network"):
        api.Network(2, 2, {})
    with pytest.raises(RuntimeError, match="network"):
        api.NetworkWithInputEncoding(2, 2, {}, {})
    assert api.Encoding(2, {"otype": "TestPolynomialOracle"}).n_output_dims == 2


def test_cache_path_validation(wrapper, tmp_path):
    api, _, fake = wrapper
    api.rtc_set_cache_dir(tmp_path)
    assert fake.cache_dir == str(tmp_path)
    api.rtc_set_cache_dir(None)
    assert fake.cache_dir == ""
    with pytest.raises(OSError, match="Missing"):
        api.rtc_set_cache_dir(tmp_path / "missing")

