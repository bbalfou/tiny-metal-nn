"""Native factory dispatch coverage, distinct from mathematical proof."""

import importlib.util
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.metal

from helpers import assert_close, assert_finite

_path = Path(__file__).resolve().parents[1] / "examples" / "encoding_catalog.py"
_spec = importlib.util.spec_from_file_location("_tinymetalnn_mlx_encoding_catalog", _path)
_catalog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_catalog)


def pytest_generate_tests(metafunc):
    if "encoding_case" in metafunc.fixturenames:
        cases = list(_catalog.encoding_cases(metafunc.config.getoption("--full-configurations")))
        metafunc.parametrize("encoding_case", [(d, c) for _, d, c in cases], ids=[n for n, _, _ in cases])


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_encoding_factory_family(tcnn, encoding_case, dtype):
    dims, config = encoding_case
    model = tcnn.Encoding(dims, config, dtype=dtype)
    x = mx.linspace(0.11, 0.89, 3 * dims).reshape(3, dims)
    y = model(x)
    objective = lambda z, p: mx.sum(model.forward_with_params(z, p).astype(mx.float32))
    gx, gp = mx.grad(objective, argnums=(0, 1))(x, model.params)
    assert y.shape == (3, model.n_output_dims) and y.dtype == dtype
    assert gx.shape == x.shape and gp.shape == model.params.shape
    assert gp.dtype == mx.float32
    assert_finite(y, gx, gp)
    if config.get("max_input_grad_dims") == 2:
        assert_close(gx[:, 2:], mx.zeros_like(gx[:, 2:]), rtol=0, atol=0)


@pytest.mark.parametrize("otype,width,depth", [
    ("GemmMLP", 0, 0), ("GemmMLP", 1, 1), ("GemmMLP", 7, 2),
    ("GemmMLP", 24, 1), ("GemmMLP", 33, 2), ("GemmMLP", 257, 1),
    ("FullyFusedMLP", 16, 0), ("FullyFusedMLP", 16, 1),
    ("FullyFusedMLP", 32, 2), ("FullyFusedMLP", 64, 1), ("FullyFusedMLP", 128, 1),
])
def test_network_dispatch_family(networks, otype, width, depth):
    model = networks.Network(3, 5, {
        "otype": otype, "n_neurons": width, "n_hidden_layers": depth,
        "activation": "ReLU", "output_activation": "None",
    })
    x = mx.random.uniform(shape=(5, 3))
    objective = lambda z, p: mx.mean(mx.square(model.forward_with_params(z, p).astype(mx.float32)))
    value, (gx, gp) = mx.value_and_grad(objective, argnums=(0, 1))(x, model.params)
    assert model(x).shape == (5, 5)
    assert_finite(value, gx, gp)


@pytest.mark.parametrize("otype", ["MLP", "MetalMLP", "CutlassMLP", "MegakernelMLP"])
def test_removed_network_aliases_are_rejected(networks, otype):
    with pytest.raises((RuntimeError, ValueError), match="Invalid network type"):
        networks.Network(3, 5, {"otype": otype})


def test_default_network_is_gemm(networks):
    model = networks.Network(3, 5, {})
    assert model.native_tcnn_module.hyperparams()["network"]["otype"] == "GemmMLP"


def test_invalid_fused_width_is_rejected_on_supported_hardware(networks):
    probe = networks.Network(3, 5, {"otype": "FullyFusedMLP", "n_neurons": 16})
    if probe.native_tcnn_module.hyperparams()["network"]["otype"] != "FullyFusedMLP":
        pytest.skip("This device uses the GemmMLP hardware fallback")
    with pytest.raises((RuntimeError, ValueError), match="16, 32, 64, and 128"):
        networks.Network(3, 5, {"otype": "FullyFusedMLP", "n_neurons": 24})


@pytest.mark.parametrize("activation", [
    "ReLU", "LeakyReLU", "SiLU", "Exponential", "Sine", "Sigmoid",
    "Squareplus", "Softplus", "Tanh", "None",
])
def test_network_activation_family(networks, activation):
    model = networks.Network(2, 3, {
        "otype": "GemmMLP", "n_neurons": 16, "n_hidden_layers": 1,
        "activation": activation, "output_activation": activation,
    })
    x = mx.full((3, 2), 0.1)
    objective = lambda z, p: mx.sum(model.forward_with_params(z, p).astype(mx.float32))
    value, (gx, gp) = mx.value_and_grad(objective, argnums=(0, 1))(x, model.params)
    assert_finite(value, gx, gp)


@pytest.mark.parametrize("dims,config", [
    (5, {"otype": "HashGrid"}),
    (2, {"otype": "HashGrid", "n_features_per_level": 3}),
    (2, {"otype": "HashGrid", "hash": "Prime"}),
    (11, {"otype": "Permuto"}),
    (2, {"otype": "Permuto", "interpolation": "Smoothstep"}),
    (2, {"otype": "OneBlob", "n_bins": 3}),
    (2, {"otype": "SphericalHarmonics"}),
    (3, {"otype": "SphericalHarmonics", "degree": 9}),
    (2, {"otype": "DoesNotExist"}),
])
def test_invalid_or_disabled_factory_configuration(tcnn, dims, config):
    with pytest.raises((RuntimeError, ValueError)):
        tcnn.Encoding(dims, config, dtype=mx.float32)
