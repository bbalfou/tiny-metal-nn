"""Factory coverage: each finite dispatch axis, plus an opt-in wider sweep.

These checks establish reachability, shapes and finite first derivatives. They
complement analytic/finite-difference tests; they do not assert a complete
numerical proof for the Cartesian product of all JSON configuration values.
"""

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.mps


# Use the same explicit case catalog as the runnable example.
from pathlib import Path
import importlib.util

_catalog_path = Path(__file__).resolve().parents[1] / "examples" / "encoding_catalog.py"
_catalog_spec = importlib.util.spec_from_file_location("_tinymetalnn_encoding_catalog", _catalog_path)
_catalog = importlib.util.module_from_spec(_catalog_spec)
_catalog_spec.loader.exec_module(_catalog)
encoding_cases = _catalog.encoding_cases


def pytest_generate_tests(metafunc):
    if "encoding_case" in metafunc.fixturenames:
        cases = list(encoding_cases(metafunc.config.getoption("--full-configurations")))
        metafunc.parametrize("encoding_case", [(dims, config) for _, dims, config in cases], ids=[name for name, _, _ in cases])


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_encoding_factory_family(tcnn, encoding_case, dtype):
    dims, config = encoding_case
    model = tcnn.Encoding(dims, config, dtype=dtype)
    x = torch.linspace(0.11, 0.89, 3 * dims, device="mps").reshape(3, dims).requires_grad_()
    y = model(x)
    assert y.shape == (3, model.n_output_dims)
    assert y.dtype == dtype
    assert torch.isfinite(y).all().item()
    y.float().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all().item()
    if model.params.numel():
        assert model.params.grad is not None and torch.isfinite(model.params.grad).all().item()
    if config.get("max_input_grad_dims") == 2:
        torch.testing.assert_close(x.grad[:, 2:], torch.zeros_like(x.grad[:, 2:]), rtol=0, atol=0)


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
    x = torch.rand(5, 3, device="mps", requires_grad=True)
    y = model(x)
    assert y.shape == (5, 5)
    y.float().square().mean().backward()
    assert torch.isfinite(x.grad).all().item()
    assert torch.isfinite(model.params.grad).all().item()


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


@pytest.mark.parametrize("activation", ["ReLU", "LeakyReLU", "SiLU", "Exponential", "Sine", "Sigmoid", "Squareplus", "Softplus", "Tanh", "None"])
def test_network_activation_family(networks, activation):
    model = networks.Network(2, 3, {
        "otype": "GemmMLP", "n_neurons": 16, "n_hidden_layers": 1,
        "activation": activation, "output_activation": activation,
    })
    x = torch.full((3, 2), 0.1, device="mps", requires_grad=True)
    y = model(x)
    assert torch.isfinite(y).all().item()
    y.float().mean().backward()
    assert torch.isfinite(x.grad).all().item()
    assert torch.isfinite(model.params.grad).all().item()


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
def test_invalid_or_disabled_factory_configurations_fail(tcnn, dims, config):
    with pytest.raises((RuntimeError, ValueError)):
        tcnn.Encoding(dims, config, dtype=torch.float32)

