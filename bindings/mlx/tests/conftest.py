"""Hardware gates: an installed extension is mandatory when Metal is present."""

import importlib
import sys

import pytest


def pytest_addoption(parser):
    parser.addoption("--require-metal", action="store_true", help="Fail unless real MLX Metal is available")
    parser.addoption("--full-configurations", action="store_true", help="Expand the finite encoding configuration sweep")


def pytest_configure(config):
    config.addinivalue_line("markers", "metal: executes the actual native MLX Metal extension")
    config.addinivalue_line("markers", "plumbing: real MLX CPU transforms with an explicit mathematical test fixture")


def pytest_sessionstart(session):
    if not session.config.getoption("--require-metal"):
        return
    try:
        import mlx.core as mx
    except ImportError as error:
        raise pytest.UsageError("--require-metal needs an installed MLX runtime") from error
    if sys.platform != "darwin" or not mx.metal.is_available():
        raise pytest.UsageError("--require-metal needs macOS and real MLX Metal; no native validation was performed")


@pytest.fixture
def tcnn():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("MLX Metal is unavailable; this is a native GPU test")
    # A missing or broken extension on a Metal host is a failure, never a skip.
    module = importlib.import_module("tinymetalnn_mlx")
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield module
    finally:
        mx.set_default_device(previous)


@pytest.fixture
def native(tcnn):
    return importlib.import_module("tinymetalnn_mlx._C")


@pytest.fixture
def networks(tcnn, native):
    if not native.has_networks():
        pytest.skip("This explicitly configured extension was built without networks")
    return tcnn
