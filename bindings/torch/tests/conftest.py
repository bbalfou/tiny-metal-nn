"""Native tests never substitute the Reference backend for Apple MPS."""

import importlib

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--require-mps",
        action="store_true",
        help="Fail instead of skipping native tests if an MPS device is absent.",
    )
    parser.addoption(
        "--full-configurations",
        action="store_true",
        help="Expand the encoding configuration sweep (additional shader compilations).",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "mps: requires the compiled extension and a native Apple MPS device")
    config.addinivalue_line("markers", "plumbing: CPU arithmetic oracle for Python wrapper behavior; not native GPU evidence")


def pytest_report_header(config):
    return (
        "tinymetalnn: mps tests exercise the real extension; plumbing tests use an "
        "explicit CPU arithmetic oracle and do not validate Metal"
    )


@pytest.fixture(scope="session")
def tcnn(request):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        reason = "Native MPS device unavailable; no Metal compilation or GPU execution was verified"
        if request.config.getoption("--require-mps"):
            pytest.fail(reason, pytrace=False)
        pytest.skip(reason)
    # Once hardware is present, an absent/broken extension is a failure, not a skip.
    return importlib.import_module("tinymetalnn")


@pytest.fixture(scope="session")
def native(tcnn):
    return importlib.import_module("tinymetalnn.modules")._C


@pytest.fixture
def networks(tcnn, native):
    if not native.has_networks():
        pytest.skip("This is an explicitly configured encoding-only build")
    return tcnn

