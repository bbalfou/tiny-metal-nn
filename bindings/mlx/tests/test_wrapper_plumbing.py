"""Real MLX CPU wrapper checks with an explicitly injected mathematical fixture.

These tests DO NOT execute the extension or Metal. The only substitute is a
test-scoped native facade whose operation is f(x,p)=p[0]*x**3+p[1]*sin(x), built
from real MLX operations. Its derivatives are checked against separate formulas.
No fallback or monkeypatch is installed in the production package.
"""

from enum import IntEnum
import importlib
from pathlib import Path
import pickle
import sys
import types

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytestmark = pytest.mark.plumbing

from helpers import assert_close


class _Precision(IntEnum):
    Fp32 = 0
    Fp16 = 1


@pytest.fixture
def wrapper(monkeypatch, tmp_path):
    class MathematicalFacade:
        def __init__(self, dims, config, precision, output_dims=None):
            self.dims, self.config, self.precision = dims, config, precision
            self.output_dims = dims if output_dims is None else output_dims
            if config.get("otype") == "Empty":
                self.output_dims = 0
            self.jit_fusion = False
            self.backward_backward_input_no_input_grad = False
            self.loss_scale = 128.0 if precision == _Precision.Fp16 else 1.0
            self.calls = []

        def param_precision(self):
            return self.precision

        output_precision = param_precision

        def n_input_dims(self):
            return self.dims

        def n_output_dims(self):
            return self.output_dims

        def n_params(self):
            return 0 if self.config.get("otype") in ("Identity", "Empty") else 2

        def name(self):
            return "MathematicalFixtureOnly"

        def hyperparams(self):
            return dict(self.config)

        def initial_params(self, seed, stream=None):
            values = [] if self.n_params() == 0 else [0.6 + seed * 1e-5, -0.4 - seed * 1e-5]
            return mx.array(values, dtype=mx.float32)

        def forward(self, x, params, stream=None):
            dtype = mx.float16 if self.precision == _Precision.Fp16 else mx.float32
            assert x.dtype == mx.float32 and params.dtype == dtype
            assert x.ndim == 2 and x.shape[1] == self.dims and x.shape[0] % 16 == 0
            self.calls.append((x.shape, params.shape, stream))
            if self.config.get("otype") == "Empty":
                return x[:, :0].astype(dtype)
            if self.config.get("otype") == "Identity":
                y = self.config.get("scale", 1.0) * x + self.config.get("offset", 0.0)
            else:
                y = params[0] * x**3 + params[1] * mx.sin(x)
            if self.output_dims != self.dims:
                y = mx.stack([y[:, index % self.dims] for index in range(self.output_dims)], axis=1)
            return y.astype(dtype, stream=stream)

    fake = types.ModuleType("tinymetalnn_mlx._C")
    fake.Precision = _Precision
    fake.Module = MathematicalFacade
    fake.batch_size_granularity = lambda: 16
    fake.default_loss_scale = lambda p: 128.0 if p == _Precision.Fp16 else 1.0
    fake.preferred_precision = lambda: _Precision.Fp16
    fake.has_networks = lambda: True
    fake.supports_jit_fusion = lambda: True
    fake.free_temporary_memory = lambda: None
    fake.rtc_set_cache_dir = lambda path: setattr(fake, "cache_dir", path)
    fake.rtc_set_include_dir = lambda path: setattr(fake, "include_dir", path)
    fake.guard_shapes = lambda input, params, *, stream=None: (input, params)
    fake.create_encoding = lambda dims, config, precision: MathematicalFacade(dims, config, precision)
    # Networks expose a padded native output; the public wrapper must trim it.
    fake.create_network = lambda dims, outputs, config: MathematicalFacade(dims, config, _Precision.Fp16, ((outputs + 15) // 16) * 16)
    fake.create_network_with_input_encoding = lambda dims, outputs, encoding, network: fake.create_network(dims, outputs, network)

    previous_modules = {key: value for key, value in sys.modules.items() if key == "tinymetalnn_mlx" or key.startswith("tinymetalnn_mlx.")}
    for key in previous_modules:
        del sys.modules[key]
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    monkeypatch.setenv("TCNN_MLX_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    sys.modules["tinymetalnn_mlx._C"] = fake
    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        package = importlib.import_module("tinymetalnn_mlx")
        modules = importlib.import_module("tinymetalnn_mlx.modules")
        monkeypatch.setattr(modules, "_gpu_stream", lambda stream: mx.default_stream(mx.cpu) if stream is None else stream)
        yield package, modules, fake
    finally:
        mx.set_default_device(previous_device)
        for key in tuple(sys.modules):
            if key == "tinymetalnn_mlx" or key.startswith("tinymetalnn_mlx."):
                del sys.modules[key]
        sys.modules.update(previous_modules)


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_wrapper_cast_padding_and_oracle_gradients(wrapper, dtype):
    tcnn, _, _ = wrapper
    model = tcnn.Encoding(2, {"otype": "MathematicalFixtureOnly"}, seed=0, dtype=dtype)
    x = mx.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    y = model(x)
    gx, gp = mx.grad(lambda z, p: mx.sum(model.forward_with_params(z, p).astype(mx.float32)), argnums=(0, 1))(x, model.params)
    p = np.asarray(model.params.astype(dtype), dtype=np.float64)
    z = np.asarray(x, dtype=np.float64)
    expected_y = p[0] * z**3 + p[1] * np.sin(z)
    expected_gx = 3 * p[0] * z**2 + p[1] * np.cos(z)
    expected_gp = np.array([np.sum(z**3), np.sum(np.sin(z))])
    tolerance = dict(rtol=3e-3, atol=5e-4) if dtype == mx.float16 else dict(rtol=3e-6, atol=1e-6)
    assert_close(y, expected_y, **tolerance)
    assert_close(gx, expected_gx, **tolerance)
    assert_close(gp, expected_gp, **tolerance)
    assert y.shape == x.shape and y.dtype == dtype
    assert model.params.dtype == mx.float32 and gp.dtype == mx.float32
    assert all(call[0] == (16, 2) for call in model.native_tcnn_module.calls)


def test_wrapper_second_derivatives_and_both_jvp_arguments(wrapper):
    tcnn, _, _ = wrapper
    model = tcnn.Encoding(2, {"otype": "MathematicalFixtureOnly"}, seed=0, dtype=mx.float32)
    x = mx.array([[0.15, 0.35], [0.55, 0.75]])
    p = mx.array([0.7, -0.4])
    g = mx.array([[0.3, -0.2], [0.4, 0.1]])
    vx = mx.array([[0.2, 0.3], [-0.1, 0.4]])
    vp = mx.array([0.1, -0.2])
    fn = lambda z, q: model.forward_with_params(z, q)
    grad_x = mx.grad(lambda z, q: mx.sum(fn(z, q) * g), argnums=0)
    hxx = mx.grad(lambda z: mx.sum(grad_x(z, p) * vx))(x)
    mixed = mx.grad(lambda q: mx.sum(grad_x(x, q) * vx))(p)
    _, (jvp,) = mx.jvp(fn, [x, p], [vx, vp])
    z, c, v = map(np.asarray, (x, g, vx))
    expected_hxx = (6 * 0.7 * z + 0.4 * np.sin(z)) * c * v
    expected_mixed = [np.sum(3 * z**2 * c * v), np.sum(np.cos(z) * c * v)]
    expected_jvp = (3 * 0.7 * z**2 - 0.4 * np.cos(z)) * v + 0.1 * z**3 - 0.2 * np.sin(z)
    assert_close(hxx, expected_hxx, rtol=3e-6, atol=2e-6)
    assert_close(mixed, expected_mixed, rtol=3e-6, atol=2e-6)
    assert_close(jvp, expected_jvp, rtol=3e-6, atol=2e-6)


def test_wrapper_native_factory_hook_and_network_output_trimming(wrapper):
    tcnn, modules, fake = wrapper

    class CustomModule(modules.Module):
        def __init__(self):
            self.n_input_dims, self.n_output_dims = 2, 2
            super().__init__(seed=0)

        def _native_tcnn_module(self):
            return fake.create_encoding(2, {"otype": "Identity", "scale": 2.0}, fake.Precision.Fp32)

    assert_close(CustomModule()(mx.ones((3, 2))), mx.full((3, 2), 2.0))
    for model in (tcnn.Network(2, 3, {}), tcnn.NetworkWithInputEncoding(2, 3, {}, {})):
        y = model(mx.ones((5, 2)))
        mx.eval(y)
        assert y.shape == (5, 3)
        assert model.native_tcnn_module.n_output_dims() == 16
        assert set(model.parameters()) == {"params"}


def test_wrapper_logical_shapes_reach_replay_guard(wrapper, monkeypatch):
    tcnn, _, fake = wrapper
    seen = []

    def guard(input, params, *, stream=None):
        seen.append((input.shape, input.dtype, params.shape, params.dtype))
        return input, params

    monkeypatch.setattr(fake, "guard_shapes", guard)
    model = tcnn.Encoding(2, {"otype": "MathematicalFixtureOnly"}, dtype=mx.float16)
    result = model.forward_with_params(mx.ones((3, 2)), mx.array([0.2, 0.3], dtype=mx.float64))
    mx.eval(result)
    assert seen == [((3, 2), mx.float32, (2,), mx.float16)]
    assert model.native_tcnn_module.calls[-1][0] == (16, 2)


def test_wrapper_parameters_freeze_update_and_apply(wrapper):
    tcnn, _, _ = wrapper
    config = {"otype": "MathematicalFixtureOnly", "nested": {"metadata": [1, 2]}}
    model = tcnn.Encoding(2, config, dtype=mx.float32)
    assert set(model) == {"params"}
    assert set(model.parameters()) == {"params"}
    original = model.params
    model.forward_with_params(mx.ones((2, 2)), original + 1)
    assert model.params is original
    model.apply(lambda p: p * 2)
    assert_close(model.params, original * 2)
    model.freeze()
    assert model.trainable_parameters() == {}
    model.unfreeze()
    model.update({"params": mx.array([0.1, 0.2])})
    assert_close(model.params, [0.1, 0.2])
    assert set(model.trainable_parameters()) == {"params"}


@pytest.mark.parametrize("batch", [0, 1, 17])
def test_wrapper_zero_parameters_empty_batches_and_dimensions(wrapper, batch):
    tcnn, _, _ = wrapper
    for dims, kind in ((2, "Identity"), (2, "Empty"), (0, "Identity")):
        model = tcnn.Encoding(dims, {"otype": kind}, dtype=mx.float32)
        x = mx.ones((batch, dims))
        y = model(x)
        gx, gp = mx.grad(lambda z, p: mx.sum(model.forward_with_params(z, p)), argnums=(0, 1))(x, model.params)
        _, (directional,) = mx.jvp(model, [x], [mx.ones_like(x)])
        grouped = mx.vmap(model)(mx.stack((x, x)))
        mx.eval(y, gx, gp, directional, grouped)
        assert y.shape == (batch, 0 if kind == "Empty" else dims)
        assert gx.shape == x.shape and gp.shape == (0,)
        assert directional.shape == y.shape and grouped.shape == (2, *y.shape)


def test_wrapper_is_lazy_and_preserves_cpu_double_leaf_gradients(wrapper, monkeypatch):
    tcnn, _, _ = wrapper
    model = tcnn.Encoding(2, {"otype": "Identity", "scale": 2.0}, dtype=mx.float32)
    x = mx.array([[0.2, 0.3]], dtype=mx.float64)
    with monkeypatch.context() as guard:
        guard.setattr(mx, "eval", lambda *args: pytest.fail("The wrapper eagerly evaluated a graph"))
        pending = model(x)
    assert_close(pending, [[0.4, 0.6]])
    gx = mx.grad(lambda z: mx.sum(model(z)))(x)
    assert gx.dtype == mx.float64
    assert_close(gx, [[2.0, 2.0]], rtol=0, atol=0)


def test_wrapper_functional_double_parameter_leaf_gradient(wrapper):
    tcnn, _, _ = wrapper
    model = tcnn.Encoding(2, {"otype": "MathematicalFixtureOnly"}, dtype=mx.float32)
    x = mx.array([[0.2, 0.3], [0.4, 0.5]])
    p = mx.array([0.7, -0.4], dtype=mx.float64)
    gradient = mx.grad(lambda q: mx.sum(model.forward_with_params(x, q)))(p)
    assert gradient.dtype == mx.float64
    z = np.asarray(x, dtype=np.float64)
    assert_close(gradient, [np.sum(z**3), np.sum(np.sin(z))], rtol=2e-6, atol=2e-6)


def test_wrapper_compiled_replay_and_vmap_use_live_explicit_params(wrapper):
    tcnn, _, _ = wrapper
    model = tcnn.Encoding(2, {"otype": "MathematicalFixtureOnly"}, dtype=mx.float32)
    fun = lambda x, p: model.forward_with_params(x, p)
    compiled = mx.compile(fun)
    x = mx.full((3, 2), 0.4)
    for p in (mx.array([0.2, 0.3]), mx.array([-0.1, 0.4])):
        assert_close(compiled(x, p), p[0] * x**3 + p[1] * mx.sin(x))
    ps = mx.array([[0.2, 0.3], [-0.1, 0.4]])
    mapped = mx.vmap(fun, in_axes=(None, 0))(x, ps)
    expected = mx.stack([fun(x, ps[index]) for index in range(2)])
    assert_close(mapped, expected)


@pytest.mark.parametrize("suffix", ["npz", "safetensors"])
def test_wrapper_weights_and_pickle_recreate_native_handle(wrapper, tmp_path, suffix):
    tcnn, _, _ = wrapper
    model = tcnn.Encoding(2, {"otype": "MathematicalFixtureOnly"}, seed=7, dtype=mx.float32)
    model.params = mx.array([0.2, 0.8])
    model.jit_fusion = True
    model.backward_backward_input_no_input_grad = True
    model.loss_scale = 32.0
    filename = tmp_path / f"weights.{suffix}"
    model.save_weights(str(filename))
    restored = tcnn.Encoding(2, {"otype": "MathematicalFixtureOnly"}, dtype=mx.float32)
    restored.load_weights(str(filename))
    assert_close(restored.params, model.params, rtol=0, atol=0)
    restored_pickle = pickle.loads(pickle.dumps(model))
    assert restored_pickle.native_tcnn_module is not model.native_tcnn_module
    assert restored_pickle.jit_fusion and restored_pickle.backward_backward_input_no_input_grad
    assert restored_pickle.loss_scale == 32.0
    assert set(restored_pickle.parameters()) == {"params"}
    assert_close(restored_pickle(mx.ones((3, 2))), model(mx.ones((3, 2))), rtol=0, atol=0)


def test_wrapper_mlx_nn_optimizer_updates_flat_params(wrapper):
    import mlx.optimizers as optim
    tcnn, _, _ = wrapper
    model = tcnn.Encoding(2, {"otype": "MathematicalFixtureOnly"}, seed=0, dtype=mx.float32)
    optimizer = optim.Adam(learning_rate=0.05)
    x = mx.linspace(0.1, 0.9, 34).reshape(17, 2)
    target = 0.3 * x**3 + 0.1 * mx.sin(x)
    loss = lambda m: mx.mean(mx.square(m(x) - target))
    value_and_grad = nn.value_and_grad(model, loss)
    before = loss(model).item()
    for _ in range(20):
        value, gradients = value_and_grad(model)
        optimizer.update(model, gradients)
        mx.eval(value, model.parameters(), optimizer.state)
        assert set(gradients) == {"params"}
    assert loss(model).item() < before


def test_wrapper_precise_invalid_argument_and_no_network_errors(wrapper, monkeypatch):
    tcnn, _, fake = wrapper
    model = tcnn.Encoding(2, {"otype": "Identity"}, dtype=mx.float32)
    for value in ([], mx.zeros((3,)), mx.zeros((3, 3)), mx.zeros((1, 2, 2))):
        with pytest.raises((TypeError, ValueError)):
            model(value)
    with pytest.raises(ValueError):
        model.forward_with_params(mx.zeros((2, 2)), mx.zeros((1,)))
    with pytest.raises(ValueError):
        tcnn.Encoding(2, {}, dtype=mx.bfloat16)
    monkeypatch.setattr(fake, "has_networks", lambda: False)
    with pytest.raises(RuntimeError, match="network"):
        tcnn.Network(2, 3, {})
    with pytest.raises(RuntimeError, match="network"):
        tcnn.NetworkWithInputEncoding(2, 3, {}, {})
    assert_close(model(mx.ones((2, 2))), mx.ones((2, 2)))
