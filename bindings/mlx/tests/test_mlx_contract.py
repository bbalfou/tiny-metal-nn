"""Public MLX integration, lazy ownership, stream ordering and serialization."""

import gc
import importlib.util
import inspect
import pickle
import subprocess
import sys
import threading

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytestmark = pytest.mark.metal

from helpers import GRID, NETWORK, assert_close, assert_finite


def test_public_classes_and_constructor_defaults(tcnn, native):
    from tinymetalnn_mlx.modules import Module
    assert issubclass(Module, nn.Module)
    for cls in (tcnn.Encoding, tcnn.Network, tcnn.NetworkWithInputEncoding):
        assert issubclass(cls, Module)
        assert inspect.signature(cls).parameters["seed"].default == 1337
    assert inspect.signature(tcnn.Encoding).parameters["dtype"].default is None
    assert native.Precision.Fp32.value == 0 and native.Precision.Fp16.value == 1
    assert native.Fp32 == native.Precision.Fp32 and native.Fp16 == native.Precision.Fp16
    for name in ("Info", "Debug", "Warning", "Error", "Success"):
        assert getattr(native, name) == getattr(native.LogSeverity, name)
    assert native.batch_size_granularity() > 0


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_master_parameter_tree_and_native_metadata(tcnn, native, dtype):
    model = tcnn.Encoding(2, GRID, seed=19, dtype=dtype)
    other = tcnn.Encoding(2, GRID, seed=19, dtype=dtype)
    different = tcnn.Encoding(2, GRID, seed=20, dtype=dtype)
    assert model.params.dtype == mx.float32
    assert model.dtype == dtype
    assert set(model.parameters()) == {"params"}
    assert model.params.ndim == 1 and model.params.size == model.n_params
    assert model.native_tcnn_module.n_input_dims() == 2
    assert model.native_tcnn_module.n_params() == model.n_params
    assert isinstance(model.native_tcnn_module.hyperparams(), dict)
    assert isinstance(model.native_tcnn_module.name(), str)
    assert_close(model.params, other.params, rtol=0, atol=0)
    assert mx.any(model.params != different.params).item()
    assert model.loss_scale == native.default_loss_scale(
        native.Precision.Fp16 if dtype == mx.float16 else native.Precision.Fp32
    )


def test_loss_scale_is_snapshotted_by_lazy_derivative_graphs(tcnn):
    model = tcnn.Encoding(2, {"otype": "Identity", "scale": 1.5}, dtype=mx.float32)
    x = mx.full((3, 2), 0.3)
    model.loss_scale = 8.0
    pending = mx.grad(lambda z: mx.sum(mx.square(model(z))))(x)
    model.loss_scale = 2.0
    assert_close(pending, 2 * 1.5 * 1.5 * x, rtol=2e-6, atol=2e-6)
    assert_close(mx.grad(lambda z: mx.sum(mx.square(model(z))))(x), pending, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("batch", [0, 1, 13, 255, 256, 257])
@pytest.mark.parametrize("config", [{"otype": "Identity", "scale": 1.5}, {"otype": "Empty"}, GRID])
def test_empty_padding_and_parameterless_gradients(tcnn, batch, config):
    model = tcnn.Encoding(2, config, dtype=mx.float32)
    x = mx.full((batch, 2), 0.3)
    y = model(x)
    value, (gx, gp) = mx.value_and_grad(
        lambda z, p: mx.sum(model.forward_with_params(z, p)), argnums=(0, 1)
    )(x, model.params)
    assert y.shape == (batch, model.n_output_dims)
    assert gx.shape == x.shape and gp.shape == model.params.shape
    assert_finite(y, value, gx, gp)
    if config["otype"] == "Identity":
        assert_close(y, 1.5 * x)
        assert_close(gx, mx.full_like(x, 1.5))
    if batch == 0 or config["otype"] == "Empty":
        assert_close(gx, mx.zeros_like(x), rtol=0, atol=0)
        assert_close(gp, mx.zeros_like(gp), rtol=0, atol=0)


def test_zero_width_identity(tcnn):
    model = tcnn.Encoding(0, {"otype": "Identity"}, dtype=mx.float32)
    x = mx.zeros((3, 0))
    y = model(x)
    gx = mx.grad(lambda z: mx.sum(model(z)))(x)
    mx.eval(y, gx)
    assert y.shape == (3, 0) and gx.shape == (3, 0)


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32, mx.float64, mx.int32])
def test_cpu_produced_arrays_and_input_casting(tcnn, dtype):
    # MLX arrays are unified-memory values. CPU production is valid; the native
    # operation and its casts are explicitly scheduled on the GPU.
    model = tcnn.Encoding(2, {"otype": "Identity", "scale": 2.0}, dtype=mx.float32)
    with mx.stream(mx.cpu):
        x = mx.array([[1, 2], [3, 4]], dtype=dtype)
    y = model(x)
    assert_close(y, [[2, 4], [6, 8]])
    if dtype in (mx.float16, mx.float32, mx.float64):
        # MLX seeds float64 gradient bookkeeping on the active stream, even
        # when the function casts to float32. Float64 transforms need a CPU
        # context; the model still explicitly schedules its native work on GPU.
        with mx.stream(mx.cpu if dtype == mx.float64 else mx.gpu):
            gx = mx.grad(lambda z: mx.sum(model(z)))(x)
        assert gx.dtype == dtype
        assert_close(gx, [[2, 2], [2, 2]])


def test_noncontiguous_and_offset_views(tcnn):
    model = tcnn.Encoding(2, {"otype": "Identity", "scale": 1.75, "offset": -0.5}, dtype=mx.float32)
    base = mx.arange(60, dtype=mx.float32).reshape(2, 30)
    for x in (base.T[3:16], base[:, ::3].T, base.T[4:12][::-1]):
        y, gx = mx.value_and_grad(lambda z: mx.sum(mx.sin(model(z))))(x)
        assert_close(y, mx.sum(mx.sin(1.75 * x - 0.5)))
        assert_close(gx, 1.75 * mx.cos(1.75 * x - 0.5))


@pytest.mark.parametrize("shape", [(2,), (2, 3), (1, 2, 2)])
def test_bad_input_shapes_fail(tcnn, shape):
    model = tcnn.Encoding(2, {"otype": "Identity"})
    with pytest.raises((ValueError, RuntimeError)):
        mx.eval(model(mx.zeros(shape)))


def test_explicit_cpu_operation_stream_fails(tcnn):
    model = tcnn.Encoding(2, {"otype": "Identity"})
    with pytest.raises((ValueError, RuntimeError), match="[Mm]etal|[Gg][Pp][Uu]|[Cc][Pp][Uu]"):
        mx.eval(model(mx.zeros((2, 2)), stream=mx.cpu))


@pytest.mark.parametrize("problem", ["dtype", "rank", "width", "batch", "param_dtype", "param_rank", "param_count", "cpu_stream"])
def test_native_facade_rejects_invalid_contracts(native, problem):
    module = native.create_encoding(2, {"otype": "Identity"}, native.Precision.Fp32)
    batch = native.batch_size_granularity()
    x = mx.zeros((batch, 2), dtype=mx.float32)
    params = module.initial_params(17)
    stream = mx.gpu
    if problem == "dtype":
        x = x.astype(mx.float16)
    elif problem == "rank":
        x = x.reshape(batch, 1, 2)
    elif problem == "width":
        x = mx.zeros((batch, 3))
    elif problem == "batch":
        x = mx.zeros((batch + 1, 2))
    elif problem == "param_dtype":
        params = params.astype(mx.float16)
    elif problem == "param_rank":
        params = params.reshape(0, 1)
    elif problem == "param_count":
        params = mx.zeros((1,))
    else:
        stream = mx.cpu
    with pytest.raises((RuntimeError, ValueError)):
        mx.eval(module.forward(x, params, stream=stream))


def test_native_facade_offsets_strides_and_parameter_views(native):
    batch = native.batch_size_granularity()
    identity = native.create_encoding(2, {"otype": "Identity", "scale": 1.75}, native.Precision.Fp32)
    params = identity.initial_params(17)
    base = mx.linspace(0.1, 0.9, 2 * (batch + 7)).reshape(batch + 7, 2)
    for x in (base[3 : 3 + batch], mx.linspace(0.1, 0.9, 2 * batch).reshape(2, batch).T):
        assert_close(identity.forward(x, params), x * 1.75)
    grid = native.create_encoding(2, GRID, native.Precision.Fp32)
    p = grid.initial_params(17)
    storage = mx.concatenate((mx.ones((5,)), p, mx.ones((3,))))
    x = mx.full((batch, 2), 0.3)
    assert_close(grid.forward(x, storage[5 : 5 + p.size]), grid.forward(x, p), rtol=0, atol=0)


def test_functional_double_parameter_leaf(tcnn):
    model = tcnn.Encoding(2, GRID, dtype=mx.float32)
    with mx.stream(mx.cpu):
        parameters = mx.linspace(-0.2, 0.2, model.n_params, dtype=mx.float64)
    x = mx.full((3, 2), 0.3)
    actual = mx.grad(lambda p: mx.sum(model.forward_with_params(x, p)))(parameters)
    expected = mx.grad(lambda p: mx.sum(model.forward_with_params(x, p)))(parameters.astype(mx.float32, stream=mx.cpu))
    assert actual.dtype == mx.float64
    assert_close(actual, expected, rtol=0, atol=0)


def test_mlx_parameter_updates_freeze_and_apply(tcnn):
    model = tcnn.Encoding(2, GRID, dtype=mx.float32)
    before = model.params
    model.apply(lambda value: value * 2)
    assert_close(model.params, before * 2)
    replacement = mx.full_like(model.params, 0.125)
    model.update({"params": replacement})
    assert_close(model.params, replacement, rtol=0, atol=0)
    model.freeze()
    assert model.trainable_parameters() == {}
    model.unfreeze()
    assert set(model.trainable_parameters()) == {"params"}
    model.eval()
    assert not model.training
    model.train()
    assert model.training


@pytest.mark.parametrize("suffix", ["npz", "safetensors"])
def test_mlx_weight_roundtrip(tcnn, tmp_path, suffix):
    model = tcnn.Encoding(2, GRID, seed=17, dtype=mx.float32)
    model.params = model.params + 0.125
    x = mx.full((7, 2), 0.3)
    expected = model(x)
    path = tmp_path / f"weights.{suffix}"
    model.save_weights(str(path))
    restored = tcnn.Encoding(2, GRID, seed=99, dtype=mx.float32)
    restored.load_weights(str(path))
    assert_close(restored.params, model.params, rtol=0, atol=0)
    assert_close(restored(x), expected, rtol=0, atol=0)
    assert set(restored.parameters()) == {"params"}


def test_pickle_recreates_native_factory_and_preserves_flags(tcnn):
    model = tcnn.Encoding(2, GRID, seed=17, dtype=mx.float32)
    model.params = model.params + 0.1
    model.backward_backward_input_no_input_grad = True
    model.loss_scale = 32.0
    model.jit_fusion = bool(tcnn.supports_jit_fusion())
    restored = pickle.loads(pickle.dumps(model))
    assert restored.native_tcnn_module is not model.native_tcnn_module
    assert restored.dtype == model.dtype and restored.seed == model.seed
    assert restored.jit_fusion == model.jit_fusion
    assert restored.backward_backward_input_no_input_grad
    assert restored.loss_scale == 32.0
    assert set(restored.parameters()) == {"params"}
    assert_close(restored.params, model.params, rtol=0, atol=0)
    x = mx.full((5, 2), 0.3)
    assert_close(restored(x), model(x), rtol=0, atol=0)


def test_lazy_graphs_keep_parameter_snapshots_and_owners(tcnn):
    model = tcnn.Encoding(2, GRID, dtype=mx.float32)
    x = mx.full((11, 2), 0.37)
    original_params = model.params
    expected = model.forward_with_params(x, original_params)
    mx.eval(expected)
    pending = model(x)
    pending_grad = mx.grad(lambda z: mx.sum(model.forward_with_params(z, original_params)))(x)
    model.params = model.params + 1
    changed = model(x)
    # No pending graph has run yet. Deleting its Python module must not destroy
    # the native factory state or storage needed by eventual evaluation.
    del model
    gc.collect()
    tcnn.free_temporary_memory()
    mx.async_eval(pending, pending_grad, changed)
    assert_close(pending, expected, rtol=0, atol=0)
    assert_close(changed, expected + 1, rtol=2e-5, atol=2e-5)
    assert_finite(pending_grad)


def test_stream_selection_and_mixed_mlx_ordering(tcnn):
    model = tcnn.Encoding(2, {"otype": "Identity", "scale": 1.25, "offset": 0.2}, dtype=mx.float32)
    producer = mx.new_stream(mx.gpu)
    consumer = mx.new_stream(mx.gpu)
    with mx.stream(producer):
        x = mx.sin(mx.arange(514, dtype=mx.float32).reshape(257, 2) * 0.01)
        native_y = model(x)
        native_gx = mx.grad(lambda z: mx.sum(mx.square(model(z))))(x)
    with mx.stream(consumer):
        observed = mx.cos(native_y) + native_gx
        expected = mx.cos(1.25 * x + 0.2) + 2 * 1.25 * (1.25 * x + 0.2)
    mx.async_eval(observed, expected)
    assert_close(observed, expected)
    # Explicit scheduling also composes with a producer from another stream.
    assert_close(model(x, stream=consumer), 1.25 * x + 0.2)


def test_compile_replays_dynamic_parameter_values(tcnn):
    model = tcnn.Encoding(2, GRID, dtype=mx.float32)
    fn = lambda x, p: mx.sin(model.forward_with_params(x, p))
    compiled = mx.compile(fn)
    compiled_grad = mx.compile(mx.value_and_grad(lambda x, p: mx.sum(fn(x, p)), argnums=(0, 1)))
    for batch, shift in ((3, 0.0), (3, 0.2), (7, -0.15), (0, 0.0)):
        x = mx.full((batch, 2), 0.37)
        p = model.params + shift
        assert_close(compiled(x, p), fn(x, p))
        actual_value, actual_grads = compiled_grad(x, p)
        expected_value, expected_grads = mx.value_and_grad(lambda z, q: mx.sum(fn(z, q)), argnums=(0, 1))(x, p)
        for actual, expected in zip((actual_value, *actual_grads), (expected_value, *expected_grads)):
            assert_close(actual, expected)


def test_shapeless_replay_rejects_shape_changes_explicitly(tcnn):
    model = tcnn.Encoding(2, {"otype": "Identity"}, dtype=mx.float32)
    compiled = mx.compile(lambda x: model(x), shapeless=True)
    assert_close(compiled(mx.ones((3, 2))), mx.ones((3, 2)))
    with pytest.raises((RuntimeError, ValueError), match="[Ss]hap|replay"):
        mx.eval(compiled(mx.ones((7, 2))))


def test_zero_parameter_jvp_keeps_shapeless_guard(tcnn):
    model = tcnn.Encoding(2, {"otype": "Identity"}, dtype=mx.float32)
    tangent = mx.zeros_like(model.params)
    only_tangent = lambda x: mx.jvp(lambda p: model.forward_with_params(x, p), [model.params], [tangent])[1][0]
    compiled = mx.compile(only_tangent, shapeless=True)
    assert_close(compiled(mx.ones((3, 2))), mx.zeros((3, 2)), rtol=0, atol=0)
    with pytest.raises((RuntimeError, ValueError), match="[Ss]hap|replay"):
        mx.eval(compiled(mx.ones((7, 2))))


def test_suppressed_curvature_keeps_shapeless_guard(tcnn):
    model = tcnn.Encoding(2, {"otype": "Frequency", "n_frequencies": 2}, dtype=mx.float32)
    model.backward_backward_input_no_input_grad = True
    first = mx.grad(lambda z: mx.sum(model(z)))
    compiled = mx.compile(mx.grad(lambda z: mx.sum(first(z))), shapeless=True)
    assert_close(compiled(mx.ones((3, 2))), mx.zeros((3, 2)), rtol=0, atol=0)
    with pytest.raises((RuntimeError, ValueError), match="[Ss]hap|replay"):
        mx.eval(compiled(mx.ones((7, 2))))


def test_mlx_optimizer_training_and_checkpoint(networks, tmp_path):
    import mlx.optimizers as optim
    model = networks.Network(2, 1, {**NETWORK, "n_hidden_layers": 0})
    optimizer = optim.Adam(learning_rate=0.03)
    x = mx.linspace(0.1, 0.9, 34).reshape(17, 2)
    target = 0.3 * x[:, :1] - 0.2 * x[:, 1:]
    loss = lambda m, z, y: mx.mean(mx.square(m(z).astype(mx.float32) - y))
    value_and_grad = nn.value_and_grad(model, loss)
    before = loss(model, x, target).item()
    for _ in range(30):
        value, gradients = value_and_grad(model, x, target)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, value)
        assert gradients["params"].dtype == mx.float32
    after = loss(model, x, target).item()
    assert after < before, (before, after)
    model.save_weights(str(tmp_path / "linear.npz"))


@pytest.mark.parametrize("otype", [None, "GemmMLP", "FullyFusedMLP"])
def test_jit_capability_and_no_silent_fallback(tcnn, otype):
    if not tcnn.supports_jit_fusion():
        pytest.skip("Native JIT is unavailable in this explicit build/device")
    if otype is None:
        model = tcnn.Encoding(2, {"otype": "Frequency", "n_frequencies": 2}, dtype=mx.float32)
    else:
        from tinymetalnn_mlx import _C
        if not _C.has_networks():
            pytest.skip("Encoding-only build")
        model = tcnn.Network(2, 3, {
            "otype": otype, "n_neurons": 16, "n_hidden_layers": 1,
            "activation": "Tanh", "output_activation": "None",
        })
    x = mx.full((5, 2), 0.3)
    objective = lambda z, p: mx.sum(mx.square(model.forward_with_params(z, p).astype(mx.float32)))
    results = []
    for enabled in (False, True):
        model.jit_fusion = enabled
        y = model(x)
        gx, gp = mx.grad(objective, argnums=(0, 1))(x, model.params)
        hvp = mx.grad(lambda z: mx.sum(mx.grad(objective)(z, model.params)))(x)
        mx.eval(y, gx, gp, hvp)
        results.append((y, gx, gp, hvp))
        assert model.jit_fusion == enabled, "Native JIT fell back to AOT"
    tolerance = dict(rtol=2e-2, atol=2e-2) if model.dtype == mx.float16 else dict(rtol=2e-3, atol=5e-4)
    for actual, expected in zip(results[1], results[0]):
        assert_close(actual, expected, **tolerance)


def test_lazy_logging_delivery_reentry_and_callback_exceptions(tcnn, native, tmp_path):
    if not tcnn.supports_jit_fusion():
        pytest.skip("Diagnostic trigger needs native JIT in this build/device")
    invalid_directory = tmp_path / "cache-is-a-file"
    invalid_directory.write_text("Force the source-cache diagnostic without breaking MSL compilation.")
    model = tcnn.Encoding(2, {"otype": "Frequency", "n_frequencies": 3}, dtype=mx.float32)
    model.jit_fusion = True
    caller = threading.get_ident()
    records = []

    def callback(severity, message):
        assert threading.get_ident() == caller
        assert model.jit_fusion
        mx.synchronize()
        records.append((severity, message))

    class CallbackFailure(RuntimeError):
        pass

    def raising_callback(severity, message):
        raise CallbackFailure("queued MLX diagnostic delivered on caller")

    native.rtc_set_cache_dir(str(invalid_directory))
    native.set_log_callback(callback)
    try:
        pending = model(mx.full((11, 2), 0.2))
        assert records == []
        mx.eval(pending)
        # MLX evaluation is a call into MLX, not the binding; Python callbacks
        # must remain queued instead of running on its scheduler worker.
        assert records == []
        native.flush_logs()
        assert any("cache unavailable" in message for _, message in records)
        assert_finite(pending)

        other = tcnn.Encoding(2, {"otype": "Frequency", "n_frequencies": 5}, dtype=mx.float32)
        other.jit_fusion = True
        native.set_log_callback(raising_callback)
        mx.eval(other(mx.full((7, 2), 0.3)))
        with pytest.raises(CallbackFailure, match="queued MLX diagnostic"):
            native.batch_size_granularity()  # Next binding entry also delivers.
    finally:
        native.set_log_callback(lambda severity, message: None)
        native.rtc_set_cache_dir("")
        native.flush_logs()


def test_package_has_no_torch_dependency(tcnn):
    completed = subprocess.run(
        [sys.executable, "-c", "import sys; import tinymetalnn_mlx; assert 'torch' not in sys.modules"],
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_optional_torch_extension_coexistence(tcnn):
    if importlib.util.find_spec("tinymetalnn") is None:
        pytest.skip("Optional PyTorch binding is not installed")
    import torch
    import tinymetalnn as torch_tcnn
    if not torch.backends.mps.is_available():
        pytest.skip("Optional PyTorch MPS device is unavailable")
    mlx_model = tcnn.Encoding(2, {"otype": "Frequency", "n_frequencies": 2}, dtype=mx.float32)
    torch_model = torch_tcnn.Encoding(2, {"otype": "Frequency", "n_frequencies": 2}, dtype=torch.float32)
    x = mx.full((3, 2), 0.25)
    actual = mlx_model(x)
    expected = torch_model(torch.full((3, 2), 0.25, device="mps")).cpu().detach().numpy()
    assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    assert_close(mlx_model(x), expected, rtol=3e-5, atol=3e-6)
