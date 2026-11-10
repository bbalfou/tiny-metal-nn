"""Public/low-level contracts, tensor views, ownership, and ordering on MPS."""

import gc
import importlib
import inspect
import io
import pickle
import threading

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.mps


GRID = {
    "otype": "HashGrid",
    "n_levels": 2,
    "n_features_per_level": 2,
    "log2_hashmap_size": 4,
    "base_resolution": 4,
    "per_level_scale": 2.0,
}
NETWORK = {
    "otype": "GemmMLP",
    "n_neurons": 16,
    "n_hidden_layers": 1,
    "activation": "Tanh",
    "output_activation": "None",
}


def test_public_signatures(tcnn):
    assert set(tcnn.__all__) == {
        "supports_jit_fusion", "free_temporary_memory", "NetworkWithInputEncoding", "Network", "Encoding"
    }
    assert str(inspect.signature(tcnn.Encoding)) == "(n_input_dims, encoding_config, seed=1337, dtype=None)"
    assert str(inspect.signature(tcnn.Network)) == "(n_input_dims, n_output_dims, network_config, seed=1337)"
    assert str(inspect.signature(tcnn.NetworkWithInputEncoding)) == (
        "(n_input_dims, n_output_dims, encoding_config, network_config, seed=1337)"
    )
    assert issubclass(tcnn.Encoding, importlib.import_module("tinymetalnn.modules").Module)
    assert issubclass(tcnn.Network, torch.nn.Module)


def test_native_enums_and_encoding_only_capability(tcnn, native):
    assert native.Precision.Fp32.value == 0
    assert native.Precision.Fp16.value == 1
    assert [getattr(native.LogSeverity, name).value for name in ("Info", "Debug", "Warning", "Error", "Success")] == list(range(5))
    if not native.has_networks():
        with pytest.raises(RuntimeError, match="network"):
            tcnn.Network(2, 3, NETWORK)
        with pytest.raises(RuntimeError, match="network"):
            tcnn.NetworkWithInputEncoding(2, 3, {"otype": "Identity"}, NETWORK)
    assert tcnn.Encoding(2, {"otype": "Identity"}, dtype=torch.float32).params.numel() == 0


@pytest.mark.parametrize("dtype", [None, torch.float16, torch.float32])
def test_master_parameter_and_execution_precision(tcnn, native, dtype):
    encoding = tcnn.Encoding(2, GRID, dtype=dtype)
    expected_dtype = dtype or (torch.float16 if native.preferred_precision() == native.Precision.Fp16 else torch.float32)
    assert encoding.params.dtype == torch.float32
    assert encoding.params.device.type == "mps"
    assert list(dict(encoding.named_parameters())) == ["params"]
    assert encoding.dtype == expected_dtype
    assert encoding.loss_scale == (128.0 if expected_dtype == torch.float16 else 1.0)
    x = torch.rand(13, 2, device="mps", requires_grad=True)
    y = encoding(x)
    assert y.shape == (13, 4)
    assert y.dtype == expected_dtype
    y.float().square().mean().backward()
    assert encoding.params.grad is not None
    assert encoding.params.grad.dtype == torch.float32
    assert torch.isfinite(x.grad).all().item()


def test_seed_and_metadata(tcnn):
    a = tcnn.Encoding(2, GRID, seed=41, dtype=torch.float32)
    b = tcnn.Encoding(2, GRID, seed=41, dtype=torch.float32)
    c = tcnn.Encoding(2, GRID, seed=42, dtype=torch.float32)
    torch.testing.assert_close(a.params, b.params, rtol=0, atol=0)
    assert not torch.equal(a.params, c.params)
    assert a.native_tcnn_module.n_params() == a.params.numel()
    assert a.native_tcnn_module.n_input_dims() == 2
    assert isinstance(a.native_tcnn_module.hyperparams(), dict)
    assert isinstance(a.native_tcnn_module.name(), str)
    assert "seed=41" in repr(a)
    with pytest.raises(ValueError):
        tcnn.Encoding(2, GRID, dtype=torch.bfloat16)
    with pytest.raises(AttributeError):
        del a.jit_fusion
    with pytest.raises(AttributeError):
        del a.backward_backward_input_no_input_grad


@pytest.mark.parametrize("batch", [0, 1, 13, 255, 256, 257])
@pytest.mark.parametrize("otype", ["Identity", "Empty", "HashGrid"])
def test_empty_and_unaligned_batches(tcnn, batch, otype):
    config = GRID if otype == "HashGrid" else {"otype": otype}
    encoding = tcnn.Encoding(2, config, dtype=torch.float32)
    x = torch.rand(batch, 2, device="mps", requires_grad=True)
    output = encoding(x)
    assert output.shape == (batch, encoding.n_output_dims)
    assert output.device.type == "mps"
    output.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all().item()
    if batch == 0 or otype == "Empty":
        torch.testing.assert_close(x.grad, torch.zeros_like(x))
        if encoding.params.grad is not None:
            torch.testing.assert_close(encoding.params.grad, torch.zeros_like(encoding.params))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_input_adaptation_and_noncontiguous_input(tcnn, dtype):
    encoding = tcnn.Encoding(3, {"otype": "Identity", "scale": 1.5, "offset": -0.25}, dtype=torch.float32)
    x = torch.arange(21, dtype=dtype).reshape(3, 7).t().requires_grad_()
    assert not x.is_contiguous()
    with pytest.warns(UserWarning, match="MPS"):
        y = encoding(x)
    torch.testing.assert_close(y.cpu(), (x * 1.5 - 0.25).float())
    y.sum().backward()
    assert x.grad.dtype == dtype
    torch.testing.assert_close(x.grad, torch.full_like(x, 1.5))


@pytest.mark.parametrize("otype", ["Empty", "Identity"])
def test_zero_input_width(tcnn, otype):
    model = tcnn.Encoding(0, {"otype": otype}, dtype=torch.float32)
    x = torch.empty(3, 0, device="mps", requires_grad=True)
    y = model(x)
    assert y.shape == (3, 0)
    y.sum().backward()
    assert x.grad is not None and x.grad.shape == (3, 0)


@pytest.mark.parametrize("shape", [(3,), (3, 3), (1, 2, 2)])
def test_wrapper_rejects_wrong_shape(tcnn, shape):
    encoding = tcnn.Encoding(2, {"otype": "Identity"})
    with pytest.raises((ValueError, RuntimeError)):
        encoding(torch.zeros(shape, device="mps"))


def native_identity(native, requires_grad=True):
    module = native.create_encoding(2, {"otype": "Identity", "scale": 1.75, "offset": -0.5}, native.Precision.Fp32)
    params = module.initial_params(1337)
    x = torch.rand(native.batch_size_granularity(), 2, device="mps", requires_grad=requires_grad)
    return module, x, params


def test_saved_context_across_current_mps_streams(native):
    """Exercise real public stream selection when supplied by installed Torch.

    Torch 2.7 has only a default MPS stream. Newer versions exposing the public
    Stream/context-manager APIs must preserve context dependencies in both
    directions without a host wait between the three native submissions.
    """
    if not all(hasattr(torch.mps, name) for name in ("Stream", "stream")):
        pytest.skip("Installed PyTorch does not expose public MPS stream selection")
    if native.has_networks():
        module = native.create_network(2, 3, NETWORK)
    else:
        module = native.create_encoding(2, GRID, native.Precision.Fp32)
    count = native.batch_size_granularity()
    x = torch.rand(count, 2, device="mps", requires_grad=True)
    params = module.initial_params(1337).to(
        torch.float16 if module.param_precision() == native.Precision.Fp16 else torch.float32
    ).requires_grad_()
    dtype = torch.float16 if module.output_precision() == native.Precision.Fp16 else torch.float32
    dy = torch.full((count, module.n_output_dims()), 0.5, dtype=dtype, device="mps", requires_grad=True)
    vector = torch.full_like(x, 0.25)
    reference_x = x * 1.25 + 0.03125
    reference_dy = dy * 0.75
    reference_vector = vector * 1.5
    reference_context, reference_output = module.fwd(reference_x, params)
    reference_first = module.bwd(reference_context, reference_x, params, reference_output, reference_dy)
    reference_second = module.bwd_bwd_input(reference_context, reference_x, params, reference_vector, reference_dy)
    reference_observed_output = reference_output * 3 + 2
    reference_observed_first = tuple(value * 2 for value in reference_first)
    reference_observed_second = tuple(value * 0.5 for value in reference_second)
    torch.mps.synchronize()

    producer = torch.mps.Stream()
    consumer = torch.mps.Stream()
    with torch.mps.stream(producer):
        # These Torch producers must run before the native operation, and the
        # Torch consumers after it, on this exact selected stream. Merely moving
        # all native calls onto default would fail to preserve these dependencies.
        stream_x = x * 1.25 + 0.03125
        context, output = module.fwd(stream_x, params)
        observed_output = output * 3 + 2
    with torch.mps.stream(consumer):
        stream_dy = dy * 0.75
        first = module.bwd(context, stream_x, params, output, stream_dy)
        observed_first = tuple(value * 2 for value in first)
    with torch.mps.stream(producer):
        stream_vector = vector * 1.5
        second = module.bwd_bwd_input(context, stream_x, params, stream_vector, stream_dy)
        observed_second = tuple(value * 0.5 for value in second)
    producer.synchronize()
    consumer.synchronize()
    tolerance = dict(rtol=3e-3, atol=3e-3) if dtype == torch.float16 else dict(rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(output, reference_output, **tolerance)
    torch.testing.assert_close(observed_output, reference_observed_output, **tolerance)
    for actual, expected in zip((*first, *second), (*reference_first, *reference_second)):
        assert (actual is None) == (expected is None)
        if actual is not None:
            torch.testing.assert_close(actual, expected, **tolerance)
    for actual, expected in zip((*observed_first, *observed_second), (*reference_observed_first, *reference_observed_second)):
        torch.testing.assert_close(actual, expected, **tolerance)


@pytest.mark.parametrize("bad", ["cpu", "half_input", "rank1", "rank3", "width", "strides", "params_dtype", "params_rank", "params_size"])
def test_low_level_fwd_validation(native, bad):
    module, x, params = native_identity(native)
    if bad == "cpu":
        x = x.cpu()
    elif bad == "half_input":
        x = x.half()
    elif bad == "rank1":
        x = x.flatten()
    elif bad == "rank3":
        x = x.unsqueeze(0)
    elif bad == "width":
        x = torch.zeros(x.shape[0], 3, device="mps")
    elif bad == "strides":
        x = torch.zeros(2, x.shape[0], device="mps").t()
    elif bad == "params_dtype":
        params = params.half()
    elif bad == "params_rank":
        params = params.reshape(0, 1)
    elif bad == "params_size":
        params = torch.zeros(1, device="mps")
    with pytest.raises((RuntimeError, ValueError)):
        module.fwd(x, params)


@pytest.mark.parametrize("bad", ["cpu", "dtype", "rank", "width", "batch", "output_dtype"])
def test_low_level_backward_validation(native, bad):
    module, x, params = native_identity(native)
    ctx, y = module.fwd(x, params)
    dy = torch.ones_like(y)
    if bad == "cpu":
        dy = dy.cpu()
    elif bad == "dtype":
        dy = dy.half()
    elif bad == "rank":
        dy = dy.flatten()
    elif bad == "width":
        dy = torch.ones(y.shape[0], y.shape[1] + 1, device="mps")
    elif bad == "batch":
        dy = torch.ones(y.shape[0] + 1, y.shape[1], device="mps")
    elif bad == "output_dtype":
        y = y.half()
    with pytest.raises((RuntimeError, ValueError)):
        module.bwd(ctx, x, params, y, dy)


@pytest.mark.parametrize("bad", ["cpu", "dtype", "rank", "width", "batch"])
def test_low_level_double_backward_validation(native, bad):
    module, x, params = native_identity(native)
    ctx, y = module.fwd(x, params)
    dy = torch.ones_like(y, requires_grad=True)
    v = torch.ones_like(x)
    if bad == "cpu":
        v = v.cpu()
    elif bad == "dtype":
        v = v.half()
    elif bad == "rank":
        v = v.flatten()
    elif bad == "width":
        v = torch.ones(x.shape[0], x.shape[1] + 1, device="mps")
    elif bad == "batch":
        v = torch.ones(x.shape[0] + 1, x.shape[1], device="mps")
    with pytest.raises((RuntimeError, ValueError)):
        module.bwd_bwd_input(ctx, x, params, v, dy)


def test_inference_context_cannot_be_differentiated(native):
    module, x, params = native_identity(native, requires_grad=False)
    ctx, y = module.fwd(x, params)
    with pytest.raises((RuntimeError, ValueError), match="context|inference"):
        module.bwd(ctx, x, params, y, torch.ones_like(y))


def test_storage_offsets_for_inputs_parameters_and_adjoints(native):
    module = native.create_encoding(2, GRID, native.Precision.Fp32)
    count = native.batch_size_granularity()
    x_storage = torch.rand(count + 3, 2, device="mps")
    x = x_storage[3:].requires_grad_()
    initial = module.initial_params(12)
    p_storage = torch.cat((torch.zeros(7, device="mps"), initial))
    params = p_storage[7:].requires_grad_()
    assert x.storage_offset() != 0 and x.is_contiguous()
    assert params.storage_offset() != 0 and params.is_contiguous()
    ctx, output = module.fwd(x, params)
    _, expected = module.fwd(x.detach().clone(), initial)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    dy_storage = torch.ones(count + 5, output.shape[1], device="mps")
    dy = dy_storage[5:]
    assert dy.storage_offset() != 0 and dy.is_contiguous()
    dx, dp = module.bwd(ctx, x, params, output, dy)
    x_copy, p_copy = x.detach().clone().requires_grad_(), initial.clone().requires_grad_()
    other_ctx, other_y = module.fwd(x_copy, p_copy)
    other_dx, other_dp = module.bwd(other_ctx, x_copy, p_copy, other_y, dy.clone())
    torch.testing.assert_close(dx, other_dx, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(dp, other_dp, rtol=1e-4, atol=1e-5)


def test_pytorch_native_ordering_and_temporary_lifetimes(tcnn):
    encoding = tcnn.Encoding(2, {"otype": "Identity", "scale": 1.125, "offset": -0.125}, dtype=torch.float32)
    value = torch.linspace(0.1, 0.9, 74, device="mps").reshape(37, 2)
    expected = value.cpu()
    # Deliberately no synchronize() or item() in this loop. PyTorch produces
    # temporaries, native code consumes them, and PyTorch consumes its output.
    for _ in range(12):
        value = encoding(value * 0.75 + 0.05) * 0.8 + 0.1
        expected = ((expected * 0.75 + 0.05) * 1.125 - 0.125) * 0.8 + 0.1
        gc.collect()
    tcnn.free_temporary_memory()
    torch.testing.assert_close(value.cpu(), expected, rtol=2e-5, atol=2e-6)


def test_context_survives_interleaved_forward_and_memory_cleanup(tcnn):
    encoding = tcnn.Encoding(2, {"otype": "Frequency", "n_frequencies": 2}, dtype=torch.float32)
    x1 = torch.rand(19, 2, device="mps", requires_grad=True)
    x2 = torch.rand(23, 2, device="mps", requires_grad=True)
    y1, y2 = encoding(x1), encoding(x2)
    grad1 = torch.autograd.grad(y1.sum(), x1, retain_graph=True)[0]
    tcnn.free_temporary_memory()
    grad2 = torch.autograd.grad(y2.sum(), x2)[0]
    repeated = torch.autograd.grad(y1.sum(), x1)[0]
    torch.testing.assert_close(repeated, grad1, rtol=0, atol=0)
    assert torch.isfinite(grad2).all().item()


def test_frozen_parameters_and_input_only_gradients(tcnn):
    encoding = tcnn.Encoding(2, GRID, dtype=torch.float32)
    encoding.params.requires_grad_(False)
    x = torch.rand(11, 2, device="mps", requires_grad=True)
    encoding(x).sum().backward()
    assert encoding.params.grad is None
    assert torch.isfinite(x.grad).all().item()


def test_pickle_and_state_dict_roundtrips(tcnn):
    model = tcnn.Encoding(2, GRID, seed=19, dtype=torch.float32)
    model.backward_backward_input_no_input_grad = True
    x = torch.rand(17, 2, device="mps")
    with torch.no_grad():
        expected = model(x)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.backward_backward_input_no_input_grad
    assert restored.seed == 19
    with torch.no_grad():
        torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    other = tcnn.Encoding(2, GRID, seed=20, dtype=torch.float32)
    other.load_state_dict(torch.load(stream, map_location="mps", weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(other(x), expected, rtol=0, atol=0)


def test_cache_configuration_and_jit_flag(tcnn, tmp_path):
    api = importlib.import_module("tinymetalnn.modules")
    api.rtc_set_cache_dir(tmp_path)
    try:
        with pytest.raises(OSError, match="Missing"):
            api.rtc_set_cache_dir(tmp_path / "does-not-exist")
        model = tcnn.Encoding(2, {"otype": "Frequency", "n_frequencies": 2}, dtype=torch.float32)
        assert isinstance(tcnn.supports_jit_fusion(), bool)
        model.jit_fusion = False
        assert not model.jit_fusion
        if tcnn.supports_jit_fusion():
            model.jit_fusion = True
            assert model.jit_fusion
            assert pickle.loads(pickle.dumps(model)).jit_fusion
    finally:
        api.rtc_set_cache_dir("")


def test_log_callback_reentry_and_exception_delivery(tcnn, native, tmp_path):
    if not tcnn.supports_jit_fusion():
        pytest.skip("The configured native build does not support JIT")
    # A cache path naming a file forces an ordinary source-cache diagnostic;
    # source generation and native execution must still proceed without cache.
    invalid_directory = tmp_path / "cache-is-a-file"
    invalid_directory.write_text("not a directory", encoding="utf-8")
    model = tcnn.Encoding(2, {"otype": "Frequency", "n_frequencies": 3}, dtype=torch.float32)
    model.jit_fusion = True
    caller = threading.get_ident()
    messages = []

    def callback(severity, message):
        assert threading.get_ident() == caller
        assert model.jit_fusion  # Must run after the module mutex is released.
        torch.mps.synchronize()  # Must not run on the MPS serial dispatch queue.
        messages.append((severity, message))

    try:
        native.rtc_set_cache_dir(str(invalid_directory))
        native.set_log_callback(callback)
        result = model(torch.rand(11, 2, device="mps"))
        assert torch.isfinite(result).all().item()
        assert any("cache unavailable" in message for _, message in messages)
        other = tcnn.Encoding(2, {"otype": "Frequency", "n_frequencies": 5}, dtype=torch.float32)
        other.jit_fusion = True

        class CallbackFailure(RuntimeError):
            pass

        def raising_callback(severity, message):
            raise CallbackFailure("diagnostic callback failure")

        native.set_log_callback(raising_callback)
        with pytest.raises(CallbackFailure, match="diagnostic callback failure"):
            other(torch.rand(11, 2, device="mps"))
        torch.mps.synchronize()
    finally:
        # The upstream interface has no callback getter. A no-op callable is a
        # valid reset and avoids assigning an empty std::function.
        native.set_log_callback(lambda severity, message: None)
        native.rtc_set_cache_dir("")


@pytest.mark.parametrize("combined", [False, True])
def test_network_state_and_optimizer(networks, combined):
    if combined:
        model = networks.NetworkWithInputEncoding(2, 3, {"otype": "Identity"}, NETWORK)
    else:
        model = networks.Network(2, 3, NETWORK)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.rand(37, 2, device="mps", requires_grad=True)
    before = model.params.detach().clone()
    y = model(x)
    assert y.shape == (37, 3)
    y.float().square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all().item()
    assert model.params.grad is not None and torch.isfinite(model.params.grad).all().item()
    optimizer.step()
    assert not torch.equal(model.params, before)
    restored = pickle.loads(pickle.dumps(model))
    with torch.no_grad():
        torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)

