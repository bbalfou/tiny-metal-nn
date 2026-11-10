# Adapted from the supplied Python binding interface for native MLX Metal.

import gc
import importlib
import os
import tempfile

import mlx.core as mx
import mlx.nn as nn


if mx.__version__ != "0.32.4":
    # Check before loading the shared object: its native array/Primitive ABI is
    # tied to this MLX release even when pip dependency checks were bypassed.
    raise ImportError(
        f"tinymetalnn_mlx targets MLX 0.32.4, but found {mx.__version__}. "
        "Install the pinned MLX version before loading the native extension."
    )


try:
    _C = importlib.import_module("tinymetalnn_mlx._C")
except ImportError as error:
    raise ImportError(
        "Could not load the tinymetalnn_mlx Metal extension. Build bindings/mlx "
        "against MLX 0.32.4 using `python -m pip install --no-build-isolation .`; "
        "the native extension must match the installed MLX C++ ABI."
    ) from error


def rtc_set_cache_dir(dir):
    """Select a writable native RTC cache directory; an empty path disables it."""
    if not dir:
        _C.rtc_set_cache_dir("")
        return
    if not os.path.isdir(dir):
        raise OSError(f"Missing RTC cache directory {dir}")
    try:
        with tempfile.TemporaryFile(dir=dir):
            pass
    except OSError as error:
        raise OSError(f"Invalid RTC cache directory {dir}") from error
    _C.rtc_set_cache_dir(str(dir))


_C.rtc_set_include_dir(os.path.join(os.path.dirname(__file__), "rtc", "include"))
_rtc_cache_dir = os.environ.get(
    "TCNN_MLX_CACHE_DIR",
    os.environ.get(
        "TCNN_METAL_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), "Library", "Caches", "tinymetalnn_mlx"),
    ),
)
try:
    if _rtc_cache_dir:
        os.makedirs(_rtc_cache_dir, exist_ok=True)
    rtc_set_cache_dir(_rtc_cache_dir)
except OSError:
    _C.rtc_set_cache_dir("")


def _mlx_precision(precision):
    if precision == _C.Precision.Fp16:
        return mx.float16
    if precision == _C.Precision.Fp32:
        return mx.float32
    raise ValueError(f"Unknown native precision {precision}")


def _gpu_stream(stream):
    """Resolve the current GPU stream without evaluating any graph or array."""
    if not mx.metal.is_available():
        raise EnvironmentError(
            "tinymetalnn_mlx execution requires an available MLX Metal device. "
            "This binding does not execute the reference CPU backend."
        )
    if stream is None:
        return mx.default_stream(mx.gpu)
    if isinstance(stream, (mx.Stream, mx.ThreadLocalStream)):
        if stream.device != mx.gpu:
            raise ValueError("tinymetalnn_mlx requires a Metal GPU stream")
        return stream
    if isinstance(stream, (mx.Device, mx.DeviceType)):
        if stream != mx.gpu:
            raise ValueError("tinymetalnn_mlx requires a Metal GPU device")
        return mx.default_stream(stream)
    raise TypeError("stream must be an MLX GPU Stream, ThreadLocalStream, Device, or None")


def supports_jit_fusion():
    """Whether the native runtime and current Metal device support model fusion."""
    return _C.supports_jit_fusion()


def free_temporary_memory():
    """Release unused native workspaces, retaining storage owned by live graphs."""
    gc.collect()
    _C.free_temporary_memory()


class Module(nn.Module):
    """Common native MLX module contract and a flat FP32 parameter tree.

    Subclasses implement ``_native_tcnn_module()`` and establish their input and
    logical output dimensions. ``params`` is an ordinary MLX trainable array;
    optimizers, freezing, and weight serialization use ``mlx.nn.Module`` itself.
    Forward calls build lazy, differentiable native primitives. They do not
    evaluate arrays, synchronize the device, or implement gradients in Python.
    """

    def __init__(self, seed=1337):
        super().__init__()
        self._native_module = self._native_tcnn_module()
        self.seed = seed
        self.dtype = _mlx_precision(self.native_tcnn_module.param_precision())
        stream = _gpu_stream(None)
        self.params = self.native_tcnn_module.initial_params(seed, stream=stream).astype(
            mx.float32, stream=stream
        )

    @property
    def native_tcnn_module(self):
        """The native facade; it is excluded from MLX's parameter/state tree."""
        return self._native_module

    @property
    def n_params(self):
        return self.native_tcnn_module.n_params()

    @property
    def output_dtype(self):
        return _mlx_precision(self.native_tcnn_module.output_precision())

    @property
    def loss_scale(self):
        """Loss scale applied by the native primitive's differentiation rules."""
        return self.native_tcnn_module.loss_scale

    @loss_scale.setter
    def loss_scale(self, value):
        self.native_tcnn_module.loss_scale = value

    @loss_scale.deleter
    def loss_scale(self):
        raise AttributeError("`loss_scale` can not be deleted")

    def __call__(self, x, *, stream=None):
        return self.forward(x, stream=stream)

    def forward(self, x, *, stream=None):
        """Build a lazy native forward operation using this module's parameters."""
        return self.forward_with_params(x, self.params, stream=stream)

    def prepare_inference_params(self, *, stream=None):
        """Evaluate a fixed-weight snapshot once in native parameter precision.

        Reuse the returned array with ``forward_with_params``. This explicit
        snapshot is detached from differentiation and is not stored on the
        module. Call this method again after updates, ``load_weights``, or
        dtype changes to use the new weights. Existing snapshots intentionally
        retain their original values, including after indexed MLX updates.

        Preparation evaluates the conversion once; ordinary forward/training
        still uses the current parameter tree and its usual lazy cast. Cross-
        stream consumers retain MLX's normal array dependency ordering.
        """
        stream = _gpu_stream(stream)
        # A distinct MLX descriptor protects the snapshot from indexed updates
        # even when the master already has native dtype and astype is a no-op.
        params = mx.array(self.params)
        parameter_stream = mx.cpu if params.dtype == mx.float64 else stream
        params = params.astype(
            _mlx_precision(self.native_tcnn_module.param_precision()),
            stream=parameter_stream,
        )
        params = mx.stop_gradient(mx.contiguous(params, stream=stream), stream=stream)
        mx.eval(params)
        return params

    def forward_with_params(self, input, params, *, stream=None):
        """Apply explicit flat parameters without changing the module state.

        This functional entry point is useful with ``mx.grad``, ``mx.jvp``,
        ``mx.vmap``, and ``mx.compile``. The native primitive supplies each
        transformation. ``nn.Module.apply`` retains its usual MLX meaning of
        mapping over a module's parameter tree.
        """
        if not isinstance(input, mx.array):
            raise TypeError("input must be an mlx.core.array")
        if input.ndim != 2 or input.shape[1] != self.n_input_dims:
            raise ValueError(
                f"Expected input of shape [batch, {self.n_input_dims}], "
                f"but got {tuple(input.shape)}"
            )
        if not isinstance(params, mx.array):
            raise TypeError("params must be an mlx.core.array")
        if params.ndim != 1 or params.size != self.n_params:
            raise ValueError(
                f"Expected a flat parameter array of shape [{self.n_params}], "
                f"but got {tuple(params.shape)}"
            )
        stream = _gpu_stream(stream)
        batch_size = input.shape[0]
        granularity = int(_C.batch_size_granularity())
        padded_batch_size = (batch_size + granularity - 1) // granularity * granularity

        # Metal has no float64 storage/arithmetic. A float64 CPU leaf must cast
        # on the CPU so MLX's cast VJP can also return its float64 gradient there.
        # This is a lazy dtype conversion; native model execution still uses GPU.
        if input.dtype == mx.float64:
            input = input.astype(mx.float32, stream=mx.cpu)

        # Every operation selects its stream explicitly, including output
        # slicing, without changing the caller's default device/stream state.
        input = input.astype(mx.float32, stream=stream)
        # The functional API may also receive float64 CPU parameter leaves.
        # Keep their cast VJP on the CPU for the same reason as input casts.
        parameter_stream = mx.cpu if params.dtype == mx.float64 else stream
        params = params.astype(
            _mlx_precision(self.native_tcnn_module.param_precision()),
            stream=parameter_stream,
        )
        # Preserve the logical shapes before padding: otherwise distinct
        # batch lengths could share one native padded shape in a shapeless
        # compiled replay and silently reuse a trace-time output slice.
        input, params = _C.guard_shapes(input, params, stream=stream)
        origin = mx.array([0, 0], dtype=mx.int32)
        if padded_batch_size != batch_size:
            padded = mx.zeros(
                (padded_batch_size, self.n_input_dims),
                dtype=mx.float32,
                stream=stream,
            )
            # DynamicSliceUpdate and its gradient support shape inference.
            # MLX 0.32.4's Pad and the static Slice used by Concatenate's VJP
            # do not, so use the native dynamic update for zero row padding.
            input = mx.slice_update(padded, input, origin, axes=(0, 1), stream=stream)
        input = mx.contiguous(input, stream=stream)
        params = mx.contiguous(params, stream=stream)
        output = self.native_tcnn_module.forward(input, params, stream=stream)
        # The dynamic slice primitive supports output_shapes and transforms;
        # MLX 0.32.4's Python indexing (static Slice) does not support shapeless
        # tracing. The logical guard above protects these fixed prefix lengths.
        return mx.slice(
            output,
            origin,
            axes=(0, 1),
            slice_size=(batch_size, self.n_output_dims),
            stream=stream,
        )

    def __getstate__(self):
        # dict-subclass pickling already carries the MLX parameter tree. Native
        # handles are recreated from the preserved constructor configuration.
        state = self.__dict__.copy()
        state["_native_tcnn_state"] = {
            "jit_fusion": self.jit_fusion,
            "backward_backward_input_no_input_grad": self.backward_backward_input_no_input_grad,
            "loss_scale": self.loss_scale,
        }
        del state["_native_module"]
        # MLX Dtype and native enum instances need not implement Python pickle.
        # dtype is derived from the facade; Encoding's precision selects it.
        state.pop("dtype", None)
        if "precision" in state:
            precision = state.pop("precision")
            if precision == _C.Precision.Fp16:
                state["_native_precision_name"] = "Fp16"
            elif precision == _C.Precision.Fp32:
                state["_native_precision_name"] = "Fp32"
            else:
                raise ValueError(f"Unknown native precision {precision}")
        return state

    def __setstate__(self, state):
        state = state.copy()
        native_state = state.pop("_native_tcnn_state", {})
        precision_name = state.pop("_native_precision_name", None)
        self.__dict__.update(state)
        if precision_name is not None:
            self.precision = getattr(_C.Precision, precision_name)
        self._native_module = self._native_tcnn_module()
        self.dtype = _mlx_precision(self.native_tcnn_module.param_precision())
        for key, value in native_state.items():
            setattr(self.native_tcnn_module, key, value)

    def extra_repr(self):
        return (
            f"n_input_dims={self.n_input_dims}, n_output_dims={self.n_output_dims}, "
            f"seed={self.seed}, dtype={self.dtype}, "
            f"hyperparams={self.native_tcnn_module.hyperparams()}"
        )

    def _extra_repr(self):
        return self.extra_repr()

    @property
    def jit_fusion(self):
        return self.native_tcnn_module.jit_fusion

    @jit_fusion.setter
    def jit_fusion(self, value):
        self.native_tcnn_module.jit_fusion = value

    @jit_fusion.deleter
    def jit_fusion(self):
        raise AttributeError("`jit_fusion` can not be deleted")

    @property
    def backward_backward_input_no_input_grad(self):
        return self.native_tcnn_module.backward_backward_input_no_input_grad

    @backward_backward_input_no_input_grad.setter
    def backward_backward_input_no_input_grad(self, value):
        self.native_tcnn_module.backward_backward_input_no_input_grad = value

    @backward_backward_input_no_input_grad.deleter
    def backward_backward_input_no_input_grad(self):
        raise AttributeError("`backward_backward_input_no_input_grad` can not be deleted")


class NetworkWithInputEncoding(Module):
    """An input encoding followed by a native Metal network in an MLX graph."""

    def __init__(
        self, n_input_dims, n_output_dims, encoding_config, network_config, seed=1337
    ):
        if not _C.has_networks():
            raise RuntimeError(
                "Cannot create `NetworkWithInputEncoding` because tiny-metal-nn "
                "was not compiled with neural network support."
            )
        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims
        # MLX stores public dictionary assignments in its parameter tree. These
        # are constructor metadata, so keep them as ordinary Python attributes.
        object.__setattr__(self, "encoding_config", encoding_config)
        object.__setattr__(self, "network_config", network_config)
        super().__init__(seed=seed)

    def _native_tcnn_module(self):
        return _C.create_network_with_input_encoding(
            self.n_input_dims,
            self.n_output_dims,
            self.encoding_config,
            self.network_config,
        )


class Network(Module):
    """A native Metal network with flat FP32 MLX trainable parameters."""

    def __init__(self, n_input_dims, n_output_dims, network_config, seed=1337):
        if not _C.has_networks():
            raise RuntimeError(
                "Cannot create `Network` because tiny-metal-nn was not compiled "
                "with neural network support."
            )
        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims
        object.__setattr__(self, "network_config", network_config)
        super().__init__(seed=seed)

    def _native_tcnn_module(self):
        return _C.create_network(self.n_input_dims, self.n_output_dims, self.network_config)


class Encoding(Module):
    """A native encoding with optional ``mx.float16`` or ``mx.float32`` precision.

    ``dtype=None`` selects the native preferred precision. Inputs use float32;
    outputs use the encoding's precision. Trainable encoding parameters remain
    a flat FP32 master array in MLX's parameter tree.
    """

    def __init__(self, n_input_dims, encoding_config, seed=1337, dtype=None):
        self.n_input_dims = n_input_dims
        object.__setattr__(self, "encoding_config", encoding_config)
        if dtype is None:
            self.precision = _C.preferred_precision()
        elif dtype == mx.float32:
            self.precision = _C.Precision.Fp32
        elif dtype == mx.float16:
            self.precision = _C.Precision.Fp16
        else:
            raise ValueError("dtype must be mlx.core.float16, mlx.core.float32, or None")
        super().__init__(seed=seed)
        self.n_output_dims = self.native_tcnn_module.n_output_dims()

    def _native_tcnn_module(self):
        return _C.create_encoding(self.n_input_dims, self.encoding_config, self.precision)
