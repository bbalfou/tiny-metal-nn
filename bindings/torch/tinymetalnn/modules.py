# Adapted from the supplied Python bindings for the native Metal/MPS backend.

import gc
import importlib
import os
import tempfile
import warnings

import torch


if not torch.backends.mps.is_available():
    raise EnvironmentError(
        "tinymetalnn requires macOS and a PyTorch installation with an available "
        "MPS device. The PyTorch bindings do not use the reference CPU backend."
    )

try:
    # Metal compiles its shaders for the actual device at runtime. CPU extension
    # modules do not pretend to represent CUDA compute capabilities.
    _C = importlib.import_module("tinymetalnn_bindings._C")
except ImportError as error:
    raise ImportError(
        "Could not load the tinymetalnn MPS extension. Build bindings/torch "
        "against the installed PyTorch with `python -m pip install "
        "--no-build-isolation .`; rebuild it after changing PyTorch versions."
    ) from error


def rtc_set_cache_dir(dir):
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


# The .metal include assets are packaged just as the reference packages its RTC
# headers. Generated native model kernels are otherwise self-contained.
_rtc_dir = os.path.join(os.path.dirname(__file__), "rtc")
_C.rtc_set_include_dir(os.path.join(_rtc_dir, "include"))
_rtc_cache_dir = os.environ.get(
    "TCNN_METAL_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), "Library", "Caches", "tinymetalnn"),
)
try:
    if _rtc_cache_dir:
        os.makedirs(_rtc_cache_dir, exist_ok=True)
    rtc_set_cache_dir(_rtc_cache_dir)
except OSError:
    # Runtime compilation still works when a persistent cache cannot be written.
    _C.rtc_set_cache_dir("")


def _torch_precision(tcnn_precision):
    if tcnn_precision == _C.Precision.Fp16:
        return torch.half
    if tcnn_precision == _C.Precision.Fp32:
        return torch.float
    raise ValueError(f"Unknown precision {tcnn_precision}")


def supports_jit_fusion():
    return _C.supports_jit_fusion()


def free_temporary_memory():
    # Native contexts can own temporary allocations until Python releases them.
    gc.collect()
    _C.free_temporary_memory()


def null_tensor_like(tensor):
    return torch.empty([], dtype=tensor.dtype, device=tensor.device)


def null_tensor_to_none(tensor):
    return None if len(tensor.shape) == 0 else tensor


class _module_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, native_tcnn_module, input, params, loss_scale):
        ctx.set_materialize_grads(False)
        native_ctx, output = native_tcnn_module.fwd(input, params)
        ctx.save_for_backward(input, params, output)
        ctx.native_tcnn_module = native_tcnn_module
        ctx.native_ctx = native_ctx
        ctx.loss_scale = loss_scale
        return output

    @staticmethod
    def backward(ctx, doutput):
        if doutput is None:
            return None, None, None, None

        input, params, output = ctx.saved_tensors
        if doutput.device.type != "mps":
            warnings.warn(
                "doutput must be an MPS tensor, but isn't. "
                "This indicates suboptimal performance.",
                stacklevel=2,
            )
            doutput = doutput.to(input.device)

        input_grad, params_grad = _module_function_backward.apply(
            ctx, doutput, input, params, output
        )
        return (
            None,
            null_tensor_to_none(input_grad),
            null_tensor_to_none(params_grad),
            None,
        )


class _module_function_backward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ctx_fwd, doutput, input, params, output):
        # In particular, an unused parameter-gradient output must remain None
        # during double backward rather than becoming an artificial zero tensor.
        ctx.set_materialize_grads(False)
        ctx.ctx_fwd = ctx_fwd
        ctx.save_for_backward(input, params, doutput)
        with torch.no_grad():
            scaled_grad = (doutput * ctx_fwd.loss_scale).contiguous()
            input_grad, params_grad = ctx_fwd.native_tcnn_module.bwd(
                ctx_fwd.native_ctx, input, params, output, scaled_grad
            )
            input_grad = (
                null_tensor_like(input)
                if input_grad is None
                else input_grad / ctx_fwd.loss_scale
            )
            params_grad = (
                null_tensor_like(params)
                if params_grad is None
                else params_grad / ctx_fwd.loss_scale
            )
        return input_grad, params_grad

    @staticmethod
    def backward(ctx, dinput_grad, dparams_grad):
        # This is the same second-order family provided by the reference:
        # derivatives of dL/dinput with respect to input, parameters, and doutput.
        # Derivatives through dL/dparams are not implemented by that interface.
        if dparams_grad is not None:
            raise RuntimeError(
                "tinymetalnn supports second-order derivatives through input "
                "gradients only. Second-order derivatives through parameter "
                "gradients are not supported by the upstream binding contract."
            )
        if dinput_grad is None:
            return None, None, None, None, None

        input, params, doutput = ctx.saved_tensors
        # Preserve doutput.requires_grad: it selects the native derivative with
        # respect to doutput even though backward usually runs with grad disabled.
        with torch.enable_grad():
            scaled_doutput = (doutput * ctx.ctx_fwd.loss_scale).contiguous()
        with torch.no_grad():
            doutput_grad, params_grad, input_grad = (
                ctx.ctx_fwd.native_tcnn_module.bwd_bwd_input(
                    ctx.ctx_fwd.native_ctx,
                    input,
                    params,
                    dinput_grad.contiguous(),
                    scaled_doutput,
                )
            )
            # The doutput derivative uses dinput_grad alone. The parameter and
            # input derivatives also use the scaled doutput and must be unscaled.
            if params_grad is not None:
                params_grad = params_grad / ctx.ctx_fwd.loss_scale
            if input_grad is not None:
                input_grad = input_grad / ctx.ctx_fwd.loss_scale
        return None, doutput_grad, input_grad, params_grad, None


class Module(torch.nn.Module):
    def __init__(self, seed=1337):
        super(Module, self).__init__()
        # Keep the supplied customization point and attribute spelling so that
        # extension code can continue overriding the same intermediate class.
        self.native_tcnn_module = self._native_tcnn_module()
        self.dtype = _torch_precision(self.native_tcnn_module.param_precision())
        self.seed = seed
        initial_params = self.native_tcnn_module.initial_params(seed)
        self.params = torch.nn.Parameter(initial_params, requires_grad=True)
        self.register_parameter(name="params", param=self.params)
        self.loss_scale = _C.default_loss_scale(
            self.native_tcnn_module.param_precision()
        )

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.n_input_dims:
            raise ValueError(
                f"Expected input of shape [batch, {self.n_input_dims}], "
                f"but got {tuple(x.shape)}"
            )
        if x.device.type != "mps":
            warnings.warn(
                "input must be an MPS tensor, but isn't. "
                "This indicates suboptimal performance.",
                stacklevel=2,
            )
            # The facade consumes float32 inputs. Cast before the transfer so
            # float64 CPU inputs retain that contract without trying to allocate
            # an unsupported float64 MPS tensor. Both copies stay differentiable.
            x = x.to(dtype=torch.float32).to(device="mps")

        batch_size = x.shape[0]
        granularity = int(_C.batch_size_granularity())
        padded_batch_size = (batch_size + granularity - 1) // granularity * granularity
        x_padded = (
            x
            if batch_size == padded_batch_size
            else torch.nn.functional.pad(
                x, [0, 0, 0, padded_batch_size - batch_size]
            )
        )
        input = x_padded.to(torch.float).contiguous()
        params = self.params.to(
            _torch_precision(self.native_tcnn_module.param_precision())
        ).contiguous()
        if torch.is_grad_enabled():
            output = _module_function.apply(
                self.native_tcnn_module, input, params, self.loss_scale
            )
        else:
            # Parameter leaves can still require grad inside no_grad(). Passing
            # detached tensors selects the native inference entry intentionally.
            _, output = self.native_tcnn_module.fwd(input.detach(), params.detach())
        return output[:batch_size, : self.n_output_dims]

    def __getstate__(self):
        """Serialize Python state and native options, without native handles."""
        state = self.__dict__.copy()
        state["_native_tcnn_state"] = {
            "jit_fusion": self.jit_fusion,
            "backward_backward_input_no_input_grad": (
                self.backward_backward_input_no_input_grad
            ),
        }
        del state["native_tcnn_module"]
        return state

    def __setstate__(self, state):
        state = state.copy()
        native_state = state.pop("_native_tcnn_state", {})
        self.__dict__.update(state)
        self.native_tcnn_module = self._native_tcnn_module()
        for key, value in native_state.items():
            setattr(self.native_tcnn_module, key, value)

    def extra_repr(self):
        return (
            f"n_input_dims={self.n_input_dims}, n_output_dims={self.n_output_dims}, "
            f"seed={self.seed}, dtype={self.dtype}, "
            f"hyperparams={self.native_tcnn_module.hyperparams()}"
        )

    @property
    def jit_fusion(self):
        return self.native_tcnn_module.jit_fusion

    @jit_fusion.setter
    def jit_fusion(self, val):
        self.native_tcnn_module.jit_fusion = val

    @jit_fusion.deleter
    def jit_fusion(self):
        raise AttributeError("`jit_fusion` can not be deleted")

    @property
    def backward_backward_input_no_input_grad(self):
        return self.native_tcnn_module.backward_backward_input_no_input_grad

    @backward_backward_input_no_input_grad.setter
    def backward_backward_input_no_input_grad(self, val):
        self.native_tcnn_module.backward_backward_input_no_input_grad = val

    @backward_backward_input_no_input_grad.deleter
    def backward_backward_input_no_input_grad(self):
        raise AttributeError("`backward_backward_input_no_input_grad` can not be deleted")


class NetworkWithInputEncoding(Module):
    """An input encoding followed by a native Metal neural network.

    Maps a ``[batch, n_input_dims]`` tensor to ``[batch, n_output_dims]``.
    Inputs are converted to float32; outputs use the native network precision.
    Configuration dictionaries retain the supplied C++ factory semantics.
    ``params`` is the flat float32 trainable master parameter tensor, initialized
    by the native library using ``seed`` (1337 by default).
    """

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
        self.encoding_config = encoding_config
        self.network_config = network_config
        super(NetworkWithInputEncoding, self).__init__(seed=seed)

    def _native_tcnn_module(self):
        return _C.create_network_with_input_encoding(
            self.n_input_dims,
            self.n_output_dims,
            self.encoding_config,
            self.network_config,
        )


class Network(Module):
    """A native Metal neural network with the supplied factory configuration.

    Maps ``[batch, n_input_dims]`` float32 inputs to ``[batch, n_output_dims]``
    outputs in the native network precision. ``params`` remains a flat float32
    trainable master tensor; the execution precision is available as ``dtype``.
    """

    def __init__(self, n_input_dims, n_output_dims, network_config, seed=1337):
        if not _C.has_networks():
            raise RuntimeError(
                "Cannot create `Network` because tiny-metal-nn was not compiled "
                "with neural network support."
            )
        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims
        self.network_config = network_config
        super(Network, self).__init__(seed=seed)

    def _native_tcnn_module(self):
        return _C.create_network(
            self.n_input_dims, self.n_output_dims, self.network_config
        )


class Encoding(Module):
    """A native input encoding, including trainable encodings.

    Maps ``[batch, n_input_dims]`` float32 inputs to ``[batch, n_output_dims]``
    outputs; ``n_output_dims`` is determined by the encoding configuration.
    ``dtype`` may be ``torch.float32``, ``torch.float16``, or ``None`` to select
    the native library's preferred precision. Both precisions use the same
    factory and configuration space as the C++ facade.
    """

    def __init__(self, n_input_dims, encoding_config, seed=1337, dtype=None):
        self.n_input_dims = n_input_dims
        self.encoding_config = encoding_config
        if dtype is None:
            self.precision = _C.preferred_precision()
        elif dtype == torch.float32:
            self.precision = _C.Precision.Fp32
        elif dtype == torch.float16:
            self.precision = _C.Precision.Fp16
        else:
            raise ValueError(f"Encoding only supports fp32 or fp16 precision, but got {dtype}")
        super(Encoding, self).__init__(seed=seed)
        self.n_output_dims = self.native_tcnn_module.n_output_dims()

    def _native_tcnn_module(self):
        return _C.create_encoding(
            self.n_input_dims, self.encoding_config, self.precision
        )
