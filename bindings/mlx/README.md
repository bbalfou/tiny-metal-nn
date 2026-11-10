# tiny-metal-nn for MLX Metal

`tinymetalnn_mlx` binds the existing native Metal library to MLX's lazy arrays,
neural-network modules, optimizers, and function transformations. It is a
separate extension and package from the PyTorch binding. It has no PyTorch
dependency and does not execute tensors through the Reference backend.

The wheel and native tests have been built and run on an Apple M3 Max with
MLX 0.32.4, nanobind 3.1.0, and Python 3.13, including coexistence with the
PyTorch extension in one process.

The network types are `GemmMLP` (default, arbitrary positive hidden widths)
and `FullyFusedMLP` (widths 16/32/64/128 on Apple GPU family 9 or newer).
The factory falls back to `GemmMLP` on older hardware and rejects invalid
fused widths on supported hardware. Both FP16 and FP32 builds support fusion.

## Installation

Use a Mac with an MLX Metal device, macOS 14 or newer, Apple's developer tools,
and Python 3.10 or newer. The extension links MLX's C++ API and is pinned to
**MLX 0.32.4 and nanobind 3.1.0**. Rebuild it when changing the MLX ABI; a Python
package import alone does not establish binary compatibility.

From the repository root:

```sh
python -m pip install "mlx==0.32.4" "nanobind==3.1.0" "cmake>=3.27" \
  "setuptools>=68" wheel ninja pytest numpy
python -m pip install --no-build-isolation ./bindings/mlx
```

Install Apple's command-line tools with `xcode-select --install` if needed.
`setup.py` builds the `tinymetalnn_mlx` CMake target using nanobind.
CMake builds the repository's native runtime and neural library;
the MLX adapter adds lazy primitives and uses the framework's command buffers.
The source distribution carries a private snapshot of the required native
source, and the package includes native RTC sources.

## Programming model

The constructor family mirrors the supplied binding interface:

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import tinymetalnn_mlx as tcnn

model = tcnn.NetworkWithInputEncoding(
    n_input_dims=2,
    n_output_dims=3,
    encoding_config={"otype": "Frequency", "n_frequencies": 4},
    network_config={
        "otype": "GemmMLP", "n_neurons": 32, "n_hidden_layers": 2,
        "activation": "ReLU", "output_activation": "None",
    },
    seed=1337,
)
x = mx.random.uniform(shape=(513, 2), stream=mx.gpu)
y = model(x)                         # Builds a lazy native graph.
mx.eval(y)                           # Runs it through MLX Metal.

optimizer = optim.Adam(learning_rate=1e-3)
def loss(m, values, targets):
    return mx.mean(mx.square(m(values).astype(mx.float32) - targets))

value_and_grad = nn.value_and_grad(model, loss)
value, gradients = value_and_grad(model, x, mx.zeros((513, 3)))
optimizer.update(model, gradients)
mx.eval(value, model.parameters(), optimizer.state)
```

`Module` derives from `mlx.nn.Module`; `Encoding`, `Network`, and
`NetworkWithInputEncoding` retain that intermediate abstraction. Subclasses
can implement `_native_tcnn_module()`. The flat trainable array is `params`,
stored in FP32. Native execution casts it to the chosen parameter precision.
`Encoding(..., dtype=mx.float16)` and `dtype=mx.float32` select either encoding
precision; `None` uses the compiled preferred precision. Network precision is
a build choice.

Inputs have shape `[batch, n_input_dims]`. The wrapper handles input casts,
strides, row padding, and logical output slicing in the MLX graph. The native
primitive receives FP32 inputs and execution-precision parameters. Both
`forward(x, *, stream=None)` and `__call__(x, *, stream=None)` resolve the
current GPU stream. An explicit CPU operation stream fails clearly. Arrays
produced on CPU can be inputs to GPU operations through MLX's unified-memory
model. A float64 input casts to FP32 on the CPU before GPU execution, allowing
the cast's reverse derivative to return to a float64 CPU leaf. For float64
inputs, call `mx.grad` inside `with mx.stream(mx.cpu):`; MLX's own transform
bookkeeping needs CPU float64 support. The model still schedules its native
execution on the GPU.

Use `forward_with_params(input, params, *, stream=None)` for pure functional
parameter transforms. It does not mutate the module. `nn.Module.apply` keeps
its normal MLX meaning of mapping over the parameter tree.

```python
encoding = tcnn.Encoding(2, {"otype": "Frequency", "n_frequencies": 3}, dtype=mx.float32)
input_gradient = mx.grad(lambda z: mx.sum(encoding(z)))(x)
_, (directional_derivative,) = mx.jvp(encoding, [x], [mx.ones_like(x)])
grouped = mx.vmap(encoding)(mx.stack((x, x + 0.01)))
compiled = mx.compile(lambda z: mx.sin(encoding(z)))
mx.eval(input_gradient, directional_derivative, grouped, compiled(x))
```

### Fixed-weight inference

Ordinary calls keep the trainable FP32 parameter tree and cast it on each
uncompiled execution. For repeated inference, prepare native precision once
and pass that array through the existing functional API:

```python
params = model.prepare_inference_params(stream=mx.gpu)  # Evaluate once.
y = model.forward_with_params(x, params, stream=mx.gpu)
mx.eval(y)

# Refresh explicitly whenever the serving weights change.
model.load_weights("weights.npz")
params = model.prepare_inference_params()
```

The returned array is an evaluated, detached fixed-weight snapshot. It is not
cached on the module or included in its parameter tree. Existing snapshots
intentionally keep their old values after `update`, indexed parameter edits,
`load_weights`, and `set_dtype`; call `prepare_inference_params` again to serve
new weights. Ordinary calls continue using current parameters and remain
differentiable. Preparation and consumers accept explicit GPU streams, with
MLX array dependencies ordering cross-stream work. Call preparation outside
`mx.compile`/differentiation; a compiled serving function that needs refreshed
weights should take the prepared array as an explicit argument:

```python
infer = mx.compile(lambda x, p: model.forward_with_params(x, p))
mx.eval(infer(x, params))
```

Eligible 2D, two-feature CoherentPrime HashGrid inference on Apple GPU family 7+
groups levels into one dispatch. Other geometries, hashes and backends retain
compatible kernels. Dispatch is automatic: old `grouped_inference` and
`grouped_parameter_gradients` checkpoint keys are ignored and omitted from
serialized configurations. Eligible 2D/3D HashGrid training and analytic encoders
use the optimized shared native implementations described in the
[root README](../../README.md#hashgrid-training). Whole-model JIT retains its
separate implementation.

### Transformations and limits

| Operation | Implementation and scope |
| --- | --- |
| First reverse derivatives | Native input and parameter backward paths. |
| Input second derivatives | Native input double backward, including mixed parameter and output-adjoint derivatives. |
| Input JVP | Native second-order facade's derivative with respect to the output adjoint. |
| Parameter JVP | Exact native backward basis evaluations and MLX reductions. This can be expensive: cost grows with the **padded output element count**, with no arbitrary size cap. |
| Parameter-gradient derivative with respect to input or output adjoint | Exact native mixed-derivative/basis construction. |
| `mx.vmap` | Mapped inputs, mapped parameters, arbitrary mapped axes, nested mappings, and transformed gradients; independent slices build native operations and are stacked. This preserves semantics without promising a fused batched kernel. |
| `mx.compile` | Normal shape-specialized compilation and replay; custom primitives remain native graph operations. |
| `shapeless=True` | Outside the supported compilation mode. Some graphs cannot complete an initial shapeless trace because the pinned MLX primitives lack shape rules. Where tracing succeeds, native guards explicitly reject changed logical shapes. Use ordinary `mx.compile`, which specializes for new shapes. |
| Parameter Hessians and third or higher derivatives | Explicit errors in the affected derivative paths; the supplied upstream facade does not provide those operations. |

`jit_fusion` enables the core library's existing runtime model fusion. It is
independent of `mx.compile`; the examples verify that requested native JIT did
not silently fall back. `supports_jit_fusion()` reports the actual native
capability. `backward_backward_input_no_input_grad` suppresses the input
curvature destination while retaining other supported second derivatives.
`loss_scale` is configurable; native operations snapshot their flags/scaling
when the lazy graph is built.

MLX's ordinary `train()`, `eval()`, `freeze()`, `unfreeze()`, `update()`,
`apply()`, `save_weights()` and `load_weights()` operate on the parameter tree.
Weight files use MLX's `.npz` or `.safetensors` conventions. Pickling recreates
the native facade from constructor configuration and preserves parameters and
native flags. Graphs own their needed native resources independently of the
Python module's lifetime. `free_temporary_memory()` releases unused pooled
resources; it does not invalidate live computations.

## Build variants

Set these environment variables when installing and rebuild each variant:

| Variable | Default | Meaning |
| --- | --- | --- |
| `TCNN_HALF_PRECISION` | `1` | Native network execution precision; `0` selects FP32. Encoding constructors retain both types. |
| `TCNN_NO_NETWORKS` | `0` | `1` creates an encoding-only build. |
| `TCNN_BUILD_WITH_RTC` | `1` | `0` disables native RTC/model fusion. |
| `TCNN_BUILD_USE_FAST_MATH` | `0` | Native math-mode choice; verify appropriate tolerances when enabling it. |
| `TCNN_BUILD_NO_FWD_BWD` | `0` | `1` omits AOT forward/backward; requires RTC and explicitly enabled JIT models. |

`setup.py --no-networks` and `--no-jit` retain corresponding build switches.
The independent RTC cache directory is `TCNN_MLX_CACHE_DIR`, falling back to
`TCNN_METAL_CACHE_DIR` and then `~/Library/Caches/tinymetalnn_mlx`.
`rtc_set_cache_dir("")` disables source caching.

### Deferred native diagnostics

The low-level `_C.set_log_callback(callback)` interface installs a Python
diagnostic callback. Native execution is lazy and may run on an MLX scheduler
thread, so diagnostics are first recorded as native values. Call
`_C.flush_logs()` after `mx.eval()` or `mx.synchronize()` to deliver them on the
calling Python thread; the next binding entry also flushes pending messages.
Callbacks can reenter MLX or query a module after native locks have unwound.
Callback exceptions propagate when messages are delivered, after the GPU work
may already have completed; they do not cancel previously submitted work.

## Examples and native acceptance

See [examples/README.md](examples/README.md) for the three runnable examples
and [tests/README.md](tests/README.md) for coverage and the build matrix.
From the repository root after installing the package:

```sh
python -m pytest bindings/mlx/tests --require-metal -q
```

See the root [build guide](../../README.md) for independent or combined CMake
builds, formatting, and C++ installation.
