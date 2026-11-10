# tiny-metal-nn for PyTorch MPS

The Python package uses setuptools and the shared native CMake build,
PyTorch-compatible pybind11 headers, the existing non-template C++ `Module` facade, and two custom autograd
functions for first derivatives and input double backward. Native execution
uses the repository's Metal kernels and `metalrt` runtime.

The wheel and native tests have been built and run on an Apple M3 Max with
PyTorch 2.15.0 and Python 3.13. The MLX extension can be loaded alongside this
package without sharing internal runtime state.

## Install

Use a Mac and a Python/PyTorch installation for which
`torch.backends.mps.is_available()` is true. The extension requires PyTorch
2.7 or newer, Apple's command-line developer tools, and the macOS version
required by that PyTorch installation. The underlying runtime's minimum is
macOS 13. Apple Silicon is the ordinary supported PyTorch wheel environment;
the binding uses actual device capabilities, not hard-coded chip names.

From the repository root:

```sh
xcode-select --install  # Only if Apple's command-line tools are not installed.
python3 -m pip install "torch>=2.7" "cmake>=3.27" setuptools wheel ninja pytest
python3 -m pip install --no-build-isolation ./bindings/torch
```

Build against the PyTorch installation that will load the extension. Rebuild
after changing PyTorch versions. `--no-build-isolation` prevents pip from
choosing an unrelated PyTorch version in a temporary build environment.
The package builds the `tinymetalnn_torch` CMake target, which links the core
static library. MLX is not discovered or required. See the root
[build guide](../../README.md) for standalone and combined CMake builds.

## Use

```python
import torch
import tinymetalnn as tcnn

model = tcnn.NetworkWithInputEncoding(
    n_input_dims=2,
    n_output_dims=3,
    encoding_config={
        "otype": "HashGrid",
        "n_levels": 4,
        "n_features_per_level": 4,
        "log2_hashmap_size": 12,
        "base_resolution": 16,
        "per_level_scale": 2.0,
    },
    network_config={
        "otype": "FullyFusedMLP",
        "n_neurons": 32,
        "n_hidden_layers": 2,
        "activation": "ReLU",
        "output_activation": "None",
    },
    seed=1337,
)

x = torch.rand(513, 2, device="mps", requires_grad=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer.zero_grad(set_to_none=True)
loss = model(x).float().square().mean()
loss.backward()
optimizer.step()

assert model.params.dtype == torch.float32
assert x.grad.shape == x.shape
```

The public constructors retain the reference signatures:

```python
NetworkWithInputEncoding(n_input_dims, n_output_dims, encoding_config, network_config, seed=1337)
Network(n_input_dims, n_output_dims, network_config, seed=1337)
Encoding(n_input_dims, encoding_config, seed=1337, dtype=None)
```

`Encoding` accepts `dtype=None`, `torch.float16`, or `torch.float32`. Network
precision follows the native build. Parameters are initialized as one flat
FP32 `torch.nn.Parameter`; casts into the execution precision remain in the
autograd graph. Inputs are converted to contiguous FP32, batch rows are padded
to the native granularity, and returned outputs are sliced to the requested
batch and logical output width. CPU inputs warn and transfer to MPS; FP32
conversion precedes that transfer so CPU FP64 inputs remain usable.

Configuration dictionaries pass directly to the existing C++ factories.
The network types are `GemmMLP` (default, arbitrary hidden widths) and
`FullyFusedMLP` (widths 16/32/64/128, Apple GPU family 9 or newer). The factory
uses `GemmMLP` on older devices; unsupported fused widths on eligible devices
raise an error. Both FP16 and FP32 builds support fusion.
`model.jit_fusion = tcnn.supports_jit_fusion()` enables the existing generated
Metal path. No CUDA architecture numbers or CUDA runtime dependencies are used.

## Examples and native acceptance

Run the examples and existing native tests from the repository root:

```sh
python3 bindings/torch/examples/train_function.py
python3 bindings/torch/examples/input_derivatives.py
python3 bindings/torch/examples/encoding_catalog.py

python3 -m pytest bindings/torch/tests --require-mps -q
```

The three examples exercise training with combined and separate modules,
parameter layout, checkpoints and pickle, first and mixed second derivatives,
all encoding families, and FP16/FP32 encoding storage. The catalog runs 27
configurations per precision by default. The expanded finite-axis sweep runs
193 per precision:

```sh
python3 bindings/torch/examples/encoding_catalog.py --full-configurations
python3 -m pytest bindings/torch/tests --require-mps --full-configurations -q
```

Use `--jit` on the examples to require the generated Metal execution path;
they reject silent JIT fallback. See [examples/README.md](examples/README.md)
and [tests/README.md](tests/README.md) for the test matrix and numerical oracles.
`--require-mps` fails when native hardware is unavailable, preventing an all-skip
run from being mistaken for GPU acceptance.

Run native tests with `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` as an additional
correctness gate. Performance measurements should use a separate run without
shader instrumentation.

## Build controls

| Control | Meaning |
| --- | --- |
| `TCNN_HALF_PRECISION=0` | Build networks with FP32 execution; default is FP16. Both explicit encoding precisions remain available. |
| `TCNN_NO_NETWORKS=1` | Encoding-only facade and extension; network constructors report the disabled capability. Equivalent to the upstream-style `setup.py --no-networks` switch. |
| `TCNN_BUILD_WITH_RTC=0` | Disable automatic JIT capability; ordinary Metal kernels still execute. `setup.py --no-jit` is also accepted. |
| `TCNN_BUILD_USE_FAST_MATH=1` | Enable the existing relaxed-math option for generated MSL. Default is the native project's strict setting. |
| `TCNN_BUILD_NO_FWD_BWD=1` | Existing JIT-only model build; requires RTC and callers to enable JIT. Use the `--jit` examples as described in the test guide. |
| `TCNN_METAL_CACHE_DIR` | Persistent native RTC cache; defaults to `~/Library/Caches/tinymetalnn`. Empty disables persistent caching. |

For example, install a fresh FP32 build with
`TCNN_HALF_PRECISION=0 python3 -m pip install --no-build-isolation .` from this
directory. Use a clean build directory when switching build flags.
The encoding-only mode retains shared native translation units containing
matrix kernels; it does not promise to remove every network-related source byte.

`python3 setup.py sdist` creates a self-contained source distribution. Only the
distribution's staging tree receives a `_native/` source snapshot. Normal
checkout builds compile the canonical runtime and NN sources directly. Wheels
include RTC Metal assets.

## Derivative and execution boundaries

The second-order interface differentiates **input gradients** with respect to
inputs, parameters, and upstream output adjoints. Second derivatives originating
from parameter-gradient outputs are outside the supplied upstream binding
interface and raise explicitly. `backward_backward_input_no_input_grad=True`
suppresses only the second input derivative. Arbitrary higher-order autograd,
`torch.compile`, `torch.func` transforms, and graph export are not claimed.

Shared MPS storage is imported with its real byte offset and allocator lease.
Private storage uses GPU-to-GPU staging through the same command buffer.
Forward, backward, and input double backward enqueue native work on the
captured current PyTorch MPS stream and return without a deliberate host GPU
wait. Saved contexts retain their resources and establish GPU event dependencies
when reused on another stream. Parameter initialization retains the existing
C++ facade's synchronous setup behavior and copies initialized values to the
PyTorch allocation on the GPU.

Low-level Python logging callbacks run on the calling thread after native
submission locks are released, so they can query a module or use PyTorch.
Callback exceptions propagate before the binding call returns; already
submitted GPU work continues.

The PyTorch adapter lives entirely here. Framework-neutral Metal buffer import
and borrowed submission are in `metalrt/interop.h`; the core library has no
PyTorch dependency. The independent `bindings/mlx` uses those APIs and the same
`tcnn::cpp::Module` facade without importing this Python package. Native symbols
are hidden and pybind registrations are module-local to keep framework adapters
independent.

