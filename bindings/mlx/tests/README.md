# MLX binding tests

The suite distinguishes **native Metal tests** from **portable wrapper
plumbing**. Passing the latter does not establish native compilation, Metal
queue safety, or native derivative correctness.

## Native execution

After installing the native extension on a Mac, run from the repository root:

```sh
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  python -m pytest bindings/mlx/tests \
  --require-metal -m metal -q -ra

python -m pytest bindings/mlx/tests \
  --require-metal -m metal --full-configurations -q -ra
```

`--require-metal` fails before test execution if macOS or MLX Metal is absent.
Without that flag, native tests explicitly skip on a host without Metal. If
Metal exists but the extension cannot import, that is a failure. Encoding-only
builds explicitly skip network cases. Optional Torch coexistence skips only
when that separate binding or its MPS device is unavailable. JIT-specific
tests report a skip if the native build/device has no JIT capability.

| File | What it verifies on native hardware |
| --- | --- |
| `test_mlx_inference.py` | Fixed native-precision snapshots, explicit refresh after indexed updates/checkpoint reloads/dtype changes, cross-stream consumers, grouped HashGrid parity across precisions/interpolation/batches, and unsupported-configuration fallbacks. |
| `test_mlx_contract.py` | Constructor/metadata contracts, FP32 parameter tree, configurable loss-scale snapshots, batch/output padding, empty tensors, CPU-produced/float64 inputs and explicit parameters, strides and offsets, low-level validation, MLX update/freeze/apply, weight and pickle roundtrips, lazy ownership, asynchronous mixed MLX/native stream work, deferred callback delivery/reentry/errors, compiled replay, optimizer training, JIT, and standalone/coinstalled binding behavior. |
| `test_mlx_derivatives.py` | Independent Frequency/Composite oracles; input VJP/JVP and forward/reverse Hessian-vector products; exact grid parameter JVP; mixed derivatives in both orders; adjoint derivatives; suppression; explicit parameter-Hessian/third-derivative errors; mapped inputs/parameters, arbitrary axes/nesting/empty maps; compiled transforms. |
| `test_mlx_configurations.py` | The same 27/193 encoding cases per precision as the example; network implementation/width/depth families, activation families, and invalid factory configurations. |

The tests do not silently replace unsupported operations with finite
differences. Finite differences appear only as an independent oracle for
selected mixed derivatives, with FP32 execution and explicit tolerances.
The parameter-JVP oracle uses grid linearity. Frequency uses dimension-major,
frequency-major, sine/cosine ordering from the supplied upstream kernel.

## Portable plumbing

With an actual MLX CPU installation, this command exercises the Python wrapper
without requiring the native extension:

```sh
python -m pytest bindings/mlx/tests -m plumbing -q -ra
```

The scoped fixture injects a test facade for
`f(x,p) = p[0] * x**3 + p[1] * sin(x)`, implemented with real MLX operations.
Separate NumPy formulas check gradients, second derivatives and both JVP
arguments. Further checks cover lazy casts/padding/trimming, the intermediate
factory hook, parameter trees, MLX optimizers, ordinary compilation/vectorizing,
empty tensors, float64 CPU leaf gradients, serialization and error paths.
The fixture restores imported modules and routing after every test. It is
never installed in production and never counted as native GPU evidence.

## Mac build matrix

Rebuild and run the native suite/examples for the relevant variants:

| Build | Verification |
| --- | --- |
| Default FP16 networks | Full native suite, both encoding precisions, all examples; JIT variants when supported. |
| `TCNN_HALF_PRECISION=0` | Repeat the suite and examples with FP32 networks. |
| `TCNN_NO_NETWORKS=1` | Encoding suite/catalog/derivative examples; network factories must report unavailability. |
| `TCNN_BUILD_WITH_RTC=0` | AOT suite/examples; JIT capability must be false. |
| `TCNN_BUILD_USE_FAST_MATH=1` | Repeat numerical checks and inspect any tolerance differences. |
| `TCNN_BUILD_NO_FWD_BWD=1` | Build with RTC and run the JIT-enabled examples; the AOT comparison suite does not apply. |

Ordinary shape-specialized `mx.compile` is covered. `shapeless=True` is outside
the supported compilation mode: some operations in the pinned MLX dependency
lack shape rules, including paths introduced by vectorization. Focused guard
tests exercise graphs that can complete an initial trace and require a changed
shape to fail explicitly; this must not silently reuse traced padding or basis
dimensions, and does not promise that every same-shape shapeless graph can run.
Arbitrary parameter Hessians and third
derivatives remain outside the supplied facade's derivative contract and are
tested as explicit, narrow failures.
