# PyTorch binding validation

Run these commands from `tiny-metal-nn/bindings/torch` after installing the
extension into the active Python environment:

```bash
python -m pytest tests -v --require-mps
python -m pytest tests/test_mps_configurations.py -v --require-mps --full-configurations
```

`--require-mps` turns absent hardware into a failure, so a Mac validation job
cannot pass by skipping every native test. Once MPS is available, failure to
import the compiled extension is always an error. Without `--require-mps`,
native tests skip explicitly on unsupported hosts.

## What each suite checks

| Suite | Evidence |
| --- | --- |
| `test_mps_contract.py` | Public signatures, intermediate `Module` class, flattened FP32 master parameters, execution precision, batch/output padding, zero-sized inputs and outputs, direct native validation, nonzero tensor storage offsets, CPU input adaptation, native/PyTorch ordering, context lifetimes, retained graphs, checkpoints, pickle, cache configuration, optimizer integration, cross-stream context reuse where public MPS stream selection exists, logging callback reentry and exception delivery |
| `test_mps_derivatives.py` | Frequency and Composite forward/first/second derivatives against independent Torch formulas; trainable grid mixed derivatives against finite differences; all supported second-backward destinations; optional input Hessian suppression; explicit failure for second derivatives through parameter gradients; native JIT against AOT |
| `test_mps_configurations.py` | Each encoding family, both encoding precisions, feature/dimension/storage/interpolation axes, Permuto, Composite reductions, LoD, general hidden widths and fused widths, hidden depth, activation family, invalid and disabled factory choices |
| `test_wrapper_plumbing.py` | **CPU-only Python/autograd plumbing evidence.** A closed-form polynomial oracle checks loss scaling, FP32 master/execution casts, first and mixed second derivatives, padding, missing gradients, state restoration, and build capability handling. It does not load or validate the native extension. |

The explicit CPU oracle lives only in the tests. The installed package has no
CPU execution fallback. Its fake device-transfer routing is scoped to one
pytest fixture and restored afterwards. In a restricted CPU environment where
PyTorch cannot initialize `cpuinfo` for FP16 conversions, only the two affected
FP16 oracle cases skip with that exact reason; FP32 arithmetic still runs.

The default catalog selects a small number of cases across all axes. The
expanded catalog takes the full Cartesian product of enabled Grid dimensions
`2/3/4`, feature widths `1/2/4/8`, three storage kinds, and three interpolation
modes, plus every enabled Permuto dimension and feature width. The supplied
JSON grid factory enables CoherentPrime only; direct C++ BaseConvert template
support is a separate contract and is not exposed by upstream Python.

Continuous values such as resolutions, level scales, seeds, level counts, and
arbitrary nested configuration trees are sampled rather than claimed to be
exhaustive. Factory tests establish native reachability, output shape and
finite gradients; the numerical-oracle suites provide stronger evidence for
specific derivatives.

## Build matrix

Use a fresh build or force a rebuild for each variant and install against the
same active PyTorch version. From this directory:

```bash
# Normal FP16 network ABI; encodings test both precisions independently.
TCNN_HALF_PRECISION=1 python -m pip install --no-build-isolation --no-deps --force-reinstall .
python -m pytest tests -v --require-mps

# FP32 network ABI.
TCNN_HALF_PRECISION=0 python -m pip install --no-build-isolation --no-deps --force-reinstall .
python -m pytest tests -v --require-mps

# Encodings only; network tests report the configured capability skip.
TCNN_NO_NETWORKS=1 python -m pip install --no-build-isolation --no-deps --force-reinstall .
python -m pytest tests -v --require-mps

# AOT execution without automatic runtime model fusion.
TCNN_BUILD_WITH_RTC=0 python -m pip install --no-build-isolation --no-deps --force-reinstall .
python -m pytest tests -v --require-mps
```

The inherited `TCNN_BUILD_NO_FWD_BWD=1` configuration intentionally requires
`jit_fusion=True` before execution. The general suite compares AOT with JIT,
so validate this special build with the JIT-enabled examples instead:

```bash
TCNN_BUILD_NO_FWD_BWD=1 TCNN_BUILD_WITH_RTC=1 python -m pip install --no-build-isolation --no-deps --force-reinstall .
python examples/encoding_catalog.py --jit
python examples/input_derivatives.py --jit
python examples/train_function.py --jit
```

Native test results must be recorded with the Mac model/GPU family, macOS,
PyTorch, Python, compiler, selected build flags, and whether shader validation
was enabled. CPU oracle results, source inspection, and a successful Reference
C++ run cannot establish Objective-C++/MSL compilation, Metal correctness, or
asynchronous MPS resource/queue safety.

