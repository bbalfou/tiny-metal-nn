# Native Python / PyTorch examples

These examples require a Mac with an available PyTorch MPS device and the
compiled `tinymetalnn` extension. They use generated tensors, so no dataset,
image download, NumPy, or plotting package is required. They stop clearly when
MPS is unavailable; they never substitute the C++ Reference backend.

From the repository root, install the binding into the Python environment
containing PyTorch, then run the examples:

```bash
python -m pip install --no-build-isolation ./tiny-metal-nn/bindings/torch
cd tiny-metal-nn/bindings/torch

python examples/train_function.py --steps 150
python examples/encoding_catalog.py
python examples/input_derivatives.py
```

## Training, composition, and persistence

`train_function.py` constructs all three public module classes. It compares a
`NetworkWithInputEncoding` with separate `Encoding` and `Network` modules using
matching native parameter arrays, trains both with PyTorch Adam, checks input
gradients, and round-trips a `state_dict` checkpoint and a pickled module. The
default batch size is 513 to exercise batch padding. FP32 master parameters
remain visible to the optimizer even when native execution uses FP16.

```bash
python examples/train_function.py --checkpoint results/model.pt
python examples/train_function.py --network FullyFusedMLP
python examples/train_function.py --jit
```

The default checkpoint is `tinymetalnn_training.pt` in the current directory.
The combined model stores network parameters first, followed by encoding
parameters. Its encoding has 16 features so that comparison with the separate
modules does not depend on different encoding-specific padded feature values.
`FullyFusedMLP` uses the existing native device-family dispatch; the selected
library may legitimately choose its GEMM path on an older supported GPU.

## Encoding configuration catalog

`encoding_catalog.py` exercises parameter-free and trainable encodings in both
FP16 and FP32. It includes Empty, Identity, Frequency, TriangleWave, OneBlob,
SphericalHarmonics, Hash/Dense/Tiled Grid, Permuto, Composite reductions,
NRC/OneBlobFrequency, and hard/soft LoD. Every case uses an uneven batch and
noncontiguous input, then checks native output and first derivatives.

```bash
python examples/encoding_catalog.py --dtype fp32
python examples/encoding_catalog.py --full-configurations
python examples/encoding_catalog.py --dtype fp16 --jit
```

The full sweep expands finite factory axes: enabled grid dimensions, features
per level, grid storage and interpolation; every enabled Permuto dimension and
feature width; all spherical-harmonic degrees; and LoD aliases/bases. Scales,
seeds, table sizes, and arbitrary composition trees remain representative
samples. The pytest suite imports the same catalog rather than duplicating it.

## First and second derivatives

`input_derivatives.py` compares native Frequency output, input gradients,
Hessian-vector products, and output-adjoint derivatives with independent Torch
formulas computed in CPU FP64. It then trains a native HashGrid through a loss
on its **input gradient**, exercising the mixed derivative into trainable
parameters. Finally it checks optional suppression of the second input
derivative while preserving the output-adjoint derivative.

```bash
python examples/input_derivatives.py --dtype fp32
python examples/input_derivatives.py --dtype fp16
python examples/input_derivatives.py --dtype fp32 --jit
```

The CPU calculations here are mathematical comparison oracles: the model being
tested still executes on native Metal. Second derivatives originating from
`dL/dparams` are outside the supplied upstream Python contract and raise an
explicit error. The supported family differentiates `dL/dinput` with respect to
input, parameters, and the upstream output adjoint. See `../tests/README.md` for
the negative-path, native-lifetime, checkpoint, and build-configuration tests.

`--jit` requires the real native capability and checks that a JIT compilation
failure did not silently turn the model back to AOT. A successful run without
`--jit` does not claim JIT validation.

