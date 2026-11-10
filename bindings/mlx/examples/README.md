# Native MLX examples

Install the binding as described in [the binding README](../README.md). Run
these commands from `tiny-metal-nn/bindings/mlx`. Each example requires real
MLX Metal and fails clearly if the native device is absent.

## Training, composition, compilation, and weight files

```sh
python examples/train_function.py --compile --checkpoint mlx-function.npz
python examples/train_function.py --compile --jit --checkpoint mlx-function-jit.npz
```

`train_function.py` builds a combined encoding/network and separate `Encoding`
and `Network` modules. It maps the network-first native parameter layout so
both models begin with identical values, checks their initial outputs, and
trains them with `mlx.nn.value_and_grad` and `mlx.optimizers.Adam`. The encoding
has 16 features so its width agrees with the combined network's input padding.
It also checks input gradients, native MLX weight-file roundtrips, and optional
compiled training steps. The outputs are combined/separate weight files and
a JSON configuration/result record.

`--steps` and `--batch-size` control workload. The defaults are 150 steps and
an odd batch of 513, which exercises row padding. Loss must decrease for both
models. `--jit` separately enables the core library's runtime fusion and checks
the native capability and final fusion flags.

## Input, parameter, and higher derivatives

```sh
python examples/input_derivatives.py --dtype fp32
python examples/input_derivatives.py --dtype fp16 --jit
```

`input_derivatives.py` compares Frequency values, input VJPs, Hessian-vector
products, adjoint derivatives, and input JVPs with an independent NumPy float64
oracle. It then exercises `mx.compile`, `mx.vmap`, the input-curvature
suppression flag, exact parameter JVP for a trainable grid, and training through
an input-gradient penalty. Both precision paths use native Metal execution.

Parameter JVP uses exact native backward basis evaluations; it is deliberately
tested on a narrow output because its cost grows with the padded output shape.
This example does not claim arbitrary parameter Hessians or third derivatives.

## Configuration and precision catalog

```sh
python examples/encoding_catalog.py
python examples/encoding_catalog.py --full-configurations
python examples/encoding_catalog.py --dtype fp32 --jit
```

The default catalog contains **27 cases per precision** and the expanded
catalog contains **193 cases per precision**. Both test forward values and
input/parameter gradients. The case definitions are standalone and shared with
the MLX pytest configuration sweep; importing the case generator does not load
MLX or the native extension.

Coverage includes analytic encodings, enabled Grid dimensions/features/storage
types/interpolations, Permuto dimensions/features and limited input gradients,
Composite concatenation/sum/product, aliases, and hard/soft LoD. The expanded
sweep covers the enabled finite Grid product and Permuto dimension/feature
product. Continuous scales, resolutions, table sizes and seeds are sampled.
No finite catalog is an exhaustive numerical proof of all JSON configurations.

Networks, activation families, zero-width/zero-batch cases, storage views,
stream ordering, lazy graph lifetimes, mapped parameters, higher derivatives,
and explicit failure paths receive additional coverage in [the native tests](../tests/README.md).
