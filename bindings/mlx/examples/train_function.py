#!/usr/bin/env python3
"""Train combined and separate native encodings/networks with MLX optimizers.

The two models start with identical flattened parameters. The example checks
their outputs, trains both, and roundtrips native MLX weight files. Use
``--compile`` to compile the stateful training step, and ``--jit`` for the
neural library's independent native model-fusion path.
"""

import argparse
from functools import partial
import json
from pathlib import Path


ENCODING = {
    "otype": "HashGrid", "n_levels": 8, "n_features_per_level": 2,
    "log2_hashmap_size": 8, "base_resolution": 4,
    "per_level_scale": 1.5, "interpolation": "Smoothstep",
}
NETWORK = {
    "otype": "GemmMLP", "n_neurons": 32, "n_hidden_layers": 2,
    "activation": "ReLU", "output_activation": "None",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=513)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--checkpoint", type=Path, default=Path("mlx-function.npz"))
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--jit", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        parser.error("--steps and --batch-size must be positive")

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    if not mx.metal.is_available():
        raise SystemExit("This example requires MLX Metal and the native tinymetalnn_mlx extension.")
    import tinymetalnn_mlx as tcnn
    from tinymetalnn_mlx import _C
    if not _C.has_networks():
        raise SystemExit("This extension was explicitly built without networks; run the encoding examples instead.")
    if args.jit and not tcnn.supports_jit_fusion():
        raise SystemExit("Native JIT fusion is unavailable in this build/device.")
    mx.set_default_device(mx.gpu)
    mx.random.seed(17)

    combined = tcnn.NetworkWithInputEncoding(2, 3, ENCODING, NETWORK, seed=17)
    combined.jit_fusion = args.jit

    class Separate(nn.Module):
        def __init__(self):
            super().__init__()
            # Sixteen logical encoding features match the combined network's
            # required alignment, so the parameter layouts agree exactly.
            self.encoding = tcnn.Encoding(2, ENCODING, seed=17, dtype=combined.dtype)
            self.network = tcnn.Network(self.encoding.n_output_dims, 3, NETWORK, seed=17)
            self.encoding.jit_fusion = self.network.jit_fusion = args.jit

        def __call__(self, x):
            return self.network(self.encoding(x))

    separate = Separate()
    split = separate.network.n_params
    assert combined.n_params == split + separate.encoding.n_params
    # The facade stores network parameters first, then encoding parameters.
    separate.network.params = combined.params[:split]
    separate.encoding.params = combined.params[split:]

    x = mx.random.uniform(low=0.05, high=0.95, shape=(args.batch_size, 2))
    target = mx.stack((
        mx.sin(6.283185307179586 * x[:, 0]) * mx.cos(6.283185307179586 * x[:, 1]),
        x[:, 0] * x[:, 1],
        mx.sin(3.141592653589793 * (x[:, 0] + x[:, 1])),
    ), axis=1)
    tolerance = 4e-3 if combined.dtype == mx.float16 else 4e-5
    assert mx.allclose(combined(x), separate(x), rtol=tolerance, atol=tolerance).item()

    def loss(model, values, targets):
        return mx.mean(mx.square(model(values).astype(mx.float32) - targets))

    trained = []
    for name, model in (("combined", combined), ("separate", separate)):
        optimizer = optim.Adam(learning_rate=args.learning_rate)
        value_and_grad = nn.value_and_grad(model, loss)

        def step(values, targets):
            value, gradients = value_and_grad(model, values, targets)
            optimizer.update(model, gradients)
            return value

        if args.compile:
            state = [model.state, optimizer.state]
            step = partial(mx.compile, inputs=state, outputs=state)(step)
        before = loss(model, x, target).item()
        for iteration in range(args.steps):
            value = step(x, target)
            mx.eval(value, model.parameters(), optimizer.state)
            if iteration == 0 or (iteration + 1) % 50 == 0:
                print(json.dumps({"model": name, "step": iteration + 1, "loss": value.item()}))
        after = loss(model, x, target).item()
        assert after < before, f"{name}: loss did not decrease ({before} -> {after})"
        input_gradient = mx.grad(lambda values: mx.sum(model(values).astype(mx.float32)))(x[:7])
        assert mx.all(mx.isfinite(input_gradient)).item()
        trained.append({"model": name, "initial_loss": before, "final_loss": after})

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    combined.save_weights(str(args.checkpoint))
    separate_path = args.checkpoint.with_name(args.checkpoint.stem + "-separate" + args.checkpoint.suffix)
    separate.save_weights(str(separate_path))
    restored = tcnn.NetworkWithInputEncoding(2, 3, ENCODING, NETWORK, seed=99)
    restored.jit_fusion = args.jit
    restored.load_weights(str(args.checkpoint))
    restored_separate = Separate()
    restored_separate.load_weights(str(separate_path))
    assert mx.allclose(restored(x), combined(x), rtol=0, atol=0).item()
    assert mx.allclose(restored_separate(x), separate(x), rtol=0, atol=0).item()
    if args.jit:
        assert all(m.jit_fusion for m in (combined, separate.encoding, separate.network, restored))
    metadata = {"encoding": ENCODING, "network": NETWORK, "input_dims": 2, "output_dims": 3,
                "dtype": str(combined.dtype), "jit": args.jit, "compiled_step": args.compile,
                "backend": "native MLX Metal", "results": trained}
    args.checkpoint.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    tcnn.free_temporary_memory()
    print(json.dumps({"status": "passed on native MLX Metal", "checkpoints": [str(args.checkpoint), str(separate_path)], **metadata}))


if __name__ == "__main__":
    main()
