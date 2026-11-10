#!/usr/bin/env python3
"""Train combined and separate native modules, then round-trip their state.

Run after installing bindings/torch on macOS:
    python examples/train_function.py --steps 150
    python examples/train_function.py --steps 150 --jit
"""

import argparse
import json
from pathlib import Path
import pickle

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=513, help="An odd default exercises native batch padding")
    parser.add_argument("--network", choices=["GemmMLP", "FullyFusedMLP"], default="GemmMLP")
    parser.add_argument("--jit", action="store_true", help="Require native automatic model fusion")
    parser.add_argument("--checkpoint", type=Path, default=Path("tinymetalnn_training.pt"))
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps and batch-size must be positive")
    if not torch.backends.mps.is_available():
        raise SystemExit("This example requires a real MPS device and the compiled tinymetalnn extension; it has no CPU fallback.")
    import tinymetalnn as tcnn
    from tinymetalnn.modules import _C

    if not _C.has_networks():
        raise SystemExit("This example needs a build with network support (unset TCNN_NO_NETWORKS).")
    if args.jit and not tcnn.supports_jit_fusion():
        raise SystemExit("Native JIT fusion is unavailable in this build/device.")
    torch.manual_seed(17)
    # Sixteen encoding features avoid different padded feature values between
    # a grid+network composition and a network's implicit Identity encoding.
    encoding_config = {
        "otype": "HashGrid", "n_levels": 4, "n_features_per_level": 4,
        "log2_hashmap_size": 9, "base_resolution": 8, "per_level_scale": 2.0,
        "interpolation": "Smoothstep",
    }
    network_config = {
        "otype": args.network, "n_neurons": 32, "n_hidden_layers": 2,
        "activation": "Tanh", "output_activation": "None",
    }
    combined = tcnn.NetworkWithInputEncoding(2, 1, encoding_config, network_config, seed=17)
    encoding = tcnn.Encoding(2, encoding_config, seed=17, dtype=combined.dtype)
    network = tcnn.Network(encoding.n_output_dims, 1, network_config, seed=17)
    separate = torch.nn.Sequential(encoding, network)
    network_parameters = network.params.numel()
    assert combined.params.numel() == network_parameters + encoding.params.numel()
    # The native compound model preserves the upstream network-first layout.
    with torch.no_grad():
        network.params.copy_(combined.params[:network_parameters])
        encoding.params.copy_(combined.params[network_parameters:])
    for model in (combined, encoding, network):
        model.jit_fusion = args.jit

    coordinates = torch.rand(args.batch_size, 2, device="mps")
    targets = (
        0.4 * torch.sin(2 * torch.pi * coordinates[:, :1])
        + 0.3 * torch.cos(2 * torch.pi * coordinates[:, 1:])
        + 0.2 * coordinates[:, :1]
    )
    tolerance = {"rtol": 1e-2, "atol": 2e-3} if combined.dtype == torch.float16 else {"rtol": 2e-5, "atol": 2e-6}
    with torch.no_grad():
        torch.testing.assert_close(combined(coordinates), separate(coordinates), **tolerance)

    summaries = {}
    for name, model in (("combined", combined), ("separate", separate)):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        with torch.no_grad():
            initial_loss = (model(coordinates).float() - targets).square().mean().item()
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(coordinates)
            loss = (prediction.float() - targets).square().mean()
            loss.backward()
            for parameter in model.parameters():
                assert parameter.dtype == torch.float32
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
            optimizer.step()
            if step in {0, args.steps - 1} or (step + 1) % 50 == 0:
                print(f"{name:8s} step={step + 1:4d} loss={loss.item():.7f}")
        with torch.no_grad():
            final_loss = (model(coordinates).float() - targets).square().mean().item()
        if not final_loss < initial_loss:
            raise AssertionError(f"{name} did not improve: {initial_loss} -> {final_loss}")
        summaries[name] = {"initial_loss": initial_loss, "final_loss": final_loss}

    # Exercise input gradients as well as the optimizer/parameter path.
    differentiable_coordinates = coordinates[:7].detach().requires_grad_()
    input_gradient = torch.autograd.grad(combined(differentiable_coordinates).sum(), differentiable_coordinates)[0]
    assert torch.isfinite(input_gradient).all().item()

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoding_config": encoding_config, "network_config": network_config,
        "combined": combined.state_dict(), "separate": separate.state_dict(),
    }, args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="mps", weights_only=True)
    restored = tcnn.NetworkWithInputEncoding(2, 1, checkpoint["encoding_config"], checkpoint["network_config"])
    restored.load_state_dict(checkpoint["combined"])
    restored.jit_fusion = args.jit
    copied = pickle.loads(pickle.dumps(combined))
    with torch.no_grad():
        expected = combined(coordinates)
        torch.testing.assert_close(restored(coordinates), expected, **tolerance)
        torch.testing.assert_close(copied(coordinates), expected, **tolerance)
    if args.jit:
        assert combined.jit_fusion and encoding.jit_fusion and network.jit_fusion, "JIT fell back; inspect native diagnostics"
    tcnn.free_temporary_memory()
    print(json.dumps({
        "backend": "native MPS / Metal", "execution_dtype": str(combined.dtype),
        "master_parameter_dtype": str(combined.params.dtype), "jit": args.jit,
        "batch_size": args.batch_size, "checkpoint": str(args.checkpoint), "results": summaries,
    }, indent=2))


if __name__ == "__main__":
    main()

