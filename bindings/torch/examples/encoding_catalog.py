#!/usr/bin/env python3
"""Exercise the native encoding factory families in both execution precisions.

    python examples/encoding_catalog.py
    python examples/encoding_catalog.py --full-configurations
    python examples/encoding_catalog.py --dtype fp32 --jit

The default cases cover each dispatch/configuration axis economically. The
expanded sweep includes all enabled grid dimensions, feature widths, grid
storage kinds and interpolations, and all Permuto dimension/feature choices.
The pytest configuration suite imports this catalog to keep their cases equal.
Continuous JSON ranges (scales, table sizes, seeds) are represented by selected
values, not claimed to be an exhaustive Cartesian product.
"""

import argparse
import itertools
import json


def encoding_cases(full=False):
    yield "empty", 2, {"otype": "Empty"}
    yield "identity", 3, {"otype": "Identity", "scale": 0.75, "offset": 0.1}
    yield "frequency", 2, {"otype": "Frequency", "n_frequencies": 3}
    yield "triangle", 2, {"otype": "TriangleWave", "n_frequencies": 3}
    for bins in ((1, 2, 4, 8, 16, 32) if full else (4,)):
        yield f"oneblob-{bins}", 2, {"otype": "OneBlob", "n_bins": bins}
    for degree in (range(1, 9) if full else (1, 4, 8)):
        yield f"sh-{degree}", 3, {"otype": "SphericalHarmonics", "degree": degree}
    if full:
        grid_axes = itertools.product((2, 3, 4), (1, 2, 4, 8), ("HashGrid", "DenseGrid", "TiledGrid"), ("Nearest", "Linear", "Smoothstep"))
    else:
        # Representatives cover every axis without 108 shader cases. The JSON
        # factory enables CoherentPrime: its grid dimensions are 2, 3, and 4.
        grid_axes = [
            (2, 1, "HashGrid", "Nearest"), (2, 2, "HashGrid", "Linear"),
            (3, 4, "DenseGrid", "Smoothstep"), (4, 8, "TiledGrid", "Linear"),
            (2, 4, "TiledGrid", "Nearest"), (3, 2, "HashGrid", "Smoothstep"),
        ]
    for dims, features, otype, interpolation in grid_axes:
        yield f"{otype}-d{dims}-f{features}-{interpolation}", dims, {
            "otype": otype, "n_levels": 2, "n_features_per_level": features,
            "base_resolution": 4, "per_level_scale": 1.5, "log2_hashmap_size": 4,
            "interpolation": interpolation,
        }
    yield "grid-total-features", 2, {
        "otype": "Grid", "type": "Hash", "n_features": 8, "n_features_per_level": 4,
        "base_resolution": 4, "log2_hashmap_size": 4, "stochastic_interpolation": True,
    }
    dimensions = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24) if full else (1, 3, 12, 24)
    features = (1, 2, 4, 8) if full else (2,)
    for dims, feature in itertools.product(dimensions, features):
        yield f"permuto-d{dims}-f{feature}", dims, {
            "otype": "Permuto", "n_levels": 2, "n_features_per_level": feature,
            "log2_hashmap_size": 4, "base_scale": 2.0, "per_level_scale": 1.5, "seed": 17,
        }
    yield "permuto-limited-input-gradient", 4, {
        "otype": "Permuto", "n_grid_features": 4, "n_features_per_level": 2,
        "log2_hashmap_size": 4, "base_scale": 2.0, "max_input_grad_dims": 2,
    }
    for reduction in ("Concatenation", "Sum", "Product"):
        yield f"composite-{reduction}", 2, {
            "otype": "Composite", "reduction": reduction,
            "nested": [
                {"otype": "Identity", "n_dims_to_encode": 2, "dims_to_encode_begin": 0, "scale": 0.5},
                {"otype": "Identity", "n_dims_to_encode": 2, "dims_to_encode_begin": 0, "scale": 1.5},
            ],
        }
    for alias in ("NRC", "OneBlobFrequency"):
        yield alias, 9, {"otype": alias, "n_frequencies": 2, "n_bins": 4}
    for mode in (("Hard", "Discontinuous", "Soft", "Continuous") if full else ("Hard", "Soft")):
        for base in (("HashGrid", "Permuto") if full else ("HashGrid",)):
            yield f"lod-{mode}-{base}", 3, {
                "otype": "MultiLevelEncodingLoD", "lod_type": mode,
                "base": {"otype": base, "n_levels": 2, "n_features_per_level": 2,
                         "base_resolution": 4, "base_scale": 2.0, "per_level_scale": 1.5, "log2_hashmap_size": 4},
            }



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["both", "fp16", "fp32"], default="both")
    parser.add_argument("--full-configurations", action="store_true")
    parser.add_argument("--jit", action="store_true")
    args = parser.parse_args()
    import torch
    if not torch.backends.mps.is_available():
        raise SystemExit("This example requires a real MPS device and the compiled tinymetalnn extension; it does not run the Reference backend.")
    import tinymetalnn as tcnn
    if args.jit and not tcnn.supports_jit_fusion():
        raise SystemExit("Native JIT fusion is unavailable in this build/device.")
    dtypes = [torch.float16, torch.float32] if args.dtype == "both" else [torch.float16 if args.dtype == "fp16" else torch.float32]
    completed = 0
    for dtype in dtypes:
        for name, dims, config in encoding_cases(args.full_configurations):
            model = tcnn.Encoding(dims, config, seed=17, dtype=dtype)
            model.jit_fusion = args.jit
            # Transposition and an odd batch exercise the wrapper's contiguous
            # conversion and row padding across all of these factory families.
            x = torch.linspace(0.11, 0.89, 5 * dims, device="mps").reshape(dims, 5).t().detach().requires_grad_()
            y = model(x)
            assert y.shape == (5, model.n_output_dims) and y.dtype == dtype
            assert torch.isfinite(y).all().item()
            y.float().sum().backward()
            assert x.grad is not None and torch.isfinite(x.grad).all().item()
            if model.params.numel():
                assert model.params.dtype == torch.float32
                assert model.params.grad is not None and torch.isfinite(model.params.grad).all().item()
            if config.get("max_input_grad_dims") == 2:
                torch.testing.assert_close(x.grad[:, 2:], torch.zeros_like(x.grad[:, 2:]), rtol=0, atol=0)
            if args.jit:
                assert model.jit_fusion, f"{name}: JIT fell back; inspect native diagnostics"
            completed += 1
            print(json.dumps({
                "case": name, "dtype": str(dtype), "input_dims": dims,
                "output_dims": model.n_output_dims, "parameters": model.params.numel(),
                "input_gradient_norm": x.grad.norm().item(), "status": "passed on native MPS",
            }))
    tcnn.free_temporary_memory()
    print(f"PASS {completed} native encoding/configuration/precision cases")


if __name__ == "__main__":
    main()
