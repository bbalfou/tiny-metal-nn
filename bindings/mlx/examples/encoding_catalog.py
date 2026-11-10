#!/usr/bin/env python3
"""Exercise the native MLX encoding factories and both storage precisions.

Run ``python examples/encoding_catalog.py`` for an economical dispatch sweep,
or add ``--full-configurations`` for the enabled finite configuration product.
``--jit`` requires and checks native runtime fusion. All execution uses Metal.
The case generator is independent of MLX and is also used by the native tests.
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
        grid_axes = itertools.product(
            (2, 3, 4), (1, 2, 4, 8), ("HashGrid", "DenseGrid", "TiledGrid"),
            ("Nearest", "Linear", "Smoothstep"),
        )
    else:
        grid_axes = [
            (2, 1, "HashGrid", "Nearest"), (2, 2, "HashGrid", "Linear"),
            (3, 4, "DenseGrid", "Smoothstep"), (4, 8, "TiledGrid", "Linear"),
            (2, 4, "TiledGrid", "Nearest"), (3, 2, "HashGrid", "Smoothstep"),
        ]
    for dims, features, otype, interpolation in grid_axes:
        yield f"{otype}-d{dims}-f{features}-{interpolation}", dims, {
            "otype": otype, "n_levels": 2, "n_features_per_level": features,
            "base_resolution": 4, "per_level_scale": 1.5,
            "log2_hashmap_size": 4, "interpolation": interpolation,
        }
    yield "grid-total-features", 2, {
        "otype": "Grid", "type": "Hash", "n_features": 8,
        "n_features_per_level": 4, "base_resolution": 4,
        "log2_hashmap_size": 4, "stochastic_interpolation": True,
    }
    dimensions = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24) if full else (1, 3, 12, 24)
    features = (1, 2, 4, 8) if full else (2,)
    for dims, feature in itertools.product(dimensions, features):
        yield f"permuto-d{dims}-f{feature}", dims, {
            "otype": "Permuto", "n_levels": 2, "n_features_per_level": feature,
            "log2_hashmap_size": 4, "base_scale": 2.0,
            "per_level_scale": 1.5, "seed": 17,
        }
    yield "permuto-limited-input-gradient", 4, {
        "otype": "Permuto", "n_grid_features": 4,
        "n_features_per_level": 2, "log2_hashmap_size": 4,
        "base_scale": 2.0, "max_input_grad_dims": 2,
    }
    for reduction in ("Concatenation", "Sum", "Product"):
        yield f"composite-{reduction}", 2, {
            "otype": "Composite", "reduction": reduction,
            "nested": [
                {"otype": "Identity", "n_dims_to_encode": 2,
                 "dims_to_encode_begin": 0, "scale": 0.5},
                {"otype": "Identity", "n_dims_to_encode": 2,
                 "dims_to_encode_begin": 0, "scale": 1.5},
            ],
        }
    for alias in ("NRC", "OneBlobFrequency"):
        yield alias, 9, {"otype": alias, "n_frequencies": 2, "n_bins": 4}
    for mode in (("Hard", "Discontinuous", "Soft", "Continuous") if full else ("Hard", "Soft")):
        for base in (("HashGrid", "Permuto") if full else ("HashGrid",)):
            yield f"lod-{mode}-{base}", 3, {
                "otype": "MultiLevelEncodingLoD", "lod_type": mode,
                "base": {
                    "otype": base, "n_levels": 2, "n_features_per_level": 2,
                    "base_resolution": 4, "base_scale": 2.0,
                    "per_level_scale": 1.5, "log2_hashmap_size": 4,
                },
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("both", "fp16", "fp32"), default="both")
    parser.add_argument("--full-configurations", action="store_true")
    parser.add_argument("--jit", action="store_true")
    args = parser.parse_args()

    import mlx.core as mx
    if not mx.metal.is_available():
        raise SystemExit("This example requires MLX Metal and the native tinymetalnn_mlx extension.")
    import tinymetalnn_mlx as tcnn
    if args.jit and not tcnn.supports_jit_fusion():
        raise SystemExit("Native JIT fusion is unavailable in this build/device.")
    dtypes = (mx.float16, mx.float32) if args.dtype == "both" else (
        mx.float16 if args.dtype == "fp16" else mx.float32,
    )
    completed = 0
    for dtype in dtypes:
        for name, dims, config in encoding_cases(args.full_configurations):
            model = tcnn.Encoding(dims, config, seed=17, dtype=dtype)
            model.jit_fusion = args.jit
            # A non-contiguous view and odd row count exercise input adaptation.
            x = mx.linspace(0.11, 0.89, 5 * dims, stream=mx.gpu).reshape(dims, 5).T
            y = model(x)
            loss = lambda z, p: mx.sum(model.forward_with_params(z, p).astype(mx.float32))
            gx, gp = mx.grad(loss, argnums=(0, 1))(x, model.params)
            mx.eval(y, gx, gp)
            assert y.shape == (5, model.n_output_dims) and y.dtype == dtype
            assert gx.shape == x.shape and gp.shape == model.params.shape
            assert model.params.dtype == mx.float32 and gp.dtype == mx.float32
            assert all(mx.all(mx.isfinite(a)).item() for a in (y, gx, gp))
            if config.get("max_input_grad_dims") == 2:
                assert mx.all(gx[:, 2:] == 0).item()
            if args.jit:
                assert model.jit_fusion, f"{name}: JIT fell back; inspect native diagnostics"
            completed += 1
            print(json.dumps({
                "case": name, "dtype": str(dtype), "input_dims": dims,
                "output_dims": model.n_output_dims, "parameters": model.params.size,
                "input_gradient_norm": mx.sqrt(mx.sum(gx * gx)).item(),
                "status": "passed on native MLX Metal",
            }))
    tcnn.free_temporary_memory()
    print(f"PASS {completed} native MLX Metal encoding configurations")


if __name__ == "__main__":
    main()
