"""Small numerical oracles; no substitute execution backend is installed."""

import math

import mlx.core as mx
import numpy as np


GRID = {
    "otype": "HashGrid", "n_levels": 2, "n_features_per_level": 2,
    "base_resolution": 4, "per_level_scale": 1.5, "log2_hashmap_size": 4,
    "interpolation": "Smoothstep",
}
NETWORK = {
    "otype": "GemmMLP", "n_neurons": 16, "n_hidden_layers": 1,
    "activation": "Tanh", "output_activation": "None",
}


def assert_close(actual, expected, rtol=2e-5, atol=2e-6):
    mx.eval(actual)
    if isinstance(expected, mx.array):
        mx.eval(expected)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=rtol, atol=atol)


def assert_finite(*arrays):
    mx.eval(*arrays)
    for array in arrays:
        assert mx.all(mx.isfinite(array)).item()


def frequency_reference(x, frequencies):
    """Independent float64 oracle in dimension/frequency/sin-cos order."""
    x = np.asarray(x, dtype=np.float64)
    scales = math.pi * 2.0 ** np.arange(frequencies, dtype=np.float64)
    angles = x[:, :, None] * scales[None, None, :]
    output = np.stack((np.sin(angles), np.cos(angles)), axis=-1)
    first = np.stack((np.cos(angles), -np.sin(angles)), axis=-1) * scales[None, None, :, None]
    second = -output * scales[None, None, :, None] ** 2
    return output.reshape(x.shape[0], -1), first, second
