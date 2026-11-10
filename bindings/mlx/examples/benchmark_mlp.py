"""Run from repository root: .venv/bin/python tiny-metal-nn/bindings/mlx/examples/benchmark_mlp.py"""
import json
import statistics
import time
import mlx.core as mx
import tinymetalnn_mlx as tcnn

# Aligned dimensions avoid hidden padding work; no biases in either model.
dims = [16, 64, 64, 16]
model = tcnn.Network(16, 16, {'otype': 'FullyFusedMLP', 'n_neurons': 64,
    'n_hidden_layers': 2, 'activation': 'ReLU', 'output_activation': 'None'}, seed=42)
weights = []
offset = 0
for ni, no in zip(dims, dims[1:]):
    weights.append(model.params[offset:offset + ni*no].reshape(no, ni).astype(mx.float16))
    offset += ni*no
assert offset == model.n_params
mx.eval(weights)

def plain(x):
    x = x.astype(mx.float16)
    for i, w in enumerate(weights):
        x = x @ w.T
        if i < len(weights)-1:
            x = mx.maximum(x, 0)
    return x

functions = {'native': model, 'mlx': plain, 'mlx_compiled': mx.compile(plain)}

def measure(fn, x):
    for _ in range(10):
        mx.eval(fn(x))
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(50):
            mx.eval(fn(x))
        samples.append((time.perf_counter()-start)*1e6/50)
    return statistics.median(samples)

results = []
for batch in [1, 128, 1024, 16384, 65536]:
    x = mx.random.uniform(shape=(batch, 16)).astype(mx.float16).astype(mx.float32)
    mx.eval(x)
    a, b = model(x), plain(x)
    mx.eval(a, b)
    error = float(mx.max(mx.abs(a.astype(mx.float32)-b.astype(mx.float32))).item())
    assert mx.allclose(a, b, atol=0.003, rtol=0.03).item(), (batch, error)
    row = {'batch': batch, 'max_abs_error': error}
    row.update({name+'_us': measure(fn, x) for name, fn in functions.items()})
    results.append(row)
    print(json.dumps(row), flush=True)
print('Inference only; median of 5 rounds of 50 synchronized calls; 10 warmups.')
