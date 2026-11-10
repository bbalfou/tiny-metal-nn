"""Inference snapshots and automatic dispatch, including old checkpoint compatibility."""
import pytest
import mlx.core as mx
from helpers import GRID, assert_close

pytestmark = pytest.mark.metal


@pytest.mark.parametrize('dtype', [mx.float16, mx.float32])
def test_fixed_params_refresh_update_checkpoint_and_stream(tcnn, tmp_path, dtype):
    model = tcnn.Encoding(2, GRID, dtype=dtype)
    producer, consumer = mx.new_stream(mx.gpu), mx.new_stream(mx.gpu)
    with mx.stream(producer):
        model.params = mx.full(model.params.shape, 0.125)
    p = model.prepare_inference_params(stream=producer)
    x = mx.random.uniform(shape=(257, 2), stream=producer)
    assert p.dtype == dtype and set(model.parameters()) == {'params'}
    y = model.forward_with_params(x, p, stream=consumer)
    assert_close(y, model(x, stream=consumer), atol=0, rtol=0)
    model.save_weights(str(tmp_path / 'fixed.npz'))
    # Indexed update of the SAME Python array object must not defeat refresh.
    original = model.params
    model.params[:] = 0.25
    assert original is model.params
    fresh = model.prepare_inference_params(stream=producer)
    assert_close(model.forward_with_params(x, p, stream=consumer), y, atol=0, rtol=0)
    assert_close(model.forward_with_params(x, fresh, stream=consumer), model(x), atol=0, rtol=0)
    assert mx.any(fresh != p).item()
    model.load_weights(str(tmp_path / 'fixed.npz'))
    reloaded = model.prepare_inference_params(stream=producer)
    assert_close(reloaded, p, atol=0, rtol=0)
    assert_close(model(x), y, atol=0, rtol=0)
    # Same-dtype preparation still snapshots the descriptor before mutations.
    model.set_dtype(dtype)
    same_dtype = model.prepare_inference_params()
    model.params[:] = 0.5
    assert_close(same_dtype, reloaded, atol=0, rtol=0)
    assert mx.all(model.prepare_inference_params() == 0.5).item()


@pytest.mark.parametrize('dtype', [mx.float16, mx.float32])
@pytest.mark.parametrize('batch', [0, 1, 257, 65536])
@pytest.mark.parametrize('interpolation', ['Nearest', 'Linear', 'Smoothstep'])
def test_old_checkpoint_flag_cannot_disable_optimized_inference(tcnn, dtype, batch, interpolation):
    config = dict(otype='HashGrid', n_levels=16, n_features_per_level=2,
                  log2_hashmap_size=19, base_resolution=16, per_level_scale=2.0,
                  interpolation=interpolation)
    grouped = tcnn.Encoding(2, config, dtype=dtype)
    restored = tcnn.Encoding(2, dict(config, grouped_inference=False), dtype=dtype)
    # Avoid tiny initialization values masking interpolation/indexing errors.
    mx.random.seed(17)
    grouped.params = mx.random.uniform(-0.5, 0.5, shape=grouped.params.shape)
    p = grouped.prepare_inference_params()
    x = mx.random.uniform(-0.2, 1.2, shape=(batch, 2))
    a, b = grouped.forward_with_params(x, p), restored.forward_with_params(x, p)
    assert_close(a, b, atol=0, rtol=0)
    assert 'grouped_inference' not in grouped.native_tcnn_module.hyperparams()
    assert 'grouped_inference' not in restored.native_tcnn_module.hyperparams()


@pytest.mark.parametrize('config', [
    dict(otype='HashGrid', n_levels=3, n_features_per_level=4),
    dict(otype='HashGrid', n_levels=3, n_features_per_level=2, dimensions=3),
    dict(otype='DenseGrid', n_levels=3, n_features_per_level=2),
    dict(otype='TiledGrid', n_levels=3, n_features_per_level=2),
])
def test_unsupported_configurations_accept_old_checkpoint_flags(tcnn, config):
    config = config.copy()
    dims = config.pop('dimensions', 2)
    a = tcnn.Encoding(dims, dict(config, log2_hashmap_size=8), dtype=mx.float32)
    b = tcnn.Encoding(dims, dict(config, log2_hashmap_size=8, grouped_inference=False), dtype=mx.float32)
    p = a.prepare_inference_params()
    x = mx.random.uniform(shape=(257, dims))
    assert_close(a.forward_with_params(x, p), b.forward_with_params(x, p), atol=0, rtol=0)
