// One sample owner per level; identical interpolation/order to per-level AOT.
// Y partitions levels into independent threadgroups without dispatch barriers.
#include "first_order.metal"
using namespace tcnn_encoding;
struct HashGridInferenceLevel {
    Args args;
    uint parameter_offset;
};
#define HASH_GRID_INFERENCE_ENTRY(T, S) \
kernel void encoding_hash_grid_inference_##S( \
    device const float* input [[buffer(0)]], device const T* parameters [[buffer(1)]], \
    device const float* limits [[buffer(2)]], device T* output [[buffer(3)]], \
    constant HashGridInferenceLevel* levels [[buffer(4)]], \
    uint2 position [[thread_position_in_grid]]) { \
    constant auto& level = levels[position.y]; \
    GridPolicy<2, 2>::forward<T>(level.args, input, parameters + level.parameter_offset, \
                               limits, output, position.x); \
}
HASH_GRID_INFERENCE_ENTRY(float, f32)
HASH_GRID_INFERENCE_ENTRY(half, f16)
#undef HASH_GRID_INFERENCE_ENTRY
