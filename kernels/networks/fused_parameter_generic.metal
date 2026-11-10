// Canonical split-K partial accumulation without SIMD-group matrix operations.
// This capability specialization preserves direct FullyFusedMLP use on older
// devices; normal eligible devices use the tiled matrix specialization.
#ifndef TCNN_FUSED_TRAIN_SCALAR
#define TCNN_FUSED_TRAIN_SCALAR half
#endif
#include "fused_mlp_math.metal"
#include "fused/parameter_generic.metal"
using ParameterScalar = TCNN_FUSED_TRAIN_SCALAR;
inline tcnn_fused::ParameterArguments generic_parameter_arguments(
    constant FusedArgs& a, constant tcnn_fused::ParameterDispatch& p) {
    return {a.x.cols, a.shape.inputs, a.shape.width, a.shape.outputs, a.shape.hidden_layers,
            {a.x.rs, a.x.cs}, {a.direction.rs, a.direction.cs}, p};
}
[[kernel, max_total_threads_per_threadgroup(256)]]
void fused_mlp_parameter_partials_generic(
    device const ParameterScalar* input [[buffer(0)]], device const ParameterScalar* hidden [[buffer(1)]],
    device const ParameterScalar* delta [[buffer(2)]], device float* partials [[buffer(3)]],
    constant FusedArgs& a [[buffer(4)]], constant tcnn_fused::ParameterDispatch& p [[buffer(5)]],
    uint2 index [[thread_position_in_grid]]) {
    tcnn_fused::GenericParameterPartial<ParameterScalar, false>::run(
        input, hidden, delta, nullptr, nullptr, nullptr, partials, generic_parameter_arguments(a, p), index.x, index.y);
}
[[kernel, max_total_threads_per_threadgroup(256)]]
void fused_mlp_second_parameter_partials_generic(
    device const ParameterScalar* input [[buffer(0)]], device const ParameterScalar* direction [[buffer(1)]],
    device const ParameterScalar* hidden [[buffer(2)]], device const ParameterScalar* tangent [[buffer(3)]],
    device const ParameterScalar* delta [[buffer(4)]], device const ParameterScalar* tangent_delta [[buffer(5)]],
    device float* partials [[buffer(6)]], constant FusedArgs& a [[buffer(7)]],
    constant tcnn_fused::ParameterDispatch& p [[buffer(8)]], uint2 index [[thread_position_in_grid]]) {
    tcnn_fused::GenericParameterPartial<ParameterScalar, true>::run(
        input, hidden, delta, direction, tangent, tangent_delta, partials, generic_parameter_arguments(a, p), index.x, index.y);
}
[[kernel, max_total_threads_per_threadgroup(256)]]
void fused_mlp_parameter_reduce_generic(
    device const float* partials [[buffer(0)]], device ParameterScalar* gradient [[buffer(1)]],
    constant tcnn_fused::ParameterReduceArguments& a [[buffer(2)]], uint vector [[thread_position_in_grid]]) {
    tcnn_fused::ParameterReduction<ParameterScalar>::run(partials, gradient, a, vector);
}
