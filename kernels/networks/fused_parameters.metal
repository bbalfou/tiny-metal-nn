#ifndef TCNN_FUSED_TRAIN_WIDTH
#define TCNN_FUSED_TRAIN_WIDTH 64
#endif
#ifndef TCNN_FUSED_TRAIN_SCALAR
#define TCNN_FUSED_TRAIN_SCALAR half
#endif

#include "fused_mlp_math.metal"
#include "fused/parameter_kernel.metal"

using ParameterScalar = TCNN_FUSED_TRAIN_SCALAR;
using ParameterShape = tcnn_fused::ParameterTuning<TCNN_FUSED_TRAIN_WIDTH, sizeof(ParameterScalar)>;
using ParameterAtom = tcnn_fused::ParameterMmaAtom<ParameterScalar>;
using ParameterCopy = tcnn_fused::TrainingCopyAtom<ParameterScalar>;

inline tcnn_fused::ParameterArguments parameter_arguments(constant FusedArgs& a,
                                                          constant tcnn_fused::ParameterDispatch& p) {
    return {a.x.cols, a.shape.inputs, a.shape.width, a.shape.outputs, a.shape.hidden_layers,
            {a.x.rs, a.x.cs}, {a.direction.rs, a.direction.cs}, p};
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void fused_mlp_parameter_partials(
    device const uchar* input [[buffer(0)]], device const uchar* hidden [[buffer(1)]],
    device const uchar* delta [[buffer(2)]], device float* partials [[buffer(3)]],
    constant FusedArgs& a [[buffer(4)]], constant tcnn_fused::ParameterDispatch& p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    alignas(16) threadgroup ParameterScalar scratch[ParameterShape::ScratchElements];
    alignas(16) threadgroup float output[ParameterShape::OutputElements];
    tcnn_fused::ParameterGradient<ParameterShape, ParameterAtom, ParameterCopy, false>::run(
        reinterpret_cast<device const ParameterScalar*>(input), reinterpret_cast<device const ParameterScalar*>(hidden),
        reinterpret_cast<device const ParameterScalar*>(delta), nullptr, nullptr, nullptr,
        partials, scratch, output, parameter_arguments(a, p), group.x, group.y, tid, sg);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void fused_mlp_second_parameter_partials(
    device const uchar* input [[buffer(0)]], device const uchar* direction [[buffer(1)]],
    device const uchar* hidden [[buffer(2)]], device const uchar* tangent [[buffer(3)]],
    device const uchar* delta [[buffer(4)]], device const uchar* tangent_delta [[buffer(5)]],
    device float* partials [[buffer(6)]], constant FusedArgs& a [[buffer(7)]],
    constant tcnn_fused::ParameterDispatch& p [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    alignas(16) threadgroup ParameterScalar scratch[ParameterShape::ScratchElements];
    alignas(16) threadgroup float output[ParameterShape::OutputElements];
    tcnn_fused::ParameterGradient<ParameterShape, ParameterAtom, ParameterCopy, true>::run(
        reinterpret_cast<device const ParameterScalar*>(input), reinterpret_cast<device const ParameterScalar*>(hidden),
        reinterpret_cast<device const ParameterScalar*>(delta), reinterpret_cast<device const ParameterScalar*>(direction),
        reinterpret_cast<device const ParameterScalar*>(tangent), reinterpret_cast<device const ParameterScalar*>(tangent_delta),
        partials, scratch, output, parameter_arguments(a, p), group.x, group.y, tid, sg);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void fused_mlp_parameter_reduce(
    device const float* partials [[buffer(0)]], device ParameterScalar* gradient [[buffer(1)]],
    constant tcnn_fused::ParameterReduceArguments& a [[buffer(2)]], uint vector [[thread_position_in_grid]]
) {
    tcnn_fused::ParameterReduction<ParameterScalar>::run(partials, gradient, a, vector);
}
