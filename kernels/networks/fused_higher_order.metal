#ifndef TCNN_FUSED_TRAIN_WIDTH
#define TCNN_FUSED_TRAIN_WIDTH 64
#endif
#ifndef TCNN_FUSED_TRAIN_SCALAR
#define TCNN_FUSED_TRAIN_SCALAR half
#endif
#ifndef TCNN_FUSED_TRAIN_ACTIVATION
#define TCNN_FUSED_TRAIN_ACTIVATION 0
#endif
#ifndef TCNN_FUSED_TRAIN_OUTPUT_ACTIVATION
#define TCNN_FUSED_TRAIN_OUTPUT_ACTIVATION 9
#endif

#include "fused_mlp_math.metal"
#include "fused/higher_order_kernel.metal"

using DerivativeScalar = TCNN_FUSED_TRAIN_SCALAR;
using DerivativeShape = tcnn_fused::HigherOrderTuning<TCNN_FUSED_TRAIN_WIDTH, sizeof(DerivativeScalar)>;
using DerivativeAtom = tcnn_fused::HigherOrderMmaAtom<DerivativeScalar>;
using DerivativeCopy = tcnn_fused::HigherOrderCopy<DerivativeScalar>;
using DerivativeMath = tcnn_fused::HigherOrderMath<DerivativeScalar>;
using DerivativeHidden = tcnn_fused::Activation<TCNN_FUSED_TRAIN_ACTIVATION>;
using DerivativeOutput = tcnn_fused::Activation<TCNN_FUSED_TRAIN_OUTPUT_ACTIVATION>;
using TangentKernel = tcnn_fused::HigherOrderTangent<DerivativeShape, DerivativeAtom, DerivativeCopy,
                                                DerivativeMath, DerivativeHidden, DerivativeOutput>;
using SecondKernel = tcnn_fused::HigherOrderBackward<DerivativeShape, DerivativeAtom, DerivativeCopy,
                                                  DerivativeMath, DerivativeHidden, DerivativeOutput>;

inline tcnn_fused::HigherOrderArguments higher_order_arguments(constant FusedArgs& a) {
    return {a.x.cols, a.shape.inputs, a.shape.outputs, a.shape.hidden_layers,
            {a.direction.rs, a.direction.cs}, {a.incoming.rs, a.incoming.cs},
            {a.dx.rs, a.dx.cs}, {a.ddy.rs, a.ddy.cs}, a.write_dx != 0, a.write_ddy != 0};
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void fused_mlp_tangent_simd(
    device const uchar* direction [[buffer(0)]], device const uchar* weights [[buffer(1)]],
    device const uchar* pre [[buffer(2)]], device const uchar* hidden [[buffer(3)]],
    device uchar* tangent_pre [[buffer(4)]], device uchar* tangent [[buffer(5)]], device uchar* ddy [[buffer(6)]],
    constant FusedArgs& a [[buffer(7)]], uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    if (ulong(group.x) * DerivativeShape::Samples >= ulong(a.x.cols)) return;
    alignas(16) threadgroup DerivativeScalar scratch[DerivativeShape::ScratchElements];
    TangentKernel::run(reinterpret_cast<device const DerivativeScalar*>(direction),
        reinterpret_cast<device const DerivativeScalar*>(weights), reinterpret_cast<device const DerivativeScalar*>(pre),
        reinterpret_cast<device const DerivativeScalar*>(hidden), reinterpret_cast<device DerivativeScalar*>(tangent_pre),
        reinterpret_cast<device DerivativeScalar*>(tangent), reinterpret_cast<device DerivativeScalar*>(ddy),
        scratch, higher_order_arguments(a), group.x, tid, sg);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void fused_mlp_second_backward_simd(
    device const uchar* weights [[buffer(0)]], device const uchar* pre [[buffer(1)]],
    device const uchar* hidden [[buffer(2)]], device const uchar* tangent_pre [[buffer(3)]],
    device const uchar* incoming [[buffer(4)]], device uchar* delta [[buffer(5)]],
    device uchar* tangent_delta [[buffer(6)]], device uchar* dx [[buffer(7)]], constant FusedArgs& a [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    if (ulong(group.x) * DerivativeShape::Samples >= ulong(a.x.cols)) return;
    alignas(16) threadgroup DerivativeScalar scratch[DerivativeShape::ScratchElements];
    SecondKernel::run(reinterpret_cast<device const DerivativeScalar*>(weights),
        reinterpret_cast<device const DerivativeScalar*>(pre), reinterpret_cast<device const DerivativeScalar*>(hidden),
        reinterpret_cast<device const DerivativeScalar*>(tangent_pre), reinterpret_cast<device const DerivativeScalar*>(incoming),
        reinterpret_cast<device DerivativeScalar*>(delta), reinterpret_cast<device DerivativeScalar*>(tangent_delta),
        reinterpret_cast<device DerivativeScalar*>(dx), scratch, higher_order_arguments(a), group.x, tid, sg);
}
