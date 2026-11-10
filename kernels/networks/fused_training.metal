// The public training ABI is unchanged. Host specialization supplies only
// scalar/width/activation constants; depth and matrix strides remain runtime.
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
#include "fused/training_kernel.metal"

using TrainingScalar = TCNN_FUSED_TRAIN_SCALAR;
using TrainingForwardTuning = tcnn_fused::TrainingTuning<TCNN_FUSED_TRAIN_WIDTH, sizeof(TrainingScalar), false>;
using TrainingBackwardTuning = tcnn_fused::TrainingTuning<TCNN_FUSED_TRAIN_WIDTH, sizeof(TrainingScalar), true>;
using TrainingForwardShape = TrainingForwardTuning::Shape;
using TrainingBackwardShape = TrainingBackwardTuning::Shape;
using TrainingAtom = tcnn_fused::TrainingMmaAtom<TrainingScalar>;
using TrainingCopy = tcnn_fused::TrainingCopyAtom<TrainingScalar>;
using TrainingHiddenActivation = tcnn_fused::Activation<TCNN_FUSED_TRAIN_ACTIVATION>;
using TrainingOutputActivation = tcnn_fused::Activation<TCNN_FUSED_TRAIN_OUTPUT_ACTIVATION>;
using TrainingForwardKernel = tcnn_fused::TrainingForward<TrainingForwardShape, TrainingAtom, TrainingCopy,
                                                         TrainingHiddenActivation, TrainingOutputActivation>;
using TrainingBackwardKernel = tcnn_fused::TrainingBackward<TrainingBackwardShape, TrainingAtom, TrainingCopy,
                                                           TrainingHiddenActivation, TrainingOutputActivation>;

inline tcnn_fused::TrainingArguments training_arguments(constant FusedArgs& a) {
    return tcnn_fused::TrainingArguments{
        a.x.cols, a.shape.hidden_layers,
        {a.x.rs, a.x.cs}, {a.output.rs, a.output.cs},
        {a.incoming.rs, a.incoming.cs}, {a.dx.rs, a.dx.cs}, a.write_dx != 0
    };
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void fused_mlp_forward_simd(
    device const uchar* input [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device uchar* pre [[buffer(2)]],
    device uchar* hidden [[buffer(3)]],
    constant FusedArgs& a [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    if (ulong(group.x) * TrainingForwardShape::Samples >= ulong(a.x.cols))
        return;
    alignas(16) threadgroup TrainingScalar scratch[TrainingForwardTuning::Scratch::Elements];
    TrainingForwardKernel::run(
        reinterpret_cast<device const TrainingScalar*>(input), reinterpret_cast<device const TrainingScalar*>(weights),
        reinterpret_cast<device TrainingScalar*>(pre), reinterpret_cast<device TrainingScalar*>(hidden),
        scratch, training_arguments(a), group.x, tid, sg
    );
}

#if TCNN_FUSED_TRAIN_WIDTH == 64
[[kernel, max_total_threads_per_threadgroup(64)]]
#else
[[kernel, max_total_threads_per_threadgroup(128)]]
#endif
void fused_mlp_backward_simd(
    device const uchar* weights [[buffer(0)]],
    device const uchar* pre [[buffer(1)]],
    device const uchar* hidden [[buffer(2)]],
    device const uchar* output [[buffer(3)]],
    device const uchar* incoming [[buffer(4)]],
    device uchar* delta [[buffer(5)]],
    device uchar* dx [[buffer(6)]],
    constant FusedArgs& a [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    if (ulong(group.x) * TrainingBackwardShape::Samples >= ulong(a.x.cols))
        return;
    alignas(16) threadgroup TrainingScalar scratch[TrainingBackwardTuning::Scratch::Elements];
    TrainingBackwardKernel::run(
        reinterpret_cast<device const TrainingScalar*>(weights), reinterpret_cast<device const TrainingScalar*>(pre),
        reinterpret_cast<device const TrainingScalar*>(hidden), reinterpret_cast<device const TrainingScalar*>(output),
        reinterpret_cast<device const TrainingScalar*>(incoming), reinterpret_cast<device TrainingScalar*>(delta),
        reinterpret_cast<device TrainingScalar*>(dx), scratch, training_arguments(a), group.x, tid, sg
    );
}
