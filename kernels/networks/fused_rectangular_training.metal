// Persistent rectangular fused training. Boundary dimensions, depth, strides,
// and tail sizes are runtime data; hidden geometry and activations specialize.
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
#include "fused/training_rectangular.metal"
#include "fused/saved_forward.metal"

using TrainingScalar = TCNN_FUSED_TRAIN_SCALAR;
using TrainingForwardTuning = tcnn_fused::RectangularTuning<TCNN_FUSED_TRAIN_WIDTH, sizeof(TrainingScalar), false>;
using TrainingBackwardTuning = tcnn_fused::RectangularTuning<TCNN_FUSED_TRAIN_WIDTH, sizeof(TrainingScalar), true>;
using TrainingForwardShape = TrainingForwardTuning::Shape;
using TrainingBackwardShape = TrainingBackwardTuning::Shape;
using TrainingAtom = tcnn_fused::TrainingMmaAtom<TrainingScalar>;
using TrainingCopy = tcnn_fused::TrainingPackedCopyAtom<TrainingScalar>;
using TrainingHiddenActivation = tcnn_fused::Activation<TCNN_FUSED_TRAIN_ACTIVATION>;
using TrainingOutputActivation = tcnn_fused::Activation<TCNN_FUSED_TRAIN_OUTPUT_ACTIVATION>;
using TrainingForwardKernel = tcnn_fused::RectangularForward<TrainingForwardShape, TrainingAtom, TrainingCopy,
                                                         TrainingHiddenActivation, TrainingOutputActivation>;
using TrainingBackwardKernel = tcnn_fused::RectangularBackward<TrainingBackwardShape, TrainingAtom, TrainingCopy,
                                                           TrainingHiddenActivation, TrainingOutputActivation>;

inline tcnn_fused::RectangularArguments training_arguments(constant FusedArgs& a) {
    return tcnn_fused::RectangularArguments{{
        a.x.cols, a.shape.hidden_layers,
        {a.x.rs, a.x.cs}, {a.output.rs, a.output.cs},
        {a.incoming.rs, a.incoming.cs}, {a.dx.rs, a.dx.cs}, a.write_dx != 0
    }, a.shape.inputs, a.shape.outputs};
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void fused_mlp_forward_rectangular_simd(
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
    // Aligned FP16 boundaries use the compact inference matrix schedule,
    // retaining preactivations and activations for ordinary training backward.
#if TCNN_SAVED_TRAIN_INPUTS > 0
    using Tuning = tcnn_fused::InferenceTuning<TCNN_FUSED_TRAIN_WIDTH, 2,
        TCNN_SAVED_TRAIN_INPUTS, TCNN_SAVED_TRAIN_OUTPUTS>;
    static_assert(Tuning::Samples == TrainingForwardShape::Samples &&
                  Tuning::Threads == TrainingForwardShape::Threads &&
                  Tuning::ScratchElements <= TrainingForwardTuning::Scratch::Elements,
                  "saved forward must match the host launch and scratch allocation");
    using Kernel = tcnn_fused::inference_detail::SavedForward<Tuning,
        tcnn_fused::InferenceAtom<TrainingScalar>, tcnn_fused::TrainingPackedCopyAtom<TrainingScalar>,
        TrainingHiddenActivation, TrainingOutputActivation>;
    alignas(16) threadgroup TrainingScalar scratch[Tuning::ScratchElements];
    Kernel::run(reinterpret_cast<device const TrainingScalar*>(input),
        reinterpret_cast<device const TrainingScalar*>(weights),
        reinterpret_cast<device TrainingScalar*>(pre), reinterpret_cast<device TrainingScalar*>(hidden),
        scratch, training_arguments(a), group.x, tid, sg);
    return;
#else
    alignas(16) threadgroup TrainingScalar scratch[TrainingForwardTuning::Scratch::Elements];
    TrainingForwardKernel::run(
        reinterpret_cast<device const TrainingScalar*>(input), reinterpret_cast<device const TrainingScalar*>(weights),
        reinterpret_cast<device TrainingScalar*>(pre), reinterpret_cast<device TrainingScalar*>(hidden),
        scratch, training_arguments(a), group.x, tid, sg
    );
#endif
}

[[kernel, max_total_threads_per_threadgroup(TrainingBackwardShape::Threads)]]
void fused_mlp_backward_rectangular_simd(
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
