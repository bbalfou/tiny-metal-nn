// Canonical private/shared-state inference. Host specialization selects masked
// boundary tiles when dimensions, alignment, layouts, or batch tails require it.
#ifndef TCNN_FUSED_INFERENCE_WIDTH
#define TCNN_FUSED_INFERENCE_WIDTH 32
#define TCNN_FUSED_INFERENCE_INPUTS 32
#define TCNN_FUSED_INFERENCE_OUTPUTS 32
#define TCNN_FUSED_INFERENCE_SCALAR half
#define TCNN_FUSED_INFERENCE_ACTIVATION 0
#define TCNN_FUSED_INFERENCE_OUTPUT_ACTIVATION 9
#define TCNN_FUSED_INFERENCE_PRELOAD 4
#define TCNN_FUSED_INFERENCE_THREADS 128
#define TCNN_FUSED_INFERENCE_BOUNDARY 0
#endif
#include "fused/inference_kernel.metal"

using InferenceScalar = TCNN_FUSED_INFERENCE_SCALAR;
using InferenceAtom = tcnn_fused::InferenceAtom<InferenceScalar>;
using InferenceActivation = tcnn_fused::Activation<TCNN_FUSED_INFERENCE_ACTIVATION>;
using InferenceOutputActivation = tcnn_fused::Activation<TCNN_FUSED_INFERENCE_OUTPUT_ACTIVATION>;
#if TCNN_FUSED_INFERENCE_BOUNDARY
using InferenceTuning = tcnn_fused::BoundaryInferenceTuning<
    TCNN_FUSED_INFERENCE_WIDTH, sizeof(InferenceScalar), TCNN_FUSED_INFERENCE_INPUTS, TCNN_FUSED_INFERENCE_OUTPUTS>;
#else
using InferenceTuning = tcnn_fused::InferenceTuning<
    TCNN_FUSED_INFERENCE_WIDTH, sizeof(InferenceScalar), TCNN_FUSED_INFERENCE_INPUTS,
    TCNN_FUSED_INFERENCE_OUTPUTS, TCNN_FUSED_INFERENCE_PRELOAD>;
#endif

[[kernel, max_total_threads_per_threadgroup(TCNN_FUSED_INFERENCE_THREADS)]]
void fused_mlp_inference(
    device const InferenceScalar* input [[buffer(0)]],
    device const InferenceScalar* params [[buffer(1)]],
    device InferenceScalar* output [[buffer(2)]],
    constant tcnn_fused::InferenceArguments& a [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
#if TCNN_FUSED_INFERENCE_BOUNDARY
    using Kernel = tcnn_fused::inference_detail::Boundary<InferenceTuning,
        InferenceAtom, tcnn_fused::InferenceCopy<InferenceScalar>, InferenceActivation, InferenceOutputActivation>;
    alignas(16) threadgroup InferenceScalar scratch[InferenceTuning::ScratchElements];
    Kernel::run(input, params, output, scratch, a, group.x, tid, sg);
#elif TCNN_FUSED_INFERENCE_WIDTH == 16
    (void)tid; (void)sg;
    using Kernel = tcnn_fused::inference_detail::Private<InferenceTuning,
        InferenceAtom, InferenceActivation, InferenceOutputActivation>;
    Kernel::run(input, params, output, a, group.x);
#else
    using Kernel = tcnn_fused::inference_detail::Shared<InferenceTuning,
        InferenceAtom, tcnn_fused::InferenceCopy<InferenceScalar>, InferenceActivation, InferenceOutputActivation>;
    alignas(16) threadgroup InferenceScalar scratch[InferenceTuning::ScratchElements];
    Kernel::run(input, params, output, scratch, a, group.x, tid, sg);
#endif
}
