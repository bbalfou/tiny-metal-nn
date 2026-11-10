#ifndef TCNN_FUSED_PARAMETER_GENERIC_METAL
#define TCNN_FUSED_PARAMETER_GENERIC_METAL
#include "parameter_traits.metal"
#include "parameter_reduction.metal"

namespace tcnn_fused {
// A capability specialization of the canonical split-K algorithm for devices
// without the matrix backend. Each invocation owns one parameter and one batch
// partition. It exports the identical bounded FP32 partial array and uses the
// common deterministic reduction; it never reduces the full batch in one GPU
// invocation when the launch plan contains multiple partitions.
template <class Scalar, bool Second>
struct GenericParameterPartial {
    static TCNN_FUSED_INLINE void run(
        device const Scalar* input, device const Scalar* hidden, device const Scalar* delta,
        device const Scalar* direction, device const Scalar* tangent, device const Scalar* tangent_delta,
        device float* partials, ParameterArguments args, unsigned index, unsigned split
    ) {
        const unsigned long parameter = static_cast<unsigned long>(index) + args.dispatch.tile_offset;
        if (parameter >= args.dispatch.parameters || split >= args.dispatch.splits) return;
        unsigned layer = 0;
        unsigned long offset = 0;
        for (; layer < args.hidden_layers; ++layer) {
            const unsigned long count = static_cast<unsigned long>(args.width) * parameter_inputs(args, layer);
            if (parameter - offset < count) break;
            offset += count;
        }
        const unsigned rows = parameter_rows(args, layer), columns = parameter_inputs(args, layer);
        const unsigned row = unsigned((parameter - offset) / columns), column = unsigned((parameter - offset) % columns);
        const unsigned long saved = static_cast<unsigned long>(layer) * args.batch * args.width;
        const unsigned long previous = layer ? static_cast<unsigned long>(layer - 1) * args.batch * args.width : 0;
        // Match the matrix specialization's complete K32 panel partitions.
        const unsigned long panels = (static_cast<unsigned long>(args.batch) + 31) / 32;
        const unsigned long per_split = (panels + args.dispatch.splits - 1) / args.dispatch.splits;
        const unsigned long begin = static_cast<unsigned long>(split) * per_split * 32;
        const unsigned long end = begin + per_split * 32;
        float primal_sum = 0, tangent_sum = 0;
        for (unsigned long sample = begin; sample < end && sample < args.batch; ++sample) {
            const float value = float(layer ? hidden[previous + sample * args.width + column]
                : input[sample * args.input.sample_stride + static_cast<unsigned long>(column) * args.input.row_stride]);
            primal_sum += float(delta[saved + sample * rows + row]) * value;
            if (Second) {
                const float value_tangent = float(layer ? tangent[previous + sample * args.width + column]
                    : direction[sample * args.direction.sample_stride + static_cast<unsigned long>(column) * args.direction.row_stride]);
                tangent_sum += float(tangent_delta[saved + sample * rows + row]) * value_tangent;
            }
        }
        const unsigned long base = static_cast<unsigned long>(split) * args.dispatch.parameters + parameter;
        partials[base] = primal_sum;
        if (Second)
            partials[static_cast<unsigned long>(args.dispatch.splits) * args.dispatch.parameters + base] = tangent_sum;
    }
};
} // namespace tcnn_fused
#endif
