#ifndef TCNN_FUSED_PARAMETER_REDUCTION_METAL
#define TCNN_FUSED_PARAMETER_REDUCTION_METAL
#include "parameter_traits.metal"
#include "traits.metal"
namespace tcnn_fused {
// Deterministic split reduction. The input partition order is fixed; no
// floating-point atomics or low-precision partial stores are used. Four adjacent
// parameters share each reduction loop and remain coalesced across GPU lanes.
template <class Scalar>
struct ParameterReduction {
    static TCNN_FUSED_INLINE void run(device const float* partials, device Scalar* gradient,
                                       ParameterReduceArguments args, unsigned vector) {
        const unsigned long first = static_cast<unsigned long>(vector) * 4;
        if (first >= args.parameters || args.mode == 0) return;
        float primal[4] = {0, 0, 0, 0}, tangent[4] = {0, 0, 0, 0};
        for (unsigned split = 0; split < args.splits; ++split) {
            const unsigned long base = static_cast<unsigned long>(split) * args.parameters + first;
            for (unsigned i = 0; i < 4; ++i) {
                if (first + i < args.parameters) {
                    primal[i] += partials[base + i];
                    if (args.second)
                        tangent[i] += partials[static_cast<unsigned long>(args.splits) * args.parameters + base + i];
                }
            }
        }
        for (unsigned i = 0; i < 4; ++i) {
            if (first + i < args.parameters) {
                const float accumulated = primal[i] + (args.mode == 2 ? float(gradient[first + i]) : 0.0f);
                // Existing second-backward ABI performs two stores: T(primal)
                // followed by T(T(primal) + tangent). Keep this rounding point.
                gradient[first + i] = args.second ? Scalar(float(Scalar(accumulated)) + tangent[i]) : Scalar(accumulated);
            }
        }
    }
};
} // namespace tcnn_fused
#endif
