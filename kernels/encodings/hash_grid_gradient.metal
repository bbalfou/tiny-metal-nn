// Canonical grouped HashGrid contribution enumeration. Native reduction and
// portable correctness checks share this policy; Reference keeps its separate
// feature-owned gather implementation in math.metal.
#ifndef TCNN_HASH_GRID_GRADIENT
#define TCNN_HASH_GRID_GRADIENT
#include "first_order.metal"
#ifndef __METAL_VERSION__
#include <tiny-metal-nn/detail/kernels/encodings/hash_grid_parameter_gradient.h>
#define HG_CONSTANT
#define HG_DEVICE
#define HG_THREAD
#else
#define HG_CONSTANT constant
#define HG_DEVICE device
#define HG_THREAD thread
#endif

namespace tcnn_encoding {
#ifdef __METAL_VERSION__
struct HashGridGradientLevel {
    Args args;
    uint parameter_offset;
};
static_assert(sizeof(HashGridGradientLevel) == 376, "Grouped HashGrid gradient host/MSL layout.");
#else
using tcnn::detail::kernels::encodings::HashGridGradientLevel;
#endif

template <uint D, uint F>
struct HashGridGradientSample {
    bool active;
    uint cell[D];
    float fraction[D], upstream[F], random;

    template <typename T>
    void prepare(HG_CONSTANT Args& a, HG_DEVICE const float* input,
                 HG_DEVICE const float* limits, HG_DEVICE const T* incoming, uint sample) {
        active = sample < a.x.cols;
        if (active) active = !inactive(a, limits, sample);
        random = 0;
        for (uint f = 0; f < F; ++f) upstream[f] = 0;
        for (uint d = 0; d < D; ++d) { cell[d] = 0; fraction[d] = 0; }
        if (!active) return;
        // Metal 4.1 does not infer thread address space for these member
        // arrays when binding the coordinates helper's pointer arguments.
        uint local_cell[D];
        float local_fraction[D], slope[D];
        GridPolicy<D, F>::coordinates(a, input, sample, local_cell, local_fraction, slope);
        for (uint d = 0; d < D; ++d) {
            cell[d] = local_cell[d];
            fraction[d] = local_fraction[d];
        }
        for (uint f = 0; f < F; ++f)
            upstream[f] = float(incoming[at(a.dy, a.level * F + f, sample)]);
        if (a.stochastic && a.interpolation)
            random = random_sample(sample + a.level * a.x.cols);
    }

    static uint corners(HG_CONSTANT Args& a) {
        return a.interpolation == 0 || a.stochastic ? 1u : (1u << D);
    }

    uint corner(HG_CONSTANT Args& a, uint ordinal, HG_THREAD float& weight) const {
        weight = active ? 1.0f : 0.0f;
        if (!active) return 0xffffffffu;
        const bool nearest = a.interpolation == 0;
        const bool stochastic = a.stochastic && !nearest;
        uint point[D];
        for (uint d = 0; d < D; ++d) {
            const bool upper = stochastic ? random < fraction[d] : (!nearest && (ordinal & (1u << d)) != 0);
            point[d] = cell[d] + uint(upper);
            if (!nearest && !stochastic) weight *= upper ? fraction[d] : 1.0f - fraction[d];
        }
        return GridPolicy<D, F>::index(a, point) * F;
    }
};
}
#undef HG_CONSTANT
#undef HG_DEVICE
#undef HG_THREAD
#endif
