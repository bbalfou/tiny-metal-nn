// Adaptive grouped HashGrid gradients. A small GPU classifier chooses between
// the existing coherent SIMD reduction and the independent tiled scatter.
// Classification changes performance only: both routes compute the full result.
#include "hash_grid_gradient.metal"
#ifndef TCNN_ENCODING_DIMS
#define TCNN_ENCODING_DIMS 2
#endif
#ifndef TCNN_ENCODING_FEATURES
#define TCNN_ENCODING_FEATURES 2
#endif
#ifndef HG_SAMPLES
#define HG_SAMPLES 4
#endif
#ifndef TCNN_HASH_GRID_ADAPTIVE
#define TCNN_HASH_GRID_ADAPTIVE 0
#endif
#ifndef HG_PARAMETER_COUNT
#define HG_PARAMETER_COUNT 11184640
#endif
#ifndef HG_TILED_LANES
#define HG_TILED_LANES 8
#endif
using namespace tcnn_encoding;

template <class T, uint D, uint F>
inline void hash_grid_parameter_gradient(device const float* input, device const float* limits,
                                         device const T* incoming, device float* partials,
                                         constant HashGridGradientLevel* levels, uint2 position,
                                         uint lane) {
    constant auto& level = levels[position.y];
    constant auto& args = level.args;
#if TCNN_HASH_GRID_ADAPTIVE
    if (partials[HG_PARAMETER_COUNT + args.level] == 0.0f)
        return;
#endif
    HashGridGradientSample<D, F> sample;
    sample.prepare(args, input, limits, incoming, position.x);
    if (D == 3)
        sample.active &= sample.upstream[0] != 0.0f || sample.upstream[1] != 0.0f;
    // Check the cell once, rather than running a separate all-lanes test
    // for every corner. Inactive lanes cannot determine the shared cell.
    const bool stochastic = args.stochastic && args.interpolation != 0;
    bool same_coordinates = true;
    if (!stochastic) {
        for (uint d = 0; d < D; ++d) {
            const uint first = simd_min(sample.active ? sample.cell[d] : 0xffffffffu);
            same_coordinates &= !sample.active || sample.cell[d] == first;
        }
    }
    const bool same_cell = simd_all(same_coordinates);
    // Tail and disabled-level lanes must reach every collective. The host
    // rounds X to complete 128-thread groups; inactive lanes contribute zero.
    for (uint corner = 0; corner < sample.corners(args); ++corner) {
        float weight;
        const uint address = sample.corner(args, corner, weight);
        uint first = 0xffffffffu;
        bool same_destination = same_cell;
        if (same_cell) {
            first = simd_min(address);
            // Stochastic interpolation can choose different corners even in
            // one shared cell. Only combine genuinely equal destinations.
            if (stochastic)
                same_destination = simd_all(!sample.active || address == first);
        }
        if (same_destination) {
            for (uint feature = 0; feature < F; ++feature) {
                const float total = simd_sum(weight * sample.upstream[feature]);
                if (lane == 0 && first != 0xffffffffu && total != 0.0f)
                    add_gradient(partials + level.parameter_offset + first + feature, total);
            }
        } else if (sample.active) {
            for (uint feature = 0; feature < F; ++feature)
                add_gradient(partials + level.parameter_offset + address + feature,
                             weight * sample.upstream[feature]);
        }
    }
}

#define HASH_GRID_PARAMETER_ENTRY(T, S)                                                            \
    kernel void encoding_hash_grid_parameter_gradient_##S(                                         \
        device const float* input [[buffer(0)]], device const float* limits [[buffer(1)]],         \
        device const T* incoming [[buffer(2)]], device float* partials [[buffer(3)]],              \
        constant HashGridGradientLevel* levels [[buffer(4)]],                                      \
        uint2 position [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {     \
        hash_grid_parameter_gradient<T, TCNN_ENCODING_DIMS, TCNN_ENCODING_FEATURES>(               \
            input, limits, incoming, partials, levels, position, lane);                            \
    }
HASH_GRID_PARAMETER_ENTRY(float, f32)
HASH_GRID_PARAMETER_ENTRY(half, f16)
#undef HASH_GRID_PARAMETER_ENTRY

// This changes only algorithm selection. Both routes compute the same gradient,
// so an unrepresentative probe can affect performance, never correctness.
kernel void encoding_hash_grid_classify(device const float* input [[buffer(0)]],
                                        device const float* limits [[buffer(1)]],
                                        device const half* incoming [[buffer(2)]],
                                        device float* scratch [[buffer(3)]],
                                        constant HashGridGradientLevel* levels [[buffer(4)]],
                                        device uint* counts [[buffer(5)]],
                                        constant uint2& tuning [[buffer(6)]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    bool same = true;
    constant auto& a = levels[min(lane, 15u)].args;
    if (lane < 16) {
        uint first[TCNN_ENCODING_DIMS];
        bool seen = false;
        for (uint p = 0; p < 32; ++p) {
            uint sample = uint(ulong(p) * (a.x.cols - 1) / 31);
            // Framework padding can have arbitrary coordinates but contributes
            // no gradient. It must not defeat coherent SIMD aggregation.
            if (inactive(a, limits, sample) ||
                (incoming[at(a.dy, a.level * 2, sample)] == half(0) &&
                 incoming[at(a.dy, a.level * 2 + 1, sample)] == half(0)))
                continue;
            for (uint d = 0; d < TCNN_ENCODING_DIMS; ++d) {
                const uint cell = uint(int(floor(fma(a.scale, input[at(a.x, d, sample)], 0.5f))));
                if (!seen) first[d] = cell;
                else same &= cell == first[d];
            }
            seen = true;
        }
        scratch[HG_PARAMETER_COUNT + lane] = same ? 1.0f : 0.0f;
    }
    bool any = simd_any(lane < 16 && same), all = simd_all(lane >= 16 || same);
    if (lane == 0) {
        uint padded = (a.x.cols + 127) / 128 * 128;
        counts[0] = all ? 0 : (padded / tuning.y * HG_TILED_LANES + tuning.x - 1) / tuning.x;
        counts[1] = 16;
        counts[2] = 1;
        counts[3] = any ? (padded + tuning.x - 1) / tuning.x : 0;
        counts[4] = 16;
        counts[5] = 1;
    }
}

kernel void encoding_hash_grid_scatter_tiled_f16(device const float* input [[buffer(0)]],
                                                 device const float* limits [[buffer(1)]],
                                                 device const half* incoming [[buffer(2)]],
                                                 device float* partials [[buffer(3)]],
                                                 constant HashGridGradientLevel* levels
                                                 [[buffer(4)]],
                                                 uint2 position [[thread_position_in_grid]]) {
    constant auto& level = levels[position.y];
    constant auto& args = level.args;
    if (partials[HG_PARAMETER_COUNT + args.level] != 0.0f)
        return;
    {
#if TCNN_ENCODING_DIMS == 3
        // Two lanes own the Z faces of each sample. Each lane reuses one
        // coordinate transform across four XY corners and both features.

        for (uint i = 0; i < HG_SAMPLES; ++i) {
            const uint sample = (position.x / 2) * HG_SAMPLES + i;
            if (sample >= args.x.cols || inactive(args, limits, sample)) continue;
            const float grad0 = float(incoming[at(args.dy, args.level * 2, sample)]);
            const float grad1 = float(incoming[at(args.dy, args.level * 2 + 1, sample)]);
            if (grad0 == 0.0f && grad1 == 0.0f) continue;
            uint cell[3]; float fraction[3];
            for (uint d = 0; d < 3; ++d) {
                const float p = fma(args.scale, input[at(args.x, d, sample)], 0.5f);
                cell[d] = uint(int(floor(p)));
                fraction[d] = p - floor(p);
            }
            #pragma unroll
            for (uint xy = 0; xy < 4; ++xy) {
                const uint corner = xy + (position.x % 2) * 4;
                const uint px = cell[0] + (corner & 1), py = cell[1] + ((corner >> 1) & 1), pz = cell[2] + (corner >> 2);
                const uint index = args.level < 3 ? px + py * args.resolution + pz * args.resolution * args.resolution
                    : px ^ (py * 2654435761u) ^ (pz * 805459861u);
                const uint addr = (index & (args.table_size - 1)) * 2 + level.parameter_offset;
                const float w = ((corner & 1) ? fraction[0] : 1 - fraction[0]) *
                                ((corner & 2) ? fraction[1] : 1 - fraction[1]) *
                                ((corner & 4) ? fraction[2] : 1 - fraction[2]);
                if (grad0 != 0.0f) add_gradient(partials + addr, w * grad0);
                if (grad1 != 0.0f) add_gradient(partials + addr + 1, w * grad1);
            }
        }
#else
        for (uint i = 0; i < HG_SAMPLES; ++i) {
            const uint sample = (position.x / ((1u << TCNN_ENCODING_DIMS) * 2u)) * HG_SAMPLES + i;
            if (sample >= args.x.cols || inactive(args, limits, sample))
                continue;
            const uint c = (position.x / 2) % (1u << TCNN_ENCODING_DIMS), f = position.x % 2;
            const float grad = float(incoming[at(args.dy, args.level * 2 + f, sample)]);
            // Exact zero suppression: retain FP32 accumulation for every nonzero
            // incoming FP16 gradient. No thresholding or precision reduction.
            if (grad == 0.0f)
                continue;
            const float x = fma(args.scale, input[at(args.x, 0, sample)], 0.5f);
            const float y = fma(args.scale, input[at(args.x, 1, sample)], 0.5f);
            const float tx = x - floor(x), ty = y - floor(y);
            const uint px = uint(int(floor(x))) + (c & 1), py = uint(int(floor(y))) + (c >> 1);
            uint addr = args.level < 6 ? px + py * args.resolution : px ^ (py * 2654435761u);
            addr = (addr & (args.table_size - 1)) * 2 + level.parameter_offset;
            const float w = ((c & 1) ? tx : 1 - tx) * ((c & 2) ? ty : 1 - ty);
            add_gradient(partials + addr + f, w * grad);
        }
#endif
        return;
    }
}
