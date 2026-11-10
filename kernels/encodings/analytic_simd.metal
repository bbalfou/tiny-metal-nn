/* Metal2 native encoding traversal adapted to Metal1's unchanged ABI.
 * Sources: metal2_encodings/{policies,spectral_policies,
 * spectral_sh}.metalh. See first_order.metal for grid provenance.
 * SIMD collectives are reached by all lanes, including padded guard lanes. */
#ifndef TCNN_ENCODING_ANALYTIC_SIMD
#define TCNN_ENCODING_ANALYTIC_SIMD
#include "first_order.metal"

namespace tcnn_encoding {

template <typename T, uint Kind, uint Degree>
inline void analytic_forward(
    constant Args& a,
    device const float* input,
    device T* output,
    uint2 position,
    uint lane,
    bool SoA,
    uint Rows
) {
    if (Kind == 5) {
        const uint sample = position.x;
        if (sample >= a.x.cols) return;
        const float x = input[at(a.x, 0, sample)] * 2.0f - 1.0f;
        const float y = input[at(a.x, 1, sample)] * 2.0f - 1.0f;
        const float z = input[at(a.x, 2, sample)] * 2.0f - 1.0f;
        for (uint row = 0; row < a.resolution; ++row)
            output[at(a.y, row, sample)] = T(1);
        #pragma unroll
        for (uint basis = 0; basis < Degree * Degree; ++basis)
            output[at(a.y, a.resolution + basis, sample)] =
                T(SphericalHarmonics::template evaluate<false>(basis, x, y, z).x);
        return;
    }
    if (SoA && Kind >= 2 && Kind <= 4) {
        const uint sample = position.x, first = position.y * 4;
        if (sample >= a.x.cols) return;
        // Power-of-two bins >=4 ensure a tile never straddles coordinates.
        if (Kind == 4 && Degree >= 4 && first < a.width) {
            const uint bin = first % Degree, dimension = first / Degree;
            const float coordinate = input[at(a.x, dimension, sample)];
            float left = blob_boundary(float(bin) / float(Degree) - coordinate, Degree);
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                const float right = blob_boundary(float(bin + i + 1) / float(Degree) - coordinate, Degree);
                output[at(a.y, first + i, sample)] = T(right - left);
                left = right;
            }
        } else {
            #pragma unroll
            for (uint i = 0; i < 4; ++i)
                if (first + i < Rows)
                    output[at(a.y, first + i, sample)] = T(analytic_value<Kind, Degree>(a, input, first + i, sample));
        }
        return;
    }
    // Amortize coordinate loads across a larger contiguous OneBlob tile.
    if (Kind == 4 && Degree >= 32 && !SoA && a.y.rs == 1 &&
        a.y.cs % 4 == 0 && Rows % 32 == 0 && a.width % 32 == 0) {
        const uint sample = position.x / (Rows / 32);
        const uint feature = (position.x % (Rows / 32)) * 32;
        if (sample >= a.x.cols) return;
        const bool active = feature < a.width;
        const uint bin = feature % Degree, dimension = feature / Degree;
        const float coordinate = active ? input[at(a.x, dimension, sample)] : 0.0f;
        float left = blob_boundary(float(bin) / float(Degree) - coordinate, Degree);
        #pragma unroll
        for (uint pack = 0; pack < 8; ++pack) {
            vec<T, 4> result(T(1));
            if (active) {
                #pragma unroll
                for (uint i = 0; i < 4; ++i) {
                    const float right = blob_boundary(float(bin + pack * 4 + i + 1) / float(Degree) - coordinate, Degree);
                    result[i] = T(right - left);
                    left = right;
                }
            }
            *reinterpret_cast<device packed_vec<T, 4>*>(output + at(a.y, feature + pack * 4, sample)) = result;
        }
        return;
    }
    // Four adjacent bins per thread reduce thread count and scalar store overhead.
    // Boundary subtraction and periodic shifts retain the scalar arithmetic.
    if (Kind >= 2 && Kind <= 4 && (Kind != 4 || Degree >= 4) && !SoA && a.y.rs == 1 &&
        a.y.cs % 4 == 0 && Rows % 4 == 0 && a.width % 4 == 0) {
        const uint sample = position.x / (Rows / 4);
        const uint feature = (position.x % (Rows / 4)) * 4;
        if (sample >= a.x.cols) return;
        vec<T, 4> result(T(1));
        if (Kind != 4) {
            #pragma unroll
            for (uint i = 0; i < 4; ++i)
                result[i] = T(analytic_value<Kind, Degree>(a, input, feature + i, sample));
        } else if (feature < a.width) {
            const uint bin = feature % Degree, dimension = feature / Degree;
            const float coordinate = input[at(a.x, dimension, sample)];
            float left = blob_boundary(float(bin) / float(Degree) - coordinate, Degree);
            for (uint i = 0; i < 4; ++i) {
                const float right = blob_boundary(float(bin + i + 1) / float(Degree) - coordinate, Degree);
                result[i] = T(right - left);
                left = right;
            }
        }
        *reinterpret_cast<device packed_vec<T, 4>*>(output + at(a.y, feature, sample)) = result;
        return;
    }
    // Flatten AoS across samples: a group per narrow row wastes most lanes.
    // SoA already has a long contiguous sample axis and needs no division.
    const uint sample = SoA ? position.x : position.x / Rows;
    const uint feature = SoA ? position.y : position.x % Rows;
    const bool valid = sample < a.x.cols && feature < a.y.rows;
    if (Kind == 4 && !SoA) {
        const uint bin = feature & (Degree - 1u);
        const uint dimension = feature >> (31u - clz(Degree));
        const bool active = valid && feature < a.width;
        const float coordinate = active ? input[at(a.x, dimension, sample)] : 0.0f;
        const float inverse_bins = 1.0f / float(Degree);
        const float left = blob_boundary(float(bin) * inverse_bins - coordinate, Degree);
        float right = simd_shuffle_down(left, 1);
        // The source's last-bin +1 shortcut assumes coordinates in [0,1].
        // CUDA1 does not impose that bound; evaluate that boundary explicitly.
        if (bin + 1u == Degree || lane == 31u)
            right = blob_boundary(float(bin + 1u) * inverse_bins - coordinate, Degree);
        if (valid)
            output[at(a.y, feature, sample)] = active ? T(right - left) : T(1);
    } else if (valid) {
        output[at(a.y, feature, sample)] =
            T(analytic_value<Kind, Degree>(
                a,
                input,
                feature,
                sample
            ));
    }
}

template <typename T, uint Kind, uint Lanes, uint Degree>
inline void analytic_backward(
    constant Args& a,
    device const float* input,
    device const T* upstream,
    device float* gradient,
    uint index,
    uint lane
) {
    static_assert(Lanes == 8u || Lanes == 16u, "Encoding reduction policy must divide a 32-lane SIMD group.");
    if (Kind == 0 || Kind == 1) {
        if (index >= a.dims * a.x.cols)
            return;
        const uint dimension = index % a.dims, sample = index / a.dims;
        gradient[at(a.dx, dimension, sample)] = Kind == 0 ? 0.0f : float(T(float(upstream[at(a.dy, dimension, sample)]) * a.scale));
    } else if (Kind == 5) {
        const uint local = lane & (Lanes - 1u), sample = index / Lanes;
        const bool active = sample < a.x.cols;
        float3 point(0.0f), sum(0.0f);
        if (active)
            point = float3(input[at(a.x, 0, sample)], input[at(a.x, 1, sample)], input[at(a.x, 2, sample)]) * 2.0f - 1.0f;
        for (uint basis = local; basis < a.width; basis += Lanes) {
            const float4 result = SphericalHarmonics::template evaluate<true>(
                basis,
                point.x,
                point.y,
                point.z
            );
            if (active)
                sum += float(upstream[at(a.dy, a.resolution + basis, sample)]) * float3(result.y, result.z, result.w);
        }
        for (uint delta = Lanes / 2u; delta; delta /= 2u)
            sum += float3(simd_shuffle_down(sum.x, delta), simd_shuffle_down(sum.y, delta), simd_shuffle_down(sum.z, delta));
        if (active && local == 0u) {
            gradient[at(a.dx, 0, sample)] = sum.x * 2.0f;
            gradient[at(a.dx, 1, sample)] = sum.y * 2.0f;
            gradient[at(a.dx, 2, sample)] = sum.z * 2.0f;
        }
    } else if (Kind == 4) {
        const uint lanes = min(Degree, 32u), local = lane & (lanes - 1u);
        const uint coordinate_index = index / lanes;
        const bool active = coordinate_index < a.dims * a.x.cols;
        const uint sample = active ? coordinate_index / a.dims : 0u;
        const uint dimension = active ? coordinate_index % a.dims : 0u;
        const float coordinate = active ? input[at(a.x, dimension, sample)] : 0.0f;
        const float inverse_bins = 1.0f / float(Degree);
        float sum = 0.0f;
        for (uint bin = local; bin < Degree; bin += lanes) {
            const float left = blob_derivative(float(bin) * inverse_bins - coordinate, Degree);
            float right = simd_shuffle_down(left, 1);
            if (local + 1u == lanes)
                right = blob_derivative(float(bin + 1u) * inverse_bins - coordinate, Degree);
            if (active)
                sum += float(upstream[at(a.dy, dimension * Degree + bin, sample)]) * (left - right);
        }
        for (uint delta = lanes / 2u; delta; delta /= 2u)
            sum += simd_shuffle_down(sum, delta);
        if (active && local == 0u)
            gradient[at(a.dx, dimension, sample)] = sum;
    } else {
        const uint local = lane & (Lanes - 1u), coordinate_index = index / Lanes;
        const bool active = coordinate_index < a.dims * a.x.cols;
        const uint sample = active ? coordinate_index / a.dims : 0u;
        const uint dimension = active ? coordinate_index % a.dims : 0u;
        const float coordinate = active ? input[at(a.x, dimension, sample)] : 0.0f;
        const uint count = Degree * (Kind == 2 ? 2u : 1u);
        float sum = 0.0f;
        for (uint feature = local; feature < count; feature += Lanes)
            if (active)
                sum += float(upstream[at(a.dy, dimension * count + feature, sample)]) * spectral_derivative<Kind>(coordinate, feature);
        for (uint delta = Lanes / 2u; delta; delta /= 2u)
            sum += simd_shuffle_down(sum, delta);
        if (active && local == 0u)
            gradient[at(a.dx, dimension, sample)] = sum;
    }
}
} // namespace tcnn_encoding
#endif
