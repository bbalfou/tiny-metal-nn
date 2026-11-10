#ifndef TCNN_ENCODING_FIRST_ORDER
#define TCNN_ENCODING_FIRST_ORDER
#include "math.metal"
#include "spherical_harmonics_basis.metal"
#ifndef TCNN_ENCODING_FLOAT_ATOMICS
#define TCNN_ENCODING_FLOAT_ATOMICS 0
#endif

namespace tcnn_encoding {
using tcnn_encoding_math::Args;
using tcnn_encoding_math::Mat;
using tcnn_encoding_math::at;
using tcnn_encoding_math::inactive;
using tcnn_encoding_math::quartic_cdf;
using tcnn_encoding_math::quartic_pdf;
using tcnn_encoding_math::random_word;
using tcnn_encoding_math::random_sample;

#ifdef __METAL_VERSION__
#define EO_CONSTANT constant
#define EO_DEVICE device
#define EO_THREAD thread
#else
#define EO_CONSTANT
#define EO_DEVICE
#define EO_THREAD
using uint = uint32_t;
using ulong = uint64_t;
using std::floor;
using std::exp2;
using std::fma;
using std::sin;
using std::cos;
using std::fabs;
using std::ldexp;
#endif

// Exact normal power-of-two scaling from metal2's spectral policy. Keep the
// library fallback for subnormal/overflow exponents instead of narrowing the
// valid frequency configuration space to a fixed octave whitelist.
inline float scale_power_two(float value, int exponent) {
#ifdef __METAL_VERSION__
    if (exponent >= -126 && exponent <= 127)
        return value * as_type<float>(uint(exponent + 127) << 23);
#endif
    return ldexp(value, exponent);
}

inline float blob_boundary(float distance, uint bins) {
    return quartic_cdf(distance, float(bins)) + quartic_cdf(distance - 1.0f, float(bins)) + quartic_cdf(distance + 1.0f, float(bins));
}

inline float blob_derivative(float distance, uint bins) {
    return quartic_pdf(distance, float(bins)) + quartic_pdf(distance - 1.0f, float(bins)) + quartic_pdf(distance + 1.0f, float(bins));
}

template <uint Kind>
inline float spectral_value(float coordinate, uint feature) {
    if (Kind == 2) {
        // Metal1 disables fast math. Preserve its operation order exactly:
        // fma(coordinate*2^k, pi, phase) from metal2 changes phase rounding at
        // large k, so it is not an equivalent optimization for this runtime.
        const float frequency = 3.14159265358979323846f * scale_power_two(1.0f, int(feature / 2u));
        const float phase = coordinate * frequency + float(feature & 1u) * 1.57079632679489661923f;
        return sin(phase);
    }
    const float phase = scale_power_two(coordinate, int(feature) - 1) + float(feature) * 0.25f;
    return fabs(phase - floor(phase) - 0.5f) * 4.0f - 1.0f;
}

template <uint Kind>
inline float spectral_derivative(float coordinate, uint feature) {
    if (Kind == 2) {
        const int octave = int(feature / 2u);
        const float frequency = 3.14159265358979323846f * scale_power_two(1.0f, octave);
        const float phase = coordinate * frequency + float(feature & 1u) * 1.57079632679489661923f;
        return frequency * cos(phase);
    }
    const float phase = scale_power_two(coordinate, int(feature) - 1) + float(feature) * 0.25f;
    const float slope = int(floor(phase * 2.0f)) % 2 == 0 ? -1.0f : 1.0f;
    return scale_power_two(slope, int(feature) + 1);
}

template <uint Kind, uint Degree = 0>
inline float analytic_value(
    EO_CONSTANT Args& a,
    EO_DEVICE const float* input,
    uint row,
    uint sample
) {
    if (Kind == 5) {
        // Non-JIT CUDA1 and Metal1 put the padding BEFORE the SH basis.
        if (row < a.resolution)
            return 1.0f;
        return SphericalHarmonics::template evaluate<false>(
                   row - a.resolution,
                   input[at(a.x, 0, sample)] * 2.0f - 1.0f,
                   input[at(a.x, 1, sample)] * 2.0f - 1.0f,
                   input[at(a.x, 2, sample)] * 2.0f - 1.0f
        )
            .x;
    }
    if (row >= a.width || Kind == 0)
        return 1.0f;
    if (Kind == 1)
        return input[at(a.x, row, sample)] * a.scale + a.offset;
    const uint degree = Degree ? Degree : a.degree;
    const uint fan = degree * (Kind == 2 ? 2u : 1u);
    const uint dimension = row / fan, feature = row - dimension * fan;
    const float coordinate = input[at(a.x, dimension, sample)];
    if (Kind == 2 || Kind == 3)
        return spectral_value<Kind>(coordinate, feature);
    const float inverse_bins = 1.0f / float(degree);
    return blob_boundary(float(feature + 1u) * inverse_bins - coordinate, degree) -
           blob_boundary(float(feature) * inverse_bins - coordinate, degree);
}

// The same analytic derivatives without a fixed SIMD width. This capability
// specialization runs on non-Apple Metal devices and exceptionally large
// dispatches; it does not restore the superseded full-output scalar traversal.
template <typename T, uint Kind, uint Degree>
inline void analytic_input_gradient(
    EO_CONSTANT Args& a, EO_DEVICE const float* input, EO_DEVICE const T* upstream,
    EO_DEVICE float* gradient, uint index
) {
    if (index >= a.dims * a.x.cols) return;
    const uint dimension = index % a.dims, sample = index / a.dims;
    float sum = 0;
    if (Kind == 1) {
        sum = float(T(float(upstream[at(a.dy, dimension, sample)]) * a.scale));
    } else if (Kind == 5) {
        const float x = input[at(a.x, 0, sample)] * 2.0f - 1.0f;
        const float y = input[at(a.x, 1, sample)] * 2.0f - 1.0f;
        const float z = input[at(a.x, 2, sample)] * 2.0f - 1.0f;
        for (uint basis = 0; basis < a.width; ++basis) {
            const auto value = SphericalHarmonics::template evaluate<true>(basis, x, y, z);
            const float derivative = dimension == 0 ? value.y : dimension == 1 ? value.z : value.w;
            sum += float(upstream[at(a.dy, a.resolution + basis, sample)]) * derivative;
        }
        sum *= 2.0f;
    } else if (Kind != 0) {
        const float coordinate = input[at(a.x, dimension, sample)];
        const uint count = Degree * (Kind == 2 ? 2u : 1u);
        for (uint feature = 0; feature < count; ++feature) {
            const float derivative = Kind == 4
                ? blob_derivative(float(feature) / float(Degree) - coordinate, Degree) -
                  blob_derivative(float(feature + 1u) / float(Degree) - coordinate, Degree)
                : spectral_derivative<Kind>(coordinate, feature);
            sum += float(upstream[at(a.dy, dimension * count + feature, sample)]) * derivative;
        }
    }
    gradient[at(a.dx, dimension, sample)] = sum;
}

// Use the source policy's native float atomic where available. Contended CAS
// retries serialize low-resolution grid levels. Retain the older MSL fallback.
// Each buffer is zeroed in a preceding ordered runtime fill command.
inline void add_gradient(EO_DEVICE float* target, float value) {
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 410 && TCNN_ENCODING_FLOAT_ATOMICS
    atomic_fetch_add_explicit(reinterpret_cast<device atomic_float*>(target), value, memory_order_relaxed);
#elif defined(__METAL_VERSION__)
    device atomic_uint* bits = reinterpret_cast<device atomic_uint*>(target);
    uint expected = atomic_load_explicit(bits, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(
        bits,
        &expected,
        as_type<uint>(as_type<float>(expected) + value),
        memory_order_relaxed,
        memory_order_relaxed
    )) {
    }
#else
    // Test-only sequential emulation. There is no production CPU fallback.
    *target += value;
#endif
}

// One thread owns one sample at one level and reuses coordinates, corner
// addresses, and weights for all F features, matching metal2's Grid<D,F>.
// Metal1 supplies a level-local parameter pointer and scale, so no offsets
// tensor or foreign host ABI is necessary. D=1 extends the same algorithm.
template <uint D, uint F>
struct GridPolicy {
    static_assert(D >= 1 && D <= 24 && (F == 1 || F == 2 || F == 4 || F == 8), "unsupported lattice shape");
    static uint index(EO_CONSTANT Args& a, EO_THREAD const uint* point) {
        if (D == 2 && a.grid_type == 0 && a.hash_type == 1 && a.table_size &&
            (a.table_size & (a.table_size - 1)) == 0) {
            const uint address = a.resolution <= 65535u && a.resolution*a.resolution <= a.table_size
                ? point[0] + point[1]*a.resolution : point[0] ^ (point[1]*2654435761u);
            return address & (a.table_size-1);
        }
        uint stride = 1u, address = 0u;
        bool linear = true;
        for (uint d = 0; d < D; ++d) {
            if (a.resolution > 0xffffffffu / stride) {
                linear = false;
                break;
            }
            stride *= a.resolution;
        }
        if (linear) {
            uint step = 1u;
            for (uint d = 0; d < D; ++d) {
                address += point[d] * step;
                step *= a.resolution;
            }
        } else {
            stride = 0xffffffffu;
        }
        if (a.grid_type == 0u && a.table_size < stride) {
            const uint prime[4] = {1958374283u,
                                   2654435761u,
                                   805459861u,
                                   3674653429u};
            const uint reversed[4] = {2165219737u,
                                      1434869437u,
                                      2097192037u,
                                      3674653429u};
            uint hash = 0u;
            ulong rng_step = 0;
            for (uint d = 0; d < D; ++d) {
                if (a.hash_type == 3u)
                    rng_step ^= ulong(point[d]) << (d * (64u / D));
                else if (a.hash_type == 4u)
                    hash = (hash + point[d]) * 2531011u;
                else
                    hash ^= point[d] * (a.hash_type == 2u ? reversed[d] : (a.hash_type == 1u && d == 0u ? 1u : prime[d]));
            }
            address = a.hash_type == 3u ? random_word(rng_step) : hash;
        }
        return address % a.table_size;
    }

    static void coordinates(
        EO_CONSTANT Args& a,
        EO_DEVICE const float* input,
        uint sample,
        EO_THREAD uint* cell,
        EO_THREAD float* fraction,
        EO_THREAD float* slope
    ) {
        for (uint d = 0; d < D; ++d) {
            const float position = fma(a.scale, input[at(a.x, d, sample)], 0.5f);
            const float base = floor(position), t = position - base;
            cell[d] = uint(int(base));
            fraction[d] = a.interpolation == 2u ? t * t * (3.0f - 2.0f * t) : t;
            slope[d] = a.interpolation == 2u ? 6.0f * t * (1.0f - t) : 1.0f;
        }
    }

    template <typename T>
    static void forward(
        EO_CONSTANT Args& a,
        EO_DEVICE const float* input,
        EO_DEVICE const T* parameters,
        EO_DEVICE const float* limits,
        EO_DEVICE T* output,
        uint sample
    ) {
        if (sample >= a.x.cols)
            return;
        float result[F] = {};
        if (!inactive(a, limits, sample)) {
            uint cell[D];
            float fraction[D], slope[D];
            coordinates(
                a,
                input,
                sample,
                cell,
                fraction,
                slope
            );
            const bool nearest = a.interpolation == 0u;
            const uint corners = nearest ? 1u : (1u << D);
            for (uint corner = 0; corner < corners; ++corner) {
                uint point[D];
                float weight = 1.0f;
                for (uint d = 0; d < D; ++d) {
                    const bool upper = !nearest && (corner & (1u << d)) != 0u;
                    point[d] = cell[d] + uint(upper);
                    if (!nearest)
                        weight *= upper ? fraction[d] : 1.0f - fraction[d];
                }
                const uint address = index(a, point) * F;
                for (uint f = 0; f < F; ++f)
                    result[f] += weight * float(parameters[address + f]);
            }
        }
        // Preserve Metal1's FP32 interpolation accumulation then final T cast.
        // CUDA1/metal2 round each half FMA; this pre-existing numerical
        // difference is not broadened by the optimization.
        for (uint f = 0; f < F; ++f)
            output[at(a.y, a.level * F + f, sample)] = T(result[f]);
    }

    template <typename T>
    static void backward_input(
        EO_CONSTANT Args& a,
        EO_DEVICE const float* input,
        EO_DEVICE const T* parameters,
        EO_DEVICE const float* limits,
        EO_DEVICE const T* upstream,
        EO_DEVICE float* gradient,
        uint sample
    ) {
        if (sample >= a.x.cols)
            return;
        float result[D] = {};
        if (!inactive(a, limits, sample) && a.interpolation != 0u) {
            uint cell[D];
            float fraction[D], slope[D], dy[F];
            coordinates(
                a,
                input,
                sample,
                cell,
                fraction,
                slope
            );
            for (uint f = 0; f < F; ++f)
                dy[f] = float(upstream[at(a.dy, a.level * F + f, sample)]);
            // metal2's paired-face Jacobian algorithm, consumed immediately
            // rather than introducing a new saved-Jacobian context buffer.
            for (uint axis = 0; axis < D; ++axis) {
                if (axis >= a.max_input_grad_dims)
                    continue;
                for (uint corner = 0; corner < (1u << (D - 1u)); ++corner) {
                    uint point[D];
                    float weight = a.scale * slope[axis];
                    for (uint k = 0; k < D - 1u; ++k) {
                        const uint d = k >= axis ? k + 1u : k;
                        const bool upper = (corner & (1u << k)) != 0u;
                        point[d] = cell[d] + uint(upper);
                        weight *= upper ? fraction[d] : 1.0f - fraction[d];
                    }
                    point[axis] = cell[axis];
                    const uint left = index(a, point) * F;
                    ++point[axis];
                    const uint right = index(a, point) * F;
                    for (uint f = 0; f < F; ++f)
                        result[axis] += dy[f] * weight * (float(parameters[right + f]) - float(parameters[left + f]));
                }
            }
        }
        for (uint d = 0; d < D; ++d) {
            const uint address = at(a.dx, d, sample);
            gradient[address] = (a.accumulate ? gradient[address] : 0.0f) + result[d];
        }
    }

    template <typename T>
    static void backward_params(
        EO_CONSTANT Args& a,
        EO_DEVICE const float* input,
        EO_DEVICE const float* limits,
        EO_DEVICE const T* upstream,
        EO_DEVICE float* gradient,
        uint sample
    ) {
        // Unlike the older metal2 source's scatter threshold, the supplied
        // CUDA1 and Metal1 both disable level >= max_level*levels + 1e-3.
        if (sample >= a.x.cols || inactive(a, limits, sample))
            return;
        uint cell[D];
        float fraction[D], slope[D], dy[F];
        coordinates(
            a,
            input,
            sample,
            cell,
            fraction,
            slope
        );
        for (uint f = 0; f < F; ++f)
            dy[f] = float(upstream[at(a.dy, a.level * F + f, sample)]);
        const bool nearest = a.interpolation == 0u;
        const bool stochastic = a.stochastic != 0u && !nearest;
        const float random = stochastic ? random_sample(sample + a.level * a.x.cols) : 0.0f;
        const uint corners = nearest || stochastic ? 1u : (1u << D);
        for (uint corner = 0; corner < corners; ++corner) {
            uint point[D];
            float weight = 1.0f;
            for (uint d = 0; d < D; ++d) {
                const bool upper = stochastic ? random < fraction[d] : (!nearest && (corner & (1u << d)) != 0u);
                point[d] = cell[d] + uint(upper);
                if (!nearest && !stochastic)
                    weight *= upper ? fraction[d] : 1.0f - fraction[d];
            }
            const uint address = index(a, point) * F;
            for (uint f = 0; f < F; ++f)
                add_gradient(gradient + address + f, weight * dy[f]);
        }
    }
};

// Permuto has simplex vertices rather than Cartesian corners. Retain the
// authoritative simplex/ranking mathematics while sharing every geometry
// traversal across the level's features, and scatter gradients in parallel.
template <uint F>
struct PermutoPolicy {
    template <typename T>
    static void forward(EO_CONSTANT Args& a, EO_DEVICE const float* input, EO_DEVICE const T* parameters,
                        EO_DEVICE const float* limits, EO_DEVICE T* output, uint sample) {
        if (sample >= a.x.cols) return;
        float result[F] = {};
        if (!inactive(a, limits, sample)) {
            uint ids[25];
            float weight[25], derivative[25], second[25];
            const uint vertices = tcnn_encoding_math::geometry(a, input, sample, 0xffffffffu, 0xffffffffu,
                                                               ids, weight, derivative, second);
            for (uint vertex_index = 0; vertex_index < vertices; ++vertex_index)
                for (uint feature = 0; feature < F; ++feature)
                    result[feature] += weight[vertex_index] * float(parameters[ids[vertex_index] * F + feature]);
        }
        for (uint feature = 0; feature < F; ++feature)
            output[at(a.y, a.level * F + feature, sample)] = T(result[feature]);
    }

    template <typename T>
    static void backward_input(EO_CONSTANT Args& a, EO_DEVICE const float* input, EO_DEVICE const T* parameters,
                               EO_DEVICE const float* limits, EO_DEVICE const T* upstream,
                               EO_DEVICE float* gradient, uint sample) {
        if (sample >= a.x.cols) return;
        const bool active = !inactive(a, limits, sample);
        float dy[F];
        for (uint feature = 0; feature < F; ++feature)
            dy[feature] = active ? float(upstream[at(a.dy, a.level * F + feature, sample)]) : 0.0f;
        for (uint dimension = 0; dimension < a.dims; ++dimension) {
            float result = 0;
            if (active && dimension < a.max_input_grad_dims) {
                uint ids[25];
                float weight[25], derivative[25], second[25];
                const uint vertices = tcnn_encoding_math::geometry(a, input, sample, dimension, 0xffffffffu,
                                                                   ids, weight, derivative, second);
                for (uint feature = 0; feature < F; ++feature)
                    for (uint vertex_index = 0; vertex_index < vertices; ++vertex_index)
                        result += dy[feature] * derivative[vertex_index] * float(parameters[ids[vertex_index] * F + feature]);
            }
            const uint address = at(a.dx, dimension, sample);
            gradient[address] = (a.accumulate ? gradient[address] : 0.0f) + result;
        }
    }

    template <typename T>
    static void backward_params(EO_CONSTANT Args& a, EO_DEVICE const float* input, EO_DEVICE const float* limits,
                                EO_DEVICE const T* upstream, EO_DEVICE float* gradient, uint sample) {
        if (sample >= a.x.cols || inactive(a, limits, sample)) return;
        uint ids[25];
        float weight[25], derivative[25], second[25], dy[F];
        const uint vertices = tcnn_encoding_math::geometry(a, input, sample, 0xffffffffu, 0xffffffffu,
                                                           ids, weight, derivative, second);
        for (uint feature = 0; feature < F; ++feature)
            dy[feature] = float(upstream[at(a.dy, a.level * F + feature, sample)]);
        for (uint vertex_index = 0; vertex_index < vertices; ++vertex_index)
            for (uint feature = 0; feature < F; ++feature)
                add_gradient(gradient + ids[vertex_index] * F + feature, weight[vertex_index] * dy[feature]);
    }
};
} // namespace tcnn_encoding
#endif
