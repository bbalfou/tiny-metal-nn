#ifndef TCNN_JIT_ENCODING_MATH
#define TCNN_JIT_ENCODING_MATH
namespace tcnn_jit_encoding {
// A value, one coordinate derivative, a directional derivative, and their mixed
// derivative. Polynomial operations provide exact SH Hessian-vector products.
struct Dual {
    float v, d, t, dt;

    Dual(
        float value = 0,
        float derivative = 0,
        float tangent = 0,
        float mixed = 0
    )
        : v(value), d(derivative), t(tangent), dt(mixed) {}
};

inline Dual operator+(Dual a, Dual b) {
    return {a.v + b.v,
            a.d + b.d,
            a.t + b.t,
            a.dt + b.dt};
}

inline Dual operator-(Dual a, Dual b) {
    return {a.v - b.v,
            a.d - b.d,
            a.t - b.t,
            a.dt - b.dt};
}

inline Dual operator-(Dual a) {
    return {-a.v,
            -a.d,
            -a.t,
            -a.dt};
}

inline Dual operator*(Dual a, Dual b) {
    return {a.v * b.v,
            a.d * b.v + a.v * b.d,
            a.t * b.v + a.v * b.t,
            a.dt * b.v + a.d * b.t + a.t * b.d + a.v * b.dt};
}

/// Evaluates one SH polynomial with exact first and mixed directional derivatives.
inline Dual sh_component(
    uint row,
    Dual x,
    Dual y,
    Dual z
) {
    Dual xy = x * y, xz = x * z, yz = y * z, x2 = x * x, y2 = y * y, z2 = z * z;
    Dual x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;
    Dual x6 = x4 * x2, y6 = y4 * y2, z6 = z4 * z2;
    switch (row) {
    case 0:
        return 0.28209479177387814f;
    case 1:
        return -0.48860251190291987f * y;
    case 2:
        return 0.48860251190291987f * z;
    case 3:
        return -0.48860251190291987f * x;
    case 4:
        return 1.0925484305920792f * xy;
    case 5:
        return -1.0925484305920792f * yz;
    case 6:
        return 0.94617469575755997f * z2 - 0.31539156525251999f;
    case 7:
        return -1.0925484305920792f * xz;
    case 8:
        return 0.54627421529603959f * x2 - 0.54627421529603959f * y2;
    case 9:
        return 0.59004358992664352f * y * (-3.0f * x2 + y2);
    case 10:
        return 2.8906114426405538f * xy * z;
    case 11:
        return 0.45704579946446572f * y * (1.0f - 5.0f * z2);
    case 12:
        return 0.3731763325901154f * z * (5.0f * z2 - 3.0f);
    case 13:
        return 0.45704579946446572f * x * (1.0f - 5.0f * z2);
    case 14:
        return 1.4453057213202769f * z * (x2 - y2);
    case 15:
        return 0.59004358992664352f * x * (-x2 + 3.0f * y2);
    case 16:
        return 2.5033429417967046f * xy * (x2 - y2);
    case 17:
        return 1.7701307697799304f * yz * (-3.0f * x2 + y2);
    case 18:
        return 0.94617469575756008f * xy * (7.0f * z2 - 1.0f);
    case 19:
        return 0.66904654355728921f * yz * (3.0f - 7.0f * z2);
    case 20:
        return -3.1735664074561294f * z2 + 3.7024941420321507f * z4 + 0.31735664074561293f;
    case 21:
        return 0.66904654355728921f * xz * (3.0f - 7.0f * z2);
    case 22:
        return 0.47308734787878004f * (x2 - y2) * (7.0f * z2 - 1.0f);
    case 23:
        return 1.7701307697799304f * xz * (-x2 + 3.0f * y2);
    case 24:
        return -3.7550144126950569f * x2 * y2 + 0.62583573544917614f * x4 + 0.62583573544917614f * y4;
    case 25:
        return 0.65638205684017015f * y * (10.0f * x2 * y2 - 5.0f * x4 - y4);
    case 26:
        return 8.3026492595241645f * xy * z * (x2 - y2);
    case 27:
        return -0.48923829943525038f * y * (3.0f * x2 - y2) * (9.0f * z2 - 1.0f);
    case 28:
        return 4.7935367849733241f * xy * z * (3.0f * z2 - 1.0f);
    case 29:
        return 0.45294665119569694f * y * (14.0f * z2 - 21.0f * z4 - 1.0f);
    case 30:
        return 0.1169503224534236f * z * (-70.0f * z2 + 63.0f * z4 + 15.0f);
    case 31:
        return 0.45294665119569694f * x * (14.0f * z2 - 21.0f * z4 - 1.0f);
    case 32:
        return 2.3967683924866621f * z * (x2 - y2) * (3.0f * z2 - 1.0f);
    case 33:
        return -0.48923829943525038f * x * (x2 - 3.0f * y2) * (9.0f * z2 - 1.0f);
    case 34:
        return 2.0756623148810411f * z * (-6.0f * x2 * y2 + x4 + y4);
    case 35:
        return 0.65638205684017015f * x * (10.0f * x2 * y2 - x4 - 5.0f * y4);
    case 36:
        return 1.3663682103838286f * xy * (-10.0f * x2 * y2 + 3.0f * x4 + 3.0f * y4);
    case 37:
        return 2.3666191622317521f * yz * (10.0f * x2 * y2 - 5.0f * x4 - y4);
    case 38:
        return 2.0182596029148963f * xy * (x2 - y2) * (11.0f * z2 - 1.0f);
    case 39:
        return -0.92120525951492349f * yz * (3.0f * x2 - y2) * (11.0f * z2 - 3.0f);
    case 40:
        return 0.92120525951492349f * xy * (-18.0f * z2 + 33.0f * z4 + 1.0f);
    case 41:
        return 0.58262136251873131f * yz * (30.0f * z2 - 33.0f * z4 - 5.0f);
    case 42:
        return 6.6747662381009842f * z2 - 20.024298714302954f * z4 + 14.684485723822165f * z6 - 0.31784601133814211f;
    case 43:
        return 0.58262136251873131f * xz * (30.0f * z2 - 33.0f * z4 - 5.0f);
    case 44:
        return 0.46060262975746175f * (x2 - y2) * (11.0f * z2 * (3.0f * z2 - 1.0f) - 7.0f * z2 + 1.0f);
    case 45:
        return -0.92120525951492349f * xz * (x2 - 3.0f * y2) * (11.0f * z2 - 3.0f);
    case 46:
        return 0.50456490072872406f * (11.0f * z2 - 1.0f) * (-6.0f * x2 * y2 + x4 + y4);
    case 47:
        return 2.3666191622317521f * xz * (10.0f * x2 * y2 - x4 - 5.0f * y4);
    case 48:
        return 10.247761577878714f * x2 * y4 - 10.247761577878714f * x4 * y2 + 0.6831841051919143f * x6 - 0.6831841051919143f * y6;
    case 49:
        return 0.70716273252459627f * y * (-21.0f * x2 * y4 + 35.0f * x4 * y2 - 7.0f * x6 + y6);
    case 50:
        return 5.2919213236038001f * xy * z * (-10.0f * x2 * y2 + 3.0f * x4 + 3.0f * y4);
    case 51:
        return -0.51891557872026028f * y * (13.0f * z2 - 1.0f) * (-10.0f * x2 * y2 + 5.0f * x4 + y4);
    case 52:
        return 4.1513246297620823f * xy * z * (x2 - y2) * (13.0f * z2 - 3.0f);
    case 53:
        return -0.15645893386229404f * y * (3.0f * x2 - y2) * (13.0f * z2 * (11.0f * z2 - 3.0f) - 27.0f * z2 + 3.0f);
    case 54:
        return 0.44253269244498261f * xy * z * (-110.0f * z2 + 143.0f * z4 + 15.0f);
    case 55:
        return 0.090331607582517306f * y * (-135.0f * z2 + 495.0f * z4 - 429.0f * z6 + 5.0f);
    case 56:
        return 0.068284276912004949f * z * (315.0f * z2 - 693.0f * z4 + 429.0f * z6 - 35.0f);
    case 57:
        return 0.090331607582517306f * x * (-135.0f * z2 + 495.0f * z4 - 429.0f * z6 + 5.0f);
    case 58:
        return 0.07375544874083044f * z * (x2 - y2) * (143.0f * z2 * (3.0f * z2 - 1.0f) - 187.0f * z2 + 45.0f);
    case 59:
        return -0.15645893386229404f * x * (x2 - 3.0f * y2) * (13.0f * z2 * (11.0f * z2 - 3.0f) - 27.0f * z2 + 3.0f);
    case 60:
        return 1.0378311574405206f * z * (13.0f * z2 - 3.0f) * (-6.0f * x2 * y2 + x4 + y4);
    case 61:
        return -0.51891557872026028f * x * (13.0f * z2 - 1.0f) * (-10.0f * x2 * y2 + x4 + 5.0f * y4);
    case 62:
        return 2.6459606618019f * z * (15.0f * x2 * y4 - 15.0f * x4 * y2 + x6 - y6);
    case 63:
        return 0.70716273252459627f * x * (-35.0f * x2 * y4 + 21.0f * x4 * y2 - x6 + 7.0f * y6);
    default:
        return {0, 0};
    }
}

inline float quartic_cdf(float x, float bins) {
    const float u = x * bins, u2 = u * u, u4 = u2 * u2;
    return fmax(0.0f, fmin(1.0f, (15.0f / 16.0f) * u * (1 - (2.0f / 3.0f) * u2 + (1.0f / 5.0f) * u4) + .5f));
}

inline float quartic_pdf(float x, float bins) {
    const float q = fmax(1 - x * x * bins * bins, 0.0f);
    return (15.0f / 16.0f) * q * q * bins;
}

/// Derivative of the compact quartic density, zero outside its smooth support.
inline float quartic_pdf_derivative(float x, float bins) {
    const float q = fmax(1 - x * x * bins * bins, 0.0f);
    return -(15.0f / 4.0f) * x * bins * bins * bins * q;
}

// Consecutive OneBlob bins share one boundary. A derivative-only invocation
// does not evaluate the CDF; the second-order variant adds just its density
// derivative. Keep periodic shifts identical to the native AOT contract.
template <uint Order>
inline Dual blob_boundary(float x, uint bins) {
    Dual result;
    for (int shift = -1; shift <= 1; ++shift) {
        const float q = x + float(shift);
        if (Order == 0u) result.v += quartic_cdf(q, float(bins));
        if (Order >= 1u) result.d += quartic_pdf(q, float(bins));
        if (Order >= 2u) result.dt += quartic_pdf_derivative(q, float(bins));
    }
    return result;
}

/// Reproduce PCG's 64-bit jump-ahead for both optional Rng hashing and stochastic gradients.
inline uint random_word(ulong delta) {
    const ulong mul = 0x5851f42d4c957f2dUL, inc = 3;
    ulong state = (inc + 1337) * mul + inc;
    ulong cur_mult = mul, cur_plus = inc, acc_mult = 1, acc_plus = 0;
    while (delta) {
        if (delta & 1UL) {
            acc_mult *= cur_mult;
            acc_plus = acc_plus * cur_mult + cur_plus;
        }
        cur_plus = (cur_mult + 1) * cur_plus;
        cur_mult *= cur_mult;
        delta >>= 1;
    }
    state = acc_mult * state + acc_plus;
    const uint z = uint(((state >> 18) ^ state) >> 27), rot = uint(state >> 59);
    return (z >> rot) | (z << ((~rot + 1u) & 31u));
}

// All feature math receives thread-local vectors so generated encoding and
// network functions can exchange values without intermediate device buffers.
template <uint D>
inline Dual analytic(uint kind, uint degree, uint width, uint padded, float scale, float offset,
                     const thread tcnn_jit_array<float, D>& input, uint row, uint dim,
                     const thread tcnn_jit_array<float, D>* direction = nullptr) {
    if (kind == 5u) {
        // Preserve the existing native/AOT layout: SH alignment precedes the basis.
        const uint prefix = padded - width;
        if (row < prefix) return Dual{1.0f};
        row -= prefix;
    }
    if (row >= width || kind == 0u) return Dual{1.0f};
    if (kind == 1u)
        return {input[row] * scale + offset, row == dim ? scale : 0.0f,
                direction ? (*direction)[row] * scale : 0.0f, 0.0f};
    if (kind == 5u)
        return sh_component(row,
            {input[0] * 2.0f - 1.0f, dim == 0u ? 2.0f : 0.0f,
             direction ? (*direction)[0] * 2.0f : 0.0f, 0.0f},
            {input[1] * 2.0f - 1.0f, dim == 1u ? 2.0f : 0.0f,
             direction ? (*direction)[1] * 2.0f : 0.0f, 0.0f},
            {input[2] * 2.0f - 1.0f, dim == 2u ? 2.0f : 0.0f,
             direction ? (*direction)[2] * 2.0f : 0.0f, 0.0f});
    const uint fan = degree * (kind == 2u ? 2u : 1u);
    const uint d = row / fan, k = row % fan;
    const float p = input[d], v = direction ? (*direction)[d] : 0.0f;
    if (kind == 2u) {
        const float frequency = 3.14159265358979323846f * exp2(float(k / 2u));
        const float angle = p * frequency + float(k % 2u) * 1.57079632679489661923f;
        const float value = sin(angle), derivative = frequency * cos(angle);
        return {value, d == dim ? derivative : 0.0f, derivative * v,
                d == dim ? -frequency * frequency * value * v : 0.0f};
    }
    if (kind == 3u) {
        const float phase = p * exp2(float(int(k) - 1)) + float(k) * 0.25f;
        const float derivative = exp2(float(k + 1u)) * (int(floor(phase * 2.0f)) % 2 == 0 ? -1.0f : 1.0f);
        return {fabs(phase - floor(phase) - 0.5f) * 4.0f - 1.0f,
                d == dim ? derivative : 0.0f, derivative * v, 0.0f};
    }
    float value = 0.0f, derivative = 0.0f, second = 0.0f;
    for (int shift = -1; shift <= 1; ++shift) {
        const float left = float(k) / float(degree) - p + float(shift);
        const float right = float(k + 1u) / float(degree) - p + float(shift);
        value += quartic_cdf(right, float(degree)) - quartic_cdf(left, float(degree));
        derivative += quartic_pdf(left, float(degree)) - quartic_pdf(right, float(degree));
        second += quartic_pdf_derivative(right, float(degree)) - quartic_pdf_derivative(left, float(degree));
    }
    return {value, d == dim ? derivative : 0.0f, derivative * v, d == dim ? second * v : 0.0f};
}

// Feature packets use scalar-aligned packed vectors. Composite parameter
// slices may begin on odd half offsets: an ordinary half2/half4 cast would
// impose an alignment contract the public composition API does not provide.
#ifdef __METAL_VERSION__
template <typename T> struct FeaturePacket;
template <> struct FeaturePacket<half> { using Type = packed_half2; };
template <> struct FeaturePacket<float> { using Type = packed_float2; };
#endif

template <uint F, typename T>
inline tcnn_jit_array<float, F> load_features(const device T* source) {
    tcnn_jit_array<float, F> value{};
#ifdef __METAL_VERSION__
    using Packet = typename FeaturePacket<T>::Type;
    for (uint f = 0; f + 1u < F; f += 2u) {
        const Packet pair = *reinterpret_cast<const device Packet*>(source + f);
        value[f] = float(pair[0]);
        value[f + 1u] = float(pair[1]);
    }
    if (F & 1u) value[F - 1u] = float(source[F - 1u]);
#else
    for (uint f = 0; f < F; ++f) value[f] = float(source[f]);
#endif
    return value;
}

// Adjacent feature contributions share a packed-half atomic transaction. The
// common helper handles canonical allocation origins and odd composite slices;
// float destinations retain their independent float atomic operations.
template <uint F, typename T>
inline void scatter_features(device T* destination, uint index, float weight,
                             const thread tcnn_jit_array<T, F>& upstream,
                             const thread tcnn_jit_environment* env) {
    for (uint f = 0; f + 1u < F; f += 2u)
        tcnn_jit_atomic_add_pair(destination, index + f,
            weight * float(upstream[f]), weight * float(upstream[f + 1u]), env);
    if (F & 1u) tcnn_jit_atomic_add(destination, index + F - 1u, weight * float(upstream[F - 1u]), env);
}

// Grid configuration is retained as template axes. The compiler specializes
// addressing, interpolation, feature width, coordinate width and position mode.
template <uint D, uint F, uint H, uint G, uint I, bool Fixed>
struct Grid {
    __attribute__((always_inline)) static uint index(const thread uint* point, uint resolution, uint table_size) {
        uint stride = 1u, address = 0u;
        bool linear = true;
        for (uint d = 0; d < D; ++d) {
            if (resolution > 0xffffffffu / stride) { linear = false; break; }
            stride *= resolution;
        }
        if (linear) {
            uint step = 1u;
            for (uint d = 0; d < D; ++d) { address += point[d] * step; step *= resolution; }
        } else stride = 0xffffffffu;
        if (G == 0u && table_size < stride) {
            const uint prime[4] = {1958374283u, 2654435761u, 805459861u, 3674653429u};
            const uint reversed[4] = {2165219737u, 1434869437u, 2097192037u, 3674653429u};
            uint hash = 0u;
            ulong step = 0ul;
            for (uint d = 0; d < D; ++d) {
                if (H == 3u) step ^= ulong(point[d]) << (d * (64u / D));
                else if (H == 4u) hash = (hash + point[d]) * 2531011u;
                else hash ^= point[d] * (H == 2u ? reversed[d] : (H == 1u && d == 0u ? 1u : prime[d]));
            }
            address = H == 3u ? random_word(step) : hash;
        }
        return address % table_size;
    }

    // Resolve the address policy once per sample/level. Every corner reuses
    // the lower/upper coordinate contributions instead of rebuilding the
    // resolution product, strides and coordinate multiplications per feature.
    struct CornerAddresses {
        uint lower[D], upper[D], table_size;
        bool hashed, linear;
        CornerAddresses(const thread uint* cell, uint resolution, uint size) : table_size(size) {
            uint extent = 1u;
            linear = true;
            for (uint d = 0; d < D; ++d) {
                if (resolution > 0xffffffffu / extent) { linear = false; break; }
                extent *= resolution;
            }
            if (!linear) extent = 0xffffffffu;
            hashed = G == 0u && size < extent;
            const uint prime[4] = {1958374283u, 2654435761u, 805459861u, 3674653429u};
            const uint reversed[4] = {2165219737u, 1434869437u, 2097192037u, 3674653429u};
            uint coefficient[D], base = 2531011u;
            if (H == 4u) for (uint d = D; d-- > 0u;) { coefficient[d] = base; base *= 2531011u; }
            uint stride = 1u;
            for (uint d = 0; d < D; ++d) {
                const uint multiplier = !hashed ? stride : H == 4u ? coefficient[d] :
                    H == 3u ? 1u : H == 2u ? reversed[d] : H == 1u && d == 0u ? 1u : prime[d];
                lower[d] = cell[d] * multiplier;
                upper[d] = (cell[d] + 1u) * multiplier;
                stride *= resolution;
            }
        }
        uint at(uint corner) const thread {
            if (!hashed && !linear) return 0u;
            if (hashed && H == 3u) {
                ulong step = 0ul;
                for (uint d = 0; d < D; ++d)
                    step ^= ulong(corner & (1u << d) ? upper[d] : lower[d]) << (d * (64u / D));
                return random_word(step) % table_size;
            }
            uint address = 0u;
            for (uint d = 0; d < D; ++d) {
                const uint term = corner & (1u << d) ? upper[d] : lower[d];
                if (hashed && H != 4u) address ^= term;
                else address += term;
            }
            return address % table_size;
        }
    };

    template <uint Order = 2u>
    __attribute__((always_inline)) static void coordinates(float scale, const thread tcnn_jit_array<float, D>& input,
                            thread uint* cell, thread float* fraction,
                            thread float* derivative, thread float* second) {
        for (uint d = 0; d < D; ++d) {
            float t;
            if (Fixed) {
                const float size = 4294967295.0f / scale;
                const uint fixed_size = size >= 4294967295.0f ? 0xffffffffu : max(uint(size), 1u);
                const uint fixed = as_type<uint>(input[d]) + fixed_size / 2u;
                cell[d] = fixed / fixed_size + (fixed < fixed_size / 2u ? uint(scale) : 0u);
                t = scale / 4294967295.0f * float(fixed - cell[d] * fixed_size);
            } else {
                const float coordinate = fma(scale, input[d], 0.5f), base = floor(coordinate);
                cell[d] = uint(int(base));
                t = coordinate - base;
            }
            fraction[d] = I == 2u ? t * t * (3.0f - 2.0f * t) : t;
            if (Order >= 1u) derivative[d] = I == 0u ? 0.0f : (I == 2u ? 6.0f * t * (1.0f - t) * scale : scale);
            second[d] = Order >= 2u && I == 2u ? (6.0f - 12.0f * t) * scale * scale : 0.0f;
        }
    }

    __attribute__((always_inline)) static float corner_weight(uint corner, const thread float* fraction) {
        float weight = 1.0f;
        if (I != 0u) for (uint d = 0; d < D; ++d)
            weight *= corner & (1u << d) ? fraction[d] : 1.0f - fraction[d];
        return weight;
    }

    template <typename T>
    __attribute__((always_inline)) static tcnn_jit_array<T, F> forward(float scale, const thread tcnn_jit_array<float, D>& input,
                                     const device T* params, uint table_size) {
        uint cell[D];
        float fraction[D], derivative[D], second[D], value[F] = {};
        coordinates<0>(scale, input, cell, fraction, derivative, second);
        const CornerAddresses addresses(cell, uint(ceil(scale)) + 1u, table_size);
        const uint corners = I == 0u ? 1u : (1u << D);
        for (uint corner = 0; corner < corners; ++corner) {
            const float weight = corner_weight(corner, fraction);
            const auto features = load_features<F>(params + addresses.at(corner) * F);
            for (uint f = 0; f < F; ++f) value[f] += weight * features[f];
        }
        tcnn_jit_array<T, F> result{};
        for (uint f = 0; f < F; ++f) result[f] = T(value[f]);
        return result;
    }

    // Evaluate the product rule without division: zero interpolation factors
    // remain valid and mixed derivatives are correct exactly on cell faces.
    __attribute__((always_inline)) static void weights(uint corner, const thread float* fraction, const thread float* derivative,
                        const thread float* second, const thread tcnn_jit_array<float, D>* direction,
                        thread float& weight, thread float* gradient,
                        thread float& tangent, thread float* mixed) {
        weight = 1.0f; tangent = 0.0f;
        for (uint d = 0; d < D; ++d) { gradient[d] = 0.0f; mixed[d] = 0.0f; }
        if (I == 0u) return;
        for (uint axis = 0; axis < D; ++axis) {
            const bool upper = (corner & (1u << axis)) != 0u;
            const float factor = upper ? fraction[axis] : 1.0f - fraction[axis];
            const float slope = (upper ? 1.0f : -1.0f) * derivative[axis];
            const float curvature = (upper ? 1.0f : -1.0f) * second[axis];
            const float velocity = direction ? (*direction)[axis] : 0.0f;
            for (uint d = 0; d < D; ++d) {
                if (direction)
                    mixed[d] = mixed[d] * factor + gradient[d] * slope * velocity +
                        (d == axis ? tangent * slope + weight * curvature * velocity : 0.0f);
                gradient[d] = gradient[d] * factor + (d == axis ? weight * slope : 0.0f);
            }
            tangent = tangent * factor + weight * slope * velocity;
            weight *= factor;
        }
    }

    template <typename T>
    __attribute__((always_inline)) static void backward(float scale, const thread tcnn_jit_array<float, D>& input,
                         const device T* params, uint table_size,
                         const thread tcnn_jit_array<T, F>& dy, device T* gradients,
                         thread tcnn_jit_array<float, D>* dx,
                         bool stochastic = false, uint level = 0u,
                         const thread tcnn_jit_environment* env = nullptr) {
        if (!gradients && !dx) return;
        uint cell[D];
        float fraction[D], derivative[D], second[D];
        if (dx) coordinates<1>(scale, input, cell, fraction, derivative, second);
        else coordinates<0>(scale, input, cell, fraction, derivative, second);
        const CornerAddresses addresses(cell, uint(ceil(scale)) + 1u, table_size);
        const uint corners = I == 0u ? 1u : (1u << D);
        uint selected = 0u;
        if (stochastic && I != 0u) {
            const uint sample = env ? env->index + level * env->count : level;
            const float random = float(random_word(ulong(sample)) >> 9u) * (1.0f / 8388608.0f);
            for (uint d = 0; d < D; ++d) if (random < fraction[d]) selected |= 1u << d;
        }
        // The usual training call requests only parameter gradients. It needs
        // no parameter reads, coordinate derivatives, or mixed-product arrays.
        // Stochastic interpolation selects a single corner without walking
        // the other 2^D addresses.
        if (!dx) {
            if (stochastic && I != 0u) {
                const uint address = addresses.at(selected) * F;
                scatter_features<F>(gradients, address, 1.0f, dy, env);
            } else {
                for (uint corner = 0; corner < corners; ++corner) {
                    const float weight = corner_weight(corner, fraction);
                    const uint address = addresses.at(corner) * F;
                    scatter_features<F>(gradients, address, weight, dy, env);
                }
            }
            return;
        }
        for (uint corner = 0; corner < corners; ++corner) {
            const uint address = addresses.at(corner) * F;
            float weight, tangent, gradient[D], mixed[D];
            weights(corner, fraction, derivative, second, nullptr, weight, gradient, tangent, mixed);
            const auto features = load_features<F>(params + address);
            if (gradients && (!stochastic || I == 0u || corner == selected))
                scatter_features<F>(gradients, address, stochastic && I != 0u ? 1.0f : weight, dy, env);
            for (uint f = 0; f < F; ++f) {
                for (uint d = 0; d < D; ++d)
                    (*dx)[d] += gradient[d] * features[f] * float(dy[f]);
            }
        }
    }

    template <typename T>
    __attribute__((always_inline)) static void backward_backward_input(float scale, const thread tcnn_jit_array<float, D>& input,
                         const device T* params, uint table_size,
                         const thread tcnn_jit_array<float, D>& direction,
                         const thread tcnn_jit_array<T, F>& dy, device T* gradients,
                         thread tcnn_jit_array<float, D>* dx, thread tcnn_jit_array<T, F>* ddy,
                         const thread tcnn_jit_environment* env = nullptr) {
        if (I == 0u) {
            if (ddy) for (uint f = 0; f < F; ++f) (*ddy)[f] = T(0.0f);
            return;
        }
        uint cell[D];
        float fraction[D], derivative[D], second[D], tangent_output[F] = {};
        coordinates<2>(scale, input, cell, fraction, derivative, second);
        const CornerAddresses addresses(cell, uint(ceil(scale)) + 1u, table_size);
        for (uint corner = 0; corner < (1u << D); ++corner) {
            const uint address = addresses.at(corner) * F;
            float weight, tangent, gradient[D], mixed[D];
            weights(corner, fraction, derivative, second, &direction, weight, gradient, tangent, mixed);
            tcnn_jit_array<float, F> features{};
            if (ddy || dx) features = load_features<F>(params + address);
            if (gradients) scatter_features<F>(gradients, address, tangent, dy, env);
            for (uint f = 0; f < F; ++f) {
                if (ddy) tangent_output[f] += tangent * features[f];
                if (dx) for (uint d = 0; d < D; ++d)
                    (*dx)[d] += mixed[d] * features[f] * float(dy[f]);
            }
        }
        if (ddy) for (uint f = 0; f < F; ++f) (*ddy)[f] = T(tangent_output[f]);
    }
};

// The simplex lattice preserves the source ranking/tie convention and its
// optional partial-input-gradient contract. Each level has its own seed table.
template <uint D, uint F>
struct Permuto {
    static uint geometry(const thread tcnn_jit_array<float, D>& input,
                         const thread tcnn_jit_array<float, D>& scales,
                         const thread tcnn_jit_array<float, D>& shifts,
                         uint table_size, uint max_grad_dims, uint first,
                         thread uint* ids, thread float* w, thread float* g, thread int* saved_rank = nullptr) {
    float elevated[D + 1], derivative[D + 1];
    int rem0[D + 1], rank[D + 1];
    float sum = 0, dsum = 0;
    for (int d = int(D); d > 0; --d) {
        const float cf = (input[uint(d - 1)] + shifts[d - 1]) * scales[d - 1];
        const float df = uint(d - 1) == first ? scales[d - 1] : 0;
        elevated[d] = sum - float(d) * cf;
        sum += cf;
        derivative[d] = dsum - float(d) * df;
        dsum += df;
    }
    elevated[0] = sum;
    derivative[0] = dsum;
    int rsum = 0;
    for (uint d = 0; d <= D; ++d) {
        const float q = elevated[d] / float(D + 1), up = ceil(q) * float(D + 1), down = floor(q) * float(D + 1);
        rem0[d] = int(up - elevated[d] < elevated[d] - down ? up : down);
        rsum += rem0[d];
        rank[d] = 0;
    }
    rsum /= int(D + 1);
    for (uint d = 0; d < D; ++d)
        for (uint j = d + 1; j <= D; ++j) {
            if (elevated[d] - rem0[d] < elevated[j] - rem0[j])
                ++rank[d];
            else
                ++rank[j];
        }
    for (uint d = 0; d <= D; ++d) {
        rank[d] += rsum;
        if (rank[d] < 0) {
            rank[d] += int(D + 1);
            rem0[d] += int(D + 1);
        } else if (rank[d] > int(D)) {
            rank[d] -= int(D + 1);
            rem0[d] -= int(D + 1);
        }
    }
    if (saved_rank) for (uint d = 0; d <= D; ++d) saved_rank[d] = rank[d];
    float bary[D + 2], dbary[D + 2];
    for (uint k = 0; k < D + 2; ++k) {
        bary[k] = 0;
        dbary[k] = 0;
    }
    for (uint d = 0; d <= D; ++d) {
        const float delta = (elevated[d] - rem0[d]) / float(D + 1), dd = derivative[d] / float(D + 1);
        const uint j = D - uint(rank[d]);
        bary[j] += delta;
        bary[j + 1] -= delta;
        dbary[j] += dd;
        dbary[j + 1] -= dd;
    }
    bary[0] += 1 + bary[D + 1];
    dbary[0] += dbary[D + 1];
    for (uint k = 0; k <= D; ++k) {
        uint index = 0;
        // Metal's unroller can lose the corrected coordinates/ranks for a
        // specialized 24D lattice. Keep this recurrence as a loop so vertex
        // hashes agree with AOT geometry (the permuto24 native regression).
#ifdef __METAL_VERSION__
#pragma clang loop unroll(disable)
#endif
        for (uint d = 0; d < D; ++d) {
            int key = rem0[d] + int(k);
            if (rank[d] > int(D - k))
                key -= int(D + 1);
            index = (index + uint(key)) * 2531011u;
        }
        ids[k] = index % table_size;
        w[k] = bary[k];
        g[k] = first < max_grad_dims ? dbary[k] : 0;
    }
    return D + 1;
    }
    template <typename T>
    static tcnn_jit_array<T, F> forward(const thread tcnn_jit_array<float, D>& input,
                         const device T* params, uint table_size,
                         const thread tcnn_jit_array<float, D>& scales,
                         const thread tcnn_jit_array<float, D>& shifts, uint max_grad_dims) {
        uint ids[D + 1]; float weight[D + 1], derivative[D + 1];
        const uint vertices = geometry(input, scales, shifts, table_size, max_grad_dims, 0xffffffffu, ids, weight, derivative);
        float value[F] = {};
        for (uint v = 0; v < vertices; ++v) {
            const auto features = load_features<F>(params + ids[v] * F);
            for (uint f = 0; f < F; ++f) value[f] += weight[v] * features[f];
        }
        tcnn_jit_array<T, F> result{};
        for (uint f = 0; f < F; ++f) result[f] = T(value[f]);
        return result;
    }

    template <typename T>
    static void backward(const thread tcnn_jit_array<float, D>& input,
                         const device T* params, uint table_size,
                         const thread tcnn_jit_array<float, D>& scales,
                         const thread tcnn_jit_array<float, D>& shifts, uint max_grad_dims,
                         const thread tcnn_jit_array<T, F>& dy, device T* gradients,
                         thread tcnn_jit_array<float, D>* dx,
                         const thread tcnn_jit_environment* env = nullptr) {
        if (!gradients && !dx) return;
        uint ids[D + 1]; int rank[D + 1];
        float weight[D + 1], derivative[D + 1], upstream[D + 1] = {};
        // Ranking, tie handling and vertex hashing are shared by every input
        // coordinate. Recomputing them per coordinate made the derivative
        // cubic in D even though a selected simplex is an affine map.
        const uint vertices = geometry(input, scales, shifts, table_size, max_grad_dims,
            0xffffffffu, ids, weight, derivative, rank);
        for (uint v = 0; v < vertices; ++v) {
            if (gradients) scatter_features<F>(gradients, ids[v] * F, weight[v], dy, env);
            if (dx && max_grad_dims) {
                const auto features = load_features<F>(params + ids[v] * F);
                for (uint f = 0; f < F; ++f) upstream[v] += features[f] * float(dy[f]);
            }
        }
        if (dx && max_grad_dims) {
            // Adjoint of the barycentric difference, then of elevation.
            // bary[0] includes bary[D+1], hence the cyclic upper neighbor.
            float elevated_adjoint[D + 1];
            for (uint d = 0; d <= D; ++d) {
                const uint j = D - uint(rank[d]);
                elevated_adjoint[d] = (upstream[j] - upstream[j == D ? 0u : j + 1u]) / float(D + 1);
            }
            float prefix = elevated_adjoint[0];
            for (uint d = 0; d < max_grad_dims; ++d) {
                (*dx)[d] += (prefix - float(d + 1u) * elevated_adjoint[d + 1u]) * scales[d];
                prefix += elevated_adjoint[d + 1u];
            }
        }
    }

    template <typename T>
    static void backward_backward_input(const thread tcnn_jit_array<float, D>& input,
                         const device T* params, uint table_size,
                         const thread tcnn_jit_array<float, D>& scales,
                         const thread tcnn_jit_array<float, D>& shifts, uint max_grad_dims,
                         const thread tcnn_jit_array<float, D>& direction,
                         const thread tcnn_jit_array<T, F>& dy, device T* gradients,
                         thread tcnn_jit_array<float, D>* dx, thread tcnn_jit_array<T, F>* ddy,
                         const thread tcnn_jit_environment* env = nullptr) {
        (void)dx; // A simplex is affine within its selected cell: its Hessian is zero.
        if (!gradients && !ddy) return;
        if (!max_grad_dims) {
            if (ddy) for (uint f = 0; f < F; ++f) (*ddy)[f] = T(0.0f);
            return;
        }
        uint ids[D + 1]; int rank[D + 1];
        float weight[D + 1], derivative[D + 1], elevated[D + 1], tangent[D + 2] = {};
        geometry(input, scales, shifts, table_size, max_grad_dims, 0xffffffffu, ids, weight, derivative, rank);
        float sum = 0.0f;
        for (uint d = D; d > 0u; --d) {
            const float component = d - 1u < max_grad_dims ? direction[d - 1u] * scales[d - 1u] : 0.0f;
            elevated[d] = sum - float(d) * component;
            sum += component;
        }
        elevated[0] = sum;
        for (uint d = 0; d <= D; ++d) {
            const uint j = D - uint(rank[d]);
            const float delta = elevated[d] / float(D + 1);
            tangent[j] += delta;
            tangent[j + 1u] -= delta;
        }
        tangent[0] += tangent[D + 1];
        float value[F] = {};
        for (uint v = 0; v <= D; ++v) {
            if (gradients) scatter_features<F>(gradients, ids[v] * F, tangent[v], dy, env);
            if (ddy) {
                const auto features = load_features<F>(params + ids[v] * F);
                for (uint f = 0; f < F; ++f) value[f] += tangent[v] * features[f];
            }
        }
        if (ddy) for (uint f = 0; f < F; ++f) (*ddy)[f] = T(value[f]);
    }

};
} // namespace tcnn_jit_encoding
#endif // TCNN_JIT_ENCODING_MATH
