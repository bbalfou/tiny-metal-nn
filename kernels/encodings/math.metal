#ifndef TCNN_ENCODING_MATH_METAL
#define TCNN_ENCODING_MATH_METAL

#ifdef __METAL_VERSION__
#include "encodings/abi.metal"
using namespace metal;
#define EC constant
#define ED device
#define ET thread
#else
#include <tiny-metal-nn/detail/kernels/encodings/abi.h>
#include <pcg32/pcg32.h>
#include <cmath>
#include <cstdint>
#define EC
#define ED
#define ET
#endif
namespace tcnn_encoding_math {
#ifndef __METAL_VERSION__
using uint = uint32_t;
using ulong = uint64_t;
using std::floor;
using std::ceil;
using std::sin;
using std::cos;
using std::exp2;
using std::fma;
using std::fabs;
using std::fmax;
using std::fmin;
#endif
#ifndef __METAL_VERSION__
using tcnn::detail::kernels::encodings::Mat;
using tcnn::detail::kernels::encodings::Args;
using tcnn::detail::kernels::encodings::UtilityArgs;
#endif

inline uint at(Mat m, uint r, uint c) {
    return r * m.rs + c * m.cs;
}

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

/// Computes an analytic feature, a coordinate derivative, and optional Jv/Hv data.
inline Dual analytic(
    EC Args& a,
    ED const float* x,
    uint row,
    uint col,
    uint dim,
    ED const float* direction = nullptr
) {
    // The supplied non-JIT SH implementation places constant padding BEFORE
    // its harmonics. Other analytic encodings append constant padding.
    if (a.kind == 5) {
        if (row < a.resolution)
            return {a.padding, 0};
        row -= a.resolution;
    }
    if (row >= a.width)
        return {a.padding, 0};
    if (a.kind == 0)
        return {1, 0};
    if (a.kind == 1)
        return {x[at(a.x, row, col)] * a.scale + a.offset,
                row == dim ? a.scale : 0,
                direction ? direction[at(a.ddx, row, col)] * a.scale : 0,
                0};
    if (a.kind == 5)
        return sh_component(
            row,
            {x[at(a.x, 0, col)] * 2 - 1,
             dim == 0 ? 2.0f : 0.0f,
             direction ? 2 * direction[at(a.ddx, 0, col)] : 0,
             0},
            {x[at(a.x, 1, col)] * 2 - 1,
             dim == 1 ? 2.0f : 0.0f,
             direction ? 2 * direction[at(a.ddx, 1, col)] : 0,
             0},
            {x[at(a.x, 2, col)] * 2 - 1,
             dim == 2 ? 2.0f : 0.0f,
             direction ? 2 * direction[at(a.ddx, 2, col)] : 0,
             0}
        );
    const uint fan = a.kind == 2 ? a.degree * 2 : a.degree;
    const uint d = row / fan, k = row % fan;
    const float p = x[at(a.x, d, col)];
    const float tangent = direction ? direction[at(a.ddx, d, col)] : 0;
    if (a.kind == 2) {
        const float frequency = 3.14159265358979323846f * exp2(float(k / 2));
        const float angle = p * frequency + (k % 2) * 1.57079632679489661923f;
        const float value = sin(angle), derivative = frequency * cos(angle);
        return {value,
                d == dim ? derivative : 0,
                derivative * tangent,
                d == dim ? -frequency * frequency * value * tangent : 0};
    }
    if (a.kind == 3) {
        const float val = p * exp2(float(int(k) - 1)) + k * .25f;
        const float derivative = exp2(float(k + 1)) * ((int(floor(val * 2)) % 2) == 0 ? -1.0f : 1.0f);
        return {fabs(val - floor(val) - .5f) * 4 - 1,
                d == dim ? derivative : 0,
                derivative * tangent,
                0};
    }
    float value = 0, derivative = 0, second = 0;
    for (int shift = -1; shift <= 1; ++shift) {
        const float left = float(k) / a.degree - p + shift, right = float(k + 1) / a.degree - p + shift;
        value += quartic_cdf(right, float(a.degree)) - quartic_cdf(left, float(a.degree));
        derivative += quartic_pdf(left, float(a.degree)) - quartic_pdf(right, float(a.degree));
        second += quartic_pdf_derivative(right, float(a.degree)) - quartic_pdf_derivative(left, float(a.degree));
    }
    return {value,
            d == dim ? derivative : 0,
            derivative * tangent,
            d == dim ? second * tangent : 0};
}

/// Reproduce PCG's 64-bit jump-ahead for both optional Rng hashing and stochastic gradients.
inline uint random_word(ulong delta) {
#ifndef __METAL_VERSION__
    pcg32 rng{1337};
    rng.advance(static_cast<int64_t>(delta));
    return rng.nextUInt();
#else
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
#endif
}

// Grid interpolation can have 2^24 vertices in the optional direct BaseConvert
// path. Visit bounded chunks instead of allocating an exponential thread array.
// Ordinary 2D/3D/4D grids and all Permuto dimensions fit one chunk.
inline uint vertex_count(EC Args& a) {
    return a.kind == 6 ? (a.interpolation == 0 ? 1u : (1u << a.dims)) : a.dims + 1;
}

/// Computes one bounded chunk of addresses and first/mixed derivatives.
inline uint geometry(
    EC Args& a,
    ED const float* x,
    uint col,
    uint first,
    uint second,
    ET uint* ids,
    ET float* w,
    ET float* g,
    ET float* h,
    uint first_vertex = 0
) {
    if (a.kind == 6) {
        float p[24], dp[24], ddp[24];
        uint base[24];
        for (uint d = 0; d < a.dims; ++d) {
            const float u = fma(a.scale, x[at(a.x, d, col)], .5f), b = floor(u), t = u - b;
            base[d] = uint(int(b));
            p[d] = a.interpolation == 2 ? t * t * (3 - 2 * t) : t;
            dp[d] = a.interpolation == 2 ? 6 * t * (1 - t) * a.scale : a.scale;
            ddp[d] = a.interpolation == 2 ? (6 - 12 * t) * a.scale * a.scale : 0;
        }
        const uint total = vertex_count(a);
        const uint n = total - first_vertex < 25u ? total - first_vertex : 25u;
        uint dense_stride = 1;
        bool linear_address = true;
        for (uint d = 0; d < a.dims; ++d) {
            if (a.resolution > 0xffffffffu / dense_stride) {
                linear_address = false;
                break;
            }
            dense_stride *= a.resolution;
        }
        for (uint slot = 0; slot < n; ++slot) {
            const uint v = first_vertex + slot;
            uint stride = 1, index = 0, hash = 0;
            ulong rng_step = 0;
            float weight = 1, grad = 1, hess = 1;
            const uint prime[4] = {1958374283u,
                                   2654435761u,
                                   805459861u,
                                   3674653429u};
            const uint reversed[4] = {2165219737u,
                                      1434869437u,
                                      2097192037u,
                                      3674653429u};
            for (uint d = 0; d < a.dims; ++d) {
                const bool upper = (v & (1u << d)) != 0;
                const uint key = base[d] + uint(upper);
                if (linear_address) {
                    index += key * stride;
                    stride *= a.resolution;
                } else
                    stride = 0xffffffffu;
                if (a.hash_type == 3)
                    rng_step ^= ulong(key) << (d * (64 / a.dims));
                else if (a.hash_type == 4)
                    hash = (hash + key) * 2531011u;
                else
                    hash ^= key * (a.hash_type == 2 ? reversed[d] : a.hash_type == 1 && d == 0 ? 1u : prime[d]);
                const float factor = upper ? p[d] : 1 - p[d], s = upper ? 1.0f : -1.0f;
                weight *= factor;
                grad *= d == first ? s * dp[d] : factor;
                if (d == first && d == second)
                    hess *= s * ddp[d];
                else if (d == first || d == second)
                    hess *= s * dp[d];
                else
                    hess *= factor;
            }
            if (a.grid_type == 0 && a.table_size < stride)
                index = a.hash_type == 3 ? random_word(rng_step) : hash;
            ids[slot] = index % a.table_size;
            w[slot] = a.interpolation == 0 ? 1 : weight;
            g[slot] = a.interpolation == 0 || first >= a.dims ? 0 : grad;
            h[slot] = a.interpolation == 0 || first >= a.dims || second >= a.dims ? 0 : hess;
        }
        return n;
    }
    float elevated[25], derivative[25];
    int rem0[25], rank[25];
    float sum = 0, dsum = 0;
    for (int d = int(a.dims); d > 0; --d) {
        const float cf = (x[at(a.x, uint(d - 1), col)] + a.shifts[d - 1]) * a.scales[d - 1];
        const float df = uint(d - 1) == first ? a.scales[d - 1] : 0;
        elevated[d] = sum - float(d) * cf;
        sum += cf;
        derivative[d] = dsum - float(d) * df;
        dsum += df;
    }
    elevated[0] = sum;
    derivative[0] = dsum;
    int rsum = 0;
    for (uint d = 0; d <= a.dims; ++d) {
        const float q = elevated[d] / float(a.dims + 1), up = ceil(q) * float(a.dims + 1), down = floor(q) * float(a.dims + 1);
        rem0[d] = int(up - elevated[d] < elevated[d] - down ? up : down);
        rsum += rem0[d];
        rank[d] = 0;
    }
    rsum /= int(a.dims + 1);
    for (uint d = 0; d < a.dims; ++d)
        for (uint j = d + 1; j <= a.dims; ++j) {
            if (elevated[d] - rem0[d] < elevated[j] - rem0[j])
                ++rank[d];
            else
                ++rank[j];
        }
    for (uint d = 0; d <= a.dims; ++d) {
        rank[d] += rsum;
        if (rank[d] < 0) {
            rank[d] += int(a.dims + 1);
            rem0[d] += int(a.dims + 1);
        } else if (rank[d] > int(a.dims)) {
            rank[d] -= int(a.dims + 1);
            rem0[d] -= int(a.dims + 1);
        }
    }
    float bary[26], dbary[26];
    for (uint k = 0; k < a.dims + 2; ++k) {
        bary[k] = 0;
        dbary[k] = 0;
    }
    for (uint d = 0; d <= a.dims; ++d) {
        const float delta = (elevated[d] - rem0[d]) / float(a.dims + 1), dd = derivative[d] / float(a.dims + 1);
        const uint j = a.dims - uint(rank[d]);
        bary[j] += delta;
        bary[j + 1] -= delta;
        dbary[j] += dd;
        dbary[j + 1] -= dd;
    }
    bary[0] += 1 + bary[a.dims + 1];
    dbary[0] += dbary[a.dims + 1];
    for (uint k = 0; k <= a.dims; ++k) {
        uint index = 0;
        for (uint d = 0; d < a.dims; ++d) {
            int key = rem0[d] + int(k);
            if (rank[d] > int(a.dims - k))
                key -= int(a.dims + 1);
            index = (index + uint(key)) * 2531011u;
        }
        ids[k] = index % a.table_size;
        w[k] = bary[k];
        g[k] = first < a.max_input_grad_dims ? dbary[k] : 0;
        h[k] = 0;
    }
    return a.dims + 1;
}

inline bool inactive(EC Args& a, ED const float* limits, uint col) {
    return float(a.level) >= (a.has_limits ? limits[col] : a.max_level) * a.levels + 1e-3f;
}

/// Convert the same PCG output bits as the uploaded next_float() implementation.
inline float random_sample(uint idx) {
    return float(random_word(ulong(idx)) >> 9) * (1.0f / 8388608.0f);
}

#ifndef __METAL_VERSION__
// Independent Reference first-order callbacks; no native entrypoints.
/// Each forward thread writes one feature/sample; no input or parameter readback occurs.
template <typename T>
inline void forward(
    EC Args& a,
    ED const float* x,
    ED const T* params,
    ED const float* limits,
    ED T* out,
    uint i
) {
    const uint width = a.kind < 6 ? a.y.rows : a.features;
    if (i >= width * a.x.cols)
        return;
    const uint row = i % width, col = i / width;
    if (a.kind < 6) {
        out[at(a.y, row, col)] = T(analytic(
                                       a,
                                       x,
                                       row,
                                       col,
                                       0xffffffffu
        )
                                       .v);
        return;
    }
    const uint output_row = a.level * a.features + row;
    if (inactive(a, limits, col)) {
        out[at(a.y, output_row, col)] = T(0);
        return;
    }
    uint ids[25];
    float w[25], g[25], h[25];
    float value = 0;
    for (uint begin = 0; begin < vertex_count(a); begin += 25) {
        const uint n = geometry(
            a,
            x,
            col,
            0xffffffffu,
            0xffffffffu,
            ids,
            w,
            g,
            h,
            begin
        );
        for (uint k = 0; k < n; ++k)
            value += w[k] * float(params[ids[k] * a.features + row]);
    }
    out[at(a.y, output_row, col)] = T(value);
}

/// Computes input gradients or a Hessian-vector product; successive levels accumulate.
template <typename T>
inline void backward_input(
    EC Args& a,
    ED const float* x,
    ED const T* params,
    ED const float* limits,
    ED const T* dy,
    ED float* dx,
    ED const float* ddx,
    uint i
) {
    if (i >= a.dims * a.x.cols)
        return;
    const uint dim = i % a.dims, col = i / a.dims;
    float value = 0;
    if (a.kind < 6) {
        if (a.mode != 0 && (a.kind == 0 || a.kind == 1 || a.kind == 3)) {
            dx[at(a.dx, dim, col)] = 0;
            return;
        }
        for (uint r = 0; r < a.dy.rows; ++r) {
            const Dual feature = analytic(
                a,
                x,
                r,
                col,
                dim,
                a.mode ? ddx : nullptr
            );
            value += float(dy[at(a.dy, r, col)]) * (a.mode ? feature.dt : feature.d);
        }
        if (a.kind == 1 && a.mode == 0)
            value = float(T(value));
        dx[at(a.dx, dim, col)] = value;
        return;
    }
    if (!inactive(a, limits, col) && dim < a.max_input_grad_dims) {
        uint ids[25];
        float w[25], g[25], h[25];
        if (a.mode == 0) {
            for (uint begin = 0; begin < vertex_count(a); begin += 25) {
                const uint n = geometry(
                    a,
                    x,
                    col,
                    dim,
                    0xffffffffu,
                    ids,
                    w,
                    g,
                    h,
                    begin
                );
                for (uint f = 0; f < a.features; ++f)
                    for (uint k = 0; k < n; ++k)
                        value += float(dy[at(a.dy, a.level * a.features + f, col)]) * g[k] * float(params[ids[k] * a.features + f]);
            }
        } else if (a.kind == 6) {
            for (uint other = 0; other < a.dims; ++other) {
                for (uint begin = 0; begin < vertex_count(a); begin += 25) {
                    const uint n = geometry(
                        a,
                        x,
                        col,
                        dim,
                        other,
                        ids,
                        w,
                        g,
                        h,
                        begin
                    );
                    for (uint f = 0; f < a.features; ++f)
                        for (uint k = 0; k < n; ++k)
                            value += float(dy[at(a.dy, a.level * a.features + f, col)]) * h[k] * float(params[ids[k] * a.features + f]) *
                                     ddx[at(a.ddx, other, col)];
                }
            }
        }
    }
    dx[at(a.dx, dim, col)] = (a.accumulate ? dx[at(a.dx, dim, col)] : 0) + value;
}

#endif

/// Computes the derivative with respect to upstream output gradients in double backward.
template <typename T>
inline void backward_output(
    EC Args& a,
    ED const float* x,
    ED const T* params,
    ED const float* limits,
    ED const float* ddx,
    ED T* ddy,
    uint i
) {
    const uint width = a.kind < 6 ? a.ddy.rows : a.features;
    if (i >= width * a.x.cols)
        return;
    const uint row = i % width, col = i / width;
    float value = 0;
    if (a.kind < 6) {
        value = analytic(
                    a,
                    x,
                    row,
                    col,
                    0xffffffffu,
                    ddx
        )
                    .t;
        ddy[at(a.ddy, row, col)] = T(value);
        return;
    }
    if (!inactive(a, limits, col)) {
        uint ids[25];
        float w[25], g[25], h[25];
        for (uint d = 0; d < a.max_input_grad_dims; ++d) {
            for (uint begin = 0; begin < vertex_count(a); begin += 25) {
                const uint n = geometry(
                    a,
                    x,
                    col,
                    d,
                    0xffffffffu,
                    ids,
                    w,
                    g,
                    h,
                    begin
                );
                for (uint k = 0; k < n; ++k)
                    value += g[k] * float(params[ids[k] * a.features + row]) * ddx[at(a.ddx, d, col)];
            }
        }
    }
    ddy[at(a.ddy, a.level * a.features + row, col)] = T(value);
}

#ifndef __METAL_VERSION__
// Independent Reference parameter-gradient gather.
/// One lane owns one feature's entire level gradient, avoiding atomics and write races.
template <typename T>
inline void backward_params(
    EC Args& a,
    ED const float* x,
    ED const float* limits,
    ED const T* dy,
    ED float* gradients,
    ED const float* ddx,
    uint feature
) {
    if (feature >= a.features)
        return;
    for (uint v = 0; v < a.table_size; ++v)
        gradients[v * a.features + feature] = 0;
    uint ids[25];
    float w[25], g[25], h[25];
    for (uint col = 0; col < a.x.cols; ++col) {
        if (inactive(a, limits, col))
            continue;
        const float grad = float(dy[at(a.dy, a.level * a.features + feature, col)]);
        if (a.mode == 0) {
            if (a.stochastic && a.kind == 6 && a.interpolation != 0) {
                const float sample = random_sample(col + a.level * a.x.cols);
                uint selected = 0;
                for (uint d = 0; d < a.dims; ++d) {
                    const float u = fma(a.scale, x[at(a.x, d, col)], .5f), t = u - floor(u),
                                p = a.interpolation == 2 ? t * t * (3 - 2 * t) : t;
                    if (sample < p)
                        selected |= 1u << d;
                }
                geometry(
                    a,
                    x,
                    col,
                    0xffffffffu,
                    0xffffffffu,
                    ids,
                    w,
                    g,
                    h,
                    selected
                );
                gradients[ids[0] * a.features + feature] += grad;
            } else {
                for (uint begin = 0; begin < vertex_count(a); begin += 25) {
                    const uint n = geometry(
                        a,
                        x,
                        col,
                        0xffffffffu,
                        0xffffffffu,
                        ids,
                        w,
                        g,
                        h,
                        begin
                    );
                    for (uint k = 0; k < n; ++k)
                        gradients[ids[k] * a.features + feature] += w[k] * grad;
                }
            }
        } else {
            for (uint d = 0; d < a.max_input_grad_dims; ++d) {
                for (uint begin = 0; begin < vertex_count(a); begin += 25) {
                    const uint n = geometry(
                        a,
                        x,
                        col,
                        d,
                        0xffffffffu,
                        ids,
                        w,
                        g,
                        h,
                        begin
                    );
                    for (uint k = 0; k < n; ++k)
                        gradients[ids[k] * a.features + feature] += g[k] * grad * ddx[at(a.ddx, d, col)];
                }
            }
        }
    }
}

#endif

/// Matrix utility for padding, slicing copies, reductions, and LoD weighting.
template <typename T>
inline void utility(
    EC UtilityArgs& a,
    ED const T* src,
    ED T* dst,
    ED const T* aux,
    ED const float* limits,
    uint i
) {
    if (i >= a.dst.rows * a.dst.cols)
        return;
    const uint r = i % a.dst.rows, c = i / a.dst.rows;
    if (a.mode == 0) {
        dst[at(a.dst, r, c)] = T(a.value);
        return;
    }
    if (a.mode == 1) {
        dst[at(a.dst, r, c)] = src[at(a.src, r, c)];
        return;
    }
    if (a.mode == 2) {
        dst[at(a.dst, r, c)] = T(float(dst[at(a.dst, r, c)]) + float(src[at(a.src, r, c)]));
        return;
    }
    if (a.mode == 3 || a.mode == 4) {
        float result = a.mode == 3 ? 0.0f : 1.0f;
        for (uint n = 0; n < a.nested; ++n) {
            const float v = float(src[at(a.src, n * a.width + r, c)]);
            result = a.mode == 3 ? result + v : result * v;
        }
        dst[at(a.dst, r, c)] = T(result);
        return;
    }
    if (a.mode == 5 || a.mode == 6) {
        float result = float(src[at(a.src, r, c)]);
        if (a.mode == 6)
            for (uint n = 0; n < a.nested; ++n)
                if (n != a.index)
                    result *= float(aux[at(a.aux, n * a.width + r, c)]);
        dst[at(a.dst, r, c)] = T(result);
        return;
    }
    if (a.mode == 8) {
        float result = 0;
        for (uint n = 0; n < a.nested; ++n) {
            float term = float(src[at(a.src, n * a.width + r, c)]);
            for (uint other = 0; other < a.nested; ++other)
                if (other != n)
                    term *= float(aux[at(a.aux, other * a.width + r, c)]);
            result += term;
        }
        dst[at(a.dst, r, c)] = T(result);
        return;
    }
    float weight = 1;
    if (r < a.levels * a.features) {
        const float lf = limits[c] * a.levels + 1e-3f;
        const uint level = r / a.features;
        weight = fmax(0.0f, fmin(1.0f, lf - float(level)));
    }
    dst[at(a.dst, r, c)] = weight == 0 ? T(0) : T(weight * float(src[at(a.src, r, c)]));
}

/// Product reduction's second-order adjoint for one child's ordinary output.
/// Multiplication excluding two factors remains defined even when an output is zero.
template <typename T>
inline void product_second(
    EC UtilityArgs& a,
    ED const T* dy,
    ED const T* values,
    ED const T* tangents,
    ED T* dst,
    uint i
) {
    if (i >= a.dst.rows * a.dst.cols)
        return;
    const uint r = i % a.dst.rows, c = i / a.dst.rows;
    float result = 0;
    for (uint other = 0; other < a.nested; ++other)
        if (other != a.index) {
            float term = float(tangents[at(a.aux, other * a.width + r, c)]);
            for (uint k = 0; k < a.nested; ++k)
                if (k != a.index && k != other)
                    term *= float(values[at(a.aux, k * a.width + r, c)]);
            result += term;
        }
    dst[at(a.dst, r, c)] = T(float(dy[at(a.src, r, c)]) * result);
}
} // namespace tcnn_encoding_math

#undef EC
#undef ED
#undef ET

#endif // TCNN_ENCODING_MATH_METAL
