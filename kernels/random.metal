#include <metal_stdlib>
using namespace metal;

namespace tcnn_random {
using Word = uint;
using Wide = ulong;

struct Args {
    Wide state, inc;
    Word count, threads, fp16, kind;
    float first, second;
};

// Advance PCG by an arbitrary stride, preserving the uploaded generator's sequence.
inline Wide pcg_advance(Wide state, Wide inc, Wide delta) {
    Wide mult = 0x5851f42d4c957f2dUL, plus = inc, acc_mult = 1, acc_plus = 0;
    while (delta) {
        if (delta & 1) {
            acc_mult *= mult;
            acc_plus = acc_plus * mult + plus;
        }
        plus = (mult + 1) * plus;
        mult *= mult;
        delta >>= 1;
    }
    return acc_mult * state + acc_plus;
}

// Produce one of PCG's 23-bit uniform floating-point samples.
inline float pcg_next(thread Wide& state, Wide inc) {
    Wide old = state;
    state = old * 0x5851f42d4c957f2dUL + inc;
    Word value = Word(((old >> 18) ^ old) >> 27), rot = Word(old >> 59);
    Word bits = (value >> rot) | (value << ((-rot) & 31));
    return as_type<float>((bits >> 9) | 0x3f800000u) - 1.0f;
}

inline void uniform_block16(constant const Args& a, Word first_thread, thread float* values) {
    Wide state = pcg_advance(a.state, a.inc, Wide(first_thread) * 4);
    for (Word lane = 0; lane < 4; ++lane)
        for (Word plane = 0; plane < 4; ++plane) {
            const float value = pcg_next(state, a.inc);
            // Keep the scalar kernel's range expression and rounding contract.
            values[plane * 4 + lane] = value * (a.second - a.first) + a.first;
        }
}
} // namespace tcnn_random

using RandomArgs = tcnn_random::Args;
using tcnn_random::pcg_advance;
using tcnn_random::pcg_next;

// Match the reference's four output planes per physical thread.
kernel void tcnn_random_pcg(device uchar* out [[buffer(0)]], constant RandomArgs& a [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.threads)
        return;
    ulong state = pcg_advance(a.state, a.inc, ulong(i) * 4);
    for (uint j = 0; j < 4; ++j) {
        uint idx = i + a.threads * j;
        if (idx >= a.count)
            return;
        float value = pcg_next(state, a.inc);
        if (a.fp16)
            value = float(half(value));
        float transformed = a.kind ? (-log(1.0f / clamp(value, 1e-9f, 1.0f - 1e-9f) - 1.0f)) * a.second * 0.551328895f + a.first
                                   : value * (a.second - a.first) + a.first;
        if (a.fp16)
            reinterpret_cast<device half*>(out)[idx] = half(transformed);
        else
            reinterpret_cast<device float*>(out)[idx] = transformed;
    }
}

// Canonical large FP32 uniform path. Each block replaces four independent PCG
// jumps while preserving the four-plane ordering, masked tail and host advance.
kernel void tcnn_random_uniform_f32(
    device float* out [[buffer(0)]],
    constant RandomArgs& a [[buffer(1)]],
    uint block [[thread_position_in_grid]]
) {
    const ulong first = ulong(block) * 4;
    if (first >= a.threads || first >= a.count)
        return;
    float values[16];
    tcnn_random::uniform_block16(a, uint(first), values);
    for (uint plane = 0; plane < 4; ++plane) {
        const ulong index = first + ulong(a.threads) * plane;
        if (index + 4 <= a.count)
            reinterpret_cast<device float4*>(out)[index / 4] =
                float4(values[plane * 4], values[plane * 4 + 1], values[plane * 4 + 2], values[plane * 4 + 3]);
        else
            for (uint lane = 0; lane < 4; ++lane)
                if (index + lane < a.count)
                    out[index + lane] = values[plane * 4 + lane];
    }
}
