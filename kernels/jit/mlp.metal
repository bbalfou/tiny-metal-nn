// Activation arithmetic shared by generated native MLP device functions.
// The casts match the existing Metal AOT parameter/activation precision rules.
#ifndef TCNN_JIT_MLP_METAL
#define TCNN_JIT_MLP_METAL
#include <metal_stdlib>
using namespace metal;

#ifndef TCNN_JIT_HAS_SIMDGROUP_MATRIX
#define TCNN_JIT_HAS_SIMDGROUP_MATRIX 1
#endif
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 230 && TCNN_JIT_HAS_SIMDGROUP_MATRIX
#include "mlp_cooperative.metal"
#endif

inline float tcnn_jit_mlp_tanh(float x) {
    if (abs(x) >= 4.0f)
        return copysign(1.0f - 2.0f / (exp(2.0f * abs(x)) + 1.0f), x);
    return tanh(x);
}

template <typename T>
inline T tcnn_jit_mlp_activate(T pre, uint activation) {
    const float x = float(pre);
    switch (activation) {
    case 0:
        return T(max(x, 0.0f));
    case 1:
        return T(x * float(T(x > 0.0f ? 1.0f : 0.01f)));
    case 2:
        return T(x / (1.0f + exp(-x)));
    case 3:
        return T(exp(x));
    case 4:
        return T(sin(x));
    case 5:
        return T(1.0f / (1.0f + exp(-x)));
    case 6: {
        const float z = 10.0f * x;
        return T(0.5f * (z + sqrt(z * z + 4.0f)) / 10.0f);
    }
    case 7:
        return T(log(exp(10.0f * x) + 1.0f) / 10.0f);
    case 8:
        return T(tcnn_jit_mlp_tanh(x));
    default:
        return pre;
    }
}

// Invertible activations use their rounded stored output, exactly as ordinary
// network backward; Sine and SiLU retain and consume their preactivations.
template <typename T>
inline T tcnn_jit_mlp_derivative(T pre, T output, uint activation) {
    const float x = float(pre), y = float(output);
    switch (activation) {
    case 0:
        return T(y > 0.0f ? 1.0f : 0.0f);
    case 1:
        return T(y > 0.0f ? 1.0f : 0.01f);
    case 2: {
        const float s = 1.0f / (1.0f + exp(-x));
        return T(s + x * s * (1.0f - s));
    }
    case 3:
        return output;
    case 4:
        return T(cos(x));
    case 5:
        return T(y * float(T(1.0f - y)));
    case 6: {
        const float z = 10.0f * y;
        return T(z * z / (z * z + 1.0f));
    }
    case 7:
        return T(1.0f - exp(-10.0f * y));
    case 8:
        return T(1.0f - y * y);
    default:
        return T(1.0f);
    }
}

template <typename T>
inline T tcnn_jit_mlp_curvature(T pre, T output, uint activation) {
    const float x = float(pre), y = float(output);
    switch (activation) {
    case 2: {
        const float s = 1.0f / (1.0f + exp(-x));
        return T(s * (1.0f - s) * (2.0f + x * (1.0f - 2.0f * s)));
    }
    case 3:
        return output;
    case 4:
        return T(-sin(x));
    case 5: {
        const float first = y * float(T(1.0f - y));
        return T(first * (1.0f - 2.0f * y));
    }
    case 6: {
        const float z = 10.0f * y, d = z * z + 1.0f;
        return T(20.0f * z * (z * z / d) / (d * d));
    }
    case 7: {
        const float e = exp(-10.0f * y);
        return T(10.0f * e * (1.0f - e));
    }
    case 8:
        return T(-2.0f * y * (1.0f - y * y));
    default:
        return T(0.0f);
    }
}
#endif
