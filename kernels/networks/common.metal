#ifndef TCNN_NETWORK_COMMON_METAL
#define TCNN_NETWORK_COMMON_METAL
// Baseline NN compute kernels. The matching C++ declarations live in nn.cpp.
// All matrices carry explicit strides; half and float storage share the same ABI.
#include <metal_stdlib>
using namespace metal;

struct Mat {
    uint rows, cols, rs, cs, precision;
};

struct Dense {
    Mat x, pre, y;
    uint inputs, outputs, activation;
};

struct Delta {
    Mat values, gradient, delta;
    uint activation, values_are_output;
};

struct SecondDelta {
    Mat pre, output, tangent, adjoint, tangent_adjoint, delta, tangent_delta;
    uint activation;
};

struct InputGradient {
    Mat delta, dx;
    uint inputs, outputs, precision;
};

struct WeightGradient {
    Mat x, delta;
    uint inputs, outputs, precision, mode;
};

struct Copy {
    Mat src, dst;
};

struct Unary {
    uint count, width, dim, mode, precision;
    float value;
};

struct ActivationArgs {
    uint count, precision, activation, mode;
};

struct MatrixActivation {
    Mat values, gradient, result;
    uint activation, mode;
};

struct Extract {
    uint count, dim, fan_in, fan_out, layout, precision;
};

inline uint index(Mat m, uint row, uint col) {
    return row * m.rs + col * m.cs;
}

inline float read_value(device const uchar* p, uint i, uint precision) {
    return precision == 2 ? float(((device const half*)p)[i]) : ((device const float*)p)[i];
}

inline void write_value(
    device uchar* p,
    uint i,
    uint precision,
    float value
) {
    if (precision == 2)
        ((device half*)p)[i] = half(value);
    else
        ((device float*)p)[i] = value;
}

// Preserve explicit T casts in the supplied half-precision activation arithmetic.
inline float round_to_precision(float value, uint precision) {
    return precision == 2 ? float(half(value)) : value;
}

// Metal's tanh can round to +/-1 before the correctly rounded float result
// saturates. Compute its small tail explicitly to retain nonzero derivatives.
inline float tanh_value(float x) {
    if (abs(x) >= 4.0f)
        return copysign(1.0f - 2.0f / (exp(2.0f * abs(x)) + 1.0f), x);
    return tanh(x);
}

// Activation ordering is the uploaded Activation enum, including k=10 softplus.
inline float activate(float x, uint act, uint precision) {
    switch (act) {
    case 0:
        return max(x, 0.0f);
    case 1:
        return x * round_to_precision(x > 0 ? 1.0f : 0.01f, precision);
    case 2:
        return x * (1.0f / (1.0f + exp(-x)));
    case 3:
        return exp(x);
    case 4:
        return sin(x);
    case 5:
        return 1.0f / (1.0f + exp(-x));
    case 6: {
        float t = x * 10.0f;
        return 0.5f * (t + sqrt(t * t + 4.0f)) / 10.0f;
    }
    case 7:
        return log(exp(x * 10.0f) + 1.0f) / 10.0f;
    case 8:
        return tanh_value(x);
    default:
        return x;
    }
}

// Derivative with preactivation input. This also handles noninvertible SiLU/Sine.
inline float derivative(float x, uint act) {
    switch (act) {
    case 0:
        return x > 0 ? 1.0f : 0.0f;
    case 1:
        return x > 0 ? 1.0f : 0.01f;
    case 2: {
        float s = 1.0f / (1.0f + exp(-x));
        return s + x * (s * (1.0f - s));
    }
    case 3:
        return exp(x);
    case 4:
        return cos(x);
    case 5: {
        float s = 1.0f / (1.0f + exp(-x));
        return s * (1.0f - s);
    }
    case 6: {
        float t = x * 10.0f;
        float y = 0.5f * (t + sqrt(t * t + 4.0f));
        return y * y / (y * y + 1.0f);
    }
    case 7: {
        float t = exp(x * 10.0f);
        return t / (t + 1.0f);
    }
    case 8: {
        float t = tanh_value(x);
        return 1.0f - t * t;
    }
    default:
        return 1.0f;
    }
}

// Derivative from output values where the activation is invertible.
inline float derivative_output(float y, uint act, uint precision) {
    switch (act) {
    case 0:
        return y > 0 ? 1.0f : 0.0f;
    case 1:
        return y > 0 ? 1.0f : 0.01f;
    case 3:
        return y;
    case 5:
        return y * round_to_precision(1.0f - y, precision);
    case 6: {
        float z = 10.0f * y;
        return z * z / (z * z + 1.0f);
    }
    case 7:
        return 1.0f - exp(-10.0f * y);
    case 8:
        return 1.0f - y * y;
    default:
        return 1.0f;
    }
}

// Round the derivative to T before multiplying by the incoming T gradient.
inline float activation_gradient(
    float value,
    float gradient,
    uint act,
    uint precision,
    bool values_are_output
) {
    if (act == 9)
        return gradient;
    float factor = values_are_output ? derivative_output(value, act, precision) : derivative(value, act);
    return gradient * round_to_precision(factor, precision);
}

// Curvature for tangent-over-reverse differentiation. Rounded outputs preserve
// saturation behavior, while noninvertible Sine/SiLU use their saved input.
inline float second_derivative(
    float x,
    float y,
    uint act,
    uint precision
) {
    switch (act) {
    case 2: {
        float s = 1.0f / (1.0f + exp(-x));
        return s * (1.0f - s) * (2.0f + x * (1.0f - 2.0f * s));
    }
    case 3:
        return y;
    case 4:
        return -sin(x);
    case 5:
        return derivative_output(y, act, precision) * (1.0f - 2.0f * y);
    case 6: {
        float z = 10.0f * y, d = z * z + 1.0f;
        return 20.0f * z * derivative_output(y, act, precision) / (d * d);
    }
    case 7: {
        float e = exp(-10.0f * y);
        return 10.0f * e * (1.0f - e);
    }
    case 8:
        return -2.0f * y * (1.0f - y * y);
    default:
        return 0.0f;
    }
}

#endif
