#pragma once
#include <tiny-metal-nn/common.h>
#include <cmath>

namespace tcnn {
static constexpr float SQRT2 = 1.41421356237309504880f;
static constexpr float K_ACT = 10.0f;

// Logistic and clipped inverse logistic use the reference's constants.
inline float logistic(const float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float logit(const float x) {
    return -std::log(1.0f / std::fmin(std::fmax(x, 1e-9f), 1.0f - 1e-9f) - 1.0f);
}

// Normalize exponentials in place, matching the uploaded helper's arithmetic.
template <uint32_t N> void softmax(float vals[N]) {
    float total = 0;
    for (uint32_t i = 0; i < N; ++i) {
        vals[i] = std::exp(vals[i]);
        total += vals[i];
    }
    for (uint32_t i = 0; i < N; ++i)
        vals[i] /= total;
}

template <typename T> T relu(T val) {
    return T(std::max(float(val), 0.0f));
}

// Evaluate each public activation using exactly the reference's parameterization.
template <typename T> T activation(Activation act, T val) {
    const float x = float(val);
    switch (act) {
    case Activation::ReLU:
        return T(std::max(x, 0.0f));
    case Activation::LeakyReLU:
        return T(x * float(T(x > 0 ? 1.0f : 0.01f)));
    case Activation::SiLU:
        return T(x * logistic(x));
    case Activation::Exponential:
        return T(std::exp(x));
    case Activation::Sine:
        return T(std::sin(x));
    case Activation::Sigmoid:
        return T(logistic(x));
    case Activation::Squareplus: {
        const float z = x * K_ACT;
        return T(0.5f * (z + std::sqrt(z * z + 4)) / K_ACT);
    }
    case Activation::Softplus:
        return T(std::log(std::exp(x * K_ACT) + 1.0f) / K_ACT);
    case Activation::Tanh:
        return T(std::tanh(x));
    case Activation::None:
        return val;
    }
    return val;
}

// Backpropagate through an activation using its saved pre-activation value.
template <typename T> T activation_backward_in(Activation act, T val, T forward_val_in) {
    const float x = float(forward_val_in);
    float derivative = 1;
    switch (act) {
    case Activation::ReLU:
        derivative = x > 0 ? 1.0f : 0.0f;
        break;
    case Activation::LeakyReLU:
        derivative = x > 0 ? 1.0f : 0.01f;
        break;
    case Activation::SiLU: {
        const float s = logistic(x);
        derivative = s + x * (s * (1 - s));
        break;
    }
    case Activation::Exponential:
        derivative = std::exp(x);
        break;
    case Activation::Sine:
        derivative = std::cos(x);
        break;
    case Activation::Sigmoid: {
        const float s = logistic(x);
        derivative = s * (1 - s);
        break;
    }
    case Activation::Squareplus: {
        const float z = x * K_ACT, y = 0.5f * (z + std::sqrt(z * z + 4));
        derivative = y * y / (y * y + 1);
        break;
    }
    case Activation::Softplus: {
        const float tmp = std::exp(x * K_ACT);
        derivative = tmp / (tmp + 1);
        break;
    }
    case Activation::Tanh: {
        const float t = std::tanh(x);
        derivative = 1 - t * t;
        break;
    }
    case Activation::None:
        break;
    }
    return T(float(val) * float(T(derivative)));
}

// Post-activation backprop is undefined for SiLU/Sine in the uploaded reference.
// Reject those cases explicitly so callers can select the pre-activation variant.
template <typename T> T activation_backward(Activation act, T val, T forward_val) {
    const float y = float(forward_val);
    float derivative = 1;
    switch (act) {
    case Activation::ReLU:
        derivative = y > 0 ? 1.0f : 0.0f;
        break;
    case Activation::LeakyReLU:
        derivative = y > 0 ? 1.0f : 0.01f;
        break;
    case Activation::SiLU:
    case Activation::Sine:
        throw std::invalid_argument("SiLU/Sine backward requires pre-activations");
    case Activation::Exponential:
        derivative = y;
        break;
    case Activation::Sigmoid:
        derivative = float(T(y * float(T(1 - y))));
        break;
    case Activation::Squareplus: {
        const float z = y * K_ACT;
        derivative = z * z / (z * z + 1);
        break;
    }
    case Activation::Softplus:
        derivative = 1 - std::exp(-y * K_ACT);
        break;
    case Activation::Tanh:
        derivative = 1 - y * y;
        break;
    case Activation::None:
        break;
    }
    return T(float(val) * float(T(derivative)));
}

// Vector wrappers preserve the public template arguments and result layout.
template <typename T, uint32_t N, size_t A = sizeof(T)>
tvec<T, N, A> vec_activation(Activation act, const tvec<T, N, A>& v) {
    tvec<T, N, A> result;
    for (uint32_t i = 0; i < N; ++i)
        result[i] = activation(act, v[i]);
    return result;
}

template <Activation act, typename T, uint32_t N, size_t A = sizeof(T)>
tvec<T, N, A> vec_activation(tvec<T, N, A>& v) {
    v = vec_activation(act, v);
    return v;
}

template <typename T, uint32_t N, size_t A = sizeof(T)>
tvec<T, N, A> vec_activation_backward_in(Activation act, const tvec<T, N, A>& v,
                                         const tvec<T, N, A>& forward_v_in) {
    tvec<T, N, A> result;
    for (uint32_t i = 0; i < N; ++i)
        result[i] = activation_backward_in(act, v[i], forward_v_in[i]);
    return result;
}

template <Activation act, typename T, uint32_t N, size_t A = sizeof(T)>
tvec<T, N, A> vec_activation_backward_in(tvec<T, N, A>& v, const tvec<T, N, A>& forward_v_in) {
    v = vec_activation_backward_in(act, v, forward_v_in);
    return v;
}

template <typename T, uint32_t N, size_t A = sizeof(T)>
tvec<T, N, A> vec_activation_backward(Activation act, const tvec<T, N, A>& v,
                                      const tvec<T, N, A>& forward_v) {
    tvec<T, N, A> result;
    for (uint32_t i = 0; i < N; ++i)
        result[i] = activation_backward(act, v[i], forward_v[i]);
    return result;
}
} // namespace tcnn
