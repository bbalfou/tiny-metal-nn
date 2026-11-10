#include <tiny-metal-nn/networks/gemm_mlp.h>
#include <tiny-metal-nn/networks/fully_fused_mlp.h>
#include <tiny-metal-nn/detail/kernels/networks/gemm_mlp.h>
#include <tiny-metal-nn/detail/shaders.h>
#include <tiny-metal-nn/detail/trainer_dispatch.h>
#include <tiny-metal-nn/detail/network_selection.h>
#include <tiny-metal-nn/metal.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <mutex>

namespace tcnn {
namespace detail {
using namespace network_kernels;

// Translate a strided C++ matrix to the fixed-width shader descriptor.
template <class T> Mat describe(const GPUMatrixDynamic<T>& m) {
    const auto v = m.view();
    Mat result{m.m(), m.n(), v.stride_i, v.stride_j, sizeof(T)};
    nn_check(m.n_elements() <= UINT32_MAX, "matrix dispatch exceeds 32-bit indexing");
    if (m.m() && m.n())
        nn_check(size_t(m.m() - 1) * v.stride_i + size_t(m.n() - 1) * v.stride_j < UINT32_MAX,
                 "matrix span exceeds 32-bit indexing");
    return result;
}

size_t at(Mat m, uint32_t r, uint32_t c) {
    return size_t(r) * m.rs + size_t(c) * m.cs;
}

// Resolve only registered backend allocations, preserving retained command ownership.
template <class T>
metalrt::Argument matrix_arg(uint32_t index, const GPUMatrixDynamic<T>& m, metalrt::Access access) {
    const auto a = describe(m);
    size_t count = a.rows && a.cols ? at(a, a.rows - 1, a.cols - 1) + 1 : 0;
    return metalrt::Argument::buffer(
        index, metalrt::resolve_pointer(m.data(), metalrt::checked_mul(count, sizeof(T))), access);
}

template <class T>
metalrt::Argument raw_arg(uint32_t index, const T* p, size_t count, metalrt::Access access) {
    return metalrt::Argument::buffer(
        index, metalrt::resolve_pointer(p, metalrt::checked_mul(count, sizeof(T))), access);
}

float read(const void* p, size_t i, uint32_t precision) {
    return precision == 2 ? float(static_cast<const half*>(p)[i]) : static_cast<const float*>(p)[i];
}

void write(void* p, size_t i, uint32_t precision, float value) {
    if (precision == 2)
        static_cast<half*>(p)[i] = half(value);
    else
        static_cast<float*>(p)[i] = value;
}

// Emulate the reference's explicit T casts even though scalar arithmetic uses float.
float round_to_precision(float value, uint32_t precision) {
    return precision == 2 ? float(half(value)) : value;
}

float activate(float x, uint32_t act, uint32_t precision) {
    switch (static_cast<Activation>(act)) {
    case Activation::ReLU:
        return std::max(x, 0.0f);
    case Activation::LeakyReLU:
        return x * round_to_precision(x > 0 ? 1.0f : 0.01f, precision);
    case Activation::SiLU:
        return x * (1 / (1 + std::exp(-x)));
    case Activation::Exponential:
        return std::exp(x);
    case Activation::Sine:
        return std::sin(x);
    case Activation::Sigmoid:
        return 1 / (1 + std::exp(-x));
    case Activation::Squareplus: {
        float t = x * 10;
        return 0.5f * (t + std::sqrt(t * t + 4)) / 10;
    }
    case Activation::Softplus:
        return std::log(std::exp(x * 10) + 1) / 10;
    case Activation::Tanh:
        return std::tanh(x);
    default:
        return x;
    }
}

float derivative(float x, uint32_t act) {
    switch (static_cast<Activation>(act)) {
    case Activation::ReLU:
        return x > 0 ? 1 : 0;
    case Activation::LeakyReLU:
        return x > 0 ? 1 : 0.01f;
    case Activation::SiLU: {
        float s = 1 / (1 + std::exp(-x));
        return s + x * (s * (1 - s));
    }
    case Activation::Exponential:
        return std::exp(x);
    case Activation::Sine:
        return std::cos(x);
    case Activation::Sigmoid: {
        float s = 1 / (1 + std::exp(-x));
        return s * (1 - s);
    }
    case Activation::Squareplus: {
        float t = x * 10;
        float y = 0.5f * (t + std::sqrt(t * t + 4));
        return y * y / (y * y + 1);
    }
    case Activation::Softplus: {
        float t = std::exp(x * 10);
        return t / (t + 1);
    }
    case Activation::Tanh: {
        float t = std::tanh(x);
        return 1 - t * t;
    }
    default:
        return 1;
    }
}

float derivative_output(float y, uint32_t act, uint32_t precision) {
    switch (static_cast<Activation>(act)) {
    case Activation::ReLU:
        return y > 0 ? 1 : 0;
    case Activation::LeakyReLU:
        return y > 0 ? 1 : 0.01f;
    case Activation::Exponential:
        return y;
    case Activation::Sigmoid:
        return y * round_to_precision(1 - y, precision);
    case Activation::Squareplus: {
        float z = 10 * y;
        return z * z / (z * z + 1);
    }
    case Activation::Softplus:
        return 1 - std::exp(-10 * y);
    case Activation::Tanh:
        return 1 - y * y;
    default:
        return 1;
    }
}

// Activation transfer rounds the derivative to T before multiplying by
// the incoming T gradient. A postactivation sigmoid has an additional inner cast.
float activation_gradient(float value, float gradient, uint32_t act, uint32_t precision,
                          bool values_are_output) {
    if (static_cast<Activation>(act) == Activation::None)
        return gradient;
    const float factor =
        values_are_output ? derivative_output(value, act, precision) : derivative(value, act);
    return gradient * round_to_precision(factor, precision);
}

// Second derivatives are evaluated from the saved input and rounded output.
// Piecewise-linear activations use zero curvature away from their boundary.
float second_derivative(float x, float y, uint32_t act, uint32_t precision) {
    switch (static_cast<Activation>(act)) {
    case Activation::SiLU: {
        float s = 1 / (1 + std::exp(-x));
        return s * (1 - s) * (2 + x * (1 - 2 * s));
    }
    case Activation::Exponential:
        return y;
    case Activation::Sine:
        return -std::sin(x);
    case Activation::Sigmoid:
        return derivative_output(y, act, precision) * (1 - 2 * y);
    case Activation::Squareplus: {
        float z = 10 * y, d = z * z + 1;
        return 20 * z * derivative_output(y, act, precision) / (d * d);
    }
    case Activation::Softplus: {
        float e = std::exp(-10 * y);
        return 10 * e * (1 - e);
    }
    case Activation::Tanh:
        return -2 * y * (1 - y * y);
    default:
        return 0;
    }
}

// The reference backend executes these independent scalar equivalents on its queue.
metalrt::CpuKernel cpu_kernel(const std::string& name) {
    return [name](const metalrt::ComputeCommand& c) {
        if (name == "gemm_dense") {
            const auto a = c.scalar<Dense>(4);
            const auto* x = c.buffer<const uint8_t>(0);
            const auto* w = c.buffer<const uint8_t>(1);
            auto* pre = c.buffer<uint8_t>(2);
            auto* y = c.buffer<uint8_t>(3);
            for (uint32_t col = 0; col < a.y.cols; ++col)
                for (uint32_t row = 0; row < a.outputs; ++row) {
                    float sum = 0;
                    for (uint32_t k = 0; k < a.inputs; ++k)
                        sum += read(w, size_t(row) * a.inputs + k, a.y.precision) *
                               read(x, at(a.x, k, col), a.x.precision);
                    if (a.pre.precision == 2)
                        sum = float(half(sum));
                    write(pre, at(a.pre, row, col), a.pre.precision, sum);
                    write(y, at(a.y, row, col), a.y.precision,
                          activate(sum, a.activation, a.y.precision));
                }
        } else if (name == "gemm_delta") {
            const auto a = c.scalar<Delta>(3);
            const auto* values = c.buffer<const uint8_t>(0);
            const auto* g = c.buffer<const uint8_t>(1);
            auto* delta = c.buffer<uint8_t>(2);
            for (uint32_t col = 0; col < a.delta.cols; ++col)
                for (uint32_t row = 0; row < a.delta.rows; ++row)
                    write(delta, at(a.delta, row, col), a.delta.precision,
                          activation_gradient(
                              read(values, at(a.values, row, col), a.values.precision),
                              read(g, at(a.gradient, row, col), a.gradient.precision), a.activation,
                              a.delta.precision, a.values_are_output != 0));
        } else if (name == "gemm_second_delta") {
            const auto a = c.scalar<SecondDelta>(7);
            for (uint32_t col = 0; col < a.delta.cols; ++col)
                for (uint32_t row = 0; row < a.delta.rows; ++row) {
                    const float x = read(c.reference_buffer(0), at(a.pre, row, col),
                                         a.pre.precision),
                                y = read(c.reference_buffer(1), at(a.output, row, col),
                                         a.output.precision);
                    const bool input_form = a.activation == uint32_t(Activation::Sine) ||
                                            a.activation == uint32_t(Activation::SiLU);
                    const float first = round_to_precision(
                        input_form ? derivative(x, a.activation)
                                   : derivative_output(y, a.activation, a.delta.precision),
                        a.delta.precision);
                    const float second =
                        round_to_precision(second_derivative(x, y, a.activation, a.delta.precision),
                                           a.delta.precision);
                    const float tangent =
                        read(c.reference_buffer(2), at(a.tangent, row, col), a.tangent.precision);
                    const float adjoint = read(c.reference_buffer(3), at(a.adjoint, row, col),
                                               a.adjoint.precision),
                                tangent_adjoint =
                                    read(c.reference_buffer(4), at(a.tangent_adjoint, row, col),
                                         a.tangent_adjoint.precision);
                    write(c.reference_buffer(5), at(a.delta, row, col), a.delta.precision,
                          adjoint * first + tangent_adjoint * second * tangent);
                    write(c.reference_buffer(6), at(a.tangent_delta, row, col),
                          a.tangent_delta.precision, tangent_adjoint * first);
                }
        } else if (name == "gemm_dense_dx") {
            const auto a = c.scalar<InputGradient>(3);
            const auto* d = c.buffer<const uint8_t>(0);
            const auto* w = c.buffer<const uint8_t>(1);
            auto* dx = c.buffer<uint8_t>(2);
            for (uint32_t col = 0; col < a.dx.cols; ++col)
                for (uint32_t row = 0; row < a.inputs; ++row) {
                    float sum = 0;
                    for (uint32_t k = 0; k < a.outputs; ++k)
                        sum += read(w, size_t(k) * a.inputs + row, a.precision) *
                               read(d, at(a.delta, k, col), a.delta.precision);
                    write(dx, at(a.dx, row, col), a.dx.precision, sum);
                }
        } else if (name == "gemm_dense_dw") {
            const auto a = c.scalar<WeightGradient>(3);
            const auto* x = c.buffer<const uint8_t>(0);
            const auto* d = c.buffer<const uint8_t>(1);
            auto* dw = c.buffer<uint8_t>(2);
            for (uint32_t row = 0; row < a.outputs; ++row)
                for (uint32_t k = 0; k < a.inputs; ++k) {
                    float sum = 0;
                    for (uint32_t col = 0; col < a.x.cols; ++col)
                        sum += read(d, at(a.delta, row, col), a.delta.precision) *
                               read(x, at(a.x, k, col), a.x.precision);
                    size_t i = size_t(row) * a.inputs + k;
                    if (a.mode == 2)
                        sum += read(dw, i, a.precision);
                    write(dw, i, a.precision, sum);
                }
        } else if (name == "nn_copy" || name == "nn_accumulate") {
            const auto a = c.scalar<Copy>(2);
            const auto* src = c.buffer<const uint8_t>(0);
            auto* dst = c.buffer<uint8_t>(1);
            for (uint32_t col = 0; col < a.dst.cols; ++col)
                for (uint32_t row = 0; row < a.dst.rows; ++row)
                    write(dst, at(a.dst, row, col), a.dst.precision,
                          read(src, at(a.src, row, col), a.src.precision) +
                              (name == "nn_accumulate"
                                   ? read(dst, at(a.dst, row, col), a.dst.precision)
                                   : 0.0f));
        } else if (name == "nn_unary") {
            const auto a = c.scalar<Unary>(1);
            auto* v = c.buffer<uint8_t>(0);
            for (uint32_t i = 0; i < a.count; ++i)
                write(v, i, a.precision,
                      a.mode == 0 ? read(v, i, a.precision) * a.value
                                  : (i % a.width == a.dim ? a.value : 0));
        } else if (name == "nn_matrix_activation") {
            const auto a = c.scalar<MatrixActivation>(3);
            const auto* v = c.buffer<const uint8_t>(0);
            const auto* g = c.buffer<const uint8_t>(1);
            auto* out = c.buffer<uint8_t>(2);
            for (uint32_t col = 0; col < a.result.cols; ++col)
                for (uint32_t row = 0; row < a.result.rows; ++row) {
                    const float x = read(v, at(a.values, row, col), a.values.precision);
                    write(out, at(a.result, row, col), a.result.precision,
                          a.mode == 0
                              ? activate(x, a.activation, a.result.precision)
                              : activation_gradient(
                                    x, read(g, at(a.gradient, row, col), a.gradient.precision),
                                    a.activation, a.result.precision, a.mode == 2));
                }
        } else if (name == "nn_activation") {
            const auto a = c.scalar<ActivationArgs>(3);
            const auto* v = c.buffer<const uint8_t>(0);
            const auto* g = c.buffer<const uint8_t>(1);
            auto* out = c.buffer<uint8_t>(2);
            for (uint32_t i = 0; i < a.count; ++i) {
                const float x = read(v, i, a.precision);
                write(out, i, a.precision,
                      a.mode == 0 ? activate(x, a.activation, a.precision)
                                  : activation_gradient(x, read(g, i, a.precision), a.activation,
                                                        a.precision, a.mode == 2));
            }
        } else if (name == "nn_add_perturbation") {
            const auto a = c.scalar<ActivationArgs>(3);
            const auto* input = c.buffer<const uint8_t>(0);
            const auto* perturbation = c.buffer<const float>(1);
            auto* output = c.buffer<uint8_t>(2);
            for (uint32_t i = 0; i < a.count; ++i)
                write(output, i, a.precision, read(input, i, a.precision) + perturbation[i]);
        } else if (name == "nn_extract") {
            const auto a = c.scalar<Extract>(2);
            const auto* src = c.buffer<const uint8_t>(0);
            auto* out = c.buffer<float>(1);
            uint32_t n = a.count / a.fan_out;
            for (uint32_t i = 0; i < a.count; ++i) {
                uint32_t row = i % a.fan_out, col = i / a.fan_out;
                float value = read(
                    src, a.layout == 1 ? size_t(col) * a.fan_in + a.dim : size_t(a.dim) * n + col,
                    a.precision);
                out[i] = a.fan_out == 1 ? value
                                        : (row == 0 ? std::max(-value, 0.0f)
                                                    : (row == 1 ? std::max(value, 0.0f)
                                                                : (row == 2 ? 0.0f : 1.0f)));
            }
        } else
            throw std::logic_error("Unknown NN kernel: " + name);
    };
}

// Compile each named MSL pipeline once for each backend device identity.
metalrt::Kernel kernel_for(metalStream_t stream, const std::string& name) {
    struct Cache {
        metalrt::Device device;
        std::map<std::string, metalrt::Kernel> kernels;
    };

    static std::mutex mutex;
    static std::vector<Cache> caches;
    auto device = resolve_stream(stream)->device();
    std::lock_guard<std::mutex> lock{mutex};
    auto it = std::find_if(caches.begin(), caches.end(),
                           [&](const auto& cache) { return cache.device == device; });
    if (it == caches.end()) {
        caches.push_back({device, {}});
        it = std::prev(caches.end());
    }
    auto kernel = it->kernels.find(name);
    if (kernel == it->kernels.end())
        kernel = it->kernels
                     .emplace(name, device.kernel(shader_source(name.rfind("gemm_", 0) == 0
                                                                    ? GemmMLPTraits::source
                                                                    : "networks.metal"),
                                                  name, cpu_kernel(name)))
                     .first;
    return kernel->second;
}

void dispatch(metalStream_t stream, const std::string& name, size_t count,
              std::vector<metalrt::Argument> args) {
    nn_check(count <= UINT32_MAX, "NN dispatch exceeds 32-bit indexing");
    if (count)
        metalrt::launch_linear(resolve_stream(stream), kernel_for(stream, name), count,
                               std::move(args));
}

template <class IN, class OUT>
void copy_matrix(metalStream_t stream, const GPUMatrixDynamic<IN>& src,
                 GPUMatrixDynamic<OUT>& dst) {
    nn_check(src.m() >= dst.m() && src.n() == dst.n(), "matrix copy/trim shape mismatch");
    if (!dst.n_elements())
        return;
    auto sa = matrix_arg(0, src, metalrt::Access::Read),
         da = matrix_arg(1, dst, metalrt::Access::Write);
    if (sa.resource == da.resource && sizeof(IN) == sizeof(OUT) &&
        src.view().stride_i == dst.view().stride_i && src.view().stride_j == dst.view().stride_j)
        return;
    nn_check(sa.resource.buffer != da.resource.buffer ||
                 sa.resource.offset + sa.resource.length <= da.resource.offset ||
                 da.resource.offset + da.resource.length <= sa.resource.offset,
             "overlapping matrix conversion is not supported");
    dispatch(stream, "nn_copy", dst.n_elements(),
             {sa, da, metalrt::Argument::scalar(2, Copy{describe(src), describe(dst)})});
}

template <class T>
void dense_forward(metalStream_t stream, const GPUMatrixDynamic<T>& input,
                   const GPUMatrix<T, RM>& weights, GPUMatrixDynamic<T>& preactivation,
                   GPUMatrixDynamic<T>& output, Activation activation) {
    if (!output.n_elements())
        return;
    dispatch(stream, "gemm_dense", output.n_elements(),
             {matrix_arg(0, input, metalrt::Access::Read),
              matrix_arg(1, weights, metalrt::Access::Read),
              matrix_arg(2, preactivation, metalrt::Access::Write),
              matrix_arg(3, output, metalrt::Access::Write),
              metalrt::Argument::scalar(4, Dense{describe(input), describe(preactivation),
                                                 describe(output), weights.n(), weights.m(),
                                                 uint32_t(activation)})});
}

template <class T>
void dense_delta(metalStream_t stream, const GPUMatrixDynamic<T>& values,
                 const GPUMatrixDynamic<T>& dL_doutput, GPUMatrixDynamic<T>& delta,
                 Activation activation, bool values_are_output) {
    nn_check(values.m() == delta.m() && values.n() == delta.n() && dL_doutput.m() == delta.m() &&
                 dL_doutput.n() == delta.n(),
             "activation derivative shape mismatch");
    nn_check(!values_are_output ||
                 (activation != Activation::Sine && activation != Activation::SiLU),
             "activation output cannot determine its derivative; use saved preactivations");
    if (!delta.n_elements())
        return;
    dispatch(
        stream, "gemm_delta", delta.n_elements(),
        {matrix_arg(0, values, metalrt::Access::Read),
         matrix_arg(1, dL_doutput, metalrt::Access::Read),
         matrix_arg(2, delta, metalrt::Access::Write),
         metalrt::Argument::scalar(3, Delta{describe(values), describe(dL_doutput), describe(delta),
                                            uint32_t(activation), values_are_output ? 1u : 0u})});
}

template <class T>
void dense_second_delta(metalStream_t stream, const GPUMatrixDynamic<T>& preactivation,
                        const GPUMatrixDynamic<T>& output,
                        const GPUMatrixDynamic<T>& tangent_preactivation,
                        const GPUMatrixDynamic<T>& adjoint,
                        const GPUMatrixDynamic<T>& tangent_adjoint, GPUMatrixDynamic<T>& delta,
                        GPUMatrixDynamic<T>& tangent_delta, Activation activation) {
    if (!delta.n_elements())
        return;
    dispatch(stream, "gemm_second_delta", delta.n_elements(),
             {matrix_arg(0, preactivation, metalrt::Access::Read),
              matrix_arg(1, output, metalrt::Access::Read),
              matrix_arg(2, tangent_preactivation, metalrt::Access::Read),
              matrix_arg(3, adjoint, metalrt::Access::Read),
              matrix_arg(4, tangent_adjoint, metalrt::Access::Read),
              matrix_arg(5, delta, metalrt::Access::Write),
              matrix_arg(6, tangent_delta, metalrt::Access::Write),
              metalrt::Argument::scalar(7, SecondDelta{describe(preactivation), describe(output),
                                                       describe(tangent_preactivation),
                                                       describe(adjoint), describe(tangent_adjoint),
                                                       describe(delta), describe(tangent_delta),
                                                       uint32_t(activation)})});
}

template <class T>
void accumulate_matrix(metalStream_t stream, const GPUMatrixDynamic<T>& source,
                       GPUMatrixDynamic<T>& destination) {
    nn_check(source.m() == destination.m() && source.n() == destination.n(),
             "matrix accumulation shape mismatch");
    if (!destination.n_elements())
        return;
    dispatch(stream, "nn_accumulate", destination.n_elements(),
             {matrix_arg(0, source, metalrt::Access::Read),
              matrix_arg(1, destination, metalrt::Access::ReadWrite),
              metalrt::Argument::scalar(2, Copy{describe(source), describe(destination)})});
}

template <class T>
void activation_matrix(metalStream_t stream, const GPUMatrixDynamic<T>& values,
                       const GPUMatrixDynamic<T>& gradient, GPUMatrixDynamic<T>& result,
                       Activation activation, uint32_t mode) {
    nn_check(values.m() == result.m() && values.n() == result.n() && gradient.m() == result.m() &&
                 gradient.n() == result.n(),
             "activation matrix shape mismatch");
    nn_check(mode <= 2, "invalid activation operation");
    if (!result.n_elements())
        return;
    auto input = matrix_arg(0, values, metalrt::Access::Read),
         incoming = matrix_arg(1, gradient, metalrt::Access::Read),
         output = matrix_arg(2, result, metalrt::Access::Write);
    auto safe_alias = [&](const metalrt::Argument& source, const GPUMatrixDynamic<T>& view) {
        const auto& a = source.resource;
        const auto& b = output.resource;
        return a.buffer != b.buffer || a.offset + a.length <= b.offset ||
               b.offset + b.length <= a.offset ||
               (a == b && view.view().stride_i == result.view().stride_i &&
                view.view().stride_j == result.view().stride_j);
    };
    nn_check(safe_alias(input, values) && (mode == 0 || safe_alias(incoming, gradient)),
             "partially overlapping activation matrices are not supported");
    dispatch(stream, "nn_matrix_activation", result.n_elements(),
             {input, incoming, output,
              metalrt::Argument::scalar(3, MatrixActivation{describe(values), describe(gradient),
                                                            describe(result), uint32_t(activation),
                                                            mode})});
}

template <class T>
void dense_input_gradient(metalStream_t stream, const GPUMatrixDynamic<T>& delta,
                          const GPUMatrix<T, RM>& weights, GPUMatrixDynamic<T>& dL_dinput) {
    if (!dL_dinput.n_elements())
        return;
    dispatch(stream, "gemm_dense_dx", dL_dinput.n_elements(),
             {matrix_arg(0, delta, metalrt::Access::Read),
              matrix_arg(1, weights, metalrt::Access::Read),
              matrix_arg(2, dL_dinput, metalrt::Access::Write),
              metalrt::Argument::scalar(3, InputGradient{describe(delta), describe(dL_dinput),
                                                         weights.n(), weights.m(), sizeof(T)})});
}

template <class T>
void dense_weight_gradient(metalStream_t stream, const GPUMatrixDynamic<T>& input,
                           const GPUMatrixDynamic<T>& delta, GPUMatrix<T, RM>& gradients,
                           GradientMode param_gradients_mode) {
    if (!input.n()) {
        if (param_gradients_mode == GradientMode::Overwrite)
            resolve_stream(stream)->fill(
                metalrt::resolve_pointer(gradients.data(), gradients.n_bytes()), 0);
        return;
    }
    dispatch(
        stream, "gemm_dense_dw", gradients.n_elements(),
        {matrix_arg(0, input, metalrt::Access::Read), matrix_arg(1, delta, metalrt::Access::Read),
         matrix_arg(2, gradients,
                    param_gradients_mode == GradientMode::Accumulate ? metalrt::Access::ReadWrite
                                                                     : metalrt::Access::Write),
         metalrt::Argument::scalar(3, WeightGradient{describe(input), describe(delta),
                                                     gradients.n(), gradients.m(), sizeof(T),
                                                     uint32_t(param_gradients_mode)})});
}

#define TCNN_INSTANTIATE_DISPATCH(T)                                                               \
    template void dense_forward<T>(metalStream_t, const GPUMatrixDynamic<T>&,                      \
                                   const GPUMatrix<T, RM>&, GPUMatrixDynamic<T>&,                  \
                                   GPUMatrixDynamic<T>&, Activation);                              \
    template void dense_delta<T>(metalStream_t, const GPUMatrixDynamic<T>&,                        \
                                 const GPUMatrixDynamic<T>&, GPUMatrixDynamic<T>&, Activation,     \
                                 bool);                                                            \
    template void dense_second_delta<T>(metalStream_t, const GPUMatrixDynamic<T>&,                 \
                                        const GPUMatrixDynamic<T>&, const GPUMatrixDynamic<T>&,    \
                                        const GPUMatrixDynamic<T>&, const GPUMatrixDynamic<T>&,    \
                                        GPUMatrixDynamic<T>&, GPUMatrixDynamic<T>&, Activation);   \
    template void accumulate_matrix<T>(metalStream_t, const GPUMatrixDynamic<T>&,                  \
                                       GPUMatrixDynamic<T>&);                                      \
    template void activation_matrix<T>(metalStream_t, const GPUMatrixDynamic<T>&,                  \
                                       const GPUMatrixDynamic<T>&, GPUMatrixDynamic<T>&,           \
                                       Activation, uint32_t);                                      \
    template void dense_input_gradient<T>(metalStream_t, const GPUMatrixDynamic<T>&,               \
                                          const GPUMatrix<T, RM>&, GPUMatrixDynamic<T>&);          \
    template void dense_weight_gradient<T>(metalStream_t, const GPUMatrixDynamic<T>&,              \
                                           const GPUMatrixDynamic<T>&, GPUMatrix<T, RM>&,          \
                                           GradientMode);
TCNN_INSTANTIATE_DISPATCH(float)
TCNN_INSTANTIATE_DISPATCH(half)
#undef TCNN_INSTANTIATE_DISPATCH
template void copy_matrix<float, float>(metalStream_t, const GPUMatrixDynamic<float>&,
                                        GPUMatrixDynamic<float>&);
template void copy_matrix<half, float>(metalStream_t, const GPUMatrixDynamic<half>&,
                                       GPUMatrixDynamic<float>&);
template void copy_matrix<float, half>(metalStream_t, const GPUMatrixDynamic<float>&,
                                       GPUMatrixDynamic<half>&);
template void copy_matrix<half, half>(metalStream_t, const GPUMatrixDynamic<half>&,
                                      GPUMatrixDynamic<half>&);

// Training perturbation and reduction share the same cache and scalar ABI.
template <class T>
void add_perturbation(metalStream_t stream, const GPUMatrix<T>& input,
                      const GPUMatrix<float>& perturbation, GPUMatrix<T>& output) {
    if (!output.n_elements())
        return;
    dispatch(stream, "nn_add_perturbation", output.n_elements(),
             {matrix_arg(0, input, metalrt::Access::Read),
              matrix_arg(1, perturbation, metalrt::Access::Read),
              matrix_arg(2, output, metalrt::Access::Write),
              metalrt::Argument::scalar(
                  3, ActivationArgs{static_cast<uint32_t>(output.n_elements()), sizeof(T), 0, 0})});
}

template void add_perturbation<float>(metalStream_t, const GPUMatrix<float>&,
                                      const GPUMatrix<float>&, GPUMatrix<float>&);
template void add_perturbation<half>(metalStream_t, const GPUMatrix<half>&, const GPUMatrix<float>&,
                                     GPUMatrix<half>&);

} // namespace detail

// Public tensor helpers retain the source's argument ordering and names.
template <class T>
void one_hot_batched(metalStream_t stream, const uint32_t num_elements, const uint32_t width,
                     const uint32_t one_hot_dim, T* out, float scale) {
    nn_check(width && one_hot_dim < width, "invalid one-hot dimension");
    if (!num_elements)
        return;
    detail::dispatch(stream, "nn_unary", num_elements,
                     {detail::raw_arg(0, out, num_elements, metalrt::Access::Write),
                      metalrt::Argument::scalar(1, detail::Unary{num_elements, width, one_hot_dim,
                                                                 1, sizeof(T), scale})});
}

template <class T>
void mult(metalStream_t stream, const uint32_t num_elements, T* inout, float factor) {
    if (!num_elements)
        return;
    detail::dispatch(
        stream, "nn_unary", num_elements,
        {detail::raw_arg(0, inout, num_elements, metalrt::Access::ReadWrite),
         metalrt::Argument::scalar(1, detail::Unary{num_elements, 0, 0, 0, sizeof(T), factor})});
}

template <class T>
void trim_and_cast_from(metalStream_t stream, const MatrixLayout layout,
                        const uint32_t num_elements, const uint32_t input_width,
                        const uint32_t output_width, const T* in, float* out) {
    nn_check(output_width && input_width >= output_width && num_elements % output_width == 0,
             "invalid trim dimensions");
    GPUMatrixDynamic<T> input{const_cast<T*>(in), input_width, num_elements / output_width, layout};
    GPUMatrixDynamic<float> output{out, output_width, num_elements / output_width, layout};
    detail::copy_matrix(stream, input, output);
}

template <class T>
void activation_gpu(metalStream_t stream, const uint32_t num_elements, const Activation act,
                    const T* in, T* out) {
    nn_check(num_elements % (16 / sizeof(T)) == 0,
             "activation element count must be vector aligned");
    if (!num_elements || (act == Activation::None && in == out))
        return;
    detail::dispatch(stream, "nn_activation", num_elements,
                     {detail::raw_arg(0, in, num_elements, metalrt::Access::Read),
                      detail::raw_arg(1, in, num_elements, metalrt::Access::Read),
                      detail::raw_arg(2, out, num_elements, metalrt::Access::Write),
                      metalrt::Argument::scalar(
                          3, detail::ActivationArgs{num_elements, sizeof(T), uint32_t(act), 0})});
}

template <class T>
void activation_backward_gpu(metalStream_t stream, const uint32_t num_elements,
                             const Activation act, const T* values, const T* gradients_out,
                             T* gradients_in) {
    nn_check(num_elements % (16 / sizeof(T)) == 0,
             "activation element count must be vector aligned");
    if (!num_elements)
        return;
    detail::dispatch(stream, "nn_activation", num_elements,
                     {detail::raw_arg(0, values, num_elements, metalrt::Access::Read),
                      detail::raw_arg(1, gradients_out, num_elements, metalrt::Access::Read),
                      detail::raw_arg(2, gradients_in, num_elements, metalrt::Access::Write),
                      metalrt::Argument::scalar(
                          3, detail::ActivationArgs{num_elements, sizeof(T), uint32_t(act), 1})});
}

template <class T>
void activation_backward_output_gpu(metalStream_t stream, const uint32_t num_elements,
                                    const Activation act, const T* output_values,
                                    const T* gradients_out, T* gradients_in) {
    nn_check(act != Activation::Sine && act != Activation::SiLU,
             "activation output cannot determine its derivative; use activation_backward_gpu");
    nn_check(num_elements % (16 / sizeof(T)) == 0,
             "activation element count must be vector aligned");
    if (!num_elements)
        return;
    detail::dispatch(stream, "nn_activation", num_elements,
                     {detail::raw_arg(0, output_values, num_elements, metalrt::Access::Read),
                      detail::raw_arg(1, gradients_out, num_elements, metalrt::Access::Read),
                      detail::raw_arg(2, gradients_in, num_elements, metalrt::Access::Write),
                      metalrt::Argument::scalar(
                          3, detail::ActivationArgs{num_elements, sizeof(T), uint32_t(act), 2})});
}

template <class T>
void extract_dimension_pos_neg(metalStream_t stream, const uint32_t num_elements,
                               const uint32_t dim, const uint32_t fan_in, const uint32_t fan_out,
                               const T* encoded, MatrixLayout layout, float* output) {
    nn_check(fan_out && dim < fan_in && num_elements % fan_out == 0,
             "invalid activation visualization shape");
    if (!num_elements)
        return;
    // The shader indexes the source in uint even when its logical output is small.
    const size_t source_elements =
        metalrt::checked_mul(size_t(num_elements / fan_out), size_t(fan_in));
    if (source_elements > UINT32_MAX)
        throw std::overflow_error("activation visualization source span exceeds 32-bit indexing");
    detail::dispatch(
        stream, "nn_extract", num_elements,
        {detail::raw_arg(0, encoded, source_elements, metalrt::Access::Read),
         detail::raw_arg(1, output, num_elements, metalrt::Access::Write),
         metalrt::Argument::scalar(
             2, detail::Extract{num_elements, dim, fan_in, fan_out, uint32_t(layout), sizeof(T)})});
}

#define TCNN_INSTANTIATE_HELPERS(T)                                                                \
    template void one_hot_batched<T>(metalStream_t, uint32_t, uint32_t, uint32_t, T*, float);      \
    template void mult<T>(metalStream_t, uint32_t, T*, float);                                     \
    template void trim_and_cast_from<T>(metalStream_t, MatrixLayout, uint32_t, uint32_t, uint32_t, \
                                        const T*, float*);                                         \
    template void activation_gpu<T>(metalStream_t, uint32_t, Activation, const T*, T*);            \
    template void activation_backward_gpu<T>(metalStream_t, uint32_t, Activation, const T*,        \
                                             const T*, T*);                                        \
    template void activation_backward_output_gpu<T>(metalStream_t, uint32_t, Activation, const T*, \
                                                    const T*, T*);                                 \
    template void extract_dimension_pos_neg<T>(metalStream_t, uint32_t, uint32_t, uint32_t,        \
                                               uint32_t, const T*, MatrixLayout, float*);
TCNN_INSTANTIATE_HELPERS(float)
TCNN_INSTANTIATE_HELPERS(half)
#undef TCNN_INSTANTIATE_HELPERS

// Both storage precisions support native fusion on Apple GPU family 9 or later.
std::string detail::select_network_for_capabilities(const json& network,
                                                    const metalrt::Capabilities& capabilities) {
    const auto name = network.value("otype", std::string{"GemmMLP"});
    if (equals_case_insensitive(name, "GemmMLP"))
        return "GemmMLP";
    if (equals_case_insensitive(name, "FullyFusedMLP"))
        return fully_fused_mlp_supported(capabilities) ? "FullyFusedMLP" : "GemmMLP";
    throw std::invalid_argument("Invalid network type: " + name);
}

std::string select_network(const json& network) {
    const auto name = network.value("otype", std::string{"GemmMLP"});
    const bool request_fused = equals_case_insensitive(name, "FullyFusedMLP");
    const auto capabilities =
        request_fused ? metalrt::Device::current().capabilities() : metalrt::Capabilities{};
    if (request_fused && !detail::fully_fused_mlp_supported(capabilities))
        log_warning("FullyFusedMLP requires Metal Apple GPU family 9 or newer. "
                    "Falling back to GemmMLP for device '{}'.",
                    capabilities.name);
    return detail::select_network_for_capabilities(network, capabilities);
}

uint32_t minimum_alignment(const json& network) {
    if (select_network(network) == "FullyFusedMLP") {
        const auto width = network.value("n_neurons", 128u);
        nn_check(detail::network_kernels::FullyFusedMLPTraits::valid_width(width),
                 "FullyFusedMLP only supports 16, 32, 64, and 128 neurons; use GemmMLP for "
                 "other widths");
    }
    return 16;
}

std::vector<std::string> builtin_networks() {
    return {"FullyFusedMLP", "GemmMLP"};
}

template <class T> Network<T, T>* create_network(const json& network) {
    const auto name = select_network(network);
    const uint32_t input = network.at("n_input_dims"), output = network.at("n_output_dims"),
                   width = network.value("n_neurons", 128u),
                   hidden = network.value("n_hidden_layers", 5u);
    const auto activation = string_to_activation(network.value("activation", std::string{"ReLU"})),
               output_activation =
                   string_to_activation(network.value("output_activation", std::string{"None"}));
    if (name == "GemmMLP")
        return new GemmMLP<T>{input, width, output, hidden, activation, output_activation};
    switch (width) {
    case 16:
        return new FullyFusedMLP<T, 16>{input, output, hidden, activation, output_activation};
    case 32:
        return new FullyFusedMLP<T, 32>{input, output, hidden, activation, output_activation};
    case 64:
        return new FullyFusedMLP<T, 64>{input, output, hidden, activation, output_activation};
    case 128:
        return new FullyFusedMLP<T, 128>{input, output, hidden, activation, output_activation};
    default:
        throw std::invalid_argument("FullyFusedMLP only supports 16, 32, 64, and 128 neurons; "
                                    "use GemmMLP for other widths");
    }
}

template Network<float, float>* create_network<float>(const json&);
template Network<half, half>* create_network<half>(const json&);
} // namespace tcnn
