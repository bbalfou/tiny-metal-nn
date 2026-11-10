#include <tiny-metal-nn/loss.h>
#include <tiny-metal-nn/losses/l2.h>
#include <tiny-metal-nn/losses/relative_l2.h>
#include <tiny-metal-nn/losses/relative_l2_luminance.h>
#include <tiny-metal-nn/losses/l1.h>
#include <tiny-metal-nn/losses/relative_l1.h>
#include <tiny-metal-nn/losses/mape.h>
#include <tiny-metal-nn/losses/smape.h>
#include <tiny-metal-nn/losses/cross_entropy.h>
#include <tiny-metal-nn/losses/variance_is.h>
#include <tiny-metal-nn/detail/shaders.h>
#include <tiny-metal-nn/detail/kernels/losses/traits.h>
#include <tiny-metal-nn/metal.h>
#include <metalrt/runtime.h>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <limits>
#include <iomanip>
#include <locale>
#include <map>
#include <mutex>
#include <sstream>

namespace tcnn {
namespace detail {
namespace {
using loss_kernels::LossArgs;
using loss_kernels::LossMatrix;
using LossBindings = loss_kernels::Bindings;

// Converts a view into a compact, layout-independent kernel descriptor.
template <typename T> LossMatrix loss_matrix(const GPUMatrix<T>& matrix) {
    const auto view = matrix.view();
    if (matrix.n_elements() > UINT32_MAX)
        throw std::overflow_error("loss dispatch exceeds 32-bit indexing");
    // Logical element count alone does not bound a strided view's uint shader address.
    if (matrix.m() && matrix.n()) {
        const uint64_t span =
            uint64_t(matrix.m() - 1) * view.stride_i + uint64_t(matrix.n() - 1) * view.stride_j + 1;
        if (span > UINT32_MAX)
            throw std::overflow_error("loss matrix span exceeds 32-bit indexing");
    }
    return {matrix.m(), matrix.n(), view.stride_i, view.stride_j};
}

// Maps a logical row and sample to the underlying matrix span.
size_t loss_index(const LossMatrix& matrix, uint32_t row, uint32_t col) {
    return size_t(row) * matrix.stride_i + size_t(col) * matrix.stride_j;
}

// Binds the entire accessed span, retaining the allocation for asynchronous execution.
template <typename T>
metalrt::Argument loss_argument(uint32_t index, const GPUMatrix<T>& matrix,
                                metalrt::Access access) {
    const auto desc = loss_matrix(matrix);
    const size_t count =
        desc.rows && desc.cols ? loss_index(desc, desc.rows - 1, desc.cols - 1) + 1 : 0;
    return metalrt::Argument::buffer(
        index, metalrt::resolve_pointer(matrix.data(), metalrt::checked_mul(count, sizeof(T))),
        access);
}

// Refuses destructive output aliases before enqueuing any work.
void loss_disjoint(const metalrt::Argument& lhs, const metalrt::Argument& rhs) {
    const auto& a = lhs.resource;
    const auto& b = rhs.resource;
    if (a.buffer == b.buffer && a.offset < b.offset + b.length && b.offset < a.offset + a.length)
        throw std::invalid_argument("loss inputs and outputs must not overlap");
}

// The scalar callback is used only by the explicitly selected Reference backend.
template <typename T> metalrt::CpuKernel loss_reference(LossKind kind) {
    return [kind](const metalrt::ComputeCommand& command) {
        const auto args = command.scalar<LossArgs>(LossBindings::arguments);
        const auto* prediction = command.buffer<const T>(LossBindings::prediction);
        const auto* target = command.buffer<const float>(LossBindings::target);
        auto* values = command.buffer<float>(LossBindings::values);
        auto* gradients = command.buffer<T>(LossBindings::gradients);
        const auto* pdf_data = command.buffer<const float>(LossBindings::pdf);
        const auto* constant_gradient =
            command.buffer<const float>(LossBindings::constant_gradient);
        const float n_total = float(args.target.rows) * args.target.cols;
        for (uint32_t col = 0; col < args.prediction.cols; ++col) {
            for (uint32_t row = 0; row < args.prediction.rows; ++row) {
                float value = 0, gradient = 0;
                if (row < args.target.rows) {
                    const float p = float(prediction[loss_index(args.prediction, row, col)]);
                    const float t = target[loss_index(args.target, row, col)];
                    const float pdf =
                        args.has_pdf ? pdf_data[loss_index(args.pdf, row, col)] : 1.0f;
                    const float difference = p - t;
                    float denominator = 1.0f;
                    if (kind == LossKind::RelativeL2)
                        denominator = p * p + 0.01f;
                    if (kind == LossKind::RelativeL2Luminance) {
                        float r = float(prediction[loss_index(args.prediction, 0, col)]);
                        float g = float(prediction[loss_index(args.prediction, 1, col)]);
                        float b = float(prediction[loss_index(args.prediction, 2, col)]);
                        if (args.target.rows >= 6) {
                            r += float(prediction[loss_index(args.prediction, 3, col)]);
                            g += float(prediction[loss_index(args.prediction, 4, col)]);
                            b += float(prediction[loss_index(args.prediction, 5, col)]);
                        }
                        const float luminance = 0.299f * r + 0.587f * g + 0.114f * b;
                        denominator = luminance * luminance + 0.01f;
                    }
                    if (kind == LossKind::L2 || kind == LossKind::RelativeL2 ||
                        kind == LossKind::RelativeL2Luminance) {
                        value = difference * difference / denominator / pdf / n_total;
                        gradient = 2.0f * difference / denominator / pdf / n_total;
                    } else if (kind == LossKind::CrossEntropy) {
                        const float factor = -t / pdf / n_total;
                        value = factor * std::log(p);
                        gradient = factor / p;
                    } else if (kind == LossKind::Variance) {
                        const float factor = t * t / pdf / n_total;
                        value = factor / p - factor / pdf;
                        gradient = -factor / (p * p);
                    } else if (kind == LossKind::Constant) {
                        gradient = constant_gradient[row] / pdf;
                    } else {
                        if (kind == LossKind::RelativeL1)
                            denominator = std::abs(p) + 0.01f;
                        if (kind == LossKind::Mape)
                            denominator = std::abs(t) + 0.01f;
                        if (kind == LossKind::Smape)
                            denominator = 0.5f * (std::abs(t) + std::abs(p)) + 0.01f;
                        const float scale = 1.0f / denominator / pdf;
                        value = std::abs(difference) * scale / n_total;
                        gradient = std::copysign(scale, difference) / n_total;
                    }
                }
                values[loss_index(args.values, row, col)] = value;
                gradients[loss_index(args.gradients, row, col)] = T(args.loss_scale * gradient);
            }
        }
    };
}

// Caches each objective/precision independently so replacing one shader cannot
// silently route a different objective through the same implementation.
template <typename T> metalrt::Kernel loss_kernel(metalrt::Stream& stream, LossKind kind) {
    struct Cached {
        metalrt::Device device;
        LossKind kind;
        metalrt::Kernel kernel;
    };

    static std::mutex mutex;
    static std::vector<Cached> cache;
    std::lock_guard<std::mutex> lock{mutex};
    for (const auto& item : cache)
        if (item.device == stream.device() && item.kind == kind)
            return item.kernel;
    const auto& traits = loss_kernels::kernel_descriptor(kind);
    const auto kernel =
        stream.device().kernel(shader_source(traits.source),
                               std::is_same<T, float>::value ? traits.entry_f32 : traits.entry_f16,
                               loss_reference<T>(kind));
    cache.push_back({stream.device(), kind, kernel});
    return kernel;
}
} // namespace

// Validates logical extents, snapshots constants, and dispatches without reading GPU inputs on the
// host.
template <typename T>
void evaluate_loss(metalStream_t stream, const float loss_scale, const GPUMatrix<T>& prediction,
                   const GPUMatrix<float>& target, GPUMatrix<float>& values,
                   GPUMatrix<T>& gradients, const GPUMatrix<float>* data_pdf, LossKind kind,
                   const std::vector<float>& constant_gradient) {
    if (prediction.n() != target.n() || target.m() > prediction.m() || target.m() == 0 ||
        values.m() != prediction.m() || values.n() != prediction.n() ||
        gradients.m() != prediction.m() || gradients.n() != prediction.n() ||
        (data_pdf && (data_pdf->m() != target.m() || data_pdf->n() != target.n())))
        throw std::invalid_argument("loss matrix shape mismatch");
    if (kind == LossKind::RelativeL2Luminance && prediction.m() < 3)
        throw std::invalid_argument("RelativeL2Luminance requires three prediction channels");
    if (kind == LossKind::Constant && constant_gradient.size() != target.m())
        throw std::invalid_argument("constant_gradient size must equal target dimensions");
    if (!prediction.n_elements())
        return;
    // Validate every descriptor before resolving pointers or recording a dispatch.
    LossArgs args{loss_matrix(prediction),
                  loss_matrix(target),
                  loss_matrix(values),
                  loss_matrix(gradients),
                  loss_matrix(data_pdf ? *data_pdf : target),
                  static_cast<uint32_t>(kind),
                  data_pdf ? 1u : 0u,
                  loss_scale};
    auto& runtime = *resolve_stream(stream);
    auto p = loss_argument(LossBindings::prediction, prediction, metalrt::Access::Read);
    auto t = loss_argument(LossBindings::target, target, metalrt::Access::Read);
    auto v = loss_argument(LossBindings::values, values, metalrt::Access::Write);
    auto g = loss_argument(LossBindings::gradients, gradients, metalrt::Access::Write);
    auto pdf =
        loss_argument(LossBindings::pdf, data_pdf ? *data_pdf : target, metalrt::Access::Read);
    loss_disjoint(p, v);
    loss_disjoint(p, g);
    loss_disjoint(t, v);
    loss_disjoint(t, g);
    loss_disjoint(pdf, v);
    loss_disjoint(pdf, g);
    loss_disjoint(v, g);
    auto constant = loss_argument(LossBindings::constant_gradient, target, metalrt::Access::Read);
    if (kind == LossKind::Constant) {
        auto snapshot = runtime.device().allocate(constant_gradient.size() * sizeof(float));
        std::memcpy(snapshot.contents(), constant_gradient.data(), snapshot.size());
        constant = metalrt::Argument::buffer(LossBindings::constant_gradient, snapshot.view(),
                                             metalrt::Access::Read);
    }
    metalrt::launch_linear(
        &runtime, loss_kernel<T>(runtime, kind), prediction.n_elements(),
        {p, t, v, g, pdf, constant, metalrt::Argument::scalar(LossBindings::arguments, args)},
        loss_kernels::kernel_descriptor(kind).threads_per_threadgroup);
}

// Substitute named tokens in a handwritten MSL asset without embedding device
// implementations in public host headers or constructing vendor-specific code.
void replace_msl_token(std::string& source, const std::string& token, const std::string& value) {
    size_t position = 0;
    while ((position = source.find(token, position)) != std::string::npos) {
        source.replace(position, token.size(), value);
        position += value.size();
    }
}

// Emit a finite decimal literal that compiles as float in Metal and ordinary C++.
std::string msl_float(float value) {
    if (!std::isfinite(value))
        throw std::invalid_argument("Generated loss constants must be finite.");
    std::ostringstream text;
    text.imbue(std::locale::classic());
    text << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
    auto result = text.str();
    if (result.find_first_of(".eE") == std::string::npos)
        result += ".0";
    return result + "f";
}

// Specialize only host configuration; all objective arithmetic remains in .metal.
std::string loss_device_body(LossKind kind, uint32_t n_dims,
                             const std::vector<float>& constant_gradient) {
    if (!n_dims)
        throw std::invalid_argument("A generated loss function needs a positive dimension count.");
    if (kind == LossKind::RelativeL2Luminance && n_dims < 3)
        throw std::invalid_argument("Luminance loss needs at least three dimensions.");
    if (kind == LossKind::Constant && constant_gradient.size() != n_dims)
        throw std::invalid_argument("Generated constant-gradient dimensions do not match.");
    std::string constants;
    for (uint32_t i = 0; i < n_dims; ++i) {
        if (i)
            constants += ", ";
        constants += msl_float(kind == LossKind::Constant ? constant_gradient[i] : 0.0f);
    }
    auto body = shader_source("loss_function_body.metal");
    replace_msl_token(body, "@KIND@", std::to_string(static_cast<uint32_t>(kind)));
    replace_msl_token(body, "@DIMS@", std::to_string(n_dims));
    replace_msl_token(body, "@CONSTANTS@", constants);
    return body;
}

// The generated helper uses Metal address spaces and metal::array, so callers
// can append a kernel entry point and compile it through MetalRtcKernel/metalrt.
std::string make_loss_function(const std::string& name, uint32_t n_dims, const std::string& body,
                               const char* type) {
    const auto letter = [](unsigned char c) { return std::isalpha(c) || c == '_'; };
    if (name.empty() || !letter(static_cast<unsigned char>(name.front())) ||
        !std::all_of(name.begin() + 1, name.end(),
                     [](unsigned char c) { return std::isalnum(c) || c == '_'; }))
        throw std::invalid_argument("A Metal loss function name must be an identifier.");
    if (!n_dims)
        throw std::invalid_argument("A generated loss function needs a positive dimension count.");
    auto source = shader_source("loss_function.metal");
    replace_msl_token(source, "@BODY@", body);
    replace_msl_token(source, "@NAME@", name);
    replace_msl_token(source, "@DIMS@", std::to_string(n_dims));
    replace_msl_token(source, "@TYPE@", type);
    return source;
}

template void evaluate_loss<float>(metalStream_t, float, const GPUMatrix<float>&,
                                   const GPUMatrix<float>&, GPUMatrix<float>&, GPUMatrix<float>&,
                                   const GPUMatrix<float>*, LossKind, const std::vector<float>&);
template void evaluate_loss<tcnn::half>(metalStream_t, float, const GPUMatrix<tcnn::half>&,
                                        const GPUMatrix<float>&, GPUMatrix<float>&,
                                        GPUMatrix<tcnn::half>&, const GPUMatrix<float>*, LossKind,
                                        const std::vector<float>&);
} // namespace detail

// Keep the host signature while moving the device implementation into MSL assets.
template <typename T>
std::string Loss<T>::generate_device_function_from_body(const std::string& name, uint32_t n_dims,
                                                        const std::string& body) const {
    return detail::make_loss_function(name, n_dims, body,
                                      std::is_same<T, float>::value ? "float" : "half");
}

template std::string Loss<float>::generate_device_function_from_body(const std::string&, uint32_t,
                                                                     const std::string&) const;
template std::string Loss<half>::generate_device_function_from_body(const std::string&, uint32_t,
                                                                    const std::string&) const;

// Creates Metal objectives with the names and default used by the supplied host API.
template <typename T> Loss<T>* create_loss(const json& params) {
    const auto name = to_lower(params.value("otype", "RelativeL2"));
    if (name == "l2")
        return new L2Loss<T>{};
    if (name == "relativel2")
        return new RelativeL2Loss<T>{};
    if (name == "relativel2luminance")
        return new RelativeL2LuminanceLoss<T>{};
    if (name == "l1")
        return new L1Loss<T>{};
    if (name == "relativel1")
        return new RelativeL1Loss<T>{};
    if (name == "mape")
        return new MapeLoss<T>{};
    if (name == "smape")
        return new SmapeLoss<T>{};
    if (name == "crossentropy")
        return new CrossEntropyLoss<T>{};
    if (name == "variance")
        return new VarianceIsLoss<T>{};
    throw std::runtime_error{"Loss '" + params.value("otype", "RelativeL2") + "' not found"};
}

// Returns stable enumeration order while preserving the uploaded set of factory names.
std::vector<std::string> builtin_losses() {
    return {"L2",   "RelativeL2", "RelativeL2Luminance", "L1",      "RelativeL1",
            "Mape", "Smape",      "CrossEntropy",        "Variance"};
}

template Loss<float>* create_loss(const json&);
template Loss<tcnn::half>* create_loss(const json&);
} // namespace tcnn
