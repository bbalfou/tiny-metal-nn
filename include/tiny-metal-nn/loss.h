#pragma once
#include <tiny-metal-nn/common.h>
#include <tiny-metal-nn/gpu_matrix.h>
#include <tiny-metal-nn/object.h>

namespace tcnn {
// Evaluates per-element losses and loss-scaled gradients using the reference matrix signatures.
template <typename T> class Loss : public ObjectWithMutableHyperparams {
  public:
    // Prediction and gradients include padded rows; target and optional PDF contain logical rows.
    virtual void evaluate(metalStream_t stream, const float loss_scale,
                          const GPUMatrix<T>& prediction, const GPUMatrix<float>& target,
                          GPUMatrix<float>& values, GPUMatrix<T>& gradients,
                          const GPUMatrix<float>* data_pdf = nullptr) const = 0;

    // Uses the current device's default stream, retaining the reference overload.
    void evaluate(const float loss_scale, const GPUMatrix<T>& prediction,
                  const GPUMatrix<float>& target, GPUMatrix<float>& values, GPUMatrix<T>& gradients,
                  const GPUMatrix<float>* data_pdf = nullptr) const {
        evaluate(nullptr, loss_scale, prediction, target, values, gradients, data_pdf);
    }

    // Produces an MSL helper for use from a caller-owned Metal kernel.
    virtual std::string generate_device_function(const std::string& name, uint32_t n_dims) const {
        (void)name;
        (void)n_dims;
        throw std::runtime_error{this->name() + ": device code generation is not supported."};
    }

    // Wraps an MSL body with typed metal::array inputs and an optional FP32 value.
    // Device source comes from a .metal asset; this header contains host API only.
    std::string generate_device_function_from_body(const std::string& name, uint32_t n_dims,
                                                   const std::string& body) const;
};

// Creates a loss using the reference's case-insensitive names and RelativeL2 default.
template <typename T> Loss<T>* create_loss(const json& params);

// Constructs a named built-in loss with its reference defaults.
template <typename T> std::unique_ptr<Loss<T>> default_loss(const std::string& name) {
    return std::unique_ptr<Loss<T>>{create_loss<T>({{"otype", name}})};
}

// Lists the nine factory-registered losses in the uploaded reference.
std::vector<std::string> builtin_losses();

namespace detail {
enum class LossKind : uint32_t {
    L2 = 0,
    RelativeL2 = 1,
    RelativeL2Luminance = 2,
    L1 = 3,
    RelativeL1 = 4,
    Mape = 5,
    Smape = 6,
    CrossEntropy = 7,
    Variance = 8,
    Constant = 9
};
// Shared dispatch preserves each uploaded formula, including stop-gradient denominators.
template <typename T>
void evaluate_loss(metalStream_t stream, const float loss_scale, const GPUMatrix<T>& prediction,
                   const GPUMatrix<float>& target, GPUMatrix<float>& values,
                   GPUMatrix<T>& gradients, const GPUMatrix<float>* data_pdf, LossKind kind,
                   const std::vector<float>& constant_gradient = {});
// Specializes the MSL loss-body asset for a configured objective and dimensions.
std::string loss_device_body(LossKind kind, uint32_t n_dims,
                             const std::vector<float>& constant_gradient = {});
} // namespace detail
} // namespace tcnn
