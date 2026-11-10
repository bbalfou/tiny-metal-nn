#pragma once
#include <tiny-metal-nn/loss.h>

namespace tcnn {
// Supplies caller-chosen per-dimension gradients; reported loss values are identically zero.
template <typename T> class ConstantGradientLoss : public Loss<T> {
  public:
    // Copies the dimension-wise gradient so its lifetime is independent of the caller's vector.
    ConstantGradientLoss(const std::vector<float>& constant_gradient)
        : m_constant_gradient{constant_gradient} {}

    // Applies loss_scale and inverse PDF without dividing by the number of samples.
    void evaluate(metalStream_t stream, const float loss_scale, const GPUMatrix<T>& prediction,
                  const GPUMatrix<float>& target, GPUMatrix<float>& values, GPUMatrix<T>& gradients,
                  const GPUMatrix<float>* data_pdf = nullptr) const override {
        detail::evaluate_loss(stream, loss_scale, prediction, target, values, gradients, data_pdf,
                              detail::LossKind::Constant, m_constant_gradient);
    }

    // ConstantGradientLoss has no mutable hyperparameters in the uploaded API.
    void update_hyperparams(const json& params) override { (void)params; }

    // Returns the reference type name and copied dimension-wise gradient.
    json hyperparams() const override {
        return {{"otype", "ConstantGradient"}, {"value", m_constant_gradient}};
    }

    // Generates a callable MSL helper with the configured gradient constants.
    std::string generate_device_function(const std::string& name, uint32_t n_dims) const override {
        return this->generate_device_function_from_body(
            name, n_dims,
            detail::loss_device_body(detail::LossKind::Constant, n_dims, m_constant_gradient));
    }

  private:
    std::vector<float> m_constant_gradient;
};
} // namespace tcnn
