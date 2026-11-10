#pragma once
#include <tiny-metal-nn/loss.h>

namespace tcnn {
// Absolute error with the reference positive subgradient at an exactly zero residual.
template <typename T> class L1Loss : public Loss<T> {
  public:
    // Evaluates the uploaded formula and zeros every padded prediction/gradient row.
    void evaluate(metalStream_t stream, const float loss_scale, const GPUMatrix<T>& prediction,
                  const GPUMatrix<float>& target, GPUMatrix<float>& values, GPUMatrix<T>& gradients,
                  const GPUMatrix<float>* data_pdf = nullptr) const override {
        detail::evaluate_loss(stream, loss_scale, prediction, target, values, gradients, data_pdf,
                              detail::LossKind::L1);
    }

    // This reference loss has no mutable hyperparameters.
    void update_hyperparams(const json& params) override { (void)params; }

    // Returns the canonical configuration name from the uploaded repository.
    json hyperparams() const override { return {{"otype", "L1"}}; }

    // Generates a callable MSL helper from the handwritten Metal loss template.
    std::string generate_device_function(const std::string& name, uint32_t n_dims) const override {
        return this->generate_device_function_from_body(
            name, n_dims, detail::loss_device_body(detail::LossKind::L1, n_dims));
    }
};
} // namespace tcnn
