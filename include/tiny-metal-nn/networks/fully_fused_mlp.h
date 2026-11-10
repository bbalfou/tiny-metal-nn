#pragma once
#include <tiny-metal-nn/detail/mlp.h>
#include <tiny-metal-nn/detail/network_selection.h>

namespace tcnn {
// Whole-model Metal MLP for supported widths on Apple GPU family 9 or newer.
template <typename T, uint32_t WIDTH> class FullyFusedMLP : public detail::BaselineMLP<T> {
    using Base = detail::BaselineMLP<T>;

  public:
    using type_t = T;

    bool jit_allows_float_gradient_accumulation() const override {
        return typeid(*this) == typeid(FullyFusedMLP);
    }

    bool jit_supports_fused_optimizer() const override {
        return typeid(*this) == typeid(FullyFusedMLP);
    }

    bool jit_supports_compact_training_context() const override {
        return typeid(*this) == typeid(FullyFusedMLP);
    }

    // Construct a bias-free MLP; the final matrix includes output padding to 16.
    FullyFusedMLP(uint32_t input_width, uint32_t output_width, uint32_t n_hidden_layers,
                  Activation activation, Activation output_activation)
        : Base{input_width,     WIDTH,
               output_width,    n_hidden_layers,
               activation,      output_activation,
               "FullyFusedMLP", detail::network_kernels::Path::FullyFused} {
        static_assert(WIDTH == 16 || WIDTH == 32 || WIDTH == 64 || WIDTH == 128,
                      "FullyFusedMLP supports 16, 32, 64, and 128 neurons");
        nn_check(detail::fully_fused_mlp_supported(metalrt::Device::current().capabilities()),
                 "FullyFusedMLP requires Metal Apple GPU family 9 or newer; use GemmMLP");
    }

    uint32_t num_forward_activations() const override { return this->m_n_hidden_layers; }

    std::pair<const T*, MatrixLayout> forward_activations(const Context& ctx,
                                                          uint32_t layer) const override {
        nn_check(layer < this->m_n_hidden_layers, "forward activation layer out of range");
        if (this->is_jit_context(ctx))
            return this->jit_forward_activation(ctx, forward_activation_device_offset(layer),
                                                this->m_network_width);
        const auto& forward = this->checked_context(ctx);
        return {forward.hidden.at(layer).data(), CM};
    }

    uint32_t forward_activation_device_offset(uint32_t layer) const override {
        nn_check(layer < this->m_n_hidden_layers, "forward activation layer out of range");
        const size_t offset =
            (size_t(this->m_input_width) + (2u * size_t(layer) + 1u) * this->m_network_width) *
            sizeof(T);
        nn_check(offset <= UINT32_MAX, "forward activation offset exceeds uint32_t");
        return static_cast<uint32_t>(offset);
    }
};
} // namespace tcnn
