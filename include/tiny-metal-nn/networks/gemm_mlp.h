#pragma once
#include <tiny-metal-nn/detail/mlp.h>

namespace tcnn {
// General layerwise Metal MLP with arbitrary positive hidden widths.
template <typename T> class GemmMLP : public detail::BaselineMLP<T> {
  public:
    using type_t = T;

    GemmMLP(uint32_t input_width, uint32_t network_width, uint32_t output_width,
            uint32_t n_hidden_layers, Activation activation, Activation output_activation)
        : detail::BaselineMLP<T>{
              input_width, network_width,     output_width, n_hidden_layers,
              activation,  output_activation, "GemmMLP",    detail::network_kernels::Path::Gemm} {}

    bool jit_allows_float_gradient_accumulation() const override {
        return typeid(*this) == typeid(GemmMLP);
    }

    bool jit_supports_fused_optimizer() const override { return typeid(*this) == typeid(GemmMLP); }

    bool jit_supports_compact_training_context() const override {
        return typeid(*this) == typeid(GemmMLP);
    }
};
} // namespace tcnn
