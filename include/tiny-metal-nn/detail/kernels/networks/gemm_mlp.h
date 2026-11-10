// Network option traits. Entry names and scalar argument slots are ABI version 1.
#pragma once
#include <tiny-metal-nn/detail/kernels/networks/abi.h>

namespace tcnn {
namespace detail {
namespace network_kernels {
struct GemmMLPTraits {
    static constexpr Path path = Path::Gemm;
    static constexpr uint32_t abi_version = ABI_VERSION;
    static constexpr const char* source = "networks/gemm_mlp.metal";
    static constexpr const char* forward = "gemm_dense";
    static constexpr const char* delta = "gemm_delta";
    static constexpr const char* input_gradient = "gemm_dense_dx";
    static constexpr const char* parameter_gradient = "gemm_dense_dw";
    static constexpr const char* second_delta = "gemm_second_delta";
    using ForwardArgs = Dense;
    using DeltaArgs = Delta;
    using InputGradientArgs = InputGradient;
    using ParameterGradientArgs = WeightGradient;
    using SecondDeltaArgs = SecondDelta;

    using ForwardBindings = DenseForwardBindings;
    using ActivationGradientBindings = DeltaBindings;
    using InputGradientBindings = DenseInputGradientBindings;
    using ParameterGradientBindings = DenseParameterGradientBindings;
    using SecondOrderActivationBindings = SecondDeltaBindings;

    enum : uint32_t {
        forward_constants = 4,
        delta_constants = 3,
        input_gradient_constants = 3,
        parameter_gradient_constants = 3,
        second_delta_constants = 7,
        required_alignment = 16
    };
};
} // namespace network_kernels
} // namespace detail
} // namespace tcnn
