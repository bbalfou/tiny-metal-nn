#pragma once
#include <tiny-metal-nn/detail/kernels/encodings/traits_base.h>
namespace tcnn {
namespace detail {
namespace kernels {
namespace encodings {
// All first-order families share the unchanged v1 buffer/constant contract.
struct FirstOrderTraits : TraitsBase {
    static constexpr const char* source = "encodings/forward_backward.metal";
    static constexpr const char* forward_f32 = "encoding_forward_f32";
    static constexpr const char* forward_f16 = "encoding_forward_f16";
    static constexpr const char* input_gradient_f32 = "encoding_input_gradient_f32";
    static constexpr const char* input_gradient_f16 = "encoding_input_gradient_f16";
    static constexpr const char* parameter_gradient_f32 = "encoding_parameter_gradient_f32";
    static constexpr const char* parameter_gradient_f16 = "encoding_parameter_gradient_f16";
};
} // namespace encodings
} // namespace kernels
} // namespace detail
} // namespace tcnn
