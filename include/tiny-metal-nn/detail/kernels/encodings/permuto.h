// Required Permuto higher-order kernel contract; first-order operations use FirstOrderTraits.
// Buffer bindings and matrix layouts retain encoding ABI v1.
#pragma once
#include <tiny-metal-nn/detail/kernels/encodings/traits_base.h>

namespace tcnn {
namespace detail {
namespace kernels {
namespace encodings {
struct PermutoTraits : TraitsBase {
    static constexpr const char* source = "encodings/permuto.metal";
    static constexpr const char* prefix = "enc_permuto";
    static constexpr bool has_parameter_gradient_kernel = true;
    static constexpr uint32_t kind = 7;
    static constexpr uint32_t max_input_dimensions = 24;
    static constexpr const char* input_second_gradient_f32 =
        "enc_permuto_input_second_gradient_f32";
    static constexpr const char* input_second_gradient_f16 =
        "enc_permuto_input_second_gradient_f16";
    static constexpr const char* output_gradient_f32 = "enc_permuto_output_gradient_f32";
    static constexpr const char* output_gradient_f16 = "enc_permuto_output_gradient_f16";
    static constexpr const char* parameter_second_gradient_f32 =
        "enc_permuto_parameter_second_gradient_f32";
    static constexpr const char* parameter_second_gradient_f16 =
        "enc_permuto_parameter_second_gradient_f16";

    static constexpr KernelDescriptor descriptor() { return {source, prefix}; }
};
} // namespace encodings
} // namespace kernels
} // namespace detail
} // namespace tcnn
