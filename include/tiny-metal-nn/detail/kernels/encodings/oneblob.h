// Required OneBlob higher-order kernel contract; first-order operations use FirstOrderTraits.
// Buffer bindings and matrix layouts retain encoding ABI v1.
#pragma once
#include <tiny-metal-nn/detail/kernels/encodings/traits_base.h>

namespace tcnn {
namespace detail {
namespace kernels {
namespace encodings {
struct OneBlobTraits : TraitsBase {
    static constexpr const char* source = "encodings/oneblob.metal";
    static constexpr const char* prefix = "enc_oneblob";
    static constexpr bool has_parameter_gradient_kernel = false;
    static constexpr uint32_t kind = 4;
    static constexpr const char* input_second_gradient_f32 =
        "enc_oneblob_input_second_gradient_f32";
    static constexpr const char* input_second_gradient_f16 =
        "enc_oneblob_input_second_gradient_f16";
    static constexpr const char* output_gradient_f32 = "enc_oneblob_output_gradient_f32";
    static constexpr const char* output_gradient_f16 = "enc_oneblob_output_gradient_f16";

    static constexpr KernelDescriptor descriptor() { return {source, prefix}; }
};
} // namespace encodings
} // namespace kernels
} // namespace detail
} // namespace tcnn
