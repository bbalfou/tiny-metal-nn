// Required TriangleWave higher-order kernel contract; first-order operations use FirstOrderTraits.
// Buffer bindings and matrix layouts retain encoding ABI v1.
#pragma once
#include <tiny-metal-nn/detail/kernels/encodings/traits_base.h>

namespace tcnn {
namespace detail {
namespace kernels {
namespace encodings {
struct TriangleWaveTraits : TraitsBase {
    static constexpr const char* source = "encodings/triangle_wave.metal";
    static constexpr const char* prefix = "enc_triangle_wave";
    static constexpr bool has_parameter_gradient_kernel = false;
    static constexpr uint32_t kind = 3;
    static constexpr const char* input_second_gradient_f32 =
        "enc_triangle_wave_input_second_gradient_f32";
    static constexpr const char* input_second_gradient_f16 =
        "enc_triangle_wave_input_second_gradient_f16";
    static constexpr const char* output_gradient_f32 = "enc_triangle_wave_output_gradient_f32";
    static constexpr const char* output_gradient_f16 = "enc_triangle_wave_output_gradient_f16";

    static constexpr KernelDescriptor descriptor() { return {source, prefix}; }
};
} // namespace encodings
} // namespace kernels
} // namespace detail
} // namespace tcnn
