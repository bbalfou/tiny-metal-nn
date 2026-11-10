// Required HashGridRng higher-order kernel contract; first-order operations use FirstOrderTraits.
// Buffer bindings and matrix layouts retain encoding ABI v1.
#pragma once
#include <tiny-metal-nn/detail/kernels/encodings/traits_base.h>

namespace tcnn {
namespace detail {
namespace kernels {
namespace encodings {
struct HashGridRngTraits : TraitsBase {
    static constexpr const char* source = "encodings/hash_grid_rng.metal";
    static constexpr const char* prefix = "enc_hash_grid_rng";
    static constexpr bool has_parameter_gradient_kernel = true;
    static constexpr uint32_t kind = 6;
    static constexpr uint32_t grid_type = 0;
    static constexpr uint32_t max_input_dimensions = 4;
    static constexpr uint32_t hash_type = 3;
    static constexpr const char* input_second_gradient_f32 =
        "enc_hash_grid_rng_input_second_gradient_f32";
    static constexpr const char* input_second_gradient_f16 =
        "enc_hash_grid_rng_input_second_gradient_f16";
    static constexpr const char* output_gradient_f32 = "enc_hash_grid_rng_output_gradient_f32";
    static constexpr const char* output_gradient_f16 = "enc_hash_grid_rng_output_gradient_f16";
    static constexpr const char* parameter_second_gradient_f32 =
        "enc_hash_grid_rng_parameter_second_gradient_f32";
    static constexpr const char* parameter_second_gradient_f16 =
        "enc_hash_grid_rng_parameter_second_gradient_f16";

    static constexpr KernelDescriptor descriptor() { return {source, prefix}; }
};
} // namespace encodings
} // namespace kernels
} // namespace detail
} // namespace tcnn
