// Baseline CompositeProduct encoding kernel contract. Replace only its matching MSL
// entrypoint bodies when optimizing; these names, bindings and layouts are stable.
#pragma once
#include <tiny-metal-nn/detail/kernels/encodings/traits_base.h>

namespace tcnn {
namespace detail {
namespace kernels {
namespace encodings {
struct CompositeProductTraits : TraitsBase {
    static constexpr const char* source = "encodings/composite_product.metal";
    static constexpr const char* prefix = "enc_composite_product";
    static constexpr bool has_parameter_gradient_kernel = false;
    static constexpr const char* utility_f32 = "enc_composite_product_utility_f32";
    static constexpr const char* utility_f16 = "enc_composite_product_utility_f16";
    static constexpr const char* product_second_f32 = "enc_composite_product_product_second_f32";
    static constexpr const char* product_second_f16 = "enc_composite_product_product_second_f16";
    static constexpr const char* cast_gradient_f32 = "enc_composite_product_cast_gradient_f32";
    static constexpr const char* cast_gradient_f16 = "enc_composite_product_cast_gradient_f16";
    static constexpr const char* levels_f32 = "enc_composite_product_levels";

    static constexpr KernelDescriptor descriptor() { return {source, prefix}; }
};
} // namespace encodings
} // namespace kernels
} // namespace detail
} // namespace tcnn
