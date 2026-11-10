// Baseline MultiLevelLoDSoft encoding kernel contract. Replace only its matching MSL
// entrypoint bodies when optimizing; these names, bindings and layouts are stable.
#pragma once
#include <tiny-metal-nn/detail/kernels/encodings/traits_base.h>

namespace tcnn {
namespace detail {
namespace kernels {
namespace encodings {
struct MultiLevelLoDSoftTraits : TraitsBase {
    static constexpr const char* source = "encodings/multi_level_lod_soft.metal";
    static constexpr const char* prefix = "enc_multi_level_lod_soft";
    static constexpr bool has_parameter_gradient_kernel = false;
    static constexpr const char* utility_f32 = "enc_multi_level_lod_soft_utility_f32";
    static constexpr const char* utility_f16 = "enc_multi_level_lod_soft_utility_f16";
    static constexpr const char* product_second_f32 = "enc_multi_level_lod_soft_product_second_f32";
    static constexpr const char* product_second_f16 = "enc_multi_level_lod_soft_product_second_f16";
    static constexpr const char* cast_gradient_f32 = "enc_multi_level_lod_soft_cast_gradient_f32";
    static constexpr const char* cast_gradient_f16 = "enc_multi_level_lod_soft_cast_gradient_f16";
    static constexpr const char* levels_f32 = "enc_multi_level_lod_soft_levels";

    static constexpr KernelDescriptor descriptor() { return {source, prefix}; }
};
} // namespace encodings
} // namespace kernels
} // namespace detail
} // namespace tcnn
