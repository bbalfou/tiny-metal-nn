// ConstantGradient: independent optimization contract for its Metal entry points.
#pragma once
#include <tiny-metal-nn/detail/kernels/losses/base.h>

namespace tcnn {
namespace detail {
namespace loss_kernels {
template <> struct LossKernelTraits<LossKind::Constant> : KernelTraitsBase {
    static constexpr LossKind kind = LossKind::Constant;
    static constexpr const char* name = "ConstantGradient";
    static constexpr const char* source = "losses/constant.metal";
    static constexpr const char* entry_f32 = "tcnn_loss_constant_f32";
    static constexpr const char* entry_f16 = "tcnn_loss_constant_f16";
};
} // namespace loss_kernels
} // namespace detail
} // namespace tcnn
