// RelativeL1: independent optimization contract for its Metal entry points.
#pragma once
#include <tiny-metal-nn/detail/kernels/losses/base.h>

namespace tcnn {
namespace detail {
namespace loss_kernels {
template <> struct LossKernelTraits<LossKind::RelativeL1> : KernelTraitsBase {
    static constexpr LossKind kind = LossKind::RelativeL1;
    static constexpr const char* name = "RelativeL1";
    static constexpr const char* source = "losses/relative_l1.metal";
    static constexpr const char* entry_f32 = "tcnn_loss_relative_l1_f32";
    static constexpr const char* entry_f16 = "tcnn_loss_relative_l1_f16";
};
} // namespace loss_kernels
} // namespace detail
} // namespace tcnn
