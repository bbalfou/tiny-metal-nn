// CrossEntropy: independent optimization contract for its Metal entry points.
#pragma once
#include <tiny-metal-nn/detail/kernels/losses/base.h>

namespace tcnn {
namespace detail {
namespace loss_kernels {
template <> struct LossKernelTraits<LossKind::CrossEntropy> : KernelTraitsBase {
    static constexpr LossKind kind = LossKind::CrossEntropy;
    static constexpr const char* name = "CrossEntropy";
    static constexpr const char* source = "losses/cross_entropy.metal";
    static constexpr const char* entry_f32 = "tcnn_loss_cross_entropy_f32";
    static constexpr const char* entry_f16 = "tcnn_loss_cross_entropy_f16";
};
} // namespace loss_kernels
} // namespace detail
} // namespace tcnn
