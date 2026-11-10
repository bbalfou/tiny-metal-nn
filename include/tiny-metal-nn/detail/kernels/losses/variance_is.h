// Variance: independent optimization contract for its Metal entry points.
#pragma once
#include <tiny-metal-nn/detail/kernels/losses/base.h>

namespace tcnn {
namespace detail {
namespace loss_kernels {
template <> struct LossKernelTraits<LossKind::Variance> : KernelTraitsBase {
    static constexpr LossKind kind = LossKind::Variance;
    static constexpr const char* name = "Variance";
    static constexpr const char* source = "losses/variance_is.metal";
    static constexpr const char* entry_f32 = "tcnn_loss_variance_is_f32";
    static constexpr const char* entry_f16 = "tcnn_loss_variance_is_f16";
};
} // namespace loss_kernels
} // namespace detail
} // namespace tcnn
