// L2: independent optimization contract for its Metal entry points.
#pragma once
#include <tiny-metal-nn/detail/kernels/losses/base.h>

namespace tcnn {
namespace detail {
namespace loss_kernels {
template <> struct LossKernelTraits<LossKind::L2> : KernelTraitsBase {
    static constexpr LossKind kind = LossKind::L2;
    static constexpr const char* name = "L2";
    static constexpr const char* source = "losses/l2.metal";
    static constexpr const char* entry_f32 = "tcnn_loss_l2_f32";
    static constexpr const char* entry_f16 = "tcnn_loss_l2_f16";
};
} // namespace loss_kernels
} // namespace detail
} // namespace tcnn
