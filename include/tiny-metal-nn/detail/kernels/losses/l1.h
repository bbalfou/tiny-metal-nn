// L1: independent optimization contract for its Metal entry points.
#pragma once
#include <tiny-metal-nn/detail/kernels/losses/base.h>

namespace tcnn {
namespace detail {
namespace loss_kernels {
template <> struct LossKernelTraits<LossKind::L1> : KernelTraitsBase {
    static constexpr LossKind kind = LossKind::L1;
    static constexpr const char* name = "L1";
    static constexpr const char* source = "losses/l1.metal";
    static constexpr const char* entry_f32 = "tcnn_loss_l1_f32";
    static constexpr const char* entry_f16 = "tcnn_loss_l1_f16";
};
} // namespace loss_kernels
} // namespace detail
} // namespace tcnn
