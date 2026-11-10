// Mape: independent optimization contract for its Metal entry points.
#pragma once
#include <tiny-metal-nn/detail/kernels/losses/base.h>

namespace tcnn {
namespace detail {
namespace loss_kernels {
template <> struct LossKernelTraits<LossKind::Mape> : KernelTraitsBase {
    static constexpr LossKind kind = LossKind::Mape;
    static constexpr const char* name = "Mape";
    static constexpr const char* source = "losses/mape.metal";
    static constexpr const char* entry_f32 = "tcnn_loss_mape_f32";
    static constexpr const char* entry_f16 = "tcnn_loss_mape_f16";
};
} // namespace loss_kernels
} // namespace detail
} // namespace tcnn
