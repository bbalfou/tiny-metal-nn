// Smape: independent optimization contract for its Metal entry points.
#pragma once
#include <tiny-metal-nn/detail/kernels/losses/base.h>

namespace tcnn {
namespace detail {
namespace loss_kernels {
template <> struct LossKernelTraits<LossKind::Smape> : KernelTraitsBase {
    static constexpr LossKind kind = LossKind::Smape;
    static constexpr const char* name = "Smape";
    static constexpr const char* source = "losses/smape.metal";
    static constexpr const char* entry_f32 = "tcnn_loss_smape_f32";
    static constexpr const char* entry_f16 = "tcnn_loss_smape_f16";
};
} // namespace loss_kernels
} // namespace detail
} // namespace tcnn
