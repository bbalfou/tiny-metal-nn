// Enumerates every independently replaceable built-in loss implementation.
#pragma once
#include <tiny-metal-nn/detail/kernels/losses/l2.h>
#include <tiny-metal-nn/detail/kernels/losses/relative_l2.h>
#include <tiny-metal-nn/detail/kernels/losses/relative_l2_luminance.h>
#include <tiny-metal-nn/detail/kernels/losses/l1.h>
#include <tiny-metal-nn/detail/kernels/losses/relative_l1.h>
#include <tiny-metal-nn/detail/kernels/losses/mape.h>
#include <tiny-metal-nn/detail/kernels/losses/smape.h>
#include <tiny-metal-nn/detail/kernels/losses/cross_entropy.h>
#include <tiny-metal-nn/detail/kernels/losses/variance_is.h>
#include <tiny-metal-nn/detail/kernels/losses/constant.h>
#include <array>
#include <stdexcept>

namespace tcnn {
namespace detail {
namespace loss_kernels {
struct KernelDescriptor {
    LossKind kind;
    const char* name;
    const char* source;
    const char* entry_f32;
    const char* entry_f16;
    uint32_t abi_version;
    uint32_t argument_bytes;
    uint32_t threads_per_threadgroup;
};

template <LossKind Kind> constexpr KernelDescriptor descriptor() {
    using Traits = LossKernelTraits<Kind>;
    return {Traits::kind,
            Traits::name,
            Traits::source,
            Traits::entry_f32,
            Traits::entry_f16,
            Traits::abi_version,
            sizeof(typename Traits::Arguments),
            Traits::threads_per_threadgroup};
}

inline constexpr std::array<KernelDescriptor, 10> descriptors = {
    {descriptor<LossKind::L2>(), descriptor<LossKind::RelativeL2>(),
     descriptor<LossKind::RelativeL2Luminance>(), descriptor<LossKind::L1>(),
     descriptor<LossKind::RelativeL1>(), descriptor<LossKind::Mape>(),
     descriptor<LossKind::Smape>(), descriptor<LossKind::CrossEntropy>(),
     descriptor<LossKind::Variance>(), descriptor<LossKind::Constant>()}};

inline const KernelDescriptor& kernel_descriptor(LossKind kind) {
    const auto index = static_cast<uint32_t>(kind);
    if (index >= descriptors.size())
        throw std::invalid_argument("Unknown loss kernel kind.");
    return descriptors[index];
}
} // namespace loss_kernels
} // namespace detail
} // namespace tcnn
