// Common metadata only; every objective owns its own traits specialization.
#pragma once
#include <tiny-metal-nn/loss.h>
#include <tiny-metal-nn/detail/kernels/losses/abi.h>

namespace tcnn {
namespace detail {
namespace loss_kernels {
struct KernelTraitsBase {
    using Arguments = LossArgs;
    using BufferBindings = Bindings;
    static constexpr uint32_t abi_version = ABI_VERSION;
    static constexpr uint32_t threads_per_threadgroup = 128;
    static constexpr bool supports_float = true;
    static constexpr bool supports_half = true;
    static constexpr bool supports_strided_views = true;
};

template <LossKind Kind> struct LossKernelTraits;
} // namespace loss_kernels
} // namespace detail
} // namespace tcnn
