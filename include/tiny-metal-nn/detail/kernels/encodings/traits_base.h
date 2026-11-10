// Per-option optimization contracts for the native C++ encoding dispatch paths.
#pragma once
#include <tiny-metal-nn/detail/kernels/encodings/abi.h>

namespace tcnn {
namespace detail {
namespace kernels {
namespace encodings {
struct KernelDescriptor {
    const char* source;
    const char* prefix;
};

struct TraitsBase {
    using Arguments = Args;
    using UtilityArguments = UtilityArgs;
    using Forward = ForwardBindings;
    using InputGradient = InputGradientBindings;
    using OutputGradient = OutputGradientBindings;
    using ParameterGradient = ParameterGradientBindings;
    using Utility = UtilityBindings;
    using ProductSecond = ProductSecondBindings;
    using CastGradient = CastGradientBindings;
    using Levels = LevelsBindings;
    static constexpr uint32_t abi_version = ABI_VERSION;
    static constexpr bool supports_fp16 = true;
    static constexpr bool supports_fp32 = true;
    static constexpr bool uses_float_accumulation = true;
    static constexpr bool supports_input_backward = true;
    static constexpr bool supports_input_second_backward = true;
    static constexpr bool requires_float_atomics = false;
};
} // namespace encodings
} // namespace kernels
} // namespace detail
} // namespace tcnn
