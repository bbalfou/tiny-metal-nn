// Launch-budget controls for MetalRtcKernel. These validate a dispatch; they do
// not pretend to tune driver registers, cache carve-outs, or vendor machine code.
#pragma once
#include <cstdint>
enum class MetalFunctionAttribute : uint32_t {
    MaxThreadsPerThreadgroup = 0,
    MaxThreadgroupMemoryBytes = 1
};
