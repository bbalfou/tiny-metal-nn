#pragma once
#include <tiny-metal-nn/object.h>

namespace tcnn {
namespace detail {
// M3-series GPUs expose Apple family 9. Newer GPU families remain eligible.
inline bool fully_fused_mlp_supported(const metalrt::Capabilities& capabilities) {
    return capabilities.backend == metalrt::BackendKind::Metal &&
           capabilities.apple_gpu_family >= 9;
}

// Pure factory selection for the supplied device. Unsupported hardware uses
// GemmMLP; invalid fused widths on supported hardware are rejected by the factory.
std::string select_network_for_capabilities(const json& network,
                                            const metalrt::Capabilities& capabilities);
} // namespace detail
} // namespace tcnn
