#pragma once
#include <tiny-metal-nn/detail/kernels/encodings/abi.h>

namespace tcnn::detail::kernels::encodings {
// Immutable per-submission descriptors retain the existing first-order Args ABI.
struct HashGridInferenceLevel {
    Args args;
    uint32_t parameter_offset;
};
static_assert(sizeof(HashGridInferenceLevel) == 376 &&
                  offsetof(HashGridInferenceLevel, parameter_offset) == 372,
              "Grouped HashGrid inference host/MSL layout.");
struct HashGridInferenceTraits {
    static constexpr const char* source = "encodings/hash_grid_inference.metal";
    static constexpr const char* entry_f32 = "encoding_hash_grid_inference_f32";
    static constexpr const char* entry_f16 = "encoding_hash_grid_inference_f16";
    static constexpr uint32_t threads = 128;
};
} // namespace tcnn::detail::kernels::encodings
