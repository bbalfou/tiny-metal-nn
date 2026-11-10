#pragma once
#include <tiny-metal-nn/detail/kernels/encodings/abi.h>

namespace tcnn {
namespace detail {
namespace kernels {
namespace encodings {

// Private grouped-gradient binding contract. The existing per-level v1 Args
// and entrypoints remain unchanged for the required fallback operations.
struct HashGridGradientLevel {
    Args args;
    uint32_t parameter_offset;
};
static_assert(std::is_trivially_copyable<HashGridGradientLevel>::value,
              "Level descriptors must be plain data.");
static_assert(sizeof(HashGridGradientLevel) == 376 && alignof(HashGridGradientLevel) == 4,
              "Grouped HashGrid gradient host/MSL layout.");
static_assert(offsetof(HashGridGradientLevel, parameter_offset) == 372,
              "Grouped HashGrid parameter offset.");

struct HashGridParameterGradientTraits {
    using Level = HashGridGradientLevel;
    static constexpr const char* source = "encodings/hash_grid_parameter_gradient.metal";
    static constexpr const char* entry_f32 = "encoding_hash_grid_parameter_gradient_f32";
    static constexpr const char* entry_f16 = "encoding_hash_grid_parameter_gradient_f16";
    static constexpr uint32_t input = 0, limits = 1, output_gradient = 2, gradient_scratch = 3,
                              levels = 4;
    static constexpr uint32_t threads = 128;

    // Measured families: 2D log15/scale1.5 in both precisions, and
    // 2D/3D log19/scale2 in FP16. Other configurations retain per-level gradients.
    // See benchmarks/hashgrid_backward/RESULTS.md for paired training evidence.
    static constexpr bool supports_configuration(uint32_t dimensions, uint32_t level_count,
                                                 uint32_t features, uint32_t log2_hashmap_size,
                                                 uint32_t base_resolution, float per_level_scale,
                                                 uint32_t interpolation, uint32_t hash_type,
                                                 bool fixed_point, bool stochastic,
                                                 bool half_precision = false) {
        return (dimensions == 2 || dimensions == 3) && level_count == 16 && features == 2 &&
               base_resolution == 16 && interpolation == 1 && hash_type == 1 && !fixed_point &&
               !stochastic &&
               ((dimensions == 2 && log2_hashmap_size == 15 && per_level_scale == 1.5f) ||
                (half_precision && log2_hashmap_size == 19 && per_level_scale == 2.0f));
    }

    static constexpr bool supports_dispatch(uint32_t apple_gpu_family, uint32_t batch,
                                            bool large_table = false) {
        // The adaptive grid has two threads per sample, rounded to 128 samples.
        // This is a uint32 indexing bound, not a measured batch-size cutoff.
        constexpr uint32_t max_adaptive_batch = (UINT32_MAX / 2 / threads) * threads;
        return apple_gpu_family >= 7 && batch > 0 &&
               batch <= (large_table ? max_adaptive_batch : 4096u);
    }
};

} // namespace encodings
} // namespace kernels
} // namespace detail
} // namespace tcnn
