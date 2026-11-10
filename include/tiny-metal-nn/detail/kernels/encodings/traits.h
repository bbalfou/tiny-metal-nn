#pragma once
#include <tiny-metal-nn/detail/kernels/encodings/first_order.h>
#include <tiny-metal-nn/detail/kernels/encodings/hash_grid_parameter_gradient.h>
#include <tiny-metal-nn/detail/kernels/encodings/empty.h>
#include <tiny-metal-nn/detail/kernels/encodings/identity.h>
#include <tiny-metal-nn/detail/kernels/encodings/frequency.h>
#include <tiny-metal-nn/detail/kernels/encodings/triangle_wave.h>
#include <tiny-metal-nn/detail/kernels/encodings/oneblob.h>
#include <tiny-metal-nn/detail/kernels/encodings/spherical_harmonics.h>
#include <tiny-metal-nn/detail/kernels/encodings/hash_grid.h>
#include <tiny-metal-nn/detail/kernels/encodings/dense_grid.h>
#include <tiny-metal-nn/detail/kernels/encodings/tiled_grid.h>
#include <tiny-metal-nn/detail/kernels/encodings/hash_grid_prime.h>
#include <tiny-metal-nn/detail/kernels/encodings/hash_grid_reversed_prime.h>
#include <tiny-metal-nn/detail/kernels/encodings/hash_grid_rng.h>
#include <tiny-metal-nn/detail/kernels/encodings/hash_grid_base_convert.h>
#include <tiny-metal-nn/detail/kernels/encodings/permuto.h>
#include <tiny-metal-nn/detail/kernels/encodings/composite_concatenation.h>
#include <tiny-metal-nn/detail/kernels/encodings/composite_sum.h>
#include <tiny-metal-nn/detail/kernels/encodings/composite_product.h>
#include <tiny-metal-nn/detail/kernels/encodings/multi_level_lod_hard.h>
#include <tiny-metal-nn/detail/kernels/encodings/multi_level_lod_soft.h>
#include <tiny-metal-nn/detail/kernels/encodings/utility.h>

namespace tcnn {
namespace detail {
namespace kernels {
namespace encodings {
inline KernelDescriptor analytic_kernel(uint32_t kind) {
    switch (kind) {
    case 0:
        return EmptyTraits::descriptor();
    case 1:
        return IdentityTraits::descriptor();
    case 2:
        return FrequencyTraits::descriptor();
    case 3:
        return TriangleWaveTraits::descriptor();
    case 4:
        return OneBlobTraits::descriptor();
    default:
        return SphericalHarmonicsTraits::descriptor();
    }
}

inline KernelDescriptor lattice_kernel(bool permuto, uint32_t grid_type, uint32_t hash_type) {
    if (permuto)
        return PermutoTraits::descriptor();
    if (grid_type == 1)
        return DenseGridTraits::descriptor();
    if (grid_type == 2)
        return TiledGridTraits::descriptor();
    switch (hash_type) {
    case 0:
        return HashGridPrimeTraits::descriptor();
    case 2:
        return HashGridReversedPrimeTraits::descriptor();
    case 3:
        return HashGridRngTraits::descriptor();
    case 4:
        return HashGridBaseConvertTraits::descriptor();
    default:
        return HashGridTraits::descriptor();
    }
}

inline KernelDescriptor composite_kernel(uint32_t reduction) {
    return reduction == 0   ? CompositeConcatenationTraits::descriptor()
           : reduction == 1 ? CompositeSumTraits::descriptor()
                            : CompositeProductTraits::descriptor();
}

inline KernelDescriptor lod_kernel(bool soft) {
    return soft ? MultiLevelLoDSoftTraits::descriptor() : MultiLevelLoDHardTraits::descriptor();
}
} // namespace encodings
} // namespace kernels
} // namespace detail
} // namespace tcnn
