// Stable v1 encoding dispatch ABI. Buffer indices and structure layout are frozen.
// All matrices use element strides. Inputs, input gradients and parameter scratch
// use float; output/parameter storage is the dispatched half or float precision.
#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tcnn {
namespace detail {
namespace kernels {
namespace encodings {
using uint = uint32_t;

struct Mat {
    uint rows, cols, rs, cs;
};

struct Args {
    Mat x, y, dy, dx, ddx, ddy;
    uint dims, width, degree, kind, level, levels, features, table_size, resolution, interpolation,
        grid_type, hash_type, max_input_grad_dims, has_limits, mode, accumulate, stochastic;
    float scale, offset, max_level, padding;
    float scales[24], shifts[24];
};

struct UtilityArgs {
    Mat src, dst, aux;
    uint mode, width, index, nested, levels, features, has_limits;
    float value;
};

inline constexpr uint32_t ABI_VERSION = 1;
static_assert(std::is_standard_layout<Mat>::value && std::is_trivially_copyable<Args>::value,
              "Encoding constants must be plain data.");
static_assert(sizeof(Mat) == 16 && alignof(Mat) == 4, "Encoding matrix ABI v1.");
static_assert(sizeof(Args) == 372 && alignof(Args) == 4, "Encoding argument ABI v1.");
static_assert(sizeof(UtilityArgs) == 80 && alignof(UtilityArgs) == 4, "Encoding utility ABI v1.");
static_assert(offsetof(Args, dims) == 96 && offsetof(Args, kind) == 108 &&
                  offsetof(Args, hash_type) == 140,
              "Encoding dimension ABI v1.");
static_assert(offsetof(Args, scale) == 164 && offsetof(Args, scales) == 180 &&
                  offsetof(Args, shifts) == 276,
              "Encoding scale ABI v1.");
static_assert(offsetof(UtilityArgs, mode) == 48 && offsetof(UtilityArgs, value) == 76,
              "Encoding utility offsets v1.");

struct ForwardBindings {
    static constexpr uint32_t input = 0, params = 1, limits = 2, output = 3, arguments = 4;
};

struct InputGradientBindings {
    static constexpr uint32_t input = 0, params = 1, limits = 2, output_gradient = 3,
                              input_gradient = 4, direction = 5, arguments = 6;
};

struct OutputGradientBindings {
    static constexpr uint32_t input = 0, params = 1, limits = 2, direction = 3, output_gradient = 4,
                              arguments = 5;
};

struct ParameterGradientBindings {
    static constexpr uint32_t input = 0, limits = 1, output_gradient = 2, gradient_scratch = 3,
                              direction = 4, arguments = 5;
};

struct UtilityBindings {
    static constexpr uint32_t source = 0, destination = 1, auxiliary = 2, limits = 3, arguments = 4;
};

struct ProductSecondBindings {
    static constexpr uint32_t output_gradient = 0, values = 1, tangents = 2, destination = 3,
                              arguments = 4;
};

struct CastGradientBindings {
    static constexpr uint32_t source = 0, destination = 1, count = 2, accumulate = 3;
};

struct LevelsBindings {
    static constexpr uint32_t input = 0, levels = 1, limits = 2, arguments = 3;
};
} // namespace encodings
} // namespace kernels
} // namespace detail
} // namespace tcnn
