// Mirror of detail/kernels/encodings/abi.h, ABI_VERSION = 1.
// Keep field types, field order, sizes and buffer slots unchanged when optimizing.
#ifndef TCNN_ENCODING_ABI_METAL
#define TCNN_ENCODING_ABI_METAL
#include <metal_stdlib>
using namespace metal;

namespace tcnn_encoding_math {
struct Mat {
    uint rows, cols, rs, cs;
};

struct Args {
    Mat x, y, dy, dx, ddx, ddy;
    uint dims, width, degree, kind, level, levels, features, table_size, resolution, interpolation, grid_type, hash_type,
        max_input_grad_dims, has_limits, mode, accumulate, stochastic;
    float scale, offset, max_level, padding;
    float scales[24], shifts[24];
};

struct UtilityArgs {
    Mat src, dst, aux;
    uint mode, width, index, nested, levels, features, has_limits;
    float value;
};

static_assert(sizeof(Mat) == 16 && alignof(Mat) == 4, "Encoding matrix ABI v1.");
static_assert(sizeof(Args) == 372 && alignof(Args) == 4, "Encoding arguments ABI v1.");
static_assert(sizeof(UtilityArgs) == 80 && alignof(UtilityArgs) == 4, "Encoding utility ABI v1.");
static_assert(__builtin_offsetof(Args, kind) == 108 && __builtin_offsetof(Args, hash_type) == 140, "Encoding selectors ABI v1.");
static_assert(
    __builtin_offsetof(Args, scale) == 164 && __builtin_offsetof(Args, scales) == 180 && __builtin_offsetof(Args, shifts) == 276,
    "Encoding scales ABI v1."
);
static_assert(__builtin_offsetof(UtilityArgs, mode) == 48 && __builtin_offsetof(UtilityArgs, value) == 76, "Encoding utility offsets v1.");
}
#endif
