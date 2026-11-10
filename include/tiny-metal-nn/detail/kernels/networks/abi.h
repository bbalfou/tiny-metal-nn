// Frozen scalar-packed network kernel ABI, version 1.
#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tcnn {
namespace detail {
namespace network_kernels {
inline constexpr uint32_t ABI_VERSION = 1;
enum class Path : uint32_t { Gemm, FullyFused };

struct Mat {
    uint32_t rows, cols, rs, cs, precision;
};

struct Dense {
    Mat x, pre, y;
    uint32_t inputs, outputs, activation;
};

struct Delta {
    Mat values, gradient, delta;
    uint32_t activation, values_are_output;
};

struct SecondDelta {
    Mat pre, output, tangent, adjoint, tangent_adjoint, delta, tangent_delta;
    uint32_t activation;
};

struct InputGradient {
    Mat delta, dx;
    uint32_t inputs, outputs, precision;
};

struct WeightGradient {
    Mat x, delta;
    uint32_t inputs, outputs, precision, mode;
};

struct Copy {
    Mat src, dst;
};

struct Unary {
    uint32_t count, width, dim, mode, precision;
    float value;
};

struct ActivationArgs {
    uint32_t count, precision, activation, mode;
};

struct MatrixActivation {
    Mat values, gradient, result;
    uint32_t activation, mode;
};

struct Extract {
    uint32_t count, dim, fan_in, fan_out, layout, precision;
};

// Explicit binding tables accompany the POD records so optimized implementations
// can replace the same entrypoint without changing host-side argument packing.
struct DenseForwardBindings {
    static constexpr uint32_t input = 0, weights = 1, preactivation = 2, output = 3, arguments = 4;
};

struct DeltaBindings {
    static constexpr uint32_t values = 0, output_gradient = 1, delta = 2, arguments = 3;
};

struct DenseInputGradientBindings {
    static constexpr uint32_t delta = 0, weights = 1, input_gradient = 2, arguments = 3;
};

struct DenseParameterGradientBindings {
    static constexpr uint32_t input = 0, delta = 1, parameter_gradient = 2, arguments = 3;
};

struct SecondDeltaBindings {
    static constexpr uint32_t preactivation = 0, output = 1, tangent_preactivation = 2, adjoint = 3,
                              tangent_adjoint = 4, delta = 5, tangent_delta = 6, arguments = 7;
};

static_assert(sizeof(Mat) == 20 && sizeof(Dense) == 72 && sizeof(Delta) == 68 && sizeof(Copy) == 40,
              "MSL constant layout mismatch");
static_assert(sizeof(SecondDelta) == 144 && offsetof(SecondDelta, activation) == 140,
              "MSL second derivative layout mismatch");
static_assert(sizeof(MatrixActivation) == 68 && offsetof(MatrixActivation, mode) == 64,
              "MSL matrix activation layout mismatch");
static_assert(sizeof(InputGradient) == 52 && sizeof(WeightGradient) == 56 && sizeof(Unary) == 24 &&
                  sizeof(ActivationArgs) == 16 && sizeof(Extract) == 24,
              "MSL helper constant layout mismatch");
static_assert(offsetof(Mat, precision) == 16 && offsetof(Dense, inputs) == 60 &&
                  offsetof(Delta, activation) == 60 && offsetof(InputGradient, precision) == 48 &&
                  offsetof(WeightGradient, mode) == 52 && offsetof(Unary, value) == 20,
              "MSL constant field offsets must match scalar-packed shader structs");

struct FusedShape {
    uint32_t inputs, width, outputs, hidden_layers, activation, output_activation;
};

// All matrix descriptors are element strides; precision is sizeof(storage T).
// Saved activations are layer-major, each layer a column-major [width,batch].
struct FusedArgs {
    Mat x, output, incoming, direction, dx, ddy;
    FusedShape shape;
    uint32_t mode, write_dx, write_ddy;
};

static_assert(sizeof(FusedShape) == 24 && sizeof(FusedArgs) == 156, "fused MLP constant ABI");
static_assert(alignof(Mat) == 4 && alignof(FusedShape) == 4 && alignof(FusedArgs) == 4,
              "network constant alignment ABI");
static_assert(offsetof(FusedArgs, shape) == 120 && offsetof(FusedArgs, mode) == 144,
              "fused MLP constant offsets");
static_assert(std::is_standard_layout<FusedArgs>::value &&
                  std::is_trivially_copyable<FusedArgs>::value,
              "shader ABI must be POD");
} // namespace network_kernels
} // namespace detail
} // namespace tcnn
