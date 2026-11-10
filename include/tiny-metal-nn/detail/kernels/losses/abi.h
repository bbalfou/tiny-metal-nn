// Stable host/device contract for baseline loss kernels. Version changes require a new ABI.
#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tcnn {
namespace detail {
namespace loss_kernels {
inline constexpr uint32_t ABI_VERSION = 1;

// Strides count elements, not bytes. A bound view begins at its logical (0, 0).
struct LossMatrix {
    uint32_t rows, cols, stride_i, stride_j;
};

// Keep kind in the original position. A specialized entry point fixes the
// objective; kind records the matching LossKind for host diagnostics.
struct LossArgs {
    LossMatrix prediction, target, values, gradients, pdf;
    uint32_t kind, has_pdf;
    float loss_scale;
};

struct Bindings {
    static constexpr uint32_t prediction = 0;
    static constexpr uint32_t target = 1;
    static constexpr uint32_t values = 2;
    static constexpr uint32_t gradients = 3;
    static constexpr uint32_t pdf = 4;
    static constexpr uint32_t constant_gradient = 5;
    static constexpr uint32_t arguments = 6;
};

static_assert(std::is_standard_layout<LossMatrix>::value &&
                  std::is_trivially_copyable<LossMatrix>::value,
              "loss matrix ABI");
static_assert(std::is_standard_layout<LossArgs>::value &&
                  std::is_trivially_copyable<LossArgs>::value,
              "loss argument ABI");
static_assert(sizeof(LossMatrix) == 16 && alignof(LossMatrix) == 4,
              "loss matrix ABI size/alignment");
static_assert(sizeof(LossArgs) == 92 && alignof(LossArgs) == 4, "loss argument ABI size/alignment");
static_assert(offsetof(LossArgs, prediction) == 0 && offsetof(LossArgs, target) == 16,
              "loss input ABI offsets");
static_assert(offsetof(LossArgs, values) == 32 && offsetof(LossArgs, gradients) == 48,
              "loss output ABI offsets");
static_assert(offsetof(LossArgs, pdf) == 64 && offsetof(LossArgs, kind) == 80,
              "loss PDF/kind ABI offsets");
static_assert(offsetof(LossArgs, has_pdf) == 84 && offsetof(LossArgs, loss_scale) == 88,
              "loss scalar ABI offsets");
} // namespace loss_kernels
} // namespace detail
} // namespace tcnn
