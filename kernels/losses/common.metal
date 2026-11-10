// Shared loss ABI and one-element launch plumbing. Objective formulas live in
// separate assets and are selected by entry point, never by a device-side switch.
#ifndef TCNN_LOSS_KERNEL_COMMON
#define TCNN_LOSS_KERNEL_COMMON
#include <metal_stdlib>
using namespace metal;

struct LossMatrix {
    uint rows, cols, stride_i, stride_j;
};

struct LossArgs {
    LossMatrix prediction, target, values, gradients, pdf;
    uint kind, has_pdf;
    float loss_scale;
};

static_assert(sizeof(LossMatrix) == 16, "loss matrix ABI");
static_assert(sizeof(LossArgs) == 92, "loss argument ABI");

inline uint loss_index(LossMatrix matrix, uint row, uint col) {
    return row * matrix.stride_i + col * matrix.stride_j;
}

template <typename T>
inline float2 loss_formula(
    device const T* prediction,
    device const float* constant_gradient,
    constant LossArgs& args,
    uint row,
    uint col,
    float p,
    float t,
    float pdf
);

// All seven bindings are mandatory. The host binds a harmless retained input to
// optional PDF/constant slots when disabled. No padded prediction is evaluated.
template <typename T>
inline void loss_element(
    device const T* prediction,
    device const float* target,
    device float* values,
    device T* gradients,
    device const float* pdf_data,
    device const float* constant_gradient,
    constant LossArgs& args,
    uint i
) {
    if (i >= args.prediction.rows * args.prediction.cols)
        return;
    const uint row = i % args.prediction.rows, col = i / args.prediction.rows;
    float2 result = float2(0.0f);
    if (row < args.target.rows) {
        const float p = float(prediction[loss_index(args.prediction, row, col)]);
        const float t = target[loss_index(args.target, row, col)];
        const float pdf = args.has_pdf ? pdf_data[loss_index(args.pdf, row, col)] : 1.0f;
        result = loss_formula(
            prediction,
            constant_gradient,
            args,
            row,
            col,
            p,
            t,
            pdf
        );
    }
    values[loss_index(args.values, row, col)] = result.x;
    gradients[loss_index(args.gradients, row, col)] = T(args.loss_scale * result.y);
}
#endif
