// RelativeL2Luminance baseline. Change this implementation without changing its traits or ABI.
#include "common.metal"

// Return the scalar value and unscaled gradient for one logical element.
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
) {
    float r = float(prediction[loss_index(args.prediction, 0, col)]);
    float g = float(prediction[loss_index(args.prediction, 1, col)]);
    float b = float(prediction[loss_index(args.prediction, 2, col)]);
    if (args.target.rows >= 6) {
        r += float(prediction[loss_index(args.prediction, 3, col)]);
        g += float(prediction[loss_index(args.prediction, 4, col)]);
        b += float(prediction[loss_index(args.prediction, 5, col)]);
    }
    const float luminance = 0.299f * r + 0.587f * g + 0.114f * b;
    const float d = p - t, denominator = luminance * luminance + 0.01f;
    const float n = float(args.target.rows) * float(args.target.cols);
    return float2(d * d / denominator / pdf / n, 2.0f * d / denominator / pdf / n);
}

// Values accumulate in FP32; the final gradient is stored in float.
kernel void tcnn_loss_relative_l2_luminance_f32(
    device const float* prediction [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* values [[buffer(2)]],
    device float* gradients [[buffer(3)]],
    device const float* pdf [[buffer(4)]],
    device const float* constant_gradient [[buffer(5)]],
    constant LossArgs& args [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
    loss_element(
        prediction,
        target,
        values,
        gradients,
        pdf,
        constant_gradient,
        args,
        i
    );
}

// Values accumulate in FP32; the final gradient is stored in half.
kernel void tcnn_loss_relative_l2_luminance_f16(
    device const half* prediction [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* values [[buffer(2)]],
    device half* gradients [[buffer(3)]],
    device const float* pdf [[buffer(4)]],
    device const float* constant_gradient [[buffer(5)]],
    constant LossArgs& args [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
    loss_element(
        prediction,
        target,
        values,
        gradients,
        pdf,
        constant_gradient,
        args,
        i
    );
}
