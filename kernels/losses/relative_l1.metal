// RelativeL1 baseline. Change this implementation without changing its traits or ABI.
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
    const float d = p - t, n = float(args.target.rows) * float(args.target.cols);
    const float scale = 1.0f / (abs(p) + 0.01f) / pdf;
    return float2(abs(d) * scale / n, copysign(scale, d) / n);
}

// Values accumulate in FP32; the final gradient is stored in float.
kernel void tcnn_loss_relative_l1_f32(
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
kernel void tcnn_loss_relative_l1_f16(
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
