// Network utility kernels; model kernels have dedicated assets.
#include "networks/common.metal"

// Copy logical entries between layouts and storage precisions, optionally trimming.
kernel void nn_copy(
    device const uchar* src [[buffer(0)]],
    device uchar* dst [[buffer(1)]],
    constant Copy& a [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= a.dst.rows * a.dst.cols)
        return;
    uint row = i % a.dst.rows, col = i / a.dst.rows;
    write_value(
        dst,
        index(a.dst, row, col),
        a.dst.precision,
        read_value(src, index(a.src, row, col), a.src.precision)
    );
}

// Accumulate independent chain-rule contributions into a strided destination.
kernel void nn_accumulate(
    device const uchar* src [[buffer(0)]],
    device uchar* dst [[buffer(1)]],
    constant Copy& a [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= a.dst.rows * a.dst.cols)
        return;
    uint row = i % a.dst.rows, col = i / a.dst.rows;
    write_value(
        dst,
        index(a.dst, row, col),
        a.dst.precision,
        read_value(dst, index(a.dst, row, col), a.dst.precision) + read_value(src, index(a.src, row, col), a.src.precision)
    );
}

// Scale or write a batched one-hot vector for input-gradient helpers.
kernel void nn_unary(device uchar* values [[buffer(0)]], constant Unary& a [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.count)
        return;
    write_value(
        values,
        i,
        a.precision,
        a.mode == 0 ? read_value(values, i, a.precision) * a.value : (i % a.width == a.dim ? a.value : 0.0f)
    );
}

// Matrix helpers preserve row/column strides, including noncontiguous slices.
kernel void nn_matrix_activation(
    device const uchar* values [[buffer(0)]],
    device const uchar* gradients [[buffer(1)]],
    device uchar* out [[buffer(2)]],
    constant MatrixActivation& a [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= a.result.rows * a.result.cols)
        return;
    uint row = i % a.result.rows, col = i / a.result.rows;
    float x = read_value(values, index(a.values, row, col), a.values.precision);
    float value = a.mode == 0 ? activate(x, a.activation, a.result.precision)
                              : activation_gradient(
                                    x,
                                    read_value(gradients, index(a.gradient, row, col), a.gradient.precision),
                                    a.activation,
                                    a.result.precision,
                                    a.mode == 2
                                );
    write_value(
        out,
        index(a.result, row, col),
        a.result.precision,
        value
    );
}

// Standalone activation and derivative helpers retain the reference pointer API.
kernel void nn_activation(
    device const uchar* values [[buffer(0)]],
    device const uchar* gradients [[buffer(1)]],
    device uchar* out [[buffer(2)]],
    constant ActivationArgs& a [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= a.count)
        return;
    float x = read_value(values, i, a.precision);
    float v = a.mode == 0 ? activate(x, a.activation, a.precision)
                          : activation_gradient(
                                x,
                                read_value(gradients, i, a.precision),
                                a.activation,
                                a.precision,
                                a.mode == 2
                            );
    write_value(
        out,
        i,
        a.precision,
        v
    );
}

// Map positive/negative components to signed or red/green visualization channels.
kernel void nn_extract(
    device const uchar* encoded [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant Extract& a [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= a.count)
        return;
    uint row = i % a.fan_out, col = i / a.fan_out, n = a.count / a.fan_out;
    float value = read_value(encoded, a.layout == 1 ? col * a.fan_in + a.dim : a.dim * n + col, a.precision);
    output[i] = a.fan_out == 1 ? value : (row == 0 ? max(-value, 0.0f) : (row == 1 ? max(value, 0.0f) : (row == 2 ? 0.0f : 1.0f)));
}

// Add a separately generated logistic perturbation before evaluating the loss.
kernel void nn_add_perturbation(
    device const uchar* input [[buffer(0)]],
    device const float* perturbation [[buffer(1)]],
    device uchar* output [[buffer(2)]],
    constant ActivationArgs& a [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < a.count)
        write_value(
            output,
            i,
            a.precision,
            read_value(input, i, a.precision) + perturbation[i]
        );
}
