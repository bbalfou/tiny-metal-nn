// General GemmMLP baseline. One element/parameter per thread, deterministic reductions.
// Replace these bodies without changing the bindings in detail/kernels/networks/abi.h.
#include "common.metal"

// One matrix output per thread: an intentionally simple replacement point for GEMM.
kernel void gemm_dense(
    device const uchar* x [[buffer(0)]],
    device const uchar* w [[buffer(1)]],
    device uchar* pre [[buffer(2)]],
    device uchar* y [[buffer(3)]],
    constant Dense& a [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= a.outputs * a.y.cols)
        return;
    uint row = i % a.outputs, col = i / a.outputs;
    float sum = 0;
    for (uint k = 0; k < a.inputs; ++k)
        sum += read_value(w, row * a.inputs + k, a.y.precision) * read_value(x, index(a.x, k, col), a.x.precision);
    // Materialize the preactivation in T, matching the ordinary unfused pass.
    if (a.pre.precision == 2)
        sum = float(half(sum));
    write_value(
        pre,
        index(a.pre, row, col),
        a.pre.precision,
        sum
    );
    write_value(
        y,
        index(a.y, row, col),
        a.y.precision,
        activate(sum, a.activation, a.y.precision)
    );
}

// Apply the local chain rule independently at each element.
kernel void gemm_delta(
    device const uchar* values [[buffer(0)]],
    device const uchar* gradient [[buffer(1)]],
    device uchar* delta [[buffer(2)]],
    constant Delta& a [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= a.delta.rows * a.delta.cols)
        return;
    uint row = i % a.delta.rows, col = i / a.delta.rows;
    write_value(
        delta,
        index(a.delta, row, col),
        a.delta.precision,
        activation_gradient(
            read_value(values, index(a.values, row, col), a.values.precision),
            read_value(gradient, index(a.gradient, row, col), a.gradient.precision),
            a.activation,
            a.delta.precision,
            a.values_are_output != 0
        )
    );
}

// Reverse the pair (activation, directional derivative) for general MLP second
// backward. Each thread owns one pair of adjoints; no floating atomics are used.
kernel void gemm_second_delta(
    device const uchar* pre [[buffer(0)]],
    device const uchar* output [[buffer(1)]],
    device const uchar* tangent [[buffer(2)]],
    device const uchar* adjoint [[buffer(3)]],
    device const uchar* tangent_adjoint [[buffer(4)]],
    device uchar* delta [[buffer(5)]],
    device uchar* tangent_delta [[buffer(6)]],
    constant SecondDelta& a [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= a.delta.rows * a.delta.cols)
        return;
    uint row = i % a.delta.rows, col = i / a.delta.rows;
    float x = read_value(pre, index(a.pre, row, col), a.pre.precision),
          y = read_value(output, index(a.output, row, col), a.output.precision);
    bool input_form = a.activation == 2 || a.activation == 4;
    float first = round_to_precision(
        input_form ? derivative(x, a.activation) : derivative_output(y, a.activation, a.delta.precision), a.delta.precision
    );
    float second = round_to_precision(
        second_derivative(
            x,
            y,
            a.activation,
            a.delta.precision
        ),
        a.delta.precision
    );
    float t = read_value(tangent, index(a.tangent, row, col), a.tangent.precision);
    float da = read_value(adjoint, index(a.adjoint, row, col), a.adjoint.precision),
          dt = read_value(tangent_adjoint, index(a.tangent_adjoint, row, col), a.tangent_adjoint.precision);
    write_value(
        delta,
        index(a.delta, row, col),
        a.delta.precision,
        da * first + dt * second * t
    );
    write_value(
        tangent_delta,
        index(a.tangent_delta, row, col),
        a.tangent_delta.precision,
        dt * first
    );
}

// Multiply local derivatives by the transposed weights.
kernel void gemm_dense_dx(
    device const uchar* delta [[buffer(0)]],
    device const uchar* w [[buffer(1)]],
    device uchar* dx [[buffer(2)]],
    constant InputGradient& a [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= a.inputs * a.dx.cols)
        return;
    uint row = i % a.inputs, col = i / a.inputs;
    float sum = 0;
    for (uint k = 0; k < a.outputs; ++k)
        sum += read_value(w, k * a.inputs + row, a.precision) * read_value(delta, index(a.delta, k, col), a.delta.precision);
    write_value(
        dx,
        index(a.dx, row, col),
        a.dx.precision,
        sum
    );
}

// One thread owns a parameter, so gradient accumulation needs no float atomics.
kernel void gemm_dense_dw(
    device const uchar* x [[buffer(0)]],
    device const uchar* delta [[buffer(1)]],
    device uchar* dw [[buffer(2)]],
    constant WeightGradient& a [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= a.inputs * a.outputs)
        return;
    uint row = i / a.inputs, k = i % a.inputs;
    float sum = 0;
    for (uint col = 0; col < a.x.cols; ++col)
        sum += read_value(delta, index(a.delta, row, col), a.delta.precision) * read_value(x, index(a.x, k, col), a.x.precision);
    if (a.mode == 2)
        sum += read_value(dw, i, a.precision);
    write_value(
        dw,
        i,
        a.precision,
        sum
    );
}
