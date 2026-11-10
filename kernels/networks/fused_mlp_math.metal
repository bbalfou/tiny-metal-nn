// Scalar whole-model algorithms. Shared with the explicitly selected Reference
// callback, as for the encoding kernels; GemmMLP is an independent layered oracle.
#ifndef TCNN_FUSED_MLP_MATH
#define TCNN_FUSED_MLP_MATH
#ifdef __METAL_VERSION__
#include "common.metal"
#define FUSED_DEVICE device

struct FusedShape {
    uint inputs, width, outputs, hidden_layers, activation, output_activation;
};

struct FusedArgs {
    Mat x, output, incoming, direction, dx, ddy;
    FusedShape shape;
    uint mode, write_dx, write_ddy;
};

static_assert(sizeof(FusedShape) == 24, "FusedShape ABI mismatch");
static_assert(sizeof(FusedArgs) == 156, "FusedArgs ABI mismatch");
#else
#define FUSED_DEVICE
namespace tcnn {
namespace detail {
namespace fused_math {
using uint = uint32_t;
using uchar = uint8_t;
using network_kernels::Mat;
using network_kernels::FusedShape;
using network_kernels::FusedArgs;

inline uint index(Mat m, uint row, uint col) {
    return row * m.rs + col * m.cs;
}

inline float read_value(const uchar* p, uint i, uint precision) {
    return detail::read(p, i, precision);
}

inline void write_value(
    uchar* p,
    uint i,
    uint precision,
    float value
) {
    detail::write(
        p,
        i,
        precision,
        value
    );
}
#endif

inline uint fused_rows(FusedShape s, uint layer) {
    return layer == s.hidden_layers ? s.outputs : s.width;
}

inline uint fused_inputs(FusedShape s, uint layer) {
    return layer == 0 ? s.inputs : s.width;
}

inline uint fused_weight_offset(FusedShape s, uint layer) {
    return layer == 0 ? 0 : s.inputs * s.width + (layer - 1) * s.width * s.width;
}

inline uint fused_saved_index(
    FusedArgs a,
    uint layer,
    uint row,
    uint col
) {
    return layer * a.shape.width * a.x.cols + col * fused_rows(a.shape, layer) + row;
}

inline uint fused_activation(FusedShape s, uint layer) {
    return layer == s.hidden_layers ? s.output_activation : s.activation;
}

inline bool fused_input_form(uint activation) {
    return activation == 2 || activation == 4;
}

inline void fused_forward_sample(
    FusedArgs a,
    FUSED_DEVICE const uchar* x,
    FUSED_DEVICE const uchar* weights,
    FUSED_DEVICE uchar* pre,
    FUSED_DEVICE uchar* hidden,
    uint col
) {
    if (col >= a.x.cols)
        return;
    const uint precision = a.x.precision;
    for (uint layer = 0; layer <= a.shape.hidden_layers; ++layer) {
        const uint rows = fused_rows(a.shape, layer), inputs = fused_inputs(a.shape, layer);
        const uint weight_offset = fused_weight_offset(a.shape, layer);
        for (uint row = 0; row < rows; ++row) {
            float sum = 0;
            for (uint k = 0; k < inputs; ++k) {
                const float v = layer == 0 ? read_value(x, index(a.x, k, col), precision)
                                           : read_value(
                                                 hidden,
                                                 fused_saved_index(
                                                     a,
                                                     layer - 1,
                                                     k,
                                                     col
                                                 ),
                                                 precision
                                             );
                sum += read_value(weights, weight_offset + row * inputs + k, precision) * v;
            }
            const uint out = fused_saved_index(
                a,
                layer,
                row,
                col
            );
            sum = round_to_precision(sum, precision);
            write_value(
                pre,
                out,
                precision,
                sum
            );
            write_value(
                hidden,
                out,
                precision,
                activate(sum, fused_activation(a.shape, layer), precision)
            );
        }
    }
}

inline void fused_backward_sample(
    FusedArgs a,
    FUSED_DEVICE const uchar* weights,
    FUSED_DEVICE const uchar* pre,
    FUSED_DEVICE const uchar* hidden,
    FUSED_DEVICE const uchar* output,
    FUSED_DEVICE const uchar* incoming,
    FUSED_DEVICE uchar* delta,
    FUSED_DEVICE uchar* dx,
    uint col
) {
    if (col >= a.x.cols)
        return;
    const uint precision = a.x.precision;
    for (uint reverse = 0; reverse <= a.shape.hidden_layers; ++reverse) {
        const uint layer = a.shape.hidden_layers - reverse, rows = fused_rows(a.shape, layer);
        const uint activation = fused_activation(a.shape, layer);
        for (uint row = 0; row < rows; ++row) {
            float gradient = 0;
            if (layer == a.shape.hidden_layers)
                gradient = read_value(incoming, index(a.incoming, row, col), precision);
            else {
                const uint next_rows = fused_rows(a.shape, layer + 1), next_weights = fused_weight_offset(a.shape, layer + 1);
                for (uint k = 0; k < next_rows; ++k)
                    gradient += read_value(weights, next_weights + k * rows + row, precision) * read_value(
                                                                                                    delta,
                                                                                                    fused_saved_index(
                                                                                                        a,
                                                                                                        layer + 1,
                                                                                                        k,
                                                                                                        col
                                                                                                    ),
                                                                                                    precision
                                                                                                );
                gradient = round_to_precision(gradient, precision);
            }
            const uint saved = fused_saved_index(
                a,
                layer,
                row,
                col
            );
            const bool input_form = fused_input_form(activation);
            const float value = input_form                       ? read_value(pre, saved, precision)
                                : layer == a.shape.hidden_layers ? read_value(output, index(a.output, row, col), precision)
                                                                 : read_value(hidden, saved, precision);
            write_value(
                delta,
                saved,
                precision,
                activation_gradient(
                    value,
                    gradient,
                    activation,
                    precision,
                    !input_form
                )
            );
        }
    }
    if (a.write_dx)
        for (uint row = 0; row < a.shape.inputs; ++row) {
            float sum = 0;
            for (uint k = 0; k < fused_rows(a.shape, 0); ++k)
                sum += read_value(weights, k * a.shape.inputs + row, precision) * read_value(
                                                                                      delta,
                                                                                      fused_saved_index(
                                                                                          a,
                                                                                          0,
                                                                                          k,
                                                                                          col
                                                                                      ),
                                                                                      precision
                                                                                  );
            write_value(
                dx,
                index(a.dx, row, col),
                precision,
                sum
            );
        }
}

#ifndef __METAL_VERSION__
// Independent scalar parameter oracle for the explicit Reference backend only.
// Each reduction thread owns one parameter across the entire batch. No atomic
// float support, inter-thread synchronization, or dynamic shared storage is needed.
inline void fused_parameter_reference(
    FusedArgs a,
    FUSED_DEVICE const uchar* x,
    FUSED_DEVICE const uchar* hidden,
    FUSED_DEVICE const uchar* delta,
    FUSED_DEVICE uchar* gradient,
    uint parameter
) {
    const uint precision = a.x.precision;
    for (uint layer = 0; layer <= a.shape.hidden_layers; ++layer) {
        const uint start = fused_weight_offset(a.shape, layer), inputs = fused_inputs(a.shape, layer);
        const uint count = inputs * fused_rows(a.shape, layer);
        if (parameter < start || parameter - start >= count)
            continue;
        const uint row = (parameter - start) / inputs, k = (parameter - start) % inputs;
        float sum = 0;
        for (uint col = 0; col < a.x.cols; ++col) {
            const float v = layer == 0 ? read_value(x, index(a.x, k, col), precision)
                                       : read_value(
                                             hidden,
                                             fused_saved_index(
                                                 a,
                                                 layer - 1,
                                                 k,
                                                 col
                                             ),
                                             precision
                                         );
            sum += read_value(
                       delta,
                       fused_saved_index(
                           a,
                           layer,
                           row,
                           col
                       ),
                       precision
                   ) *
                   v;
        }
        if (a.mode == 2)
            sum += read_value(gradient, parameter, precision);
        write_value(
            gradient,
            parameter,
            precision,
            sum
        );
        return;
    }
}

#endif

inline void fused_tangent_sample(
    FusedArgs a,
    FUSED_DEVICE const uchar* direction,
    FUSED_DEVICE const uchar* weights,
    FUSED_DEVICE const uchar* pre,
    FUSED_DEVICE const uchar* hidden,
    FUSED_DEVICE uchar* tangent_pre,
    FUSED_DEVICE uchar* tangent,
    FUSED_DEVICE uchar* ddy,
    uint col
) {
    if (col >= a.x.cols)
        return;
    const uint precision = a.x.precision;
    for (uint layer = 0; layer <= a.shape.hidden_layers; ++layer) {
        const uint rows = fused_rows(a.shape, layer), inputs = fused_inputs(a.shape, layer);
        const uint weight_offset = fused_weight_offset(a.shape, layer), activation = fused_activation(a.shape, layer);
        for (uint row = 0; row < rows; ++row) {
            float sum = 0;
            for (uint k = 0; k < inputs; ++k) {
                const float v = layer == 0 ? read_value(direction, index(a.direction, k, col), precision)
                                           : read_value(
                                                 tangent,
                                                 fused_saved_index(
                                                     a,
                                                     layer - 1,
                                                     k,
                                                     col
                                                 ),
                                                 precision
                                             );
                sum += read_value(weights, weight_offset + row * inputs + k, precision) * v;
            }
            const uint saved = fused_saved_index(
                a,
                layer,
                row,
                col
            );
            sum = round_to_precision(sum, precision);
            write_value(
                tangent_pre,
                saved,
                precision,
                sum
            );
            const bool input_form = fused_input_form(activation);
            const float value = read_value(input_form ? pre : hidden, saved, precision);
            const float result = activation_gradient(
                value,
                sum,
                activation,
                precision,
                !input_form
            );
            write_value(
                tangent,
                saved,
                precision,
                result
            );
            if (a.write_ddy && layer == a.shape.hidden_layers)
                write_value(
                    ddy,
                    index(a.ddy, row, col),
                    precision,
                    result
                );
        }
    }
}

inline void fused_second_backward_sample(
    FusedArgs a,
    FUSED_DEVICE const uchar* weights,
    FUSED_DEVICE const uchar* pre,
    FUSED_DEVICE const uchar* hidden,
    FUSED_DEVICE const uchar* tangent_pre,
    FUSED_DEVICE const uchar* incoming,
    FUSED_DEVICE uchar* delta,
    FUSED_DEVICE uchar* tangent_delta,
    FUSED_DEVICE uchar* dx,
    uint col
) {
    if (col >= a.x.cols)
        return;
    const uint precision = a.x.precision;
    for (uint reverse = 0; reverse <= a.shape.hidden_layers; ++reverse) {
        const uint layer = a.shape.hidden_layers - reverse, rows = fused_rows(a.shape, layer);
        const uint activation = fused_activation(a.shape, layer);
        for (uint row = 0; row < rows; ++row) {
            float adjoint = 0, tangent_adjoint = 0;
            if (layer == a.shape.hidden_layers)
                tangent_adjoint = read_value(incoming, index(a.incoming, row, col), precision);
            else {
                const uint next_rows = fused_rows(a.shape, layer + 1), next_weights = fused_weight_offset(a.shape, layer + 1);
                for (uint k = 0; k < next_rows; ++k) {
                    const float weight = read_value(weights, next_weights + k * rows + row, precision);
                    const uint next = fused_saved_index(
                        a,
                        layer + 1,
                        k,
                        col
                    );
                    adjoint += weight * read_value(delta, next, precision);
                    tangent_adjoint += weight * read_value(tangent_delta, next, precision);
                }
                adjoint = round_to_precision(adjoint, precision);
                tangent_adjoint = round_to_precision(tangent_adjoint, precision);
            }
            const uint saved = fused_saved_index(
                a,
                layer,
                row,
                col
            );
            const float x = read_value(pre, saved, precision), y = read_value(hidden, saved, precision);
            const float first = round_to_precision(
                fused_input_form(activation) ? derivative(x, activation) : derivative_output(y, activation, precision), precision
            );
            const float second = round_to_precision(
                second_derivative(
                    x,
                    y,
                    activation,
                    precision
                ),
                precision
            );
            const float t = read_value(tangent_pre, saved, precision);
            write_value(
                delta,
                saved,
                precision,
                adjoint * first + tangent_adjoint * second * t
            );
            write_value(
                tangent_delta,
                saved,
                precision,
                tangent_adjoint * first
            );
        }
    }
    if (a.write_dx)
        for (uint row = 0; row < a.shape.inputs; ++row) {
            float sum = 0;
            for (uint k = 0; k < fused_rows(a.shape, 0); ++k)
                sum += read_value(weights, k * a.shape.inputs + row, precision) * read_value(
                                                                                      delta,
                                                                                      fused_saved_index(
                                                                                          a,
                                                                                          0,
                                                                                          k,
                                                                                          col
                                                                                      ),
                                                                                      precision
                                                                                  );
            write_value(
                dx,
                index(a.dx, row, col),
                precision,
                sum
            );
        }
}

#ifndef __METAL_VERSION__
inline void fused_second_parameter_reference(
    FusedArgs a,
    FUSED_DEVICE const uchar* x,
    FUSED_DEVICE const uchar* direction,
    FUSED_DEVICE const uchar* hidden,
    FUSED_DEVICE const uchar* tangent,
    FUSED_DEVICE const uchar* delta,
    FUSED_DEVICE const uchar* tangent_delta,
    FUSED_DEVICE uchar* gradient,
    uint parameter
) {
    const uint precision = a.x.precision;
    for (uint layer = 0; layer <= a.shape.hidden_layers; ++layer) {
        const uint start = fused_weight_offset(a.shape, layer), inputs = fused_inputs(a.shape, layer);
        const uint count = inputs * fused_rows(a.shape, layer);
        if (parameter < start || parameter - start >= count)
            continue;
        const uint row = (parameter - start) / inputs, k = (parameter - start) % inputs;
        float primal_sum = 0, tangent_sum = 0;
        for (uint col = 0; col < a.x.cols; ++col) {
            const float v = layer == 0 ? read_value(x, index(a.x, k, col), precision)
                                       : read_value(
                                             hidden,
                                             fused_saved_index(
                                                 a,
                                                 layer - 1,
                                                 k,
                                                 col
                                             ),
                                             precision
                                         );
            const float t = layer == 0 ? read_value(direction, index(a.direction, k, col), precision)
                                       : read_value(
                                             tangent,
                                             fused_saved_index(
                                                 a,
                                                 layer - 1,
                                                 k,
                                                 col
                                             ),
                                             precision
                                         );
            primal_sum += read_value(
                              delta,
                              fused_saved_index(
                                  a,
                                  layer,
                                  row,
                                  col
                              ),
                              precision
                          ) *
                          v;
            tangent_sum += read_value(
                               tangent_delta,
                               fused_saved_index(
                                   a,
                                   layer,
                                   row,
                                   col
                               ),
                               precision
                           ) *
                           t;
        }
        if (a.mode == 2)
            primal_sum += read_value(gradient, parameter, precision);
        // Preserve the two stores in the existing layerwise second-backward ABI.
        primal_sum = round_to_precision(primal_sum, precision);
        write_value(
            gradient,
            parameter,
            precision,
            primal_sum + tangent_sum
        );
        return;
    }
}
#endif
#ifndef __METAL_VERSION__
}
}
}
#endif
#undef FUSED_DEVICE
#endif
