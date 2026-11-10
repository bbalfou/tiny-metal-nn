// Derivative operations required by the public encoding interfaces. These are
// separate from canonical first-order traversal and cannot select old kernels.
#ifndef TCNN_ENCODING_HIGHER_ORDER
#define TCNN_ENCODING_HIGHER_ORDER
#include "math.metal"
#ifdef __METAL_VERSION__
#define EH_CONSTANT constant
#define EH_DEVICE device
#else
#define EH_CONSTANT
#define EH_DEVICE
#endif
namespace tcnn_encoding_math {

template <typename T>
inline void input_second_gradient(EH_CONSTANT Args& a, EH_DEVICE const float* input, EH_DEVICE const T* parameters,
                                  EH_DEVICE const float* limits, EH_DEVICE const T* upstream, EH_DEVICE float* gradient,
                                  EH_DEVICE const float* direction, uint index) {
    if (index >= a.dims * a.x.cols) return;
    const uint dimension = index % a.dims, sample = index / a.dims;
    float result = 0;
    if (a.kind < 6) {
        if (a.kind != 0 && a.kind != 1 && a.kind != 3)
            for (uint feature = 0; feature < a.dy.rows; ++feature)
                result += float(upstream[at(a.dy, feature, sample)]) *
                    analytic(a, input, feature, sample, dimension, direction).dt;
        gradient[at(a.dx, dimension, sample)] = result;
        return;
    }
    if (a.kind == 6 && !inactive(a, limits, sample) && dimension < a.max_input_grad_dims) {
        uint ids[25];
        float weights[25], first[25], second[25];
        for (uint other = 0; other < a.dims; ++other)
            for (uint begin = 0; begin < vertex_count(a); begin += 25) {
                const uint count = geometry(a, input, sample, dimension, other, ids, weights, first, second, begin);
                for (uint feature = 0; feature < a.features; ++feature)
                    for (uint vertex_index = 0; vertex_index < count; ++vertex_index)
                        result += float(upstream[at(a.dy, a.level * a.features + feature, sample)]) * second[vertex_index] *
                            float(parameters[ids[vertex_index] * a.features + feature]) * direction[at(a.ddx, other, sample)];
            }
    }
    const uint address = at(a.dx, dimension, sample);
    gradient[address] = (a.accumulate ? gradient[address] : 0.0f) + result;
}

template <typename T>
inline void parameter_second_gradient(EH_CONSTANT Args& a, EH_DEVICE const float* input, EH_DEVICE const float* limits,
                                      EH_DEVICE const T* upstream, EH_DEVICE float* gradients,
                                      EH_DEVICE const float* direction, uint feature) {
    if (feature >= a.features) return;
    for (uint vertex_index = 0; vertex_index < a.table_size; ++vertex_index)
        gradients[vertex_index * a.features + feature] = 0;
    uint ids[25];
    float weights[25], first[25], second[25];
    for (uint sample = 0; sample < a.x.cols; ++sample) {
        if (inactive(a, limits, sample)) continue;
        const float upstream_value = float(upstream[at(a.dy, a.level * a.features + feature, sample)]);
        for (uint dimension = 0; dimension < a.max_input_grad_dims; ++dimension)
            for (uint begin = 0; begin < vertex_count(a); begin += 25) {
                const uint count = geometry(a, input, sample, dimension, 0xffffffffu, ids, weights, first, second, begin);
                for (uint vertex_index = 0; vertex_index < count; ++vertex_index)
                    gradients[ids[vertex_index] * a.features + feature] += first[vertex_index] * upstream_value *
                        direction[at(a.ddx, dimension, sample)];
            }
    }
}
} // namespace tcnn_encoding_math
#undef EH_CONSTANT
#undef EH_DEVICE
#endif
