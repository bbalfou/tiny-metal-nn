// Canonical first-order encoding operations. Compile-time constants specialize
// geometry; batches, matrix strides, level controls and scales retain ABI v1.
#ifndef TCNN_ENCODING_KIND
#define TCNN_ENCODING_KIND 1
#endif
#ifndef TCNN_ENCODING_DIMS
#define TCNN_ENCODING_DIMS 2
#endif
#ifndef TCNN_ENCODING_FEATURES
#define TCNN_ENCODING_FEATURES 2
#endif
#ifndef TCNN_ENCODING_DEGREE
#define TCNN_ENCODING_DEGREE 1
#endif
// Layout and traversal variants specialize a shared Metal library. Geometry
// remains template-specialized until the policy instantiation design is settled.
#include "first_order.metal"
constant bool TCNN_ENCODING_SOA [[function_constant(0)]];
constant uint TCNN_ENCODING_OUTPUT_ROWS [[function_constant(1)]];
constant bool TCNN_ENCODING_SIMD [[function_constant(2)]];
#include "analytic_simd.metal"
using namespace tcnn_encoding;

template <class T>
inline void encoding_forward_impl(constant Args& a, device const float* input, device const T* parameters,
                                  device const float* limits, device T* output, uint2 position, uint lane) {
    if (TCNN_ENCODING_KIND == 6) {
        GridPolicy<TCNN_ENCODING_DIMS, TCNN_ENCODING_FEATURES>::template forward<T>(a, input, parameters, limits, output, position.x);
    } else if (TCNN_ENCODING_KIND == 7) {
        PermutoPolicy<TCNN_ENCODING_FEATURES>::template forward<T>(a, input, parameters, limits, output, position.x);
    } else {
        if (TCNN_ENCODING_SIMD) {
            analytic_forward<T, TCNN_ENCODING_KIND, TCNN_ENCODING_DEGREE>(
                a, input, output, position, lane, TCNN_ENCODING_SOA, TCNN_ENCODING_OUTPUT_ROWS);
        } else {
            const uint sample = TCNN_ENCODING_SOA ? position.x : position.x / TCNN_ENCODING_OUTPUT_ROWS;
            const uint feature = TCNN_ENCODING_SOA ? position.y : position.x % TCNN_ENCODING_OUTPUT_ROWS;
            if (sample < a.x.cols && feature < a.y.rows)
                output[at(a.y, feature, sample)] = T(analytic_value<TCNN_ENCODING_KIND, TCNN_ENCODING_DEGREE>(a, input, feature, sample));
        }
    }
}

template <class T>
inline void encoding_input_gradient_impl(constant Args& a, device const float* input, device const T* parameters,
                                         device const float* limits, device const T* upstream,
                                         device float* gradient, uint index, uint lane) {
    if (TCNN_ENCODING_KIND == 6) {
        GridPolicy<TCNN_ENCODING_DIMS, TCNN_ENCODING_FEATURES>::template backward_input<T>(
            a, input, parameters, limits, upstream, gradient, index);
    } else if (TCNN_ENCODING_KIND == 7) {
        PermutoPolicy<TCNN_ENCODING_FEATURES>::template backward_input<T>(a, input, parameters, limits, upstream, gradient, index);
    } else {
#if TCNN_ENCODING_GRADIENT_SOA
        analytic_input_gradient<T, TCNN_ENCODING_KIND, TCNN_ENCODING_DEGREE>(a, input, upstream, gradient, index);
#else
        if (TCNN_ENCODING_SIMD) {
            analytic_backward<T, TCNN_ENCODING_KIND, 8, TCNN_ENCODING_DEGREE>(a, input, upstream, gradient, index, lane);
        } else {
            analytic_input_gradient<T, TCNN_ENCODING_KIND, TCNN_ENCODING_DEGREE>(a, input, upstream, gradient, index);
        }
#endif
    }
}

#define ENCODING_FIRST_ORDER_ENTRIES(T, S) \
kernel void encoding_forward_##S( \
    device const float* input [[buffer(0)]], device const T* parameters [[buffer(1)]], \
    device const float* limits [[buffer(2)]], device T* output [[buffer(3)]], constant Args& a [[buffer(4)]], \
    uint2 position [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    encoding_forward_impl(a, input, parameters, limits, output, position, lane); \
} \
kernel void encoding_input_gradient_##S( \
    device const float* input [[buffer(0)]], device const T* parameters [[buffer(1)]], device const float* limits [[buffer(2)]], \
    device const T* upstream [[buffer(3)]], device float* gradient [[buffer(4)]], device const float* direction [[buffer(5)]], \
    constant Args& a [[buffer(6)]], uint index [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    encoding_input_gradient_impl(a, input, parameters, limits, upstream, gradient, index, lane); \
} \
kernel void encoding_parameter_gradient_##S( \
    device const float* input [[buffer(0)]], device const float* limits [[buffer(1)]], device const T* upstream [[buffer(2)]], \
    device float* gradient [[buffer(3)]], device const float* direction [[buffer(4)]], constant Args& a [[buffer(5)]], \
    uint sample [[thread_position_in_grid]]) { \
    if (TCNN_ENCODING_KIND == 7) \
        PermutoPolicy<TCNN_ENCODING_FEATURES>::template backward_params<T>(a, input, limits, upstream, gradient, sample); \
    else if (TCNN_ENCODING_KIND == 6) \
        GridPolicy<TCNN_ENCODING_DIMS, TCNN_ENCODING_FEATURES>::template backward_params<T>(a, input, limits, upstream, gradient, sample); \
}
ENCODING_FIRST_ORDER_ENTRIES(float, f32)
ENCODING_FIRST_ORDER_ENTRIES(half, f16)
