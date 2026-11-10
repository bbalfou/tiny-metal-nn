// Required higher-order and utility entrypoints; first-order operations use forward_backward.metal.
// Buffer slots are mirrored by detail/kernels/encodings/abi.h.
#include "encodings/higher_order.metal"
using namespace tcnn_encoding_math;

#define ENCODING_VALUE_ENTRIES(P, T, S)                                                                                                    \
    kernel void P##_input_second_gradient_##S(                                                                                                                \
        device const float* x [[buffer(0)]],                                                                                               \
        device const T* params [[buffer(1)]],                                                                                              \
        device const float* limits [[buffer(2)]],                                                                                          \
        device const T* dy [[buffer(3)]],                                                                                                  \
        device float* dx [[buffer(4)]],                                                                                                    \
        device const float* ddx [[buffer(5)]],                                                                                             \
        constant Args& a [[buffer(6)]],                                                                                                    \
        uint i [[thread_position_in_grid]]                                                                                                 \
    ) {                                                                                                                                    \
        input_second_gradient(                                                                                                                    \
            a,                                                                                                                             \
            x,                                                                                                                             \
            params,                                                                                                                        \
            limits,                                                                                                                        \
            dy,                                                                                                                            \
            dx,                                                                                                                            \
            ddx,                                                                                                                           \
            i                                                                                                                              \
        );                                                                                                                                 \
    }                                                                                                                                      \
    kernel void P##_output_gradient_##S(                                                                                                               \
        device const float* x [[buffer(0)]],                                                                                               \
        device const T* params [[buffer(1)]],                                                                                              \
        device const float* limits [[buffer(2)]],                                                                                          \
        device const float* ddx [[buffer(3)]],                                                                                             \
        device T* ddy [[buffer(4)]],                                                                                                       \
        constant Args& a [[buffer(5)]],                                                                                                    \
        uint i [[thread_position_in_grid]]                                                                                                 \
    ) {                                                                                                                                    \
        backward_output(                                                                                                                   \
            a,                                                                                                                             \
            x,                                                                                                                             \
            params,                                                                                                                        \
            limits,                                                                                                                        \
            ddx,                                                                                                                           \
            ddy,                                                                                                                           \
            i                                                                                                                              \
        );                                                                                                                                 \
    }

#define ENCODING_PARAMETER_ENTRIES(P, T, S)                                                                                                \
    kernel void P##_parameter_second_gradient_##S(                                                                                                                \
        device const float* x [[buffer(0)]],                                                                                               \
        device const float* limits [[buffer(1)]],                                                                                          \
        device const T* dy [[buffer(2)]],                                                                                                  \
        device float* gradients [[buffer(3)]],                                                                                             \
        device const float* ddx [[buffer(4)]],                                                                                             \
        constant Args& a [[buffer(5)]],                                                                                                    \
        uint i [[thread_position_in_grid]]                                                                                                 \
    ) {                                                                                                                                    \
        parameter_second_gradient(                                                                                                                   \
            a,                                                                                                                             \
            x,                                                                                                                             \
            limits,                                                                                                                        \
            dy,                                                                                                                            \
            gradients,                                                                                                                     \
            ddx,                                                                                                                           \
            i                                                                                                                              \
        );                                                                                                                                 \
    }

#define ENCODING_UTILITY_ENTRIES(P, T, S)                                                                                                  \
    kernel void P##_utility_##S(                                                                                                           \
        device const T* src [[buffer(0)]],                                                                                                 \
        device T* dst [[buffer(1)]],                                                                                                       \
        device const T* aux [[buffer(2)]],                                                                                                 \
        device const float* limits [[buffer(3)]],                                                                                          \
        constant UtilityArgs& a [[buffer(4)]],                                                                                             \
        uint i [[thread_position_in_grid]]                                                                                                 \
    ) {                                                                                                                                    \
        utility(                                                                                                                           \
            a,                                                                                                                             \
            src,                                                                                                                           \
            dst,                                                                                                                           \
            aux,                                                                                                                           \
            limits,                                                                                                                        \
            i                                                                                                                              \
        );                                                                                                                                 \
    }                                                                                                                                      \
    kernel void P##_product_second_##S(                                                                                                    \
        device const T* dy [[buffer(0)]],                                                                                                  \
        device const T* values [[buffer(1)]],                                                                                              \
        device const T* tangents [[buffer(2)]],                                                                                            \
        device T* dst [[buffer(3)]],                                                                                                       \
        constant UtilityArgs& a [[buffer(4)]],                                                                                             \
        uint i [[thread_position_in_grid]]                                                                                                 \
    ) {                                                                                                                                    \
        product_second(                                                                                                                    \
            a,                                                                                                                             \
            dy,                                                                                                                            \
            values,                                                                                                                        \
            tangents,                                                                                                                      \
            dst,                                                                                                                           \
            i                                                                                                                              \
        );                                                                                                                                 \
    }                                                                                                                                      \
    kernel void P##_cast_gradient_##S(                                                                                                     \
        device const float* src [[buffer(0)]],                                                                                             \
        device T* dst [[buffer(1)]],                                                                                                       \
        constant uint& count [[buffer(2)]],                                                                                                \
        constant uint& accumulate [[buffer(3)]],                                                                                           \
        uint i [[thread_position_in_grid]]                                                                                                 \
    ) {                                                                                                                                    \
        if (i < count)                                                                                                                     \
            dst[i] = T(src[i] + (accumulate ? float(dst[i]) : 0));                                                                         \
    }

#define ENCODING_LEVELS_ENTRY(P)                                                                                                           \
    kernel void P##_levels(                                                                                                                \
        device const float* src [[buffer(0)]],                                                                                             \
        device float* levels [[buffer(1)]],                                                                                                \
        device const float* limits [[buffer(2)]],                                                                                          \
        constant UtilityArgs& a [[buffer(3)]],                                                                                             \
        uint i [[thread_position_in_grid]]                                                                                                 \
    ) {                                                                                                                                    \
        if (i < a.src.cols)                                                                                                                \
            levels[i] = fmin(src[at(a.src, a.index, i)], a.has_limits ? limits[i] : a.value);                                              \
    }
