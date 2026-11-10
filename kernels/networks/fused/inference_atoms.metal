#ifndef TCNN_FUSED_INFERENCE_ATOMS
#define TCNN_FUSED_INFERENCE_ATOMS
#include "atoms.metal"

#ifdef __METAL_VERSION__
namespace tcnn_fused {

// Exact compiler-derived element mapping used by metal2/_fp32_load.py. Do not
// replace it with assumptions about the mapping from a SIMD lane to a matrix.
constant float inference_coordinates[64] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
    32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,
    48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63
};

template <class T> struct InferenceDeviceLoad;
template <> struct InferenceDeviceLoad<half> {
    static TCNN_FUSED_INLINE void load(thread simdgroup_half8x8& value,
        device const half* pointer, unsigned long stride, bool transpose) {
        simdgroup_load(value, pointer, stride, ulong2(0), transpose);
    }
};
template <> struct InferenceDeviceLoad<float> {
    static TCNN_FUSED_INLINE void load(thread simdgroup_float8x8& value,
        device const float* pointer, unsigned long stride, bool transpose) {
        simdgroup_float8x8 coordinates;
        coordinates.thread_elements() = __metal_simdgroup_matrix_8x8_load(
            inference_coordinates + 0, 8ul, ulong2(0), false, 0);
        for (unsigned e = 0; e < 2; ++e) {
            const unsigned index = unsigned(coordinates.thread_elements()[e]);
            const unsigned r = index / 8, c = index % 8;
            value.thread_elements()[e] = transpose ? pointer[c * stride + r] : pointer[r * stride + c];
        }
    }
};

template <class T>
struct InferenceAtom {
    using Scalar = T;
    using Base = SimdgroupMmaAtom<T>;
    using Operand = typename Base::Operand;
    using Accumulator = typename Base::Accumulator;
    static TCNN_FUSED_INLINE void zero(thread Accumulator& c) { Base::zero(c); }
    static TCNN_FUSED_INLINE void mma(thread Accumulator& c, thread Operand& a, thread Operand& b) {
        Base::mma(c, a, b);
    }
    static TCNN_FUSED_INLINE void load_shared(thread Operand& a, threadgroup const T* p,
        unsigned offset, unsigned stride, bool transpose = false) {
        simdgroup_load(a, p + offset, stride, ulong2(0), transpose);
    }
    static TCNN_FUSED_INLINE void load_device(thread Operand& a, device const T* p,
        unsigned long offset, unsigned stride, bool transpose = false) {
        InferenceDeviceLoad<T>::load(a, p + offset, stride, transpose);
    }
    static TCNN_FUSED_INLINE void store_shared(thread const Operand& a, threadgroup T* p,
        unsigned offset, unsigned stride) {
        simdgroup_store(a, p + offset, stride, ulong2(0), false);
    }
    static TCNN_FUSED_INLINE void store_device(thread const Operand& a, device T* p,
        unsigned long offset, unsigned stride, bool transpose = false) {
        simdgroup_store(a, p + offset, stride, ulong2(0), transpose);
    }
    template <class Policy>
    static TCNN_FUSED_INLINE void convert(thread Operand& a, thread Accumulator& c) {
        Base::template convert<Policy>(a, c);
    }
};

template <class T>
struct InferenceCopy {
    enum : unsigned { VectorElements = 16 / sizeof(T) };
    static TCNN_FUSED_INLINE void to_shared(threadgroup T* destination, unsigned offset,
        device const T* source, unsigned long source_offset) {
        *reinterpret_cast<threadgroup uint4*>(destination + offset) =
            *reinterpret_cast<device const uint4*>(source + source_offset);
    }
    static TCNN_FUSED_INLINE void barrier() { threadgroup_barrier(mem_flags::mem_threadgroup); }
};

} // namespace tcnn_fused
#endif
#endif
