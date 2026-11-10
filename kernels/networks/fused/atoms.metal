#ifndef TCNN_FUSED_ATOMS_METAL
#define TCNN_FUSED_ATOMS_METAL
#include "traits.metal"

#ifdef __METAL_VERSION__
#include <metal_simdgroup_matrix>

namespace tcnn_fused {

template <> struct ScalarTraits<half> {
    using Scalar = half;
    using Vector = half2;
    using Operand = simdgroup_half8x8;
    using Accumulator = simdgroup_float8x8;
};
template <> struct ScalarTraits<float> {
    using Scalar = float;
    using Vector = float2;
    using Operand = simdgroup_float8x8;
    using Accumulator = simdgroup_float8x8;
};

template <class T>
struct VectorCopyAtom {
    using Scalar = T;
    static TCNN_FUSED_INLINE void copy(threadgroup T* destination, device const T* source, unsigned vector_index) {
        reinterpret_cast<threadgroup uint4*>(destination)[vector_index] =
            reinterpret_cast<device const uint4*>(source)[vector_index];
    }
    static TCNN_FUSED_INLINE void barrier() { threadgroup_barrier(mem_flags::mem_threadgroup); }
};

template <class T, class Policy>
struct FragmentConversion {
    using Types = ScalarTraits<T>;
    static TCNN_FUSED_INLINE void run(thread typename Types::Operand& a, thread typename Types::Accumulator& c) {
        a.thread_elements()[0] = Policy::template apply<T>(c.thread_elements()[0]);
        a.thread_elements()[1] = Policy::template apply<T>(c.thread_elements()[1]);
    }
};

// Keep the supplied paired conversion, half rounding, and two element writes.
// All other/custom activation policies use the same scalar customization point.
template <class T>
struct FragmentConversion<T, Relu> {
    using Types = ScalarTraits<T>;
    static TCNN_FUSED_INLINE void run(thread typename Types::Operand& a, thread typename Types::Accumulator& c) {
        using V = typename Types::Vector;
        V v = max(V(float2(c.thread_elements()[0], c.thread_elements()[1])), V(0));
        a.thread_elements()[0] = v.x;
        a.thread_elements()[1] = v.y;
    }
};

// These are the native hardware atoms. The collective templates depend on this
// small interface, so a host test can inject independent reference atoms while
// compiling and executing the same tile/layer schedule.
template <class T>
struct SimdgroupMmaAtom {
    using Scalar = T;
    using Types = ScalarTraits<T>;
    using Operand = typename Types::Operand;
    using Accumulator = typename Types::Accumulator;
    static TCNN_FUSED_INLINE void load_a(thread Operand& a, device const T* input, unsigned offset, unsigned stride) {
        simdgroup_load(a, input + offset, stride, ulong2(0), false);
    }
    static TCNN_FUSED_INLINE void load_b(thread Operand& b, threadgroup const T* weights, unsigned offset,
                                         unsigned stride, bool transpose) {
        simdgroup_load(b, weights + offset, stride, ulong2(0), transpose);
    }
    static TCNN_FUSED_INLINE void zero(thread Accumulator& c) { c = make_filled_simdgroup_matrix<float, 8, 8>(float(0)); }
    static TCNN_FUSED_INLINE void mma(thread Accumulator& c, thread Operand& a, thread Operand& b) {
        simdgroup_multiply_accumulate(c, a, b, c);
    }
    template <class Policy>
    static TCNN_FUSED_INLINE void convert(thread Operand& a, thread Accumulator& c) {
        FragmentConversion<T, Policy>::run(a, c);
    }
    static TCNN_FUSED_INLINE void store(thread const Operand& a, device T* output, unsigned offset, unsigned stride) {
        simdgroup_store(a, output + offset, stride, ulong2(0), false);
    }
};

} // namespace tcnn_fused
#endif // __METAL_VERSION__
#endif
