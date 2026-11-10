#ifndef TCNN_FUSED_HIGHER_ORDER_ATOMS_METAL
#define TCNN_FUSED_HIGHER_ORDER_ATOMS_METAL
#include "training_atoms.metal"

#ifdef __METAL_VERSION__
namespace tcnn_fused {

// Packed vectors have scalar alignment. GPUMatrix subviews therefore keep the
// SIMD path even when their first element is not on a sixteen-byte boundary.
template <class T>
using HigherOrderCopy = TrainingPackedCopyAtom<T>;

template <class T>
struct HigherOrderMath {
    template <class Policy>
    static TCNN_FUSED_INLINE T first(T pre, T hidden) {
        const float value = Policy::Kind == 2 || Policy::Kind == 4
            ? derivative(float(pre), Policy::Kind)
            : derivative_output(float(hidden), Policy::Kind, sizeof(T));
        return T(value);
    }
    template <class Policy>
    static TCNN_FUSED_INLINE T second(T pre, T hidden) {
        return T(second_derivative(float(pre), float(hidden), Policy::Kind, sizeof(T)));
    }
};

template <class T>
struct HigherOrderMmaAtom {
    using Base = TrainingMmaAtom<T>;
    using Scalar = T;
    using Operand = typename Base::Operand;
    using Accumulator = typename Base::Accumulator;
    static TCNN_FUSED_INLINE void zero(thread Accumulator& c) { Base::zero(c); }
    static TCNN_FUSED_INLINE void mma(thread Accumulator& c, thread Operand& a, thread Operand& b) {
        Base::mma(c, a, b);
    }
    static TCNN_FUSED_INLINE void load_shared(thread Operand& a, threadgroup const T* source,
                                              unsigned offset, unsigned stride, bool transpose = false) {
        Base::load_shared(a, source, offset, stride, transpose);
    }
    static TCNN_FUSED_INLINE void load_device(thread Operand& a, device const T* source,
                                              ulong offset, unsigned stride) {
        Base::load_device(a, source, offset, stride);
    }
    static TCNN_FUSED_INLINE void store_shared(thread const Operand& a, threadgroup T* destination,
                                               unsigned offset, unsigned stride) {
        Base::store_shared(a, destination, offset, stride);
    }
    static TCNN_FUSED_INLINE void store_device(thread const Operand& a, device T* destination,
                                               ulong offset, unsigned stride) {
        Base::store_device(a, destination, offset, stride);
    }
    template <class Policy>
    static TCNN_FUSED_INLINE void convert(thread Operand& a, thread Accumulator& c) {
        Base::template convert<Policy>(a, c);
    }
    template <class Math, class Policy>
    static TCNN_FUSED_INLINE void tangent(thread Operand& result, thread Operand& tangent_pre,
                                          thread Operand& pre, thread Operand& hidden) {
        for (unsigned i = 0; i < 2; ++i)
            result.thread_elements()[i] = T(float(tangent_pre.thread_elements()[i]) *
                float(Math::template first<Policy>(pre.thread_elements()[i], hidden.thread_elements()[i])));
    }
    template <class Math, class Policy>
    static TCNN_FUSED_INLINE void second(thread Operand& delta, thread Operand& tangent_delta,
                                         thread Operand& adjoint, thread Operand& tangent_adjoint,
                                         thread Operand& pre, thread Operand& hidden, thread Operand& tangent_pre) {
        for (unsigned i = 0; i < 2; ++i) {
            const T first = Math::template first<Policy>(pre.thread_elements()[i], hidden.thread_elements()[i]);
            const T second = Math::template second<Policy>(pre.thread_elements()[i], hidden.thread_elements()[i]);
            delta.thread_elements()[i] = T(float(adjoint.thread_elements()[i]) * float(first) +
                float(tangent_adjoint.thread_elements()[i]) * float(second) * float(tangent_pre.thread_elements()[i]));
            tangent_delta.thread_elements()[i] = T(float(tangent_adjoint.thread_elements()[i]) * float(first));
        }
    }
};

} // namespace tcnn_fused
#endif
#endif
