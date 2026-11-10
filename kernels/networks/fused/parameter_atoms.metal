#ifndef TCNN_FUSED_PARAMETER_ATOMS_METAL
#define TCNN_FUSED_PARAMETER_ATOMS_METAL
#include "training_atoms.metal"

#ifdef __METAL_VERSION__
namespace tcnn_fused {
template <class T>
struct ParameterMmaAtom {
    using Base = TrainingMmaAtom<T>;
    using Scalar = T;
    using Operand = typename Base::Operand;
    using Accumulator = typename Base::Accumulator;
    static TCNN_FUSED_INLINE void zero(thread Accumulator& a) { Base::zero(a); }
    static TCNN_FUSED_INLINE void mma(thread Accumulator& c, thread Operand& a, thread Operand& b) {
        Base::mma(c, a, b);
    }
    static TCNN_FUSED_INLINE void load_shared(thread Operand& a, threadgroup const T* source,
                                              unsigned offset, unsigned stride, bool transpose = false) {
        Base::load_shared(a, source, offset, stride, transpose);
    }
    static TCNN_FUSED_INLINE void store_accumulator(thread const Accumulator& a, threadgroup float* destination,
                                                    unsigned offset, unsigned stride) {
        simdgroup_store(a, destination + offset, stride, ulong2(0), false);
    }
};
} // namespace tcnn_fused
#endif
#endif
