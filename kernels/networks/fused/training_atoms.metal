#ifndef TCNN_FUSED_TRAINING_ATOMS_METAL
#define TCNN_FUSED_TRAINING_ATOMS_METAL
#include "atoms.metal"

#ifdef __METAL_VERSION__
#include "../common.metal"

namespace tcnn_fused {

template <unsigned Bytes> struct TrainingTransferVector;
template <> struct TrainingTransferVector<8> { using Type = uint2; };
template <> struct TrainingTransferVector<16> { using Type = uint4; };

// Byte-preserving vector transfers. The collective checks complete sample
// tiles and matrix strides; launch selection checks the buffer bases.
template <class T>
struct TrainingCopyAtom {
    enum : unsigned { VectorElements = 16 / sizeof(T), IncomingElements = 4 };
    static TCNN_FUSED_INLINE void to_shared(threadgroup T* destination, unsigned destination_index,
                                            device const T* source, ulong source_index) {
        *reinterpret_cast<threadgroup uint4*>(destination + destination_index) =
            *reinterpret_cast<device const uint4*>(source + source_index);
    }
    static TCNN_FUSED_INLINE void to_device(device T* destination, ulong destination_index,
                                            threadgroup const T* source, unsigned source_index) {
        *reinterpret_cast<device uint4*>(destination + destination_index) =
            *reinterpret_cast<threadgroup const uint4*>(source + source_index);
    }
    static TCNN_FUSED_INLINE void incoming_panel(threadgroup T* destination, unsigned destination_index,
                                                 device const T* source, ulong source_index,
                                                 device T* saved, ulong saved_index) {
        // Four half elements reproduce the supplied backward input-panel
        // transfer; four floats keep the corresponding FP32 transfer at 16 B.
        using V = typename TrainingTransferVector<4 * sizeof(T)>::Type;
        const V value = *reinterpret_cast<device const V*>(source + source_index);
        *reinterpret_cast<threadgroup V*>(destination + destination_index) = value;
        *reinterpret_cast<device V*>(saved + saved_index) = value;
    }
    static TCNN_FUSED_INLINE void barrier() { threadgroup_barrier(mem_flags::mem_threadgroup); }
};

template <class T> struct TrainingPackedVector;
template <> struct TrainingPackedVector<half> { using Type = packed_half4; };
template <> struct TrainingPackedVector<float> { using Type = packed_float4; };

// Metal's packed_half4/packed_float4 require only scalar alignment (MSL
// specification Table 2.4). This policy keeps vector transfers available for
// arbitrary GPUMatrix subviews without assuming a 16-byte base address.
// Each call moves exactly 16 bytes; masks/strides are checked by the collective.
template <class T>
struct TrainingPackedCopyAtom {
    using Base = TrainingCopyAtom<T>;
    enum : unsigned { VectorElements = 16 / sizeof(T), IncomingElements = Base::IncomingElements };
    static TCNN_FUSED_INLINE void barrier() { Base::barrier(); }
    static TCNN_FUSED_INLINE void incoming_panel(threadgroup T* destination, unsigned destination_index,
                                                 device const T* source, ulong source_index,
                                                 device T* saved, ulong saved_index) {
        // This entry is only used when the caller has established the original
        // incoming-panel alignment contract; rectangular gathers use packets.
        Base::incoming_panel(destination, destination_index, source, source_index, saved, saved_index);
    }
    static TCNN_FUSED_INLINE void to_shared(threadgroup T* destination, unsigned destination_index,
                                            device const T* source, unsigned long source_index) {
        using Packet = typename TrainingPackedVector<T>::Type;
        for (unsigned i = 0; i < VectorElements; i += 4)
            *reinterpret_cast<threadgroup Packet*>(destination + destination_index + i) =
                *reinterpret_cast<device const Packet*>(source + source_index + i);
    }
    static TCNN_FUSED_INLINE void to_device(device T* destination, unsigned long destination_index,
                                            threadgroup const T* source, unsigned source_index) {
        using Packet = typename TrainingPackedVector<T>::Type;
        for (unsigned i = 0; i < VectorElements; i += 4)
            *reinterpret_cast<device Packet*>(destination + destination_index + i) =
                *reinterpret_cast<threadgroup const Packet*>(source + source_index + i);
    }
};

// The kernel depends on this fragment interface, rather than on the layout of
// thread_elements(). Tests may supply an independent matrix atom with the same
// interface. Native execution delegates multiplication/conversion to the same
// atoms as fused inference.
template <class T>
struct TrainingMmaAtom {
    using Scalar = T;
    using Base = SimdgroupMmaAtom<T>;
    using Operand = typename Base::Operand;
    using Accumulator = typename Base::Accumulator;
    static TCNN_FUSED_INLINE void zero(thread Accumulator& c) { Base::zero(c); }
    static TCNN_FUSED_INLINE void mma(thread Accumulator& c, thread Operand& a, thread Operand& b) {
        Base::mma(c, a, b);
    }
    static TCNN_FUSED_INLINE void load_shared(thread Operand& a, threadgroup const T* source,
                                              unsigned offset, unsigned stride, bool transpose = false) {
        simdgroup_load(a, source + offset, stride, ulong2(0), transpose);
    }
    static TCNN_FUSED_INLINE void load_device(thread Operand& a, device const T* source,
                                              ulong offset, unsigned stride) {
        simdgroup_load(a, source + offset, stride, ulong2(0), false);
    }
    static TCNN_FUSED_INLINE void store_shared(thread const Operand& a, threadgroup T* destination,
                                               unsigned offset, unsigned stride) {
        simdgroup_store(a, destination + offset, stride, ulong2(0), false);
    }
    static TCNN_FUSED_INLINE void store_device(thread const Operand& a, device T* destination,
                                               ulong offset, unsigned stride) {
        simdgroup_store(a, destination + offset, stride, ulong2(0), false);
    }
    template <class Policy>
    static TCNN_FUSED_INLINE void convert(thread Operand& a, thread Accumulator& c) {
        Base::template convert<Policy>(a, c);
    }
    template <class Policy>
    static TCNN_FUSED_INLINE T gradient(T value, float incoming) {
        // The upstream contract rounds the matrix product and derivative to T
        // before multiplying. Sine and SiLU consume saved preactivations.
        return T(activation_gradient(float(value), float(T(incoming)), Policy::Kind, sizeof(T),
                                     Policy::Kind != 2 && Policy::Kind != 4));
    }
    template <class Policy>
    static TCNN_FUSED_INLINE void differentiate(thread Operand& result, thread Accumulator& c,
                                                 thread Operand& saved) {
        result.thread_elements()[0] = gradient<Policy>(saved.thread_elements()[0], c.thread_elements()[0]);
        result.thread_elements()[1] = gradient<Policy>(saved.thread_elements()[1], c.thread_elements()[1]);
    }
};

} // namespace tcnn_fused
#endif // __METAL_VERSION__
#endif
