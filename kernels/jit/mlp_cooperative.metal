// Native, shape-specialized matrix collectives for callable JIT MLPs.
// These are independent of the AOT kernels: parameter and context storage stay
// canonical, and one native SIMD-group owns 32 application samples.
#ifndef TCNN_JIT_MLP_COOPERATIVE_METAL
#define TCNN_JIT_MLP_COOPERATIVE_METAL

#ifndef TCNN_JIT_HAS_SIMDGROUP_MATRIX
// Automatic RTC dispatch sets this from actual native device capabilities.
// Independent device-code consumers targeting older/other GPUs can set it to 0.
#define TCNN_JIT_HAS_SIMDGROUP_MATRIX 1
#endif
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 230 && TCNN_JIT_HAS_SIMDGROUP_MATRIX
#include <metal_simdgroup_matrix>
#define TCNN_JIT_MLP_INLINE __attribute__((always_inline)) inline
#else
#define TCNN_JIT_MLP_INLINE inline
#endif

namespace tcnn_jit_mlp {

// Lanes is 32 on the GPU. A portable test can provide a whole-matrix atom and
// Lanes=1 to execute this exact tile schedule without pretending to emulate
// Metal instructions or their unspecified fragment-to-lane mapping.
template <class Atom, unsigned Inputs, unsigned Outputs, unsigned Stride,
          bool Backward, unsigned Lanes = 32>
struct Product {
    using T = typename Atom::Scalar;
    enum : unsigned {
        Columns = (Outputs + 7) / 8, KTiles = (Inputs + 7) / 8,
        SampleRows = Outputs >= 32 ? 1 : 4
    };
    static_assert(Stride >= KTiles * 8 && Stride >= Columns * 8, "JIT state plane must cover both matrix dimensions");

    static TCNN_JIT_MLP_INLINE void run(threadgroup T* state, device const T* weights,
                                        threadgroup T* masked_weights, unsigned lane) {
        // Wide layers retire eight samples per pass instead of keeping all
        // 32 samples' output fragments live. The full width-64 tile spilled
        // heavily on M3 Max. Narrow outputs retain the original reuse schedule.
        // Each pass only overwrites samples whose inputs it has consumed.
        // Do not unroll the passes: their accumulator lifetimes must stay apart.
#pragma nounroll
        for (unsigned sample_tile = 0; sample_tile < 4; sample_tile += SampleRows) {
            typename Atom::Accumulator result[SampleRows * Columns];
#pragma unroll
            for (unsigned i = 0; i < SampleRows * Columns; ++i) Atom::zero(result[i]);
#pragma unroll
            for (unsigned kt = 0; kt < KTiles; ++kt) {
                typename Atom::Operand a[SampleRows];
#pragma unroll
                for (unsigned row = 0; row < SampleRows; ++row)
                    // Feature-major shared state makes the per-sample callable
                    // gather/activation/scatter coalesced across native lanes.
                    Atom::load_shared(a[row], state, kt * 8 * 32 + (sample_tile + row) * 8, 32, true);
#pragma unroll
                for (unsigned column = 0; column < Columns; ++column) {
                    typename Atom::Operand b;
                    if (Inputs % 8 == 0 && Outputs % 8 == 0) {
                        // Read the established row-major weights. Forward uses W^T;
                        // backpropagation uses W, without conversion or copies.
                        Atom::load_device(b, weights,
                            Backward ? kt * 8 * Outputs + column * 8
                                     : column * 8 * Inputs + kt * 8,
                            Backward ? Outputs : Inputs, !Backward);
                    } else {
                        // Rectangular/odd layers stage only the boundary-safe 8x8
                        // weight panel. No padded allocation is assumed of callers.
                        for (unsigned e = lane; e < 64; e += Lanes) {
                            const unsigned k = kt * 8 + e / 8, out = column * 8 + e % 8;
                            masked_weights[e] = k < Inputs && out < Outputs
                                ? weights[Backward ? k * Outputs + out : out * Inputs + k] : T(0);
                        }
                        Atom::barrier();
                        Atom::load_shared(b, masked_weights, 0, 8, false);
                        Atom::barrier();
                    }
#pragma unroll
                    for (unsigned row = 0; row < SampleRows; ++row)
                        Atom::mma(result[row * Columns + column], a[row], b);
                }
            }
            // This pass has consumed its samples before their state is reused.
            // Accumulators remain resident until this retirement barrier.
            Atom::barrier();
#pragma unroll
            for (unsigned row = 0; row < SampleRows; ++row)
#pragma unroll
                for (unsigned column = 0; column < Columns; ++column)
                    Atom::store_rounded(result[row * Columns + column], state,
                                        column * 8 * 32 + (sample_tile + row) * 8, 32, true);
            Atom::barrier();
        }
        if (Outputs % 8 != 0) {
            // Zero padding explicitly: Inf * a masked zero weight is NaN, so
            // the matrix product alone cannot establish a zero padding invariant.
            // Unused features must never contaminate the next affine product.
            for (unsigned sample = lane; sample < 32; sample += Lanes)
                for (unsigned column = Outputs; column < Columns * 8; ++column)
                    state[column * 32 + sample] = T(0);
            Atom::barrier();
        }
    }
};

// Compute one eight-column dW panel = Delta^T * X, reducing all 32 samples in
// FP32 matrix accumulators before global accumulation. This replaces 32 atomic
// updates per parameter with one. The caller stages X once per input panel;
// its reuse across every output tile avoids rereading each saved input per row.
template <class Atom, unsigned Rows, unsigned Inputs, unsigned Stride, unsigned Lanes = 32>
struct GradientPanel {
    using T = typename Atom::Scalar;
    enum : unsigned { RowTiles = (Rows + 7) / 8 };

    template <class Environment>
    static TCNN_JIT_MLP_INLINE void run(threadgroup const T* delta, threadgroup const T* input,
                                        threadgroup float* sums, device T* gradients,
                                        unsigned first_input, unsigned lane,
                                        const thread Environment* env) {
        typename Atom::Accumulator result[RowTiles];
#pragma unroll
        for (unsigned row = 0; row < RowTiles; ++row) Atom::zero(result[row]);
#pragma unroll
        for (unsigned sample = 0; sample < 32; sample += 8) {
            typename Atom::Operand b;
            Atom::load_shared(b, input, sample, 32, true);
#pragma unroll
            for (unsigned row = 0; row < RowTiles; ++row) {
                typename Atom::Operand a;
                Atom::load_shared(a, delta, row * 8 * 32 + sample, 32, false);
                Atom::mma(result[row], a, b);
            }
        }
        // Retire all matrix fragments together. Each tile has its own shared
        // slice, so global updates need only one publication/reuse barrier.
#pragma unroll
        for (unsigned row = 0; row < RowTiles; ++row)
            Atom::store_float(result[row], sums, row * 64, 8);
        Atom::barrier();
        for (unsigned e = lane * 2; e < RowTiles * 64; e += Lanes * 2) {
            const unsigned output = e / 8, in = first_input + e % 8;
            if (output < Rows && in < Inputs) {
                if (in + 1 < Inputs)
                    Atom::add_gradient_pair(gradients, output * Inputs + in, sums[e], sums[e + 1], env);
                else
                    Atom::add_gradient(gradients, output * Inputs + in, sums[e], env);
            }
        }
        Atom::barrier();
    }
};

#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 230 && TCNN_JIT_HAS_SIMDGROUP_MATRIX
template <class T> struct MatrixTypes;
template <> struct MatrixTypes<half> { using Operand = simdgroup_half8x8; };
template <> struct MatrixTypes<float> { using Operand = simdgroup_float8x8; };

// Conversion only accesses the fragment's documented per-thread element array;
// no assumption is made about which sample/feature a particular lane owns.
template <class T>
struct NativeAtom {
    using Scalar = T;
    using Operand = typename MatrixTypes<T>::Operand;
    using Accumulator = simdgroup_float8x8;

    static TCNN_JIT_MLP_INLINE void zero(thread Accumulator& result) {
        result = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    static TCNN_JIT_MLP_INLINE void mma(thread Accumulator& result, thread Operand& a, thread Operand& b) {
        simdgroup_multiply_accumulate(result, a, b, result);
    }
    static TCNN_JIT_MLP_INLINE void load_shared(thread Operand& result, threadgroup const T* source,
                                               unsigned offset, unsigned stride, bool transpose) {
        simdgroup_load(result, source + offset, stride, ulong2(0), transpose);
    }
    static TCNN_JIT_MLP_INLINE void load_device(thread Operand& result, device const T* source,
                                               unsigned offset, unsigned stride, bool transpose) {
        simdgroup_load(result, source + offset, stride, ulong2(0), transpose);
    }
    static TCNN_JIT_MLP_INLINE void store_rounded(thread Accumulator& result, threadgroup T* target,
                                                 unsigned offset, unsigned stride, bool transpose) {
        Operand rounded;
        rounded.thread_elements()[0] = T(result.thread_elements()[0]);
        rounded.thread_elements()[1] = T(result.thread_elements()[1]);
        simdgroup_store(rounded, target + offset, stride, ulong2(0), transpose);
    }
    static TCNN_JIT_MLP_INLINE void store_float(thread Accumulator& result, threadgroup float* target,
                                               unsigned offset, unsigned stride) {
        simdgroup_store(result, target + offset, stride, ulong2(0), false);
    }
    static TCNN_JIT_MLP_INLINE void add_gradient(device T* target, unsigned index, float value,
                                                const thread tcnn_jit_environment* env) {
        tcnn_jit_atomic_add(target, index, value, env);
    }
    static TCNN_JIT_MLP_INLINE void add_gradient_pair(device T* target, unsigned index, float first, float second,
                                                     const thread tcnn_jit_environment* env) {
        tcnn_jit_atomic_add_pair(target, index, first, second, env);
    }
    static TCNN_JIT_MLP_INLINE void barrier() {
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
};

// The source-derived dispatch metadata supplies a distinct aligned slice for
// each *native* SIMD-group. Partial sample groups and caller-defined scalar
// environments use the original per-element arithmetic without any barrier.
inline bool available(const thread tcnn_jit_environment* env, unsigned bytes) {
    return env && env->simd_shared_memory && env->simd_shared_bytes >= bytes &&
           env->lane_count == 32u && env->index >= env->lane_index &&
           env->index - env->lane_index <= env->count &&
           env->count - (env->index - env->lane_index) >= 32u;
}
#endif

} // namespace tcnn_jit_mlp
#undef TCNN_JIT_MLP_INLINE
#endif
