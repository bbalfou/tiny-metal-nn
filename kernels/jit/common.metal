// Callable JIT ABI for native Metal. Kernel wrappers pass a native SIMD-group
// context; per-element function prologues select one sample. Cooperative
// functions and compositions own disjoint, explicitly aligned group ranges.
#ifndef TCNN_JIT_COMMON
#define TCNN_JIT_COMMON
#include <metal_stdlib>
using namespace metal;
#ifndef TCNN_JIT_FLOAT_ATOMICS
#define TCNN_JIT_FLOAT_ATOMICS 0
#endif
#ifndef TCNN_JIT_CTX
#define TCNN_JIT_CTX device
#endif

template<typename T> struct tcnn_jit_empty_array {
    T unused;
    constexpr uint size() const thread { return 0; }
    thread T& operator[](uint) thread { return unused; }
    const thread T& operator[](uint) const thread { return unused; }
};
template<typename T, uint N> struct tcnn_jit_array_type { using type = metal::array<T,N>; };
template<typename T> struct tcnn_jit_array_type<T,0> { using type = tcnn_jit_empty_array<T>; };
template<typename T, uint N> using tcnn_jit_array = typename tcnn_jit_array_type<T,N>::type;

struct tcnn_jit_matrix_strides { uint row, column; };
struct tcnn_jit_environment {
    const device float* runtime_inputs;
    uint index, count, offset;
    device uchar* gradient_base;
    uint has_level_override;
    float level_override;
    threadgroup uchar* shared_memory;
    uint thread_index, group_size;
    uint lane_index, lane_count, context_is_per_element;
    // The wrapper slices cooperative workspace using native SIMD-group IDs.
    // Keep shared_memory as the full threadgroup base for existing extensions.
    // Old manual consumers may initialize only the original fields. Optional
    // optimization state must still default to the original scalar/typed path.
    threadgroup uchar* simd_shared_memory = nullptr;
    uint simd_shared_bytes = 0;
    device float* float_gradients = nullptr;
    uint float_gradient_count = 0, dense_gradient_count = 0, gradient_partitions = 0;
};

inline ulong tcnn_jit_float_gradient_index(ulong absolute, const thread tcnn_jit_environment* env) {
    // Dense weights are touched by every sample group. Bounded disjoint bins
    // reduce contention while sparse encoding entries retain direct indexing.
    if (absolute < env->dense_gradient_count && env->gradient_partitions) {
        const uint partition = (env->index / 32u) % env->gradient_partitions;
        return ulong(env->float_gradient_count) + ulong(partition) * env->dense_gradient_count + absolute;
    }
    return absolute;
}

// Native FP32 atomic addition requires MSL 4.1 and Apple9 or newer.
// Other targets retain integer CAS; packed-half updates also use CAS.
inline void tcnn_jit_atomic_add(device float* base, uint index, float value,
    const thread tcnn_jit_environment* env = nullptr) {
    if (value == 0.0f) return;
    if (env && env->float_gradients) {
        device float* origin = env->gradient_base ? reinterpret_cast<device float*>(env->gradient_base) : base;
        const ulong absolute = ulong(base - origin) + index;
        base = env->float_gradients + tcnn_jit_float_gradient_index(absolute, env);
        index = 0u;
    }
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 410 && TCNN_JIT_FLOAT_ATOMICS
    atomic_fetch_add_explicit(reinterpret_cast<device atomic_float*>(base + index), value, memory_order_relaxed);
#else
    device atomic_uint* word = reinterpret_cast<device atomic_uint*>(base + index);
    uint old = atomic_load_explicit(word, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(word, &old,
        as_type<uint>(as_type<float>(old) + value), memory_order_relaxed, memory_order_relaxed)) {}
#endif
}
inline void tcnn_jit_atomic_add(device half* base, uint index, float value,
    const thread tcnn_jit_environment* env = nullptr) {
    if (value == 0.0f) return;
    device half* origin = env && env->gradient_base
        ? reinterpret_cast<device half*>(env->gradient_base) : base;
    const ulong absolute = ulong(base - origin) + index;
    if (env && env->float_gradients) {
        tcnn_jit_atomic_add(env->float_gradients + tcnn_jit_float_gradient_index(absolute, env), 0u, value);
        return;
    }
    device atomic_uint* word = reinterpret_cast<device atomic_uint*>(origin) + absolute / 2;
    const uint shift = uint(absolute & 1) * 16;
    const uint mask = 0xffffu << shift;
    uint old = atomic_load_explicit(word, memory_order_relaxed);
    do {
        const half previous = as_type<half>(ushort((old >> shift) & 0xffffu));
        const uint next = (old & ~mask) | (uint(as_type<ushort>(half(float(previous) + value))) << shift);
        if (atomic_compare_exchange_weak_explicit(word, &old, next,
            memory_order_relaxed, memory_order_relaxed)) break;
    } while (true);
}

inline void tcnn_jit_atomic_add_pair(device float* base, uint index, float first, float second,
    const thread tcnn_jit_environment* env = nullptr) {
    tcnn_jit_atomic_add(base, index, first, env);
    tcnn_jit_atomic_add(base, index + 1u, second, env);
}

// Adjacent grid features usually share a half word. Publish both through one
// CAS instead of making their updates contend on the same word twice. An odd
// child offset deliberately retains the scalar helpers and their canaries.
inline void tcnn_jit_atomic_add_pair(device half* base, uint index, float first, float second,
    const thread tcnn_jit_environment* env = nullptr) {
    if (first == 0.0f && second == 0.0f) return;
    device half* origin = env && env->gradient_base
        ? reinterpret_cast<device half*>(env->gradient_base) : base;
    const ulong absolute = ulong(base - origin) + index;
    if (env && env->float_gradients) {
        // A pair can straddle the dense-prefix boundary in an extension.
        tcnn_jit_atomic_add(env->float_gradients + tcnn_jit_float_gradient_index(absolute, env), 0u, first);
        tcnn_jit_atomic_add(env->float_gradients + tcnn_jit_float_gradient_index(absolute + 1ul, env), 0u, second);
        return;
    }
    if (absolute & 1ul) {
        tcnn_jit_atomic_add(base, index, first, env);
        tcnn_jit_atomic_add(base, index + 1u, second, env);
        return;
    }
    device atomic_uint* word = reinterpret_cast<device atomic_uint*>(origin) + absolute / 2;
    uint old = atomic_load_explicit(word, memory_order_relaxed);
    do {
        const half a = as_type<half>(ushort(old & 0xffffu));
        const half b = as_type<half>(ushort(old >> 16));
        const uint low = first == 0.0f ? old & 0xffffu : uint(as_type<ushort>(half(float(a) + first)));
        const uint high = second == 0.0f ? old & 0xffff0000u : uint(as_type<ushort>(half(float(b) + second))) << 16;
        const uint next = low | high;
        if (atomic_compare_exchange_weak_explicit(word, &old, next,
            memory_order_relaxed, memory_order_relaxed)) break;
    } while (true);
}
#endif
