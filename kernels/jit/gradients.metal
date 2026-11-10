// One final conversion after automatic JIT FP32 accumulation. Dense MLP
// parameters use bounded independent stripes; sparse encoders use the base.
// Typed scratch contains Accumulate's prior value and direct extension stores.
#include <metal_stdlib>
using namespace metal;

struct JitGradientArgs {
    uint parameters, dense_parameters, partitions, reserved;
};

template <typename T> struct JitGradientPacket;
template <> struct JitGradientPacket<half> { using Type = packed_half4; };
template <> struct JitGradientPacket<float> { using Type = packed_float4; };

template <typename T>
inline void tcnn_jit_finish_gradients(
    const device T* typed, const device float* wide, device T* result,
    constant JitGradientArgs& args, uint tid) {
    const ulong first = ulong(tid) * 4ul;
    if (first >= args.parameters) return;
    // Packed packets require only scalar alignment. This covers odd half
    // views and unaligned stripe starts without reading neighboring canaries.
    if (first + 3ul < args.parameters &&
        (first >= args.dense_parameters || first + 3ul < args.dense_parameters)) {
        float4 sum = float4(0.0f);
        if (first < args.dense_parameters) {
            for (uint partition = 0; partition < args.partitions; ++partition) {
                const ulong offset = ulong(args.parameters) + ulong(partition) * args.dense_parameters + first;
                const packed_float4 value = *reinterpret_cast<const device packed_float4*>(wide + offset);
                sum += float4(value.x, value.y, value.z, value.w);
            }
        } else {
            const packed_float4 value = *reinterpret_cast<const device packed_float4*>(wide + first);
            sum = float4(value.x, value.y, value.z, value.w);
        }
        using Packet = typename JitGradientPacket<T>::Type;
        const Packet previous = *reinterpret_cast<const device Packet*>(typed + first);
        Packet output;
        for (uint k = 0; k < 4u; ++k) output[k] = T(float(previous[k]) + sum[k]);
        *reinterpret_cast<device Packet*>(result + first) = output;
    } else {
        // Only the dense/sparse boundary and the exact final tail need scalar
        // loads. No special parameter-count or pointer-alignment restriction.
        for (uint k = 0; k < 4u && first + k < args.parameters; ++k) {
            const ulong i = first + k;
            float sum = 0.0f;
            if (i < args.dense_parameters) {
                for (uint partition = 0; partition < args.partitions; ++partition)
                    sum += wide[ulong(args.parameters) + ulong(partition) * args.dense_parameters + i];
            } else {
                sum = wide[i];
            }
            result[i] = T(float(typed[i]) + sum);
        }
    }
}

#define TCNN_JIT_GRADIENT_ENTRY(T, NAME) \
kernel void NAME(const device T* typed [[buffer(0)]], const device float* wide [[buffer(1)]], \
    device T* result [[buffer(2)]], constant JitGradientArgs& args [[buffer(3)]], \
    uint tid [[thread_position_in_grid]]) { \
    tcnn_jit_finish_gradients(typed, wide, result, args, tid); \
}
TCNN_JIT_GRADIENT_ENTRY(half, tcnn_jit_finish_gradients_f16)
TCNN_JIT_GRADIENT_ENTRY(float, tcnn_jit_finish_gradients_f32)
#undef TCNN_JIT_GRADIENT_ENTRY
