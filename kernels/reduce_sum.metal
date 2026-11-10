// Independently implemented hierarchical FP32 accumulation. Public FP16/FP32/FP64
// inputs are 16-byte aligned; intermediate partials may have any element count.
#include <metal_stdlib>
using namespace metal;
#include "reduction/convert.metal"

struct ReductionArgs {
    uint count, groups, accumulate, reserved;
};

static_assert(sizeof(ReductionArgs) == 16, "reduction argument ABI");

// Every thread participates in the barrier, including lanes outside the input.
// SIMD width comes from Metal, rather than assuming an Apple-specific width.
inline void finish_sum(
    float value,
    device float* output,
    constant ReductionArgs& args,
    threadgroup float* partials,
    uint group,
    uint lane,
    uint simd_index,
    uint simd_width,
    uint simd_count
) {
    value = simd_sum(value);
    if (lane == 0)
        partials[simd_index] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_index == 0) {
        value = 0.0f;
        for (uint i = lane; i < simd_count; i += simd_width)
            value += partials[i];
        value = simd_sum(value);
        if (lane == 0) {
            if (args.accumulate)
                output[group] += value;
            else
                output[group] = value;
        }
    }
}

kernel void nn_reduce_sum_f32(
    device const float4* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ReductionArgs& args [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 group_size [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint simd_count [[simdgroups_per_threadgroup]]
) {
    threadgroup float partials[256];
    float4 sum = float4(0.0f);
    const uint stride = args.groups * group_size.x;
    for (uint i = group.x * group_size.x + tid; i < args.count / 4; i += stride)
        sum += input[i];
    finish_sum(
        (sum.x + sum.y) + (sum.z + sum.w),
        output,
        args,
        partials,
        group.x,
        lane,
        simd_index,
        simd_width,
        simd_count
    );
}

kernel void nn_reduce_sum_f16(
    device const half4* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ReductionArgs& args [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 group_size [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint simd_count [[simdgroups_per_threadgroup]]
) {
    threadgroup float partials[256];
    float4 low = float4(0.0f), high = float4(0.0f);
    const uint stride = args.groups * group_size.x;
    for (uint i = group.x * group_size.x + tid; i < args.count / 8; i += stride) {
        low += float4(input[2 * i]);
        high += float4(input[2 * i + 1]);
    }
    const float4 sum = low + high;
    finish_sum(
        (sum.x + sum.y) + (sum.z + sum.w),
        output,
        args,
        partials,
        group.x,
        lane,
        simd_index,
        simd_width,
        simd_count
    );
}

// FP64 is a storage format here, not GPU arithmetic. The supplied upstream
// identity functor narrows each input to float before accumulating in float.
kernel void nn_reduce_sum_f64(
    device const ulong2* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ReductionArgs& args [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 group_size [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint simd_count [[simdgroups_per_threadgroup]]
) {
    threadgroup float partials[256];
    float2 sum = float2(0.0f);
    const uint stride = args.groups * group_size.x;
    for (uint i = group.x * group_size.x + tid; i < args.count / 2; i += stride) {
        const ulong2 bits = input[i];
        sum += float2(as_type<float>(reduction_f64_to_f32_bits(bits.x)), as_type<float>(reduction_f64_to_f32_bits(bits.y)));
    }
    finish_sum(
        sum.x + sum.y,
        output,
        args,
        partials,
        group.x,
        lane,
        simd_index,
        simd_width,
        simd_count
    );
}

// Scalar loads avoid padding, overreads, and alignment requirements for a
// non-multiple-of-four number of partials. The partial buffer remains FP32.
kernel void nn_reduce_sum_partials(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ReductionArgs& args [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 group_size [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint simd_count [[simdgroups_per_threadgroup]]
) {
    threadgroup float partials[256];
    float sum = 0.0f;
    const uint stride = args.groups * group_size.x;
    for (uint i = group.x * group_size.x + tid; i < args.count; i += stride)
        sum += input[i];
    finish_sum(
        sum,
        output,
        args,
        partials,
        group.x,
        lane,
        simd_index,
        simd_width,
        simd_count
    );
}
