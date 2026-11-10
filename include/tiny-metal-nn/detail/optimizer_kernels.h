#pragma once
#include <tiny-metal-nn/gpu_memory.h>
#include <tiny-metal-nn/gpu_memory_json.h>
#include <tiny-metal-nn/metal.h>

namespace tcnn {
namespace detail {
// Buffers must describe one validated JIT gradient plan. Null typed input is
// Overwrite; typed may alias the destination for Accumulate. Commands retain
// their resolved resources after this transient host descriptor expires.
struct AdamJitGradientInput {
    const void* typed;
    const float* wide;
    uint32_t parameters, dense_parameters, partitions;
};
struct AdamState {
    float first, second;
    uint32_t step;
};
static_assert(sizeof(AdamState) == 12 && alignof(AdamState) == 4, "Adam state Metal ABI");

// Helpers dispatch native Metal kernels with explicit Reference callbacks.
template <typename T>
void optimizer_sgd_step(metalStream_t, uint32_t, float, float, float, float*, T*, const T*);
template <typename T>
void optimizer_adam_step(metalStream_t, uint32_t, uint32_t, float, float, float, float, float,
                         float, float, bool, bool, bool, float, float, float, float, float, float,
                         float, float*, T*, const T*, float*, float*, uint32_t*,
                         const AdamJitGradientInput* = nullptr, AdamState* = nullptr);
template <typename T>
void optimizer_ema_step_full_precision(metalStream_t, uint32_t, float, float, float, const T*, T*,
                                       float*);
template <typename T>
void optimizer_ema_step_half_precision(metalStream_t, uint32_t, float, float, float, const T*, T*);
template <typename T>
void optimizer_average_step(metalStream_t, uint32_t, uint32_t, const T*, T*, T*);
template <typename T> void optimizer_lookahead_step(metalStream_t, uint32_t, float, float*, T*, T*);
template <typename T>
void optimizer_gradient_update(metalStream_t, uint32_t, bool, uint32_t, const T*, float*);
template <typename T> void optimizer_cast(metalStream_t, uint32_t, const float*, T*);
template <typename T> void optimizer_cast_from(metalStream_t, uint32_t, const T*, float*);
template <typename T>
void optimizer_novo_moment(metalStream_t, uint32_t, float, float, const T*, float*);
template <typename T>
void optimizer_novo_step(metalStream_t, uint32_t, float, float, float, float, float, float, float*,
                         T*, const T*, float*, const float*);
} // namespace detail
} // namespace tcnn
