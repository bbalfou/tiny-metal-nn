#pragma once
#include <tiny-metal-nn/gpu_memory.h>
#include <tiny-metal-nn/object.h>
#include <cstring>
#include <type_traits>

namespace tcnn {
// Keep the reference workspace calculation and pointer-based reduction signatures.
uint32_t reduce_sum_workspace_size(uint32_t n_elements);

namespace detail {
void reduce_sum_dispatch(metalStream_t stream, const void* input, uint32_t input_bytes,
                         float* workspace, uint32_t n_elements);
}

// Enqueue a GPU reduction, adding to workspace[0] without a host read or wait.
// Intermediate FP32 partials use bounded, capture-safe runtime storage.
// The upstream identity functor takes float: double storage is narrowed per
// element before summation, rather than accumulated using FP64 arithmetic.
template <typename T>
void reduce_sum(T* device_pointer, float* workspace, uint32_t n_elements, metalStream_t stream) {
    if constexpr (std::is_same<std::remove_cv_t<T>, float>::value ||
                  std::is_same<std::remove_cv_t<T>, half>::value ||
                  std::is_same<std::remove_cv_t<T>, double>::value) {
        nn_check(n_elements % (16 / sizeof(T)) == 0,
                 "reduce_sum byte count must be a multiple of 16");
        nn_check(reinterpret_cast<uintptr_t>(device_pointer) % 16 == 0,
                 "reduce_sum input must be aligned to 16 bytes");
        detail::reduce_sum_dispatch(stream, device_pointer, sizeof(T), workspace, n_elements);
    } else
        unsupported_nn(
            "reduction element type (only float, tcnn::half, and double storage are available)");
}

// Return a scalar at an explicit CPU synchronization boundary, as in the reference.
template <typename T>
float reduce_sum(T* device_pointer, uint32_t n_elements, metalStream_t stream) {
    auto& runtime = *detail::resolve_stream(stream);
    if (runtime.is_recording())
        throw metalrt::RecordingDependencyError(
            "A host-scalar reduction cannot synchronize an active recording");
    auto workspace = allocate_workspace(stream, sizeof(float));
    auto transaction = runtime.transaction();
    runtime.fill(workspace.buffer_view(), 0);
    reduce_sum(device_pointer, reinterpret_cast<float*>(workspace.data()), n_elements, stream);
    transaction.finish();
    runtime.synchronize();
    float sum = 0;
    // Shared allocation readback after this stream completes. Avoid a second,
    // device-wide wait through GPUMemory::copy_to_host.
    std::memcpy(&sum, workspace.data(), sizeof(sum));
    return sum;
}

} // namespace tcnn
