// Stable host boundary for the whole-model kernels. All storage stays on the
// selected backend; optional derivative outputs do not change parameter gradients.
#pragma once
#include <tiny-metal-nn/object.h>
#include <tiny-metal-nn/detail/kernels/networks/fully_fused_mlp.h>

namespace tcnn {
namespace detail {
size_t fused_storage_count(network_kernels::FusedShape shape, uint32_t batch_size);

enum class FusedTrainingPass : uint32_t { Forward, Backward, Tangent, SecondBackward };

struct FusedTrainingConfig {
    uint32_t samples_per_threadgroup, threads, k_panel, row_groups, column_groups,
        threadgroup_bytes;
};

// Pure inspection helpers, also used by native benchmarks/tests. Unsupported
// widths or storage precisions return a zero configuration. Matrix strides and
// rectangular boundary layers are supported by the general tiled entrypoints.
// Rectangular forward reports general-path geometry and a scratch upper bound;
// a shape-specialized saved-forward pipeline can use less shared memory.
FusedTrainingConfig fused_training_config(uint32_t width, uint32_t precision,
                                          FusedTrainingPass pass);
FusedTrainingConfig fused_rectangular_training_config(uint32_t width, uint32_t precision,
                                                      FusedTrainingPass pass);
FusedTrainingConfig fused_higher_order_config(uint32_t width, uint32_t precision);

struct FusedParameterConfig {
    uint32_t tile_rows, tile_columns, k_panel, threads, splits, threadgroup_bytes;
};
FusedParameterConfig fused_parameter_config(network_kernels::FusedShape shape, uint32_t precision,
                                            uint32_t batch);

bool fused_training_eligible(const metalrt::Capabilities& capabilities,
                             const network_kernels::Mat& input, network_kernels::FusedShape shape,
                             size_t n_params);

// Pure eligibility query for the canonical inference backend. Boundary tiles
// handle strides, scalar alignment, projections, nonlinear output and tails.
bool fused_inference_eligible(const metalrt::Capabilities& capabilities,
                              const network_kernels::Mat& input, const network_kernels::Mat& output,
                              network_kernels::FusedShape shape, size_t n_params,
                              uintptr_t input_address, uintptr_t weights_address,
                              uintptr_t output_address);

// Returns false without submitting work when the supplied inference schedule
// does not apply. Successful dispatch retains the runtime's buffer ownership,
// stream ordering, and graph-capture behavior; no parameter copy is cached.
template <class T>
bool try_fused_inference(metalStream_t stream, const GPUMatrixDynamic<T>& input, const T* weights,
                         size_t n_params, GPUMatrixDynamic<T>& output,
                         network_kernels::FusedShape shape);

struct FusedInferenceConfig {
    uint32_t samples_per_threadgroup, threads, k_panel, row_groups, column_groups,
        output_row_groups, threadgroup_bytes, preloaded_layers;
    bool private_state, direct_device_weights, boundary_tiles;
};
FusedInferenceConfig fused_inference_config(network_kernels::FusedShape shape, uint32_t precision,
                                            bool boundary_tiles = false);

template <class T>
void fused_forward(metalStream_t stream, const GPUMatrixDynamic<T>& input, const T* weights,
                   size_t n_params, GPUMatrixDynamic<T>& pre, GPUMatrixDynamic<T>& hidden,
                   network_kernels::FusedShape shape);

template <class T>
void fused_backward(metalStream_t stream, const GPUMatrixDynamic<T>& input,
                    const GPUMatrixDynamic<T>& output, const GPUMatrixDynamic<T>& incoming,
                    GPUMatrixDynamic<T>* dinput, const T* weights, T* gradients, size_t n_params,
                    const GPUMatrixDynamic<T>& pre, const GPUMatrixDynamic<T>& hidden,
                    network_kernels::FusedShape shape, GradientMode mode);

template <class T>
void fused_second_backward(metalStream_t stream, const GPUMatrixDynamic<T>& input,
                           const GPUMatrixDynamic<T>& direction,
                           const GPUMatrixDynamic<T>& incoming, GPUMatrixDynamic<T>* doutput,
                           GPUMatrixDynamic<T>* dinput, const T* weights, T* gradients,
                           size_t n_params, const GPUMatrixDynamic<T>& pre,
                           const GPUMatrixDynamic<T>& hidden, network_kernels::FusedShape shape,
                           GradientMode mode);
} // namespace detail
} // namespace tcnn
