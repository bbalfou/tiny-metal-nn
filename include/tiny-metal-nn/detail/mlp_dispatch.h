// Internal baseline Metal kernels. Public interfaces remain in network.h.
#pragma once
#include <tiny-metal-nn/object.h>
#include <tiny-metal-nn/detail/kernels/networks/abi.h>

namespace tcnn {
namespace detail {
// Compute W*x, retaining preactivation values for noninvertible activations.
template <class T>
void dense_forward(metalStream_t stream, const GPUMatrixDynamic<T>& input,
                   const GPUMatrix<T, RM>& weights, GPUMatrixDynamic<T>& preactivation,
                   GPUMatrixDynamic<T>& output, Activation activation);
// Form dL/dz using the saved input for nonfused activations or the rounded
// output for fused activations, preserving the uploaded precision boundaries.
template <class T>
void dense_delta(metalStream_t stream, const GPUMatrixDynamic<T>& values,
                 const GPUMatrixDynamic<T>& dL_doutput, GPUMatrixDynamic<T>& delta,
                 Activation activation, bool values_are_output = false);
// Propagate a local derivative through the transpose of a weight matrix.
template <class T>
void dense_input_gradient(metalStream_t stream, const GPUMatrixDynamic<T>& delta,
                          const GPUMatrix<T, RM>& weights, GPUMatrixDynamic<T>& dL_dinput);
// Reduce a batch's outer products deterministically into the parameter gradient.
template <class T>
void dense_weight_gradient(metalStream_t stream, const GPUMatrixDynamic<T>& input,
                           const GPUMatrixDynamic<T>& delta, GPUMatrix<T, RM>& gradients,
                           GradientMode param_gradients_mode);
// Differentiate the activation's ordinary and tangent outputs in one GPU pass.
template <class T>
void dense_second_delta(metalStream_t stream, const GPUMatrixDynamic<T>& preactivation,
                        const GPUMatrixDynamic<T>& output,
                        const GPUMatrixDynamic<T>& tangent_preactivation,
                        const GPUMatrixDynamic<T>& adjoint,
                        const GPUMatrixDynamic<T>& tangent_adjoint, GPUMatrixDynamic<T>& delta,
                        GPUMatrixDynamic<T>& tangent_delta, Activation activation);
// Add a strided matrix to another on the GPU, retaining both allocations.
template <class T>
void accumulate_matrix(metalStream_t stream, const GPUMatrixDynamic<T>& source,
                       GPUMatrixDynamic<T>& destination);
} // namespace detail
} // namespace tcnn
