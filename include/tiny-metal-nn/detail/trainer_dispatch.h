#pragma once
#include <tiny-metal-nn/gpu_matrix.h>

namespace tcnn {
namespace detail {
// Add the reference Trainer's FP32 logistic perturbation to a typed output tensor.
template <class T>
void add_perturbation(metalStream_t stream, const GPUMatrix<T>& input,
                      const GPUMatrix<float>& perturbation, GPUMatrix<T>& output);
} // namespace detail
} // namespace tcnn
