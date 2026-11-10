#pragma once
#include <tiny-metal-nn/rtc_kernel.h>
#include <tiny-metal-nn/gpu_memory.h>
#include <string>
#include <vector>

namespace tcnn {
namespace detail {
// All generated device definitions live in .metal assets or .cpp generators.
// This common prelude is guarded and can be concatenated by arbitrary models.
std::string jit_device_prelude();
// Composable generators advertise their cooperative workspace in a source
// annotation. Sequential children reuse the largest region; each native SIMD
// group receives a distinct slice even when children have different shapes.
uint32_t jit_simdgroup_scratch_bytes(const std::string& source);
uint32_t jit_threadgroup_scratch_bytes(const std::string& source, uint32_t threads);
void jit_validate_identifier(const std::string& name);
std::string jit_forward_function(const std::string& name, const std::string& in,
                                 const std::string& out, const std::string& params,
                                 const std::string& body, uint32_t context_bytes = 0,
                                 bool per_element = true);
std::string jit_backward_function(const std::string& name, const std::string& in,
                                  const std::string& out, const std::string& params,
                                  const std::string& body, uint32_t context_bytes = 0,
                                  bool per_element = true);
std::string jit_second_function(const std::string& name, const std::string& in,
                                const std::string& out, const std::string& params,
                                const std::string& body, uint32_t context_bytes = 0,
                                bool per_element = true);
std::string jit_forward_kernel_source(const std::string& name, const std::string& function,
                                      const std::string& input_type,
                                      const std::string& parameter_type,
                                      const std::string& output_type, uint32_t input_width,
                                      uint32_t context_bytes);
std::string jit_backward_kernel_source(const std::string& name, const std::string& function,
                                       const std::string& input_type,
                                       const std::string& parameter_type,
                                       const std::string& output_type, uint32_t input_width,
                                       uint32_t output_width, uint32_t context_bytes,
                                       bool second = false, size_t gradient_parameters = 0);
std::unique_ptr<MetalRtcKernel> jit_compile_kernel(const std::string& name,
                                                   const std::string& source, uint32_t kind);
std::string jit_training_kernel_source(const std::string& name, const std::string& forward,
                                       const std::string& backward, const std::string& loss,
                                       const std::string& input_type,
                                       const std::string& parameter_type,
                                       const std::string& output_type, uint32_t input_width,
                                       uint32_t output_width, uint32_t context_bytes,
                                       bool context_in_threadgroup, uint32_t backward_scratch_bytes,
                                       bool external_gradient, size_t gradient_parameters = 0);
// Snapshots per-sample controls using queued native copies. Each slot contains
// exactly batch floats; generated compositions shift an environment slot offset.
GPUMemoryArena::Allocation jit_collect_runtime_inputs(metalStream_t stream,
                                                      const std::vector<const float*>& inputs,
                                                      uint32_t batch);
// Packed-half atomic updates always operate on a word-aligned, word-padded
// allocation, including odd parameter counts and odd composite child offsets.
GPUMemoryArena::Allocation jit_prepare_gradients(metalStream_t stream, const void* gradients,
                                                 size_t bytes, bool accumulate);
void jit_finish_gradients(metalStream_t stream, void* gradients,
                          const GPUMemoryArena::Allocation& scratch, size_t bytes);
// Automatic differentiation of explicitly additive implementations reduces
// into float. Typed scratch retains prior and independent direct contributions;
// finalization combines them with the float contributions in one conversion.
struct JitGradientPlan {
    uint32_t parameters = 0, dense_parameters = 0, partitions = 0;
    size_t elements() const;
};
JitGradientPlan jit_gradient_plan(const std::string& source, size_t parameters);
GPUMemoryArena::Allocation jit_prepare_float_gradients(metalStream_t stream,
                                                       const JitGradientPlan& plan);
void jit_finish_float_gradients(metalStream_t stream, void* gradients,
                                const GPUMemoryArena::Allocation& typed,
                                const GPUMemoryArena::Allocation& wide, const JitGradientPlan& plan,
                                size_t element_size);
metalrt::Argument jit_float_gradient_argument(uint32_t slot,
                                              const GPUMemoryArena::Allocation& wide);

struct JitKernelCache {
    std::string source;
    metalrt::Device device;
    std::unique_ptr<MetalRtcKernel> kernel;
    MetalRtcKernel* get(const std::string& name, const std::string& code, uint32_t kind) {
        const auto selected = metalrt::Device::current();
        if (!kernel || device != selected || source != code) {
            auto replacement = jit_compile_kernel(name, code, kind);
            kernel = std::move(replacement);
            source = code;
            device = selected;
        }
        return kernel.get();
    }
};
} // namespace detail
} // namespace tcnn
