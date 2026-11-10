// MLX-only adaptation. Core NN and metalrt interfaces contain no MLX types.
#pragma once

#include <mlx/array.h>
#include <mlx/stream.h>
#include <metalrt/runtime.h>
#include <tiny-metal-nn/metal.h>

#include <functional>
#include <memory>
#include <vector>

namespace tinymetalnn {
namespace mlx_bindings {

struct ArrayBuffer {
    metalrt::BufferView original;
    metalrt::Buffer staging;

    metalrt::BufferView view() const {
        return staging ? staging.view(0, original.length) : original;
    }
    void* data() const { return view().data(); }
    template <class T> T* data_as() const { return static_cast<T*>(data()); }
};

struct MLXSession {
    metalrt::Device device;
    std::shared_ptr<metalrt::Stream> queue;
    std::unique_ptr<tcnn::metal::Stream> nn_stream;
    std::vector<ArrayBuffer> inputs;
    std::vector<ArrayBuffer> outputs;
};

// Called only from Primitive::eval_gpu with evaluated inputs. Allocates MLX
// outputs, retains allocator Data leases, and batches native work on MLX's
// unchanged current command buffer. Explicit GPU handoffs order MLX's usual
// untracked resources in both directions. Never evaluates, commits, or waits
// for the MLX stream; its scheduler retains that ownership.
void execute_mlx(mlx::core::Stream stream, const std::vector<mlx::core::array>& inputs,
                 std::vector<mlx::core::array>& outputs,
                 const std::function<void(MLXSession&)>& operation);

// Select the MLX Metal device for synchronous host setup/factories and restore
// the previous runtime selection. Does not touch or synchronize an MLX stream.
void with_mlx_device(mlx::core::Stream stream,
                     const std::function<void(metalrt::Device)>& operation);

} // namespace mlx_bindings
} // namespace tinymetalnn
