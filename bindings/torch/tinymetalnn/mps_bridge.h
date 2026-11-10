// Framework-specific adaptation stays in bindings/torch. The runtime interop
// API and the neural-network facade contain no ATen, Python, or pybind types.
#pragma once

#include <ATen/core/Tensor.h>
#include <metalrt/runtime.h>
#include <tiny-metal-nn/metal.h>

#include <functional>
#include <memory>

namespace tinymetalnn {
namespace torch_bindings {

struct MPSSession {
    metalrt::Device device;
    std::shared_ptr<metalrt::Stream> queue;
    std::unique_ptr<tcnn::metal::Stream> nn_stream;
    metalrt::Buffer empty_buffer;
};

// Runs on the captured current PyTorch MPS stream's serial dispatch queue.
// Commits that stream asynchronously even when an operation throws after an
// encoded prefix. Callers must release Python's GIL before entering this call.
void execute_mps(const std::function<void(MPSSession&)>& operation);

// A view retains allocator storage, including for aliased tensor slices. Shared
// storage uses its registered real address; Private storage uses GPU-only blits
// to a Shared buffer accepted by the unchanged C++ facade.
struct TensorBuffer {
    metalrt::BufferView original;
    metalrt::Buffer staging;
    metalrt::Access access = metalrt::Access::Read;

    void* data() const { return staging ? staging.contents() : original.data(); }
    template <class T> T* data_as() const { return static_cast<T*>(data()); }
};

TensorBuffer bind_tensor(MPSSession& session, const at::Tensor& tensor, metalrt::Access access);
void copy_back(MPSSession& session, const TensorBuffer& tensor);

} // namespace torch_bindings
} // namespace tinymetalnn
