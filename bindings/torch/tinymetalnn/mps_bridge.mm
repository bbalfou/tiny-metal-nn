// Native PyTorch MPS adapter. No tensors are evaluated on a CPU backend.
#include "mps_bridge.h"
#include "native_log.h"

#include <ATen/mps/MPSStream.h>
#include <torch/mps.h>
#include <metalrt/interop.h>

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <exception>
#include <map>
#include <mutex>

namespace tinymetalnn {
namespace torch_bindings {
namespace {

class DeviceScope {
  public:
    explicit DeviceScope(metalrt::Device device) : previous_{metalrt::Device::current()} {
        metalrt::Device::set_current(std::move(device));
    }
    ~DeviceScope() { metalrt::Device::set_current(previous_); }

  private:
    metalrt::Device previous_;
};

// Submission is serialized, not GPU execution. This also keeps the existing
// facade's stream-less parameter initializer from synchronizing a half-encoded
// invocation on another PyTorch stream. No lock is held by GPU callbacks.
std::mutex submission_mutex;

std::shared_ptr<MPSSession> session_for(at::mps::MPSStream* stream) {
    // Accessed only while submission_mutex is held. PyTorch owns its streams;
    // retaining this adapter never transfers command-buffer commit ownership.
    static std::map<at::mps::MPSStream*, std::shared_ptr<MPSSession>> sessions;
    auto& result = sessions[stream];
    if (!result) {
        auto session = std::make_shared<MPSSession>();
        session->device = metalrt::import_metal_device((__bridge void*)stream->device());
        session->queue = metalrt::borrow_metal_stream(session->device, [stream]() -> void* {
            // MPSCommandBuffer is a mutable commitAndContinue wrapper. Retain
            // the exact underlying command buffer for completion and errors.
            return (__bridge void*)[stream->commandBuffer() rootCommandBuffer];
        });
        session->nn_stream = std::make_unique<tcnn::metal::Stream>(session->queue);
        // AGX indirect threadgroup encoding can crash on borrowed MPS command
        // buffers. Keep GPU-side algorithm selection with guarded direct grids.
        tcnn::detail::set_indirect_dispatch_enabled(session->nn_stream->get(), false);
        session->empty_buffer = session->device.allocate(16);
        result = std::move(session);
    }
    return result;
}
} // namespace

void execute_mps(const std::function<void(MPSSession&)>& operation) {
    TORCH_CHECK(torch::mps::is_available(),
                "tiny-metal-nn requires an available PyTorch MPS device.");
    // Capture on the calling thread: querying current_stream inside a GCD block
    // would lose thread-local stream selection in multistream PyTorch versions.
    auto* stream = at::mps::getCurrentMPSStream();
    auto log_records = current_native_logs();
    std::lock_guard<std::mutex> submission_lock{submission_mutex};
    __block std::exception_ptr failure;
    dispatch_sync(stream->queue(), ^{
      @autoreleasepool {
          NativeLogCapture log_capture{log_records};
          try {
              stream->endKernelCoalescing();
              auto session = session_for(stream);
              DeviceScope device_scope{session->device};
              session->queue->check_errors();
              operation(*session);
              session->queue->check_errors();
          } catch (...) {
              failure = std::current_exception();
          }
          // Work encoded before an exception must still be committed so its
          // lifetime completion handlers run. This does not wait for the GPU.
          try {
              stream->synchronize(at::mps::SyncType::COMMIT);
          } catch (...) {
              if (!failure)
                  failure = std::current_exception();
          }
      }
    });
    if (failure)
        std::rethrow_exception(failure);
}

TensorBuffer bind_tensor(MPSSession& session, const at::Tensor& tensor, metalrt::Access access) {
    TORCH_CHECK(tensor.defined() && tensor.device().is_mps(), "Expected an MPS tensor.");
    TORCH_CHECK(tensor.is_contiguous(), "Native tensor views must be contiguous.");
    const auto bytes =
        metalrt::checked_mul(static_cast<size_t>(tensor.numel()), tensor.element_size());
    if (!bytes)
        return {session.empty_buffer.view(0, 0), {}, access};

    const auto storage = tensor.storage();
    const auto offset =
        metalrt::checked_mul(static_cast<size_t>(tensor.storage_offset()), tensor.element_size());
    TORCH_CHECK(offset <= storage.nbytes() && bytes <= storage.nbytes() - offset,
                "MPS tensor view exceeds its storage.");
    // MPS storage().data() is an opaque id<MTLBuffer>, not a scalar address.
    // This is the storage representation used by ATen's getMTLBufferStorage.
    id<MTLBuffer> native_buffer = (__bridge id<MTLBuffer>)storage.data();
    TORCH_CHECK(native_buffer != nil, "Nonempty MPS tensor has no Metal buffer.");
    auto storage_owner = std::make_shared<c10::Storage>(storage);
    std::shared_ptr<void> owner{storage_owner, storage.unsafeGetStorageImpl()};
    auto imported = metalrt::import_metal_buffer(session.device, (__bridge void*)native_buffer,
                                                 std::move(owner));
    TensorBuffer result{imported.view(offset, bytes), {}, access};
    if (imported.storage_mode() == metalrt::StorageMode::Private) {
        result.staging = session.device.allocate(bytes);
        if (access != metalrt::Access::Write)
            session.queue->copy(result.original, result.staging.view(0, bytes));
    }
    return result;
}

void copy_back(MPSSession& session, const TensorBuffer& tensor) {
    if (tensor.staging && tensor.access != metalrt::Access::Read && tensor.original.length)
        session.queue->copy(tensor.staging.view(0, tensor.original.length), tensor.original);
}

} // namespace torch_bindings
} // namespace tinymetalnn
