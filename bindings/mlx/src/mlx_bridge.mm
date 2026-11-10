// Native MLX Primitive bridge. No tensor evaluation, CPU backend, or other
// framework is used to implement an MLX GPU operation.
#include "mlx_bridge.h"

#include <mlx/allocator.h>
#include <mlx/backend/metal/device.h>
#include <metalrt/interop.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>

namespace tinymetalnn {
namespace mlx_bindings {
namespace {
namespace mx = mlx::core;

// Serialize NN host submission/setup, not GPU completion. The core has a
// stream-less parameter initializer; no live adapter submission may publish
// its fresh initialization storage while that initializer writes on the host.
std::mutex submission_mutex;

struct HandoffResources {
    id<MTLEvent> event = nil;
    std::vector<std::shared_ptr<mx::array::Data>> storage;
};

void require(bool condition, const char* message) {
    if (!condition)
        throw std::invalid_argument(message);
}

class DeviceScope {
  public:
    explicit DeviceScope(metalrt::Device device) : previous_{metalrt::Device::current()} {
        metalrt::Device::set_current(std::move(device));
    }
    ~DeviceScope() { metalrt::Device::set_current(previous_); }

  private:
    metalrt::Device previous_;
};

metalrt::Device import_device(mx::Stream stream) {
    require(stream.device == mx::Device::gpu, "tiny-metal-nn MLX primitives require a GPU stream.");
    auto& device = mx::metal::device(stream.device);
    return metalrt::import_metal_device(static_cast<void*>(device.mtl_device()));
}

void validate_input(const mx::array& input) {
    require(input.flags().row_contiguous, "Native MLX inputs must be row-contiguous.");
    require(input.data_shared_ptr() != nullptr,
            "MLX input storage is unavailable; invoke the bridge from eval_gpu.");
    require(input.offset() >= 0, "MLX array has a negative storage byte offset.");
    if (input.nbytes())
        require(input.buffer().ptr() != nullptr, "Nonempty MLX array has no Metal storage.");
}

ArrayBuffer bind_array(MLXSession& session, const mx::array& array, const metalrt::Buffer& empty,
                       bool input) {
    const auto bytes = metalrt::checked_mul(array.size(), array.itemsize());
    if (!bytes)
        return {empty.view(0, 0), {}};
    const auto offset = static_cast<size_t>(array.offset());
    const auto& storage = array.data_shared_ptr();
    // Hold allocator Data, not ArrayDesc/Primitive/graph objects. A Metal retain
    // alone cannot prevent MLX's caching allocator from reusing this storage.
    std::shared_ptr<void> owner{storage, storage.get()};
    auto imported =
        metalrt::import_metal_buffer(session.device, array.buffer().ptr(), std::move(owner),
                                     metalrt::MetalBufferSynchronization::Explicit);
    require(offset <= imported.size() && bytes <= imported.size() - offset,
            "MLX array byte range exceeds its Metal buffer.");
    ArrayBuffer result{imported.view(offset, bytes), {}};
    if (imported.storage_mode() == metalrt::StorageMode::Private) {
        result.staging = session.device.allocate(bytes);
        if (input)
            session.queue->copy(result.original, result.view());
    }
    return result;
}

MTL::ComputePipelineState* publication_kernel(mx::Stream stream) {
    auto& device = mx::metal::device(stream.device);
    // MLX's CommandEncoder owns its fence/output map and commit thresholds. An
    // ordinary one-thread dispatch publishes externally written outputs through
    // those same mechanisms without accessing MLX's private encoder fields.
    auto* library = device.get_library("tinymetalnn_mlx_handoff_v1", []() {
        return std::string{"#include <metal_stdlib>\n"
                           "using namespace metal;\n"
                           "kernel void tinymetalnn_mlx_handoff_v1(uint i "
                           "[[thread_position_in_grid]]) { (void)i; }\n"};
    });
    return device.get_kernel("tinymetalnn_mlx_handoff_v1", library);
}

// Events are the documented command-buffer boundary between independent
// encoders. Signal drains prior passes; wait prevents following passes from
// starting early even for untracked buffers. The event is retained explicitly
// because MLX creates command buffers with unretained resource references.
void handoff(id<MTLCommandBuffer> command_buffer, id<MTLEvent> event, uint64_t value) {
    [command_buffer encodeSignalEvent:event value:value];
    [command_buffer encodeWaitForEvent:event value:value];
}

void publish_outputs(mx::metal::CommandEncoder& encoder, MTL::ComputePipelineState* kernel,
                     std::vector<mx::array>& outputs) {
    encoder.set_compute_pipeline_state(kernel);
    for (auto& output : outputs)
        if (output.nbytes())
            encoder.set_output_array(output, 0);
    encoder.dispatch_threads(MTL::Size{1, 1, 1}, MTL::Size{1, 1, 1});
    // Record the outputs against MLX's own fence, including consumers in a
    // subsequent command buffer. The preceding event already ordered writes.
    encoder.end_encoding();
}
} // namespace

void with_mlx_device(mx::Stream stream, const std::function<void(metalrt::Device)>& operation) {
    std::lock_guard<std::mutex> lock{submission_mutex};
    @autoreleasepool {
        auto device = import_device(stream);
        DeviceScope scope{device};
        operation(device);
    }
}

void execute_mlx(mx::Stream stream, const std::vector<mx::array>& inputs,
                 std::vector<mx::array>& outputs,
                 const std::function<void(MLXSession&)>& operation) {
    std::lock_guard<std::mutex> lock{submission_mutex};
    @autoreleasepool {
        auto device = import_device(stream);
        DeviceScope device_scope{device};
        for (const auto& input : inputs)
            validate_input(input);
        for (auto& output : outputs)
            output.set_data(
                mx::allocator::malloc(metalrt::checked_mul(output.size(), output.itemsize())));

        auto& encoder = mx::metal::get_command_encoder(stream);
        auto* marker = publication_kernel(stream);
        auto* current = encoder.get_command_buffer();
        id<MTLCommandBuffer> command_buffer =
            (__bridge id<MTLCommandBuffer>)static_cast<void*>(current);
        id<MTLDevice> native_device = (__bridge id<MTLDevice>)metalrt::native_metal_device(device);
        id<MTLEvent> event = [native_device newEvent];
        if (!command_buffer || !event)
            throw std::runtime_error("Unable to create MLX/native Metal handoff.");
        auto handoff_resources = std::make_shared<HandoffResources>();
        handoff_resources->event = event;
        handoff_resources->storage.reserve(inputs.size() + outputs.size());
        for (const auto& input : inputs)
            handoff_resources->storage.push_back(input.data_shared_ptr());
        for (const auto& output : outputs)
            handoff_resources->storage.push_back(output.data_shared_ptr());
        [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
          (void)completed;
          (void)handoff_resources;
        }];

        // Register actual dependencies before ending MLX's current encoder.
        // end_encoding() adds waits on its private prior-output fence map. A
        // barrier also opens an encoder for operations without input arrays.
        for (const auto& input : inputs)
            if (input.nbytes())
                encoder.set_input_array(input, 0);
        encoder.barrier();
        encoder.end_encoding();
        handoff(command_buffer, event, 1);

        MLXSession session;
        session.device = device;
        // A transient stream keeps no stale CommandEncoder pointer after MLX
        // clear_streams() or thread teardown. Borrowed destruction never waits.
        session.queue = metalrt::borrow_metal_stream(
            device, [command_buffer]() -> void* { return (__bridge void*)command_buffer; });
        session.nn_stream = std::make_unique<tcnn::metal::Stream>(session.queue);
        auto empty = device.allocate(16);
        std::exception_ptr failure;
        try {
            auto recording = session.queue->batch();
            session.inputs.reserve(inputs.size());
            session.outputs.reserve(outputs.size());
            for (const auto& input : inputs)
                session.inputs.push_back(bind_array(session, input, empty, true));
            for (const auto& output : outputs)
                session.outputs.push_back(bind_array(session, output, empty, false));
            operation(session);
            for (const auto& output : session.outputs)
                if (output.staging)
                    session.queue->copy(output.view(), output.original);
            recording.finish();
            session.queue->check_errors();
        } catch (...) {
            failure = std::current_exception();
        }

        // Restore MLX's encoder/fence accounting after an encoded prefix too.
        // MLX remains responsible for committing work and propagating GPU
        // errors; the bridge never rotates its scheduler's command buffer.
        try {
            handoff(command_buffer, event, 2);
            publish_outputs(encoder, marker, outputs);
        } catch (...) {
            try {
                encoder.end_encoding();
            } catch (...) {
            }
            if (!failure)
                failure = std::current_exception();
        }
        if (failure)
            std::rethrow_exception(failure);
    }
}

} // namespace mlx_bindings
} // namespace tinymetalnn
