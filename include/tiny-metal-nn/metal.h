// Explicit Metal backend selection and RAII queue ownership for the NN host API.
#pragma once
#include <tiny-metal-nn/metal_runtime.h>
#include <metalrt/runtime.h>
#include <memory>

namespace tcnn {
namespace detail {
// Validate a Metal-owned handle and return its underlying runtime queue.
metalrt::Stream* resolve_stream(metalStream_t stream);
// Register a queue without transferring ownership; its caller must outlive the handle.
metalStream_t register_stream(metalrt::Stream* stream);
void unregister_stream(metalStream_t stream);
// Adapter capability: some borrowed command buffers cannot encode GPU-sized
// dispatches reliably. Kernels with guarded direct grids can use that route.
void set_indirect_dispatch_enabled(metalStream_t stream, bool enabled);
bool indirect_dispatch_enabled(metalStream_t stream);
// Synchronous device transfers still execute actual runtime copy/fill commands.
// They first complete producer work and then wait for their owning-device queue.
void copy_device_sync(metalrt::BufferView source, metalrt::BufferView destination);
void fill_device_sync(metalrt::BufferView destination, uint8_t value);
} // namespace detail

namespace metal {
// Select the metalrt device used by nullptr streams and subsequent NN allocations.
inline void set_default_device(metalrt::Device device) {
    metalrt::Device::set_current(std::move(device));
}

// Obtain the runtime queue for advanced graph/event integration without changing NN signatures.
inline metalrt::Stream& runtime_stream(metalStream_t stream = nullptr) {
    return *detail::resolve_stream(stream);
}

// Own a queue and its Metal handle, or borrow an application-owned queue.
class Stream {
  public:
    explicit Stream(metalrt::Device device = metalrt::Device::current(),
                    metalrt::DispatchPolicy policy = metalrt::DispatchPolicy::Direct);
    explicit Stream(metalrt::Stream& stream);
    // Retain an alias of an existing runtime queue without changing its
    // ordering role. Suitable for contexts whose lazy work outlives the caller.
    explicit Stream(std::shared_ptr<metalrt::Stream> stream);
    ~Stream();
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    Stream(Stream&&) noexcept;
    Stream& operator=(Stream&&) noexcept;

    // Pass this handle unchanged to every neural and Metal host operation.
    metalStream_t get() const { return m_handle; }

    operator metalStream_t() const { return m_handle; }

    // Access graph recording, event submission, counters and synchronization.
    metalrt::Stream& runtime() const { return *detail::resolve_stream(m_handle); }

  private:
    std::unique_ptr<metalrt::Stream> m_owned;
    std::shared_ptr<metalrt::Stream> m_retained;
    metalStream_t m_handle = nullptr;
};
} // namespace metal
} // namespace tcnn
