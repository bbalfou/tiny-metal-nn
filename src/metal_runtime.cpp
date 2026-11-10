// Metal host resource operations implemented entirely through metalrt.
#include <tiny-metal-nn/metal.h>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <utility>

struct MetalStreamState {
    metalrt::Stream* stream = nullptr;
    bool owns = false;
    bool indirect_dispatch = true;
};

struct MetalEventState {
    metalrt::EventPoint point;
    metalrt::Device device;
};

namespace {
std::mutex registry_mutex;
std::unordered_set<metalStream_t> streams;
std::unordered_set<metalEvent_t> events;
std::unordered_map<void*, metalrt::Buffer> allocations;
thread_local metalError_t last_error = metalSuccess;
thread_local metalError_t message_error = metalSuccess;
thread_local std::string last_message;

// Contain C++ exceptions at the status-returning host boundary.
template <class F> metalError_t checked(F&& fn) noexcept {
    try {
        fn();
        return metalSuccess;
    } catch (const std::bad_alloc& e) {
        last_message = e.what();
        return message_error = last_error = metalErrorMemoryAllocation;
    } catch (const std::invalid_argument& e) {
        last_message = e.what();
        return message_error = last_error = metalErrorInvalidValue;
    } catch (const std::exception& e) {
        last_message = e.what();
        return message_error = last_error = metalErrorUnknown;
    } catch (...) {
        last_message = "unknown Metal execution error";
        return message_error = last_error = metalErrorUnknown;
    }
}

// Reject a foreign event pointer without dereferencing it.
MetalEventState& event_for(metalEvent_t event) {
    std::lock_guard<std::mutex> lock{registry_mutex};
    if (!events.count(event))
        throw std::invalid_argument("event is not owned by tiny-metal-nn");
    return *event;
}

// Resolve a registered range; host pointers legitimately produce an empty result.
metalrt::BufferView optional_view(const void* pointer, size_t bytes, bool host_endpoint) {
    // Probe registration independently of the requested span. A registered range
    // that is too short must never be reclassified as arbitrary host memory.
    // Explicit host endpoints may start exactly at the end of an unrelated
    // allocation. Default/device endpoints must still reject nonempty end views.
    metalrt::BufferView probe;
    try {
        probe = metalrt::resolve_pointer(pointer, 0);
    } catch (const std::exception&) {
        return {};
    }
    if (host_endpoint && probe.offset == probe.buffer.size())
        return {};
    return metalrt::resolve_pointer(pointer, bytes);
}

// Stage host input immediately, enqueue GPU copies, and complete host readback explicitly.
void copy_bytes(metalStream_t stream, void* dst, const void* src, size_t bytes,
                metalMemcpyKind kind) {
    if (!bytes)
        return;
    if (!dst || !src)
        throw std::invalid_argument("null memcpy pointer");
    auto& queue = *tcnn::detail::resolve_stream(stream);
    auto source =
        optional_view(src, bytes, kind == metalMemcpyHostToDevice || kind == metalMemcpyHostToHost);
    auto destination =
        optional_view(dst, bytes, kind == metalMemcpyDeviceToHost || kind == metalMemcpyHostToHost);
    if (kind == metalMemcpyDefault)
        kind = source ? (destination ? metalMemcpyDeviceToDevice : metalMemcpyDeviceToHost)
                      : (destination ? metalMemcpyHostToDevice : metalMemcpyHostToHost);
    if (kind == metalMemcpyDeviceToDevice) {
        if (!source || !destination)
            throw std::invalid_argument("device memcpy needs registered allocations");
        if (source == destination)
            return;
        queue.copy(source, destination);
    } else if (kind == metalMemcpyHostToDevice) {
        if (!destination)
            throw std::invalid_argument("host-to-device destination is unregistered");
        if (source)
            throw std::invalid_argument("host-to-device source is a registered device allocation");
        auto staging = queue.device().allocate(bytes);
        std::memcpy(staging.contents(), src, bytes);
        queue.copy(staging.view(), destination);
    } else if (kind == metalMemcpyDeviceToHost) {
        if (!source)
            throw std::invalid_argument("device-to-host source is unregistered");
        if (destination)
            throw std::invalid_argument(
                "device-to-host destination is a registered device allocation");
        if (source.buffer.device() != queue.device())
            throw std::invalid_argument(
                "device-to-host source and stream belong to different devices");
        // metalrt has no host callback command; this permitted synchronous readback
        // keeps the host destination lifetime and completion semantics unambiguous.
        queue.host([&] { std::memmove(dst, src, bytes); });
    } else if (kind == metalMemcpyHostToHost) {
        if (source || destination)
            throw std::invalid_argument("host-to-host copy cannot bypass device-buffer ownership");
        std::memmove(dst, src, bytes);
    } else
        throw std::invalid_argument("invalid memcpy direction");
}
} // namespace

namespace tcnn {
namespace detail {
metalrt::Stream* resolve_stream(metalStream_t stream) {
    if (!stream)
        return &metalrt::ordered_default_stream();
    std::lock_guard<std::mutex> lock{registry_mutex};
    if (!streams.count(stream))
        throw std::invalid_argument("stream is not a tiny-metal-nn Metal stream");
    return stream->stream;
}

metalStream_t register_stream_with_ordering(metalrt::Stream* stream,
                                            metalrt::StreamOrdering ordering) {
    if (!stream)
        throw std::invalid_argument("cannot register a null runtime stream");
    auto handle = std::make_unique<MetalStreamState>();
    handle->stream = stream;
    std::lock_guard<std::mutex> lock{registry_mutex};
    const bool already_registered =
        std::any_of(streams.begin(), streams.end(),
                    [stream](metalStream_t handle) { return handle->stream == stream; });
    if (!already_registered && stream->ordering() != metalrt::StreamOrdering::Default)
        stream->set_ordering(ordering);
    streams.insert(handle.get());
    return handle.release();
}

metalStream_t register_stream(metalrt::Stream* stream) {
    return register_stream_with_ordering(stream, metalrt::StreamOrdering::Blocking);
}

void unregister_stream(metalStream_t stream) {
    if (!stream)
        return;
    std::lock_guard<std::mutex> lock{registry_mutex};
    if (!streams.erase(stream))
        throw std::invalid_argument("stream is not registered");
    delete stream;
}

void set_indirect_dispatch_enabled(metalStream_t stream, bool enabled) {
    std::lock_guard<std::mutex> lock{registry_mutex};
    if (!streams.count(stream))
        throw std::invalid_argument("stream is not registered");
    stream->indirect_dispatch = enabled;
}

bool indirect_dispatch_enabled(metalStream_t stream) {
    if (!stream)
        return true;
    std::lock_guard<std::mutex> lock{registry_mutex};
    if (!streams.count(stream))
        throw std::invalid_argument("stream is not registered");
    return stream->indirect_dispatch;
}

void copy_device_sync(metalrt::BufferView source, metalrt::BufferView destination) {
    if (!source.length && !destination.length)
        return;
    if (source.buffer.device() != destination.buffer.device())
        throw std::invalid_argument("device copy requires one owning device");
    auto device = destination.buffer.device();
    metalrt::synchronize_device(device);
    metalrt::Stream queue{device, metalrt::DispatchPolicy::Direct};
    queue.copy(std::move(source), std::move(destination));
    queue.synchronize();
}

void fill_device_sync(metalrt::BufferView destination, uint8_t value) {
    if (!destination.length)
        return;
    auto device = destination.buffer.device();
    metalrt::synchronize_device(device);
    metalrt::Stream queue{device, metalrt::DispatchPolicy::Direct};
    queue.fill(std::move(destination), value);
    queue.synchronize();
}
} // namespace detail

namespace metal {
Stream::Stream(metalrt::Device device, metalrt::DispatchPolicy policy)
    : m_owned{std::make_unique<metalrt::Stream>(std::move(device), policy)},
      m_handle{detail::register_stream(m_owned.get())} {}

Stream::Stream(metalrt::Stream& stream) : m_handle{detail::register_stream(&stream)} {}

Stream::Stream(std::shared_ptr<metalrt::Stream> stream) : m_retained{std::move(stream)} {
    if (!m_retained)
        throw std::invalid_argument("cannot retain a null runtime stream");
    m_handle = detail::register_stream_with_ordering(m_retained.get(), m_retained->ordering());
}

Stream::~Stream() {
    if (m_handle) {
        try {
            detail::unregister_stream(m_handle);
        } catch (...) {
        }
    }
}

Stream::Stream(Stream&& other) noexcept
    : m_owned{std::move(other.m_owned)}, m_retained{std::move(other.m_retained)},
      m_handle{std::exchange(other.m_handle, nullptr)} {}

Stream& Stream::operator=(Stream&& other) noexcept {
    if (this != &other) {
        if (m_handle) {
            try {
                detail::unregister_stream(m_handle);
            } catch (...) {
            }
        }
        m_owned = std::move(other.m_owned);
        m_retained = std::move(other.m_retained);
        m_handle = std::exchange(other.m_handle, nullptr);
    }
    return *this;
}
} // namespace metal
} // namespace tcnn

extern "C" {
const char* metalGetErrorString(metalError_t error) {
    if (error == metalSuccess)
        return "success";
    if (error == message_error && !last_message.empty())
        return last_message.c_str();
    return metalGetErrorName(error);
}

const char* metalGetErrorName(metalError_t error) {
    switch (error) {
    case metalSuccess:
        return "metalSuccess";
    case metalErrorInvalidValue:
        return "metalErrorInvalidValue";
    case metalErrorMemoryAllocation:
        return "metalErrorMemoryAllocation";
    case metalErrorInitializationError:
        return "metalErrorInitializationError";
    case metalErrorInvalidDevice:
        return "metalErrorInvalidDevice";
    case metalErrorInvalidResourceHandle:
        return "metalErrorInvalidResourceHandle";
    case metalErrorNotReady:
        return "metalErrorNotReady";
    case metalErrorNotSupported:
        return "metalErrorNotSupported";
    default:
        return "metalErrorUnknown";
    }
}

metalError_t metalGetLastError() {
    auto result = last_error;
    last_error = metalSuccess;
    return result;
}

metalError_t metalPeekAtLastError() {
    return last_error;
}

metalError_t metalGetDevice(int* device) {
    return checked([&] {
        if (!device)
            throw std::invalid_argument("null device output");
        (void)metalrt::Device::current();
        *device = 0;
    });
}

metalError_t metalSetDevice(int device) {
    return checked([&] {
        if (device != 0)
            throw std::invalid_argument("only selected Metal device index 0 exists");
        (void)metalrt::Device::current();
    });
}

metalError_t metalGetDeviceCount(int* count) {
    return checked([&] {
        if (!count)
            throw std::invalid_argument("null device count");
        (void)metalrt::Device::current();
        *count = 1;
    });
}

metalError_t metalDeviceSynchronize() {
    return checked([] { metalrt::synchronize_device(); });
}

metalError_t metalStreamCreate(metalStream_t* stream) {
    return metalStreamCreateWithFlags(stream, metalStreamDefault);
}

metalError_t metalStreamCreateWithFlags(metalStream_t* stream, unsigned int flags) {
    return checked([&] {
        if (!stream || flags > metalStreamNonBlocking)
            throw std::invalid_argument("invalid stream creation argument");
        auto queue = std::make_unique<metalrt::Stream>();
        auto handle = tcnn::detail::register_stream_with_ordering(
            queue.get(), flags == metalStreamNonBlocking ? metalrt::StreamOrdering::Independent
                                                         : metalrt::StreamOrdering::Blocking);
        handle->owns = true;
        queue.release();
        *stream = handle;
    });
}

metalError_t metalStreamDestroy(metalStream_t stream) {
    return checked([&] {
        if (!stream)
            return;
        auto* queue = tcnn::detail::resolve_stream(stream);
        if (!stream->owns)
            throw std::invalid_argument("borrowed Metal stream must be destroyed by its owner");
        // Stream owns completion-aware retirement, including work deferred
        // behind another queue's still-cancellable recording.
        try {
            queue->synchronize();
        } catch (const metalrt::RecordingDependencyError&) {
            // The queue destructor retains this unresolved work until its
            // producer commits. Other errors still reach the status boundary.
        }
        tcnn::detail::unregister_stream(stream);
        delete queue;
    });
}

metalError_t metalStreamSynchronize(metalStream_t stream) {
    return checked([&] { tcnn::detail::resolve_stream(stream)->synchronize(); });
}

metalError_t metalStreamQuery(metalStream_t stream) {
    bool ready = false;
    auto result = checked([&] {
        auto* queue = tcnn::detail::resolve_stream(stream);
        queue->check_errors();
        ready = queue->completion_ticket().ready();
    });
    return result == metalSuccess && !ready ? metalErrorNotReady : result;
}

metalError_t metalMalloc(void** ptr, size_t size) {
    return checked([&] {
        if (!ptr)
            throw std::invalid_argument("null allocation output");
        if (!size) {
            *ptr = nullptr;
            return;
        }
        auto buffer = metalrt::Device::current().allocate(size);
        void* data = buffer.contents();
        std::lock_guard<std::mutex> lock{registry_mutex};
        allocations.emplace(data, std::move(buffer));
        *ptr = data;
    });
}

metalError_t metalMallocManaged(void** ptr, size_t size, unsigned int flags) {
    if (flags != metalMemAttachGlobal)
        return checked(
            [] { throw std::invalid_argument("Metal shared allocation flags are invalid"); });
    return metalMalloc(ptr, size);
}

metalError_t metalFree(void* ptr) {
    return checked([&] {
        if (!ptr)
            return;
        metalrt::Buffer buffer;
        {
            std::lock_guard<std::mutex> lock{registry_mutex};
            auto it = allocations.find(ptr);
            if (it == allocations.end())
                throw std::invalid_argument("metalFree needs a Metal host allocation");
            buffer = it->second;
        }
        metalrt::synchronize_device(buffer.device());
        std::lock_guard<std::mutex> lock{registry_mutex};
        allocations.erase(ptr);
    });
}

metalError_t metalMemset(void* devPtr, int value, size_t count) {
    return checked([&] {
        if (!count)
            return;
        tcnn::detail::fill_device_sync(metalrt::resolve_pointer(devPtr, count),
                                       static_cast<uint8_t>(value));
    });
}

metalError_t metalMemsetAsync(void* devPtr, int value, size_t count, metalStream_t stream) {
    return checked([&] {
        if (count)
            tcnn::detail::resolve_stream(stream)->fill(metalrt::resolve_pointer(devPtr, count),
                                                       static_cast<uint8_t>(value));
    });
}

metalError_t metalMemcpy(void* dst, const void* src, size_t count, metalMemcpyKind kind) {
    return checked([&] {
        if (!count)
            return;
        metalrt::synchronize_device();
        copy_bytes(nullptr, dst, src, count, kind);
        tcnn::detail::resolve_stream(nullptr)->synchronize();
    });
}

metalError_t metalMemcpyAsync(void* dst, const void* src, size_t count, metalMemcpyKind kind,
                              metalStream_t stream) {
    return checked([&] { copy_bytes(stream, dst, src, count, kind); });
}

metalError_t metalEventCreate(metalEvent_t* event) {
    return metalEventCreateWithFlags(event, metalEventDefault);
}

metalError_t metalEventCreateWithFlags(metalEvent_t* event, unsigned int flags) {
    return checked([&] {
        if (!event || flags > 3)
            throw std::invalid_argument("invalid event arguments");
        auto handle = std::make_unique<MetalEventState>();
        handle->device = metalrt::Device::current();
        std::lock_guard<std::mutex> lock{registry_mutex};
        events.insert(handle.get());
        *event = handle.release();
    });
}

metalError_t metalEventDestroy(metalEvent_t event) {
    return checked([&] {
        std::lock_guard<std::mutex> lock{registry_mutex};
        if (!events.erase(event))
            throw std::invalid_argument("unknown event");
        delete event;
    });
}

metalError_t metalEventRecord(metalEvent_t event, metalStream_t stream) {
    return checked([&] {
        auto& handle = event_for(event);
        auto* queue = tcnn::detail::resolve_stream(stream);
        if (queue->device() != handle.device)
            throw std::invalid_argument("cross-device event");
        metalrt::TimelineEvent timeline{handle.device};
        handle.point = queue->signal(timeline);
    });
}

metalError_t metalEventSynchronize(metalEvent_t event) {
    return checked([&] { event_for(event).point.wait(); });
}

metalError_t metalEventQuery(metalEvent_t event) {
    bool ready = false;
    auto result = checked([&] { ready = event_for(event).point.ready(); });
    return result == metalSuccess && !ready ? metalErrorNotReady : result;
}

metalError_t metalStreamWaitEvent(metalStream_t stream, metalEvent_t event, unsigned int flags) {
    return checked([&] {
        if (flags)
            throw std::invalid_argument("nonzero event wait flags");
        auto& handle = event_for(event);
        if (handle.point)
            tcnn::detail::resolve_stream(stream)->wait(handle.point);
    });
}
}
