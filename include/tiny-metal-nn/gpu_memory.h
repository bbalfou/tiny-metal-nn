#pragma once
#include <tiny-metal-nn/common_host.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <tuple>
#include <map>
#include <unordered_map>
#include <vector>

namespace tcnn {
// Compatibility accounting: requested bytes held by live GPUMemory owners and
// workspace cache entries. Captured Buffer references may outlive these owners;
// this is not a physical-device memory-usage query.
inline std::atomic<size_t>& total_n_bytes_allocated() {
    static std::atomic<size_t> count{0};
    return count;
}

template <class T> class GPUMemory {
    static_assert(std::is_trivially_copyable<T>::value,
                  "GPU buffers require trivially copyable elements");

  public:
    using Type = T;
    using View = T*;
    using ConstView = const T*;

    GPUMemory() : m_device{metalrt::Device::current()} {}

    // Allocate the requested number of elements on the selected device.
    GPUMemory(size_t size, bool managed = false)
        : m_device{metalrt::Device::current()}, m_managed{managed} {
        resize(size);
    }

    GPUMemory(GPUMemory&& other)
        : m_buffer{std::move(other.m_buffer)}, m_size{other.m_size},
          m_device{std::move(other.m_device)}, m_managed{other.m_managed} {
        other.m_size = 0;
    }

    GPUMemory& operator=(GPUMemory&& other) {
        if (this != &other) {
            using std::swap;
            swap(m_buffer, other.m_buffer);
            swap(m_size, other.m_size);
            swap(m_device, other.m_device);
            swap(m_managed, other.m_managed);
        }
        return *this;
    }

    explicit GPUMemory(const GPUMemory& other)
        : m_device{other.m_device}, m_managed{other.m_managed} {
        resize(other.m_size);
        copy_from_device(other);
    }

    GPUMemory& operator=(const GPUMemory&) = delete;

    ~GPUMemory() { free_memory(); }

    void check_guards() const {} // Dispatch binding and view creation validate ranges.

    void allocate_memory(size_t bytes) {
        if (bytes % sizeof(T))
            throw std::invalid_argument("allocation byte count is not an element multiple");
        resize(bytes / sizeof(T));
    }

    void free_memory() {
        total_n_bytes_allocated().fetch_sub(m_size * sizeof(T), std::memory_order_relaxed);
        m_buffer = {};
        m_size = 0;
    }

    void resize(size_t size) {
        if (size == m_size)
            return;
        const auto bytes = detail::checked_mul(size, sizeof(T));
        if (!m_device)
            m_device = metalrt::Device::current();
        auto next =
            bytes ? m_device.allocate(bytes, metalrt::StorageMode::Shared) : metalrt::Buffer{};
        total_n_bytes_allocated().fetch_sub(m_size * sizeof(T), std::memory_order_relaxed);
        m_buffer = std::move(next);
        m_size = size;
        total_n_bytes_allocated().fetch_add(bytes, std::memory_order_relaxed);
    }

    void enlarge(size_t size) {
        if (size > m_size)
            resize(size);
    }

    T* data() const { return m_buffer ? m_buffer.template data<T>() : nullptr; }

    View view() const { return data(); }

    ConstView const_view() const { return data(); }

    bool managed() const { return m_managed; }

    metalrt::Device device() const { return m_device; }

    const metalrt::Buffer& buffer() const { return m_buffer; }

    metalrt::BufferView buffer_view(size_t offset_elements = 0, size_t count = SIZE_MAX) const {
        if (count == SIZE_MAX) {
            if (offset_elements > m_size)
                throw std::out_of_range("buffer offset");
            count = m_size - offset_elements;
        }
        detail::check_range(offset_elements, count, m_size);
        if (!m_buffer)
            return {};
        return m_buffer.view(detail::checked_mul(offset_elements, sizeof(T)),
                             detail::checked_mul(count, sizeof(T)));
    }

    // at() retains the reference's managed-only host-access contract.
    T& at(size_t idx) const {
        if (!m_managed)
            throw std::runtime_error("GPUMemory::at() not permitted if not managed");
        if (idx >= m_size)
            throw std::out_of_range("GPUMemory::at");
        return data()[idx];
    }

    T& operator[](size_t idx) const { return data()[idx]; }

    T& operator[](uint32_t idx) const { return data()[idx]; }

    size_t get_num_elements() const { return m_size; }

    size_t size() const { return m_size; }

    size_t get_bytes() const { return m_size * sizeof(T); }

    size_t n_bytes() const { return get_bytes(); }

    size_t bytes() const { return get_bytes(); }

    void memset(int value, size_t count, size_t offset = 0) {
        detail::check_range(offset, count, m_size);
        if (!count)
            return;
        detail::fill_device_sync(buffer_view(offset, count), static_cast<uint8_t>(value));
    }

    void memset(int value) { memset(value, m_size); }

    void copy_from_host(const T* input, size_t count) {
        detail::check_range(0, count, m_size);
        if (!count)
            return;
        if (!input)
            throw std::invalid_argument("null host source");
        metalrt::synchronize_buffer(buffer_view(0, count));
        std::memmove(data(), input, count * sizeof(T));
    }

    void copy_from_host(const T* input) { copy_from_host(input, m_size); }

    void copy_from_host(const std::vector<T>& input, size_t count) {
        detail::check_range(0, count, input.size());
        copy_from_host(input.data(), count);
    }

    void copy_from_host(const std::vector<T>& input) { copy_from_host(input, m_size); }

    void enlarge_and_copy_from_host(const T* input, size_t count) {
        enlarge(count);
        copy_from_host(input, count);
    }

    void enlarge_and_copy_from_host(const std::vector<T>& input, size_t count) {
        detail::check_range(0, count, input.size());
        enlarge_and_copy_from_host(input.data(), count);
    }

    void enlarge_and_copy_from_host(const std::vector<T>& input) {
        enlarge_and_copy_from_host(input.data(), input.size());
    }

    void resize_and_copy_from_host(const T* input, size_t count) {
        resize(count);
        copy_from_host(input, count);
    }

    void resize_and_copy_from_host(const std::vector<T>& input, size_t count) {
        detail::check_range(0, count, input.size());
        resize_and_copy_from_host(input.data(), count);
    }

    void resize_and_copy_from_host(const std::vector<T>& input) {
        resize_and_copy_from_host(input.data(), input.size());
    }

    void copy_to_host(T* output, size_t count) const {
        detail::check_range(0, count, m_size);
        if (!count)
            return;
        if (!output)
            throw std::invalid_argument("null host destination");
        metalrt::synchronize_device(m_device);
        std::memmove(output, data(), count * sizeof(T));
    }

    void copy_to_host(T* output) const { copy_to_host(output, m_size); }

    void copy_to_host(std::vector<T>& output, size_t count) const {
        detail::check_range(0, count, output.size());
        copy_to_host(output.data(), count);
    }

    void copy_to_host(std::vector<T>& output) const { copy_to_host(output, m_size); }

    void copy_from_device(const GPUMemory& other, size_t count) {
        detail::check_range(0, count, other.m_size);
        if (m_device != other.m_device)
            throw std::invalid_argument("device copy requires the same Device");
        if (this == &other || !count)
            return;
        enlarge(count);
        detail::copy_device_sync(other.buffer_view(0, count), buffer_view(0, count));
    }

    void copy_from_device(const GPUMemory& other) { copy_from_device(other, other.m_size); }

    // The uploaded copy(size) retains all source elements, enlarging size if needed.
    GPUMemory copy(size_t new_size) const {
        GPUMemory result;
        result.m_device = m_device;
        result.resize(new_size);
        result.copy_from_device(*this);
        return result;
    }

    GPUMemory copy() const { return copy(m_size); }

  private:
    metalrt::Buffer m_buffer;
    size_t m_size = 0;
    metalrt::Device m_device;
    bool m_managed = false;
};

inline size_t align_to_cacheline(size_t bytes) {
    return next_multiple(bytes, size_t{128});
}

// A capture-safe pool using independently retained storage. Each
// allocation is an independent MTLBuffer. Reuse requires exclusive ownership of
// BOTH the GPUMemory wrapper and its underlying runtime Buffer allocation.
class GPUMemoryArena : public std::enable_shared_from_this<GPUMemoryArena> {
  public:
    // The uploaded public arena constructor uses the active device.
    GPUMemoryArena() : m_device{metalrt::Device::current()} {}

    GPUMemoryArena(const GPUMemoryArena&) = delete;
    GPUMemoryArena& operator=(const GPUMemoryArena&) = delete;

    GPUMemoryArena(GPUMemoryArena&& other)
        : std::enable_shared_from_this<GPUMemoryArena>{}, m_device{other.m_device} {
        std::lock_guard<std::mutex> lock(other.m_mutex);
        m_pool = std::move(other.m_pool);
        m_fallback_memory = std::move(other.m_fallback_memory);
        m_allocated_intervals = std::move(other.m_allocated_intervals);
        m_high_water = std::exchange(other.m_high_water, 0);
        m_append_end = std::exchange(other.m_append_end, 0);
    }

    GPUMemoryArena& operator=(GPUMemoryArena&&) = delete;

    metalrt::Device device() const { return m_device; }

    class Allocation {
      public:
        Allocation() = default;

        Allocation(metalStream_t stream, std::shared_ptr<GPUMemory<uint8_t>> memory,
                   std::shared_ptr<GPUMemoryArena> arena, size_t requested)
            : m_stream{stream}, m_memory{std::move(memory)}, m_workspace{std::move(arena)},
              m_requested{requested} {}

        // Retain the backing allocation of a raw arena offset across later growth.
        Allocation(metalStream_t stream, size_t offset,
                   const std::shared_ptr<GPUMemoryArena>& workspace)
            : m_stream{stream}, m_memory{workspace->backing_memory()}, m_workspace{workspace},
              m_requested{workspace->block_size(offset)}, m_offset{offset} {}

        ~Allocation() { release(); }

        Allocation(const Allocation&) = delete;
        Allocation& operator=(const Allocation&) = delete;

        Allocation(Allocation&& other) { *this = std::move(other); }

        Allocation& operator=(Allocation&& other) {
            if (this != &other) {
                release();
                m_stream = other.m_stream;
                m_memory = std::move(other.m_memory);
                m_workspace = std::move(other.m_workspace);
                m_requested = std::exchange(other.m_requested, 0);
                m_offset = std::exchange(other.m_offset, SIZE_MAX);
            }
            return *this;
        }

        uint8_t* data() {
            return m_memory ? m_memory->data() + (m_offset == SIZE_MAX ? 0 : m_offset) : nullptr;
        }

        const uint8_t* data() const {
            return m_memory ? m_memory->data() + (m_offset == SIZE_MAX ? 0 : m_offset) : nullptr;
        }

        metalStream_t stream() const { return m_stream; }

        size_t size() const { return m_requested; }

        metalrt::BufferView buffer_view() const {
            return m_memory
                       ? m_memory->buffer_view(m_offset == SIZE_MAX ? 0 : m_offset, m_requested)
                       : metalrt::BufferView{};
        }

        const std::shared_ptr<GPUMemory<uint8_t>>& backing_memory() const { return m_memory; }

        explicit operator bool() const { return static_cast<bool>(m_memory); }

      private:
        void release() noexcept {
            if (m_offset != SIZE_MAX && m_workspace) {
                try {
                    m_workspace->free(m_offset);
                } catch (...) {
                }
            }
            m_offset = SIZE_MAX;
            m_memory.reset();
            m_workspace.reset();
        }

        metalStream_t m_stream = nullptr;
        std::shared_ptr<GPUMemory<uint8_t>> m_memory;
        std::shared_ptr<GPUMemoryArena> m_workspace;
        size_t m_requested = 0;
        size_t m_offset = SIZE_MAX;
    };

    Allocation acquire(metalStream_t stream, size_t bytes) {
        if (!bytes)
            return {};
        auto self = shared_from_this();
        if (m_device != metalrt::Device::current())
            throw std::invalid_argument("workspace arena belongs to another selected device");
        std::lock_guard<std::mutex> lock(m_mutex);
        std::shared_ptr<GPUMemory<uint8_t>>* best = nullptr;
        for (auto& entry : m_pool) {
            if (entry.use_count() == 1 && entry->buffer().use_count() == 1 &&
                entry->size() >= bytes && (!best || entry->size() < (*best)->size()))
                best = &entry;
        }
        if (best)
            return {stream, *best, std::move(self), bytes};
        auto entry = std::make_shared<GPUMemory<uint8_t>>(align_to_cacheline(bytes), true);
        m_pool.push_back(entry);
        return {stream, std::move(entry), std::move(self), bytes};
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        size_t bytes = 0;
        for (const auto& p : m_pool)
            bytes = detail::checked_add(bytes, p->n_bytes());
        bytes = detail::checked_add(bytes, m_high_water);
        return bytes;
    }

    bool in_use() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (const auto& p : m_pool)
            if (p.use_count() > 1 || p->buffer().use_count() > 1)
                return true;
        return !m_allocated_intervals.empty();
    }

    void free_unused() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_pool.erase(std::remove_if(m_pool.begin(), m_pool.end(),
                                    [](const auto& p) {
                                        return p.use_count() == 1 && p->buffer().use_count() == 1;
                                    }),
                     m_pool.end());
    }

    // The raw-offset surface uses the reference's portable backing-buffer fallback.
    // Stream workspaces use acquire() above so captured work never forces arena growth.
    uint8_t* data() {
        std::lock_guard<std::mutex> lock{m_mutex};
        return m_fallback_memory ? m_fallback_memory->data() : nullptr;
    }

    std::shared_ptr<GPUMemory<uint8_t>> backing_memory() {
        std::lock_guard<std::mutex> lock{m_mutex};
        return m_fallback_memory;
    }

    // Allocate a cache-line-aligned interval; live Buffer views defer hole reuse.
    size_t allocate(size_t n_bytes) {
        const auto bytes = align_to_cacheline(std::max(n_bytes, size_t{1}));
        std::lock_guard<std::mutex> lock{m_mutex};
        size_t offset = 0;
        if (m_fallback_memory && m_fallback_memory->buffer().use_count() > 1)
            offset = m_append_end;
        else
            for (const auto& block : m_allocated_intervals) {
                if (block.first >= offset && block.first - offset >= bytes)
                    break;
                offset = std::max(offset, detail::checked_add(block.first, block.second));
            }
        const auto end = detail::checked_add(offset, bytes);
        enlarge_locked(end);
        m_allocated_intervals.emplace(offset, bytes);
        m_high_water = std::max(m_high_water, end);
        m_append_end = std::max(m_append_end, end);
        return offset;
    }

    // Release exactly one previously allocated offset, rejecting double frees.
    void free(size_t start) {
        std::lock_guard<std::mutex> lock{m_mutex};
        if (!m_allocated_intervals.erase(start))
            throw std::invalid_argument("arena offset was not allocated");
        m_high_water = m_allocated_intervals.empty()
                           ? 0
                           : detail::checked_add(m_allocated_intervals.rbegin()->first,
                                                 m_allocated_intervals.rbegin()->second);
    }

    // Preserve any existing backing storage before growing the portable fallback.
    void enlarge(size_t n_bytes) {
        std::lock_guard<std::mutex> lock{m_mutex};
        enlarge_locked(n_bytes);
    }

  private:
    // Caller holds the arena mutex so allocation and growth form one transaction.
    void enlarge_locked(size_t n_bytes) {
        if (m_fallback_memory && m_fallback_memory->size() >= n_bytes)
            return;
        if (m_device != metalrt::Device::current())
            throw std::invalid_argument("arena belongs to another selected device");
        if (m_fallback_memory && m_fallback_memory->size())
            metalrt::synchronize_buffer(m_fallback_memory->buffer_view());
        const size_t capacity =
            next_multiple(detail::checked_add(n_bytes, n_bytes / 2), size_t{256});
        auto replacement = std::make_shared<GPUMemory<uint8_t>>(capacity);
        if (m_fallback_memory && m_fallback_memory->size())
            std::memcpy(replacement->data(), m_fallback_memory->data(), m_fallback_memory->size());
        m_fallback_memory = std::move(replacement);
        m_append_end = m_high_water;
    }

    size_t block_size(size_t offset) {
        std::lock_guard<std::mutex> lock{m_mutex};
        auto it = m_allocated_intervals.find(offset);
        if (it == m_allocated_intervals.end())
            throw std::invalid_argument("unknown arena offset");
        return it->second;
    }

    metalrt::Device m_device;
    mutable std::mutex m_mutex;
    std::vector<std::shared_ptr<GPUMemory<uint8_t>>> m_pool;
    std::shared_ptr<GPUMemory<uint8_t>> m_fallback_memory;
    std::map<size_t, size_t> m_allocated_intervals;
    size_t m_high_water = 0;
    // Retired intervals may still be retained by recorded GPU commands.
    size_t m_append_end = 0;
};

inline std::unordered_map<metalStream_t, std::shared_ptr<GPUMemoryArena>>&
stream_gpu_memory_arenas() {
    static std::unordered_map<metalStream_t, std::shared_ptr<GPUMemoryArena>> arenas;
    return arenas;
}

inline std::unordered_map<int, std::shared_ptr<GPUMemoryArena>>& global_gpu_memory_arenas() {
    static std::unordered_map<int, std::shared_ptr<GPUMemoryArena>> arenas;
    return arenas;
}

inline std::mutex& gpu_memory_arenas_mutex() {
    static std::mutex mutex;
    return mutex;
}

inline GPUMemoryArena::Allocation allocate_workspace(metalStream_t stream, size_t bytes) {
    if (!bytes)
        return {};
    auto device = metalrt::stream_or_default(detail::resolve_stream(stream)).device();
    std::shared_ptr<GPUMemoryArena> arena;
    {
        std::lock_guard<std::mutex> lock(gpu_memory_arenas_mutex());
        auto& slot = stream ? stream_gpu_memory_arenas()[stream] : global_gpu_memory_arenas()[0];
        if (device != metalrt::Device::current())
            throw std::invalid_argument("workspace stream belongs to another selected device");
        if (!slot || slot->device() != device)
            slot = std::make_shared<GPUMemoryArena>();
        arena = slot;
    }
    return arena->acquire(stream, bytes);
}

inline void free_gpu_memory_arena(metalStream_t stream) {
    std::lock_guard<std::mutex> lock(gpu_memory_arenas_mutex());
    if (stream)
        stream_gpu_memory_arenas().erase(stream);
    else
        global_gpu_memory_arenas().erase(0);
}

inline void free_all_gpu_memory_arenas() {
    std::lock_guard<std::mutex> lock(gpu_memory_arenas_mutex());
    stream_gpu_memory_arenas().clear();
    global_gpu_memory_arenas().clear();
}

template <class First, class FirstSize>
std::tuple<First*> allocate_workspace_and_distribute(metalStream_t stream,
                                                     GPUMemoryArena::Allocation* alloc,
                                                     size_t offset, FirstSize count) {
    if (!alloc)
        throw std::invalid_argument("null Allocation destination");
    if constexpr (std::is_signed<FirstSize>::value)
        if (count < 0)
            throw std::invalid_argument("negative workspace size");
    auto end = detail::checked_add(
        offset, align_to_cacheline(detail::checked_mul(static_cast<size_t>(count), sizeof(First))));
    *alloc = allocate_workspace(stream, end);
    return std::tuple<First*>{alloc->data() ? reinterpret_cast<First*>(alloc->data() + offset)
                                            : nullptr};
}

template <class First, class... Types, class FirstSize, class... Sizes,
          std::enable_if_t<(sizeof...(Types) > 0) && sizeof...(Types) == sizeof...(Sizes), int> = 0>
std::tuple<First*, Types*...>
allocate_workspace_and_distribute(metalStream_t stream, GPUMemoryArena::Allocation* alloc,
                                  size_t offset, FirstSize count, Sizes... counts) {
    if constexpr (std::is_signed<FirstSize>::value)
        if (count < 0)
            throw std::invalid_argument("negative workspace size");
    auto end = detail::checked_add(
        offset, align_to_cacheline(detail::checked_mul(static_cast<size_t>(count), sizeof(First))));
    auto nested = allocate_workspace_and_distribute<Types...>(stream, alloc, end, counts...);
    auto first = alloc->data() ? reinterpret_cast<First*>(alloc->data() + offset) : nullptr;
    return std::tuple_cat(std::tuple<First*>{first}, nested);
}

template <class... Types, class... Sizes,
          std::enable_if_t<sizeof...(Types) == sizeof...(Sizes) && (sizeof...(Types) > 0), int> = 0>
std::tuple<Types*...> allocate_workspace_and_distribute(metalStream_t stream,
                                                        GPUMemoryArena::Allocation* alloc,
                                                        Sizes... counts) {
    return allocate_workspace_and_distribute<Types...>(stream, alloc, size_t{0}, counts...);
}
} // namespace tcnn
