#pragma once
#include <tiny-metal-nn/common.h>
#include <tiny-metal-nn/gpu_memory.h>
#include <pcg32/pcg32.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace tcnn {
template <class T> class GPUMatrixDynamic;
template <class T, MatrixLayout Layout> class GPUMatrix;

class GPUMatrixBase {
  public:
    virtual ~GPUMatrixBase() = default;
    virtual size_t n_bytes() const = 0;
    virtual void set_data_unsafe(void* data) = 0;

    static void allocate_shared_memory(GPUMemory<char>& memory,
                                       const std::vector<GPUMatrixBase*>& matrices) {
        const auto bytes = shared_size(matrices);
        memory.enlarge(bytes);
        size_t offset = 0;
        for (auto* matrix : matrices) {
            matrix->set_data_unsafe(memory.data() ? memory.data() + offset : nullptr);
            offset = detail::checked_add(offset, matrix->n_bytes());
        }
    }

    static GPUMemoryArena::Allocation
    allocate_shared_memory(metalStream_t stream, const std::vector<GPUMatrixBase*>& matrices) {
        auto alloc = allocate_workspace(stream, shared_size(matrices));
        size_t offset = 0;
        for (auto* matrix : matrices) {
            matrix->set_data_unsafe(alloc.data() ? alloc.data() + offset : nullptr);
            offset = detail::checked_add(offset, matrix->n_bytes());
        }
        return alloc;
    }
    template <class T>
    static void allocate_shared_memory(GPUMemory<char>& memory,
                                       std::vector<GPUMatrixDynamic<T>>& matrices);
    template <class T, MatrixLayout Layout>
    static void allocate_shared_memory(GPUMemory<char>& memory,
                                       std::vector<GPUMatrix<T, Layout>>& matrices);
    template <class T>
    static GPUMemoryArena::Allocation
    allocate_shared_memory(metalStream_t stream, std::vector<GPUMatrixDynamic<T>>& matrices);
    template <class T, MatrixLayout Layout>
    static GPUMemoryArena::Allocation
    allocate_shared_memory(metalStream_t stream, std::vector<GPUMatrix<T, Layout>>& matrices);

  private:
    static size_t shared_size(const std::vector<GPUMatrixBase*>& matrices) {
        size_t bytes = 0;
        for (auto* matrix : matrices) {
            if (!matrix)
                throw std::invalid_argument("null matrix in shared allocation");
            bytes = detail::checked_add(bytes, matrix->n_bytes());
        }
        return bytes;
    }
};

template <class T> class GPUMatrixDynamic : public GPUMatrixBase {
  public:
    using Type = T;
    using View = MatrixView<T>;
    using ConstView = MatrixView<const T>;

    GPUMatrixDynamic(uint32_t m, uint32_t n, MatrixLayout layout = CM)
        : m_rows{m}, m_cols{n}, m_layout{layout}, m_device{metalrt::Device::current()} {
        set_stride_contiguous();
        m_malloc_allocation = std::make_shared<GPUMemory<uint8_t>>(n_bytes(), true);
        m_data = reinterpret_cast<T*>(m_malloc_allocation->data());
        retain_resource();
    }

    GPUMatrixDynamic(uint32_t m, uint32_t n, metalStream_t stream, MatrixLayout layout = CM)
        : m_rows{m}, m_cols{n}, m_layout{layout},
          m_device{metalrt::stream_or_default(detail::resolve_stream(stream)).device()} {
        set_stride_contiguous();
        m_arena_allocation =
            std::make_shared<GPUMemoryArena::Allocation>(allocate_workspace(stream, n_bytes()));
        m_data = reinterpret_cast<T*>(m_arena_allocation->data());
        retain_resource();
    }

    explicit GPUMatrixDynamic(
        T* data, uint32_t m, uint32_t n, MatrixLayout layout = CM, uint32_t stride = 0,
        std::shared_ptr<GPUMemory<uint8_t>> malloc_allocation = nullptr,
        std::shared_ptr<GPUMemoryArena::Allocation> arena_allocation = nullptr)
        : m_data{data}, m_rows{m}, m_cols{n}, m_stride{stride}, m_layout{layout},
          m_malloc_allocation{std::move(malloc_allocation)},
          m_arena_allocation{std::move(arena_allocation)}, m_device{metalrt::Device::current()} {
        if (m_malloc_allocation)
            m_device = m_malloc_allocation->device();
        if (m_arena_allocation && m_arena_allocation->backing_memory())
            m_device = m_arena_allocation->backing_memory()->device();
        if (!m_stride)
            set_stride_contiguous();
        validate_shape();
        retain_resource();
    }

    GPUMatrixDynamic() : m_device{metalrt::Device::current()} {}

    GPUMatrixDynamic(const GPUMatrixDynamic&) = delete;
    GPUMatrixDynamic& operator=(const GPUMatrixDynamic&) = delete;

    GPUMatrixDynamic(GPUMatrixDynamic&& other) { swap(other); }

    GPUMatrixDynamic& operator=(GPUMatrixDynamic&& other) {
        swap(other);
        return *this;
    }

    ~GPUMatrixDynamic() override = default;

    void swap(GPUMatrixDynamic& other) noexcept {
        using std::swap;
        swap(m_data, other.m_data);
        swap(m_rows, other.m_rows);
        swap(m_cols, other.m_cols);
        swap(m_stride, other.m_stride);
        swap(m_layout, other.m_layout);
        swap(m_malloc_allocation, other.m_malloc_allocation);
        swap(m_arena_allocation, other.m_arena_allocation);
        swap(m_resource, other.m_resource);
        swap(m_device, other.m_device);
    }

    void set_data_unsafe(void* data) override {
        // "unsafe" preserves the source name: shape/capacity is caller supplied.
        // Unlike a naked pointer view, a registered runtime allocation is retained.
        auto next = static_cast<T*>(data);
        if (next && reinterpret_cast<uintptr_t>(next) % alignof(T))
            throw std::invalid_argument("matrix pointer is misaligned");
        metalrt::BufferView resource;
        if (next && n_storage_bytes())
            resource = metalrt::resolve_pointer(next, n_storage_bytes());
        m_data = next;
        m_resource = std::move(resource);
        if (m_resource)
            m_device = m_resource.buffer.device();
        // Allocation ownership describes how resize obtains storage, independently
        // of the pointer currently viewed. Upstream setters retain that provenance,
        // including when the new pointer refers to a different allocation.
    }

    void set_size_unsafe(uint32_t rows, uint32_t cols, uint32_t stride = 0) {
        const auto old_rows = m_rows, old_cols = m_cols, old_stride = m_stride;
        m_rows = rows;
        m_cols = cols;
        m_stride = stride ? stride : stride_contiguous();
        try {
            validate_shape();
            retain_resource();
        } catch (...) {
            m_rows = old_rows;
            m_cols = old_cols;
            m_stride = old_stride;
            throw;
        }
    }

    void set(T* data, uint32_t rows, uint32_t cols, uint32_t stride = 0) {
        GPUMatrixDynamic replacement{
            data, rows, cols, m_layout, stride, m_malloc_allocation, m_arena_allocation};
        swap(replacement);
    }

    void resize(uint32_t rows, uint32_t cols) {
        if (m_arena_allocation) {
            GPUMatrixDynamic replacement{rows, cols, m_arena_allocation->stream(), m_layout};
            swap(replacement);
        } else if (m_malloc_allocation || !m_data) {
            GPUMatrixDynamic replacement;
            replacement.m_rows = rows;
            replacement.m_cols = cols;
            replacement.m_layout = m_layout;
            replacement.m_device = m_device;
            replacement.set_stride_contiguous();
            replacement.m_malloc_allocation =
                std::make_shared<GPUMemory<uint8_t>>(replacement.n_bytes(), true);
            replacement.m_data = reinterpret_cast<T*>(replacement.m_malloc_allocation->data());
            replacement.retain_resource();
            swap(replacement);
        } else
            throw std::runtime_error(
                "GPUMatrix::resize requires owned memory; use set for a borrowed view");
    }

    uint32_t stride_contiguous() const { return m_layout == CM ? m_rows : m_cols; }

    bool is_contiguous() const { return m_stride == stride_contiguous(); }

    void set_stride_contiguous() {
        m_stride = stride_contiguous();
        validate_shape();
        retain_resource();
    }

    GPUMatrixDynamic slice(uint32_t offset_rows, uint32_t new_rows, uint32_t offset_cols,
                           uint32_t new_cols) const {
        detail::check_range(offset_rows, new_rows, m_rows);
        detail::check_range(offset_cols, new_cols, m_cols);
        const auto offset =
            m_layout == CM
                ? detail::checked_add(offset_rows, detail::checked_mul(offset_cols, m_stride))
                : detail::checked_add(offset_cols, detail::checked_mul(offset_rows, m_stride));
        // Empty slices carry no pointer; this avoids manufacturing a pointer beyond
        // the allocation when both end coordinates are selected simultaneously.
        T* begin = (m_data && new_rows && new_cols) ? m_data + offset : nullptr;
        return GPUMatrixDynamic{
            begin, new_rows, new_cols, m_layout, m_stride, m_malloc_allocation, m_arena_allocation};
    }

    GPUMatrixDynamic slice_rows(uint32_t offset, uint32_t count) const {
        return slice(offset, count, 0, m_cols);
    }

    GPUMatrixDynamic slice_cols(uint32_t offset, uint32_t count) const {
        return slice(0, m_rows, offset, count);
    }

    GPUMatrixDynamic alias() const { return slice(0, m_rows, 0, m_cols); }

    View view() const { return {m_data, row_stride(), column_stride()}; }

    ConstView const_view() const { return {m_data, row_stride(), column_stride()}; }

    uint32_t rows() const { return m_rows; }

    uint32_t fan_out() const { return m_rows; }

    uint32_t m() const { return m_rows; }

    uint32_t cols() const { return m_cols; }

    uint32_t fan_in() const { return m_cols; }

    uint32_t n() const { return m_cols; }

    uint32_t stride() const { return m_stride; }

    uint32_t row_stride() const { return m_layout == CM ? 1u : m_stride; }

    uint32_t column_stride() const { return m_layout == CM ? m_stride : 1u; }

    PitchedPtr<T> pitched_ptr() { return {m_data, m_stride}; }

    PitchedPtr<const T> pitched_ptr() const { return {m_data, m_stride}; }

    uint32_t n_elements() const {
        const auto count = detail::checked_mul(m_rows, m_cols);
        if (count > UINT32_MAX)
            throw std::overflow_error("matrix element count exceeds uint32_t");
        return static_cast<uint32_t>(count);
    }

    size_t n_bytes() const override { return detail::checked_mul(n_elements(), sizeof(T)); }

    size_t n_storage_bytes() const {
        if (!m_rows || !m_cols)
            return 0;
        const auto major = m_layout == CM ? m_cols : m_rows,
                   minor = m_layout == CM ? m_rows : m_cols;
        return detail::checked_mul(
            detail::checked_add(detail::checked_mul(major - 1, m_stride), minor), sizeof(T));
    }

    MatrixLayout layout() const { return m_layout; }

    MatrixLayout transposed_layout() const { return m_layout == CM ? RM : CM; }

    T* data() const { return m_data; }

    metalrt::Device device() const { return m_device; }

    const metalrt::BufferView& buffer_view() const { return m_resource; }

    void memset(int value) {
        check_data();
        if (!n_elements())
            return;
        if (!is_contiguous())
            throw std::invalid_argument("memset requires a contiguous matrix");
        detail::fill_device_sync(metalrt::resolve_pointer(m_data, n_bytes()),
                                 static_cast<uint8_t>(value));
    }

    void memset_async(metalStream_t stream, int value) {
        check_data();
        if (!n_elements())
            return;
        if (!is_contiguous())
            throw std::invalid_argument("memset_async requires a contiguous matrix");
        auto& s = metalrt::stream_or_default(detail::resolve_stream(stream));
        if (s.device() != m_device)
            throw std::invalid_argument("matrix and stream belong to different Devices");
        s.fill(metalrt::resolve_pointer(m_data, n_bytes()), static_cast<uint8_t>(value));
    }

    std::vector<T> to_cpu_vector() {
        check_data();
        std::vector<T> result(n_elements());
        if (result.empty())
            return result;
        if (!is_contiguous())
            throw std::invalid_argument(
                "to_cpu_vector requires a contiguous matrix, as in the reference");
        metalrt::synchronize_device(m_device);
        std::memcpy(result.data(), m_data, n_bytes());
        return result;
    }

    void initialize_uniform(pcg32& rng, float low, float high) {
        initialize([&](uint32_t, uint32_t) {
            return static_cast<T>(low + detail::rng_next_float(rng, 0) * (high - low));
        });
    }

    void initialize_xavier_uniform(pcg32& rng, float scale = 1) {
        const auto width = static_cast<double>(fan_in()) + fan_out();
        if (width == 0)
            return;
        scale *= static_cast<float>(std::sqrt(6.0 / width));
        initialize_uniform(rng, -scale, scale);
    }

    void initialize_fa_uniform_forward(pcg32& rng, float scale = 1) {
        if (!fan_in())
            return;
        scale *= std::sqrt(1.0f / static_cast<float>(fan_in()));
        initialize_uniform(rng, -scale, scale);
    }

    void initialize_fa_uniform_backward(pcg32& rng, float scale = 1) {
        if (!fan_out())
            return;
        scale *= std::sqrt(1.0f / static_cast<float>(fan_out()));
        initialize_uniform(rng, -scale, scale);
    }

    void initialize_siren_uniform(pcg32& rng, float scale = 1) {
        if (!fan_in())
            return;
        scale *= std::sqrt(6.0f / static_cast<float>(fan_in()));
        initialize_uniform(rng, -scale, scale);
    }

    void initialize_siren_uniform_first(pcg32& rng, float scale = 1) {
        if (!fan_in())
            return;
        scale *= 30.0f / static_cast<float>(fan_in());
        initialize_uniform(rng, -scale, scale);
    }

    void initialize_constant(float value) {
        initialize([&](uint32_t, uint32_t) { return static_cast<T>(value); });
    }

    void initialize_diagonal(float value = 1) {
        if (m_rows != m_cols)
            throw std::invalid_argument("diagonal initialization requires a square matrix");
        initialize([&](uint32_t i, uint32_t j) { return static_cast<T>(i == j ? value : 0); });
    }

    GPUMatrixDynamic transposed() const {
        return GPUMatrixDynamic{m_data,
                                m_cols,
                                m_rows,
                                transposed_layout(),
                                m_stride,
                                m_malloc_allocation,
                                m_arena_allocation};
    }

    GPUMatrix<T, RM> rm() const;
    GPUMatrix<T, CM> cm() const;

  private:
    void validate_shape() const {
        if (m_layout != CM && m_layout != RM)
            throw std::invalid_argument("invalid matrix layout");
        if (m_rows && m_cols && m_stride < stride_contiguous())
            throw std::invalid_argument("matrix stride is smaller than its minor dimension");
        (void)n_storage_bytes();
    }

    void retain_resource() {
        metalrt::BufferView next;
        if (m_data && reinterpret_cast<uintptr_t>(m_data) % alignof(T))
            throw std::invalid_argument("matrix pointer is misaligned");
        if (m_data && n_storage_bytes())
            next = metalrt::resolve_pointer(m_data, n_storage_bytes());
        m_resource = std::move(next);
        if (m_resource)
            m_device = m_resource.buffer.device();
    }

    void check_data() const {
        if (n_elements() && !m_data)
            throw std::runtime_error("matrix has no bound storage");
    }

    template <class F> void initialize(F&& value) {
        check_data();
        if (!n_elements())
            return;
        metalrt::synchronize_buffer(m_resource);
        if (m_layout == CM) {
            for (uint32_t j = 0; j < m_cols; ++j)
                for (uint32_t i = 0; i < m_rows; ++i)
                    view()(i, j) = value(i, j);
        } else {
            for (uint32_t i = 0; i < m_rows; ++i)
                for (uint32_t j = 0; j < m_cols; ++j)
                    view()(i, j) = value(i, j);
        }
    }

    T* m_data = nullptr;
    uint32_t m_rows = 0, m_cols = 0, m_stride = 0;
    MatrixLayout m_layout = CM;
    std::shared_ptr<GPUMemory<uint8_t>> m_malloc_allocation;
    std::shared_ptr<GPUMemoryArena::Allocation> m_arena_allocation;
    metalrt::BufferView m_resource;
    metalrt::Device m_device;
};

template <class T, MatrixLayout Layout = CM> class GPUMatrix : public GPUMatrixDynamic<T> {
  public:
    static constexpr MatrixLayout static_layout = Layout;
    static constexpr MatrixLayout static_transposed_layout = Layout == RM ? CM : RM;

    GPUMatrix(uint32_t rows, uint32_t cols) : GPUMatrixDynamic<T>{rows, cols, Layout} {}

    GPUMatrix(uint32_t rows, uint32_t cols, metalStream_t stream)
        : GPUMatrixDynamic<T>{rows, cols, stream, Layout} {}

    explicit GPUMatrix(T* data, uint32_t rows, uint32_t cols, uint32_t stride = 0,
                       std::shared_ptr<GPUMemory<uint8_t>> allocation = nullptr,
                       std::shared_ptr<GPUMemoryArena::Allocation> arena = nullptr)
        : GPUMatrixDynamic<T>{
              data, rows, cols, Layout, stride, std::move(allocation), std::move(arena)} {}

    GPUMatrix() : GPUMatrixDynamic<T>{nullptr, 0, 0, Layout} {}

    GPUMatrix(const GPUMatrix&) = delete;
    GPUMatrix& operator=(const GPUMatrix&) = delete;

    GPUMatrix(GPUMatrix&& other) noexcept : GPUMatrixDynamic<T>{std::move(other)} {}

    GPUMatrix& operator=(GPUMatrix&& other) noexcept {
        GPUMatrixDynamic<T>::operator=(std::move(other));
        return *this;
    }

    GPUMatrix(GPUMatrixDynamic<T>&& other) noexcept : GPUMatrixDynamic<T>{nullptr, 0, 0, Layout} {
        *this = std::move(other);
    }

    GPUMatrix& operator=(GPUMatrixDynamic<T>&& other) {
        if (other.layout() != Layout)
            throw std::invalid_argument("GPUMatrix static layout does not match dynamic layout");
        GPUMatrixDynamic<T>::operator=(std::move(other));
        return *this;
    }

    GPUMatrix slice(uint32_t row, uint32_t rows, uint32_t col, uint32_t cols) const {
        return GPUMatrixDynamic<T>::slice(row, rows, col, cols);
    }

    GPUMatrix slice_rows(uint32_t offset, uint32_t rows) const {
        return GPUMatrixDynamic<T>::slice_rows(offset, rows);
    }

    GPUMatrix slice_cols(uint32_t offset, uint32_t cols) const {
        return GPUMatrixDynamic<T>::slice_cols(offset, cols);
    }

    GPUMatrix alias() const { return GPUMatrixDynamic<T>::alias(); }

    GPUMatrix<T, static_transposed_layout> transposed() const {
        return GPUMatrixDynamic<T>::transposed();
    }
};

template <class T> GPUMatrix<T, RM> GPUMatrixDynamic<T>::rm() const {
    if (m_layout != RM)
        throw std::invalid_argument("matrix is not row major");
    return alias();
}

template <class T> GPUMatrix<T, CM> GPUMatrixDynamic<T>::cm() const {
    if (m_layout != CM)
        throw std::invalid_argument("matrix is not column major");
    return alias();
}

template <class T>
void GPUMatrixBase::allocate_shared_memory(GPUMemory<char>& memory,
                                           std::vector<GPUMatrixDynamic<T>>& matrices) {
    std::vector<GPUMatrixBase*> ptrs;
    for (auto& m : matrices)
        ptrs.push_back(&m);
    allocate_shared_memory(memory, ptrs);
}

template <class T, MatrixLayout Layout>
void GPUMatrixBase::allocate_shared_memory(GPUMemory<char>& memory,
                                           std::vector<GPUMatrix<T, Layout>>& matrices) {
    std::vector<GPUMatrixBase*> ptrs;
    for (auto& m : matrices)
        ptrs.push_back(&m);
    allocate_shared_memory(memory, ptrs);
}

template <class T>
GPUMemoryArena::Allocation
GPUMatrixBase::allocate_shared_memory(metalStream_t stream,
                                      std::vector<GPUMatrixDynamic<T>>& matrices) {
    std::vector<GPUMatrixBase*> ptrs;
    for (auto& m : matrices)
        ptrs.push_back(&m);
    return allocate_shared_memory(stream, ptrs);
}

template <class T, MatrixLayout Layout>
GPUMemoryArena::Allocation
GPUMatrixBase::allocate_shared_memory(metalStream_t stream,
                                      std::vector<GPUMatrix<T, Layout>>& matrices) {
    std::vector<GPUMatrixBase*> ptrs;
    for (auto& m : matrices)
        ptrs.push_back(&m);
    return allocate_shared_memory(stream, ptrs);
}
} // namespace tcnn
