#pragma once
#include <tiny-metal-nn/common_host.h>
#include <tiny-metal-nn/metal_kernel_attributes.h>
#include <tiny-metal-nn/metal_runtime.h>
#include <memory>
#include <array>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tcnn {
// True only for a selected native device with a runtime MSL compiler. Reference
// callbacks never masquerade as a compiler or an automatic-fusion backend.
bool supports_jit_fusion(int device);

inline bool supports_jit_fusion() {
    return supports_jit_fusion(metal_device());
}

// Configure an optional expanded-MSL and compiled Metal binary-archive cache,
// and a quoted-include search directory. Configure before concurrent compilation.
std::string& rtc_cache_dir();
void rtc_set_cache_dir(const std::string& dir);
std::string& rtc_include_dir();
void rtc_set_include_dir(const std::string& dir);

namespace detail {
template <class T> struct RtcMatrixView : std::false_type {};
template <class T, class STRIDE_T>
struct RtcMatrixView<MatrixView<T, STRIDE_T>> : std::true_type {};
} // namespace detail

// Compile a concrete MSL kernel entry and launch it on the selected runtime device.
// Arguments map in order to [[buffer(i)]]: pointers/views are retained buffers,
// trivially-copyable values are constants, and shmem_size uses [[threadgroup(0)]].
class MetalRtcKernel {
  public:
    // Extra includes are named MSL headers, copied during construction. A Reference
    // callback is explicit test-backend support; a native device always executes MSL.
    MetalRtcKernel(const std::string& name, const std::string& kernel_code,
                   const std::vector<std::pair<std::string, const char*>>& extra_includes = {},
                   metalrt::CpuKernel reference = {});
    ~MetalRtcKernel();
    MetalRtcKernel(const MetalRtcKernel&) = delete;
    MetalRtcKernel& operator=(const MetalRtcKernel&) = delete;
    // Release this wrapper's pipeline reference; recorded commands retain their own.
    void clear();
    // Apply Metal launch limits without inventing vendor register/occupancy settings.
    void set(MetalFunctionAttribute attr, int value);
    // Accept logical null launch arguments. Native active buffer slots also need
    // the mask ABI below. Handwritten kernels reject host nulls by default.
    void set_allow_null_arguments(bool allow) { m_allow_null_arguments = allow; }
    // Generated shaders reserve a uint mask slot and reconstruct null pointers
    // from its bits. This opt-in lets Metal bind validation-safe fallback
    // resources without changing the meaning of optional arguments.
    void set_optional_buffer_mask_argument(uint32_t slot);
    // Defaults fill missing slots only; explicitly supplied arguments win. Each
    // launch copies the bindings, so queued work retains its own resources.
    void set_default_arguments(std::vector<metalrt::Argument> arguments);
    // Fill a hidden pointer slot from another buffer binding, rounded down to
    // its containing four-byte word. This preserves packed-half atomics for
    // odd subviews without copying unrelated allocation contents. An explicitly
    // supplied or default base takes precedence over automatic derivation.
    void set_pointer_base_argument(uint32_t source_slot, uint32_t base_slot);
    // Generated functions may declare a scratch slot even when a specialization
    // needs no working bytes. Reserve a nonzero aligned slot in that case.
    void set_min_threadgroup_memory_bytes(uint32_t bytes);
    // Generated cooperative functions use 32-lane SIMD groups. Their scratch
    // is separate from the optional per-sample context at the end of the slot.
    // Partial groups use scalar device functions and require no SIMD scratch.
    void set_threadgroup_memory_layout(uint32_t bytes_per_simdgroup,
                                       uint32_t context_bytes_per_thread = 0,
                                       uint32_t minimum_scratch_bytes = 0);
    uint32_t required_threadgroup_memory_bytes(uint32_t threads) const;
    uint32_t max_threads_per_threadgroup() const;
    uint32_t static_threadgroup_memory_bytes() const;
    uint32_t thread_execution_width() const;

    // blocks counts threadgroups; threads is the size of each threadgroup.
    template <typename... Types>
    void launch(MetalSize3 blocks, MetalSize3 threads, uint32_t shmem_size, metalStream_t stream,
                Types&&... args) {
        if (!blocks.x || !blocks.y || !blocks.z)
            return;
        std::vector<metalrt::Argument> bindings;
        bindings.reserve(2 * sizeof...(args) + (shmem_size ? 1 : 0));
        uint32_t index = 0;
        (append_argument(bindings, index, std::forward<Types>(args)), ...);
        launch_impl(blocks, threads, shmem_size, stream, std::move(bindings));
    }

    // Preserve the convenient one-dimensional launch form.
    template <typename... Types>
    void launch(uint32_t blocks, uint32_t threads, uint32_t shmem_size, metalStream_t stream,
                Types&&... args) {
        launch(MetalSize3{blocks, 1, 1}, MetalSize3{threads, 1, 1}, shmem_size, stream,
               std::forward<Types>(args)...);
    }

  private:
    // Translate host values into runtime-owned bindings, never unmanaged GPU addresses.
    template <typename U> metalrt::Argument make_argument(uint32_t index, U&& value) const {
        using T = std::decay_t<U>;
        if constexpr (std::is_same<T, metalrt::BufferView>::value) {
            if (!value && m_allow_null_arguments)
                return metalrt::Argument::null_buffer(index);
            return metalrt::Argument::buffer(index, value, metalrt::Access::ReadWrite);
        } else if constexpr (std::is_same<T, metalrt::Buffer>::value) {
            if (!value && m_allow_null_arguments)
                return metalrt::Argument::null_buffer(index);
            return metalrt::Argument::buffer(index, value.view(), metalrt::Access::ReadWrite);
        } else if constexpr (std::is_same<T, std::nullptr_t>::value) {
            if (!m_allow_null_arguments)
                throw std::invalid_argument("MetalRtcKernel: null buffer argument");
            return metalrt::Argument::null_buffer(index);
        } else if constexpr (std::is_pointer<T>::value) {
            if (!value) {
                if (m_allow_null_arguments)
                    return metalrt::Argument::null_buffer(index);
                throw std::invalid_argument("MetalRtcKernel: null buffer argument");
            }
            auto probe = metalrt::resolve_pointer(value, 0);
            auto view = probe.buffer.view(probe.offset);
            constexpr auto access = std::is_const<std::remove_pointer_t<T>>::value
                                        ? metalrt::Access::Read
                                        : metalrt::Access::ReadWrite;
            return metalrt::Argument::buffer(index, std::move(view), access);
        } else {
            static_assert(std::is_trivially_copyable<T>::value,
                          "MSL constants must be trivially copyable; use BufferView for GPU data");
            return metalrt::Argument::scalar(index, value);
        }
    }

    template <typename U>
    void append_argument(std::vector<metalrt::Argument>& arguments, uint32_t& index,
                         U&& value) const {
        using T = std::decay_t<U>;
        if constexpr (std::is_same<T, metalrt::Argument>::value) {
            // Internal optional resources may follow a reserved generated ABI
            // slot. Explicit indices do not renumber ordinary public arguments.
            if (value.kind == metalrt::Argument::Kind::Threadgroup || value.index >= 31)
                throw std::invalid_argument(
                    "MetalRtcKernel: explicit arguments require a buffer slot below 31");
            arguments.emplace_back(std::forward<U>(value));
        } else if constexpr (detail::RtcMatrixView<T>::value) {
            // A host address inside a MatrixView cannot be sent to an MSL
            // constant buffer. Bind the retained resource and its two strides
            // separately, preserving the public launch(..., matrix.view(), ...).
            arguments.emplace_back(make_argument(index++, value.data));
            const std::array<decltype(value.stride_i), 2> strides{{value.stride_i, value.stride_j}};
            arguments.emplace_back(metalrt::Argument::scalar(index++, strides));
        } else {
            arguments.emplace_back(make_argument(index++, std::forward<U>(value)));
        }
    }

    // Validate the launch and enqueue one real compute command through metalrt.
    void launch_impl(MetalSize3 blocks, MetalSize3 threads, uint32_t shmem_size,
                     metalStream_t stream, std::vector<metalrt::Argument> bindings);
    struct Impl;
    std::unique_ptr<Impl> m_impl;
    bool m_allow_null_arguments = false;
};
} // namespace tcnn
