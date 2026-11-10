#pragma once
#include <tiny-metal-nn/common.h>
#include <metalrt/runtime.h>
#include <fmt/format.h>
#include <atomic>
#include <cctype>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef CHECK_THROW
#define CHECK_THROW(x)                                                                             \
    do {                                                                                           \
        if (!(x))                                                                                  \
            throw std::runtime_error(std::string("check failed: ") + #x);                          \
    } while (0)
#endif
#ifndef METAL_CHECK_THROW
#define METAL_CHECK_THROW(x)                                                                       \
    do {                                                                                           \
        const metalError_t result_ = (x);                                                          \
        if (result_ != metalSuccess)                                                               \
            throw std::runtime_error(std::string(#x " failed: ") + metalGetErrorString(result_));  \
    } while (0)
#endif
#ifndef METAL_CHECK_PRINT
#define METAL_CHECK_PRINT(x)                                                                       \
    do {                                                                                           \
        const metalError_t result_ = (x);                                                          \
        if (result_ != metalSuccess)                                                               \
            tcnn::log_error("{}", metalGetErrorString(result_));                                   \
    } while (0)
#endif

namespace tcnn {

using namespace fmt::literals;

enum class LogSeverity { Info, Debug, Warning, Error, Success };

namespace detail {

inline std::mutex& log_mutex() {
    static std::mutex m;
    return m;
}

inline std::atomic<bool>& verbose_flag();

inline std::function<void(LogSeverity, const std::string&)>& mutable_log_callback() {
    static std::function<void(LogSeverity, const std::string&)> fn = [](LogSeverity severity,
                                                                        const std::string& s) {
        if (severity == LogSeverity::Error || severity == LogSeverity::Warning || verbose_flag())
            std::cerr << s << '\n';
    };
    return fn;
}

inline std::atomic<bool>& verbose_flag() {
    static std::atomic<bool> flag{false};
    return flag;
}

} // namespace detail

inline const std::function<void(LogSeverity, const std::string&)>& log_callback() {
    return detail::mutable_log_callback();
}

inline void set_log_callback(const std::function<void(LogSeverity, const std::string&)>& callback) {
    std::lock_guard<std::mutex> lock(detail::log_mutex());
    detail::mutable_log_callback() = callback;
}

template <class... Ts> void log(LogSeverity severity, const std::string& message, Ts&&... args) {
    std::function<void(LogSeverity, const std::string&)> callback;
    {
        std::lock_guard<std::mutex> lock(detail::log_mutex());
        callback = detail::mutable_log_callback();
    }
    callback(severity, fmt::format(fmt::runtime(message), std::forward<Ts>(args)...));
}

template <class... Ts> void log_info(const std::string& s, Ts&&... a) {
    log(LogSeverity::Info, s, std::forward<Ts>(a)...);
}

template <class... Ts> void log_debug(const std::string& s, Ts&&... a) {
    log(LogSeverity::Debug, s, std::forward<Ts>(a)...);
}

template <class... Ts> void log_warning(const std::string& s, Ts&&... a) {
    log(LogSeverity::Warning, s, std::forward<Ts>(a)...);
}

template <class... Ts> void log_error(const std::string& s, Ts&&... a) {
    log(LogSeverity::Error, s, std::forward<Ts>(a)...);
}

template <class... Ts> void log_success(const std::string& s, Ts&&... a) {
    log(LogSeverity::Success, s, std::forward<Ts>(a)...);
}

inline bool verbose() {
    return detail::verbose_flag();
}

inline void set_verbose(bool v) {
    detail::verbose_flag() = v;
}

inline std::string to_lower(std::string s) {
    for (auto& c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

inline std::string to_upper(std::string s) {
    for (auto& c : s)
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

inline bool equals_case_insensitive(const std::string& a, const std::string& b) {
    return to_lower(a) == to_lower(b);
}

struct CaseInsensitiveHash {
    size_t operator()(const std::string& s) const { return std::hash<std::string>{}(to_lower(s)); }
};

struct CaseInsensitiveEqual {
    bool operator()(const std::string& a, const std::string& b) const {
        return equals_case_insensitive(a, b);
    }
};
template <class T>
using ci_hashmap = std::unordered_map<std::string, T, CaseInsensitiveHash, CaseInsensitiveEqual>;

// Hash mixing helpers retain the constants and public names of the uploaded source.
template <typename T> T xorshift(T n, int i) {
    return n ^ (n >> i);
}

inline uint32_t distribute(uint32_t n) {
    return 3423571495u * xorshift(0x55555555u * xorshift(n, 16), 16);
}

inline uint64_t distribute(uint64_t n) {
    return 17316035218449499591ull * xorshift(0x5555555555555555ull * xorshift(n, 32), 32);
}

template <typename T, typename S>
constexpr typename std::enable_if<std::is_unsigned<T>::value, T>::type rotl(const T n, const S i) {
    const T m = std::numeric_limits<T>::digits - 1, c = i & m;
    return (n << c) | (n >> ((T(0) - c) & m));
}

template <typename T> size_t hash_combine(size_t seed, const T& v) {
    return rotl(seed, std::numeric_limits<size_t>::digits / 3) ^ distribute(std::hash<T>{}(v));
}

// Report a positive power of two and optionally its exponent.
inline bool is_pot(uint32_t num, uint32_t* log2 = nullptr) {
    if (log2)
        *log2 = 0;
    if (!num)
        return false;
    while (num % 2 == 0) {
        num /= 2;
        if (log2)
            ++*log2;
    }
    return num == 1;
}

inline uint32_t powi(uint32_t base, uint32_t exponent) {
    uint32_t result = 1;
    for (uint32_t i = 0; i < exponent; ++i)
        result *= base;
    return result;
}

inline std::string to_string(Activation a) {
    static const char* names[] = {"ReLU",    "LeakyReLU",  "SiLU",     "Exponential", "Sine",
                                  "Sigmoid", "Squareplus", "Softplus", "Tanh",        "None"};
    const auto i = static_cast<size_t>(a);
    if (i >= 10)
        throw std::invalid_argument("invalid activation");
    return names[i];
}

inline Activation string_to_activation(const std::string& s) {
    for (size_t i = 0; i < 10; ++i)
        if (equals_case_insensitive(s, to_string(static_cast<Activation>(i))))
            return static_cast<Activation>(i);
    throw std::invalid_argument("unknown activation: " + s);
}

template <class E, size_t N>
inline std::string enum_to_string(E value, const char* const (&names)[N]) {
    auto i = static_cast<size_t>(value);
    if (i >= N)
        throw std::invalid_argument("invalid enumeration value");
    return names[i];
}

template <class E, size_t N>
inline E string_to_enum(const std::string& value, const char* const (&names)[N]) {
    for (size_t i = 0; i < N; ++i)
        if (equals_case_insensitive(value, names[i]))
            return static_cast<E>(i);
    throw std::invalid_argument("unknown enumeration value: " + value);
}

inline std::string to_string(GridType value) {
    const char* names[] = {"Hash", "Dense", "Tiled"};
    return enum_to_string(value, names);
}

inline GridType string_to_grid_type(const std::string& value) {
    if (equals_case_insensitive(value, "Tile"))
        return GridType::Tiled;
    const char* names[] = {"Hash", "Dense", "Tiled"};
    return string_to_enum<GridType>(value, names);
}

inline std::string to_string(HashType value) {
    const char* names[] = {"Prime", "CoherentPrime", "ReversedPrime", "Rng", "BaseConvert"};
    return enum_to_string(value, names);
}

inline HashType string_to_hash_type(const std::string& value) {
    const char* names[] = {"Prime", "CoherentPrime", "ReversedPrime", "Rng", "BaseConvert"};
    return string_to_enum<HashType>(value, names);
}

inline std::string to_string(InterpolationType value) {
    const char* names[] = {"Nearest", "Linear", "Smoothstep"};
    return enum_to_string(value, names);
}

inline InterpolationType string_to_interpolation_type(const std::string& value) {
    const char* names[] = {"Nearest", "Linear", "Smoothstep"};
    return string_to_enum<InterpolationType>(value, names);
}

inline std::string to_string(ReductionType value) {
    const char* names[] = {"Concatenation", "Sum", "Product"};
    return enum_to_string(value, names);
}

inline ReductionType string_to_reduction_type(const std::string& value) {
    const char* names[] = {"Concatenation", "Sum", "Product"};
    return string_to_enum<ReductionType>(value, names);
}

// Load the MSL preamble from a shader asset; host headers contain no device bodies.
std::string generate_device_code_preamble();

inline std::string bytes_to_string(size_t bytes) {
    const char* suffixes[] = {"B", "KB", "MB", "GB", "TB", "PB", "EB"};
    double value = static_cast<double>(bytes);
    size_t i = 0;
    while (value >= 1024 && i < 6) {
        value /= 1024;
        ++i;
    }
    std::ostringstream s;
    s.precision(3);
    s << value << ' ' << suffixes[i];
    return s.str();
}

inline std::vector<std::string> split(const std::string& s, const std::string& delim) {
    std::vector<std::string> out;
    size_t pos = 0;
    for (;;) {
        auto end = s.find_first_of(delim, pos);
        out.push_back(s.substr(pos, end == std::string::npos ? end : end - pos));
        if (end == std::string::npos)
            break;
        pos = end + 1;
    }
    return out;
}

template <class T> std::string join(const T& values, const std::string& delim) {
    std::ostringstream out;
    bool first = true;
    for (const auto& v : values) {
        if (!first)
            out << delim;
        first = false;
        out << v;
    }
    return out.str();
}

template <class... Ts> std::string dfmt(uint32_t indent, const std::string& format, Ts&&... args) {
    // Preserve the supplied host helper: trim leading/trailing empty or tab-only
    // lines, remove common tab indentation, then indent and apply full fmt syntax.
    uint32_t input_indent = std::numeric_limits<uint32_t>::max();
    uint32_t n_empty_leading = 0, n_empty_trailing = 0;
    bool leading = true;

    std::vector<std::string> lines = split(format, "\n");
    for (const auto& line : lines) {
        bool empty = true;
        uint32_t line_indent = 0;
        for (uint32_t i = 0; i < line.length(); ++i) {
            if (empty && line[i] == '\t') {
                line_indent = i + 1;
            } else {
                empty = false;
                break;
            }
        }

        if (empty) {
            if (leading)
                ++n_empty_leading;
            ++n_empty_trailing;
            continue;
        }

        n_empty_trailing = 0;
        leading = false;
        input_indent = std::min(input_indent, line_indent);
    }

    if (input_indent == std::numeric_limits<uint32_t>::max())
        return "";

    lines.erase(lines.end() - n_empty_trailing, lines.end());
    lines.erase(lines.begin(), lines.begin() + n_empty_leading);

    for (auto& line : lines) {
        if (line.length() >= input_indent) {
            line = line.substr(input_indent);
            line = line.insert(0, indent, '\t');
        }
    }

    return fmt::format(fmt::runtime(join(lines, "\n")), std::forward<Ts>(args)...);
}

inline std::string to_snake_case(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        if (std::isupper(c) && !out.empty())
            out += '_';
        out += static_cast<char>(std::tolower(c));
    }
    return out;
}

template <class T> std::string type_to_string() {
    throw std::runtime_error("type_to_string is unsupported for this type");
}

template <> inline std::string type_to_string<float>() {
    return "float";
}

template <> inline std::string type_to_string<half>() {
    return "half";
}

template <> inline std::string type_to_string<uint32_t>() {
    return "uint32_t";
}

template <> inline std::string type_to_string<uint16_t>() {
    return "uint16_t";
}

template <> inline std::string type_to_string<uint8_t>() {
    return "uint8_t";
}

template <> inline std::string type_to_string<char>() {
    return "char";
}

template <> inline std::string type_to_string<bool>() {
    return "bool";
}

template <> inline std::string type_to_string<double>() {
    return "double";
}

template <> inline std::string type_to_string<int>() {
    return "int";
}

// Format vector values using the same generated-type spelling as the reference.
template <typename T, uint32_t N, size_t A> std::string to_string(const tvec<T, N, A>& v) {
    return "tvec<" + type_to_string<T>() + ", " + std::to_string(N) + ", " + std::to_string(A) +
           ">(" + join(v, ", ") + ")";
}

// Device index zero denotes the device selected through tcnn::metal. The runtime
// exposes one selected device, rather than inventing a second enumeration layer.
inline int metal_device() {
    int device;
    METAL_CHECK_THROW(metalGetDevice(&device));
    return device;
}

inline void set_metal_device(int device) {
    METAL_CHECK_THROW(metalSetDevice(device));
}

inline int metal_device_count() {
    int count;
    METAL_CHECK_THROW(metalGetDeviceCount(&count));
    return count;
}

inline std::string metal_device_name(int device) {
    set_metal_device(device);
    return metalrt::Device::current().capabilities().name;
}

inline std::string metal_device_name() {
    return metal_device_name(metal_device());
}

// Query actual backend capabilities and allocation/compilation counters.
inline metalrt::Capabilities metal_device_capabilities(int device) {
    set_metal_device(device);
    return metalrt::Device::current().capabilities();
}

inline metalrt::Capabilities metal_device_capabilities() {
    return metal_device_capabilities(metal_device());
}

inline metalrt::DeviceStatistics metal_device_statistics(int device) {
    set_metal_device(device);
    return metalrt::Device::current().statistics();
}

inline metalrt::DeviceStatistics metal_device_statistics() {
    return metal_device_statistics(metal_device());
}

// The retained-buffer allocator has no reserved virtual-address capability.
inline bool metal_supports_virtual_memory(int device) {
    set_metal_device(device);
    return false;
}

inline bool metal_supports_virtual_memory() {
    return metal_supports_virtual_memory(metal_device());
}

inline size_t metal_memory_granularity(int device) {
    set_metal_device(device);
    return 256;
}

inline size_t metal_memory_granularity() {
    return metal_memory_granularity(metal_device());
}

struct MemoryInfo {
    size_t total = 0, free = 0, used = 0;
};

// Query the actual Metal device's threadgroup-memory ceiling. The explicit
// Reference backend has no native hardware ceiling to report.
inline size_t metal_max_shmem(int device) {
    set_metal_device(device);
    const auto capabilities = metalrt::Device::current().capabilities();
    if (capabilities.backend != metalrt::BackendKind::Metal)
        throw std::runtime_error("The Reference backend has no native threadgroup-memory limit");
    return capabilities.max_threadgroup_memory_bytes;
}

inline size_t metal_max_shmem() {
    return metal_max_shmem(metal_device());
}

// Return the selected native compiler's __METAL_VERSION__ value using a small
// Metal kernel. The explicit Reference backend returns zero (no MSL compiler).
int metal_runtime_version();
std::string metal_runtime_version_string();

inline MemoryInfo metal_memory_info() {
    throw std::runtime_error("metalrt does not expose device memory availability");
}

// General host cleanup helper with the uploaded constructor/disarm surface.
// Cleanup failures are logged instead of escaping a destructor during unwinding.
class ScopeGuard {
  public:
    ScopeGuard() = default;

    ScopeGuard(const std::function<void()>& callback) : callback_{callback} {}

    ScopeGuard(std::function<void()>&& callback) : callback_{std::move(callback)} {}

    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;

    ScopeGuard(ScopeGuard&& other) : callback_{std::move(other.callback_)} { other.callback_ = {}; }

    ScopeGuard& operator=(ScopeGuard&& other) {
        callback_.swap(other.callback_);
        return *this;
    }

    ~ScopeGuard() noexcept {
        try {
            if (callback_)
                callback_();
        } catch (...) {
            try {
                log_error("ScopeGuard cleanup callback failed");
            } catch (...) {
            }
        }
    }

    void disarm() { callback_ = {}; }

  private:
    std::function<void()> callback_;
};

template <class T> class Lazy {
  public:
    template <class F> T& get(F&& create) {
        if (!value_)
            value_ = create();
        return value_;
    }

  private:
    T value_{};
};

// Dispatch a concrete Metal kernel over a linear range using retained bindings.
// Any dynamic threadgroup allocation is bound at MSL threadgroup index zero.
inline void linear_kernel(const metalrt::Kernel& kernel, uint32_t shmem_size, metalStream_t stream,
                          uint32_t n_elements, std::vector<metalrt::Argument> args = {}) {
    if (!n_elements)
        return;
    if (shmem_size)
        args.push_back(metalrt::Argument::threadgroup(0, shmem_size));
    metalrt::launch_linear(detail::resolve_stream(stream), kernel, n_elements, std::move(args),
                           N_THREADS_LINEAR);
}

// Dispatch a compiled Metal kernel over a linear range.
inline void parallel_for_gpu(metalStream_t stream, size_t n, const metalrt::Kernel& kernel,
                             std::vector<metalrt::Argument> args = {}) {
    if (n > UINT32_MAX)
        throw std::overflow_error("linear dispatch exceeds 32-bit indexing");
    linear_kernel(kernel, 0, stream, static_cast<uint32_t>(n), std::move(args));
}

inline void parallel_for_gpu(size_t n, const metalrt::Kernel& kernel,
                             std::vector<metalrt::Argument> args = {}) {
    parallel_for_gpu(nullptr, n, kernel, std::move(args));
}

} // namespace tcnn
