// Compile handwritten MSL and bind its host arguments through the standalone runtime.
#include <tiny-metal-nn/rtc_kernel.h>
#include <metalrt/detail/cache_key.h>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>

namespace tcnn {
bool supports_jit_fusion(int device) {
    if (device < -1 || device > 0)
        throw std::invalid_argument("Unknown selected Metal device index");
#if !defined(TCNN_RTC) || !TCNN_RTC
    return false;
#else
    try {
        const auto capabilities = metalrt::Device::current().capabilities();
        return capabilities.backend == metalrt::BackendKind::Metal &&
               capabilities.runtime_compilation;
    } catch (const std::runtime_error&) {
        // A capability query must also work in a build with no native device.
        return false;
    }
#endif
}

std::string& rtc_cache_dir() {
    static std::string dir;
    return dir;
}

void rtc_set_cache_dir(const std::string& dir) {
    rtc_cache_dir() = dir;
}

std::string& rtc_include_dir() {
    static std::string dir;
    return dir;
}

void rtc_set_include_dir(const std::string& dir) {
    rtc_include_dir() = dir;
}

namespace {
// Read a quoted MSL include. Angle-bracket SDK includes remain for Apple's compiler.
std::string read_include(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot read Metal include: " + path.string());
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

// Expand user-provided quoted includes with cycle and size checks before compilation.
std::string expand_source(const std::string& source,
                          const std::map<std::string, std::string>& includes,
                          const std::filesystem::path& directory, std::set<std::string>& active,
                          uint32_t depth = 0) {
    if (depth > 32 || source.size() > 8u * 1024u * 1024u)
        throw std::invalid_argument("Metal include expansion exceeds its depth or size limit");
    std::istringstream input(source);
    std::ostringstream output;
    std::string line;
    while (std::getline(input, line)) {
        auto start = line.find_first_not_of(" \t");
        if (start != std::string::npos && line.compare(start, 8, "#include") == 0) {
            auto first = line.find_first_not_of(" \t", start + 8);
            if (first != std::string::npos && line[first] == '"') {
                auto last = line.find('"', first + 1);
                if (last == std::string::npos)
                    throw std::invalid_argument("Unclosed quoted Metal include");
                const auto name = line.substr(first + 1, last - first - 1);
                if (!active.insert(name).second)
                    throw std::invalid_argument("Cyclic Metal include: " + name);
                auto found = includes.find(name);
                std::string body;
                if (found != includes.end())
                    body = found->second;
                else {
                    if (directory.empty())
                        throw std::invalid_argument("No Metal include supplied for: " + name);
                    const std::filesystem::path relative(name);
                    if (relative.is_absolute())
                        throw std::invalid_argument(
                            "Metal includes must be relative to rtc_include_dir");
                    for (const auto& component : relative)
                        if (component == "..")
                            throw std::invalid_argument("Metal include escapes rtc_include_dir");
                    body = read_include(directory / relative);
                }
                output << expand_source(body, includes, directory, active, depth + 1) << '\n';
                active.erase(name);
                if (output.tellp() > 8 * 1024 * 1024)
                    throw std::invalid_argument("Expanded Metal source is too large");
                continue;
            }
        }
        output << line << '\n';
        if (output.tellp() > 8 * 1024 * 1024)
            throw std::invalid_argument("Expanded Metal source is too large");
    }
    return output.str();
}

// Save expanded source for diagnostics. Compiled pipeline persistence is owned
// by metalrt; the source file alone is never treated as a compiled cache hit.
void cache_source(const std::string& entry, const std::string& source) {
    const auto directory = rtc_cache_dir();
    if (directory.empty())
        return;
    std::filesystem::create_directories(directory);
    const auto filename = std::filesystem::path(directory) /
                          (entry + "-" + metalrt::detail::kernel_cache_digest(source) + ".metal");
    if (std::filesystem::is_regular_file(filename))
        return;
    static std::atomic<uint64_t> sequence{0};
    auto temporary = filename;
    temporary += "." + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                 "." + std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)) + ".tmp";
    try {
        std::ofstream output(temporary, std::ios::binary);
        if (!output || !(output << source) || !output.flush())
            throw std::runtime_error("Cannot write expanded MSL source cache");
        output.close();
        std::filesystem::rename(temporary, filename);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}
} // namespace

struct MetalRtcKernel::Impl {
    metalrt::Device device;
    metalrt::Kernel kernel;
    uint32_t max_threads = 0;
    uint32_t max_threadgroup_bytes = 0;
    uint32_t min_threadgroup_bytes = 0;
    uint32_t scratch_bytes_per_simdgroup = 0;
    uint32_t context_bytes_per_thread = 0;
    uint32_t minimum_scratch_bytes = 0;
    uint32_t optional_buffer_mask = UINT32_MAX;
    std::vector<metalrt::Argument> defaults;
    std::map<uint32_t, uint32_t> pointer_bases;
};

// Compile once against the selected device; actual native compile errors propagate to callers.
MetalRtcKernel::MetalRtcKernel(
    const std::string& name, const std::string& kernel_code,
    const std::vector<std::pair<std::string, const char*>>& extra_includes,
    metalrt::CpuKernel reference)
    : m_impl{std::make_unique<Impl>()} {
#if !defined(TCNN_RTC) || !TCNN_RTC
    throw std::runtime_error(
        "tiny-metal-nn was built without runtime compilation (TCNN_BUILD_WITH_RTC=OFF)");
#endif
    if (name.empty() ||
        !(std::isalpha(static_cast<unsigned char>(name.front())) || name.front() == '_'))
        throw std::invalid_argument("Metal kernel entry must be an identifier");
    for (unsigned char c : name)
        if (!(std::isalnum(c) || c == '_'))
            throw std::invalid_argument("Metal kernel entry must be an identifier");
    if (kernel_code.empty())
        throw std::invalid_argument("Metal kernel source is empty");
    std::map<std::string, std::string> includes;
    for (const auto& item : extra_includes) {
        if (item.first.empty() || !item.second || !includes.emplace(item.first, item.second).second)
            throw std::invalid_argument("Metal extra include is null, empty, or duplicated");
    }
    std::set<std::string> active;
    const auto expanded = expand_source(kernel_code, includes, rtc_include_dir(), active);
    m_impl->device = metalrt::Device::current();
    // Matrix collectives are a real Apple7+ hardware capability, not an
    // implication of runtime compilation. Other Metal devices retain the
    // generated scalar implementation, including in JIT-only builds.
    const auto source = std::string{"#define TCNN_JIT_HAS_SIMDGROUP_MATRIX "} +
                        (m_impl->device.capabilities().apple_gpu_family >= 7 ? "1\n" : "0\n") +
                        "#define TCNN_JIT_FLOAT_ATOMICS " +
                        (m_impl->device.capabilities().apple_gpu_family >= 9 ? "1\n" : "0\n") +
                        expanded;
    if (m_impl->device.capabilities().backend == metalrt::BackendKind::Reference && !reference)
        throw std::runtime_error("The Reference backend cannot compile MSL; supply an explicit "
                                 "reference callback or use a Metal device");
    try {
        cache_source(name, source);
    } catch (const std::exception& error) {
        log_warning("MSL source cache unavailable: {}", error.what());
    }
    metalrt::KernelCompileOptions options;
#if defined(TCNN_RTC_USE_FAST_MATH) && TCNN_RTC_USE_FAST_MATH
    options.fast_math = true;
#endif
    m_impl->kernel =
        m_impl->device.kernel_cached(source, name, rtc_cache_dir(), options, std::move(reference));
}

MetalRtcKernel::~MetalRtcKernel() = default;

void MetalRtcKernel::clear() {
    m_impl->kernel = {};
}

void MetalRtcKernel::set_default_arguments(std::vector<metalrt::Argument> arguments) {
    std::set<uint32_t> indices;
    for (const auto& argument : arguments) {
        if (argument.index >= 31 || argument.kind == metalrt::Argument::Kind::Threadgroup ||
            !indices.insert(argument.index).second)
            throw std::invalid_argument(
                "MetalRtcKernel: defaults require distinct buffer indices below 31");
        if (argument.index == m_impl->optional_buffer_mask)
            throw std::invalid_argument(
                "MetalRtcKernel: default conflicts with the reserved optional-buffer mask slot");
        if (argument.kind != metalrt::Argument::Kind::Buffer &&
            argument.kind != metalrt::Argument::Kind::Bytes &&
            argument.kind != metalrt::Argument::Kind::NullBuffer)
            throw std::invalid_argument("MetalRtcKernel: invalid default argument kind");
        if (argument.kind == metalrt::Argument::Kind::Buffer &&
            (!argument.resource || !argument.resource.length ||
             argument.resource.buffer.device() != m_impl->device))
            throw std::invalid_argument(
                "MetalRtcKernel: default buffer is empty or belongs to another device");
        if (argument.kind == metalrt::Argument::Kind::Bytes && argument.value.empty())
            throw std::invalid_argument("MetalRtcKernel: default constant is empty");
        if (argument.kind == metalrt::Argument::Kind::NullBuffer &&
            (argument.resource || argument.resource.offset || argument.resource.length ||
             !argument.value.empty() || argument.threadgroup_bytes))
            throw std::invalid_argument("MetalRtcKernel: malformed null default buffer");
    }
    m_impl->defaults = std::move(arguments);
}

void MetalRtcKernel::set_optional_buffer_mask_argument(uint32_t slot) {
    if (slot >= 31)
        throw std::invalid_argument("MetalRtcKernel: optional-buffer mask slot must be below 31");
    for (const auto& argument : m_impl->defaults)
        if (argument.index == slot)
            throw std::invalid_argument(
                "MetalRtcKernel: optional-buffer mask conflicts with a default argument");
    for (const auto& mapping : m_impl->pointer_bases)
        if (mapping.first == slot || mapping.second == slot)
            throw std::invalid_argument(
                "MetalRtcKernel: optional-buffer mask conflicts with a pointer-base binding");
    m_impl->optional_buffer_mask = slot;
}

void MetalRtcKernel::set_min_threadgroup_memory_bytes(uint32_t bytes) {
    if (bytes % 16)
        throw std::invalid_argument(
            "MetalRtcKernel: minimum threadgroup memory must be a multiple of 16 bytes");
    m_impl->min_threadgroup_bytes = bytes;
}

void MetalRtcKernel::set_threadgroup_memory_layout(uint32_t bytes_per_simdgroup,
                                                   uint32_t context_bytes_per_thread,
                                                   uint32_t minimum_scratch_bytes) {
    if (bytes_per_simdgroup % 16 || minimum_scratch_bytes % 4)
        throw std::invalid_argument("MetalRtcKernel: generated workspace alignment is invalid");
    m_impl->scratch_bytes_per_simdgroup = bytes_per_simdgroup;
    m_impl->context_bytes_per_thread = context_bytes_per_thread;
    m_impl->minimum_scratch_bytes = minimum_scratch_bytes;
}

uint32_t MetalRtcKernel::required_threadgroup_memory_bytes(uint32_t threads) const {
    const auto scratch =
        std::max(size_t{m_impl->minimum_scratch_bytes},
                 metalrt::checked_mul(threads / 32u, m_impl->scratch_bytes_per_simdgroup));
    const auto total = metalrt::checked_add(
        scratch, metalrt::checked_mul(threads, m_impl->context_bytes_per_thread));
    const auto aligned = metalrt::checked_add(total, size_t{15}) & ~size_t{15};
    if (aligned > UINT32_MAX)
        throw std::invalid_argument(
            "MetalRtcKernel: generated threadgroup memory exceeds uint32_t");
    return std::max(m_impl->min_threadgroup_bytes, static_cast<uint32_t>(aligned));
}

void MetalRtcKernel::set_pointer_base_argument(uint32_t source_slot, uint32_t base_slot) {
    if (source_slot >= 31 || base_slot >= 31 || source_slot == base_slot)
        throw std::invalid_argument(
            "MetalRtcKernel: pointer-base bindings require distinct buffer slots below 31");
    if (source_slot == m_impl->optional_buffer_mask || base_slot == m_impl->optional_buffer_mask)
        throw std::invalid_argument(
            "MetalRtcKernel: pointer-base binding conflicts with the optional-buffer mask");
    if (m_impl->pointer_bases.count(source_slot))
        throw std::invalid_argument("MetalRtcKernel: pointer-base bindings cannot be chained");
    for (const auto& mapping : m_impl->pointer_bases)
        if (mapping.second == base_slot)
            throw std::invalid_argument("MetalRtcKernel: pointer-base bindings cannot be chained");
    m_impl->pointer_bases[base_slot] = source_slot;
}

uint32_t MetalRtcKernel::max_threads_per_threadgroup() const {
    const auto compiled = static_cast<uint32_t>(m_impl->kernel.max_threads_per_threadgroup());
    uint32_t limit = m_impl->max_threads ? std::min(compiled, m_impl->max_threads) : compiled;
    auto memory = m_impl->device.capabilities().max_threadgroup_memory_bytes;
    if (m_impl->max_threadgroup_bytes)
        memory = memory ? std::min(size_t{memory}, size_t{m_impl->max_threadgroup_bytes})
                        : m_impl->max_threadgroup_bytes;
    if (memory && (m_impl->scratch_bytes_per_simdgroup || m_impl->context_bytes_per_thread ||
                   m_impl->minimum_scratch_bytes)) {
        const auto fixed = static_threadgroup_memory_bytes();
        if (fixed > memory || m_impl->min_threadgroup_bytes > memory - fixed)
            return 0;
        const auto budget = (uint64_t(memory) - fixed) & ~uint64_t{15};
        while (limit) {
            // Compare in wide arithmetic before converting the selected launch
            // size to uint32_t. A very large cooperative shape can still use
            // a sub-SIMD scalar group instead of failing while sizing 1024 lanes.
            const auto context = uint64_t(limit) * m_impl->context_bytes_per_thread;
            const auto scratch =
                std::max(uint64_t(m_impl->minimum_scratch_bytes),
                         uint64_t(limit / 32u) * m_impl->scratch_bytes_per_simdgroup);
            if (context <= budget && scratch <= budget - context)
                break;
            --limit;
        }
    }
    return limit;
}

uint32_t MetalRtcKernel::static_threadgroup_memory_bytes() const {
    return static_cast<uint32_t>(m_impl->kernel.static_threadgroup_memory_bytes());
}
uint32_t MetalRtcKernel::thread_execution_width() const {
    return static_cast<uint32_t>(m_impl->kernel.thread_execution_width());
}

// Limits are checked for every launch; they do not pretend to tune hardware registers.
void MetalRtcKernel::set(MetalFunctionAttribute attr, int value) {
    if (value < 0)
        throw std::invalid_argument("Metal kernel launch limit must be nonnegative");
    switch (attr) {
    case MetalFunctionAttribute::MaxThreadsPerThreadgroup:
        m_impl->max_threads = static_cast<uint32_t>(value);
        return;
    case MetalFunctionAttribute::MaxThreadgroupMemoryBytes:
        m_impl->max_threadgroup_bytes = static_cast<uint32_t>(value);
        return;
    }
    throw std::invalid_argument("Unknown Metal kernel attribute");
}

// Convert threadgroup counts to the runtime's total-thread grid with checked arithmetic.
void MetalRtcKernel::launch_impl(MetalSize3 blocks, MetalSize3 threads, uint32_t shmem_size,
                                 metalStream_t stream, std::vector<metalrt::Argument> bindings) {
    if (!m_impl->kernel)
        throw std::runtime_error("MetalRtcKernel has been cleared");
    if (!threads.x || !threads.y || !threads.z)
        throw std::invalid_argument("Metal threadgroup dimensions must be positive");
    auto* queue = detail::resolve_stream(stream);
    if (queue->device() != m_impl->device)
        throw std::invalid_argument("Metal kernel and stream belong to different devices");
    const size_t group_size =
        metalrt::checked_mul(metalrt::checked_mul(threads.x, threads.y), threads.z);
    if (m_impl->max_threads && group_size > m_impl->max_threads)
        throw std::invalid_argument("Metal kernel launch exceeds MaxThreadsPerThreadgroup");
    if (group_size > UINT32_MAX)
        throw std::invalid_argument("Metal kernel threadgroup exceeds uint32_t");
    shmem_size =
        std::max(shmem_size, required_threadgroup_memory_bytes(static_cast<uint32_t>(group_size)));
    if (m_impl->max_threadgroup_bytes && shmem_size > m_impl->max_threadgroup_bytes)
        throw std::invalid_argument("Metal kernel launch exceeds MaxThreadgroupMemoryBytes");
    for (const auto& fallback : m_impl->defaults) {
        const bool supplied =
            std::any_of(bindings.begin(), bindings.end(), [&](const metalrt::Argument& argument) {
                return argument.kind != metalrt::Argument::Kind::Threadgroup &&
                       argument.index == fallback.index;
            });
        if (!supplied)
            bindings.push_back(fallback);
    }
    if (shmem_size)
        bindings.push_back(metalrt::Argument::threadgroup(0, shmem_size));
    metalrt::ComputeCommand command;
    command.kernel = m_impl->kernel;
    command.grid = {metalrt::checked_mul(blocks.x, threads.x),
                    metalrt::checked_mul(blocks.y, threads.y),
                    metalrt::checked_mul(blocks.z, threads.z)};
    command.group = {threads.x, threads.y, threads.z};
    command.arguments = std::move(bindings);
    command.optional_buffer_mask = m_impl->optional_buffer_mask;
    for (const auto& mapping : m_impl->pointer_bases)
        command.pointer_base_arguments.emplace_back(mapping.second, mapping.first);
    metalrt::Command submission;
    submission.kind = metalrt::Command::Kind::Compute;
    submission.compute = std::move(command);
    queue->enqueue(std::move(submission));
}
} // namespace tcnn
