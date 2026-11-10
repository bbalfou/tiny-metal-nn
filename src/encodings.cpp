#include <tiny-metal-nn/encodings/identity.h>
#include <tiny-metal-nn/encodings/empty.h>
#include <tiny-metal-nn/encodings/frequency.h>
#include <tiny-metal-nn/encodings/triangle_wave.h>
#include <tiny-metal-nn/encodings/oneblob.h>
#include <tiny-metal-nn/encodings/spherical_harmonics.h>
#include <tiny-metal-nn/encodings/grid.h>
#include <tiny-metal-nn/encodings/internal/permuto.h>
#include <tiny-metal-nn/encodings/composite.h>
#include <tiny-metal-nn/encodings/multi_level_lod.h>
#include <tiny-metal-nn/detail/shaders.h>
#include <tiny-metal-nn/detail/kernels/encodings/traits.h>
#include <tiny-metal-nn/detail/kernels/encodings/hash_grid_inference.h>
#include <tiny-metal-nn/metal.h>
#include <tiny-metal-nn/random.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <utility>
#include "../kernels/encodings/math.metal"

namespace tcnn {
namespace encoding_detail {
namespace kernel_traits = detail::kernels::encodings;
using tcnn_encoding_math::Args;
using tcnn_encoding_math::Mat;
using tcnn_encoding_math::UtilityArgs;
static_assert(sizeof(Mat) == 16 && sizeof(Args) == 372 && sizeof(UtilityArgs) == 80,
              "Encoding constant ABI must match Metal.");
static_assert(offsetof(Args, hash_type) == 140 && offsetof(Args, scale) == 164 &&
                  offsetof(Args, scales) == 180 && offsetof(Args, shifts) == 276 &&
                  offsetof(UtilityArgs, value) == 76,
              "Encoding constant field offsets must match Metal.");

/// Converts matrix strides into the explicit ABI consumed by every encoding kernel.
template <typename T> Mat describe(const GPUMatrixDynamic<T>& matrix) {
    const auto v = matrix.view();
    if (matrix.m() && matrix.n()) {
        const uint64_t span =
            uint64_t(matrix.m() - 1) * v.stride_i + uint64_t(matrix.n() - 1) * v.stride_j + 1;
        if (span > UINT32_MAX || uint64_t(matrix.m()) * matrix.n() > UINT32_MAX)
            throw std::overflow_error("Encoding matrix exceeds 32-bit indexing.");
    }
    return {matrix.m(), matrix.n(), v.stride_i, v.stride_j};
}

inline size_t span(Mat m) {
    return m.rows && m.cols ? size_t(m.rows - 1) * m.rs + size_t(m.cols - 1) * m.cs + 1 : 0;
}

inline std::string lower(std::string value) {
    for (char& c : value)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return value;
}

inline void check(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}

/// Reference callbacks execute the same dispatch boundaries, buffers, and strides as Metal.
template <typename T> metalrt::CpuKernel cpu_kernel(const std::string& op) {
    return [op](const metalrt::ComputeCommand& c) {
        using namespace tcnn_encoding_math;
        if (op == "forward") {
            auto a = c.scalar<Args>(4);
            const auto* x = c.buffer<const float>(0);
            const auto* p = c.buffer<const T>(1);
            const auto* l = c.buffer<const float>(2);
            auto* y = c.buffer<T>(3);
            const uint32_t count = (a.kind < 6 ? a.y.rows : a.features) * a.x.cols;
            for (uint32_t i = 0; i < count; ++i)
                forward(a, x, p, l, y, i);
        } else if (op == "dx") {
            auto a = c.scalar<Args>(6);
            for (uint32_t i = 0; i < a.dims * a.x.cols; ++i)
                backward_input(a, c.buffer<const float>(0), c.buffer<const T>(1),
                               c.buffer<const float>(2), c.buffer<const T>(3), c.buffer<float>(4),
                               c.buffer<const float>(5), i);
        } else if (op == "ddy") {
            auto a = c.scalar<Args>(5);
            const uint32_t count = (a.kind < 6 ? a.ddy.rows : a.features) * a.x.cols;
            for (uint32_t i = 0; i < count; ++i)
                backward_output(a, c.buffer<const float>(0), c.buffer<const T>(1),
                                c.buffer<const float>(2), c.buffer<const float>(3), c.buffer<T>(4),
                                i);
        } else if (op == "dp") {
            auto a = c.scalar<Args>(5);
            for (uint32_t i = 0; i < a.features; ++i)
                backward_params(a, c.buffer<const float>(0), c.buffer<const float>(1),
                                c.buffer<const T>(2), c.buffer<float>(3), c.buffer<const float>(4),
                                i);
        } else if (op == "utility") {
            auto a = c.scalar<UtilityArgs>(4);
            for (uint32_t i = 0; i < a.dst.rows * a.dst.cols; ++i)
                utility(a, c.buffer<const T>(0), c.buffer<T>(1), c.buffer<const T>(2),
                        c.buffer<const float>(3), i);
        } else if (op == "product_second") {
            auto a = c.scalar<UtilityArgs>(4);
            for (uint32_t i = 0; i < a.dst.rows * a.dst.cols; ++i)
                product_second(a, c.buffer<const T>(0), c.buffer<const T>(1), c.buffer<const T>(2),
                               c.buffer<T>(3), i);
        } else if (op == "cast_gradient") {
            const auto n = c.scalar<uint32_t>(2), acc = c.scalar<uint32_t>(3);
            auto* dst = c.buffer<T>(1);
            const auto* src = c.buffer<const float>(0);
            for (uint32_t i = 0; i < n; ++i)
                dst[i] = T(src[i] + (acc ? float(dst[i]) : 0));
        } else
            throw std::logic_error("Unknown encoding kernel.");
    };
}

struct KernelCache {
    metalrt::Device device;
    metalrt::Buffer dummy;
    std::map<std::string, metalrt::Kernel> kernels;
};

std::mutex cache_mutex;
std::vector<KernelCache> caches;

/// Binds a real allocation even for unused shader slots, so native resource declarations remain
/// valid.
metalrt::Buffer dummy_buffer(metalStream_t stream) {
    auto device = metalrt::stream_or_default(detail::resolve_stream(stream)).device();
    std::lock_guard<std::mutex> lock{cache_mutex};
    auto it = std::find_if(caches.begin(), caches.end(),
                           [&](const auto& c) { return c.device == device; });
    if (it == caches.end()) {
        auto dummy = device.allocate(16);
        std::memset(dummy.contents(), 0, 16);
        caches.push_back({device, dummy, {}});
        it = std::prev(caches.end());
    }
    return it->dummy;
}

template <typename T>
metalrt::Argument raw(metalStream_t stream, uint32_t index, const T* data, size_t count,
                      metalrt::Access access) {
    if (count && !data)
        throw std::invalid_argument("Encoding requires a non-null buffer for a nonempty argument.");
    return metalrt::Argument::buffer(
        index,
        data && count ? metalrt::resolve_pointer(data, metalrt::checked_mul(count, sizeof(T)))
                      : dummy_buffer(stream).view(),
        access);
}

template <typename T>
metalrt::Argument matrix(metalStream_t stream, uint32_t index, const GPUMatrixDynamic<T>& value,
                         metalrt::Access access) {
    return raw(stream, index, value.data(), span(describe(value)), access);
}

// Canonical first-order dispatch preserves the public classes, contexts,
// buffer slots, parameter ownership and asynchronous graph lifetimes.
template <typename T>
bool launch_first_order(metalStream_t stream, const std::string& op,
                        std::vector<metalrt::Argument>& arguments) {
    if (op != "forward" && op != "dx" && op != "dp")
        return false;
    auto& queue = metalrt::stream_or_default(detail::resolve_stream(stream));
    const auto device = queue.device();
    const auto capabilities = device.capabilities();
    if (capabilities.backend == metalrt::BackendKind::Reference)
        return false;

    const uint32_t argument_slot = op == "forward" ? 4u : op == "dx" ? 6u : 5u;
    const auto constants = std::find_if(arguments.begin(), arguments.end(), [&](const auto& value) {
        return value.index == argument_slot && value.kind == metalrt::Argument::Kind::Bytes;
    });
    if (constants == arguments.end() || constants->value.size() != sizeof(Args))
        throw std::logic_error("Encoding dispatch received an incompatible constant ABI.");
    Args a{};
    std::memcpy(&a, constants->value.data(), sizeof(a));
    if (a.mode)
        return false; // Required higher-order operations have separate kernels.
    if (a.kind > 7u || (a.kind >= 6u && (a.dims < 1u || a.dims > 24u ||
                                         !(a.features == 1u || a.features == 2u ||
                                           a.features == 4u || a.features == 8u))))
        throw std::logic_error("Invalid canonical encoding shape.");

    // Apple analytic derivatives use eight cooperating lanes; other Metal
    // devices execute the same formulas with one owner per coordinate. Avoid
    // enlarging the public indexing bounds when a large valid dispatch would
    // overflow after the eight-lane expansion.
    bool simd = capabilities.apple_gpu_family >= 6;
    const bool float_atomics = capabilities.apple_gpu_family >= 9;
    constexpr uint32_t threads = 128;
    const bool soa = a.kind < 6u && a.y.cs < a.y.rs;
    // One coordinate per lane makes SoA upstream reads contiguous across samples.
    const bool gradient_soa = op == "dx" && a.kind >= 2u && a.kind <= 4u && a.dy.cs < a.dy.rs;
    metalrt::Size3 grid{a.x.cols, 1, 1};
    if (a.kind < 6u && op == "forward") {
        grid = soa ? metalrt::Size3{a.x.cols, a.y.rows, 1}
                   : metalrt::Size3{metalrt::checked_mul(a.x.cols, a.y.rows), 1, 1};
        // SoA tiles preserve contiguous sample stores while amortizing coordinate loads.
        if (simd && soa && a.kind >= 2u && a.kind <= 4u)
            grid.y = a.y.rows / 4u + (a.y.rows % 4u != 0);
        if (simd && a.kind == 5u)
            grid = {a.x.cols, 1, 1};
        // Aligned AoS tiles use packed stores; OneBlob also shares boundaries.
        if (simd && a.kind >= 2 && a.kind <= 4 && (a.kind != 4 || a.degree >= 4) && !soa &&
            a.y.rs == 1 && a.y.cs % 4 == 0 && a.y.rows % 4 == 0 && a.width % 4 == 0)
            grid.x /= 4;
        // Eligible OneBlob rows expand the four-feature tile to 32 features.
        if (simd && a.kind == 4 && a.degree >= 32 && !soa && a.y.rs == 1 && a.y.cs % 4 == 0 &&
            a.y.rows % 32 == 0 && a.width % 32 == 0)
            grid.x /= 8;
    } else if (a.kind < 6u && op == "dx") {
        const uint32_t lanes = (a.kind <= 1u || gradient_soa) ? 1u
                               : a.kind == 4u                 ? std::min(a.degree, 32u)
                                                              : 8u;
        const size_t coordinates = a.kind == 5u ? a.x.cols : metalrt::checked_mul(a.dims, a.x.cols);
        if (coordinates > UINT32_MAX / std::max(lanes, 1u))
            simd = false;
        grid.x = simd ? metalrt::checked_mul(coordinates, lanes)
                      : metalrt::checked_mul(a.dims, a.x.cols);
    }
    if (grid.x > UINT32_MAX || grid.y > UINT32_MAX)
        throw std::overflow_error("Encoding dispatch exceeds 32-bit thread coordinates.");

    // Sample-owned lattice scatter requires one ordered GPU clear. Empty
    // batches still clear parameter scratch before the unchanged cast/mode step.
    if (op == "dp") {
        const auto scratch =
            std::find_if(arguments.begin(), arguments.end(), [](const auto& value) {
                return value.index == 3u && value.kind == metalrt::Argument::Kind::Buffer;
            });
        if (scratch == arguments.end())
            throw std::logic_error("Encoding gradient scratch is missing.");
        queue.fill(scratch->resource, 0);
        scratch->access = metalrt::Access::ReadWrite;
    }
    if (!grid.x || !grid.y)
        return true;

    const uint32_t dimensions = a.kind >= 6u ? a.dims : 2u;
    const uint32_t features = a.kind >= 6u ? a.features : 2u;
    const uint32_t output_rows = a.kind < 6u && op == "forward" ? a.y.rows : 1u;
    const uint32_t degree = a.kind < 6u ? a.degree : 0u;
    using First = kernel_traits::FirstOrderTraits;
    const char* entry =
        op == "forward" ? (sizeof(T) == 4 ? First::forward_f32 : First::forward_f16)
        : op == "dx"
            ? (sizeof(T) == 4 ? First::input_gradient_f32 : First::input_gradient_f16)
            : (sizeof(T) == 4 ? First::parameter_gradient_f32 : First::parameter_gradient_f16);
    const std::string name = entry;
    const std::string key = name + "/k" + std::to_string(a.kind) + "/d" +
                            std::to_string(dimensions) + "/f" + std::to_string(features) + "/s" +
                            std::to_string(soa) + "/simd" + std::to_string(simd) + "/atom" +
                            std::to_string(float_atomics) + "/o" + std::to_string(output_rows) +
                            "/b" + std::to_string(degree) + "/gs" + std::to_string(gradient_soa);
    metalrt::Kernel kernel;
    {
        std::lock_guard<std::mutex> lock{cache_mutex};
        auto cache = std::find_if(caches.begin(), caches.end(),
                                  [&](const auto& value) { return value.device == device; });
        if (cache == caches.end()) {
            auto dummy = device.allocate(16);
            std::memset(dummy.contents(), 0, 16);
            caches.push_back({device, dummy, {}});
            cache = std::prev(caches.end());
        }
        auto found = cache->kernels.find(key);
        if (found == cache->kernels.end()) {
            std::ostringstream source;
            source << "#define TCNN_ENCODING_GRADIENT_SOA " << uint32_t(gradient_soa) << "\n"
                   << "#define TCNN_ENCODING_KIND " << a.kind << "\n"
                   << "#define TCNN_ENCODING_DIMS " << dimensions << "\n"
                   << "#define TCNN_ENCODING_FEATURES " << features << "\n"
                   << "#define TCNN_ENCODING_FLOAT_ATOMICS " << uint32_t(float_atomics) << "\n"
                   << "#define TCNN_ENCODING_DEGREE " << degree << "\n"
                   << detail::shader_source(First::source);
            metalrt::KernelCompileOptions options;
            // Metal exposes only constants used by this entry point. Lattice
            // and parameter-gradient entries do not use traversal constants.
            if (a.kind < 6u && op == "forward") {
                options.constants.push_back(metalrt::FunctionConstant::scalar(0, soa));
                options.constants.push_back(metalrt::FunctionConstant::scalar(1, output_rows));
            }
            if (a.kind < 6u && (op == "forward" || op == "dx") && !gradient_soa)
                options.constants.push_back(metalrt::FunctionConstant::scalar(2, simd));
            found = cache->kernels.emplace(key, device.kernel(source.str(), name, options)).first;
        }
        kernel = found->second;
    }
    queue.dispatch(kernel, grid, {threads, 1, 1}, std::move(arguments));
    return true;
}

/// Caches a compiled shader per runtime device and precision; replay never recompiles.
template <typename T>
void launch(
    metalStream_t stream, const std::string& op, size_t count, std::vector<metalrt::Argument> args,
    kernel_traits::KernelDescriptor descriptor = kernel_traits::UtilityTraits::descriptor()) {
    if (!count)
        return;
    if (count > UINT32_MAX)
        throw std::overflow_error("Encoding dispatch is too large.");
    if (launch_first_order<T>(stream, op, args))
        return;
    const auto device = metalrt::stream_or_default(detail::resolve_stream(stream)).device();
    const bool reference = device.capabilities().backend == metalrt::BackendKind::Reference;
    const std::string operation = reference     ? op
                                  : op == "dx"  ? "input_second_gradient"
                                  : op == "dp"  ? "parameter_second_gradient"
                                  : op == "ddy" ? "output_gradient"
                                                : op;
    const std::string name = (reference ? std::string{"reference_"} : std::string{}) +
                             descriptor.prefix + "_" + operation +
                             (std::is_same<T, float>::value ? "_f32" : "_f16");
    metalrt::Kernel kernel;
    {
        std::lock_guard<std::mutex> lock{cache_mutex};
        auto it = std::find_if(caches.begin(), caches.end(),
                               [&](const auto& c) { return c.device == device; });
        if (it == caches.end()) {
            auto d = device.allocate(16);
            std::memset(d.contents(), 0, 16);
            caches.push_back({device, d, {}});
            it = std::prev(caches.end());
        }
        auto k = it->kernels.find(name);
        if (k == it->kernels.end())
            k = it->kernels
                    .emplace(name, device.kernel(detail::shader_source(descriptor.source), name,
                                                 cpu_kernel<T>(op)))
                    .first;
        kernel = k->second;
    }
    metalrt::launch_linear(detail::resolve_stream(stream), kernel, count, std::move(args));
}

// Inference-only grouping. Descriptor storage is fresh and immutable, so
// concurrent streams and retained graphs cannot observe another call's metadata.
template <typename T>
void launch_hash_grid_inference(metalStream_t stream, const GPUMatrixDynamic<float>& input,
                                GPUMatrixDynamic<T>& output, const T* params,
                                size_t parameter_count, const float* limits,
                                const std::vector<kernel_traits::HashGridInferenceLevel>& levels) {
    if (!input.n())
        return;
    using Grouped = kernel_traits::HashGridInferenceTraits;
    auto& queue = metalrt::stream_or_default(detail::resolve_stream(stream));
    const auto device = queue.device();
    const std::string name = sizeof(T) == 4 ? Grouped::entry_f32 : Grouped::entry_f16;
    metalrt::Kernel kernel;
    {
        std::lock_guard<std::mutex> lock{cache_mutex};
        auto cache = std::find_if(caches.begin(), caches.end(),
                                  [&](const auto& value) { return value.device == device; });
        if (cache == caches.end()) {
            auto dummy = device.allocate(16);
            std::memset(dummy.contents(), 0, 16);
            caches.push_back({device, dummy, {}});
            cache = std::prev(caches.end());
        }
        auto found = cache->kernels.find(name);
        if (found == cache->kernels.end())
            found = cache->kernels
                        .emplace(name, device.kernel(detail::shader_source(Grouped::source), name))
                        .first;
        kernel = found->second;
    }
    auto descriptors = device.allocate(levels.size() * sizeof(levels.front()));
    std::memcpy(descriptors.contents(), levels.data(), descriptors.size());
    queue.dispatch(kernel, {input.n(), levels.size(), 1}, {Grouped::threads, 1, 1},
                   {matrix(stream, 0, input, metalrt::Access::Read),
                    raw(stream, 1, params, parameter_count, metalrt::Access::Read),
                    raw(stream, 2, limits, limits ? input.n() : 0, metalrt::Access::Read),
                    matrix(stream, 3, output, metalrt::Access::Write),
                    metalrt::Argument::buffer(4, descriptors.view(), metalrt::Access::Read)});
}

// The grouped family clears and converts its complete gradient
// once. Descriptors are immutable, privately allocated GPU buffers so graph
// capture and concurrent streams retain their own matrix/level metadata.
template <typename T>
void launch_hash_grid_parameter_gradient(
    metalStream_t stream, const GPUMatrixDynamic<float>& input, const GPUMatrixDynamic<T>& upstream,
    const float* limits, T* gradients, uint32_t parameter_count, GradientMode mode,
    const std::vector<kernel_traits::HashGridGradientLevel>& levels) {
    // 2D uses eight corner/feature lanes over four samples. 3D uses two
    // Z-face lanes per sample, each reusing coordinates for both features.
    constexpr uint32_t threads = 128;
    using Grouped = kernel_traits::HashGridParameterGradientTraits;
    auto& queue = metalrt::stream_or_default(detail::resolve_stream(stream));
    const auto device = queue.device();
    const std::string name = sizeof(T) == 4 ? Grouped::entry_f32 : Grouped::entry_f16;
    // All batch sizes in the large-table family use adaptive scatter. Compile
    // its coherent branch separately: the log15 specialization has no classification flags.
    const uint32_t dimensions = levels.front().args.dims;
    const uint32_t samples_per_lane = dimensions == 3 ? 1u : 4u;
    const uint32_t tiled_lanes = dimensions == 3 ? 2u : 8u;
    const bool adaptive = parameter_count == 11184640 || parameter_count == 14229504;
    const std::string variant =
        "/d" + std::to_string(dimensions) + "/p" + std::to_string(parameter_count);
    const std::string key = name + (adaptive ? "/adaptive" : "/grouped") + variant;
    const auto source = [&] {
        return std::string{"#define HG_SAMPLES "} + std::to_string(samples_per_lane) +
               "\n#define TCNN_HASH_GRID_ADAPTIVE " + (adaptive ? "1" : "0") +
               "\n#define TCNN_ENCODING_DIMS " + std::to_string(dimensions) +
               "\n#define HG_PARAMETER_COUNT " + std::to_string(parameter_count) +
               "\n#define HG_TILED_LANES " + std::to_string(tiled_lanes) +
               "\n#define TCNN_ENCODING_FLOAT_ATOMICS " +
               (device.capabilities().apple_gpu_family >= 9 ? "1\n" : "0\n") +
               detail::shader_source(Grouped::source);
    };
    metalrt::Kernel kernel;
    {
        std::lock_guard<std::mutex> lock{cache_mutex};
        auto cache = std::find_if(caches.begin(), caches.end(),
                                  [&](const auto& value) { return value.device == device; });
        if (cache == caches.end()) {
            auto dummy = device.allocate(16);
            std::memset(dummy.contents(), 0, 16);
            caches.push_back({device, dummy, {}});
            cache = std::prev(caches.end());
        }
        auto found = cache->kernels.find(key);
        if (found == cache->kernels.end()) {
            found = cache->kernels.emplace(key, device.kernel(source(), name)).first;
        }
        kernel = found->second;
    }
    const bool indirect = adaptive && detail::indirect_dispatch_enabled(stream);
    metalrt::Kernel classify, tiled;
    if (adaptive) {
        std::lock_guard<std::mutex> lock{cache_mutex};
        auto cache = std::find_if(caches.begin(), caches.end(),
                                  [&](const auto& c) { return c.device == device; });
        auto it = cache->kernels.find("encoding_hash_grid_classify" + variant);
        if (it == cache->kernels.end())
            it = cache->kernels
                     .emplace("encoding_hash_grid_classify" + variant,
                              device.kernel(source(), "encoding_hash_grid_classify"))
                     .first;
        classify = it->second;
        auto fast = cache->kernels.find("encoding_hash_grid_scatter_tiled_f16" + variant);
        if (fast == cache->kernels.end())
            fast = cache->kernels
                       .emplace("encoding_hash_grid_scatter_tiled_f16" + variant,
                                device.kernel(source(), "encoding_hash_grid_scatter_tiled_f16"))
                       .first;
        tiled = fast->second;
    }
    auto counts = adaptive ? device.allocate(24) : metalrt::Buffer{};
    auto descriptors = device.allocate(levels.size() * sizeof(levels.front()));
    // This buffer has not been submitted before; copying its host metadata
    // needs no device synchronization and cannot mutate a retained replay.
    std::memcpy(descriptors.contents(), levels.data(), descriptors.size());
    GPUMatrixDynamic<float> temporary{parameter_count + (adaptive ? 16 : 0), 1, stream};
    auto scratch = matrix(stream, Grouped::gradient_scratch, temporary, metalrt::Access::ReadWrite);
    queue.fill(scratch.resource, 0);
    const size_t padded_batch =
        (size_t(input.n()) + Grouped::threads - 1) / Grouped::threads * Grouped::threads;
    if (adaptive)
        queue.dispatch(
            classify, {32, 1, 1}, {32, 1, 1},
            {matrix(stream, 0, input, metalrt::Access::Read),
             raw(stream, 1, limits, limits ? input.n() : 0, metalrt::Access::Read),
             matrix(stream, 2, upstream, metalrt::Access::Read),
             matrix(stream, 3, temporary, metalrt::Access::ReadWrite),
             metalrt::Argument::buffer(4, descriptors.view(), metalrt::Access::Read),
             metalrt::Argument::buffer(5, counts.view(), metalrt::Access::Write),
             metalrt::Argument::scalar(6, std::array<uint32_t, 2>{threads, samples_per_lane})});
    if (adaptive) {
        std::vector<metalrt::Argument> arguments{
            matrix(stream, 0, input, metalrt::Access::Read),
            raw(stream, 1, limits, limits ? input.n() : 0, metalrt::Access::Read),
            matrix(stream, 2, upstream, metalrt::Access::Read),
            matrix(stream, 3, temporary, metalrt::Access::ReadWrite),
            metalrt::Argument::buffer(4, descriptors.view(), metalrt::Access::Read)};
        const size_t tiled_threads = padded_batch / samples_per_lane * tiled_lanes;
        if (dimensions == 3) {
            // Keep each level's working set cache-local while sharing one clear
            // and final conversion across the complete parameter gradient.
            for (size_t level = 0; level < levels.size(); ++level) {
                arguments.back() = metalrt::Argument::buffer(
                    4, descriptors.view(level * sizeof(levels.front()), sizeof(levels.front())),
                    metalrt::Access::Read);
                queue.dispatch(tiled, {tiled_threads, 1, 1}, {threads, 1, 1}, arguments);
            }
        } else if (indirect)
            queue.dispatch_indirect(tiled, counts.view(0, 12),
                                    {(tiled_threads + threads - 1) / threads, levels.size(), 1},
                                    {threads, 1, 1}, std::move(arguments));
        else
            queue.dispatch(tiled, {tiled_threads, levels.size(), 1}, {threads, 1, 1},
                           std::move(arguments));
    }
    if (indirect) {
        queue.dispatch_indirect(
            kernel, counts.view(12, 12), {(padded_batch + threads - 1) / threads, levels.size(), 1},
            {threads, 1, 1},
            {matrix(stream, Grouped::input, input, metalrt::Access::Read),
             raw(stream, Grouped::limits, limits, limits ? input.n() : 0, metalrt::Access::Read),
             matrix(stream, Grouped::output_gradient, upstream, metalrt::Access::Read),
             std::move(scratch),
             metalrt::Argument::buffer(Grouped::levels, descriptors.view(),
                                       metalrt::Access::Read)});
    } else {
        queue.dispatch(
            kernel, {padded_batch, levels.size(), 1}, {threads, 1, 1},
            {matrix(stream, Grouped::input, input, metalrt::Access::Read),
             raw(stream, Grouped::limits, limits, limits ? input.n() : 0, metalrt::Access::Read),
             matrix(stream, Grouped::output_gradient, upstream, metalrt::Access::Read),
             std::move(scratch),
             metalrt::Argument::buffer(Grouped::levels, descriptors.view(),
                                       metalrt::Access::Read)});
    }
    launch<T>(stream, "cast_gradient", parameter_count,
              {matrix(stream, 0, temporary, metalrt::Access::Read),
               raw(stream, 1, gradients, parameter_count,
                   mode == GradientMode::Accumulate ? metalrt::Access::ReadWrite
                                                    : metalrt::Access::Write),
               metalrt::Argument::scalar(2, parameter_count),
               metalrt::Argument::scalar(3, uint32_t(mode == GradientMode::Accumulate))});
}

/// Copies or transforms matrix values without a CPU synchronization boundary.
template <typename T>
void utility_impl(metalStream_t stream, kernel_traits::KernelDescriptor descriptor,
                  const GPUMatrixDynamic<T>* src, GPUMatrixDynamic<T>& dst, uint32_t mode,
                  float value = 0, const GPUMatrixDynamic<T>* aux = nullptr,
                  const float* limits = nullptr, uint32_t width = 0, uint32_t index = 0,
                  uint32_t nested = 0, uint32_t levels = 0, uint32_t features = 0) {
    UtilityArgs a{};
    a.src = src ? describe(*src) : Mat{};
    a.dst = describe(dst);
    a.aux = aux ? describe(*aux) : Mat{};
    a.mode = mode;
    a.width = width;
    a.index = index;
    a.nested = nested;
    a.levels = levels;
    a.features = features;
    a.value = value;
    launch<T>(
        stream, "utility", dst.n_elements(),
        {src ? matrix(stream, 0, *src, metalrt::Access::Read)
             : raw<T>(stream, 0, nullptr, 0, metalrt::Access::Read),
         matrix(stream, 1, dst, mode == 2 ? metalrt::Access::ReadWrite : metalrt::Access::Write),
         aux ? matrix(stream, 2, *aux, metalrt::Access::Read)
             : raw<T>(stream, 2, nullptr, 0, metalrt::Access::Read),
         raw(stream, 3, limits, limits ? dst.n() : 0, metalrt::Access::Read),
         metalrt::Argument::scalar(4, a)},
        descriptor);
}

/// Utility callers choose an immutable option descriptor before recording work.
template <typename T, typename... Arguments>
void utility(metalStream_t stream, Arguments&&... arguments) {
    utility_impl<T>(stream, kernel_traits::UtilityTraits::descriptor(),
                    std::forward<Arguments>(arguments)...);
}

template <typename T, typename... Arguments>
void composite_utility(metalStream_t stream, ReductionType reduction, Arguments&&... arguments) {
    utility_impl<T>(stream, kernel_traits::composite_kernel(static_cast<uint32_t>(reduction)),
                    std::forward<Arguments>(arguments)...);
}

template <typename T, typename... Arguments>
void lod_utility(metalStream_t stream, bool soft, Arguments&&... arguments) {
    utility_impl<T>(stream, kernel_traits::lod_kernel(soft), std::forward<Arguments>(arguments)...);
}

/// Accumulates the product reduction's cross-child second-order adjoint on the GPU.
template <typename T>
void product_second(metalStream_t stream, const GPUMatrixDynamic<T>& dy,
                    const GPUMatrixDynamic<T>& values, const GPUMatrixDynamic<T>& tangents,
                    GPUMatrixDynamic<T>& dst, uint32_t index, uint32_t nested) {
    UtilityArgs a{};
    a.src = describe(dy);
    a.dst = describe(dst);
    a.aux = describe(values);
    a.width = dst.m();
    a.index = index;
    a.nested = nested;
    check(values.m() == tangents.m() && values.n() == tangents.n() &&
              values.layout() == tangents.layout(),
          "Composite tangent layout mismatch.");
    launch<T>(stream, "product_second", dst.n_elements(),
              {matrix(stream, 0, dy, metalrt::Access::Read),
               matrix(stream, 1, values, metalrt::Access::Read),
               matrix(stream, 2, tangents, metalrt::Access::Read),
               matrix(stream, 3, dst, metalrt::Access::Write), metalrt::Argument::scalar(4, a)},
              kernel_traits::CompositeProductTraits::descriptor());
}

/// Analytic constructors retain the reference feature ordering, dimensions, and padding.
template <typename T>
AnalyticEncoding<T>::AnalyticEncoding(uint32_t kind, uint32_t n_dims_to_encode, uint32_t degree,
                                      float scale, float offset)
    : m_kind{kind}, m_n_dims_to_encode{n_dims_to_encode}, m_degree{degree}, m_scale{scale},
      m_offset{offset} {
    check(std::isfinite(scale) && std::isfinite(offset), "Encoding scale/offset must be finite.");
    if (kind == 4)
        check(degree && !(degree & (degree - 1)), "Number of bins must be a power of 2.");
    if (kind == 5)
        check(n_dims_to_encode == 3 && degree >= 1 && degree <= 8,
              "SphericalHarmonics requires 3 dimensions and degree 1 through 8.");
    const uint64_t width = kind == 0   ? 0
                           : kind == 1 ? n_dims_to_encode
                           : kind == 5 ? uint64_t(degree) * degree
                                       : uint64_t(n_dims_to_encode) * degree * (kind == 2 ? 2 : 1);
    check(width <= UINT32_MAX, "Encoding output width overflow.");
    m_n_output_dims = m_padded_output_width = static_cast<uint32_t>(width);
}

template <typename T>
void AnalyticEncoding<T>::set_padded_output_width(uint32_t padded_output_width) {
    check(padded_output_width >= m_n_output_dims,
          "Encoding padding is smaller than its feature width.");
    m_padded_output_width = padded_output_width;
}

template <typename T> json AnalyticEncoding<T>::hyperparams() const {
    switch (m_kind) {
    case 0:
        return {{"otype", "Empty"}};
    case 1:
        return {{"otype", "Identity"}, {"scale", m_scale}, {"offset", m_offset}};
    case 2:
        return {{"otype", "Frequency"}, {"n_frequencies", m_degree}};
    case 3:
        return {{"otype", "TriangleWave"}, {"n_frequencies", m_degree}};
    case 4:
        return {{"otype", "OneBlob"}, {"n_bins", m_degree}};
    default:
        return {{"otype", "SphericalHarmonics"}, {"degree", m_degree}};
    }
}

template <typename T> uint32_t AnalyticEncoding<T>::device_function_fwd_ctx_bytes() const {
    return m_kind == 0 || m_kind == 1 ? 0 : m_n_dims_to_encode * sizeof(float);
}

struct AnalyticContext : Context {
    const void* owner = nullptr;
    uint32_t batch = 0;
    bool prepare_input_gradients = false;
};

#ifndef TCNN_NO_FWD_BWD
template <typename T>
std::unique_ptr<Context>
AnalyticEncoding<T>::forward_impl(metalStream_t stream, const GPUMatrixDynamic<float>& input,
                                  GPUMatrixDynamic<T>* output, bool use_inference_params,
                                  bool prepare_input_gradients) {
    (void)use_inference_params;
    auto ctx = std::make_unique<AnalyticContext>();
    ctx->owner = this;
    ctx->batch = input.n();
    ctx->prepare_input_gradients = prepare_input_gradients;
    if (output && output->n_elements()) {
        Args a{};
        a.x = describe(input);
        a.y = describe(*output);
        a.kind = m_kind;
        a.width = m_n_output_dims;
        a.dims = m_n_dims_to_encode;
        a.degree = m_degree;
        a.scale = m_scale;
        a.offset = m_offset;
        a.padding = 1;
        a.resolution = m_padded_output_width - m_n_output_dims;
        launch<T>(stream, "forward", output->n_elements(),
                  {matrix(stream, 0, input, metalrt::Access::Read),
                   raw<T>(stream, 1, nullptr, 0, metalrt::Access::Read),
                   raw<float>(stream, 2, nullptr, 0, metalrt::Access::Read),
                   matrix(stream, 3, *output, metalrt::Access::Write),
                   metalrt::Argument::scalar(4, a)},
                  kernel_traits::analytic_kernel(m_kind));
    }
    return ctx;
}
#endif // TCNN_NO_FWD_BWD

#ifndef TCNN_NO_FWD_BWD
template <typename T>
void AnalyticEncoding<T>::backward_impl(metalStream_t stream, const Context& ctx,
                                        const GPUMatrixDynamic<float>& input,
                                        const GPUMatrixDynamic<T>& output,
                                        const GPUMatrixDynamic<T>& dL_doutput,
                                        GPUMatrixDynamic<float>* dL_dinput,
                                        bool use_inference_params,
                                        GradientMode param_gradients_mode) {
    (void)output;
    (void)use_inference_params;
    (void)param_gradients_mode;
    const auto& forward = dynamic_cast<const AnalyticContext&>(ctx);
    check(forward.owner == this && forward.batch == input.n(),
          "Encoding context owner or batch mismatch.");
    if (!dL_dinput)
        return;
    if (m_kind == 2 || m_kind == 3)
        check(forward.prepare_input_gradients,
              "Frequency/TriangleWave input gradients require prepare_input_gradients=true.");
    Args a{};
    a.x = describe(input);
    a.dy = describe(dL_doutput);
    a.dx = describe(*dL_dinput);
    a.kind = m_kind;
    a.width = m_n_output_dims;
    a.dims = m_n_dims_to_encode;
    a.degree = m_degree;
    a.scale = m_scale;
    a.offset = m_offset;
    a.resolution = m_padded_output_width - m_n_output_dims;
    launch<T>(stream, "dx", dL_dinput->n_elements(),
              {matrix(stream, 0, input, metalrt::Access::Read),
               raw<T>(stream, 1, nullptr, 0, metalrt::Access::Read),
               raw<float>(stream, 2, nullptr, 0, metalrt::Access::Read),
               matrix(stream, 3, dL_doutput, metalrt::Access::Read),
               matrix(stream, 4, *dL_dinput, metalrt::Access::Write),
               raw<float>(stream, 5, nullptr, 0, metalrt::Access::Read),
               metalrt::Argument::scalar(6, a)},
              kernel_traits::analytic_kernel(m_kind));
}
#endif // TCNN_NO_FWD_BWD

#ifndef TCNN_NO_FWD_BWD
template <typename T>
void AnalyticEncoding<T>::backward_backward_input_impl(
    metalStream_t stream, const Context& ctx, const GPUMatrixDynamic<float>& input,
    const GPUMatrixDynamic<float>& dL_ddLdinput, const GPUMatrixDynamic<T>& dL_doutput,
    GPUMatrixDynamic<T>* dL_ddLdoutput, GPUMatrixDynamic<float>* dL_dinput,
    bool use_inference_params, GradientMode param_gradients_mode) {
    // Join an outer batch/capture; otherwise submit this entire encoder call once.
    auto batch = detail::resolve_stream(stream)->batch();
    (void)use_inference_params;
    (void)param_gradients_mode;
    const auto& forward = dynamic_cast<const AnalyticContext&>(ctx);
    check(forward.owner == this && forward.batch == input.n(),
          "Encoding context owner or batch mismatch.");
    Args a{};
    a.x = describe(input);
    a.ddx = describe(dL_ddLdinput);
    a.dy = describe(dL_doutput);
    a.kind = m_kind;
    a.width = m_n_output_dims;
    a.dims = m_n_dims_to_encode;
    a.degree = m_degree;
    a.scale = m_scale;
    a.offset = m_offset;
    a.resolution = m_padded_output_width - m_n_output_dims;
    a.mode = 1;
    if (dL_ddLdoutput) {
        a.ddy = describe(*dL_ddLdoutput);
        launch<T>(stream, "ddy", dL_ddLdoutput->n_elements(),
                  {matrix(stream, 0, input, metalrt::Access::Read),
                   raw<T>(stream, 1, nullptr, 0, metalrt::Access::Read),
                   raw<float>(stream, 2, nullptr, 0, metalrt::Access::Read),
                   matrix(stream, 3, dL_ddLdinput, metalrt::Access::Read),
                   matrix(stream, 4, *dL_ddLdoutput, metalrt::Access::Write),
                   metalrt::Argument::scalar(5, a)},
                  kernel_traits::analytic_kernel(m_kind));
    }
    if (dL_dinput) {
        a.dx = describe(*dL_dinput);
        launch<T>(stream, "dx", dL_dinput->n_elements(),
                  {matrix(stream, 0, input, metalrt::Access::Read),
                   raw<T>(stream, 1, nullptr, 0, metalrt::Access::Read),
                   raw<float>(stream, 2, nullptr, 0, metalrt::Access::Read),
                   matrix(stream, 3, dL_doutput, metalrt::Access::Read),
                   matrix(stream, 4, *dL_dinput, metalrt::Access::Write),
                   matrix(stream, 5, dL_ddLdinput, metalrt::Access::Read),
                   metalrt::Argument::scalar(6, a)},
                  kernel_traits::analytic_kernel(m_kind));
    }
    batch.finish();
}
#endif // TCNN_NO_FWD_BWD

/// Reads strict nonnegative JSON integers, including checks before narrowing to uint32_t.
uint32_t integer(const json& config, const char* name, uint32_t fallback) {
    auto it = config.find(name);
    if (it == config.end())
        return fallback;
    check(it->is_number_integer(), "Encoding dimension/count must be an integer.");
    if (it->is_number_unsigned()) {
        auto v = it->get<uint64_t>();
        check(v <= UINT32_MAX, "Encoding integer overflow.");
        return uint32_t(v);
    }
    const auto v = it->get<int64_t>();
    check(v >= 0 && uint64_t(v) <= UINT32_MAX,
          "Encoding integer must be nonnegative and fit uint32_t.");
    return uint32_t(v);
}

/// Validates metadata once; the resulting lattice description is immutable during replay.
template <typename T>
LatticeEncoding<T>::LatticeEncoding(uint32_t n_dims_to_encode, const json& encoding, bool permuto)
    : m_n_dims_to_encode{n_dims_to_encode}, m_permuto{permuto} {
    check(encoding.is_object(), "Encoding configuration must be an object.");
    const std::set<uint32_t> permuto_dims{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24};
    check(permuto || lower(encoding.value("hash", "CoherentPrime")) == "baseconvert"
              ? permuto_dims.count(n_dims_to_encode) != 0
              : (n_dims_to_encode >= 1 && n_dims_to_encode <= 4),
          "Unsupported encoding input dimensionality for the selected lattice/hash.");
    m_n_features_per_level = integer(encoding, "n_features_per_level", 2);
    check(m_n_features_per_level == 1 || m_n_features_per_level == 2 ||
              m_n_features_per_level == 4 || m_n_features_per_level == 8,
          "Features per level must be 1, 2, 4, or 8.");
    const bool total = encoding.contains("n_features") || encoding.contains("n_grid_features");
    check(!total || !encoding.contains("n_levels"),
          "Total features and n_levels are mutually exclusive.");
    if (permuto)
        check(!(encoding.contains("n_features") && encoding.contains("n_grid_features")),
              "Feature aliases are mutually exclusive.");
    if (total) {
        auto features = integer(
            encoding, encoding.contains("n_features") ? "n_features" : "n_grid_features", 0);
        check(features && features % m_n_features_per_level == 0,
              "Features must be a positive multiple of features per level.");
        m_n_levels = features / m_n_features_per_level;
    } else
        m_n_levels = integer(encoding, "n_levels", 16);
    check(m_n_levels && m_n_levels <= (permuto ? 32u : 128u),
          "Encoding level count is out of range.");
    m_padded_output_width = m_n_levels * m_n_features_per_level;
    const uint32_t log2_hashmap_size = integer(encoding, "log2_hashmap_size", 19);
    check(log2_hashmap_size < 32, "Hash table exponent is out of range.");
    m_max_input_grad_dims =
        permuto ? integer(encoding, "max_input_grad_dims", n_dims_to_encode) : n_dims_to_encode;
    check(m_max_input_grad_dims <= n_dims_to_encode,
          "Input gradient dimension exceeds input width.");
    const auto interpolation = lower(encoding.value("interpolation", "Linear"));
    check(interpolation == "linear" ||
              (!permuto && (interpolation == "nearest" || interpolation == "smoothstep")),
          "Unsupported lattice interpolation.");
    m_interpolation = interpolation == "nearest" ? 0 : interpolation == "linear" ? 1 : 2;
    const auto otype = lower(encoding.value("otype", "Grid"));
    const auto grid_type = lower(encoding.value("type", otype == "densegrid"   ? "Dense"
                                                        : otype == "tiledgrid" ? "Tiled"
                                                                               : "Hash"));
    check(permuto || grid_type == "hash" || grid_type == "dense" || grid_type == "tiled",
          "Unsupported grid type.");
    m_grid_type = grid_type == "hash" ? 0 : grid_type == "dense" ? 1 : 2;
    m_hash_type = static_cast<uint32_t>(
        permuto ? HashType::BaseConvert
                : string_to_hash_type(encoding.value("hash", "CoherentPrime")));
    m_base_resolution = integer(encoding, "base_resolution", 16);
    check(permuto || m_base_resolution > 0, "Grid base resolution must be positive.");
    m_base_scale = encoding.value("base_scale", 16.0f);
    const float default_scale =
        !permuto && m_grid_type == 1
            ? std::exp(std::log(256.0f / float(m_base_resolution)) / float(m_n_levels - 1))
            : 2.0f;
    m_per_level_scale = encoding.value("per_level_scale", default_scale);
    check(std::isfinite(m_base_scale) && std::isfinite(m_per_level_scale) && m_per_level_scale > 0,
          "Lattice scales must be finite and per_level_scale positive.");
    m_fixed_point_pos = encoding.value("fixed_point_pos", false);
    m_stochastic_interpolation = encoding.value("stochastic_interpolation", false);
    m_has_optimized_parameter_gradients =
        !m_permuto && m_grid_type == 0 &&
        kernel_traits::HashGridParameterGradientTraits::supports_configuration(
            m_n_dims_to_encode, m_n_levels, m_n_features_per_level, log2_hashmap_size,
            m_base_resolution, m_per_level_scale, m_interpolation, m_hash_type, m_fixed_point_pos,
            m_stochastic_interpolation, sizeof(T) == 2);
    const uint32_t seed = integer(encoding, "seed", 1337);
    pcg32 rng{seed};
    uint64_t offset = 0;
    const uint64_t table_limit = uint64_t{1} << log2_hashmap_size;
    for (uint32_t level = 0; level < m_n_levels; ++level) {
        uint64_t count = table_limit;
        const float scale = m_base_scale * std::exp2(level * std::log2(m_per_level_scale));
        if (permuto) {
            const double budget = double(INT32_MAX) / (2.0 * (n_dims_to_encode + 1));
            const double max_scale = (budget - (n_dims_to_encode + 1)) / (n_dims_to_encode * 6.0);
            check(std::isfinite(scale) && std::abs(double(scale)) <= max_scale,
                  "Permuto scale exceeds its normalized-input coordinate range.");
            for (uint32_t d = 0; d < n_dims_to_encode; ++d) {
                m_scales.push_back(scale / std::sqrt(float((d + 1) * (d + 2))));
                m_shifts.push_back((rng.nextFloat() - .5f) * 10);
            }
        } else {
            const float grid_scale =
                std::exp2(level * std::log2(m_per_level_scale)) * m_base_resolution - 1;
            check(std::isfinite(grid_scale) && grid_scale >= 0 &&
                      double(grid_scale) < UINT32_MAX - 1.0,
                  "Grid resolution is out of range.");
            const uint32_t resolution = uint32_t(std::ceil(grid_scale)) + 1;
            const double dense = std::pow(double(resolution), double(n_dims_to_encode));
            count = std::min<uint64_t>(uint64_t(INT32_MAX),
                                       dense >= INT32_MAX ? uint64_t(INT32_MAX) : uint64_t(dense));
            count = (count + 7) & ~uint64_t(7);
            if (m_grid_type == 0)
                count = std::min(count, table_limit);
            if (m_grid_type == 2) {
                const double tile = std::pow(double(m_base_resolution), double(n_dims_to_encode));
                check(tile <= UINT32_MAX, "Tiled grid size overflow.");
                count = std::min(count, uint64_t(tile));
            }
            m_scales.push_back(grid_scale);
        }
        check(count > 0 && offset + count <= UINT32_MAX / m_n_features_per_level,
              "Lattice parameter count exceeds uint32_t.");
        m_offset_table.data[level] = uint32_t(offset);
        offset += count;
    }
    m_offset_table.data[m_n_levels] = uint32_t(offset);
    m_offset_table.size = m_n_levels + 1;
    if (permuto) {
        m_hyperparams = {{"otype", "Permuto"},
                         {"n_levels", m_n_levels},
                         {"n_features_per_level", m_n_features_per_level},
                         {"base_scale", m_base_scale},
                         {"per_level_scale", m_per_level_scale},
                         {"log2_hashmap_size", log2_hashmap_size},
                         {"max_input_grad_dims", m_max_input_grad_dims},
                         {"seed", seed},
                         {"scales_table", m_scales},
                         {"shifts_table", m_shifts}};
        for (const char* key : {"scales_table", "shifts_table"})
            if (encoding.contains(key)) {
                const auto& supplied = encoding.at(key);
                const auto expected = m_hyperparams.at(key).template get<std::vector<float>>();
                check(supplied.is_array() && supplied.size() == expected.size(),
                      "Permuto lattice table size mismatch.");
                for (size_t i = 0; i < expected.size(); ++i) {
                    check(supplied[i].is_number(), "Permuto lattice table values must be numeric.");
                    const double value = supplied[i].template get<double>();
                    check(std::isfinite(value) &&
                              std::abs(value - expected[i]) <=
                                  8 * std::numeric_limits<float>::epsilon() *
                                      std::max(1.0, std::abs(double(expected[i]))),
                          "Permuto lattice table does not match its seed and scales.");
                }
            }
    } else {
        m_hyperparams = {{"otype", "Grid"},
                         {"type", m_grid_type == 0   ? "Hash"
                                  : m_grid_type == 1 ? "Dense"
                                                     : "Tiled"},
                         {"n_levels", m_n_levels},
                         {"n_features_per_level", m_n_features_per_level},
                         {"base_resolution", m_base_resolution},
                         {"per_level_scale", m_per_level_scale},
                         {"interpolation", m_interpolation == 0   ? "Nearest"
                                           : m_interpolation == 1 ? "Linear"
                                                                  : "Smoothstep"},
                         {"hash", to_string(static_cast<HashType>(m_hash_type))}};
        if (m_grid_type == 0)
            m_hyperparams["log2_hashmap_size"] = log2_hashmap_size;
    }
}

template <typename T>
void LatticeEncoding<T>::set_padded_output_width(uint32_t padded_output_width) {
    check(padded_output_width >= m_n_levels * m_n_features_per_level,
          "Lattice padding is too small.");
    m_padded_output_width = padded_output_width;
}

template <typename T> size_t LatticeEncoding<T>::level_params_offset(uint32_t level) const {
    check(level < m_offset_table.size, "Out of bounds params offset request.");
    return m_offset_table.data[level];
}

/// Initializes trainable lattice values on the selected GPU using the reference PCG layout.
template <typename T>
void LatticeEncoding<T>::initialize_params(pcg32& rnd, float* params_full_precision, float scale) {
    check(std::isfinite(scale), "Initialization scale must be finite.");
    generate_random_uniform<float>(rnd, n_params(), params_full_precision, -1e-4f * scale,
                                   1e-4f * scale);
}

struct LatticeContext : Context {
    const void* owner = nullptr;
    uint32_t batch = 0;
    bool prepare_input_gradients = false;
};

/// Records per-level work with fixed resources; parameters and level limits stay on the GPU.
#ifndef TCNN_NO_FWD_BWD
template <typename T>
void LatticeEncoding<T>::run(metalStream_t stream, const GPUMatrixDynamic<float>& input,
                             GPUMatrixDynamic<T>* output, const GPUMatrixDynamic<T>* dL_doutput,
                             GPUMatrixDynamic<float>* dL_dinput,
                             const GPUMatrixDynamic<float>* dL_ddLdinput,
                             GPUMatrixDynamic<T>* dL_ddLdoutput, bool use_inference_params,
                             GradientMode param_gradients_mode, uint32_t operation,
                             bool grouped_inference) {
    if (operation == 0 && !output)
        return;
    // A lattice call may emit padding, per-level derivatives and gradient casts.
    // Join an enclosing batch/capture, otherwise submit the entire call once.
    // Recorded commands retain their buffers, including per-level temporaries.
    auto batch = detail::resolve_stream(stream)->batch();
    if (output && m_padded_output_width > m_n_levels * m_n_features_per_level) {
        auto pad = output->slice_rows(m_n_levels * m_n_features_per_level,
                                      m_padded_output_width - m_n_levels * m_n_features_per_level);
        utility<T>(stream, nullptr, pad, 0);
    }
    if (dL_ddLdoutput && m_padded_output_width > m_n_levels * m_n_features_per_level) {
        auto pad =
            dL_ddLdoutput->slice_rows(m_n_levels * m_n_features_per_level,
                                      m_padded_output_width - m_n_levels * m_n_features_per_level);
        utility<T>(stream, nullptr, pad, 0);
    }
    const T* params = use_inference_params ? this->inference_params() : this->params();
    bool grouped_parameters = false;
    if (operation == 1 && param_gradients_mode != GradientMode::Ignore &&
        m_has_optimized_parameter_gradients) {
        const auto capabilities =
            metalrt::stream_or_default(detail::resolve_stream(stream)).device().capabilities();
        grouped_parameters =
            capabilities.backend == metalrt::BackendKind::Metal &&
            kernel_traits::HashGridParameterGradientTraits::supports_dispatch(
                capabilities.apple_gpu_family, input.n(), m_per_level_scale == 2.0f);
    }
    std::vector<kernel_traits::HashGridInferenceLevel> inference_levels;
    if (grouped_inference)
        inference_levels.reserve(m_n_levels);
    std::vector<kernel_traits::HashGridGradientLevel> gradient_levels;
    if (grouped_parameters)
        gradient_levels.reserve(m_n_levels);
    for (uint32_t level = 0; level < m_n_levels; ++level) {
        Args a{};
        a.x = describe(input);
        a.y = output ? describe(*output) : Mat{};
        a.dy = dL_doutput ? describe(*dL_doutput) : Mat{};
        a.dx = dL_dinput ? describe(*dL_dinput) : Mat{};
        a.ddx = dL_ddLdinput ? describe(*dL_ddLdinput) : Mat{};
        a.ddy = dL_ddLdoutput ? describe(*dL_ddLdoutput) : Mat{};
        a.kind = m_permuto ? 7 : 6;
        a.dims = m_n_dims_to_encode;
        a.level = level;
        a.levels = m_n_levels;
        a.features = m_n_features_per_level;
        a.table_size = m_offset_table.data[level + 1] - m_offset_table.data[level];
        a.interpolation = m_interpolation;
        a.grid_type = m_grid_type;
        a.hash_type = m_hash_type;
        a.max_input_grad_dims = m_max_input_grad_dims;
        a.max_level = this->m_max_level;
        a.has_limits = this->m_max_level_gpu ? 1 : 0;
        a.mode = operation == 2 ? 1 : 0;
        a.accumulate = level ? 1 : 0;
        a.stochastic = m_stochastic_interpolation ? 1 : 0;
        if (m_permuto) {
            for (uint32_t d = 0; d < m_n_dims_to_encode; ++d) {
                a.scales[d] = m_scales[level * m_n_dims_to_encode + d];
                a.shifts[d] = m_shifts[level * m_n_dims_to_encode + d];
            }
        } else {
            a.scale = m_scales[level];
            a.resolution = uint32_t(std::ceil(a.scale)) + 1;
        }
        const size_t offset = size_t(m_offset_table.data[level]) * m_n_features_per_level,
                     count = size_t(a.table_size) * m_n_features_per_level;
        const T* p = params ? params + offset : nullptr;
        if (operation == 0 && grouped_inference) {
            inference_levels.push_back({a, uint32_t(offset)});
        } else if (operation == 0) {
            launch<T>(stream, "forward", size_t(m_n_features_per_level) * input.n(),
                      {matrix(stream, 0, input, metalrt::Access::Read),
                       raw(stream, 1, p, count, metalrt::Access::Read),
                       raw(stream, 2, this->m_max_level_gpu, this->m_max_level_gpu ? input.n() : 0,
                           metalrt::Access::Read),
                       matrix(stream, 3, *output, metalrt::Access::Write),
                       metalrt::Argument::scalar(4, a)},
                      kernel_traits::lattice_kernel(m_permuto, m_grid_type, m_hash_type));
        } else {
            if (grouped_parameters) {
                gradient_levels.push_back({a, uint32_t(offset)});
            } else if (param_gradients_mode != GradientMode::Ignore) {
                GPUMatrixDynamic<float> temporary{static_cast<uint32_t>(count), 1, stream};
                launch<T>(stream, "dp", m_n_features_per_level,
                          {matrix(stream, 0, input, metalrt::Access::Read),
                           raw(stream, 1, this->m_max_level_gpu,
                               this->m_max_level_gpu ? input.n() : 0, metalrt::Access::Read),
                           matrix(stream, 2, *dL_doutput, metalrt::Access::Read),
                           matrix(stream, 3, temporary,
                                  // Higher-order kernels accumulate with += after clearing
                                  // this buffer. A singleton indirect pass must declare
                                  // both accesses, even before the later cast reads it.
                                  metalrt::Access::ReadWrite),
                           dL_ddLdinput ? matrix(stream, 4, *dL_ddLdinput, metalrt::Access::Read)
                                        : raw<float>(stream, 4, nullptr, 0, metalrt::Access::Read),
                           metalrt::Argument::scalar(5, a)},
                          kernel_traits::lattice_kernel(m_permuto, m_grid_type, m_hash_type));
                launch<T>(stream, "cast_gradient", count,
                          {matrix(stream, 0, temporary, metalrt::Access::Read),
                           raw(stream, 1, this->gradients() + offset, count,
                               param_gradients_mode == GradientMode::Accumulate
                                   ? metalrt::Access::ReadWrite
                                   : metalrt::Access::Write),
                           metalrt::Argument::scalar(2, uint32_t(count)),
                           metalrt::Argument::scalar(
                               3, uint32_t(param_gradients_mode == GradientMode::Accumulate))});
            }
            if (dL_dinput)
                launch<T>(stream, "dx", dL_dinput->n_elements(),
                          {matrix(stream, 0, input, metalrt::Access::Read),
                           raw(stream, 1, p, count, metalrt::Access::Read),
                           raw(stream, 2, this->m_max_level_gpu,
                               this->m_max_level_gpu ? input.n() : 0, metalrt::Access::Read),
                           matrix(stream, 3, *dL_doutput, metalrt::Access::Read),
                           matrix(stream, 4, *dL_dinput,
                                  level ? metalrt::Access::ReadWrite : metalrt::Access::Write),
                           dL_ddLdinput ? matrix(stream, 5, *dL_ddLdinput, metalrt::Access::Read)
                                        : raw<float>(stream, 5, nullptr, 0, metalrt::Access::Read),
                           metalrt::Argument::scalar(6, a)},
                          kernel_traits::lattice_kernel(m_permuto, m_grid_type, m_hash_type));
            if (dL_ddLdoutput)
                launch<T>(stream, "ddy", size_t(m_n_features_per_level) * input.n(),
                          {matrix(stream, 0, input, metalrt::Access::Read),
                           raw(stream, 1, p, count, metalrt::Access::Read),
                           raw(stream, 2, this->m_max_level_gpu,
                               this->m_max_level_gpu ? input.n() : 0, metalrt::Access::Read),
                           matrix(stream, 3, *dL_ddLdinput, metalrt::Access::Read),
                           matrix(stream, 4, *dL_ddLdoutput, metalrt::Access::Write),
                           metalrt::Argument::scalar(5, a)},
                          kernel_traits::lattice_kernel(m_permuto, m_grid_type, m_hash_type));
        }
    }
    if (grouped_inference)
        launch_hash_grid_inference(stream, input, *output, params, this->n_params(),
                                   this->m_max_level_gpu, inference_levels);
    if (grouped_parameters)
        launch_hash_grid_parameter_gradient<T>(stream, input, *dL_doutput, this->m_max_level_gpu,
                                               this->gradients(), uint32_t(this->n_params()),
                                               param_gradients_mode, gradient_levels);
    // Report submission errors here; Scope's noexcept destructor can only defer them.
    batch.finish();
}
#endif // TCNN_NO_FWD_BWD

#ifndef TCNN_NO_FWD_BWD
template <typename T>
void LatticeEncoding<T>::inference_mixed_precision_impl(metalStream_t stream,
                                                        const GPUMatrixDynamic<float>& input,
                                                        GPUMatrixDynamic<T>& output,
                                                        bool use_inference_params) {
    const auto capabilities =
        metalrt::stream_or_default(detail::resolve_stream(stream)).device().capabilities();
    // Promote only the measured geometry and Apple hardware family. Other
    // grids/Permuto/reference devices retain the established per-level path.
    const bool grouped =
        !m_permuto && m_grid_type == 0 && m_n_dims_to_encode == 2 && m_n_features_per_level == 2 &&
        m_hash_type == 1 && this->n_params() <= UINT32_MAX &&
        capabilities.backend == metalrt::BackendKind::Metal && capabilities.apple_gpu_family >= 7;
    run(stream, input, &output, nullptr, nullptr, nullptr, nullptr, use_inference_params,
        GradientMode::Ignore, 0, grouped);
}

template <typename T>
std::unique_ptr<Context>
LatticeEncoding<T>::forward_impl(metalStream_t stream, const GPUMatrixDynamic<float>& input,
                                 GPUMatrixDynamic<T>* output, bool use_inference_params,
                                 bool prepare_input_gradients) {
    auto ctx = std::make_unique<LatticeContext>();
    ctx->owner = this;
    ctx->batch = input.n();
    ctx->prepare_input_gradients = prepare_input_gradients;
    run(stream, input, output, nullptr, nullptr, nullptr, nullptr, use_inference_params,
        GradientMode::Ignore, 0);
    return ctx;
}
#endif // TCNN_NO_FWD_BWD

#ifndef TCNN_NO_FWD_BWD
template <typename T>
void LatticeEncoding<T>::backward_impl(metalStream_t stream, const Context& ctx,
                                       const GPUMatrixDynamic<float>& input,
                                       const GPUMatrixDynamic<T>& output,
                                       const GPUMatrixDynamic<T>& dL_doutput,
                                       GPUMatrixDynamic<float>* dL_dinput,
                                       bool use_inference_params,
                                       GradientMode param_gradients_mode) {
    (void)output;
    const auto& forward = dynamic_cast<const LatticeContext&>(ctx);
    check(forward.owner == this && forward.batch == input.n(),
          "Lattice context owner or batch mismatch.");
    if (!m_permuto && dL_dinput)
        check(forward.prepare_input_gradients,
              "Grid input gradients require prepare_input_gradients=true.");
    run(stream, input, nullptr, &dL_doutput, dL_dinput, nullptr, nullptr, use_inference_params,
        param_gradients_mode, 1);
}
#endif // TCNN_NO_FWD_BWD

#ifndef TCNN_NO_FWD_BWD
template <typename T>
void LatticeEncoding<T>::backward_backward_input_impl(
    metalStream_t stream, const Context& ctx, const GPUMatrixDynamic<float>& input,
    const GPUMatrixDynamic<float>& dL_ddLdinput, const GPUMatrixDynamic<T>& dL_doutput,
    GPUMatrixDynamic<T>* dL_ddLdoutput, GPUMatrixDynamic<float>* dL_dinput,
    bool use_inference_params, GradientMode param_gradients_mode) {
    const auto& forward = dynamic_cast<const LatticeContext&>(ctx);
    check(forward.owner == this && forward.batch == input.n(),
          "Lattice context owner or batch mismatch.");
    if (!m_permuto && dL_ddLdoutput)
        check(forward.prepare_input_gradients,
              "Grid double backward requires prepare_input_gradients=true.");
    run(stream, input, nullptr, &dL_doutput, dL_dinput, &dL_ddLdinput, dL_ddLdoutput,
        use_inference_params, param_gradients_mode, 2);
}
#endif // TCNN_NO_FWD_BWD
} // namespace encoding_detail
} // namespace tcnn

namespace tcnn {
using encoding_detail::check;

/// Resolves child input slices and aligns child outputs before any commands are recorded.
template <typename T>
CompositeEncoding<T>::CompositeEncoding(const json& params, uint32_t n_dims_to_encode)
    : m_n_dims_to_encode{n_dims_to_encode} {
    check(params.contains("nested") && params["nested"].is_array(),
          "CompositeEncoding requires an array of nested encodings.");
    m_reduction_type = string_to_reduction_type(params.value("reduction", "Concatenation"));
    const auto& children = params["nested"];
    uint64_t total = 0;
    bool explicit_offsets = false;
    for (const auto& child : children) {
        total += encoding_detail::integer(child, "n_dims_to_encode", 0);
        explicit_offsets = explicit_offsets || child.contains("dims_to_encode_begin");
    }
    check(explicit_offsets || total <= n_dims_to_encode,
          "Composite children encode too many dimensions.");
    uint32_t unspecified = explicit_offsets ? UINT32_MAX : n_dims_to_encode - uint32_t(total),
             offset = 0;
    for (const auto& child : children) {
        uint32_t dims;
        if (child.contains("n_dims_to_encode")) {
            dims = encoding_detail::integer(child, "n_dims_to_encode", 0);
            if (child.contains("dims_to_encode_begin"))
                offset = encoding_detail::integer(child, "dims_to_encode_begin", 0);
        } else {
            check(unspecified != UINT32_MAX, "Only one child may omit n_dims_to_encode.");
            dims = unspecified;
            unspecified = UINT32_MAX;
        }
        check(offset <= n_dims_to_encode && dims <= n_dims_to_encode - offset,
              "Composite child input slice is out of range.");
        if (dims) {
            m_nested.emplace_back(create_encoding<T>(dims, child, 1));
            m_dims_to_encode_begin.push_back(offset);
        }
        offset += dims;
    }
    if (m_reduction_type == ReductionType::Concatenation) {
        uint32_t previous = 0;
        for (size_t i = 0; i + 1 < m_nested.size(); ++i) {
            const auto width = next_multiple(previous + m_nested[i]->output_width(),
                                             m_nested[i + 1]->required_output_alignment()) -
                               previous;
            m_nested[i]->set_padded_output_width(width);
            previous += width;
        }
    } else {
        const uint32_t alignment = required_output_alignment();
        for (auto& child : m_nested)
            child->set_alignment(alignment);
        for (const auto& child : m_nested)
            check(child->output_width() == m_nested.front()->output_width(),
                  "Reduced child output widths must match.");
    }
}

template <typename T> uint32_t CompositeEncoding<T>::padded_output_width() const {
    if (m_reduction_type != ReductionType::Concatenation)
        return m_nested.empty() ? 0 : m_nested.front()->padded_output_width();
    uint64_t width = 0;
    for (const auto& child : m_nested)
        width += child->padded_output_width();
    check(width <= UINT32_MAX, "Composite output width overflow.");
    return uint32_t(width);
}

template <typename T> uint32_t CompositeEncoding<T>::required_output_alignment() const {
    uint32_t alignment = 1;
    for (const auto& child : m_nested)
        alignment = lcm(alignment, child->required_output_alignment());
    return alignment;
}

template <typename T>
void CompositeEncoding<T>::set_padded_output_width(uint32_t padded_output_width) {
    if (m_nested.empty()) {
        check(padded_output_width == 0, "An empty Composite cannot produce padding.");
        return;
    }
    if (m_reduction_type == ReductionType::Concatenation) {
        const uint32_t previous =
            this->padded_output_width() - m_nested.back()->padded_output_width();
        check(padded_output_width >= previous, "Composite padding is too small.");
        m_nested.back()->set_padded_output_width(padded_output_width - previous);
    } else
        for (auto& child : m_nested)
            child->set_padded_output_width(padded_output_width);
}

template <typename T>
void CompositeEncoding<T>::set_params_impl(T* params, T* inference_params, T* gradients) {
    size_t offset = 0;
    for (auto& child : m_nested) {
        child->set_params(params ? params + offset : nullptr,
                          inference_params ? inference_params + offset : nullptr,
                          gradients ? gradients + offset : nullptr);
        offset += child->n_params();
    }
}

template <typename T>
void CompositeEncoding<T>::initialize_params(pcg32& rnd, float* params_full_precision,
                                             float scale) {
    size_t offset = 0;
    for (auto& child : m_nested) {
        child->initialize_params(
            rnd, params_full_precision ? params_full_precision + offset : nullptr, scale);
        offset += child->n_params();
    }
}

template <typename T> size_t CompositeEncoding<T>::n_params() const {
    size_t count = 0;
    for (const auto& child : m_nested)
        count += child->n_params();
    return count;
}

template <typename T> json CompositeEncoding<T>::hyperparams() const {
    json children = json::array();
    for (const auto& child : m_nested)
        children.push_back(child->hyperparams());
    return {{"otype", "Composite"}, {"nested", children}};
}

template <typename T>
std::string CompositeEncoding<T>::generate_nested_name(const std::string& name, size_t i) const {
    return name + "_" + std::to_string(i) + "_" + to_snake_case(nested(i)->name());
}

template <typename T> uint32_t CompositeEncoding<T>::device_function_fwd_ctx_bytes() const {
    size_t bytes = 0;
    for (const auto& child : m_nested) {
        const size_t child_bytes = child->device_function_fwd_ctx_bytes();
        bytes = metalrt::checked_add(bytes, (child_bytes + 3u) & ~size_t(3u));
    }
    // Product differentiation needs each factor, including zero-valued factors.
    // Keep them as floats so child contexts remain naturally aligned.
    if (m_reduction_type == ReductionType::Product)
        bytes = metalrt::checked_add(
            bytes,
            metalrt::checked_mul(metalrt::checked_mul(m_nested.size(), padded_output_width()),
                                 sizeof(float)));
    check(bytes <= UINT32_MAX, "Composite JIT context exceeds uint32_t.");
    return static_cast<uint32_t>(bytes);
}

template <typename T>
uint32_t CompositeEncoding<T>::backward_device_function_shmem_bytes(
    uint32_t n_threads, GradientMode param_gradients_mode) const {
    uint32_t bytes = 0;
    for (const auto& child : m_nested)
        bytes = std::max(
            bytes, child->backward_device_function_shmem_bytes(n_threads, param_gradients_mode));
    return bytes;
}

/// Child contexts and unreduced values stay alive until the caller releases backward state.
#ifndef TCNN_NO_FWD_BWD
template <typename T>
std::unique_ptr<Context>
CompositeEncoding<T>::forward_impl(metalStream_t stream, const GPUMatrixDynamic<float>& input,
                                   GPUMatrixDynamic<T>* output, bool use_inference_params,
                                   bool prepare_input_gradients) {
    // Join an outer batch/capture; otherwise submit this entire encoder call once.
    auto batch = detail::resolve_stream(stream)->batch();
    auto ctx = std::make_unique<ForwardContext>();
    ctx->owner = this;
    ctx->batch = input.n();
    ctx->nested.resize(m_nested.size());
    auto* unreduced = output;
    if (m_reduction_type != ReductionType::Concatenation) {
        check(uint64_t(padded_output_width()) * m_nested.size() <= UINT32_MAX,
              "Composite workspace overflow.");
        ctx->to_reduce = GPUMatrixDynamic<T>{padded_output_width() * uint32_t(m_nested.size()),
                                             input.n(), stream, preferred_output_layout()};
        unreduced = &ctx->to_reduce;
    }
    uint32_t offset = 0;
    for (size_t i = 0; i < m_nested.size(); ++i) {
        auto& child = m_nested[i];
        auto in = input.slice_rows(m_dims_to_encode_begin[i], child->input_width());
        GPUMatrixDynamic<T> out;
        if (unreduced)
            out = unreduced->slice_rows(offset, child->output_width());
        ctx->nested[i] = child->forward(stream, in, unreduced ? &out : nullptr,
                                        use_inference_params, prepare_input_gradients);
        offset += child->output_width();
    }
    if (output && m_reduction_type != ReductionType::Concatenation)
        encoding_detail::composite_utility<T>(stream, m_reduction_type, &ctx->to_reduce, *output,
                                              m_reduction_type == ReductionType::Sum ? 3 : 4, 0,
                                              nullptr, nullptr, padded_output_width(), 0,
                                              uint32_t(m_nested.size()));
    batch.finish();
    return ctx;
}
#endif // TCNN_NO_FWD_BWD

/// Backpropagates reductions explicitly; overlapping input slices are accumulated safely.
#ifndef TCNN_NO_FWD_BWD
template <typename T>
void CompositeEncoding<T>::backward_impl(metalStream_t stream, const Context& ctx,
                                         const GPUMatrixDynamic<float>& input,
                                         const GPUMatrixDynamic<T>& output,
                                         const GPUMatrixDynamic<T>& dL_doutput,
                                         GPUMatrixDynamic<float>* dL_dinput,
                                         bool use_inference_params,
                                         GradientMode param_gradients_mode) {
    // Join an outer batch/capture; otherwise submit this entire encoder call once.
    auto batch = detail::resolve_stream(stream)->batch();
    const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
    check(forward.owner == this && forward.batch == input.n() &&
              forward.nested.size() == m_nested.size(),
          "Composite context mismatch.");
    if (dL_dinput)
        encoding_detail::composite_utility<float>(stream, m_reduction_type, nullptr, *dL_dinput, 0);
    uint32_t offset = 0;
    for (size_t i = 0; i < m_nested.size(); ++i) {
        const auto& child = m_nested[i];
        auto in = input.slice_rows(m_dims_to_encode_begin[i], child->input_width());
        auto out = (m_reduction_type == ReductionType::Concatenation ? output : forward.to_reduce)
                       .slice_rows(offset, child->output_width());
        GPUMatrixDynamic<T> dy;
        if (m_reduction_type == ReductionType::Concatenation)
            dy = dL_doutput.slice_rows(offset, child->output_width());
        else {
            dy = GPUMatrixDynamic<T>{child->output_width(), input.n(), stream};
            encoding_detail::composite_utility<T>(
                stream, m_reduction_type, &dL_doutput, dy,
                m_reduction_type == ReductionType::Sum ? 5 : 6, 0, &forward.to_reduce, nullptr,
                padded_output_width(), uint32_t(i), uint32_t(m_nested.size()));
        }
        GPUMatrixDynamic<float> dx;
        if (dL_dinput)
            dx = GPUMatrixDynamic<float>{child->input_width(), input.n(), stream};
        child->backward(stream, *forward.nested[i], in, out, dy, dL_dinput ? &dx : nullptr,
                        use_inference_params, param_gradients_mode);
        if (dL_dinput) {
            auto dst = dL_dinput->slice_rows(m_dims_to_encode_begin[i], child->input_width());
            encoding_detail::composite_utility<float>(stream, m_reduction_type, &dx, dst, 2);
        }
        offset += child->output_width();
    }
    batch.finish();
}
#endif // TCNN_NO_FWD_BWD

/// Differentiate the reduction and then each child without reading intermediates
/// on the CPU. Product reductions include both tangent and ordinary-output adjoints.
#ifndef TCNN_NO_FWD_BWD
template <typename T>
void CompositeEncoding<T>::backward_backward_input_impl(
    metalStream_t stream, const Context& ctx, const GPUMatrixDynamic<float>& input,
    const GPUMatrixDynamic<float>& dL_ddLdinput, const GPUMatrixDynamic<T>& dL_doutput,
    GPUMatrixDynamic<T>* dL_ddLdoutput, GPUMatrixDynamic<float>* dL_dinput,
    bool use_inference_params, GradientMode param_gradients_mode) {
    // Join an outer batch/capture; otherwise submit this entire encoder call once.
    auto batch = detail::resolve_stream(stream)->batch();
    const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
    check(forward.owner == this && forward.batch == input.n() &&
              forward.nested.size() == m_nested.size(),
          "Composite context mismatch.");
    if (dL_dinput)
        encoding_detail::composite_utility<float>(stream, m_reduction_type, nullptr, *dL_dinput, 0);
    if (dL_ddLdoutput)
        encoding_detail::composite_utility<T>(stream, m_reduction_type, nullptr, *dL_ddLdoutput, 0);
    if (m_nested.empty()) {
        batch.finish();
        return;
    }
    uint64_t packed_width = 0;
    for (const auto& child : m_nested)
        packed_width += child->padded_output_width();
    check(packed_width <= UINT32_MAX, "Composite tangent workspace overflow.");
    GPUMatrixDynamic<T> tangents{uint32_t(packed_width), input.n(), stream,
                                 preferred_output_layout()};
    // Compute every child's Jv before the product reduction's cross terms.
    uint32_t offset = 0;
    for (size_t i = 0; i < m_nested.size(); ++i) {
        const auto& child = m_nested[i];
        const auto begin = m_dims_to_encode_begin[i];
        auto in = input.slice_rows(begin, child->input_width());
        auto direction = dL_ddLdinput.slice_rows(begin, child->input_width());
        auto tangent = tangents.slice_rows(offset, child->padded_output_width());
        auto dy = m_reduction_type == ReductionType::Concatenation
                      ? dL_doutput.slice_rows(offset, child->padded_output_width())
                      : dL_doutput.slice_rows(0, dL_doutput.m());
        child->backward_backward_input(stream, *forward.nested[i], in, direction, dy, &tangent,
                                       nullptr, use_inference_params, GradientMode::Ignore);
        offset += child->padded_output_width();
    }
    if (dL_ddLdoutput) {
        const uint32_t mode = m_reduction_type == ReductionType::Concatenation ? 1
                              : m_reduction_type == ReductionType::Sum         ? 3
                                                                               : 8;
        encoding_detail::composite_utility<T>(stream, m_reduction_type, &tangents, *dL_ddLdoutput,
                                              mode, 0, &forward.to_reduce, nullptr,
                                              padded_output_width(), 0, uint32_t(m_nested.size()));
    }
    if (!dL_dinput && param_gradients_mode == GradientMode::Ignore) {
        batch.finish();
        return;
    }
    offset = 0;
    for (size_t i = 0; i < m_nested.size(); ++i) {
        const auto& child = m_nested[i];
        const auto begin = m_dims_to_encode_begin[i];
        auto in = input.slice_rows(begin, child->input_width());
        auto direction = dL_ddLdinput.slice_rows(begin, child->input_width());
        GPUMatrixDynamic<T> dy;
        if (m_reduction_type == ReductionType::Concatenation)
            dy = dL_doutput.slice_rows(offset, child->padded_output_width());
        else if (m_reduction_type == ReductionType::Sum)
            dy = dL_doutput.slice_rows(0, dL_doutput.m());
        else {
            dy = GPUMatrixDynamic<T>{child->padded_output_width(), input.n(), stream};
            encoding_detail::composite_utility<T>(
                stream, m_reduction_type, &dL_doutput, dy, 6, 0, &forward.to_reduce, nullptr,
                padded_output_width(), uint32_t(i), uint32_t(m_nested.size()));
        }
        GPUMatrixDynamic<float> dx;
        if (dL_dinput)
            dx = GPUMatrixDynamic<float>{child->input_width(), input.n(), stream};
        child->backward_backward_input(stream, *forward.nested[i], in, direction, dy, nullptr,
                                       dL_dinput ? &dx : nullptr, use_inference_params,
                                       param_gradients_mode);
        if (dL_dinput) {
            auto dst = dL_dinput->slice_rows(begin, child->input_width());
            encoding_detail::composite_utility<float>(stream, m_reduction_type, &dx, dst, 2);
        }
        if (m_reduction_type == ReductionType::Product) {
            GPUMatrixDynamic<T> output_adjoint{child->padded_output_width(), input.n(), stream};
            encoding_detail::product_second(stream, dL_doutput, forward.to_reduce, tangents,
                                            output_adjoint, uint32_t(i), uint32_t(m_nested.size()));
            auto child_output = forward.to_reduce.slice_rows(offset, child->padded_output_width());
            const auto mode = param_gradients_mode == GradientMode::Ignore
                                  ? GradientMode::Ignore
                                  : GradientMode::Accumulate;
            child->backward(stream, *forward.nested[i], in, child_output, output_adjoint,
                            dL_dinput ? &dx : nullptr, use_inference_params, mode);
            if (dL_dinput) {
                auto dst = dL_dinput->slice_rows(begin, child->input_width());
                encoding_detail::composite_utility<float>(stream, m_reduction_type, &dx, dst, 2);
            }
        }
        offset += child->padded_output_width();
    }
    batch.finish();
}
#endif // TCNN_NO_FWD_BWD

/// LoD supports the exact Hard/Discontinuous and Soft/Continuous names in the reference.
template <typename T>
MultiLevelEncodingLoD<T>::MultiLevelEncodingLoD(uint32_t n_dims_to_encode, const json& encoding) {
    check(n_dims_to_encode > 1, "MultiLevelEncodingLoD requires at least two input dimensions.");
    const auto mode = encoding_detail::lower(encoding.value("lod_type", "Hard"));
    check(mode == "hard" || mode == "discontinuous" || mode == "soft" || mode == "continuous",
          "Invalid LoD type.");
    m_is_soft = mode == "soft" || mode == "continuous";
    const auto& base = encoding.at("base");
    check(encoding_detail::lower(base.value("otype", "")) != "multilevelencodinglod",
          "LoD cannot wrap another LoD.");
    std::shared_ptr<Encoding<T>> child{create_encoding<T>(n_dims_to_encode - 1, base, 1)};
    m_base = std::dynamic_pointer_cast<MultiLevelEncoding<T>>(child);
    check(bool(m_base), "LoD requires a multi-level base encoding.");
    m_base->set_alignment(m_base->required_output_alignment());
}

/// Extracts limits in a compute command, so GPU-produced LoD values work on graph replay.
void record_levels(metalStream_t stream, const GPUMatrixDynamic<float>& input,
                   GPUMatrixDynamic<float>& output, uint32_t position_dims, float max_level,
                   const float* max_level_gpu, bool soft) {
    using namespace encoding_detail;
    UtilityArgs a{};
    a.src = describe(input);
    a.index = position_dims;
    a.has_limits = max_level_gpu ? 1 : 0;
    a.value = max_level;
    const auto descriptor = kernel_traits::lod_kernel(soft);
    const std::string name = std::string(descriptor.prefix) + "_levels";
    const auto device = metalrt::stream_or_default(detail::resolve_stream(stream)).device();
    metalrt::Kernel kernel;
    {
        std::lock_guard<std::mutex> lock{cache_mutex};
        auto it = std::find_if(caches.begin(), caches.end(),
                               [&](const auto& c) { return c.device == device; });
        if (it == caches.end()) {
            auto d = device.allocate(16);
            std::memset(d.contents(), 0, 16);
            caches.push_back({device, d, {}});
            it = std::prev(caches.end());
        }
        auto k = it->kernels.find(name);
        if (k == it->kernels.end())
            k = it->kernels
                    .emplace(name,
                             device.kernel(
                                 detail::shader_source(descriptor.source), name,
                                 [](const metalrt::ComputeCommand& c) {
                                     auto args = c.scalar<UtilityArgs>(3);
                                     auto* dst = c.buffer<float>(1);
                                     const auto* src = c.buffer<const float>(0);
                                     const auto* limits = c.buffer<const float>(2);
                                     for (uint32_t i = 0; i < args.src.cols; ++i)
                                         dst[i] = std::fmin(
                                             src[tcnn_encoding_math::at(args.src, args.index, i)],
                                             args.has_limits ? limits[i] : args.value);
                                 }))
                    .first;
        kernel = k->second;
    }
    if (input.n())
        metalrt::launch_linear(
            detail::resolve_stream(stream), kernel, input.n(),
            {matrix(stream, 0, input, metalrt::Access::Read),
             matrix(stream, 1, output, metalrt::Access::Write),
             raw(stream, 2, max_level_gpu, max_level_gpu ? input.n() : 0, metalrt::Access::Read),
             metalrt::Argument::scalar(3, a)});
}

#ifndef TCNN_NO_FWD_BWD
template <typename T>
std::unique_ptr<Context>
MultiLevelEncodingLoD<T>::forward_impl(metalStream_t stream, const GPUMatrixDynamic<float>& input,
                                       GPUMatrixDynamic<T>* output, bool use_inference_params,
                                       bool prepare_input_gradients) {
    // Join an outer batch/capture; otherwise submit this entire encoder call once.
    auto batch = detail::resolve_stream(stream)->batch();
    auto ctx = std::make_unique<ForwardContext>();
    ctx->owner = this;
    ctx->levels = GPUMatrixDynamic<float>{1, input.n(), stream};
    ctx->base_output =
        GPUMatrixDynamic<T>{padded_output_width(), input.n(), stream, preferred_output_layout()};
    record_levels(stream, input, ctx->levels, m_base->input_width(), this->m_max_level,
                  this->m_max_level_gpu, m_is_soft);
    auto positions = input.slice_rows(0, m_base->input_width());
    auto* previous = m_base->max_level_gpu();
    m_base->set_max_level_gpu(ctx->levels.data());
    ScopeGuard restore{[&] { m_base->set_max_level_gpu(previous); }};
    ctx->base = m_base->forward(stream, positions, &ctx->base_output, use_inference_params,
                                prepare_input_gradients);
    if (output)
        encoding_detail::lod_utility<T>(stream, m_is_soft, &ctx->base_output, *output,
                                        m_is_soft ? 7 : 1, 0, nullptr, ctx->levels.data(), 0, 0, 0,
                                        n_levels(), n_features_per_level());
    batch.finish();
    return ctx;
}
#endif // TCNN_NO_FWD_BWD

#ifndef TCNN_NO_FWD_BWD
template <typename T>
void MultiLevelEncodingLoD<T>::backward_impl(metalStream_t stream, const Context& ctx,
                                             const GPUMatrixDynamic<float>& input,
                                             const GPUMatrixDynamic<T>& output,
                                             const GPUMatrixDynamic<T>& dL_doutput,
                                             GPUMatrixDynamic<float>* dL_dinput,
                                             bool use_inference_params,
                                             GradientMode param_gradients_mode) {
    // Join an outer batch/capture; otherwise submit this entire encoder call once.
    auto batch = detail::resolve_stream(stream)->batch();
    (void)output;
    const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
    check(forward.owner == this && forward.levels.n() == input.n() && forward.base,
          "LoD context mismatch.");
    auto positions = input.slice_rows(0, m_base->input_width());
    GPUMatrixDynamic<T> dy{dL_doutput.m(), dL_doutput.n(), stream, preferred_output_layout()};
    encoding_detail::lod_utility<T>(stream, m_is_soft, &dL_doutput, dy, m_is_soft ? 7 : 1, 0,
                                    nullptr, forward.levels.data(), 0, 0, 0, n_levels(),
                                    n_features_per_level());
    GPUMatrixDynamic<float> dx;
    if (dL_dinput)
        dx = dL_dinput->slice_rows(0, m_base->input_width());
    auto* previous = m_base->max_level_gpu();
    m_base->set_max_level_gpu(forward.levels.data());
    ScopeGuard restore{[&] { m_base->set_max_level_gpu(previous); }};
    m_base->backward(stream, *forward.base, positions, forward.base_output, dy,
                     dL_dinput ? &dx : nullptr, use_inference_params, param_gradients_mode);
    if (dL_dinput) {
        auto lod = dL_dinput->slice_rows(m_base->input_width(), 1);
        encoding_detail::lod_utility<float>(stream, m_is_soft, nullptr, lod, 0);
    }
    batch.finish();
}
#endif // TCNN_NO_FWD_BWD

#ifndef TCNN_NO_FWD_BWD
template <typename T>
void MultiLevelEncodingLoD<T>::backward_backward_input_impl(
    metalStream_t stream, const Context& ctx, const GPUMatrixDynamic<float>& input,
    const GPUMatrixDynamic<float>& dL_ddLdinput, const GPUMatrixDynamic<T>& dL_doutput,
    GPUMatrixDynamic<T>* dL_ddLdoutput, GPUMatrixDynamic<float>* dL_dinput,
    bool use_inference_params, GradientMode param_gradients_mode) {
    // Join an outer batch/capture; otherwise submit this entire encoder call once.
    auto batch = detail::resolve_stream(stream)->batch();
    const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
    check(forward.owner == this && forward.levels.n() == input.n() && forward.base,
          "LoD context mismatch.");
    auto positions = input.slice_rows(0, m_base->input_width());
    auto ddpositions = dL_ddLdinput.slice_rows(0, m_base->input_width());
    GPUMatrixDynamic<T> dy{dL_doutput.m(), dL_doutput.n(), stream, preferred_output_layout()};
    encoding_detail::lod_utility<T>(stream, m_is_soft, &dL_doutput, dy, m_is_soft ? 7 : 1, 0,
                                    nullptr, forward.levels.data(), 0, 0, 0, n_levels(),
                                    n_features_per_level());
    GPUMatrixDynamic<float> dx;
    if (dL_dinput)
        dx = dL_dinput->slice_rows(0, m_base->input_width());
    auto* previous = m_base->max_level_gpu();
    m_base->set_max_level_gpu(forward.levels.data());
    ScopeGuard restore{[&] { m_base->set_max_level_gpu(previous); }};
    m_base->backward_backward_input(stream, *forward.base, positions, ddpositions, dy,
                                    dL_ddLdoutput, dL_dinput ? &dx : nullptr, use_inference_params,
                                    param_gradients_mode);
    if (m_is_soft && dL_ddLdoutput)
        encoding_detail::lod_utility<T>(stream, m_is_soft, dL_ddLdoutput, *dL_ddLdoutput, 7, 0,
                                        nullptr, forward.levels.data(), 0, 0, 0, n_levels(),
                                        n_features_per_level());
    if (dL_dinput) {
        auto lod = dL_dinput->slice_rows(m_base->input_width(), 1);
        encoding_detail::lod_utility<float>(stream, m_is_soft, nullptr, lod, 0);
    }
    batch.finish();
}
#endif // TCNN_NO_FWD_BWD

/// The public JSON factory keeps the upload's enabled-hash switch; direct templates are separate.
template <typename T>
MultiLevelEncoding<T>* create_grid_encoding(uint32_t n_dims_to_encode, const json& encoding) {
    const HashType hash_type = string_to_hash_type(encoding.value("hash", "CoherentPrime"));
    if (hash_type == HashType::CoherentPrime)
        return create_grid_encoding_templated_1<T, HashType::CoherentPrime>(n_dims_to_encode,
                                                                            encoding);
    throw std::runtime_error("GridEncoding: compiled without " + to_string(hash_type) +
                             " hash support.");
}

template <typename T, uint32_t D, uint32_t F>
MultiLevelEncoding<T>* make_permuto_checked(const json& config) {
    encoding_detail::LatticeEncoding<T> validated{D, config, true};
    const auto h = validated.hyperparams();
    return new PermutoEncodingTemplated<T, D, F>{
        validated.n_levels() * F,
        h.at("log2_hashmap_size").template get<uint32_t>(),
        h.at("base_scale").template get<float>(),
        h.at("per_level_scale").template get<float>(),
        h.at("max_input_grad_dims").template get<uint32_t>(),
        h.at("seed").template get<uint32_t>()};
}

template <typename T, uint32_t F>
MultiLevelEncoding<T>* permuto_dimensions(uint32_t n_dims_to_encode, const json& encoding) {
#define PERMUTO_DIM(D)                                                                             \
    case D:                                                                                        \
        return make_permuto_checked<T, D, F>(encoding)
    switch (n_dims_to_encode) {
        PERMUTO_DIM(1);
        PERMUTO_DIM(2);
        PERMUTO_DIM(3);
        PERMUTO_DIM(4);
        PERMUTO_DIM(5);
        PERMUTO_DIM(6);
        PERMUTO_DIM(7);
        PERMUTO_DIM(8);
        PERMUTO_DIM(9);
        PERMUTO_DIM(10);
        PERMUTO_DIM(12);
        PERMUTO_DIM(16);
        PERMUTO_DIM(24);
    default:
        throw std::runtime_error("Unsupported Permuto dimensionality.");
    }
#undef PERMUTO_DIM
}

template <typename T>
MultiLevelEncoding<T>* create_permuto_encoding(uint32_t n_dims_to_encode, const json& encoding) {
    switch (encoding_detail::integer(encoding, "n_features_per_level", 2)) {
    case 1:
        return permuto_dimensions<T, 1>(n_dims_to_encode, encoding);
    case 2:
        return permuto_dimensions<T, 2>(n_dims_to_encode, encoding);
    case 4:
        return permuto_dimensions<T, 4>(n_dims_to_encode, encoding);
    case 8:
        return permuto_dimensions<T, 8>(n_dims_to_encode, encoding);
    default:
        throw std::runtime_error("Permuto: features per level must be 1, 2, 4, or 8.");
    }
}

namespace encoding_detail {
template <typename T> using Factory = std::function<Encoding<T>*(uint32_t, const json&)>;

template <typename T> struct Registry {
    std::mutex mutex;
    std::map<std::string, std::pair<std::string, Factory<T>>> factories;
};

/// Registry initialization follows the names, aliases, defaults, and case folding in encoding.cu.
template <typename T> Registry<T>& registry() {
    static Registry<T> result;
    static const bool initialized = [] {
        auto add = [&](const std::string& name, Factory<T> factory) {
            result.factories.emplace(lower(name), std::make_pair(name, std::move(factory)));
        };
        add("Composite", [](uint32_t d, const json& c) { return new CompositeEncoding<T>{c, d}; });
        add("Empty", [](uint32_t d, const json&) { return new EmptyEncoding<T>{d}; });
        add("Frequency", [](uint32_t d, const json& c) {
            return new FrequencyEncoding<T>{integer(c, "n_frequencies", 12), d};
        });
        for (const char* name : {"Grid", "HashGrid", "TiledGrid", "DenseGrid"})
            add(name, [](uint32_t d, const json& c) { return create_grid_encoding<T>(d, c); });
        add("Permuto", [](uint32_t d, const json& c) { return create_permuto_encoding<T>(d, c); });
        add("MultiLevelEncodingLoD",
            [](uint32_t d, const json& c) { return new MultiLevelEncodingLoD<T>{d, c}; });
        add("Identity", [](uint32_t d, const json& c) {
            return new IdentityEncoding<T>{d, c.value("scale", 1.0f), c.value("offset", 0.0f)};
        });
        add("OneBlob", [](uint32_t d, const json& c) {
            return new OneBlobEncoding<T>{integer(c, "n_bins", 16), d};
        });
        add("SphericalHarmonics", [](uint32_t d, const json& c) {
            return new SphericalHarmonicsEncoding<T>{integer(c, "degree", 4), d};
        });
        add("TriangleWave", [](uint32_t d, const json& c) {
            return new TriangleWaveEncoding<T>{integer(c, "n_frequencies", 12), d};
        });
        auto nrc = [](uint32_t d, const json& c) -> Encoding<T>* {
            return new CompositeEncoding<T>{{{"otype", "Composite"},
                                             {"nested",
                                              {{{"n_dims_to_encode", 3},
                                                {"otype", "TriangleWave"},
                                                {"n_frequencies", integer(c, "n_frequencies", 12)}},
                                               {{"n_dims_to_encode", 5},
                                                {"otype", "OneBlob"},
                                                {"n_bins", integer(c, "n_bins", 4)}},
                                               {{"otype", "Identity"}}}}},
                                            d};
        };
        add("OneBlobFrequency", nrc);
        add("NRC", nrc);
        return true;
    }();
    (void)initialized;
    return result;
}
} // namespace encoding_detail

template <typename T>
void register_encoding(const std::string& name,
                       const std::function<Encoding<T>*(uint32_t, const json&)>& factory) {
    check(bool(factory), "Encoding factory cannot be empty.");
    auto& r = encoding_detail::registry<T>();
    std::lock_guard<std::mutex> lock{r.mutex};
    const auto key = encoding_detail::lower(name);
    check(!r.factories.count(key), "Cannot register an encoding name twice.");
    r.factories.emplace(key, std::make_pair(name, factory));
}

template <typename T>
Encoding<T>* create_encoding(uint32_t n_dims_to_encode, const json& params, uint32_t alignment) {
    check(params.is_object(), "Encoding configuration must be an object.");
    const auto name = params.value("otype", "OneBlob");
    encoding_detail::Factory<T> factory;
    {
        auto& r = encoding_detail::registry<T>();
        std::lock_guard<std::mutex> lock{r.mutex};
        const auto it = r.factories.find(encoding_detail::lower(name));
        if (it == r.factories.end())
            throw std::runtime_error("Encoding '" + name + "' not found.");
        factory = it->second.second;
    }
    std::unique_ptr<Encoding<T>> result{factory(n_dims_to_encode, params)};
    check(bool(result), "Encoding factory returned nullptr.");
    if (alignment > 0)
        result->set_alignment(alignment);
    return result.release();
}

std::vector<std::string> builtin_encodings() {
    auto& r = encoding_detail::registry<float>();
    std::lock_guard<std::mutex> lock{r.mutex};
    std::vector<std::string> names;
    for (const auto& entry : r.factories)
        names.push_back(entry.second.first);
    return names;
}

#define INSTANTIATE_ENCODINGS(T)                                                                   \
    template class encoding_detail::AnalyticEncoding<T>;                                           \
    template class encoding_detail::LatticeEncoding<T>;                                            \
    template class CompositeEncoding<T>;                                                           \
    template class MultiLevelEncodingLoD<T>;                                                       \
    template Encoding<T>* create_encoding<T>(uint32_t, const json&, uint32_t);                     \
    template void register_encoding<T>(const std::string&,                                         \
                                       const std::function<Encoding<T>*(uint32_t, const json&)>&); \
    template MultiLevelEncoding<T>* create_grid_encoding<T>(uint32_t, const json&);                \
    template MultiLevelEncoding<T>* create_permuto_encoding<T>(uint32_t, const json&);
INSTANTIATE_ENCODINGS(float)
INSTANTIATE_ENCODINGS(half)
#undef INSTANTIATE_ENCODINGS
} // namespace tcnn
