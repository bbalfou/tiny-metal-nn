// Canonical native random kernels and a deterministic reference implementation.
#include <tiny-metal-nn/random.h>
#include <tiny-metal-nn/detail/shaders.h>
#include <cstddef>
#include <mutex>

namespace tcnn {
namespace detail {
namespace {
struct RandomArgs {
    uint64_t state, inc;
    uint32_t count, threads, fp16, kind;
    float first, second;
};

static_assert(sizeof(RandomArgs) == 40, "random constant ABI");
static_assert(alignof(RandomArgs) == 8 && offsetof(RandomArgs, count) == 16 &&
                  offsetof(RandomArgs, first) == 32,
              "random ulong/scalar offsets must match Metal");

// Cache independently for every runtime device; compile only once per device.
metalrt::Kernel random_kernel(metalrt::Device device, bool blocked_uniform) {
    struct Entry {
        metalrt::Device device;
        bool blocked_uniform;
        metalrt::Kernel kernel;
    };

    static std::vector<Entry> cache;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock{mutex};
    for (const auto& entry : cache)
        if (entry.device == device && entry.blocked_uniform == blocked_uniform)
            return entry.kernel;
    // The Reference backend deliberately retains the independent scalar PCG
    // algorithm for both native entrypoints.
    auto kernel = device.kernel(
        shader_source("random.metal"),
        blocked_uniform ? "tcnn_random_uniform_f32" : "tcnn_random_pcg",
        [](const metalrt::ComputeCommand& command) {
            const auto a = command.scalar<RandomArgs>(1);
            for (uint32_t i = 0; i < a.threads; ++i) {
                pcg32 rng;
                rng.state = a.state;
                rng.inc = a.inc;
                rng.advance(uint64_t(i) * 4);
                for (uint32_t j = 0; j < 4; ++j) {
                    const uint32_t idx = i + a.threads * j;
                    if (idx >= a.count)
                        break;
                    float value = rng.nextFloat();
                    if (a.fp16)
                        value = float(half(value));
                    const float transformed =
                        a.kind
                            ? (-std::log(1.0f / std::fmin(std::fmax(value, 1e-9f), 1.0f - 1e-9f) -
                                         1.0f)) *
                                      a.second * 0.551328895f +
                                  a.first
                            : value * (a.second - a.first) + a.first;
                    if (a.fp16)
                        command.buffer<half>(0)[idx] = half(transformed);
                    else
                        command.buffer<float>(0)[idx] = transformed;
                }
            }
        });
    cache.push_back({device, blocked_uniform, kernel});
    return kernel;
}
} // namespace

void random_pcg(metalStream_t stream, pcg32& rng, size_t n_elements, void* out, bool fp16,
                uint32_t kind, float first, float second) {
    if (n_elements > UINT32_MAX)
        throw std::overflow_error("random output exceeds 32-bit indexing");
    if (!n_elements)
        return;
    auto* queue = resolve_stream(stream);
    const uint32_t threads = next_multiple(
        div_round_up(static_cast<uint32_t>(n_elements), uint32_t{4}), N_THREADS_LINEAR);
    const RandomArgs args{rng.state, rng.inc,        static_cast<uint32_t>(n_elements),
                          threads,   uint32_t(fp16), kind,
                          first,     second};
    const auto output =
        metalrt::resolve_pointer(out, metalrt::checked_mul(n_elements, fp16 ? 2 : 4));
    // Four adjacent logical PCG streams share one jump and use float4 stores.
    // The measured crossover applies to FP32 uniform generation; half,
    // logistic, small outputs and unaligned suballocations keep the general path.
    const bool blocked_uniform =
        !fp16 && kind == 0 && n_elements >= 524288 && output.offset % 16 == 0;
    metalrt::launch_linear(queue, random_kernel(queue->device(), blocked_uniform),
                           blocked_uniform ? threads / 4 : threads,
                           {metalrt::Argument::buffer(0, output, metalrt::Access::Write),
                            metalrt::Argument::scalar(1, args)},
                           N_THREADS_LINEAR);
    rng.advance(static_cast<int64_t>(n_elements));
}
} // namespace detail
} // namespace tcnn

namespace tcnn {
// Source assembly is a host operation; the MSL text remains in its own asset.
std::string generate_device_code_preamble() {
    return detail::shader_source("device_preamble.metal");
}

// Query and cache the actual compiler version for each selected native device.
// A temporary Direct queue makes the first query an explicit completion boundary.
int metal_runtime_version() {
    auto device = metalrt::Device::current();
    if (device.capabilities().backend == metalrt::BackendKind::Reference)
        return 0;

    struct Entry {
        metalrt::Device device;
        int version;
    };

    static std::vector<Entry> cache;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock{mutex};
    for (const auto& entry : cache)
        if (entry.device == device)
            return entry.version;
    auto result = device.allocate(sizeof(uint32_t));
    auto kernel =
        device.kernel(detail::shader_source("device_queries.metal"), "tcnn_query_metal_version");
    metalrt::Stream queue{device, metalrt::DispatchPolicy::Direct};
    metalrt::launch_linear(&queue, kernel, 1,
                           {metalrt::Argument::buffer(0, result.view(), metalrt::Access::Write)},
                           1);
    queue.synchronize();
    const int version = static_cast<int>(*result.data<uint32_t>());
    if (version <= 0)
        throw std::runtime_error("native Metal compiler returned an invalid language version");
    cache.push_back({device, version});
    return version;
}

// Format the selected MSL major/minor version without claiming a CUDA-style SDK version.
std::string metal_runtime_version_string() {
    const int version = metal_runtime_version();
    if (!version)
        return "Reference (no Metal compiler)";
    return std::to_string(version / 100) + "." + std::to_string((version % 100) / 10);
}
} // namespace tcnn
