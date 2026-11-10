// Native reduction orchestration through metalrt, with a separate Reference oracle.
#include <tiny-metal-nn/reduce_sum.h>
#include <tiny-metal-nn/detail/reduction.h>
#include <tiny-metal-nn/detail/shaders.h>
#include <algorithm>
#include <array>
#include <mutex>

namespace tcnn {
namespace detail {
namespace {
enum class SumPass : uint32_t { Float = 0, Half = 1, Double = 2, Partials = 3 };

// The explicit Reference backend validates the same staged resource graph.
// Widened scalar accumulation is independent of the GPU's SIMD reduction tree;
// it is not a CPU fallback used by the native Metal backend.
metalrt::CpuKernel sum_reference(SumPass pass) {
    return [pass](const metalrt::ComputeCommand& command) {
        const auto args = command.scalar<reduction::Args>(2);
        const uint32_t width = pass == SumPass::Half     ? 8
                               : pass == SumPass::Float  ? 4
                               : pass == SumPass::Double ? 2
                                                         : 1;
        const auto* input = command.buffer<const uint8_t>(0);
        auto* output = command.buffer<float>(1);
        const uint32_t lanes = static_cast<uint32_t>(command.group.x);
        const uint32_t count = args.count / width;
        const uint32_t stride = args.groups * lanes;
        for (uint32_t group = 0; group < args.groups; ++group) {
            double sum = 0;
            for (uint32_t first = group * lanes; first < count; first += stride) {
                const uint32_t end = std::min(first + lanes, count);
                for (uint32_t pack = first; pack < end; ++pack)
                    for (uint32_t component = 0; component < width; ++component) {
                        const size_t index = size_t(pack) * width + component;
                        sum +=
                            pass == SumPass::Half
                                ? float(reinterpret_cast<const half*>(input)[index])
                            : pass == SumPass::Double
                                ? static_cast<float>(reinterpret_cast<const double*>(input)[index])
                                : reinterpret_cast<const float*>(input)[index];
                    }
            }
            if (args.accumulate)
                output[group] += static_cast<float>(sum);
            else
                output[group] = static_cast<float>(sum);
        }
    };
}

metalrt::Kernel sum_kernel(const metalrt::Device& device, SumPass pass) {
    struct Cache {
        metalrt::Device device;
        std::array<metalrt::Kernel, 4> kernels;
    };
    static std::mutex mutex;
    static std::vector<Cache> caches;
    static constexpr const char* entries[] = {"nn_reduce_sum_f32", "nn_reduce_sum_f16",
                                              "nn_reduce_sum_f64", "nn_reduce_sum_partials"};
    std::lock_guard<std::mutex> lock{mutex};
    auto found = std::find_if(caches.begin(), caches.end(),
                              [&](const auto& item) { return item.device == device; });
    if (found == caches.end()) {
        caches.push_back({device, {}});
        found = std::prev(caches.end());
    }
    auto& kernel = found->kernels[static_cast<uint32_t>(pass)];
    if (!kernel)
        kernel = device.kernel(shader_source("reduce_sum.metal"),
                               entries[static_cast<uint32_t>(pass)], sum_reference(pass));
    return kernel;
}

void sum_pass(metalrt::Stream& runtime, metalrt::Kernel kernel, metalrt::BufferView input,
              metalrt::BufferView output, uint32_t count, uint32_t groups, bool accumulate) {
    // 256 threads fits the supported Metal compute limits. All groups are full;
    // the shader uses the actual SIMD width and masks input tails before barriers.
    runtime.dispatch(
        std::move(kernel), {size_t(groups) * reduction::threads, 1, 1}, {reduction::threads, 1, 1},
        {metalrt::Argument::buffer(0, std::move(input), metalrt::Access::Read),
         metalrt::Argument::buffer(1, std::move(output),
                                   accumulate ? metalrt::Access::ReadWrite
                                              : metalrt::Access::Write),
         metalrt::Argument::scalar(2, reduction::Args{count, groups, accumulate ? 1u : 0u, 0})});
}
} // namespace

void reduce_sum_dispatch(metalStream_t stream, const void* input, uint32_t input_bytes,
                         float* workspace, uint32_t n_elements) {
    if (!n_elements)
        return;
    nn_check(input_bytes == 2 || input_bytes == 4 || input_bytes == 8,
             "invalid reduction storage type");
    auto& runtime = *resolve_stream(stream);
    const auto plan = reduction::plan(n_elements, input_bytes);
    const auto source =
        metalrt::resolve_pointer(input, metalrt::checked_mul(size_t(n_elements), input_bytes));
    const auto destination = metalrt::resolve_pointer(workspace, sizeof(float));
    const auto input_pass = input_bytes == 2   ? SumPass::Half
                            : input_bytes == 8 ? SumPass::Double
                                               : SumPass::Float;
    const auto first_kernel = sum_kernel(runtime.device(), input_pass);
    const auto partial_kernel =
        plan.first_groups > 1 ? sum_kernel(runtime.device(), SumPass::Partials) : metalrt::Kernel{};
    auto scratch = allocate_workspace(stream, size_t(plan.scratch_elements) * sizeof(float));
    // Compile and resolve all resources before recording anything. A transaction
    // also cancels the full staged suffix if runtime validation rejects a pass.
    auto transaction = runtime.transaction();
    if (plan.first_groups == 1) {
        sum_pass(runtime, first_kernel, source, destination, n_elements, 1, true);
    } else {
        auto first = scratch.buffer_view().slice(0, size_t(plan.first_groups) * sizeof(float));
        auto second =
            plan.second_capacity
                ? scratch.buffer_view().slice(size_t(plan.first_groups) * sizeof(float),
                                              size_t(plan.second_capacity) * sizeof(float))
                : metalrt::BufferView{};
        sum_pass(runtime, first_kernel, source, first, n_elements, plan.first_groups, false);
        uint32_t count = plan.first_groups;
        auto current = first;
        auto next = second;
        while (count > 1) {
            const uint32_t groups = reduction::groups_for(count, 1);
            sum_pass(runtime, partial_kernel, current.slice(0, size_t(count) * sizeof(float)),
                     groups == 1 ? destination : next.slice(0, size_t(groups) * sizeof(float)),
                     count, groups, groups == 1);
            count = groups;
            std::swap(current, next);
        }
    }
    // Each command owns its BufferViews, including scratch. Captured/replayed
    // commands therefore keep allocations alive after this pool lease expires.
    transaction.finish();
}
} // namespace detail

uint32_t reduce_sum_workspace_size(uint32_t n_elements) {
    // Preserve the upstream public calculation. The asynchronous overload only
    // consumes workspace[0]; bounded intermediate storage belongs to the runtime.
    return n_blocks_linear(n_elements);
}
} // namespace tcnn
