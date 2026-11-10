// Automatic JIT gradient accumulation uses native FP32 storage and a bounded
// striped prefix for dense weights. The caller's canonical parameter ABI stays
// unchanged; this file owns only reduction planning, storage and finalization.
#include <tiny-metal-nn/detail/jit.h>
#include <tiny-metal-nn/detail/shaders.h>
#include <tiny-metal-nn/metal.h>
#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>

namespace tcnn {
namespace detail {
namespace {
constexpr size_t max_striped_bytes = 8u * 1024u * 1024u;
constexpr uint32_t max_partitions = 64u;

uint32_t dense_requirement(const std::string& source) {
    const std::string marker = "// TCNN_JIT_DENSE_GRADIENT_PARAMETERS:";
    uint32_t maximum = 0;
    size_t position = 0;
    while ((position = source.find(marker, position)) != std::string::npos) {
        position += marker.size();
        while (position < source.size() && (source[position] == ' ' || source[position] == '\t'))
            ++position;
        const auto begin = position;
        uint32_t value = 0;
        while (position < source.size() && source[position] >= '0' && source[position] <= '9') {
            const auto digit = static_cast<uint32_t>(source[position++] - '0');
            if (value > (UINT32_MAX - digit) / 10u)
                throw std::invalid_argument("Generated JIT dense gradient count exceeds uint32_t");
            value = value * 10u + digit;
        }
        if (position == begin || (position < source.size() &&
                                  !std::isspace(static_cast<unsigned char>(source[position]))))
            throw std::invalid_argument("Invalid generated JIT dense gradient annotation");
        maximum = std::max(maximum, value);
    }
    return maximum;
}

struct GradientArgs {
    uint32_t parameters, dense_parameters, partitions, reserved;
};
static_assert(sizeof(GradientArgs) == 16, "JIT gradient constants must match the native shader");

// This oracle is used only by the explicitly selected Reference backend.
// Metal always compiles and executes the shader, including for offset views.
template <typename T> metalrt::CpuKernel gradient_reference() {
    return [](const metalrt::ComputeCommand& command) {
        const auto args = command.scalar<GradientArgs>(3);
        const auto* typed = command.buffer<const T>(0);
        const auto* wide = command.buffer<const float>(1);
        auto* destination = command.buffer<T>(2);
        for (size_t i = 0; i < args.parameters; ++i) {
            double contribution = 0;
            if (i < args.dense_parameters) {
                for (uint32_t partition = 0; partition < args.partitions; ++partition)
                    contribution += wide[size_t(args.parameters) +
                                         size_t(partition) * args.dense_parameters + i];
            } else {
                contribution = wide[i];
            }
            // Accumulate's prior value and direct typed writes by extensions
            // enter exactly once, after the widened reduction has completed.
            destination[i] = T(float(typed[i]) + static_cast<float>(contribution));
        }
    };
}
} // namespace

size_t JitGradientPlan::elements() const {
    if (dense_parameters > parameters || partitions > max_partitions ||
        (partitions && (partitions & (partitions - 1u))) ||
        (dense_parameters == 0u) != (partitions == 0u))
        throw std::invalid_argument("Invalid JIT gradient reduction plan");
    const auto extra = metalrt::checked_mul(size_t(dense_parameters), size_t(partitions));
    if (metalrt::checked_mul(extra, sizeof(float)) > max_striped_bytes)
        throw std::invalid_argument("JIT dense gradient scratch exceeds its bounded budget");
    return metalrt::checked_add(size_t(parameters), extra);
}

JitGradientPlan jit_gradient_plan(const std::string& source, size_t parameters) {
    if (parameters > UINT32_MAX)
        throw std::invalid_argument("JIT parameter count exceeds uint32_t");
    JitGradientPlan plan;
    plan.parameters = static_cast<uint32_t>(parameters);
    plan.dense_parameters = std::min(dense_requirement(source), plan.parameters);
    const auto capacity = max_striped_bytes / sizeof(float);
    if (plan.dense_parameters && plan.dense_parameters <= capacity) {
        plan.partitions = 1u;
        while (plan.partitions < max_partitions &&
               size_t(plan.partitions) * 2u <= capacity / plan.dense_parameters)
            plan.partitions *= 2u;
    } else {
        // Large dense networks remain supported. If even one whole prefix
        // would exceed the budget, atomics use the ordinary widened base.
        plan.dense_parameters = 0u;
    }
    (void)plan.elements();
    return plan;
}

GPUMemoryArena::Allocation jit_prepare_float_gradients(metalStream_t stream,
                                                       const JitGradientPlan& plan) {
    const auto bytes = metalrt::checked_mul(plan.elements(), sizeof(float));
    if (!bytes)
        return {};
    auto scratch = allocate_workspace(stream, bytes);
    METAL_CHECK_THROW(metalMemsetAsync(scratch.data(), 0, bytes, stream));
    return scratch;
}

metalrt::Argument jit_float_gradient_argument(uint32_t slot,
                                              const GPUMemoryArena::Allocation& wide) {
    return wide ? metalrt::Argument::buffer(slot, wide.buffer_view(), metalrt::Access::ReadWrite)
                : metalrt::Argument::null_buffer(slot);
}

void jit_finish_float_gradients(metalStream_t stream, void* gradients,
                                const GPUMemoryArena::Allocation& typed,
                                const GPUMemoryArena::Allocation& wide, const JitGradientPlan& plan,
                                size_t element_size) {
    const auto wide_bytes = metalrt::checked_mul(plan.elements(), sizeof(float));
    if (!plan.parameters)
        return;
    if (element_size != sizeof(half) && element_size != sizeof(float))
        throw std::invalid_argument("JIT gradients require native half or float storage");
    const auto typed_bytes = metalrt::checked_mul(size_t(plan.parameters), element_size);
    if (typed.size() < typed_bytes || wide.size() < wide_bytes)
        throw std::invalid_argument("JIT gradient finalization scratch is too small");
    auto& queue = *resolve_stream(stream);
    auto destination = metalrt::resolve_pointer(gradients, typed_bytes);
    const bool is_half = element_size == sizeof(half);
    const auto kernel = queue.device().kernel_cached(
        shader_source("jit/gradients.metal"),
        is_half ? "tcnn_jit_finish_gradients_f16" : "tcnn_jit_finish_gradients_f32",
        rtc_cache_dir(), is_half ? gradient_reference<half>() : gradient_reference<float>());
    metalrt::launch_linear(
        &queue, kernel, (size_t(plan.parameters) + 3u) / 4u,
        {metalrt::Argument::buffer(0, typed.buffer_view().slice(0, typed_bytes),
                                   metalrt::Access::Read),
         metalrt::Argument::buffer(1, wide.buffer_view().slice(0, wide_bytes),
                                   metalrt::Access::Read),
         metalrt::Argument::buffer(2, std::move(destination), metalrt::Access::Write),
         metalrt::Argument::scalar(
             3, GradientArgs{plan.parameters, plan.dense_parameters, plan.partitions, 0u})});
    // Every enqueued command retains its input/output BufferViews. Scratch
    // leases may expire immediately and captured graphs remain replayable.
}
} // namespace detail
} // namespace tcnn
