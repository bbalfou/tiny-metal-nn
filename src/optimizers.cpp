#include <tiny-metal-nn/optimizer.h>
#include <tiny-metal-nn/optimizers/sgd.h>
#include <tiny-metal-nn/optimizers/adam.h>
#include <tiny-metal-nn/optimizers/average.h>
#include <tiny-metal-nn/optimizers/batched.h>
#include <tiny-metal-nn/optimizers/composite.h>
#include <tiny-metal-nn/optimizers/ema.h>
#include <tiny-metal-nn/optimizers/exponential_decay.h>
#include <tiny-metal-nn/optimizers/lookahead.h>
#include <tiny-metal-nn/optimizers/novograd.h>
#include <tiny-metal-nn/detail/shaders.h>
#include <tiny-metal-nn/detail/jit.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>

namespace tcnn {
namespace detail {
namespace {
enum class OptimizerOperation : uint32_t {
    SGD,
    Adam,
    EmaFull,
    EmaStorage,
    Average,
    Lookahead,
    GradientUpdate,
    Cast,
    CastFrom,
    NovoMoment,
    Novo
};

struct OptimizerArgs {
    uint32_t n_elements = 0, operation = 0, n_matrix_weights = 0, flags = 0;
    float loss_scale = 1, learning_rate = 0, l2_reg = 0, beta1 = 0, beta2 = 0, epsilon = 0;
    float relative_weight_decay = 0, absolute_weight_decay = 0, weight_clipping_magnitude = 0,
          gradient_clipping_magnitude = 0;
    float non_matrix_learning_rate_factor = 1, non_matrix_l2_reg = 0, lower_lr_bound = 0,
          upper_lr_bound = std::numeric_limits<float>::max();
    float ema_decay = 0, ema_debias_old = 0, ema_debias_new = 0, alpha = 0;
    uint32_t n_samples = 0, batch_size_multiplier = 0;
};

static_assert(sizeof(OptimizerArgs) == 96, "optimizer kernel constant ABI");

// Applies reference relative (L2-style) and absolute (L1-style) weight decay.
float optimizer_weight_decay(float relative, float absolute, float weight) {
    return (1.0f - relative) * weight - std::copysign(absolute, weight);
}

// Reference-only scalar execution; native submissions run the matching external Metal kernel.
template <typename T> metalrt::CpuKernel optimizer_reference() {
    return [](const metalrt::ComputeCommand& command) {
        const auto args = command.scalar<OptimizerArgs>(8);
        auto* full = command.buffer<float>(0);
        auto* weights = command.buffer<T>(1);
        const auto* gradients = command.buffer<const T>(2);
        auto* first = command.buffer<float>(3);
        auto* second = command.buffer<float>(4);
        auto* param_steps = command.buffer<uint32_t>(5);
        auto* auxiliary = command.buffer<T>(6);
        auto* samples = command.buffer<T>(7);
        const auto operation = static_cast<OptimizerOperation>(args.operation);
        if (operation == OptimizerOperation::NovoMoment) {
            float norm = 0;
            for (uint32_t i = 0; i < args.n_elements; ++i) {
                const float gradient = float(gradients[i]);
                norm += gradient * gradient;
            }
            first[0] = args.beta2 * first[0] +
                       (1.0f - args.beta2) * norm / args.loss_scale / args.loss_scale;
            return;
        }
        for (uint32_t i = 0; i < args.n_elements; ++i) {
            switch (operation) {
            case OptimizerOperation::SGD: {
                const float weight_fp = full[i];
                const float gradient =
                    float(gradients[i]) / args.loss_scale + args.l2_reg * weight_fp;
                const float value = weight_fp - args.learning_rate * gradient;
                full[i] = value;
                weights[i] = T(value);
                break;
            }
            case OptimizerOperation::Adam: {
                float gradient = float(gradients[i]) / args.loss_scale;
                const bool matrix = i < args.n_matrix_weights;
                if (matrix ? !(args.flags & 1u)
                           : (!(args.flags & 2u) || (gradient == 0 && (args.flags & 4u))))
                    break;
                const float weight_fp = full[i];
                gradient += (matrix ? args.l2_reg : args.non_matrix_l2_reg) * weight_fp;
                if (args.gradient_clipping_magnitude != 0)
                    gradient = std::copysign(
                        std::min(std::abs(gradient), args.gradient_clipping_magnitude), gradient);
                const float first_moment = first[i] =
                    args.beta1 * first[i] + (1.0f - args.beta1) * gradient;
                const float second_moment = second[i] =
                    args.beta2 * second[i] + (1.0f - args.beta2) * gradient * gradient;
                float learning_rate =
                    args.learning_rate * (matrix ? 1.0f : args.non_matrix_learning_rate_factor);
                const auto current_step = ++param_steps[i];
                learning_rate *= std::sqrt(1.0f - std::pow(args.beta2, float(current_step))) /
                                 (1.0f - std::pow(args.beta1, float(current_step)));
                const float effective =
                    std::fmin(std::fmax(learning_rate / (std::sqrt(second_moment) + args.epsilon),
                                        args.lower_lr_bound),
                              args.upper_lr_bound);
                float value =
                    optimizer_weight_decay(args.relative_weight_decay * learning_rate,
                                           args.absolute_weight_decay * learning_rate, weight_fp) -
                    effective * first_moment;
                if (args.weight_clipping_magnitude != 0)
                    value = std::clamp(value, -args.weight_clipping_magnitude,
                                       args.weight_clipping_magnitude);
                full[i] = value;
                weights[i] = T(value);
                break;
            }
            case OptimizerOperation::EmaFull:
            case OptimizerOperation::EmaStorage: {
                const float old =
                    operation == OptimizerOperation::EmaFull ? first[i] : float(auxiliary[i]);
                const float value = (old * args.ema_decay * args.ema_debias_old +
                                     float(weights[i]) * (1.0f - args.ema_decay)) *
                                    args.ema_debias_new;
                if (operation == OptimizerOperation::EmaFull)
                    first[i] = value;
                auxiliary[i] = T(value);
                break;
            }
            case OptimizerOperation::Average: {
                const T weight = weights[i];
                auxiliary[i] =
                    T(float(auxiliary[i]) + (float(weight) - float(samples[i])) / args.n_samples);
                samples[i] = weight;
                break;
            }
            case OptimizerOperation::Lookahead: {
                const float value =
                    float(auxiliary[i]) * (1.0f - args.alpha) + full[i] * args.alpha;
                full[i] = value;
                weights[i] = auxiliary[i] = T(value);
                break;
            }
            case OptimizerOperation::GradientUpdate:
                if (args.flags & 1u)
                    first[i] = 0;
                first[i] += float(gradients[i]) / args.batch_size_multiplier;
                break;
            case OptimizerOperation::Cast:
                auxiliary[i] = T(full[i]);
                break;
            case OptimizerOperation::CastFrom:
                full[i] = float(weights[i]);
                break;
            case OptimizerOperation::Novo: {
                const float weight_fp = full[i], gradient = float(gradients[i]) / args.loss_scale;
                const float first_moment = first[i] =
                    args.beta1 * first[i] +
                    (1.0f - args.beta1) * gradient / (std::sqrt(second[0]) + args.epsilon);
                const float value =
                    optimizer_weight_decay(args.relative_weight_decay * args.learning_rate,
                                           args.absolute_weight_decay * args.learning_rate,
                                           weight_fp) -
                    args.learning_rate * first_moment;
                full[i] = value;
                weights[i] = T(value);
                break;
            }
            case OptimizerOperation::NovoMoment:
                break;
            }
        }
    };
}

// Resolves one typed range without exposing an arbitrary host pointer to the GPU.
template <typename T>
metalrt::Argument optimizer_argument(uint32_t index, const T* pointer, uint32_t count,
                                     metalrt::Access access) {
    return metalrt::Argument::buffer(
        index, metalrt::resolve_pointer(pointer, metalrt::checked_mul(count, sizeof(T))), access);
}

// Keep the finalization constants identical on native and Reference paths.
struct AdamJitGradientArgs {
    uint32_t parameters, dense_parameters, partitions, reserved;
};
template <class T> metalrt::CpuKernel optimizer_jit_reference() {
    return [](const metalrt::ComputeCommand& command) {
        const auto args = command.scalar<AdamJitGradientArgs>(11);
        const auto* typed = command.buffer<const T>(10);
        const auto* wide = command.buffer<const float>(9);
        auto* gradients = command.buffer<T>(2);
        for (uint32_t i = 0; i < args.parameters; ++i) {
            float sum = 0.0f;
            if (i < args.dense_parameters)
                for (uint32_t p = 0; p < args.partitions; ++p)
                    sum += wide[size_t(args.parameters) + size_t(p) * args.dense_parameters + i];
            else
                sum = wide[i];
            gradients[i] = T((args.reserved & 1u ? 0.0f : float(typed[i])) + sum);
        }
        optimizer_reference<T>()(command);
    };
}

// Dispatch canonical kernels or the widened-gradient Adam epilogue. Buffer
// views retain arena scratch after the transient input descriptor expires.
template <typename T>
void optimizer_dispatch(metalStream_t stream, OptimizerArgs args,
                        std::vector<metalrt::Argument> bindings,
                        const AdamJitGradientInput* input = nullptr, bool packed_state = false) {
    if (!args.n_elements)
        return;

    struct Cached {
        metalrt::Device device;
        metalrt::Kernel kernel;
        metalrt::Buffer dummy;
        metalrt::Kernel jit_kernel;
        metalrt::Kernel packed_kernel;
        metalrt::Kernel packed_jit_kernel;
    };

    static std::mutex mutex;
    static std::vector<Cached> cache;
    auto& runtime = *resolve_stream(stream);
    metalrt::Kernel kernel;
    metalrt::Buffer dummy;
    if (packed_state && (args.operation != uint32_t(OptimizerOperation::Adam) ||
                         runtime.device().capabilities().backend != metalrt::BackendKind::Metal))
        throw std::invalid_argument("Packed Adam requires native Metal");
    {
        std::lock_guard<std::mutex> lock{mutex};
        auto found = std::find_if(cache.begin(), cache.end(), [&](const Cached& item) {
            return item.device == runtime.device();
        });
        if (found == cache.end()) {
            auto buffer = runtime.device().allocate(sizeof(float));
            std::memset(buffer.contents(), 0, sizeof(float));
            cache.push_back(
                {runtime.device(),
                 runtime.device().kernel(shader_source("optimizers.metal"),
                                         std::is_same<T, float>::value ? "tcnn_optimizer_f32"
                                                                       : "tcnn_optimizer_f16",
                                         optimizer_reference<T>()),
                 buffer,
                 {},
                 {},
                 {}});
            found = std::prev(cache.end());
        }
        if (input && !packed_state && !found->jit_kernel)
            found->jit_kernel = runtime.device().kernel(shader_source("optimizers.metal"),
                                                        std::is_same<T, float>::value
                                                            ? "tcnn_optimizer_jit_adam_f32"
                                                            : "tcnn_optimizer_jit_adam_f16",
                                                        optimizer_jit_reference<T>());
        if (packed_state) {
            auto& selected = input ? found->packed_jit_kernel : found->packed_kernel;
            if (!selected)
                selected = runtime.device().kernel(
                    shader_source("optimizers.metal"),
                    input ? (std::is_same<T, float>::value ? "tcnn_optimizer_jit_packed_adam_f32"
                                                           : "tcnn_optimizer_jit_packed_adam_f16")
                          : (std::is_same<T, float>::value ? "tcnn_optimizer_packed_adam_f32"
                                                           : "tcnn_optimizer_packed_adam_f16"));
            kernel = selected;
        } else
            kernel = input ? found->jit_kernel : found->kernel;
        dummy = found->dummy;
    }
    std::vector<metalrt::Argument> arguments;
    for (uint32_t i = 0; i < 8; ++i)
        arguments.push_back(metalrt::Argument::buffer(i, dummy.view(), metalrt::Access::Read));
    for (auto& binding : bindings)
        arguments.at(binding.index) = std::move(binding);
    arguments.push_back(metalrt::Argument::scalar(8, args));
    if (input) {
        if (input->parameters != args.n_elements || input->dense_parameters > input->parameters)
            throw std::invalid_argument("JIT Adam gradient count mismatch");
        const size_t elements =
            JitGradientPlan{input->parameters, input->dense_parameters, input->partitions}
                .elements();
        arguments.push_back(metalrt::Argument::buffer(
            9, metalrt::resolve_pointer(input->wide, elements * sizeof(float)),
            metalrt::Access::Read));
        arguments.push_back(metalrt::Argument::buffer(
            10,
            input->typed
                ? metalrt::resolve_pointer(input->typed, size_t(args.n_elements) * sizeof(T))
                : dummy.view(),
            metalrt::Access::Read));
        arguments.push_back(metalrt::Argument::scalar(
            11, AdamJitGradientArgs{input->parameters, input->dense_parameters, input->partitions,
                                    input->typed ? 0u : 1u}));
    }
    const uint32_t count = args.operation == static_cast<uint32_t>(OptimizerOperation::NovoMoment)
                               ? 1
                               : args.n_elements;
    metalrt::launch_linear(&runtime, kernel, count, std::move(arguments));
}

// Initializes the operation selector and common update size.
OptimizerArgs optimizer_args(OptimizerOperation operation, uint32_t n_elements) {
    OptimizerArgs args;
    args.operation = static_cast<uint32_t>(operation);
    args.n_elements = n_elements;
    return args;
}
} // namespace

// Enqueues the reference SGD update, retaining both full-precision and stored weights.
template <typename T>
void optimizer_sgd_step(metalStream_t stream, uint32_t n_elements, float loss_scale,
                        float learning_rate, float l2_reg, float* weights_full_precision,
                        T* weights, const T* gradients) {
    if (!n_elements)
        return;
    auto args = optimizer_args(OptimizerOperation::SGD, n_elements);
    args.loss_scale = loss_scale;
    args.learning_rate = learning_rate;
    args.l2_reg = l2_reg;
    optimizer_dispatch<T>(
        stream, args,
        {optimizer_argument(0, weights_full_precision, n_elements, metalrt::Access::ReadWrite),
         optimizer_argument(1, weights, n_elements, metalrt::Access::Write),
         optimizer_argument(2, gradients, n_elements, metalrt::Access::Read)});
}

// Implements Adam/AdaBound with the uploaded per-parameter step counters and selective updates.
template <typename T>
void optimizer_adam_step(metalStream_t stream, uint32_t n_elements, uint32_t n_matrix_weights,
                         float relative_weight_decay, float absolute_weight_decay,
                         float weight_clipping_magnitude, float gradient_clipping_magnitude,
                         float loss_scale, float learning_rate,
                         float non_matrix_learning_rate_factor, bool optimize_matrix_params,
                         bool optimize_non_matrix_params, bool skip_zero_grad_non_matrix_params,
                         float beta1, float beta2, float epsilon, float lower_lr_bound,
                         float upper_lr_bound, float l2_reg, float non_matrix_l2_reg,
                         float* weights_full_precision, T* weights, const T* gradients,
                         float* first_moments, float* second_moments, uint32_t* param_steps,
                         const AdamJitGradientInput* input, AdamState* packed_state) {
    if (!n_elements)
        return;
    auto args = optimizer_args(OptimizerOperation::Adam, n_elements);
    args.n_matrix_weights = n_matrix_weights;
    args.relative_weight_decay = relative_weight_decay;
    args.absolute_weight_decay = absolute_weight_decay;
    args.weight_clipping_magnitude = weight_clipping_magnitude;
    args.gradient_clipping_magnitude = gradient_clipping_magnitude;
    args.loss_scale = loss_scale;
    args.learning_rate = learning_rate;
    args.non_matrix_learning_rate_factor = non_matrix_learning_rate_factor;
    args.flags = (optimize_matrix_params ? 1u : 0u) | (optimize_non_matrix_params ? 2u : 0u) |
                 (skip_zero_grad_non_matrix_params ? 4u : 0u);
    args.beta1 = beta1;
    args.beta2 = beta2;
    args.epsilon = epsilon;
    args.lower_lr_bound = lower_lr_bound;
    args.upper_lr_bound = upper_lr_bound;
    args.l2_reg = l2_reg;
    args.non_matrix_l2_reg = non_matrix_l2_reg;
    std::vector<metalrt::Argument> bindings{
        optimizer_argument(0, weights_full_precision, n_elements, metalrt::Access::ReadWrite),
        optimizer_argument(1, weights, n_elements, metalrt::Access::ReadWrite),
        optimizer_argument(2, gradients, n_elements,
                           input ? metalrt::Access::ReadWrite : metalrt::Access::Read)};
    if (packed_state)
        bindings.push_back(
            optimizer_argument(3, packed_state, n_elements, metalrt::Access::ReadWrite));
    else {
        bindings.push_back(
            optimizer_argument(3, first_moments, n_elements, metalrt::Access::ReadWrite));
        bindings.push_back(
            optimizer_argument(4, second_moments, n_elements, metalrt::Access::ReadWrite));
        bindings.push_back(
            optimizer_argument(5, param_steps, n_elements, metalrt::Access::ReadWrite));
    }
    optimizer_dispatch<T>(stream, args, std::move(bindings), input, packed_state != nullptr);
}

// Updates debiased EMA state in float before rounding the custom inference weights.
template <typename T>
void optimizer_ema_step_full_precision(metalStream_t stream, uint32_t n_elements, float ema_decay,
                                       float ema_debias_old, float ema_debias_new, const T* weights,
                                       T* weights_ema, float* tmp) {
    if (!n_elements)
        return;
    auto args = optimizer_args(OptimizerOperation::EmaFull, n_elements);
    args.ema_decay = ema_decay;
    args.ema_debias_old = ema_debias_old;
    args.ema_debias_new = ema_debias_new;
    optimizer_dispatch<T>(
        stream, args,
        {optimizer_argument(1, weights, n_elements, metalrt::Access::Read),
         optimizer_argument(6, weights_ema, n_elements, metalrt::Access::ReadWrite),
         optimizer_argument(3, tmp, n_elements, metalrt::Access::ReadWrite)});
}

// Updates EMA directly in the configured storage precision, matching full_precision=false.
template <typename T>
void optimizer_ema_step_half_precision(metalStream_t stream, uint32_t n_elements, float ema_decay,
                                       float ema_debias_old, float ema_debias_new, const T* weights,
                                       T* weights_ema) {
    if (!n_elements)
        return;
    auto args = optimizer_args(OptimizerOperation::EmaStorage, n_elements);
    args.ema_decay = ema_decay;
    args.ema_debias_old = ema_debias_old;
    args.ema_debias_new = ema_debias_new;
    optimizer_dispatch<T>(
        stream, args,
        {optimizer_argument(1, weights, n_elements, metalrt::Access::Read),
         optimizer_argument(6, weights_ema, n_elements, metalrt::Access::ReadWrite)});
}

// Replaces one circular sample and adjusts the fixed-window average.
template <typename T>
void optimizer_average_step(metalStream_t stream, uint32_t n_elements, uint32_t n_samples,
                            const T* weights, T* weights_current_sample, T* weights_average) {
    if (!n_elements)
        return;
    auto args = optimizer_args(OptimizerOperation::Average, n_elements);
    args.n_samples = n_samples;
    optimizer_dispatch<T>(
        stream, args,
        {optimizer_argument(1, weights, n_elements, metalrt::Access::Read),
         optimizer_argument(7, weights_current_sample, n_elements, metalrt::Access::ReadWrite),
         optimizer_argument(6, weights_average, n_elements, metalrt::Access::ReadWrite)});
}

// Interpolates slow weights before the nested update, following the uploaded order.
template <typename T>
void optimizer_lookahead_step(metalStream_t stream, uint32_t n_elements, float alpha,
                              float* weights_full_precision, T* weights, T* weights_lookahead) {
    if (!n_elements)
        return;
    auto args = optimizer_args(OptimizerOperation::Lookahead, n_elements);
    args.alpha = alpha;
    optimizer_dispatch<T>(
        stream, args,
        {optimizer_argument(0, weights_full_precision, n_elements, metalrt::Access::ReadWrite),
         optimizer_argument(1, weights, n_elements, metalrt::Access::Write),
         optimizer_argument(6, weights_lookahead, n_elements, metalrt::Access::ReadWrite)});
}

// Accumulates the mean of a configured group of minibatch gradients in float.
template <typename T>
void optimizer_gradient_update(metalStream_t stream, uint32_t n_elements, bool first,
                               uint32_t batch_size_multiplier, const T* gradients, float* pool) {
    if (!n_elements)
        return;
    auto args = optimizer_args(OptimizerOperation::GradientUpdate, n_elements);
    args.flags = first ? 1u : 0u;
    args.batch_size_multiplier = batch_size_multiplier;
    optimizer_dispatch<T>(stream, args,
                          {optimizer_argument(2, gradients, n_elements, metalrt::Access::Read),
                           optimizer_argument(3, pool, n_elements, metalrt::Access::ReadWrite)});
}

// Rounds an FP32 temporary into optimizer storage without a host transfer.
template <typename T>
void optimizer_cast(metalStream_t stream, uint32_t n_elements, const float* input, T* output) {
    if (!n_elements)
        return;
    optimizer_dispatch<T>(stream, optimizer_args(OptimizerOperation::Cast, n_elements),
                          {optimizer_argument(0, input, n_elements, metalrt::Access::Read),
                           optimizer_argument(6, output, n_elements, metalrt::Access::Write)});
}

// Expands optimizer storage into float without a host transfer.
template <typename T>
void optimizer_cast_from(metalStream_t stream, uint32_t n_elements, const T* input, float* output) {
    if (!n_elements)
        return;
    optimizer_dispatch<T>(stream, optimizer_args(OptimizerOperation::CastFrom, n_elements),
                          {optimizer_argument(1, input, n_elements, metalrt::Access::Read),
                           optimizer_argument(0, output, n_elements, metalrt::Access::Write)});
}

// Computes a layer's squared gradient norm on the GPU and updates its scalar second moment.
template <typename T>
void optimizer_novo_moment(metalStream_t stream, uint32_t n_elements, float loss_scale, float beta2,
                           const T* gradients, float* per_layer_second_moment) {
    if (!n_elements)
        return;
    auto args = optimizer_args(OptimizerOperation::NovoMoment, n_elements);
    args.loss_scale = loss_scale;
    args.beta2 = beta2;
    optimizer_dispatch<T>(
        stream, args,
        {optimizer_argument(2, gradients, n_elements, metalrt::Access::Read),
         optimizer_argument(3, per_layer_second_moment, 1, metalrt::Access::ReadWrite)});
}

// Updates one NovoGrad layer using its GPU-resident second-moment scalar.
template <typename T>
void optimizer_novo_step(metalStream_t stream, uint32_t n_elements, float relative_weight_decay,
                         float absolute_weight_decay, float loss_scale, float learning_rate,
                         float beta1, float epsilon, float* weights_full_precision, T* weights,
                         const T* gradients, float* first_moments,
                         const float* per_layer_second_moment) {
    if (!n_elements)
        return;
    auto args = optimizer_args(OptimizerOperation::Novo, n_elements);
    args.relative_weight_decay = relative_weight_decay;
    args.absolute_weight_decay = absolute_weight_decay;
    args.loss_scale = loss_scale;
    args.learning_rate = learning_rate;
    args.beta1 = beta1;
    args.epsilon = epsilon;
    optimizer_dispatch<T>(
        stream, args,
        {optimizer_argument(0, weights_full_precision, n_elements, metalrt::Access::ReadWrite),
         optimizer_argument(1, weights, n_elements, metalrt::Access::Write),
         optimizer_argument(2, gradients, n_elements, metalrt::Access::Read),
         optimizer_argument(3, first_moments, n_elements, metalrt::Access::ReadWrite),
         optimizer_argument(4, per_layer_second_moment, 1, metalrt::Access::Read)});
}

#define TCNN_INSTANTIATE_OPTIMIZER_KERNELS(T)                                                      \
    template void optimizer_sgd_step<T>(metalStream_t, uint32_t, float, float, float, float*, T*,  \
                                        const T*);                                                 \
    template void optimizer_adam_step<T>(                                                          \
        metalStream_t, uint32_t, uint32_t, float, float, float, float, float, float, float, bool,  \
        bool, bool, float, float, float, float, float, float, float, float*, T*, const T*, float*, \
        float*, uint32_t*, const AdamJitGradientInput*, AdamState*);                               \
    template void optimizer_ema_step_full_precision<T>(metalStream_t, uint32_t, float, float,      \
                                                       float, const T*, T*, float*);               \
    template void optimizer_ema_step_half_precision<T>(metalStream_t, uint32_t, float, float,      \
                                                       float, const T*, T*);                       \
    template void optimizer_average_step<T>(metalStream_t, uint32_t, uint32_t, const T*, T*, T*);  \
    template void optimizer_lookahead_step<T>(metalStream_t, uint32_t, float, float*, T*, T*);     \
    template void optimizer_gradient_update<T>(metalStream_t, uint32_t, bool, uint32_t, const T*,  \
                                               float*);                                            \
    template void optimizer_cast<T>(metalStream_t, uint32_t, const float*, T*);                    \
    template void optimizer_cast_from<T>(metalStream_t, uint32_t, const T*, float*);               \
    template void optimizer_novo_moment<T>(metalStream_t, uint32_t, float, float, const T*,        \
                                           float*);                                                \
    template void optimizer_novo_step<T>(metalStream_t, uint32_t, float, float, float, float,      \
                                         float, float, float*, T*, const T*, float*,               \
                                         const float*);
TCNN_INSTANTIATE_OPTIMIZER_KERNELS(float)
TCNN_INSTANTIATE_OPTIMIZER_KERNELS(tcnn::half)
#undef TCNN_INSTANTIATE_OPTIMIZER_KERNELS
} // namespace detail

// Canonical optimizer factory. Default and explicit Adam share automatic native
// packed-state selection in AdamOptimizer, including nested optimizer factories.
template <typename T> Optimizer<T>* create_optimizer(const json& params) {
    const auto name = to_lower(params.value("otype", "Adam"));
    if (name == "adam")
        return new AdamOptimizer<T>{params};
    if (name == "average")
        return new AverageOptimizer<T>{params};
    if (name == "batched")
        return new BatchedOptimizer<T>{params};
    if (name == "composite")
        return new CompositeOptimizer<T>{params};
    if (name == "ema")
        return new EmaOptimizer<T>{params};
    if (name == "exponentialdecay")
        return new ExponentialDecayOptimizer<T>{params};
    if (name == "lookahead")
        return new LookaheadOptimizer<T>{params};
    if (name == "novograd")
        return new NovogradOptimizer<T>{params};
    if (name == "sgd")
        return new SGDOptimizer<T>{params};
    if (name == "shampoo")
        throw std::runtime_error{"Shampoo is an unsupported vendor-library specialization; select "
                                 "a supported Metal optimizer such as Adam or NovoGrad."};
    throw std::runtime_error{"Invalid optimizer type: " + params.value("otype", "Adam")};
}

template Optimizer<float>* create_optimizer(const json&);
template Optimizer<tcnn::half>* create_optimizer(const json&);
} // namespace tcnn
