// Canonical optimizer arithmetic and the widened-gradient Adam epilogue.
#include <metal_stdlib>
using namespace metal;

struct OptimizerArgs {
    uint n_elements, operation, n_matrix_weights, flags;
    float loss_scale, learning_rate, l2_reg, beta1, beta2, epsilon;
    float relative_weight_decay, absolute_weight_decay, weight_clipping_magnitude, gradient_clipping_magnitude;
    float non_matrix_learning_rate_factor, non_matrix_l2_reg, lower_lr_bound, upper_lr_bound;
    float ema_decay, ema_debias_old, ema_debias_new, alpha;
    uint n_samples, batch_size_multiplier;
};

// Matches the reference's decoupled relative and absolute weight decay exactly.
inline float optimizer_weight_decay(float relative, float absolute, float weight) {
    return (1.0f - relative) * weight - copysign(absolute, weight);
}

// Each operation executes independent scalar updates except NovoMoment's one-thread reduction.
template <typename T>
inline void optimizer_element(
    device float* full,
    device T* weights,
    device const T* gradients,
    device float* first,
    device float* second,
    device uint* param_steps,
    device T* auxiliary,
    device T* samples,
    constant OptimizerArgs& args,
    uint i, bool supplied_gradient = false, float gradient_value = 0.0f
) {
    if (args.operation == 9) {
        if (i != 0)
            return;
        float norm = 0;
        for (uint j = 0; j < args.n_elements; ++j) {
            float gradient = float(gradients[j]);
            norm += gradient * gradient;
        }
        first[0] = args.beta2 * first[0] + (1.0f - args.beta2) * norm / args.loss_scale / args.loss_scale;
        return;
    }
    if (i >= args.n_elements)
        return;
    switch (args.operation) {
    case 0: {
        float weight_fp = full[i];
        float gradient = float(gradients[i]) / args.loss_scale + args.l2_reg * weight_fp;
        float value = weight_fp - args.learning_rate * gradient;
        full[i] = value;
        weights[i] = T(value);
        break;
    }
    case 1: {
        float gradient = (supplied_gradient ? gradient_value : float(gradients[i])) / args.loss_scale;
        bool matrix = i < args.n_matrix_weights;
        if (matrix ? !(args.flags & 1u) : (!(args.flags & 2u) || (gradient == 0 && (args.flags & 4u))))
            break;
        float weight_fp = full[i];
        gradient += (matrix ? args.l2_reg : args.non_matrix_l2_reg) * weight_fp;
        if (args.gradient_clipping_magnitude != 0)
            gradient = copysign(min(abs(gradient), args.gradient_clipping_magnitude), gradient);
        float first_moment = first[i] = args.beta1 * first[i] + (1.0f - args.beta1) * gradient;
        float second_moment = second[i] = args.beta2 * second[i] + (1.0f - args.beta2) * gradient * gradient;
        float learning_rate = args.learning_rate * (matrix ? 1.0f : args.non_matrix_learning_rate_factor);
        uint current_step = ++param_steps[i];
        learning_rate *= sqrt(1.0f - pow(args.beta2, float(current_step))) / (1.0f - pow(args.beta1, float(current_step)));
        float effective = clamp(learning_rate / (sqrt(second_moment) + args.epsilon), args.lower_lr_bound, args.upper_lr_bound);
        float value =
            optimizer_weight_decay(args.relative_weight_decay * learning_rate, args.absolute_weight_decay * learning_rate, weight_fp) -
            effective * first_moment;
        if (args.weight_clipping_magnitude != 0)
            value = clamp(value, -args.weight_clipping_magnitude, args.weight_clipping_magnitude);
        full[i] = value;
        weights[i] = T(value);
        break;
    }
    case 2:
    case 3: {
        float old = args.operation == 2 ? first[i] : float(auxiliary[i]);
        float value = (old * args.ema_decay * args.ema_debias_old + float(weights[i]) * (1.0f - args.ema_decay)) * args.ema_debias_new;
        if (args.operation == 2)
            first[i] = value;
        auxiliary[i] = T(value);
        break;
    }
    case 4: {
        T weight = weights[i];
        auxiliary[i] = T(float(auxiliary[i]) + (float(weight) - float(samples[i])) / args.n_samples);
        samples[i] = weight;
        break;
    }
    case 5: {
        float value = float(auxiliary[i]) * (1.0f - args.alpha) + full[i] * args.alpha;
        full[i] = value;
        weights[i] = auxiliary[i] = T(value);
        break;
    }
    case 6:
        if (args.flags & 1u)
            first[i] = 0;
        first[i] += float(gradients[i]) / args.batch_size_multiplier;
        break;
    case 7:
        auxiliary[i] = T(full[i]);
        break;
    case 8:
        full[i] = float(weights[i]);
        break;
    case 10: {
        float weight_fp = full[i], gradient = float(gradients[i]) / args.loss_scale;
        float first_moment = first[i] = args.beta1 * first[i] + (1.0f - args.beta1) * gradient / (sqrt(second[0]) + args.epsilon);
        float value = optimizer_weight_decay(
                          args.relative_weight_decay * args.learning_rate, args.absolute_weight_decay * args.learning_rate, weight_fp
                      ) -
                      args.learning_rate * first_moment;
        full[i] = value;
        weights[i] = T(value);
        break;
    }
    }
}

// Interleaving moments and counters reduces cache-line overfetch for sparse updates.
// FP32 moments and full-width per-parameter counters preserve the reference update.
struct AdamState { float first, second; uint step; };
template <typename T>
inline void optimizer_packed_adam_element(device float* full, device T* weights,
    device const T* gradients, device AdamState* state, constant OptimizerArgs& args,
    uint i, bool supplied_gradient = false, float gradient_value = 0.0f) {
    if (i >= args.n_elements) return;

    float gradient = (supplied_gradient ? gradient_value : float(gradients[i])) / args.loss_scale;
    bool matrix = i < args.n_matrix_weights;
    if (matrix ? !(args.flags & 1u) : (!(args.flags & 2u) || (gradient == 0 && (args.flags & 4u))))
        return;
    float weight_fp = full[i];
    gradient += (matrix ? args.l2_reg : args.non_matrix_l2_reg) * weight_fp;
    if (args.gradient_clipping_magnitude != 0)
        gradient = copysign(min(abs(gradient), args.gradient_clipping_magnitude), gradient);
    float first_moment = args.beta1 * state[i].first + (1.0f - args.beta1) * gradient;
    float second_moment = args.beta2 * state[i].second + (1.0f - args.beta2) * gradient * gradient;
    float learning_rate = args.learning_rate * (matrix ? 1.0f : args.non_matrix_learning_rate_factor);
    uint current_step = (state[i].step + 1u);
    learning_rate *= sqrt(1.0f - pow(args.beta2, float(current_step))) / (1.0f - pow(args.beta1, float(current_step)));
    float effective = clamp(learning_rate / (sqrt(second_moment) + args.epsilon), args.lower_lr_bound, args.upper_lr_bound);
    float value =
        optimizer_weight_decay(args.relative_weight_decay * learning_rate, args.absolute_weight_decay * learning_rate, weight_fp) -
        effective * first_moment;
    if (args.weight_clipping_magnitude != 0)
        value = clamp(value, -args.weight_clipping_magnitude, args.weight_clipping_magnitude);
    full[i] = value;
    weights[i] = T(value);

    state[i] = AdamState{first_moment, second_moment, current_step};
}
#define OPTIMIZER_PACKED_ADAM(T, NAME) \
kernel void NAME(device float* full [[buffer(0)]], device T* weights [[buffer(1)]], \
    device const T* gradients [[buffer(2)]], device AdamState* state [[buffer(3)]], \
    constant OptimizerArgs& args [[buffer(8)]], uint i [[thread_position_in_grid]]) { \
    optimizer_packed_adam_element(full, weights, gradients, state, args, i); \
}
OPTIMIZER_PACKED_ADAM(float, tcnn_optimizer_packed_adam_f32)
OPTIMIZER_PACKED_ADAM(half, tcnn_optimizer_packed_adam_f16)
#undef OPTIMIZER_PACKED_ADAM

// FP32 storage entry; moments and full-precision weights are always float.
kernel void tcnn_optimizer_f32(
    device float* full [[buffer(0)]],
    device float* weights [[buffer(1)]],
    device const float* gradients [[buffer(2)]],
    device float* first [[buffer(3)]],
    device float* second [[buffer(4)]],
    device uint* steps [[buffer(5)]],
    device float* auxiliary [[buffer(6)]],
    device float* samples [[buffer(7)]],
    constant OptimizerArgs& args [[buffer(8)]],
    uint i [[thread_position_in_grid]]
) {
    optimizer_element(
        full,
        weights,
        gradients,
        first,
        second,
        steps,
        auxiliary,
        samples,
        args,
        i
    );
}

// FP16 storage entry; arithmetic remains float until a stored weight is written.
kernel void tcnn_optimizer_f16(
    device float* full [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device const half* gradients [[buffer(2)]],
    device float* first [[buffer(3)]],
    device float* second [[buffer(4)]],
    device uint* steps [[buffer(5)]],
    device half* auxiliary [[buffer(6)]],
    device half* samples [[buffer(7)]],
    constant OptimizerArgs& args [[buffer(8)]],
    uint i [[thread_position_in_grid]]
) {
    optimizer_element(
        full,
        weights,
        gradients,
        first,
        second,
        steps,
        auxiliary,
        samples,
        args,
        i
    );
}

// Finalize each parameter in the Adam dispatch. Round to the public gradient
// storage type before applying the unchanged Adam arithmetic, and preserve
// the gradient destination for callers, checkpoints and subsequent Accumulate.
struct AdamJitGradientArgs { uint parameters, dense_parameters, partitions, reserved; };
template <class T>
inline void optimizer_jit_adam_element(
    device float* full, device T* weights, device T* gradients,
    device float* first, device float* second, device uint* steps,
    device const float* wide, device const T* typed,
    constant OptimizerArgs& args, constant AdamJitGradientArgs& reduction, uint i) {
    if (i >= args.n_elements) return;
    float sum = 0.0f;
    if (i < reduction.dense_parameters) {
        for (uint p = 0; p < reduction.partitions; ++p)
            sum += wide[ulong(reduction.parameters) + ulong(p) * reduction.dense_parameters + i];
    } else sum = wide[i];
    const T rounded = T((reduction.reserved & 1u ? 0.0f : float(typed[i])) + sum);
    gradients[i] = rounded;
    optimizer_element(full, weights, gradients, first, second, steps,
        weights, weights, args, i, true, float(rounded));
}
#define OPTIMIZER_JIT_ADAM(T, NAME) \
kernel void NAME(device float* full [[buffer(0)]], device T* weights [[buffer(1)]], \
    device T* gradients [[buffer(2)]], device float* first [[buffer(3)]], \
    device float* second [[buffer(4)]], device uint* steps [[buffer(5)]], \
    constant OptimizerArgs& args [[buffer(8)]], device const float* wide [[buffer(9)]], \
    device const T* typed [[buffer(10)]], constant AdamJitGradientArgs& reduction [[buffer(11)]], \
    uint i [[thread_position_in_grid]]) { \
    optimizer_jit_adam_element(full, weights, gradients, first, second, steps, wide, typed, args, reduction, i); \
}
OPTIMIZER_JIT_ADAM(float, tcnn_optimizer_jit_adam_f32)
OPTIMIZER_JIT_ADAM(half, tcnn_optimizer_jit_adam_f16)
#undef OPTIMIZER_JIT_ADAM

// Same fused gradient finalization as the planar state epilogue, with packed state.
template <class T>
inline void optimizer_jit_packed_adam_element(device float* full, device T* weights,
    device T* gradients, device AdamState* state, device const float* wide,
    device const T* typed, constant OptimizerArgs& args,
    constant AdamJitGradientArgs& reduction, uint i) {
    if (i >= args.n_elements) return;
    float sum = 0.0f;
    if (i < reduction.dense_parameters) {
        for (uint p = 0; p < reduction.partitions; ++p)
            sum += wide[ulong(reduction.parameters) + ulong(p) * reduction.dense_parameters + i];
    } else sum = wide[i];
    const T rounded = T((reduction.reserved & 1u ? 0.0f : float(typed[i])) + sum);
    gradients[i] = rounded;
    optimizer_packed_adam_element(full, weights, gradients, state, args, i, true, float(rounded));
}
#define OPTIMIZER_JIT_PACKED_ADAM(T, NAME) \
kernel void NAME(device float* full [[buffer(0)]], device T* weights [[buffer(1)]], \
    device T* gradients [[buffer(2)]], device AdamState* state [[buffer(3)]], \
    constant OptimizerArgs& args [[buffer(8)]], device const float* wide [[buffer(9)]], \
    device const T* typed [[buffer(10)]], constant AdamJitGradientArgs& reduction [[buffer(11)]], \
    uint i [[thread_position_in_grid]]) { \
    optimizer_jit_packed_adam_element(full, weights, gradients, state, wide, typed, args, reduction, i); \
}
OPTIMIZER_JIT_PACKED_ADAM(float, tcnn_optimizer_jit_packed_adam_f32)
OPTIMIZER_JIT_PACKED_ADAM(half, tcnn_optimizer_jit_packed_adam_f16)
#undef OPTIMIZER_JIT_PACKED_ADAM
