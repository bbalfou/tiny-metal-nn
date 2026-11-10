#pragma once

#include <tiny-metal-nn/common.h>
#include <tiny-metal-nn/gpu_memory.h>

#include <tiny-metal-nn/optimizer.h>
#include <tiny-metal-nn/detail/optimizer_kernels.h>
#include <tiny-metal-nn/gpu_memory_json.h>
#include <nlohmann/json.hpp>

#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

namespace tcnn {

template <typename T> class AdamOptimizer : public Optimizer<T> {
  public:
    AdamOptimizer(const json& params) { update_hyperparams(params); }

    // Allocates GPU state for the parameter count and leading matrix blocks.
    void allocate(uint32_t n_weights,
                  const std::vector<std::pair<uint32_t, uint32_t>>& layer_sizes) override {
        m_n_weights = n_weights;
        const auto capacity = m_use_packed_state ? m_packed_state.size() : m_first_moments.size();
        if (n_weights <= capacity)
            return;
        m_n_weights_covered_by_matrices = 0;
        for (const auto& layer : layer_sizes)
            m_n_weights_covered_by_matrices += layer.first * layer.second;

        // Canonical native Adam layout for every model size and parameter type.
        // Keep FP32 moments and full-width counters together; the CPU Reference
        // backend retains its planar state and reference callback.
        m_use_packed_state =
            m_packed_state.device().capabilities().backend == metalrt::BackendKind::Metal;
        if (m_use_packed_state) {
            m_packed_state.resize(n_weights);
            m_packed_state.memset(0);
            m_first_moments.free_memory();
            m_second_moments.free_memory();
            m_param_steps.free_memory();
        } else {
            m_first_moments.resize(n_weights);
            m_first_moments.memset(0);
            m_second_moments.resize(n_weights);
            m_second_moments.memset(0);
            m_param_steps.resize(n_weights);
            m_param_steps.memset(0);
        }
    }

    // Enqueues the Adam update; schedule counters advance when this API is called.
    void step(metalStream_t stream, float loss_scale, float* full, T* weights,
              const T* gradients) override {
        step_impl(stream, loss_scale, full, weights, gradients, nullptr);
    }

    // Native JIT epilogue. Null typed input means Overwrite; a typed input may
    // alias gradients for Accumulate. Final stored gradients remain observable.
    void step_with_float_gradients(metalStream_t stream, float loss_scale, float* full, T* weights,
                                   T* gradients, const detail::AdamJitGradientInput& input) {
        nn_check(input.parameters == m_n_weights, "JIT Adam parameter count mismatch");
        (void)detail::JitGradientPlan{input.parameters, input.dense_parameters, input.partitions}
            .elements();
        nn_check(!m_n_weights || input.wide, "JIT Adam requires widened gradients");
        step_impl(stream, loss_scale, full, weights, gradients, &input);
    }

  private:
    void step_impl(metalStream_t stream, float loss_scale, float* weights_full_precision,
                   T* weights, const T* gradients, const detail::AdamJitGradientInput* input) {
        ++m_current_step;

        float lower_lr_bound = 0;
        float upper_lr_bound = std::numeric_limits<float>::max();

        // AdaBound paper: https://openreview.net/pdf?id=Bkg3g2R9FX
        if (m_adabound) {
            lower_lr_bound = 0.1f - 0.1f / ((1 - m_beta2) * (float)step() + 1);
            upper_lr_bound = 0.1f + 0.1f / ((1 - m_beta2) * (float)step());
        }

        uint32_t n_weights_to_optimize = n_weights();

        detail::optimizer_adam_step<T>(
            stream, n_weights_to_optimize, m_n_weights_covered_by_matrices, m_relative_weight_decay,
            m_absolute_weight_decay, m_weight_clipping_magnitude, m_gradient_clipping_magnitude,
            loss_scale, m_base_learning_rate, m_non_matrix_learning_rate_factor,
            m_optimize_matrix_params, m_optimize_non_matrix_params,
            m_skip_zero_grad_non_matrix_params, m_beta1, m_beta2, m_epsilon, lower_lr_bound,
            upper_lr_bound, m_l2_reg, m_non_matrix_l2_reg, weights_full_precision, weights,
            gradients, m_first_moments.data(), m_second_moments.data(), m_param_steps.data(), input,
            m_use_packed_state ? m_packed_state.data() : nullptr);
    }

  public:
    // Returns the current effective learning rate without a GPU readback.
    float learning_rate() const override { return m_base_learning_rate; }

    // Changes the configured learning rate for subsequent update calls.
    void set_learning_rate(float val) override { m_base_learning_rate = val; }

    // Returns the reference host-side update count.
    uint32_t step() const override { return m_current_step; }

    // Reports the allocated flat parameter count.
    uint32_t n_weights() const override { return m_n_weights; }

    // Returns custom inference weights when this optimizer maintains them.
    T* custom_weights() const override { return nullptr; }

    // Updates only explicitly supplied hyperparameter keys.
    void update_hyperparams(const json& params) override {
        if (params.contains("beta1")) {
            m_beta1 = params["beta1"];
        }

        if (params.contains("beta2")) {
            m_beta2 = params["beta2"];
        }

        if (params.contains("epsilon")) {
            m_epsilon = params["epsilon"];
        }

        if (params.contains("learning_rate")) {
            m_base_learning_rate = params["learning_rate"];
        }

        if (params.contains("l2_reg")) {
            m_l2_reg = params["l2_reg"];
        }

        if (params.contains("adabound")) {
            m_adabound = params["adabound"];
        }

        if (params.contains("relative_decay")) {
            m_relative_weight_decay = params["relative_decay"];
        }

        if (params.contains("absolute_decay")) {
            m_absolute_weight_decay = params["absolute_decay"];
        }

        if (params.contains("clipping_magnitude")) {
            m_weight_clipping_magnitude = params["clipping_magnitude"];
        }

        if (params.contains("gradient_clipping_magnitude")) {
            m_gradient_clipping_magnitude = params["gradient_clipping_magnitude"];
        }

        if (params.contains("non_matrix_learning_rate_factor")) {
            m_non_matrix_learning_rate_factor = params["non_matrix_learning_rate_factor"];
        }

        if (params.contains("non_matrix_l2_reg")) {
            m_non_matrix_l2_reg = params["non_matrix_l2_reg"];
        }

        if (params.contains("optimize_matrix_params")) {
            m_optimize_matrix_params = params["optimize_matrix_params"];
        }

        if (params.contains("optimize_non_matrix_params")) {
            m_optimize_non_matrix_params = params["optimize_non_matrix_params"];
        }

        if (params.contains("skip_zero_grad_non_matrix_params")) {
            m_skip_zero_grad_non_matrix_params = params["skip_zero_grad_non_matrix_params"];
        }
    }

    // Describes the active reference configuration, including nested optimizers.
    json hyperparams() const override {
        return {
            {"otype", "Adam"},
            {"beta1", m_beta1},
            {"beta2", m_beta2},
            {"epsilon", m_epsilon},
            {"learning_rate", m_base_learning_rate},
            {"l2_reg", m_l2_reg},
            {"adabound", m_adabound},
            {"relative_decay", m_relative_weight_decay},
            {"absolute_decay", m_absolute_weight_decay},
            {"clipping_magnitude", m_weight_clipping_magnitude},
            {"gradient_clipping_magnitude", m_gradient_clipping_magnitude},
            {"non_matrix_learning_rate_factor", m_non_matrix_learning_rate_factor},
            {"non_matrix_l2_reg", m_non_matrix_l2_reg},
            {"optimize_matrix_params", m_optimize_matrix_params},
            {"optimize_non_matrix_params", m_optimize_non_matrix_params},
            {"skip_zero_grad_non_matrix_params", m_skip_zero_grad_non_matrix_params},
        };
    }

    // Captures state using the uploaded snapshot keys and binary tensor format.
    json serialize() const override {
        json data;
        data["current_step"] = m_current_step;
        data["base_learning_rate"] = m_base_learning_rate;
        if (m_use_packed_state) {
            // Keep the existing checkpoint schema; transpose only at the CPU boundary.
            const auto bytes =
                gpu_memory_to_json_binary(m_packed_state.data(), m_packed_state.get_bytes());
            const size_t count = m_packed_state.size();
            json::binary_t first, second, steps;
            first.resize(count * sizeof(float));
            second.resize(first.size());
            steps.resize(first.size());
            for (size_t i = 0; i < count; ++i) {
                detail::AdamState state;
                std::memcpy(&state, bytes.data() + i * sizeof(state), sizeof(state));
                std::memcpy(first.data() + i * 4, &state.first, 4);
                std::memcpy(second.data() + i * 4, &state.second, 4);
                std::memcpy(steps.data() + i * 4, &state.step, 4);
            }
            data["first_moments_binary"] = std::move(first);
            data["second_moments_binary"] = std::move(second);
            data["param_steps_binary"] = std::move(steps);
        } else {
            data["first_moments_binary"] = m_first_moments;
            data["second_moments_binary"] = m_second_moments;
            data["param_steps_binary"] = m_param_steps;
        }
        return data;
    }

    // Restores a snapshot after state allocation with the same configuration.
    void deserialize(const json& data) override {
        if (m_use_packed_state) {
            const auto first = snapshot_bytes(data.at("first_moments_binary"));
            const auto second = snapshot_bytes(data.at("second_moments_binary"));
            const auto steps = data.contains("param_steps_binary")
                                   ? snapshot_bytes(data.at("param_steps_binary"))
                                   : json::binary_t{};
            if (first.size() % 4 || first.size() != second.size() ||
                (data.contains("param_steps_binary") && steps.size() != first.size()) ||
                first.size() / 4 < m_n_weights)
                throw std::invalid_argument("Adam snapshot state shape mismatch");
            std::vector<detail::AdamState> state(first.size() / 4);
            for (size_t i = 0; i < state.size(); ++i) {
                std::memcpy(&state[i].first, first.data() + i * 4, 4);
                std::memcpy(&state[i].second, second.data() + i * 4, 4);
                if (!steps.empty())
                    std::memcpy(&state[i].step, steps.data() + i * 4, 4);
            }
            m_packed_state.resize(state.size());
            m_packed_state.copy_from_host(state.data(), state.size());
        } else {
            m_first_moments = data["first_moments_binary"];
            m_second_moments = data["second_moments_binary"];
            if (data.contains("param_steps_binary")) {
                m_param_steps = data["param_steps_binary"];
            } else {
                m_param_steps.resize(m_second_moments.size());
                m_param_steps.memset(0);
            }
        }
        m_current_step = data["current_step"];
        m_base_learning_rate = data["base_learning_rate"];
    }

  private:
    static json::binary_t snapshot_bytes(const json& value) {
        if (value.is_binary())
            return value.get_binary();
        if (!value.is_object() || !value.at("bytes").is_array())
            throw std::invalid_argument("Invalid Adam snapshot binary");
        json::binary_t result;
        const auto& bytes = value.at("bytes");
        result.resize(bytes.size());
        for (size_t i = 0; i < bytes.size(); ++i) {
            const auto v = bytes[i].template get<int>();
            if (v < 0 || v > 255)
                throw std::invalid_argument("Adam snapshot byte out of range");
            result[i] = uint8_t(v);
        }
        return result;
    }
    GPUMemory<detail::AdamState> m_packed_state;
    bool m_use_packed_state = false;
    uint32_t m_n_weights = 0;
    uint32_t m_n_weights_covered_by_matrices = 0;

    GPUMemory<float> m_first_moments;
    GPUMemory<float> m_second_moments;
    GPUMemory<uint32_t> m_param_steps;

    uint32_t m_current_step = 0;

    // Hyperparameters
    float m_base_learning_rate = 1e-3f;
    float m_beta1 = 0.9f;
    float m_beta2 = 0.999f;
    float m_epsilon = 1e-8f;
    float m_l2_reg = 1e-8f;
    float m_non_matrix_learning_rate_factor = 1.0f;
    float m_non_matrix_l2_reg = 0.0f;

    float m_relative_weight_decay = 0.0f;
    float m_absolute_weight_decay = 0.0f;
    float m_weight_clipping_magnitude = 0.0f;
    float m_gradient_clipping_magnitude = 0.0f;

    bool m_adabound = false;

    bool m_optimize_matrix_params = true;
    bool m_optimize_non_matrix_params = true;
    bool m_skip_zero_grad_non_matrix_params = true;
};

} // namespace tcnn
