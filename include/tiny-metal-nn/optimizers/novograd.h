#pragma once

#include <tiny-metal-nn/common.h>
#include <tiny-metal-nn/gpu_memory.h>
#include <tiny-metal-nn/optimizer.h>
#include <tiny-metal-nn/detail/optimizer_kernels.h>

#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

namespace tcnn {

template <typename T> class NovogradOptimizer : public Optimizer<T> {
  public:
    NovogradOptimizer(const json& params) { update_hyperparams(params); }

    // Allocates GPU state for the parameter count and leading matrix blocks.
    void allocate(uint32_t n_weights,
                  const std::vector<std::pair<uint32_t, uint32_t>>& layer_sizes) override {
        m_n_weights = n_weights;

        if (m_n_weights <= m_first_moments.size()) {
            return;
        }

        m_first_moments.resize(m_n_weights);
        m_first_moments.memset(0);

        size_t total_size = 0;

        m_layers.clear();
        for (const auto& pair : layer_sizes) {
            CHECK_THROW(uint64_t(pair.first) * pair.second <= UINT32_MAX);
            m_layers.push_back(pair.first * pair.second);
            total_size += m_layers.back();
            CHECK_THROW(total_size <= n_weights);
        }

        m_per_layer_second_moments.resize(m_layers.size());
        m_per_layer_second_moments.memset(0);
    }

    // Enqueues the reference update; schedule counters advance when this API is called.
    void step(metalStream_t stream, float loss_scale, float* weights_full_precision, T* weights,
              const T* gradients) override {
        ++m_current_step;

        uint32_t offset = 0;
        for (size_t i = 0; i < m_layers.size(); ++i) {
            // Reduces one layer on the GPU; the first call initializes its second moment exactly.
            detail::optimizer_novo_moment<T>(
                stream, m_layers[i], loss_scale, m_current_step == 1 ? 0.0f : m_beta2,
                gradients + offset, m_per_layer_second_moments.data() + i);

            detail::optimizer_novo_step<T>(
                stream, m_layers[i], m_relative_weight_decay, m_absolute_weight_decay, loss_scale,
                m_base_learning_rate,
                m_current_step == 1 ? 0.0f : m_beta1, // Set exact value on first step
                m_epsilon, weights_full_precision + offset, weights + offset, gradients + offset,
                m_first_moments.data() + offset, m_per_layer_second_moments.data() + i);

            offset += m_layers[i];
        }
    }

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

        if (params.contains("relative_decay")) {
            m_relative_weight_decay = params["relative_decay"];
        }

        if (params.contains("absolute_decay")) {
            m_absolute_weight_decay = params["absolute_decay"];
        }
    }

    // Describes the active reference configuration, including nested optimizers.
    json hyperparams() const override {
        return {
            {"otype", "Novograd"},
            {"beta1", m_beta1},
            {"beta2", m_beta2},
            {"epsilon", m_epsilon},
            {"learning_rate", m_base_learning_rate},
            {"relative_decay", m_relative_weight_decay},
            {"absolute_decay", m_absolute_weight_decay},
        };
    }

    // Captures state using the uploaded snapshot keys and binary tensor format.
    json serialize() const override {
        json data;
        data["current_step"] = m_current_step;
        data["base_learning_rate"] = m_base_learning_rate;
        data["first_moments_binary"] = m_first_moments;
        data["per_layer_second_moments_binary"] = m_per_layer_second_moments;
        return data;
    }

    // Restores a snapshot after state allocation with the same configuration.
    void deserialize(const json& data) override {
        m_first_moments = data["first_moments_binary"];
        m_per_layer_second_moments = data["per_layer_second_moments_binary"];
        m_current_step = data["current_step"];
        m_base_learning_rate = data["base_learning_rate"];
    }

  private:
    uint32_t m_n_weights = 0;

    GPUMemory<float> m_first_moments;
    GPUMemory<float> m_per_layer_second_moments;

    GPUMemory<float> m_reduction_workspace;

    std::vector<uint32_t> m_layers;

    uint32_t m_current_step = 0;

    // Hyperparameters
    float m_base_learning_rate = 1e-3f;
    float m_beta1 = 0.9f;
    float m_beta2 = 0.999f;
    float m_epsilon = 1e-8f;

    float m_relative_weight_decay = 0.0f;
    float m_absolute_weight_decay = 0.0f;
};

} // namespace tcnn
