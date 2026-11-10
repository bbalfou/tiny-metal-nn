#pragma once

#include <tiny-metal-nn/common.h>
#include <tiny-metal-nn/gpu_memory.h>
#include <tiny-metal-nn/optimizer.h>
#include <tiny-metal-nn/detail/optimizer_kernels.h>

#include <memory>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

namespace tcnn {

template <typename T> class EmaOptimizer : public Optimizer<T> {
  public:
    EmaOptimizer(const json& params) {
        m_nested.reset(create_optimizer<T>(params.value("nested", json::object())));
        update_hyperparams(params);
    }

    // Allocates GPU state for the parameter count and leading matrix blocks.
    void allocate(uint32_t n_weights,
                  const std::vector<std::pair<uint32_t, uint32_t>>& layer_sizes) override {
        m_nested->allocate(n_weights, layer_sizes);

        if (n_weights <= m_weights_ema.size()) {
            return;
        }

        m_weights_ema.resize(n_weights);
        m_weights_ema.memset(0);

        if (m_full_precision) {
            m_tmp.resize(n_weights);
            m_tmp.memset(0);
        }
    }

    // Enqueues the reference update; schedule counters advance when this API is called.
    void step(metalStream_t stream, float loss_scale, float* weights_full_precision, T* weights,
              const T* gradients) override {
        m_nested->step(stream, loss_scale, weights_full_precision, weights, gradients);

        uint32_t current_step = m_nested->step();

        float ema_debias_old = 1 - (float)std::pow(m_ema_decay, current_step - 1);
        float ema_debias_new = 1.0f / (1 - (float)std::pow(m_ema_decay, current_step));

        T* nested_custom_weights = m_nested->custom_weights();

        if (nested_custom_weights) {
            weights = nested_custom_weights;
        }

        if (m_full_precision) {
            detail::optimizer_ema_step_full_precision<T>(stream, n_weights(), m_ema_decay,
                                                         ema_debias_old, ema_debias_new, weights,
                                                         m_weights_ema.data(), m_tmp.data());
        } else {
            detail::optimizer_ema_step_half_precision<T>(stream, n_weights(), m_ema_decay,
                                                         ema_debias_old, ema_debias_new, weights,
                                                         m_weights_ema.data());
        }
    }

    // Returns the current effective learning rate without a GPU readback.
    float learning_rate() const override { return m_nested->learning_rate(); }

    // Changes the configured learning rate for subsequent update calls.
    void set_learning_rate(float val) override { m_nested->set_learning_rate(val); }

    // Returns the reference host-side update count.
    uint32_t step() const override { return m_nested->step(); }

    // Reports the allocated flat parameter count.
    uint32_t n_weights() const override { return m_nested->n_weights(); }

    // Returns custom inference weights when this optimizer maintains them.
    T* custom_weights() const override { return m_weights_ema.data(); }

    // Reports how many nested optimizers this wrapper owns.
    size_t n_nested() const override { return 1; }

    // Returns a checked nested optimizer reference.
    const std::shared_ptr<Optimizer<T>>& nested(size_t idx) const override {
        CHECK_THROW(idx == 0);
        return m_nested;
    }

    // Updates only explicitly supplied hyperparameter keys.
    void update_hyperparams(const json& params) override {
        if (params.contains("decay")) {
            m_ema_decay = params["decay"];
        }

        if (params.contains("full_precision")) {
            m_full_precision = params["full_precision"];
        }

        if (params.contains("nested")) {
            m_nested->update_hyperparams(params["nested"]);
        }
    }

    // Describes the active reference configuration, including nested optimizers.
    json hyperparams() const override {
        return {
            {"otype", "EMA"},
            {"nested", m_nested->hyperparams()},
            {"decay", m_ema_decay},
            {"full_precision", m_full_precision},
        };
    }

    // Captures state using the uploaded snapshot keys and binary tensor format.
    json serialize() const override {
        json data;
        data["nested"] = m_nested->serialize();
        data["weights_ema_binary"] = m_weights_ema;
        return data;
    }

    // Restores a snapshot after state allocation with the same configuration.
    void deserialize(const json& data) override {
        m_weights_ema = data["weights_ema_binary"];

        if (m_full_precision) {
            m_tmp.resize(m_weights_ema.size());
            detail::optimizer_cast_from<T>(nullptr, m_weights_ema.size(), m_weights_ema.data(),
                                           m_tmp.data());
        }

        m_nested->deserialize(data["nested"]);
    }

  private:
    float m_ema_decay = 0.99f;
    bool m_full_precision = false;
    std::shared_ptr<Optimizer<T>> m_nested;

    GPUMemory<T> m_weights_ema;
    GPUMemory<float> m_tmp;
};

} // namespace tcnn
