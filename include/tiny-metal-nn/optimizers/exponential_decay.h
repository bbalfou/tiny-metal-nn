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

template <typename T> class ExponentialDecayOptimizer : public Optimizer<T> {
  public:
    ExponentialDecayOptimizer(const json& params) {
        m_nested.reset(create_optimizer<T>(params.value("nested", json::object())));
        update_hyperparams(params);

        m_learning_rate_factor = 1.0f;
        m_base_learning_rate = m_nested->learning_rate();
    }

    // Allocates GPU state for the parameter count and leading matrix blocks.
    void allocate(uint32_t n_weights,
                  const std::vector<std::pair<uint32_t, uint32_t>>& layer_sizes) override {
        m_nested->allocate(n_weights, layer_sizes);
    }

    // Enqueues the reference update; schedule counters advance when this API is called.
    void step(metalStream_t stream, float loss_scale, float* weights_full_precision, T* weights,
              const T* gradients) override {
        if (step() == 0) {
            m_learning_rate_factor = 1.0f;
        }

        if (step() >= m_decay_start && (step() - m_decay_start) % m_decay_interval == 0 &&
            step() <= m_decay_end) {
            m_learning_rate_factor *= m_decay_base;
        }

        m_nested->set_learning_rate(m_base_learning_rate * m_learning_rate_factor);
        m_nested->step(stream, loss_scale, weights_full_precision, weights, gradients);
    }

    // Returns the current effective learning rate without a GPU readback.
    float learning_rate() const override { return m_base_learning_rate * m_learning_rate_factor; }

    // Changes the configured learning rate for subsequent update calls.
    void set_learning_rate(float val) override {
        m_base_learning_rate = val / m_learning_rate_factor;
        m_nested->set_learning_rate(m_base_learning_rate * m_learning_rate_factor);
    }

    // Returns the reference host-side update count.
    uint32_t step() const override { return m_nested->step(); }

    // Reports the allocated flat parameter count.
    uint32_t n_weights() const override { return m_nested->n_weights(); }

    // Returns custom inference weights when this optimizer maintains them.
    T* custom_weights() const override { return m_nested->custom_weights(); }

    // Reports how many nested optimizers this wrapper owns.
    size_t n_nested() const override { return 1; }

    // Returns a checked nested optimizer reference.
    const std::shared_ptr<Optimizer<T>>& nested(size_t idx) const override {
        CHECK_THROW(idx == 0);
        return m_nested;
    }

    // Updates only explicitly supplied hyperparameter keys.
    void update_hyperparams(const json& params) override {
        if (params.contains("decay_base")) {
            m_decay_base = params["decay_base"];
        }

        if (params.contains("decay_interval")) {
            m_decay_interval = params["decay_interval"];
            CHECK_THROW(m_decay_interval > 0);
        }

        if (params.contains("decay_start")) {
            m_decay_start = params["decay_start"];
        }

        if (params.contains("decay_end")) {
            m_decay_end = params["decay_end"];
        }

        if (params.contains("nested")) {
            m_nested->update_hyperparams(params["nested"]);
        }
    }

    // Describes the active reference configuration, including nested optimizers.
    json hyperparams() const override {
        return {
            {"otype", "ExponentialDecay"},  {"nested", m_nested->hyperparams()},
            {"decay_base", m_decay_base},   {"decay_interval", m_decay_interval},
            {"decay_start", m_decay_start}, {"decay_end", m_decay_end},
        };
    }

    // Captures state using the uploaded snapshot keys and binary tensor format.
    json serialize() const override {
        json data;
        data["nested"] = m_nested->serialize();
        data["learning_rate"] = m_base_learning_rate;
        data["learning_rate_factor"] = m_learning_rate_factor;
        return data;
    }

    // Restores a snapshot after state allocation with the same configuration.
    void deserialize(const json& data) override {
        m_base_learning_rate = data["learning_rate"];
        m_learning_rate_factor = data.value("learning_rate_factor", 1.0f);
        m_nested->deserialize(data["nested"]);
    }

  private:
    std::shared_ptr<Optimizer<T>> m_nested;

    float m_learning_rate_factor = 1.0f;
    float m_base_learning_rate;

    float m_decay_base = 0.1f;
    uint32_t m_decay_interval = 10000;
    uint32_t m_decay_start = 10000;
    uint32_t m_decay_end = 10000000;
};

} // namespace tcnn
