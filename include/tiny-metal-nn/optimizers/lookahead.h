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

template <typename T> class LookaheadOptimizer : public Optimizer<T> {
  public:
    LookaheadOptimizer(const json& params) {
        m_nested.reset(create_optimizer<T>(params.value("nested", json::object())));
        update_hyperparams(params);
    }

    // Allocates GPU state for the parameter count and leading matrix blocks.
    void allocate(uint32_t n_weights,
                  const std::vector<std::pair<uint32_t, uint32_t>>& layer_sizes) override {
        m_nested->allocate(n_weights, layer_sizes);

        if (n_weights <= m_weights_lookahead.size()) {
            return;
        }

        m_weights_lookahead.resize(n_weights);
    }

    // Enqueues the reference update; schedule counters advance when this API is called.
    void step(metalStream_t stream, float loss_scale, float* weights_full_precision, T* weights,
              const T* gradients) override {
        uint32_t current_step = m_nested->step();

        if (current_step == 0) {
            METAL_CHECK_THROW(metalMemcpy(m_weights_lookahead.data(), weights,
                                          m_weights_lookahead.size() * sizeof(T),
                                          metalMemcpyDeviceToDevice));
        }

        if (current_step % m_n_steps == 0) {
            detail::optimizer_lookahead_step<T>(stream, n_weights(), m_alpha,
                                                weights_full_precision, weights,
                                                m_weights_lookahead.data());
        }

        m_nested->step(stream, loss_scale, weights_full_precision, weights, gradients);
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
    T* custom_weights() const override { return m_weights_lookahead.data(); }

    // Reports how many nested optimizers this wrapper owns.
    size_t n_nested() const override { return 1; }

    // Returns a checked nested optimizer reference.
    const std::shared_ptr<Optimizer<T>>& nested(size_t idx) const override {
        CHECK_THROW(idx == 0);
        return m_nested;
    }

    // Updates only explicitly supplied hyperparameter keys.
    void update_hyperparams(const json& params) override {
        if (params.contains("alpha")) {
            m_alpha = params["alpha"];
        }

        if (params.contains("n_steps")) {
            m_n_steps = params["n_steps"];
            CHECK_THROW(m_n_steps > 0);
        }

        if (params.contains("nested")) {
            m_nested->update_hyperparams(params["nested"]);
        }
    }

    // Describes the active reference configuration, including nested optimizers.
    json hyperparams() const override {
        return {
            {"otype", "Lookahead"},
            {"nested", m_nested->hyperparams()},
            {"alpha", m_alpha},
            {"n_steps", m_n_steps},
        };
    }

    // Captures state using the uploaded snapshot keys and binary tensor format.
    json serialize() const override {
        json data;
        data["nested"] = m_nested->serialize();
        data["weights_lookahead_binary"] = m_weights_lookahead;
        return data;
    }

    // Restores a snapshot after state allocation with the same configuration.
    void deserialize(const json& data) override {
        m_weights_lookahead = data["weights_lookahead_binary"];
        m_nested->deserialize(data["nested"]);
    }

  private:
    float m_alpha = 0.5f;
    uint32_t m_n_steps = 16;
    std::shared_ptr<Optimizer<T>> m_nested;

    GPUMemory<T> m_weights_lookahead;
};

} // namespace tcnn
