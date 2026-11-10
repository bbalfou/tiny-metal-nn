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

template <typename T> class SGDOptimizer : public Optimizer<T> {
  public:
    SGDOptimizer(const json& params) { update_hyperparams(params); }

    // Allocates GPU state for the parameter count and leading matrix blocks.
    void allocate(uint32_t n_weights,
                  const std::vector<std::pair<uint32_t, uint32_t>>& layer_sizes) override {
        (void)layer_sizes;
        m_n_weights = n_weights;
    }

    // Enqueues the reference update; schedule counters advance when this API is called.
    void step(metalStream_t stream, float loss_scale, float* weights_full_precision, T* weights,
              const T* gradients) override {
        ++m_current_step;

        detail::optimizer_sgd_step<T>(stream, n_weights(), loss_scale, m_learning_rate, m_l2_reg,
                                      weights_full_precision, weights, gradients);
    }

    // Returns the current effective learning rate without a GPU readback.
    float learning_rate() const override { return m_learning_rate; }

    // Changes the configured learning rate for subsequent update calls.
    void set_learning_rate(float val) override { m_learning_rate = val; }

    // Returns the reference host-side update count.
    uint32_t step() const override { return m_current_step; }

    // Reports the allocated flat parameter count.
    uint32_t n_weights() const override { return m_n_weights; }

    // Returns custom inference weights when this optimizer maintains them.
    T* custom_weights() const override { return nullptr; }

    // Updates only explicitly supplied hyperparameter keys.
    void update_hyperparams(const json& params) override {
        if (params.contains("learning_rate")) {
            m_learning_rate = params["learning_rate"];
        }

        if (params.contains("l2_reg")) {
            m_l2_reg = params["l2_reg"];
        }
    }

    // Describes the active reference configuration, including nested optimizers.
    json hyperparams() const override {
        return {
            {"otype", "SGD"},
            {"learning_rate", m_learning_rate},
            {"l2_reg", m_l2_reg},
        };
    }

    // Captures state using the uploaded snapshot keys and binary tensor format.
    json serialize() const override {
        json data;
        data["current_step"] = m_current_step;
        data["learning_rate"] = m_learning_rate;
        return data;
    }

    // Restores a snapshot after state allocation with the same configuration.
    void deserialize(const json& data) override {
        m_current_step = data["current_step"];
        m_learning_rate = data["learning_rate"];
    }

  private:
    uint32_t m_n_weights = 0;
    uint32_t m_current_step = 0;

    // Hyperparameters
    float m_learning_rate = 1e-3f;
    float m_l2_reg = 1e-8f;
};

} // namespace tcnn
