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

template <typename T> class BatchedOptimizer : public Optimizer<T> {
  public:
    BatchedOptimizer(const json& params) {
        m_nested.reset(create_optimizer<T>(params.value("nested", json::object())));
        update_hyperparams(params);
    }

    // Allocates GPU state for the parameter count and leading matrix blocks.
    void allocate(uint32_t n_weights,
                  const std::vector<std::pair<uint32_t, uint32_t>>& layer_sizes) override {
        m_nested->allocate(n_weights, layer_sizes);

        m_averaged_gradients.resize(n_weights);
        m_averaged_gradients_half.resize(n_weights);
    }

    // Enqueues the reference update; schedule counters advance when this API is called.
    void step(metalStream_t stream, float loss_scale, float* weights_full_precision, T* weights,
              const T* gradients) override {
        detail::optimizer_gradient_update<T>(
            stream, m_nested->n_weights(), m_current_step % m_batch_size_multiplier == 0,
            m_batch_size_multiplier, gradients, m_averaged_gradients.data());
        ++m_current_step;

        if (m_current_step % m_batch_size_multiplier == 0) {
            if (!std::is_same<T, float>::value) {
                detail::optimizer_cast<T>(stream, m_nested->n_weights(),
                                          m_averaged_gradients.data(),
                                          m_averaged_gradients_half.data());
            }

            m_nested->step(stream, loss_scale, weights_full_precision, weights,
                           std::is_same<T, float>::value ? (T*)m_averaged_gradients.data()
                                                         : (T*)m_averaged_gradients_half.data());
        }
    }

    // Returns the current effective learning rate without a GPU readback.
    float learning_rate() const override { return m_nested->learning_rate(); }

    // Changes the configured learning rate for subsequent update calls.
    void set_learning_rate(float val) override { m_nested->set_learning_rate(val); }

    // Returns the reference host-side update count.
    uint32_t step() const override { return m_current_step; }

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
        if (params.contains("batch_size_multiplier")) {
            m_batch_size_multiplier = params["batch_size_multiplier"];
            CHECK_THROW(m_batch_size_multiplier > 0);
        }

        if (params.contains("nested")) {
            m_nested->update_hyperparams(params["nested"]);
        }
    }

    // Describes the active reference configuration, including nested optimizers.
    json hyperparams() const override {
        return {
            {"otype", "Batched"},
            {"nested", m_nested->hyperparams()},
            {"batch_size_multiplier", m_batch_size_multiplier},
        };
    }

    // Captures state using the uploaded snapshot keys and binary tensor format.
    json serialize() const override {
        json data;
        data["nested"] = m_nested->serialize();
        data["averaged_gradients_binary"] = m_averaged_gradients;
        data["averaged_gradients_half_binary"] = m_averaged_gradients_half;
        data["current_step"] = m_current_step;
        return data;
    }

    // Restores a snapshot after state allocation with the same configuration.
    void deserialize(const json& data) override {
        m_current_step = data["current_step"];
        m_averaged_gradients = data["averaged_gradients_binary"];
        m_averaged_gradients_half = data["averaged_gradients_half_binary"];
        m_nested->deserialize(data["nested"]);
    }

  private:
    uint32_t m_batch_size_multiplier = 16;
    std::shared_ptr<Optimizer<T>> m_nested;

    GPUMemory<float> m_averaged_gradients;
    GPUMemory<T> m_averaged_gradients_half;

    uint32_t m_current_step = 0;
};

} // namespace tcnn
