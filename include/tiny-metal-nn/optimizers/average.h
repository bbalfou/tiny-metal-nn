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

template <typename T> class AverageOptimizer : public Optimizer<T> {
  public:
    AverageOptimizer(const json& params) {
        m_nested.reset(create_optimizer<T>(params.value("nested", json::object())));
        update_hyperparams(params);
    }

    // Allocates GPU state for the parameter count and leading matrix blocks.
    void allocate(uint32_t n_weights,
                  const std::vector<std::pair<uint32_t, uint32_t>>& layer_sizes) override {
        m_n_weights = n_weights;
        m_layer_sizes = layer_sizes;

        m_nested->allocate(n_weights, layer_sizes);

        m_weights_samples.resize(detail::checked_mul(size_t(m_n_weights), size_t(m_n_samples)));
        m_weights_samples.memset(0);

        m_weights_average.resize(m_n_weights);
        m_weights_average.memset(0);
    }

    // Enqueues the reference update; schedule counters advance when this API is called.
    void step(metalStream_t stream, float loss_scale, float* weights_full_precision, T* weights,
              const T* gradients) override {
        m_nested->step(stream, loss_scale, weights_full_precision, weights, gradients);

        detail::optimizer_average_step<T>(stream, n_weights(), m_n_samples, weights,
                                          current_sample(), m_weights_average.data());
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
    T* custom_weights() const override { return m_weights_average.data(); }

    // Selects the circular history slot used by the current host-side step.
    uint32_t current_sample_idx() const { return step() % m_n_samples; }

    // Returns the selected sample slot without reading it on the CPU.
    T* current_sample() const {
        return m_weights_samples.data() + current_sample_idx() * m_n_weights;
    }

    // Reports how many nested optimizers this wrapper owns.
    size_t n_nested() const override { return 1; }

    // Returns a checked nested optimizer reference.
    const std::shared_ptr<Optimizer<T>>& nested(size_t idx) const override {
        CHECK_THROW(idx == 0);
        return m_nested;
    }

    // Updates only explicitly supplied hyperparameter keys.
    void update_hyperparams(const json& params) override {
        if (params.contains("n_samples")) {
            m_n_samples = params["n_samples"];
            CHECK_THROW(m_n_samples > 0);
            if (m_n_weights > 0 || !m_layer_sizes.empty()) {
                allocate(m_n_weights, m_layer_sizes);
            }
        }

        if (params.contains("nested")) {
            m_nested->update_hyperparams(params["nested"]);
        }
    }

    // Describes the active reference configuration, including nested optimizers.
    json hyperparams() const override {
        return {
            {"otype", "Average"},
            {"nested", m_nested->hyperparams()},
            {"n_samples", m_n_samples},
        };
    }

    // Captures state using the uploaded snapshot keys and binary tensor format.
    json serialize() const override {
        json data;
        data["nested"] = m_nested->serialize();
        data["weights_samples_binary"] = m_weights_samples;
        data["weights_average_binary"] = m_weights_average;
        return data;
    }

    // Restores a snapshot after state allocation with the same configuration.
    void deserialize(const json& data) override {
        m_weights_samples = data["weights_samples_binary"];
        m_weights_average = data["weights_average_binary"];
        m_nested->deserialize(data["nested"]);
    }

  private:
    uint32_t m_n_samples = 128;
    uint32_t m_n_weights = 0;
    std::shared_ptr<Optimizer<T>> m_nested;

    std::vector<std::pair<uint32_t, uint32_t>> m_layer_sizes;

    GPUMemory<T> m_weights_samples;
    GPUMemory<T> m_weights_average;
};

} // namespace tcnn
