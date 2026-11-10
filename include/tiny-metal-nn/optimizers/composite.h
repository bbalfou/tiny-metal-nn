/** @file   composite.h
 *  @author Sergei Solonets, Technical University of Munich
 *  @brief  Allow using different optimizer on different parameters.
 */

#pragma once

#include <stdint.h>
#include <tiny-metal-nn/common.h>
#include <tiny-metal-nn/gpu_memory.h>
#include <tiny-metal-nn/optimizer.h>
#include <tiny-metal-nn/detail/optimizer_kernels.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace tcnn {

// Returns complete matrix blocks inside a parameter slice; rejects partial matrix slices.
inline std::vector<std::pair<uint32_t, uint32_t>>
slice_weights(const std::vector<std::pair<uint32_t, uint32_t>>& object_layer_size, uint32_t offset,
              uint32_t size) {
    std::vector<std::pair<uint32_t, uint32_t>> layer_sizes;
    const uint64_t end = uint64_t(offset) + size;
    uint64_t cursor = 0;
    for (const auto& layer : object_layer_size) {
        const uint64_t next = cursor + uint64_t(layer.first) * layer.second;
        if ((cursor < offset && next > offset) || (cursor < end && next > end))
            throw std::runtime_error{"Invalid slice. Can't slice within a layer."};
        if (cursor >= offset && next <= end && cursor < end)
            layer_sizes.push_back(layer);
        cursor = next;
    }
    return layer_sizes;
}

template <typename T> class CompositeOptimizer : public Optimizer<T> {
  public:
    // Constructs this optimizer from the uploaded reference configuration keys and defaults.
    CompositeOptimizer(const json& params) {
        if (!params.contains("nested") || !params["nested"].is_array() ||
            params["nested"].empty()) {
            throw std::runtime_error{
                "Must provide an array of nested encodings to CompositeOptimizer."};
        }

        m_n_weights = 0;
        for (const auto& nested : params["nested"]) {
            m_offsets.emplace_back(m_n_weights);
            const uint32_t count = nested.value("n_params_to_optimize", uint32_t{0});
            CHECK_THROW(count <= UINT32_MAX - m_n_weights);
            m_n_weights += count;
            m_nested.emplace_back(std::shared_ptr<Optimizer<T>>{create_optimizer<T>(nested)});
            m_base_learning_rates.emplace_back(m_nested.back()->learning_rate());
        }

        m_offsets.emplace_back(m_n_weights);
        update_hyperparams(params);
    }

    // Allocates GPU state for the parameter count and leading matrix blocks.
    void allocate(uint32_t n_weights,
                  const std::vector<std::pair<uint32_t, uint32_t>>& layer_sizes) override {
        CHECK_THROW(m_n_weights <= n_weights);
        for (size_t i = 0; i < m_nested.size(); i++) {
            uint32_t size = m_offsets[i + 1] - m_offsets[i];
            m_nested[i]->allocate(size, slice_weights(layer_sizes, m_offsets[i], size));
            m_need_custom_weights |= m_nested[i]->custom_weights() != nullptr;
        }

        if (m_need_custom_weights) {
            m_custom_weights.resize(m_n_weights);
        }
    }

    // Enqueues the reference update; schedule counters advance when this API is called.
    void step(metalStream_t stream, float loss_scale, float* weights_full_precision, T* weights,
              const T* gradients) override {
        for (size_t i = 0; i < m_nested.size(); ++i) {
            uint32_t offset = m_offsets[i];
            m_nested[i]->step(stream, loss_scale, weights_full_precision + offset, weights + offset,
                              gradients + offset);
            if (m_need_custom_weights) {
                METAL_CHECK_THROW(metalMemcpyAsync(
                    m_custom_weights.data() + offset,
                    m_nested[i]->custom_weights() == nullptr ? weights + offset
                                                             : m_nested[i]->custom_weights(),
                    m_nested[i]->n_weights() * sizeof(T), metalMemcpyDeviceToDevice, stream));
            }
        }
    }

    // Returns the current effective learning rate without a GPU readback.
    float learning_rate() const override { return m_learning_rate_factor; }

    // Changes the configured learning rate for subsequent update calls.
    void set_learning_rate(float val) override {
        m_learning_rate_factor = val;
        for (size_t i = 0; i < m_nested.size(); i++) {
            m_nested[i]->set_learning_rate(m_base_learning_rates[i] * m_learning_rate_factor);
        }
    }

    // Returns the reference host-side update count.
    uint32_t step() const override { return m_nested[0]->step(); }

    // Reports the allocated flat parameter count.
    uint32_t n_weights() const override { return m_n_weights; }

    // Returns custom inference weights when this optimizer maintains them.
    T* custom_weights() const override { return m_custom_weights.data(); }

    // Updates only explicitly supplied hyperparameter keys.
    void update_hyperparams(const json& params) override {
        if (params.contains("nested") && params["nested"].is_array()) {
            const json::array_t& nested = params["nested"];
            CHECK_THROW(nested.size() == m_nested.size());
            for (size_t i = 0; i < m_nested.size(); i++) {
                m_nested[i]->update_hyperparams(nested[i]);
            }
        }
    }

    // Reports how many nested optimizers this wrapper owns.
    size_t n_nested() const override { return m_nested.size(); }

    // Returns a checked nested optimizer reference.
    const std::shared_ptr<Optimizer<T>>& nested(size_t idx) const override {
        CHECK_THROW(idx < m_nested.size());
        return m_nested[idx];
    }

    // Describes the active reference configuration, including nested optimizers.
    json hyperparams() const override {
        json::array_t nested;
        for (auto& n : m_nested) {
            nested.emplace_back(n->hyperparams());
        }

        return {{"otype", "Composite"}, {"nested", nested}};
    }

    // Captures state using the uploaded snapshot keys and binary tensor format.
    json serialize() const override {
        json::array_t nested;
        for (auto& n : m_nested) {
            nested.emplace_back(n->serialize());
        }

        return {{"nested", nested},
                {"base_learning_rates", m_base_learning_rates},
                {"learning_rate_factor", m_learning_rate_factor}};
    }

    // Restores a snapshot after state allocation with the same configuration.
    void deserialize(const json& data) override {
        const json::array_t& nested = data["nested"];
        CHECK_THROW(nested.size() == m_nested.size());
        for (size_t i = 0; i < m_nested.size(); i++) {
            m_nested.at(i)->deserialize(nested[i]);
        }

        data["base_learning_rates"].get_to(m_base_learning_rates);
        CHECK_THROW(m_base_learning_rates.size() == m_nested.size());
        set_learning_rate(data["learning_rate_factor"]);
    }

  private:
    std::vector<std::shared_ptr<Optimizer<T>>> m_nested;
    std::vector<float> m_base_learning_rates;
    std::vector<uint32_t> m_offsets;
    uint32_t m_n_weights = 0;
    float m_learning_rate_factor = 1.0f;
    bool m_need_custom_weights = false;
    GPUMemory<T> m_custom_weights;
};

} // namespace tcnn
