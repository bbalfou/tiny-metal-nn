#pragma once
#include <tiny-metal-nn/common.h>
#include <tiny-metal-nn/object.h>
#include <cstdint>

namespace tcnn {
// Parameter-update contract matching the uploaded reference's argument order and ownership.
template <typename T> class Optimizer : public ObjectWithMutableHyperparams {
  public:
    virtual ~Optimizer() = default;
    // Allocates state for a flat parameter vector and optional leading matrix blocks.
    virtual void allocate(uint32_t n_weights,
                          const std::vector<std::pair<uint32_t, uint32_t>>& layer_sizes = {}) = 0;

    // Derives state allocation from a model without taking ownership of its parameter storage.
    void allocate(const std::shared_ptr<ParametricObject<T>>& target) {
        if (!target)
            throw std::invalid_argument("optimizer target is null");
        if (target->n_params() > UINT32_MAX)
            throw std::overflow_error("optimizer parameter count exceeds uint32_t");
        allocate(static_cast<uint32_t>(target->n_params()), target->layer_sizes());
    }

    // Enqueues one update; loss_scale reverses the loss's gradient scaling.
    virtual void step(metalStream_t stream, float loss_scale, float* weights_full_precision,
                      T* weights, const T* gradients) = 0;
    // Reads or replaces the configured learning rate without synchronizing the GPU.
    virtual float learning_rate() const = 0;
    virtual void set_learning_rate(float val) = 0;
    // Counts host update calls; graph replay does not advance host schedules.
    virtual uint32_t step() const = 0;
    // Reports the number of parameters assigned to this optimizer.
    virtual uint32_t n_weights() const = 0;
    // Exposes averaged/slow inference weights when the concrete optimizer supplies them.
    virtual T* custom_weights() const = 0;

    // Exposes nested optimizer ownership for wrapper and composite configurations.
    virtual size_t n_nested() const { return 0; }

    virtual const std::shared_ptr<Optimizer<T>>& nested(size_t idx = 0) const {
        (void)idx;
        throw std::runtime_error{"Optimizer does not support nesting."};
    }

    // Serializes optimizer state; tensor bytes use the reference binary JSON format.
    virtual json serialize() const { return {}; }

    // Restores a snapshot into an already-configured optimizer.
    virtual void deserialize(const json& data) { (void)data; }
};

// Creates the canonical optimizer set, defaulting to Adam. Adam selects its
// optimized native state layout automatically; no separate optimizer type is needed.
template <typename T> Optimizer<T>* create_optimizer(const json& params);
} // namespace tcnn
