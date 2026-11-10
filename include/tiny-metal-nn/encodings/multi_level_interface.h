#pragma once
#include <tiny-metal-nn/encoding.h>
#include <cstring>

namespace tcnn {
static constexpr uint32_t MAX_N_LEVELS = 128;

/// Offsets count lattice vertices; multiply by n_features_per_level for scalar offsets.
struct ParamsOffsetTable {
    uint32_t data[MAX_N_LEVELS + 1] = {};
    uint32_t size = 0;
};

/// Common control interface for Grid, Permuto, and their LoD wrapper.
template <typename T> class MultiLevelEncoding : public Encoding<T> {
  public:
    virtual uint32_t n_pos_dims() const = 0;

    virtual uint32_t n_levels() const {
        const auto n = params_offset_table().size;
        return n ? n - 1 : 0;
    }

    virtual uint32_t n_features_per_level() const = 0;
    virtual size_t level_n_params(uint32_t level) const = 0;
    virtual size_t level_params_offset(uint32_t level) const = 0;
    virtual const ParamsOffsetTable& params_offset_table() const = 0;

    std::string jit_configuration_key() const override {
        uint32_t bits;
        std::memcpy(&bits, &m_max_level, sizeof(bits));
        return json{{"base", Encoding<T>::jit_configuration_key()},
                    {"max_level_bits", bits},
                    {"gpu_levels", m_max_level_gpu != nullptr}}
            .dump();
    }

    /// The normalized limit enables levels below max_level * n_levels.
    float max_level() const { return m_max_level; }

    void set_max_level(float value) { m_max_level = value; }

    /// Optional per-sample normalized limits remain GPU-resident across replay.
    float* max_level_gpu() const { return m_max_level_gpu; }

    void set_max_level_gpu(float* value) { m_max_level_gpu = value; }

  protected:
    float m_max_level = 1000.f;
    float* m_max_level_gpu = nullptr;
};
} // namespace tcnn
