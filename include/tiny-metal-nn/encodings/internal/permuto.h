#pragma once
#include <tiny-metal-nn/encodings/internal/implementation.h>

namespace tcnn {
/// Piecewise-linear permutohedral lattice; seed fixes the per-level lattice shifts.
template <typename T, uint32_t N_POS_DIMS = 3, uint32_t N_FEATURES_PER_LEVEL = 2>
class PermutoEncodingTemplated : public encoding_detail::LatticeEncoding<T> {
  public:
    // The Metal implementation selects gradient storage without vendor architecture flags.
    using grad_t = std::conditional_t<N_FEATURES_PER_LEVEL == 1, float, T>;

    bool jit_allows_float_gradient_accumulation() const override {
        return typeid(*this) == typeid(PermutoEncodingTemplated);
    }

    /// Construct seeded multi-resolution lattices and cap the dimensions receiving input gradients.
    PermutoEncodingTemplated(uint32_t n_features, uint32_t log2_hashmap_size, float base_scale,
                             float per_level_scale, uint32_t max_input_grad_dims,
                             uint32_t seed = 1337)
        : encoding_detail::LatticeEncoding<T>{N_POS_DIMS,
                                              {{"otype", "Permuto"},
                                               {"n_features", n_features},
                                               {"n_features_per_level", N_FEATURES_PER_LEVEL},
                                               {"log2_hashmap_size", log2_hashmap_size},
                                               {"base_scale", base_scale},
                                               {"per_level_scale", per_level_scale},
                                               {"max_input_grad_dims", max_input_grad_dims},
                                               {"seed", seed}},
                                              true} {}
};

/// Constructs one of the dimensions/features enabled in the supplied repository.
template <typename T>
MultiLevelEncoding<T>* create_permuto_encoding(uint32_t n_dims_to_encode, const json& encoding);
} // namespace tcnn
