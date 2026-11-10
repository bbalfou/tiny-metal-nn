#pragma once
#include <tiny-metal-nn/encodings/internal/implementation.h>
#include <cmath>
#include <type_traits>

namespace tcnn {
/// Multi-resolution regular lattice with Dense, Hash, or Tiled addressing.
template <typename T, uint32_t N_POS_DIMS = 3, uint32_t N_FEATURES_PER_LEVEL = 2,
          HashType HASH_TYPE = HashType::CoherentPrime>
class GridEncodingTemplated : public encoding_detail::LatticeEncoding<T> {
  public:
    /// Gradient storage depends on feature width, never a vendor architecture macro.
    using grad_t = std::conditional_t<N_FEATURES_PER_LEVEL == 1, float, T>;

    bool jit_allows_float_gradient_accumulation() const override {
        return typeid(*this) == typeid(GridEncodingTemplated);
    }

    bool jit_supports_fused_optimizer() const override {
        return typeid(*this) == typeid(GridEncodingTemplated);
    }

    /// Build a concrete lattice; optional hash templates remain usable directly.
    GridEncodingTemplated(uint32_t n_features, uint32_t log2_hashmap_size, uint32_t base_resolution,
                          float per_level_scale, bool stochastic_interpolation,
                          InterpolationType interpolation_type, GridType grid_type,
                          bool fixed_point_pos)
        : encoding_detail::LatticeEncoding<T>{
              N_POS_DIMS,
              {{"otype", "Grid"},
               {"n_features", n_features},
               {"n_features_per_level", N_FEATURES_PER_LEVEL},
               {"log2_hashmap_size", log2_hashmap_size},
               {"base_resolution", base_resolution},
               {"per_level_scale", per_level_scale},
               {"stochastic_interpolation", stochastic_interpolation},
               {"interpolation", to_string(interpolation_type)},
               {"type", to_string(grid_type)},
               {"fixed_point_pos", fixed_point_pos},
               {"hash", to_string(HASH_TYPE)}},
              false} {}

    // Source compatibility only: dispatch is automatic, including when an old
    // caller passes false. No alternate implementation is selected by this flag.
    [[deprecated("Grid dispatch is automatic; omit the final boolean")]]
    GridEncodingTemplated(uint32_t n_features, uint32_t log2_hashmap_size, uint32_t base_resolution,
                          float per_level_scale, bool stochastic_interpolation,
                          InterpolationType interpolation_type, GridType grid_type,
                          bool fixed_point_pos, bool)
        : GridEncodingTemplated(n_features, log2_hashmap_size, base_resolution, per_level_scale,
                                stochastic_interpolation, interpolation_type, grid_type,
                                fixed_point_pos) {}

    /// These source functions use the same selected geometry specialization as the model.
    std::string generate_lookup_device_function(const std::string& name) const {
        return encoding_detail::LatticeEncoding<T>::generate_lookup_device_function(name);
    }

    std::string generate_backward_lookup_device_function(const std::string& name) const {
        return encoding_detail::LatticeEncoding<T>::generate_backward_lookup_device_function(name);
    }

    std::string
    generate_backward_backward_input_lookup_device_function(const std::string& name) const {
        return encoding_detail::LatticeEncoding<
            T>::generate_backward_backward_input_lookup_device_function(name);
    }
};

namespace encoding_detail {
/// Parse the same constructor arguments used by both uploaded Grid helper overloads.
template <typename T, uint32_t N_FEATURES_PER_LEVEL, HashType HASH_TYPE>
MultiLevelEncoding<T>* grid_from_config(uint32_t n_dims_to_encode, const json& encoding) {
    const uint32_t log2_hashmap_size = encoding.value("log2_hashmap_size", 19u);
    const std::string encoding_type = encoding.value("otype", "Grid");
    const std::string default_type =
        equals_case_insensitive(encoding_type, "TiledGrid")
            ? "Tiled"
            : (equals_case_insensitive(encoding_type, "DenseGrid") ? "Dense" : "Hash");
    uint32_t n_features;
    if (encoding.contains("n_features") || encoding.contains("n_grid_features")) {
        n_features = encoding.contains("n_features")
                         ? encoding["n_features"].template get<uint32_t>()
                         : encoding["n_grid_features"].template get<uint32_t>();
        if (encoding.contains("n_levels"))
            throw std::runtime_error("GridEncoding: may not specify n_features and n_levels "
                                     "simultaneously (one determines the other)");
    } else
        n_features = N_FEATURES_PER_LEVEL * encoding.value("n_levels", 16u);
    const uint32_t n_levels = n_features / N_FEATURES_PER_LEVEL;
    const GridType grid_type = string_to_grid_type(encoding.value("type", default_type));
    const uint32_t base_resolution = encoding.value("base_resolution", 16u);
    const float per_level_scale =
        encoding.value("per_level_scale", grid_type == GridType::Dense
                                              ? std::exp(std::log(256.0f / float(base_resolution)) /
                                                         float(n_levels - 1))
                                              : 2.0f);
    const bool stochastic_interpolation = encoding.value("stochastic_interpolation", false);
    const InterpolationType interpolation_type =
        string_to_interpolation_type(encoding.value("interpolation", "Linear"));
    const bool fixed_point_pos = encoding.value("fixed_point_pos", false);
#define TCNN_METAL_GRID_PARAMS                                                                     \
    n_features, log2_hashmap_size, base_resolution, per_level_scale, stochastic_interpolation,     \
        interpolation_type, grid_type, fixed_point_pos
    switch (n_dims_to_encode) {
    case 1:
        if constexpr (HASH_TYPE == HashType::BaseConvert)
            return new GridEncodingTemplated<T, 1, N_FEATURES_PER_LEVEL, HASH_TYPE>{
                TCNN_METAL_GRID_PARAMS};
        break;
    case 2:
        return new GridEncodingTemplated<T, 2, N_FEATURES_PER_LEVEL, HASH_TYPE>{
            TCNN_METAL_GRID_PARAMS};
    case 3:
        return new GridEncodingTemplated<T, 3, N_FEATURES_PER_LEVEL, HASH_TYPE>{
            TCNN_METAL_GRID_PARAMS};
    case 4:
        return new GridEncodingTemplated<T, 4, N_FEATURES_PER_LEVEL, HASH_TYPE>{
            TCNN_METAL_GRID_PARAMS};
    case 5:
        if constexpr (HASH_TYPE == HashType::BaseConvert)
            return new GridEncodingTemplated<T, 5, N_FEATURES_PER_LEVEL, HASH_TYPE>{
                TCNN_METAL_GRID_PARAMS};
        break;
    case 6:
        if constexpr (HASH_TYPE == HashType::BaseConvert)
            return new GridEncodingTemplated<T, 6, N_FEATURES_PER_LEVEL, HASH_TYPE>{
                TCNN_METAL_GRID_PARAMS};
        break;
    case 7:
        if constexpr (HASH_TYPE == HashType::BaseConvert)
            return new GridEncodingTemplated<T, 7, N_FEATURES_PER_LEVEL, HASH_TYPE>{
                TCNN_METAL_GRID_PARAMS};
        break;
    case 8:
        if constexpr (HASH_TYPE == HashType::BaseConvert)
            return new GridEncodingTemplated<T, 8, N_FEATURES_PER_LEVEL, HASH_TYPE>{
                TCNN_METAL_GRID_PARAMS};
        break;
    case 9:
        if constexpr (HASH_TYPE == HashType::BaseConvert)
            return new GridEncodingTemplated<T, 9, N_FEATURES_PER_LEVEL, HASH_TYPE>{
                TCNN_METAL_GRID_PARAMS};
        break;
    case 10:
        if constexpr (HASH_TYPE == HashType::BaseConvert)
            return new GridEncodingTemplated<T, 10, N_FEATURES_PER_LEVEL, HASH_TYPE>{
                TCNN_METAL_GRID_PARAMS};
        break;
    case 12:
        if constexpr (HASH_TYPE == HashType::BaseConvert)
            return new GridEncodingTemplated<T, 12, N_FEATURES_PER_LEVEL, HASH_TYPE>{
                TCNN_METAL_GRID_PARAMS};
        break;
    case 16:
        if constexpr (HASH_TYPE == HashType::BaseConvert)
            return new GridEncodingTemplated<T, 16, N_FEATURES_PER_LEVEL, HASH_TYPE>{
                TCNN_METAL_GRID_PARAMS};
        break;
    case 24:
        if constexpr (HASH_TYPE == HashType::BaseConvert)
            return new GridEncodingTemplated<T, 24, N_FEATURES_PER_LEVEL, HASH_TYPE>{
                TCNN_METAL_GRID_PARAMS};
        break;
    default:
        break;
    }
#undef TCNN_METAL_GRID_PARAMS
    if constexpr (HASH_TYPE == HashType::BaseConvert)
        throw std::runtime_error(
            "GridEncoding: BaseConvert input dims must be 1 through 10, 12, 16, or 24.");
    throw std::runtime_error("GridEncoding: number of input dims must be 2, 3, or 4.");
}
} // namespace encoding_detail

/// Dispatch a fixed feature/hash specialization using the uploaded non-BaseConvert signature.
template <typename T, uint32_t N_FEATURES_PER_LEVEL, HashType HASH_TYPE>
typename std::enable_if<HASH_TYPE != HashType::BaseConvert, MultiLevelEncoding<T>*>::type
create_grid_encoding_templated_2(uint32_t n_dims_to_encode, const json& encoding) {
    return encoding_detail::grid_from_config<T, N_FEATURES_PER_LEVEL, HASH_TYPE>(n_dims_to_encode,
                                                                                 encoding);
}

/// Preserve the separate BaseConvert overload, including every enabled dimension in the supplied
/// reference.
template <typename T, uint32_t N_FEATURES_PER_LEVEL, HashType HASH_TYPE>
typename std::enable_if<HASH_TYPE == HashType::BaseConvert, MultiLevelEncoding<T>*>::type
create_grid_encoding_templated_2(uint32_t n_dims_to_encode, const json& encoding) {
    return encoding_detail::grid_from_config<T, N_FEATURES_PER_LEVEL, HASH_TYPE>(n_dims_to_encode,
                                                                                 encoding);
}

/// Select the uploaded feature widths while retaining the caller's template hash selection.
template <typename T, HashType HASH_TYPE>
MultiLevelEncoding<T>* create_grid_encoding_templated_1(uint32_t n_dims_to_encode,
                                                        const json& encoding) {
    const uint32_t n_features_per_level = encoding.value("n_features_per_level", 2u);
    switch (n_features_per_level) {
    case 1:
        return create_grid_encoding_templated_2<T, 1, HASH_TYPE>(n_dims_to_encode, encoding);
    case 2:
        return create_grid_encoding_templated_2<T, 2, HASH_TYPE>(n_dims_to_encode, encoding);
    case 4:
        return create_grid_encoding_templated_2<T, 4, HASH_TYPE>(n_dims_to_encode, encoding);
    case 8:
        return create_grid_encoding_templated_2<T, 8, HASH_TYPE>(n_dims_to_encode, encoding);
    default:
        throw std::runtime_error("GridEncoding: n_features_per_level must be 1, 2, 4, or 8.");
    }
}

/// Creates the CoherentPrime grid specializations enabled in the uploaded reference.
template <typename T>
MultiLevelEncoding<T>* create_grid_encoding(uint32_t n_dims_to_encode, const json& encoding);
} // namespace tcnn
