#pragma once
#include <tiny-metal-nn/encodings/internal/implementation.h>

namespace tcnn {
/// Expands coordinates into phase-shifted triangle waves and their piecewise derivatives.
template <typename T> class TriangleWaveEncoding : public encoding_detail::AnalyticEncoding<T> {
  public:
    TriangleWaveEncoding(uint32_t n_frequencies, uint32_t n_dims_to_encode)
        : encoding_detail::AnalyticEncoding<T>{3, n_dims_to_encode, n_frequencies} {}
};
} // namespace tcnn
