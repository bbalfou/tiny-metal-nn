#pragma once
#include <tiny-metal-nn/encodings/internal/implementation.h>

namespace tcnn {
/// Integrates the reference quartic density into periodic bins for each coordinate.
template <typename T> class OneBlobEncoding : public encoding_detail::AnalyticEncoding<T> {
  public:
    OneBlobEncoding(uint32_t n_bins, uint32_t n_dims_to_encode)
        : encoding_detail::AnalyticEncoding<T>{4, n_dims_to_encode, n_bins} {}
};
} // namespace tcnn
