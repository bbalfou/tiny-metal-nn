#pragma once
#include <tiny-metal-nn/encodings/internal/implementation.h>

namespace tcnn {
/// Expands each coordinate into sin/cos pairs at powers-of-two frequencies.
template <typename T> class FrequencyEncoding : public encoding_detail::AnalyticEncoding<T> {
  public:
    FrequencyEncoding(uint32_t n_frequencies, uint32_t n_dims_to_encode)
        : encoding_detail::AnalyticEncoding<T>{2, n_dims_to_encode, n_frequencies} {}
};
} // namespace tcnn
