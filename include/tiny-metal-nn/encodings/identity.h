#pragma once
#include <tiny-metal-nn/encodings/internal/implementation.h>

namespace tcnn {
/// Applies x * scale + offset; padded features are one, as in the uploaded reference.
template <typename T> class IdentityEncoding : public encoding_detail::AnalyticEncoding<T> {
  public:
    IdentityEncoding(uint32_t n_dims_to_encode, float scale = 1.0f, float offset = 0.0f)
        : encoding_detail::AnalyticEncoding<T>{1, n_dims_to_encode, 0, scale, offset} {}
};
} // namespace tcnn
