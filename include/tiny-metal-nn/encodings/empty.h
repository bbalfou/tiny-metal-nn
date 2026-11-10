#pragma once
#include <tiny-metal-nn/encodings/internal/implementation.h>

namespace tcnn {
/// Produces no unpadded features and zero input derivatives.
template <typename T> class EmptyEncoding : public encoding_detail::AnalyticEncoding<T> {
  public:
    EmptyEncoding(uint32_t n_dims_to_encode)
        : encoding_detail::AnalyticEncoding<T>{0, n_dims_to_encode, 0} {}
};
} // namespace tcnn
