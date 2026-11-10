#pragma once
#include <tiny-metal-nn/encodings/internal/implementation.h>

namespace tcnn {
/// Evaluates real SH polynomials through degree-1 on directions mapped from [0,1]^3.
template <typename T>
class SphericalHarmonicsEncoding : public encoding_detail::AnalyticEncoding<T> {
  public:
    SphericalHarmonicsEncoding(uint32_t degree, uint32_t n_dims_to_encode)
        : encoding_detail::AnalyticEncoding<T>{5, n_dims_to_encode, degree} {}
};
} // namespace tcnn
