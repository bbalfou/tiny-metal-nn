// SphericalHarmonics higher-order/utility entrypoints. Stable ABI: detail/kernels/encodings/spherical_harmonics.h.
// Required derivative and utility mathematics executes on the GPU.
#include "encodings/entrypoints.metal"
ENCODING_VALUE_ENTRIES(enc_spherical_harmonics, float, f32)
ENCODING_VALUE_ENTRIES(enc_spherical_harmonics, half, f16)
