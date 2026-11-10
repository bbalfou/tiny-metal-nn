// Permuto higher-order/utility entrypoints. Stable ABI: detail/kernels/encodings/permuto.h.
// Required derivative and utility mathematics executes on the GPU.
#include "encodings/entrypoints.metal"
ENCODING_VALUE_ENTRIES(enc_permuto, float, f32)
ENCODING_VALUE_ENTRIES(enc_permuto, half, f16)
ENCODING_PARAMETER_ENTRIES(enc_permuto, float, f32)
ENCODING_PARAMETER_ENTRIES(enc_permuto, half, f16)
