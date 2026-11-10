// Identity higher-order/utility entrypoints. Stable ABI: detail/kernels/encodings/identity.h.
// Required derivative and utility mathematics executes on the GPU.
#include "encodings/entrypoints.metal"
ENCODING_VALUE_ENTRIES(enc_identity, float, f32)
ENCODING_VALUE_ENTRIES(enc_identity, half, f16)
