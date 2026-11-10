// Empty higher-order/utility entrypoints. Stable ABI: detail/kernels/encodings/empty.h.
// Required derivative and utility mathematics executes on the GPU.
#include "encodings/entrypoints.metal"
ENCODING_VALUE_ENTRIES(enc_empty, float, f32)
ENCODING_VALUE_ENTRIES(enc_empty, half, f16)
