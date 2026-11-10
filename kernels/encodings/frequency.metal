// Frequency higher-order/utility entrypoints. Stable ABI: detail/kernels/encodings/frequency.h.
// Required derivative and utility mathematics executes on the GPU.
#include "encodings/entrypoints.metal"
ENCODING_VALUE_ENTRIES(enc_frequency, float, f32)
ENCODING_VALUE_ENTRIES(enc_frequency, half, f16)
