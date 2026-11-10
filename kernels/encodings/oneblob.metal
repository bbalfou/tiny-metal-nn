// OneBlob higher-order/utility entrypoints. Stable ABI: detail/kernels/encodings/oneblob.h.
// Required derivative and utility mathematics executes on the GPU.
#include "encodings/entrypoints.metal"
ENCODING_VALUE_ENTRIES(enc_oneblob, float, f32)
ENCODING_VALUE_ENTRIES(enc_oneblob, half, f16)
