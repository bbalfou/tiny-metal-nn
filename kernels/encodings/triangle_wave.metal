// TriangleWave higher-order/utility entrypoints. Stable ABI: detail/kernels/encodings/triangle_wave.h.
// Required derivative and utility mathematics executes on the GPU.
#include "encodings/entrypoints.metal"
ENCODING_VALUE_ENTRIES(enc_triangle_wave, float, f32)
ENCODING_VALUE_ENTRIES(enc_triangle_wave, half, f16)
