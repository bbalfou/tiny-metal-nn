// CompositeProduct higher-order/utility entrypoints. Stable ABI: detail/kernels/encodings/composite_product.h.
// Required derivative and utility mathematics executes on the GPU.
#include "encodings/entrypoints.metal"
ENCODING_UTILITY_ENTRIES(enc_composite_product, float, f32)
ENCODING_UTILITY_ENTRIES(enc_composite_product, half, f16)
ENCODING_LEVELS_ENTRY(enc_composite_product)
