// HashGridRng higher-order/utility entrypoints. Stable ABI: detail/kernels/encodings/hash_grid_rng.h.
// Required derivative and utility mathematics executes on the GPU.
#include "encodings/entrypoints.metal"
ENCODING_VALUE_ENTRIES(enc_hash_grid_rng, float, f32)
ENCODING_VALUE_ENTRIES(enc_hash_grid_rng, half, f16)
ENCODING_PARAMETER_ENTRIES(enc_hash_grid_rng, float, f32)
ENCODING_PARAMETER_ENTRIES(enc_hash_grid_rng, half, f16)
