// DenseGrid higher-order/utility entrypoints. Stable ABI: detail/kernels/encodings/dense_grid.h.
// Required derivative and utility mathematics executes on the GPU.
#include "encodings/entrypoints.metal"
ENCODING_VALUE_ENTRIES(enc_dense_grid, float, f32)
ENCODING_VALUE_ENTRIES(enc_dense_grid, half, f16)
ENCODING_PARAMETER_ENTRIES(enc_dense_grid, float, f32)
ENCODING_PARAMETER_ENTRIES(enc_dense_grid, half, f16)
