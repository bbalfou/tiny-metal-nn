// MultiLevelLoDHard higher-order/utility entrypoints. Stable ABI: detail/kernels/encodings/multi_level_lod_hard.h.
// Required derivative and utility mathematics executes on the GPU.
#include "encodings/entrypoints.metal"
ENCODING_UTILITY_ENTRIES(enc_multi_level_lod_hard, float, f32)
ENCODING_UTILITY_ENTRIES(enc_multi_level_lod_hard, half, f16)
ENCODING_LEVELS_ENTRY(enc_multi_level_lod_hard)
