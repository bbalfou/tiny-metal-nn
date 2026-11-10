// Small native queries are ordinary kernels, so the answer comes from the same
// compiler/backend that executes the application's neural and custom kernels.
#include <metal_stdlib>
using namespace metal;

// Write the Metal Shading Language compiler version used for this pipeline.
kernel void tcnn_query_metal_version(device uint* result [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    if (index == 0)
        result[0] = __METAL_VERSION__;
}
