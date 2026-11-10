// Template for a callable MSL loss function. Host code substitutes the function
// name, scalar type, feature count and body; no device implementation lives in .h.
#include <metal_stdlib>
using namespace metal;
inline metal::array<@TYPE@, @DIMS@> @NAME@(
    const uint n_elements, const float loss_scale,
    const thread metal::array<@TYPE@, @DIMS@>& prediction,
    const thread metal::array<float, @DIMS@>& target,
    const thread metal::array<float, @DIMS@>& pdf,
    thread metal::array<float, @DIMS@>* value = nullptr) {
@BODY@
}
