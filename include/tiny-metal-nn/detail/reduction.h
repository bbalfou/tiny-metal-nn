// Internal launch plan and scalar-packed ABI for the native sum reduction.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tcnn {
namespace detail {
namespace reduction {
inline constexpr uint32_t threads = 256;
inline constexpr uint32_t loads_per_thread = 4;
// This bounds scratch storage, not the number of input elements. The first
// dispatch uses a grid-stride loop once this many groups are sufficient.
inline constexpr uint32_t max_groups = 4096;

struct Args {
    uint32_t count, groups, accumulate, reserved;
};

static_assert(sizeof(Args) == 16 && alignof(Args) == 4 && offsetof(Args, accumulate) == 8,
              "reduction argument ABI");
static_assert(std::is_standard_layout<Args>::value && std::is_trivially_copyable<Args>::value,
              "reduction argument ABI must be POD");

inline constexpr uint32_t groups_for(uint32_t count, uint32_t elements_per_load) {
    const uint32_t tile = threads * loads_per_thread * elements_per_load;
    const uint32_t groups = count / tile + (count % tile != 0);
    return groups < max_groups ? groups : max_groups;
}

struct Plan {
    uint32_t first_groups = 0;
    uint32_t second_capacity = 0;
    uint32_t scratch_elements = 0;
    uint32_t dispatches = 0;
};

inline constexpr Plan plan(uint32_t count, uint32_t input_bytes) {
    Plan result;
    result.first_groups = groups_for(count, 16 / input_bytes);
    if (!result.first_groups)
        return result;
    result.dispatches = 1;
    if (result.first_groups > 1) {
        const uint32_t second = groups_for(result.first_groups, 1);
        result.second_capacity = second > 1 ? second : 0;
        result.scratch_elements = result.first_groups + result.second_capacity;
        for (uint32_t remaining = result.first_groups; remaining > 1;
             remaining = groups_for(remaining, 1))
            ++result.dispatches;
    }
    return result;
}
} // namespace reduction
} // namespace detail
} // namespace tcnn
