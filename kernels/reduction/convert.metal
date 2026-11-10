// IEEE binary64 storage -> binary32 round-to-nearest-even conversion. Metal has
// no double arithmetic; the upstream identity functor accepts float, so each
// double is narrowed before summation. The caller supplies uint and ulong.
#ifndef TCNN_REDUCTION_CONVERT
#define TCNN_REDUCTION_CONVERT
inline uint reduction_f64_to_f32_bits(ulong bits) {
    const uint sign = uint(bits >> 32) & 0x80000000u;
    const uint exponent = uint((bits >> 52) & 0x7ffu);
    const ulong fraction = bits & 0x000ffffffffffffful;
    if (exponent == 0x7ffu) {
        if (fraction == 0)
            return sign | 0x7f800000u;
        // Keep the available payload bits and quiet signaling NaNs, as an
        // ordinary floating-point conversion does. Never turn a NaN into Inf.
        return sign | 0x7fc00000u | uint(fraction >> 29);
    }
    int target_exponent = int(exponent) - 896;
    if (target_exponent >= 255)
        return sign | 0x7f800000u;
    if (target_exponent < -23)
        return sign;
    const ulong significand = fraction | (1ul << 52);
    const uint shift = target_exponent > 0 ? 29u : uint(30 - target_exponent);
    const ulong remainder = significand & ((1ul << shift) - 1);
    const ulong halfway = 1ul << (shift - 1);
    ulong rounded = significand >> shift;
    rounded += remainder > halfway || (remainder == halfway && (rounded & 1ul));
    if (target_exponent <= 0)
        return sign | uint(rounded); // Includes rounding up to minimum normal.
    if (rounded == (1ul << 24)) {
        rounded >>= 1;
        ++target_exponent;
    }
    if (target_exponent >= 255)
        return sign | 0x7f800000u;
    return sign | (uint(target_exponent) << 23) | (uint(rounded) & 0x007fffffu);
}
#endif
