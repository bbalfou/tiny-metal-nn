// IEEE binary16 host storage matching an MSL half buffer element.
// Arithmetic in GPU kernels lives in .metal assets; these conversions are CPU utilities.
#pragma once
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace tcnn {
namespace half_detail {
// Convert binary32 to binary16 using round-to-nearest-even, including subnormals.
inline uint16_t encode(float value) noexcept {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u, exponent = (bits >> 23) & 255u;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 255)
        return static_cast<uint16_t>(sign | 0x7c00u | (mantissa ? 0x0200u | (mantissa >> 13) : 0));
    const int half_exponent = static_cast<int>(exponent) - 127 + 15;
    if (half_exponent >= 31)
        return static_cast<uint16_t>(sign | 0x7c00u);
    if (half_exponent <= 0) {
        if (half_exponent < -10)
            return static_cast<uint16_t>(sign);
        mantissa |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - half_exponent),
                       retained = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1u), halfway = 1u << (shift - 1u);
        return static_cast<uint16_t>(
            sign | (retained + (remainder > halfway || (remainder == halfway && (retained & 1u)))));
    }
    const uint32_t retained = mantissa >> 13, remainder = mantissa & 0x1fffu;
    const uint32_t rounded =
        retained + (remainder > 0x1000u || (remainder == 0x1000u && (retained & 1u)));
    return static_cast<uint16_t>(sign | ((static_cast<uint32_t>(half_exponent) << 10) + rounded));
}

// Expand every binary16 representation into binary32 without numeric rounding.
inline float decode(uint16_t value) noexcept {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    const uint32_t exponent = (value >> 10) & 31u;
    uint32_t mantissa = value & 1023u, bits;
    if (!exponent) {
        if (!mantissa)
            bits = sign;
        else {
            int shift = 0;
            while ((mantissa & 1024u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            bits =
                sign | (static_cast<uint32_t>(127 - 14 - shift) << 23) | ((mantissa & 1023u) << 13);
        }
    } else if (exponent == 31)
        bits = sign | 0x7f800000u | (mantissa << 13);
    else
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
} // namespace half_detail

// Two-byte, trivially copyable parameter/output storage. Value initialization is zero.
struct alignas(2) half {
    uint16_t bits;
    half() = default;

    half(float value) noexcept : bits{half_detail::encode(value)} {}

    template <class T, std::enable_if_t<
                           std::is_arithmetic<T>::value && !std::is_same<T, float>::value, int> = 0>
    half(T value) noexcept : half{static_cast<float>(value)} {}

    operator float() const noexcept { return half_detail::decode(bits); }

    half& operator+=(half value) noexcept {
        *this = float(*this) + float(value);
        return *this;
    }

    half& operator-=(half value) noexcept {
        *this = float(*this) - float(value);
        return *this;
    }

    half& operator*=(half value) noexcept {
        *this = float(*this) * float(value);
        return *this;
    }

    half& operator/=(half value) noexcept {
        *this = float(*this) / float(value);
        return *this;
    }
};

static_assert(sizeof(half) == 2 && alignof(half) == 2 && std::is_trivially_copyable<half>::value,
              "Metal half storage layout");

// Explicit host conversions and bit inspection, with no GPU compiler intrinsics.
inline float half_to_float(half value) noexcept {
    return float(value);
}

inline half float_to_half(float value) noexcept {
    return half{value};
}

inline uint16_t half_bits(half value) noexcept {
    return value.bits;
}

inline half half_from_bits(uint16_t value) noexcept {
    half result;
    result.bits = value;
    return result;
}
} // namespace tcnn
