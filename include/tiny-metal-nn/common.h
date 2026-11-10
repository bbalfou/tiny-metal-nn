#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <tiny-metal-nn/half.h>
#include <tiny-metal-nn/metal.h>
#ifndef TCNN_HALF_PRECISION
#define TCNN_HALF_PRECISION 1
#endif

#include <tiny-metal-nn/vec.h>

namespace tcnn {

using network_precision_t = std::conditional_t<TCNN_HALF_PRECISION != 0, half, float>;
#if defined(TCNN_PARAMS_UNALIGNED)
inline constexpr bool PARAMS_ALIGNED = false;
#else
inline constexpr bool PARAMS_ALIGNED = true;
#endif

enum class Activation {
    ReLU,
    LeakyReLU,
    SiLU,
    Exponential,
    Sine,
    Sigmoid,
    Squareplus,
    Softplus,
    Tanh,
    None
};
enum class GridType { Hash, Dense, Tiled };
enum class HashType { Prime, CoherentPrime, ReversedPrime, Rng, BaseConvert };
enum class InterpolationType { Nearest, Linear, Smoothstep };
enum class ReductionType { Concatenation, Sum, Product };
enum class MatrixLayout { RowMajor = 0, SoA = 0, ColumnMajor = 1, AoS = 1 };
inline constexpr MatrixLayout RM = MatrixLayout::RowMajor;
inline constexpr MatrixLayout SoA = MatrixLayout::SoA;
inline constexpr MatrixLayout CM = MatrixLayout::ColumnMajor;
inline constexpr MatrixLayout AoS = MatrixLayout::AoS;

// Match the public batch contract of the uploaded reference.
inline constexpr uint32_t BATCH_SIZE_GRANULARITY = 256;
inline constexpr uint32_t N_THREADS_LINEAR = 128;
inline constexpr uint32_t batch_size_granularity = BATCH_SIZE_GRANULARITY;
inline constexpr uint32_t n_threads_linear = N_THREADS_LINEAR;

inline constexpr float PI() {
    return 3.14159265358979323846f;
}

template <class T> inline void host_device_swap(T& a, T& b) {
    using std::swap;
    swap(a, b);
}

template <class T> constexpr T gcd(T a, T b) {
    while (a != 0) {
        T t = b % a;
        b = a;
        a = t;
    }
    return b;
}

template <class T> constexpr T lcm(T a, T b) {
    T g = gcd(a, b);
    return g ? (a / g) * b : 0;
}

template <class T> constexpr T div_round_up(T val, T divisor) {
    if (divisor <= 0)
        throw std::invalid_argument("div_round_up divisor must be positive");
    return val / divisor + (val % divisor != 0);
}

template <class T> constexpr T next_multiple(T val, T divisor) {
    T count = div_round_up(val, divisor);
    if (count > std::numeric_limits<T>::max() / divisor)
        throw std::overflow_error("next_multiple overflow");
    return count * divisor;
}

template <class T> constexpr T previous_multiple(T val, T divisor) {
    if (divisor <= 0)
        throw std::invalid_argument("previous_multiple divisor must be positive");
    return (val / divisor) * divisor;
}

template <class T> constexpr bool is_pot(T val) {
    return (val & (val - 1)) == 0;
}

inline constexpr uint32_t next_pot(uint32_t v) {
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}
template <class T> constexpr float default_loss_scale();

template <> constexpr float default_loss_scale<float>() {
    return 1.0f;
}

template <> constexpr float default_loss_scale<half>() {
    return 128.0f;
}

template <class T> constexpr uint32_t n_blocks_linear(T n, uint32_t threads = N_THREADS_LINEAR) {
    auto result = div_round_up(n, static_cast<T>(threads));
    if (result > std::numeric_limits<uint32_t>::max())
        throw std::overflow_error("dispatch block count overflow");
    return static_cast<uint32_t>(result);
}

namespace detail {
inline size_t checked_add(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a)
        throw std::overflow_error("byte size addition overflow");
    return a + b;
}

inline size_t checked_mul(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b)
        throw std::overflow_error("byte size multiplication overflow");
    return a * b;
}

template <class RNG> auto rng_next_float(RNG& rng, int) -> decltype(rng.next_float()) {
    return rng.next_float();
}

template <class RNG> auto rng_next_float(RNG& rng, long) -> decltype(rng.nextFloat()) {
    return rng.nextFloat();
}

inline void check_range(size_t offset, size_t count, size_t size) {
    if (offset > size || count > size - offset)
        throw std::out_of_range("buffer range is out of bounds");
}
} // namespace detail

template <class T> struct PitchedPtr {
    T* ptr = nullptr;
    size_t stride_in_bytes = sizeof(T);
    PitchedPtr() = default;

    PitchedPtr(T* p, size_t stride_in_elements, size_t offset = 0, size_t extra_stride_bytes = 0)
        : ptr{p ? p + offset : nullptr},
          stride_in_bytes{detail::checked_add(detail::checked_mul(stride_in_elements, sizeof(T)),
                                              extra_stride_bytes)} {}

    // This explicit rebind intentionally has the same unrestricted pointer cast
    // as the upstream view. The caller supplies storage valid for T; the pitch
    // remains in bytes even when the element type or cv-qualification changes.
    template <class U>
    explicit PitchedPtr(PitchedPtr<U> other)
        : ptr{(T*)other.ptr}, stride_in_bytes{other.stride_in_bytes} {}

    using byte_type =
        std::conditional_t<std::is_const<T>::value, const unsigned char, unsigned char>;

    T* operator()(uint32_t y) const {
        return reinterpret_cast<T*>(reinterpret_cast<byte_type*>(ptr) + y * stride_in_bytes);
    }

    void operator+=(uint32_t y) { ptr = (*this)(y); }

    void operator-=(uint32_t y) {
        ptr = reinterpret_cast<T*>(reinterpret_cast<byte_type*>(ptr) - y * stride_in_bytes);
    }

    explicit operator bool() const { return ptr != nullptr; }
};

template <class T, class STRIDE_T = uint32_t> struct MatrixView {
    T* data = nullptr;
    STRIDE_T stride_i = 0, stride_j = 0;
    MatrixView() = default;

    MatrixView(T* p, STRIDE_T si, STRIDE_T sj) : data{p}, stride_i{si}, stride_j{sj} {}

    MatrixView(const MatrixView<std::remove_const_t<T>>& other)
        : data{other.data}, stride_i{other.stride_i}, stride_j{other.stride_j} {}

    using signed_index_t = std::make_signed_t<STRIDE_T>;
    using unsigned_index_t = std::make_unsigned_t<STRIDE_T>;

    // Signed indices support relative offsets, matching the reference overload set.
    T& operator()(signed_index_t i, signed_index_t j = 0) const {
        return data[static_cast<std::ptrdiff_t>(i) * stride_i +
                    static_cast<std::ptrdiff_t>(j) * stride_j];
    }

    void advance(signed_index_t m, signed_index_t n) {
        data +=
            static_cast<std::ptrdiff_t>(m) * stride_i + static_cast<std::ptrdiff_t>(n) * stride_j;
    }

    void advance_rows(signed_index_t m) { advance(m, signed_index_t{0}); }

    void advance_cols(signed_index_t n) { advance(signed_index_t{0}, n); }

    // Unsigned indices preserve the address arithmetic used for matrix dimensions.
    T& operator()(unsigned_index_t i, unsigned_index_t j = 0) const {
        return data[static_cast<size_t>(i) * stride_i + static_cast<size_t>(j) * stride_j];
    }

    void advance(unsigned_index_t m, unsigned_index_t n) {
        data += static_cast<size_t>(m) * stride_i + static_cast<size_t>(n) * stride_j;
    }

    void advance_rows(unsigned_index_t m) { advance(m, unsigned_index_t{0}); }

    void advance_cols(unsigned_index_t n) { advance(unsigned_index_t{0}, n); }

    template <uint32_t N> tvec<std::remove_const_t<T>, N> row(unsigned_index_t i) const {
        tvec<std::remove_const_t<T>, N> result;
        for (uint32_t j = 0; j < N; ++j)
            result[j] = (*this)(i, j);
        return result;
    }

    template <uint32_t N> tvec<std::remove_const_t<T>, N> col(unsigned_index_t j) const {
        tvec<std::remove_const_t<T>, N> result;
        for (uint32_t i = 0; i < N; ++i)
            result[i] = (*this)(i, j);
        return result;
    }

    template <class U, uint32_t N, size_t A>
    void set_row(unsigned_index_t i, const tvec<U, N, A>& v) {
        for (uint32_t j = 0; j < N; ++j)
            (*this)(i, j) = v[j];
    }

    template <class U, uint32_t N, size_t A>
    void set_col(unsigned_index_t j, const tvec<U, N, A>& v) {
        for (uint32_t i = 0; i < N; ++i)
            (*this)(i, j) = v[i];
    }

    explicit operator bool() const { return data != nullptr; }
};

template <class T> struct Interval {
    T start{}, end{};

    bool operator<(const Interval& other) const {
        return end < other.end || (end == other.end && start < other.start);
    }

    bool overlaps(const Interval& other) const { return !intersect(other).empty(); }

    Interval intersect(const Interval& other) const {
        return {std::max(start, other.start), std::min(end, other.end)};
    }

    bool valid() const { return end >= start; }

    bool empty() const { return end <= start; }

    T size() const { return end - start; }
};

// A ray stores an origin and direction and exposes the uploaded geometric helpers.
struct Ray {
    vec3 o;
    vec3 d;

    vec3 operator()(float t) const { return o + t * d; }

    void advance(float t) { o += d * t; }

    float distance_to(const vec3& p) const {
        vec3 nearest = p - o;
        nearest -= d * dot(nearest, d) / length2(d);
        return length(nearest);
    }

    bool is_valid() const { return d != vec3(0.0f); }

    static Ray invalid() { return {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}}; }
};

// Associate an index with a sortable value without changing the value's layout.
template <typename T> struct PayloadAndIdx {
    T t;
    int64_t idx;

    bool operator<(const PayloadAndIdx<T>& other) { return t < other.t; }
};

using DistAndIdx = PayloadAndIdx<float>;
using IntervalAndIdx = PayloadAndIdx<Interval<float>>;

} // namespace tcnn
