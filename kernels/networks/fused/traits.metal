#ifndef TCNN_FUSED_TRAITS_METAL
#define TCNN_FUSED_TRAITS_METAL

// A small compile-time vocabulary for square fused MLPs. This is C++14/MSL
// composition: no inheritance, runtime dispatch, or source-string generation.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#else
#include <cmath>
#endif

#define TCNN_FUSED_INLINE inline __attribute__((always_inline))

namespace tcnn_fused {

namespace math {
#ifdef __METAL_VERSION__
using metal::abs;
using metal::copysign;
using metal::exp;
using metal::log;
using metal::sin;
using metal::sqrt;
using metal::tanh;
#else
using std::abs;
using std::copysign;
using std::exp;
using std::log;
using std::sin;
using std::sqrt;
using std::tanh;
#endif
}

// The policy receives an FP32 accumulator and owns its storage-rounding
// boundary. A custom policy needs only template<class T> static T apply(float).
// Kind values preserve the port's public Activation enum.
template <unsigned Kind_> struct Activation {
    static_assert(Kind_ < 10, "unknown fused activation policy");
};
template <> struct Activation<0> {
    enum : unsigned { Kind = 0 };
    template <class T> static TCNN_FUSED_INLINE T apply(float value) {
        const T x = T(value);
        return float(x) > 0.0f ? x : T(0);
    }
};
template <> struct Activation<1> {
    enum : unsigned { Kind = 1 };
    template <class T> static TCNN_FUSED_INLINE T apply(float value) {
        const T x = T(value);
        return T(float(x) * float(T(float(x) > 0.0f ? 1.0f : 0.01f)));
    }
};
template <> struct Activation<2> {
    enum : unsigned { Kind = 2 };
    template <class T> static TCNN_FUSED_INLINE T apply(float value) {
        const float x = float(T(value));
        return T(x * (1.0f / (1.0f + math::exp(-x))));
    }
};
template <> struct Activation<3> {
    enum : unsigned { Kind = 3 };
    template <class T> static TCNN_FUSED_INLINE T apply(float value) { return T(math::exp(float(T(value)))); }
};
template <> struct Activation<4> {
    enum : unsigned { Kind = 4 };
    template <class T> static TCNN_FUSED_INLINE T apply(float value) { return T(math::sin(float(T(value)))); }
};
template <> struct Activation<5> {
    enum : unsigned { Kind = 5 };
    template <class T> static TCNN_FUSED_INLINE T apply(float value) {
        const float x = float(T(value));
        return T(1.0f / (1.0f + math::exp(-x)));
    }
};
template <> struct Activation<6> {
    enum : unsigned { Kind = 6 };
    template <class T> static TCNN_FUSED_INLINE T apply(float value) {
        const float x = float(T(value)) * 10.0f;
        return T(0.5f * (x + math::sqrt(x * x + 4.0f)) / 10.0f);
    }
};
template <> struct Activation<7> {
    enum : unsigned { Kind = 7 };
    template <class T> static TCNN_FUSED_INLINE T apply(float value) {
        const float x = float(T(value));
        return T(math::log(math::exp(x * 10.0f) + 1.0f) / 10.0f);
    }
};
template <> struct Activation<8> {
    enum : unsigned { Kind = 8 };
    template <class T> static TCNN_FUSED_INLINE T apply(float value) {
        const float x = float(T(value));
        return T(math::abs(x) >= 4.0f ? math::copysign(1.0f - 2.0f / (math::exp(2.0f * math::abs(x)) + 1.0f), x)
                                      : math::tanh(x));
    }
};
template <> struct Activation<9> {
    enum : unsigned { Kind = 9 };
    template <class T> static TCNN_FUSED_INLINE T apply(float value) { return T(value); }
};
using Relu = Activation<0>;
using Identity = Activation<9>;

template <class T> struct ScalarTraits;
template <class T> struct VectorCopyAtom;
template <class T> struct SimdgroupMmaAtom;

// Each invocation is a different template instantiation; no runtime loop or
// recursive function call is present after ordinary compiler specialization.
template <unsigned Begin, unsigned End>
struct StaticFor {
    static_assert(Begin < End, "invalid compile-time iteration interval");
    template <class Operation, class State>
    static TCNN_FUSED_INLINE void run(thread State& state) {
        Operation::template apply<Begin>(state);
        StaticFor<Begin + 1, End>::template run<Operation>(state);
    }
};
template <unsigned End>
struct StaticFor<End, End> {
    template <class Operation, class State>
    static TCNN_FUSED_INLINE void run(thread State&) {}
};

} // namespace tcnn_fused
#endif
