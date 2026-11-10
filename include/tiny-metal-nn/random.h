#pragma once
#include <tiny-metal-nn/common_host.h>
#include <pcg32/pcg32.h>

namespace tcnn {
namespace detail {
// Record the reference's four-values-per-thread PCG layout on the selected queue.
void random_pcg(metalStream_t stream, pcg32& rng, size_t n_elements, void* out, bool fp16,
                uint32_t kind, float first, float second);
} // namespace detail

// Fill GPU storage with reproducible uniform samples, advancing the caller's PCG state.
template <typename T, typename RNG>
void generate_random_uniform(metalStream_t stream, RNG& rng, size_t n_elements, T* out,
                             const T lower = (T)0.0, const T upper = (T)1.0) {
    static_assert(std::is_same<RNG, pcg32>::value,
                  "Metal random kernels currently require the uploaded pcg32 generator");
    static_assert(std::is_same<T, float>::value || std::is_same<T, half>::value,
                  "random storage must be float or half");
    detail::random_pcg(stream, rng, n_elements, out, std::is_same<T, half>::value, 0, float(lower),
                       float(upper));
}

// Use the default queue for uniform sampling.
template <typename T, typename RNG>
void generate_random_uniform(RNG& rng, size_t n_elements, T* out, const T lower = (T)0.0,
                             const T upper = (T)1.0) {
    generate_random_uniform(nullptr, rng, n_elements, out, lower, upper);
}

// Generate uniform samples, then invoke an in-place Metal transform. The shader
// receives the typed output buffer at slot 0 and a uint element count at slot 1.
template <typename T, typename RNG>
void generate_random(metalStream_t stream, RNG& rng, size_t n_elements, T* out,
                     const metalrt::Kernel& transform) {
    if (n_elements > UINT32_MAX)
        throw std::overflow_error("random transform exceeds 32-bit indexing");
    if (!n_elements)
        return;
    if (!transform)
        throw std::invalid_argument("random transform is an empty Metal kernel");
    auto* queue = detail::resolve_stream(stream);
    auto scope = queue->batch();
    generate_random_uniform(stream, rng, n_elements, out, T(0), T(1));
    metalrt::launch_linear(
        queue, transform, n_elements,
        {metalrt::Argument::buffer(
             0, metalrt::resolve_pointer(out, metalrt::checked_mul(n_elements, sizeof(T))),
             metalrt::Access::ReadWrite),
         metalrt::Argument::scalar(1, static_cast<uint32_t>(n_elements))},
        N_THREADS_LINEAR);
}

// Fill GPU storage with the reference's inverse-logit logistic distribution.
template <typename T, typename RNG>
void generate_random_logistic(metalStream_t stream, RNG& rng, size_t n_elements, T* out,
                              const T mean = (T)0.0, const T stddev = (T)1.0) {
    static_assert(std::is_same<RNG, pcg32>::value,
                  "Metal random kernels currently require the uploaded pcg32 generator");
    static_assert(std::is_same<T, float>::value || std::is_same<T, half>::value,
                  "random storage must be float or half");
    detail::random_pcg(stream, rng, n_elements, out, std::is_same<T, half>::value, 1, float(mean),
                       float(stddev));
}

// Use the default queue for logistic sampling.
template <typename T, typename RNG>
void generate_random_logistic(RNG& rng, size_t n_elements, T* out, const T mean = (T)0.0,
                              const T stddev = (T)1.0) {
    generate_random_logistic(nullptr, rng, n_elements, out, mean, stddev);
}
} // namespace tcnn
