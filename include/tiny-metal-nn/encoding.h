#pragma once
#include <tiny-metal-nn/object.h>
#include <functional>
#include <numeric>

namespace tcnn {
/// A differentiable mapping from float inputs to encoded features of type T.
/// Parameters, when present, are supplied by the caller through set_params().
template <typename T> class Encoding : public DifferentiableObject<float, T, T> {
  public:
    virtual ~Encoding() = default;

    /// Inference uses the same encoding calculation without retaining backward data.
#ifndef TCNN_NO_FWD_BWD
    void inference_mixed_precision_impl(metalStream_t stream, const GPUMatrixDynamic<float>& input,
                                        GPUMatrixDynamic<T>& output,
                                        bool use_inference_params = true) override {
        this->forward(stream, input, &output, use_inference_params, false);
    }
#endif // TCNN_NO_FWD_BWD

    /// Changes the feature width, including constant alignment padding.
    virtual void set_padded_output_width(uint32_t padded_output_width) = 0;
    virtual uint32_t required_output_alignment() const = 0;
    virtual MatrixLayout preferred_output_layout() const = 0;

    /// Composite encodings expose their children through these two functions.
    virtual size_t n_nested() const { return 0; }

    virtual const std::shared_ptr<Encoding<T>>& nested(size_t idx = 0) const {
        (void)idx;
        throw std::runtime_error("Encoding does not support nesting.");
    }

    /// Analytic encodings have no trainable state; lattice encodings override these.
    void set_params_impl(T* params, T* inference_params, T* gradients) override {
        (void)params;
        (void)inference_params;
        (void)gradients;
    }

    void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override {
        (void)rnd;
        (void)params_full_precision;
        (void)scale;
    }

    size_t n_params() const override { return 0; }

    std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override { return {}; }

    /// Rounds the current output width to the combined caller/encoding alignment.
    void set_alignment(uint32_t alignment) {
        if (!alignment || !required_output_alignment())
            throw std::invalid_argument("Encoding alignment must be positive.");
        const uint64_t a = uint64_t(alignment / std::gcd(alignment, required_output_alignment())) *
                           required_output_alignment();
        if (a > UINT32_MAX)
            throw std::overflow_error("Encoding alignment overflow.");
        set_padded_output_width(next_multiple(this->output_width(), static_cast<uint32_t>(a)));
    }
};

/// Constructs the named encoding, then applies alignment (zero leaves it unchanged).
template <typename T>
Encoding<T>* create_encoding(uint32_t n_dims_to_encode, const json& params, uint32_t alignment = 8);

/// Constructs a named encoding with an Identity child for encodings requiring nesting.
template <typename T>
std::unique_ptr<Encoding<T>> default_encoding(uint32_t n_dims_in, const std::string& name) {
    return std::unique_ptr<Encoding<T>>{
        create_encoding<T>(n_dims_in, {{"otype", name}, {"nested", {{{"otype", "Identity"}}}}})};
}

/// Returns the registered factory names, including user registrations.
std::vector<std::string> builtin_encodings();
/// Registers a case-insensitive name; duplicate names are rejected.
template <typename T>
void register_encoding(const std::string& name,
                       const std::function<Encoding<T>*(uint32_t, const json&)>& factory);
} // namespace tcnn
