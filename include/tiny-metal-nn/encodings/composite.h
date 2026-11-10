#pragma once
#include <tiny-metal-nn/encoding.h>

namespace tcnn {
/// Combines named child encodings by concatenation, sum, or product.
template <typename T> class CompositeEncoding : public Encoding<T> {
  public:
    CompositeEncoding(const json& params, uint32_t n_dims_to_encode);
#ifndef TCNN_NO_FWD_BWD
    std::unique_ptr<Context> forward_impl(metalStream_t stream,
                                          const GPUMatrixDynamic<float>& input,
                                          GPUMatrixDynamic<T>* output = nullptr,
                                          bool use_inference_params = false,
                                          bool prepare_input_gradients = false) override;
    void backward_impl(metalStream_t stream, const Context& ctx,
                       const GPUMatrixDynamic<float>& input, const GPUMatrixDynamic<T>& output,
                       const GPUMatrixDynamic<T>& dL_doutput,
                       GPUMatrixDynamic<float>* dL_dinput = nullptr,
                       bool use_inference_params = false,
                       GradientMode param_gradients_mode = GradientMode::Overwrite) override;
    /// Differentiate the input-gradient operation, including product cross-child terms.
    void backward_backward_input_impl(
        metalStream_t stream, const Context& ctx, const GPUMatrixDynamic<float>& input,
        const GPUMatrixDynamic<float>& dL_ddLdinput, const GPUMatrixDynamic<T>& dL_doutput,
        GPUMatrixDynamic<T>* dL_ddLdoutput = nullptr, GPUMatrixDynamic<float>* dL_dinput = nullptr,
        bool use_inference_params = false,
        GradientMode param_gradients_mode = GradientMode::Overwrite) override;

#endif // TCNN_NO_FWD_BWD

    uint32_t input_width() const override { return m_n_dims_to_encode; }

    uint32_t padded_output_width() const override;

    uint32_t output_width() const override { return padded_output_width(); }

    uint32_t required_input_alignment() const override { return 1; }

    void set_padded_output_width(uint32_t padded_output_width) override;
    uint32_t required_output_alignment() const override;

    MatrixLayout preferred_output_layout() const override {
        return m_nested.empty() ? AoS : m_nested.front()->preferred_output_layout();
    }

    size_t n_nested() const override { return m_nested.size(); }

    /// Returns a shared child object; its parameter region belongs to this composite.
    const std::shared_ptr<Encoding<T>>& nested(size_t idx = 0) const override {
        if (idx >= m_nested.size())
            throw std::out_of_range("Composite child index.");
        return m_nested[idx];
    }

    void set_params_impl(T* params, T* inference_params, T* gradients) override;
    void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override;
    size_t n_params() const override;
    json hyperparams() const override;
    std::string generate_nested_name(const std::string& name, size_t i) const;
    std::string generate_device_function(const std::string& name) const override;
    std::string generate_backward_device_function(const std::string& name,
                                                  uint32_t n_threads) const override;
    std::string generate_backward_backward_input_device_function(const std::string& name,
                                                                 uint32_t n_threads) const override;
    bool jit_allows_float_gradient_accumulation() const override {
        if (typeid(*this) != typeid(CompositeEncoding))
            return false;
        for (const auto& child : m_nested)
            if (child->n_params() && !child->jit_allows_float_gradient_accumulation())
                return false;
        return true;
    }

    std::vector<const float*> jit_runtime_inputs() const override {
        std::vector<const float*> result;
        for (const auto& child : m_nested) {
            auto inputs = child->jit_runtime_inputs();
            result.insert(result.end(), inputs.begin(), inputs.end());
        }
        return result;
    }
    uint32_t device_function_fwd_ctx_bytes() const override;

    bool device_function_fwd_ctx_aligned_per_element() const override {
        for (const auto& child : m_nested)
            if (!child->device_function_fwd_ctx_aligned_per_element())
                return false;
        return true;
    }

    std::string jit_configuration_key() const override {
        json children = json::array();
        for (size_t i = 0; i < m_nested.size(); ++i)
            children.push_back({{"input_begin", m_dims_to_encode_begin[i]},
                                {"key", m_nested[i]->jit_configuration_key()}});
        return json{{"base", Encoding<T>::jit_configuration_key()},
                    {"reduction", static_cast<uint32_t>(m_reduction_type)},
                    {"children", children}}
            .dump();
    }

    uint32_t backward_device_function_shmem_bytes(uint32_t n_threads,
                                                  GradientMode param_gradients_mode) const override;

  private:
    struct ForwardContext : Context {
        const void* owner = nullptr;
        uint32_t batch = 0;
        std::vector<std::unique_ptr<Context>> nested;
        GPUMatrixDynamic<T> to_reduce;
    };

    uint32_t m_n_dims_to_encode;
    ReductionType m_reduction_type = ReductionType::Concatenation;
    std::vector<std::shared_ptr<Encoding<T>>> m_nested;
    std::vector<uint32_t> m_dims_to_encode_begin;
};
} // namespace tcnn
