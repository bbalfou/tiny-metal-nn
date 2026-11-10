#pragma once
#include <tiny-metal-nn/encodings/multi_level_interface.h>

namespace tcnn {
/// Uses the final input coordinate as a per-sample level limit. The LoD coordinate
/// is intentionally nondifferentiable, including in Soft mode, matching the reference.
template <typename T> class MultiLevelEncodingLoD : public MultiLevelEncoding<T> {
  public:
    MultiLevelEncodingLoD(uint32_t n_dims_to_encode, const json& encoding);
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
    void backward_backward_input_impl(
        metalStream_t stream, const Context& ctx, const GPUMatrixDynamic<float>& input,
        const GPUMatrixDynamic<float>& dL_ddLdinput, const GPUMatrixDynamic<T>& dL_doutput,
        GPUMatrixDynamic<T>* dL_ddLdoutput = nullptr, GPUMatrixDynamic<float>* dL_dinput = nullptr,
        bool use_inference_params = false,
        GradientMode param_gradients_mode = GradientMode::Overwrite) override;

#endif // TCNN_NO_FWD_BWD

    uint32_t input_width() const override { return m_base->input_width() + 1; }

    uint32_t padded_output_width() const override { return m_base->padded_output_width(); }

    uint32_t output_width() const override { return m_base->output_width(); }

    uint32_t required_input_alignment() const override { return 1; }

    void set_padded_output_width(uint32_t padded_output_width) override {
        m_base->set_padded_output_width(padded_output_width);
    }

    uint32_t required_output_alignment() const override {
        return m_base->required_output_alignment();
    }

    MatrixLayout preferred_output_layout() const override {
        return m_base->preferred_output_layout();
    }

    void set_params_impl(T* params, T* inference_params, T* gradients) override {
        m_base->set_params(params, inference_params, gradients);
    }

    void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override {
        m_base->initialize_params(rnd, params_full_precision, scale);
    }

    size_t n_params() const override { return m_base->n_params(); }

    std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
        return m_base->layer_sizes();
    }

    uint32_t n_pos_dims() const override { return m_base->n_pos_dims(); }

    uint32_t n_levels() const override { return m_base->n_levels(); }

    uint32_t n_features_per_level() const override { return m_base->n_features_per_level(); }

    size_t level_n_params(uint32_t level) const override { return m_base->level_n_params(level); }

    size_t level_params_offset(uint32_t level) const override {
        return m_base->level_params_offset(level);
    }

    const ParamsOffsetTable& params_offset_table() const override {
        return m_base->params_offset_table();
    }

    json hyperparams() const override {
        return {{"otype", "MultiLevelEncodingLoD"},
                {"lod_type", m_is_soft ? "Soft" : "Hard"},
                {"base", m_base->hyperparams()}};
    }

    std::string generate_device_function(const std::string& name) const override;
    std::string generate_backward_device_function(const std::string& name,
                                                  uint32_t n_threads) const override;
    std::string generate_backward_backward_input_device_function(const std::string& name,
                                                                 uint32_t n_threads) const override;
    bool jit_allows_float_gradient_accumulation() const override {
        return typeid(*this) == typeid(MultiLevelEncodingLoD) &&
               (!m_base->n_params() || m_base->jit_allows_float_gradient_accumulation());
    }

    uint32_t device_function_fwd_ctx_bytes() const override {
        const size_t base = m_base->device_function_fwd_ctx_bytes();
        const size_t bytes = metalrt::checked_add((base + 3u) & ~size_t(3u), sizeof(float));
        nn_check(bytes <= UINT32_MAX, "LoD JIT context exceeds uint32_t");
        return static_cast<uint32_t>(bytes);
    }
    bool device_function_fwd_ctx_aligned_per_element() const override {
        return m_base->device_function_fwd_ctx_aligned_per_element();
    }
    std::string jit_configuration_key() const override {
        return json{{"base", MultiLevelEncoding<T>::jit_configuration_key()},
                    {"soft", m_is_soft},
                    {"nested_key", m_base->jit_configuration_key()}}
            .dump();
    }
    uint32_t backward_device_function_shmem_bytes(uint32_t n_threads,
                                                  GradientMode mode) const override {
        return m_base->backward_device_function_shmem_bytes(n_threads, mode);
    }
    std::vector<const float*> jit_runtime_inputs() const override {
        auto result = m_base->jit_runtime_inputs();
        if (this->m_max_level_gpu)
            result.insert(result.begin(), this->m_max_level_gpu);
        return result;
    }

  private:
    std::string generate_jit_device_function(const std::string& name, uint32_t n_threads,
                                             uint32_t operation) const;
    struct ForwardContext : Context {
        const void* owner = nullptr;
        GPUMatrixDynamic<float> levels;
        GPUMatrixDynamic<T> base_output;
        std::unique_ptr<Context> base;
    };

    std::shared_ptr<MultiLevelEncoding<T>> m_base;
    bool m_is_soft = false;
};
} // namespace tcnn
