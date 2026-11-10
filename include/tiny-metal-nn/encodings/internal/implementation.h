/* Metal implementations and automatic kernel dispatch for encoding interfaces. */
#pragma once
#include <tiny-metal-nn/encodings/multi_level_interface.h>

namespace tcnn {
namespace encoding_detail {
/// Shared analytic implementation; concrete public classes preserve source constructors.
template <typename T> class AnalyticEncoding : public Encoding<T> {
  public:
    AnalyticEncoding(uint32_t kind, uint32_t n_dims_to_encode, uint32_t degree, float scale = 1,
                     float offset = 0);
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

    uint32_t input_width() const override { return m_n_dims_to_encode; }

    uint32_t padded_output_width() const override { return m_padded_output_width; }

    uint32_t output_width() const override { return padded_output_width(); }

    uint32_t required_input_alignment() const override { return 1; }

    void set_padded_output_width(uint32_t padded_output_width) override;

    uint32_t required_output_alignment() const override { return 1; }

    MatrixLayout preferred_output_layout() const override { return m_kind == 5 ? SoA : AoS; }

    json hyperparams() const override;
    uint32_t device_function_fwd_ctx_bytes() const override;
    std::string generate_device_function(const std::string& name) const override;
    std::string generate_backward_device_function(const std::string& name,
                                                  uint32_t n_threads) const override;
    std::string generate_backward_backward_input_device_function(const std::string& name,
                                                                 uint32_t n_threads) const override;

  private:
    uint32_t m_kind, m_n_dims_to_encode, m_degree, m_n_output_dims, m_padded_output_width;
    float m_scale, m_offset;
};

/// A correctness-oriented lattice implementation shared by concrete Grid and Permuto classes.
template <typename T> class LatticeEncoding : public MultiLevelEncoding<T> {
  public:
    bool jit_allows_float_gradient_accumulation() const override {
        return typeid(*this) == typeid(LatticeEncoding);
    }

    LatticeEncoding(uint32_t n_dims_to_encode, const json& encoding, bool permuto);
#ifndef TCNN_NO_FWD_BWD
    void inference_mixed_precision_impl(metalStream_t stream, const GPUMatrixDynamic<float>& input,
                                        GPUMatrixDynamic<T>& output,
                                        bool use_inference_params = true) override;
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

    uint32_t input_width() const override { return m_n_dims_to_encode; }

    uint32_t padded_output_width() const override { return m_padded_output_width; }

    uint32_t output_width() const override { return padded_output_width(); }

    uint32_t required_input_alignment() const override { return 1; }

    void set_padded_output_width(uint32_t padded_output_width) override;

    uint32_t required_output_alignment() const override { return m_n_features_per_level; }

    MatrixLayout preferred_output_layout() const override { return SoA; }

    void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override;

    size_t n_params() const override {
        return size_t(m_offset_table.data[m_n_levels]) * m_n_features_per_level;
    }

    size_t level_n_params(uint32_t level) const override {
        return level_params_offset(level + 1) - level_params_offset(level);
    }

    size_t level_params_offset(uint32_t level) const override;

    const ParamsOffsetTable& params_offset_table() const override { return m_offset_table; }

    uint32_t n_pos_dims() const override { return m_n_dims_to_encode; }

    uint32_t n_levels() const override { return m_n_levels; }

    uint32_t n_features_per_level() const override { return m_n_features_per_level; }

    json hyperparams() const override { return m_hyperparams; }

    uint32_t device_function_fwd_ctx_bytes() const override {
        return m_n_dims_to_encode * sizeof(float);
    }

    std::string generate_device_function(const std::string& name) const override;
    std::string generate_backward_device_function(const std::string& name,
                                                  uint32_t n_threads) const override;
    std::string generate_backward_backward_input_device_function(const std::string& name,
                                                                 uint32_t n_threads) const override;
    std::string generate_lookup_device_function(const std::string& name) const;
    std::string generate_backward_lookup_device_function(const std::string& name) const;
    std::string
    generate_backward_backward_input_lookup_device_function(const std::string& name) const;

    std::vector<const float*> jit_runtime_inputs() const override {
        return this->m_max_level_gpu ? std::vector<const float*>{this->m_max_level_gpu}
                                     : std::vector<const float*>{};
    }

    std::string jit_configuration_key() const override {
        return json{{"base", MultiLevelEncoding<T>::jit_configuration_key()},
                    {"fixed", m_fixed_point_pos},
                    {"stochastic", m_stochastic_interpolation}}
            .dump();
    }

  private:
    std::string generate_jit_device_function(const std::string& name, uint32_t operation) const;
#ifndef TCNN_NO_FWD_BWD
    void run(metalStream_t stream, const GPUMatrixDynamic<float>& input,
             GPUMatrixDynamic<T>* output, const GPUMatrixDynamic<T>* dL_doutput,
             GPUMatrixDynamic<float>* dL_dinput, const GPUMatrixDynamic<float>* dL_ddLdinput,
             GPUMatrixDynamic<T>* dL_ddLdoutput, bool use_inference_params,
             GradientMode param_gradients_mode, uint32_t operation, bool grouped_inference = false);
#endif // TCNN_NO_FWD_BWD
    uint32_t m_n_dims_to_encode = 0, m_n_levels = 0, m_n_features_per_level = 0,
             m_padded_output_width = 0;
    uint32_t m_base_resolution = 0, m_interpolation = 0, m_grid_type = 0, m_hash_type = 0,
             m_max_input_grad_dims = 0;
    bool m_permuto = false, m_fixed_point_pos = false, m_stochastic_interpolation = false;
    bool m_has_optimized_parameter_gradients = false;
    float m_base_scale = 0, m_per_level_scale = 0;
    ParamsOffsetTable m_offset_table;
    std::vector<float> m_scales, m_shifts;
    json m_hyperparams;
};
} // namespace encoding_detail
} // namespace tcnn
