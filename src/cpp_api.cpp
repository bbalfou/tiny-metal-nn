#include <tiny-metal-nn/cpp_api.h>
#include <tiny-metal-nn/encoding.h>
#ifndef TCNN_NO_NETWORKS
#include <tiny-metal-nn/network_with_input_encoding.h>
#endif
#include <tiny-metal-nn/gpu_memory.h>
#include <tiny-metal-nn/rtc_kernel.h>

namespace tcnn {
namespace cpp {
uint32_t batch_size_granularity() {
    return tcnn::BATCH_SIZE_GRANULARITY;
}

int metal_device() {
    return tcnn::metal_device();
}

void set_metal_device(int device) {
    tcnn::set_metal_device(device);
}

void free_temporary_memory() {
    tcnn::free_all_gpu_memory_arenas();
}

bool has_networks() {
#ifdef TCNN_NO_NETWORKS
    return false;
#else
    return true;
#endif
}

float default_loss_scale(Precision p) {
    if (p == Precision::Fp32)
        return tcnn::default_loss_scale<float>();
    if (p == Precision::Fp16)
        return tcnn::default_loss_scale<half>();
    throw std::invalid_argument("Unknown Metal parameter precision");
}

Precision preferred_precision() {
    return std::is_same<network_precision_t, float>::value ? Precision::Fp32 : Precision::Fp16;
}

bool supports_jit_fusion(int device) {
    return tcnn::supports_jit_fusion(device);
}

void rtc_set_cache_dir(const std::string& dir) {
    tcnn::rtc_set_cache_dir(dir);
}

void rtc_set_include_dir(const std::string& dir) {
    tcnn::rtc_set_include_dir(dir);
}

void set_log_callback(const std::function<void(LogSeverity, const std::string&)>& callback) {
    tcnn::set_log_callback(
        callback ? std::function<void(tcnn::LogSeverity,
                                      const std::string&)>{[callback](tcnn::LogSeverity severity,
                                                                      const std::string& msg) {
            callback(static_cast<LogSeverity>(severity), msg);
        }}
                 : std::function<void(tcnn::LogSeverity, const std::string&)>{});
}

namespace {
// Implement the exact non-template virtual facade for both storage precisions.
template <class T> class ModuleImpl final : public Module {
  public:
    explicit ModuleImpl(tcnn::DifferentiableObject<float, T, T>* model)
        : Module{precision(), precision()}, m_model{model} {}

    void inference(metalStream_t stream, uint32_t n_elements, const float* input, void* output,
                   void* params) override {
        bind(params, nullptr);
        GPUMatrix<float> input_matrix{const_cast<float*>(input), m_model->input_width(),
                                      n_elements};
        GPUMatrix<T> output_matrix{static_cast<T*>(output), m_model->padded_output_width(),
                                   n_elements};
        m_model->inference_mixed_precision(stream, input_matrix, output_matrix);
    }

    Context forward(metalStream_t stream, uint32_t n_elements, const float* input, void* output,
                    void* params, bool prepare_input_gradients) override {
        bind(params, nullptr);
        GPUMatrix<float> input_matrix{const_cast<float*>(input), m_model->input_width(),
                                      n_elements};
        GPUMatrix<T> output_matrix{static_cast<T*>(output), m_model->padded_output_width(),
                                   n_elements};
        return {
            m_model->forward(stream, input_matrix, &output_matrix, false, prepare_input_gradients)};
    }

    void backward(metalStream_t stream, const Context& ctx, uint32_t n_elements, float* dL_dinput,
                  const void* dL_doutput, void* dL_dparams, const float* input, const void* output,
                  const void* params) override {
        nn_check(static_cast<bool>(ctx.ctx), "backward requires a forward context");
        bind(const_cast<void*>(params), dL_dparams);
        GPUMatrix<float> input_matrix{const_cast<float*>(input), m_model->input_width(),
                                      n_elements};
        GPUMatrix<T> output_matrix{static_cast<T*>(const_cast<void*>(output)),
                                   m_model->padded_output_width(), n_elements},
            dL_doutput_matrix{static_cast<T*>(const_cast<void*>(dL_doutput)),
                              m_model->padded_output_width(), n_elements};
        GPUMatrix<float> dL_dinput_matrix{dL_dinput, m_model->input_width(), n_elements};
        m_model->backward(stream, *ctx.ctx, input_matrix, output_matrix, dL_doutput_matrix,
                          dL_dinput ? &dL_dinput_matrix : nullptr, false,
                          dL_dparams ? GradientMode::Overwrite : GradientMode::Ignore);
    }

    void backward_backward_input(metalStream_t stream, const Context& ctx, uint32_t n_elements,
                                 const float* dL_ddLdinput, const float* input,
                                 const void* dL_doutput, void* dL_dparams, void* dL_ddLdoutput,
                                 float* dL_dinput, const void* params) override {
        nn_check(static_cast<bool>(ctx.ctx), "double backward requires a forward context");
        bind(const_cast<void*>(params), dL_dparams);
        GPUMatrix<float> input_matrix{const_cast<float*>(input), m_model->input_width(),
                                      n_elements},
            dL_ddLdinput_matrix{const_cast<float*>(dL_ddLdinput), m_model->input_width(),
                                n_elements},
            dL_dinput_matrix{dL_dinput, m_model->input_width(), n_elements};
        GPUMatrix<T> dL_doutput_matrix{static_cast<T*>(const_cast<void*>(dL_doutput)),
                                       m_model->padded_output_width(), n_elements},
            dL_ddLdoutput_matrix{static_cast<T*>(dL_ddLdoutput), m_model->padded_output_width(),
                                 n_elements};
        m_model->backward_backward_input(
            stream, *ctx.ctx, input_matrix, dL_ddLdinput_matrix, dL_doutput_matrix,
            dL_ddLdoutput ? &dL_ddLdoutput_matrix : nullptr,
            dL_dinput ? &dL_dinput_matrix : nullptr, false,
            dL_dparams ? GradientMode::Overwrite : GradientMode::Ignore);
    }

    uint32_t n_input_dims() const override { return m_model->input_width(); }

    uint32_t n_output_dims() const override { return m_model->padded_output_width(); }

    size_t n_params() const override { return m_model->n_params(); }

    void initialize_params(size_t seed, float* params_full_precision, float scale = 1) override {
        pcg32 rnd{seed};
        m_model->initialize_params(rnd, params_full_precision, scale);
    }

    json hyperparams() const override { return m_model->hyperparams(); }

    std::string name() const override { return m_model->name(); }

    bool jit_fusion() const override { return m_model->jit_fusion(); }

    void set_jit_fusion(bool val) override { m_model->set_jit_fusion(val); }

  private:
    static Precision precision() {
        return std::is_same<T, float>::value ? Precision::Fp32 : Precision::Fp16;
    }

    void bind(void* params, void* gradients) {
        m_model->set_params(static_cast<T*>(params), static_cast<T*>(params),
                            static_cast<T*>(gradients));
    }

    std::unique_ptr<tcnn::DifferentiableObject<float, T, T>> m_model;
};
} // namespace

// Preserve factories and ownership: callers delete the returned Module pointer.
Module* create_network_with_input_encoding(uint32_t n_input_dims, uint32_t n_output_dims,
                                           const json& encoding, const json& network) {
#ifdef TCNN_NO_NETWORKS
    (void)n_input_dims;
    (void)n_output_dims;
    (void)encoding;
    (void)network;
    throw std::runtime_error(
        "This tiny-metal-nn build excludes neural networks (TCNN_NO_NETWORKS).");
#else
    return new ModuleImpl<network_precision_t>{
        new tcnn::NetworkWithInputEncoding<network_precision_t>{n_input_dims, n_output_dims,
                                                                encoding, network}};
#endif
}

Module* create_network(uint32_t n_input_dims, uint32_t n_output_dims, const json& network) {
    return create_network_with_input_encoding(n_input_dims, n_output_dims, {{"otype", "Identity"}},
                                              network);
}

Module* create_encoding(uint32_t n_input_dims, const json& encoding,
                        Precision requested_precision) {
    if (requested_precision == Precision::Fp32)
        return new ModuleImpl<float>{tcnn::create_encoding<float>(n_input_dims, encoding, 0)};
    if (requested_precision == Precision::Fp16)
        return new ModuleImpl<half>{tcnn::create_encoding<half>(n_input_dims, encoding, 0)};
    throw std::invalid_argument("Unknown Metal encoding precision");
}
} // namespace cpp
} // namespace tcnn
