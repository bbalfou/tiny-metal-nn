#pragma once
#include <tiny-metal-nn/encoding.h>
#include <tiny-metal-nn/network.h>
#include <tiny-metal-nn/detail/mlp_dispatch.h>
#include <tiny-metal-nn/metal.h>

namespace tcnn {
// Compose a float-input encoding with a network, sharing the encoded device buffer.
// Parameter order is network first, encoding second, exactly as in the supplied ZIP.
template <typename T> class NetworkWithInputEncoding : public Network<float, T> {
  public:
    // Adopt compatible objects without changing the already chosen network shape.
    NetworkWithInputEncoding(std::shared_ptr<Encoding<T>> encoding,
                             std::shared_ptr<Network<T>> network)
        : m_encoding{std::move(encoding)}, m_network{std::move(network)} {
        nn_check(m_encoding && m_network, "encoding and network must be non-null");
        nn_check(m_encoding->padded_output_width() == m_network->input_width(),
                 "encoding/network width mismatch");
    }

    // Align the encoding for the selected network and inject JSON dimensions.
    NetworkWithInputEncoding(std::shared_ptr<Encoding<T>> encoding, uint32_t n_output_dims,
                             const json& network)
        : m_encoding{std::move(encoding)} {
        nn_check(static_cast<bool>(m_encoding), "encoding must be non-null");
        m_encoding->set_alignment(minimum_alignment(network));
        json local_network_config = network;
        local_network_config["n_input_dims"] = m_encoding->padded_output_width();
        local_network_config["n_output_dims"] = n_output_dims;
        m_network.reset(create_network<T>(local_network_config));
    }

    // Factory-style constructor with the same names and order as the supplied reference.
    NetworkWithInputEncoding(uint32_t n_dims_to_encode, uint32_t n_output_dims,
                             const json& encoding, const json& network)
        : NetworkWithInputEncoding{
              std::shared_ptr<Encoding<T>>{create_encoding<T>(n_dims_to_encode, encoding)},
              n_output_dims, network} {}

    virtual ~NetworkWithInputEncoding() = default;

    // Evaluate the encoding and MLP on one queue without an intermediate CPU copy.
#ifndef TCNN_NO_FWD_BWD
    void inference_mixed_precision_impl(metalStream_t stream, const GPUMatrixDynamic<float>& input,
                                        GPUMatrixDynamic<T>& output,
                                        bool use_inference_params = true) override {
        if (!input.n())
            return;
        auto batch = detail::resolve_stream(stream)->batch();
        GPUMatrixDynamic<T> network_input{m_encoding->padded_output_width(), input.n(), stream,
                                          m_encoding->preferred_output_layout()};
        m_encoding->inference_mixed_precision(stream, input, network_input, use_inference_params);
        m_network->inference_mixed_precision(stream, network_input, output, use_inference_params);
    }
#endif

    uint32_t num_encoded_dims() const { return m_encoding->padded_output_width(); }

    // Retain both child contexts and the exact encoded values used by the network.
#ifndef TCNN_NO_FWD_BWD
    std::unique_ptr<Context> forward_impl(metalStream_t stream,
                                          const GPUMatrixDynamic<float>& input,
                                          GPUMatrixDynamic<T>* output = nullptr,
                                          bool use_inference_params = false,
                                          bool prepare_input_gradients = false) override {
        auto batch = detail::resolve_stream(stream)->batch();
        auto forward = std::make_unique<ForwardContext>();
        forward->owner = this;
        if (!input.n())
            return forward;
        forward->network_input = GPUMatrixDynamic<T>{m_encoding->padded_output_width(), input.n(),
                                                     stream, m_encoding->preferred_output_layout()};
        forward->encoding_ctx = m_encoding->forward(stream, input, &forward->network_input,
                                                    use_inference_params, prepare_input_gradients);
        forward->network_ctx =
            m_network->forward(stream, forward->network_input, output, use_inference_params, true);
        return forward;
    }
#endif

    // Chain network gradients through the encoding only when those gradients are needed.
#ifndef TCNN_NO_FWD_BWD
    void backward_impl(metalStream_t stream, const Context& ctx,
                       const GPUMatrixDynamic<float>& input, const GPUMatrixDynamic<T>& output,
                       const GPUMatrixDynamic<T>& dL_doutput,
                       GPUMatrixDynamic<float>* dL_dinput = nullptr,
                       bool use_inference_params = false,
                       GradientMode param_gradients_mode = GradientMode::Overwrite) override {
        auto batch = detail::resolve_stream(stream)->batch();
        if (!input.n()) {
            zero_empty_gradient(stream, param_gradients_mode);
            return;
        }
        const auto& forward = checked_context(ctx);
        nn_check(forward.network_ctx && forward.encoding_ctx &&
                     forward.network_input.n() == input.n(),
                 "forward context batch does not match backward input");
        GPUMatrixDynamic<T> dL_dnetwork_input;
        if (m_encoding->n_params() || dL_dinput)
            dL_dnetwork_input = GPUMatrixDynamic<T>{m_encoding->padded_output_width(), input.n(),
                                                    stream, m_encoding->preferred_output_layout()};
        m_network->backward(stream, *forward.network_ctx, forward.network_input, output, dL_doutput,
                            dL_dnetwork_input.data() ? &dL_dnetwork_input : nullptr,
                            use_inference_params, param_gradients_mode);
        if (dL_dnetwork_input.data())
            m_encoding->backward(stream, *forward.encoding_ctx, input, forward.network_input,
                                 dL_dnetwork_input, dL_dinput, use_inference_params,
                                 param_gradients_mode);
    }
#endif

    // Apply the complete second-order chain rule across encoding and network.
    // Both the encoding curvature and the network curvature contribute to dx
    // and to a trainable encoding's parameter derivatives.
#ifndef TCNN_NO_FWD_BWD
    void backward_backward_input_impl(
        metalStream_t stream, const Context& ctx, const GPUMatrixDynamic<float>& input,
        const GPUMatrixDynamic<float>& dL_ddLdinput, const GPUMatrixDynamic<T>& dL_doutput,
        GPUMatrixDynamic<T>* dL_ddLdoutput = nullptr, GPUMatrixDynamic<float>* dL_dinput = nullptr,
        bool use_inference_params = false,
        GradientMode param_gradients_mode = GradientMode::Overwrite) override {
        auto batch = detail::resolve_stream(stream)->batch();
        if (!input.n()) {
            zero_empty_gradient(stream, param_gradients_mode);
            return;
        }
        const auto& forward = checked_context(ctx);
        nn_check(forward.network_ctx && forward.encoding_ctx &&
                     forward.network_input.n() == input.n(),
                 "forward context batch does not match double backward input");
        GPUMatrixDynamic<T> network_output{m_network->padded_output_width(), input.n(), stream,
                                           dL_doutput.layout()};
        m_network->inference_mixed_precision(stream, forward.network_input, network_output,
                                             use_inference_params);
        GPUMatrixDynamic<T> dL_dnetwork_input{m_encoding->padded_output_width(), input.n(), stream,
                                              m_encoding->preferred_output_layout()};
        GPUMatrixDynamic<T> dL_ddLdnetwork_input{m_encoding->padded_output_width(), input.n(),
                                                 stream, m_encoding->preferred_output_layout()};
        m_network->backward(stream, *forward.network_ctx, forward.network_input, network_output,
                            dL_doutput, &dL_dnetwork_input, use_inference_params,
                            GradientMode::Ignore);
        m_encoding->backward_backward_input(stream, *forward.encoding_ctx, input, dL_ddLdinput,
                                            dL_dnetwork_input, &dL_ddLdnetwork_input, dL_dinput,
                                            use_inference_params, param_gradients_mode);
        const bool need_encoding_adjoint =
            dL_dinput || (m_encoding->n_params() && param_gradients_mode != GradientMode::Ignore);
        GPUMatrixDynamic<T> encoding_adjoint;
        if (need_encoding_adjoint)
            encoding_adjoint = GPUMatrixDynamic<T>{m_encoding->padded_output_width(), input.n(),
                                                   stream, m_encoding->preferred_output_layout()};
        m_network->backward_backward_input(stream, *forward.network_ctx, forward.network_input,
                                           dL_ddLdnetwork_input, dL_doutput, dL_ddLdoutput,
                                           need_encoding_adjoint ? &encoding_adjoint : nullptr,
                                           use_inference_params, param_gradients_mode);
        if (need_encoding_adjoint) {
            GPUMatrixDynamic<float> extra_dL_dinput;
            if (dL_dinput)
                extra_dL_dinput =
                    GPUMatrixDynamic<float>{input.m(), input.n(), stream, dL_dinput->layout()};
            m_encoding->backward(
                stream, *forward.encoding_ctx, input, forward.network_input, encoding_adjoint,
                dL_dinput ? &extra_dL_dinput : nullptr, use_inference_params,
                param_gradients_mode == GradientMode::Ignore ? GradientMode::Ignore
                                                             : GradientMode::Accumulate);
            if (dL_dinput)
                detail::accumulate_matrix(stream, extra_dL_dinput, *dL_dinput);
        }
    }
#endif

    // Bind the original flat checkpoint representation without reordering values.
    void set_params_impl(T* params, T* inference_params, T* gradients) override {
        m_network->set_params(params, inference_params, gradients);
        size_t offset = m_network->n_params();
        m_encoding->set_params(params ? params + offset : nullptr,
                               inference_params ? inference_params + offset : nullptr,
                               gradients ? gradients + offset : nullptr);
    }

    void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override {
        m_network->initialize_params(rnd, params_full_precision, scale);
        m_encoding->initialize_params(
            rnd, params_full_precision ? params_full_precision + m_network->n_params() : nullptr,
            scale);
    }

    size_t n_params() const override {
        return metalrt::checked_add(m_network->n_params(), m_encoding->n_params());
    }

    uint32_t padded_output_width() const override { return m_network->padded_output_width(); }

    uint32_t output_width() const override { return m_network->output_width(); }

    uint32_t required_input_alignment() const override { return 1; }

    std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
        return m_network->layer_sizes();
    }

    uint32_t width(uint32_t layer) const override {
        return layer == 0 ? m_encoding->padded_output_width() : m_network->width(layer - 1);
    }

    uint32_t num_forward_activations() const override {
        return m_network->num_forward_activations() + 1;
    }

    // Layer zero exposes the encoding; subsequent layers use the network's layout.
    std::pair<const T*, MatrixLayout> forward_activations(const Context& ctx,
                                                          uint32_t layer) const override {
        if (this->is_jit_context(ctx))
            return this->jit_forward_activation(ctx, forward_activation_device_offset(layer),
                                                width(layer));
        const auto& forward = checked_context(ctx);
        if (layer && !forward.network_ctx)
            return {nullptr, CM};
        return layer == 0 ? std::make_pair(static_cast<const T*>(forward.network_input.data()),
                                           m_encoding->preferred_output_layout())
                          : m_network->forward_activations(*forward.network_ctx, layer - 1);
    }

    uint32_t forward_activation_device_offset(uint32_t layer) const override {
        nn_check(layer < num_forward_activations(), "forward activation layer out of range");
        nn_check(device_function_fwd_ctx_aligned_per_element(),
                 "custom cooperative JIT activation layouts require their own diagnostic accessor");
        return layer == 0 ? encoded_values_device_offset()
                          : m_network->forward_activation_device_offset(layer - 1);
    }

    uint32_t input_width() const override { return m_encoding->input_width(); }

    const std::shared_ptr<Encoding<T>>& encoding() const { return m_encoding; }

    json hyperparams() const override {
        return {{"otype", "NetworkWithInputEncoding"},
                {"encoding", m_encoding->hyperparams()},
                {"network", m_network->hyperparams()}};
    }

    bool jit_allows_float_gradient_accumulation() const override {
        return typeid(*this) == typeid(NetworkWithInputEncoding) &&
               (!m_network->n_params() || m_network->jit_allows_float_gradient_accumulation()) &&
               (!m_encoding->n_params() || m_encoding->jit_allows_float_gradient_accumulation());
    }

    bool jit_supports_fused_optimizer() const override {
        return typeid(*this) == typeid(NetworkWithInputEncoding) &&
               m_network->jit_supports_fused_optimizer() &&
               (!m_encoding->n_params() || m_encoding->jit_supports_fused_optimizer());
    }

    bool jit_supports_compact_training_context() const override {
        return typeid(*this) == typeid(NetworkWithInputEncoding) &&
               m_network->jit_supports_compact_training_context();
    }

    uint32_t device_function_training_fwd_ctx_bytes() const override {
        if (!jit_supports_compact_training_context())
            return device_function_fwd_ctx_bytes();
        return align_device_context(
            metalrt::checked_add(encoded_values_device_offset(true),
                                 metalrt::checked_mul(num_encoded_dims(), sizeof(T))));
    }

    std::string generate_training_device_function(const std::string& name) const override {
        return jit_supports_compact_training_context() ? generate_device_function_layout(name, true)
                                                       : generate_device_function(name);
    }

    std::string generate_training_backward_device_function(const std::string& name,
                                                           uint32_t threads) const override {
        return jit_supports_compact_training_context()
                   ? generate_backward_device_function_layout(name, threads, true)
                   : generate_backward_device_function(name, threads);
    }

    std::string generate_device_function(const std::string& name) const override {
        return generate_device_function_layout(name, false);
    }

    std::string generate_backward_device_function(const std::string& name,
                                                  uint32_t threads) const override {
        return generate_backward_device_function_layout(name, threads, false);
    }

    // Reverse the tangent of the complete composition. Both the encoding's
    // curvature and the network's curvature contribute to input/parameter grads.
    std::string
    generate_backward_backward_input_device_function(const std::string& name,
                                                     uint32_t n_threads) const override {
        const std::string encoding = name + "_encoding",
                          encoding_second = name + "_encoding_second", network = name + "_network",
                          network_second = name + "_network_second";
        std::ostringstream body;
        body << generated_child_environments() << "    " << m_encoding->generate_vec_out()
             << " encoding_gradient{}, encoding_direction{}, encoding_adjoint{};\n"
             << "    " << network
             << "(dL_dy, params, fwd_ctx, nullptr, &encoding_gradient, env ? &network_env : "
                "nullptr);\n"
             << "    " << encoding_second << "(dL_ddLdx, encoding_gradient, params + "
             << m_network->n_params() << "u, fwd_ctx + "
             << device_context_offset_expression(encoding_device_context_offset())
             << ", dL_dparams ? dL_dparams + " << m_network->n_params()
             << "u : nullptr, dL_dx, &encoding_direction, env);\n"
             << "    const bool needs_encoding_adjoint = dL_dx || (dL_dparams && "
             << m_encoding->n_params() << "u != 0u);\n"
             << "    " << network_second
             << "(encoding_direction, dL_dy, params, fwd_ctx, dL_dparams, needs_encoding_adjoint ? "
                "&encoding_adjoint : nullptr, "
             << "dL_ddLdy, env ? &network_env : nullptr);\n"
             << "    if (needs_encoding_adjoint) {\n"
             << "        " << this->generate_vec_in() << " extra_input_gradient{};\n"
             << "        " << encoding << "(encoding_adjoint, params + " << m_network->n_params()
             << "u, fwd_ctx + "
             << device_context_offset_expression(encoding_device_context_offset())
             << ", dL_dparams ? dL_dparams + " << m_network->n_params()
             << "u : nullptr, dL_dx ? &extra_input_gradient : nullptr, env);\n"
             << "        if (dL_dx)\n"
             << "            for (uint k = 0; k < " << input_width()
             << "u; ++k) (*dL_dx)[k] += extra_input_gradient[k];\n"
             << "    }\n";
        return m_network->generate_backward_device_function(network, n_threads) + "\n" +
               m_network->generate_backward_backward_input_device_function(network_second,
                                                                           n_threads) +
               "\n" + m_encoding->generate_backward_device_function(encoding, n_threads) + "\n" +
               m_encoding->generate_backward_backward_input_device_function(encoding_second,
                                                                            n_threads) +
               "\n" +
               this->generate_backward_backward_input_device_function_from_body(name, body.str());
    }

    std::vector<const float*> jit_runtime_inputs() const override {
        auto result = m_encoding->jit_runtime_inputs();
        const auto network_inputs = m_network->jit_runtime_inputs();
        result.insert(result.end(), network_inputs.begin(), network_inputs.end());
        return result;
    }

    std::string jit_configuration_key() const override {
        return json{{"base", Network<float, T>::jit_configuration_key()},
                    {"encoding", m_encoding->jit_configuration_key()},
                    {"network", m_network->jit_configuration_key()}}
            .dump();
    }

    uint32_t device_function_fwd_ctx_bytes() const override {
        return align_device_context(metalrt::checked_add(
            encoded_values_device_offset(), metalrt::checked_mul(num_encoded_dims(), sizeof(T))));
    }

    bool device_function_fwd_ctx_aligned_per_element() const override {
        return m_network->device_function_fwd_ctx_aligned_per_element() &&
               m_encoding->device_function_fwd_ctx_aligned_per_element();
    }

    uint32_t
    backward_device_function_shmem_bytes(uint32_t n_threads,
                                         GradientMode param_gradients_mode) const override {
        return std::max(
            m_network->backward_device_function_shmem_bytes(n_threads, param_gradients_mode),
            m_encoding->backward_device_function_shmem_bytes(n_threads, param_gradients_mode));
    }

  private:
    std::string generate_device_function_layout(const std::string& name, bool training) const {
        const std::string encoding = name + "_encoding", network = name + "_network";
        std::ostringstream body;
        body << generated_child_environments() << "    const auto encoded = " << encoding
             << "(input, params + " << m_network->n_params() << "u, fwd_ctx ? fwd_ctx + "
             << device_context_offset_expression(encoding_device_context_offset(training))
             << " : nullptr, env);\n"
             << "    if (fwd_ctx) {\n"
             << "        auto values = reinterpret_cast<TCNN_JIT_CTX " << type_to_string<T>()
             << "*>(fwd_ctx + "
             << device_context_offset_expression(encoded_values_device_offset(training));
        if (!device_function_fwd_ctx_aligned_per_element())
            body << " + (env ? env->lane_index : 0u) * " << size_t(num_encoded_dims()) * sizeof(T)
                 << "u";
        body << ");\n"
             << "        for (uint k = 0; k < " << num_encoded_dims()
             << "u; ++k) values[k] = encoded[k];\n"
             << "    }\n"
             << "    return " << network
             << "(encoded, params, fwd_ctx, env ? &network_env : nullptr);\n";
        return (training ? m_network->generate_training_device_function(network)
                         : m_network->generate_device_function(network)) +
               "\n" + m_encoding->generate_device_function(encoding) + "\n" +
               detail::jit_forward_function(name, this->generate_vec_in(), this->generate_vec_out(),
                                            type_to_string<T>(), body.str(),
                                            training ? device_function_training_fwd_ctx_bytes()
                                                     : device_function_fwd_ctx_bytes(),
                                            device_function_fwd_ctx_aligned_per_element());
    }

    std::string generate_backward_device_function_layout(const std::string& name,
                                                         uint32_t n_threads, bool training) const {
        const std::string encoding = name + "_encoding", network = name + "_network";
        std::ostringstream body;
        body << generated_child_environments()
             << "    const bool needs_encoding = dL_dx || (dL_dparams && " << m_encoding->n_params()
             << "u != 0u);\n"
             << "    " << m_encoding->generate_vec_out() << " encoding_gradient{};\n"
             << "    " << network
             << "(dL_dy, params, fwd_ctx, dL_dparams, needs_encoding ? &encoding_gradient : "
                "nullptr, env ? &network_env : nullptr);\n"
             << "    if (needs_encoding)\n"
             << "        " << encoding << "(encoding_gradient, params + " << m_network->n_params()
             << "u, fwd_ctx + "
             << device_context_offset_expression(encoding_device_context_offset(training))
             << ", dL_dparams ? dL_dparams + " << m_network->n_params()
             << "u : nullptr, dL_dx, env);\n";
        return (training ? m_network->generate_training_backward_device_function(network, n_threads)
                         : m_network->generate_backward_device_function(network, n_threads)) +
               "\n" + m_encoding->generate_backward_device_function(encoding, n_threads) + "\n" +
               detail::jit_backward_function(name, this->generate_vec_in(),
                                             this->generate_vec_out(), type_to_string<T>(),
                                             body.str(),
                                             training ? device_function_training_fwd_ctx_bytes()
                                                      : device_function_fwd_ctx_bytes(),
                                             device_function_fwd_ctx_aligned_per_element());
    }

    static uint32_t align_device_context(size_t bytes) {
        bytes = metalrt::checked_add(bytes, 3u) & ~size_t{3};
        nn_check(bytes <= UINT32_MAX, "composed generated forward context exceeds uint32_t");
        return static_cast<uint32_t>(bytes);
    }

    uint32_t encoding_device_context_offset(bool training = false) const {
        return align_device_context(training ? m_network->device_function_training_fwd_ctx_bytes()
                                             : m_network->device_function_fwd_ctx_bytes());
    }

    uint32_t encoded_values_device_offset(bool training = false) const {
        return align_device_context(metalrt::checked_add(
            encoding_device_context_offset(training), m_encoding->device_function_fwd_ctx_bytes()));
    }

    std::string generated_child_environments() const {
        std::ostringstream code;
        code << "    tcnn_jit_environment network_env{};\n"
             << "    if (env) { network_env = *env; network_env.offset += "
             << m_encoding->jit_runtime_inputs().size() << "u; }\n";
        return code.str();
    }

    // A cooperative child owns a SIMD-group-wide region. If either child uses
    // that layout, split the parent's group region into contiguous child group
    // regions; per-element children then select their own lane within the slice.
    std::string device_context_offset_expression(uint32_t offset) const {
        const std::string bytes = std::to_string(offset) + "u";
        return device_function_fwd_ctx_aligned_per_element()
                   ? bytes
                   : "((env ? env->lane_count : 1u) * " + bytes + ")";
    }

    struct ForwardContext : Context {
        const NetworkWithInputEncoding* owner = nullptr;
        GPUMatrixDynamic<T> network_input;
        std::unique_ptr<Context> encoding_ctx, network_ctx;
    };

    const ForwardContext& checked_context(const Context& ctx) const {
        auto* p = dynamic_cast<const ForwardContext*>(&ctx);
        nn_check(p && p->owner == this, "forward context belongs to another composed network");
        return *p;
    }

    void zero_empty_gradient(metalStream_t stream, GradientMode mode) {
        if (mode == GradientMode::Overwrite && n_params())
            detail::resolve_stream(stream)->fill(
                metalrt::resolve_pointer(this->gradients(), n_params() * sizeof(T)), 0);
    }

    std::shared_ptr<Encoding<T>> m_encoding;
    std::shared_ptr<Network<T>> m_network;
};
} // namespace tcnn
