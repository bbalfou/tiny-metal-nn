#pragma once
#include <tiny-metal-nn/network.h>
#include <tiny-metal-nn/detail/mlp_dispatch.h>
#include <tiny-metal-nn/detail/fused_mlp_dispatch.h>
#include <tiny-metal-nn/metal.h>

namespace tcnn {
namespace detail {
// Common implementation only: every numerical pass uses metalrt compute dispatch.
// One output element per thread deliberately favors an inspectable correctness
// baseline; tiled GEMM kernels can replace these passes without public API edits.
template <class T> class BaselineMLP : public Network<T> {
  public:
    bool jit_allows_float_gradient_accumulation() const override {
        return typeid(*this) == typeid(BaselineMLP);
    }

    bool jit_supports_fused_optimizer() const override {
        return typeid(*this) == typeid(BaselineMLP);
    }

    bool jit_supports_compact_training_context() const override {
        return typeid(*this) == typeid(BaselineMLP);
    }

    // Bind the flattened row-major matrices in the uploaded checkpoint order.
    void set_params_impl(T* params, T* inference_params, T* gradients) override {
        size_t current_pos = 0;
        for (size_t i = 0; i < m_weight_matrices.size(); ++i) {
            m_weight_matrices[i].set_data_unsafe(params ? params + current_pos : nullptr);
            m_weight_matrices_inference[i].set_data_unsafe(
                inference_params ? inference_params + current_pos : nullptr);
            m_gradient_matrices[i].set_data_unsafe(gradients ? gradients + current_pos : nullptr);
            current_pos += m_weight_matrices[i].n_elements();
        }
    }

    // Xavier/SIREN initialization uses the reference matrix initializers and PCG state.
    void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override {
        size_t offset = 0;
        for (size_t i = 0; i < m_weight_matrices.size(); ++i) {
            const auto& m = m_weight_matrices[i];
            GPUMatrix<float, RM> matrix{params_full_precision + offset, m.m(), m.n()};
            if (m_activation == Activation::Sine) {
                if (i == 0)
                    matrix.initialize_siren_uniform_first(rnd, scale);
                else
                    matrix.initialize_siren_uniform(rnd, scale);
            } else
                matrix.initialize_xavier_uniform(rnd, scale);
            offset += m.n_elements();
        }
    }

    // Return the same diagnostic weight/gradient matrix views as the reference.
    GPUMatrix<T, RM>& input_weight_matrix(bool inference) {
        return (inference ? m_weight_matrices_inference : m_weight_matrices).front();
    }

    GPUMatrix<T, RM>& weight_matrix_at(bool inference, uint32_t idx) {
        return (inference ? m_weight_matrices_inference : m_weight_matrices).at(1 + idx);
    }

    GPUMatrix<T, RM>& output_weight_matrix(bool inference) {
        return (inference ? m_weight_matrices_inference : m_weight_matrices).back();
    }

    GPUMatrix<T, RM>& input_gradient_matrix() { return m_gradient_matrices.front(); }

    GPUMatrix<T, RM>& gradient_matrix_at(uint32_t idx) { return m_gradient_matrices.at(1 + idx); }

    GPUMatrix<T, RM>& output_gradient_matrix() { return m_gradient_matrices.back(); }

    size_t n_params() const override { return m_total_n_params; }

    uint32_t input_width() const override { return m_input_width; }

    uint32_t padded_output_width() const override { return m_padded_output_width; }

    uint32_t output_width() const override { return m_output_width; }

    static uint32_t REQUIRED_ALIGNMENT() { return 16; }

    uint32_t required_input_alignment() const override { return REQUIRED_ALIGNMENT(); }

    std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
        std::vector<std::pair<uint32_t, uint32_t>> result;
        for (const auto& m : m_weight_matrices)
            result.emplace_back(m.m(), m.n());
        return result;
    }

    uint32_t width(uint32_t layer) const override {
        (void)layer;
        return m_network_width;
    }

    uint32_t num_forward_activations() const override {
        return m_can_fuse_activation ? m_n_hidden_layers : m_n_hidden_layers * 2;
    }

    // Access the original forward context, rejecting contexts from another model.
    std::pair<const T*, MatrixLayout> forward_activations(const Context& ctx,
                                                          uint32_t layer) const override {
        if (this->is_jit_context(ctx))
            return this->jit_forward_activation(ctx, forward_activation_device_offset(layer),
                                                m_network_width);
        const auto& forward = checked_context(ctx);
        if (m_can_fuse_activation)
            return {forward.hidden.at(layer).data(), CM};
        return {layer % 2 ? forward.hidden.at(layer / 2).data() : forward.pre.at(layer / 2).data(),
                CM};
    }

    uint32_t forward_activation_device_offset(uint32_t layer) const override {
        nn_check(layer < num_forward_activations(), "forward activation layer out of range");
        const uint32_t hidden_layer = m_can_fuse_activation ? layer : layer / 2;
        const bool output = m_can_fuse_activation || (layer % 2 != 0);
        const size_t element_offset = size_t(m_input_width) +
                                      2u * size_t(hidden_layer) * m_network_width +
                                      (output ? m_network_width : 0u);
        nn_check(element_offset * sizeof(T) <= UINT32_MAX,
                 "forward activation offset exceeds uint32_t");
        return static_cast<uint32_t>(element_offset * sizeof(T));
    }

    json hyperparams() const override {
        return {{"otype", m_otype},
                {"activation", to_string(m_activation)},
                {"output_activation", to_string(m_output_activation)},
                {"n_neurons", m_network_width},
                {"n_hidden_layers", m_n_hidden_layers}};
    }

    // Inference keeps the training shape/layout/precision contract. The supplied
    // SIMD-group specialization writes the caller's output directly; every other
    // case continues through the existing whole-model or layerwise implementation.
#ifndef TCNN_NO_FWD_BWD
    void inference_mixed_precision_impl(metalStream_t stream, const GPUMatrixDynamic<T>& input,
                                        GPUMatrixDynamic<T>& output,
                                        bool use_inference_params = true) override {
        if (m_path == network_kernels::Path::FullyFused) {
            const auto& weights =
                use_inference_params ? m_weight_matrices_inference : m_weight_matrices;
            if (try_fused_inference(stream, input, weights.front().data(), m_total_n_params, output,
                                    fused_shape()))
                return;
        }
        // The Reference backend and devices without the fused capability use
        // the same required training-forward operation as before.
        auto ctx = forward_impl(stream, input, &output, use_inference_params, false);
        (void)ctx;
    }
#endif

    // Save preactivations and hidden outputs. Buffer ownership is retained by each
    // recorded dispatch even if the host context is destroyed before completion.
#ifndef TCNN_NO_FWD_BWD
    std::unique_ptr<Context> forward_impl(metalStream_t stream, const GPUMatrixDynamic<T>& input,
                                          GPUMatrixDynamic<T>* output = nullptr,
                                          bool use_inference_params = false,
                                          bool prepare_input_gradients = false) override {
        (void)prepare_input_gradients;
        auto batch = resolve_stream(stream)->batch();
        auto forward = std::make_unique<ForwardContext>();
        forward->owner = this;
        forward->batch_size = input.n();
        const auto& weights =
            use_inference_params ? m_weight_matrices_inference : m_weight_matrices;
        forward->pre.reserve(weights.size());
        forward->hidden.reserve(weights.size());
        if (m_path == network_kernels::Path::FullyFused) {
            const auto count = fused_storage_count(fused_shape(), input.n());
            forward->fused_pre = GPUMatrixDynamic<T>{uint32_t(count), 1u, stream};
            forward->fused_hidden = GPUMatrixDynamic<T>{uint32_t(count), 1u, stream};
            size_t offset = 0;
            for (const auto& weight : weights) {
                forward->pre.emplace_back(count ? forward->fused_pre.data() + offset : nullptr,
                                          weight.m(), input.n());
                forward->hidden.emplace_back(
                    count ? forward->fused_hidden.data() + offset : nullptr, weight.m(), input.n());
                offset += size_t(weight.m()) * input.n();
            }
            fused_forward(stream, input, weights.front().data(), m_total_n_params,
                          forward->fused_pre, forward->fused_hidden, fused_shape());
            if (output)
                copy_matrix(stream, forward->hidden.back(), *output);
            return forward;
        }
        const GPUMatrixDynamic<T>* layer_input = &input;
        for (size_t i = 0; i < weights.size(); ++i) {
            forward->pre.emplace_back(weights[i].m(), input.n(), stream);
            forward->hidden.emplace_back(weights[i].m(), input.n(), stream);
            const bool last = i + 1 == weights.size();
            dense_forward(stream, *layer_input, weights[i], forward->pre.back(),
                          forward->hidden.back(), last ? m_output_activation : m_activation);
            layer_input = &forward->hidden.back();
        }
        if (output)
            copy_matrix(stream, forward->hidden.back(), *output);
        return forward;
    }
#endif

    // Reverse every layer, respecting Ignore/Overwrite/Accumulate exactly at the
    // parameter-gradient destination; dL/dinput is optional and independently computed.
#ifndef TCNN_NO_FWD_BWD
    void backward_impl(metalStream_t stream, const Context& ctx, const GPUMatrixDynamic<T>& input,
                       const GPUMatrixDynamic<T>& output, const GPUMatrixDynamic<T>& dL_doutput,
                       GPUMatrixDynamic<T>* dL_dinput = nullptr, bool use_inference_params = false,
                       GradientMode param_gradients_mode = GradientMode::Overwrite) override {
        const auto& forward = checked_context(ctx);
        nn_check(forward.batch_size == input.n(), "forward batch mismatch");
        auto batch = resolve_stream(stream)->batch();
        const auto& weights =
            use_inference_params ? m_weight_matrices_inference : m_weight_matrices;
        if (m_path == network_kernels::Path::FullyFused) {
            fused_backward(stream, input, output, dL_doutput, dL_dinput, weights.front().data(),
                           m_gradient_matrices.front().data(), m_total_n_params, forward.fused_pre,
                           forward.fused_hidden, fused_shape(), param_gradients_mode);
            return;
        }
        GPUMatrixDynamic<T> incoming;
        const GPUMatrixDynamic<T>* derivative = &dL_doutput;
        for (size_t i = weights.size(); i-- > 0;) {
            GPUMatrixDynamic<T> delta{weights[i].m(), input.n(), stream};
            const bool last = i + 1 == weights.size();
            // Invertible activations consume rounded outputs, preserving the
            // ordinary precision contract. Sine/SiLU use saved preactivations
            // for every layer, including the output layer.
            const auto activation = last ? m_output_activation : m_activation;
            const bool use_output =
                activation != Activation::Sine && activation != Activation::SiLU;
            const auto& values = use_output ? (last ? output : forward.hidden[i]) : forward.pre[i];
            dense_delta(stream, values, *derivative, delta, activation, use_output);
            if (param_gradients_mode != GradientMode::Ignore)
                dense_weight_gradient(stream, i ? forward.hidden[i - 1] : input, delta,
                                      m_gradient_matrices[i], param_gradients_mode);
            if (i) {
                GPUMatrixDynamic<T> next{weights[i].n(), input.n(), stream};
                dense_input_gradient(stream, delta, weights[i], next);
                incoming = std::move(next);
                derivative = &incoming;
            } else if (dL_dinput)
                dense_input_gradient(stream, delta, weights[i], *dL_dinput);
        }
    }
#endif

    // Compute the derivative of v dot dL/dinput by forward-over-reverse AD.
    // This supports every layer count and activation, returning J*v, the input
    // Hessian-vector term, and parameter cross derivatives without CPU readback.
#ifndef TCNN_NO_FWD_BWD
    void backward_backward_input_impl(
        metalStream_t stream, const Context& ctx, const GPUMatrixDynamic<T>& input,
        const GPUMatrixDynamic<T>& dL_ddLdinput, const GPUMatrixDynamic<T>& dL_doutput,
        GPUMatrixDynamic<T>* dL_ddLdoutput = nullptr, GPUMatrixDynamic<T>* dL_dinput = nullptr,
        bool use_inference_params = false,
        GradientMode param_gradients_mode = GradientMode::Overwrite) override {
        const auto& forward = checked_context(ctx);
        nn_check(forward.batch_size == input.n(), "forward batch mismatch");
        auto batch = resolve_stream(stream)->batch();
        const auto& weights =
            use_inference_params ? m_weight_matrices_inference : m_weight_matrices;
        if (m_path == network_kernels::Path::FullyFused) {
            fused_second_backward(stream, input, dL_ddLdinput, dL_doutput, dL_ddLdoutput, dL_dinput,
                                  weights.front().data(), m_gradient_matrices.front().data(),
                                  m_total_n_params, forward.fused_pre, forward.fused_hidden,
                                  fused_shape(), param_gradients_mode);
            return;
        }
        std::vector<GPUMatrixDynamic<T>> tangent_pre, tangent;
        tangent_pre.reserve(weights.size());
        tangent.reserve(weights.size());
        const GPUMatrixDynamic<T>* layer_tangent = &dL_ddLdinput;
        for (size_t i = 0; i < weights.size(); ++i) {
            tangent_pre.emplace_back(weights[i].m(), input.n(), stream);
            tangent.emplace_back(weights[i].m(), input.n(), stream);
            dense_forward(stream, *layer_tangent, weights[i], tangent_pre.back(), tangent.back(),
                          Activation::None);
            const auto activation = i + 1 == weights.size() ? m_output_activation : m_activation;
            const bool use_output =
                activation != Activation::Sine && activation != Activation::SiLU;
            dense_delta(stream, use_output ? forward.hidden[i] : forward.pre[i], tangent.back(),
                        tangent.back(), activation, use_output);
            layer_tangent = &tangent.back();
        }
        if (dL_ddLdoutput)
            copy_matrix(stream, tangent.back(), *dL_ddLdoutput);

        GPUMatrixDynamic<T> adjoint{m_padded_output_width, input.n(), stream};
        adjoint.memset_async(stream, 0);
        GPUMatrixDynamic<T> tangent_adjoint_storage;
        const GPUMatrixDynamic<T>* tangent_adjoint = &dL_doutput;
        for (size_t i = weights.size(); i-- > 0;) {
            GPUMatrixDynamic<T> delta{weights[i].m(), input.n(), stream};
            GPUMatrixDynamic<T> tangent_delta{weights[i].m(), input.n(), stream};
            dense_second_delta(stream, forward.pre[i], forward.hidden[i], tangent_pre[i], adjoint,
                               *tangent_adjoint, delta, tangent_delta,
                               i + 1 == weights.size() ? m_output_activation : m_activation);
            if (param_gradients_mode != GradientMode::Ignore) {
                dense_weight_gradient(stream, i ? forward.hidden[i - 1] : input, delta,
                                      m_gradient_matrices[i], param_gradients_mode);
                dense_weight_gradient(stream, i ? tangent[i - 1] : dL_ddLdinput, tangent_delta,
                                      m_gradient_matrices[i], GradientMode::Accumulate);
            }
            if (i) {
                GPUMatrixDynamic<T> next_adjoint{weights[i].n(), input.n(), stream};
                GPUMatrixDynamic<T> next_tangent_adjoint{weights[i].n(), input.n(), stream};
                dense_input_gradient(stream, delta, weights[i], next_adjoint);
                dense_input_gradient(stream, tangent_delta, weights[i], next_tangent_adjoint);
                adjoint = std::move(next_adjoint);
                tangent_adjoint_storage = std::move(next_tangent_adjoint);
                tangent_adjoint = &tangent_adjoint_storage;
            } else if (dL_dinput)
                dense_input_gradient(stream, delta, weights[i], *dL_dinput);
        }
    }
#endif

    // The generated context owns the original input and all preactivations and
    // outputs. Backward functions therefore need no hidden host-side model data.
    std::string generate_device_function(const std::string& name) const override {
        return this->generate_device_function_from_body(
            name, generate_mlp_device_code<T>(m_input_width, m_network_width, m_padded_output_width,
                                              m_output_width, m_n_hidden_layers, m_activation,
                                              m_output_activation));
    }

    std::string generate_backward_device_function(const std::string& name,
                                                  uint32_t n_threads) const override {
        return this->generate_backward_device_function_from_body(
            name, generate_backward_mlp_device_code<T>(
                      n_threads, m_input_width, m_network_width, m_padded_output_width,
                      m_output_width, m_n_hidden_layers, m_activation, m_output_activation));
    }

    std::string
    generate_backward_backward_input_device_function(const std::string& name,
                                                     uint32_t n_threads) const override {
        return this->generate_backward_backward_input_device_function_from_body(
            name, generate_backward_backward_input_mlp_device_code<T>(
                      n_threads, m_input_width, m_network_width, m_padded_output_width,
                      m_output_width, m_n_hidden_layers, m_activation, m_output_activation));
    }

    uint32_t device_function_fwd_ctx_bytes() const override {
        return mlp_device_code_fwd_ctx_bytes<T>(
            m_input_width, m_network_width, m_padded_output_width, m_output_width,
            m_n_hidden_layers, m_activation, m_output_activation);
    }

    // Only the private automatic-training boundary uses packed saved outputs.
    // Public forward/derivative callables retain their existing context ABI.
    uint32_t device_function_training_fwd_ctx_bytes() const override {
        if (!this->jit_supports_compact_training_context())
            return this->device_function_fwd_ctx_bytes();
        return mlp_jit_training_context_bytes(sizeof(T), m_input_width, m_network_width,
                                              m_padded_output_width, m_n_hidden_layers,
                                              m_activation, m_output_activation);
    }

    std::string generate_training_device_function(const std::string& name) const override {
        if (!compact_training_context())
            return this->generate_device_function(name);
        const auto bytes = device_function_training_fwd_ctx_bytes();
        return "// TCNN_JIT_PRIVATE_TRAINING_CONTEXT_BYTES: " + std::to_string(bytes) + "\n" +
               jit_forward_function(
                   name, this->generate_vec_in(), this->generate_vec_out(), type_to_string<T>(),
                   generate_mlp_body(type_to_string<T>(), m_input_width, m_network_width,
                                     m_padded_output_width, m_output_width, m_n_hidden_layers,
                                     m_activation, m_output_activation, 0, true),
                   bytes, true);
    }

    std::string generate_training_backward_device_function(const std::string& name,
                                                           uint32_t threads) const override {
        if (!compact_training_context())
            return this->generate_backward_device_function(name, threads);
        nn_check(threads != 0, "generated training backward requires a positive thread count");
        return jit_backward_function(
            name, this->generate_vec_in(), this->generate_vec_out(), type_to_string<T>(),
            generate_mlp_body(type_to_string<T>(), m_input_width, m_network_width,
                              m_padded_output_width, m_output_width, m_n_hidden_layers,
                              m_activation, m_output_activation, 1, true),
            device_function_training_fwd_ctx_bytes(), true);
    }

    bool device_function_fwd_ctx_aligned_per_element() const override { return true; }

    // SIMD-group products and dW reductions reuse one explicitly sized plane.
    uint32_t
    backward_device_function_shmem_bytes(uint32_t n_threads,
                                         GradientMode param_gradients_mode) const override {
        return backward_mlp_device_code_shmem_bytes<T>(
            n_threads, param_gradients_mode, m_input_width,
            m_n_hidden_layers ? m_network_width : 0u, m_padded_output_width);
    }

  private:
    bool compact_training_context() const {
        return this->jit_supports_compact_training_context() &&
               device_function_training_fwd_ctx_bytes() < this->device_function_fwd_ctx_bytes();
    }

  protected:
    // Allocate host descriptors for the bias-free matrices and output padding.
    BaselineMLP(uint32_t input_width, uint32_t network_width, uint32_t output_width,
                uint32_t n_hidden_layers, Activation activation, Activation output_activation,
                std::string otype, network_kernels::Path path)
        : m_input_width{input_width}, m_network_width{network_width}, m_output_width{output_width},
          m_n_hidden_layers{n_hidden_layers}, m_activation{activation},
          m_output_activation{output_activation}, m_otype{std::move(otype)}, m_path{path} {
        nn_check(input_width && output_width, "network dimensions must be positive");
        nn_check(input_width % 16 == 0, "network input width must be aligned to 16");
        nn_check(!n_hidden_layers || network_width, "network width must be positive");
        nn_check(!n_hidden_layers || m_path == network_kernels::Path::Gemm ||
                     network_width % 16 == 0,
                 "FullyFusedMLP network width must be aligned to 16");
        m_padded_output_width = next_multiple(output_width, 16u);
        m_can_fuse_activation = activation != Activation::Sine && activation != Activation::SiLU;
        uint32_t previous = input_width;
        for (uint32_t i = 0; i <= n_hidden_layers; ++i) {
            uint32_t rows = i == n_hidden_layers ? m_padded_output_width : network_width;
            m_weight_matrices.emplace_back(nullptr, rows, previous);
            m_weight_matrices_inference.emplace_back(nullptr, rows, previous);
            m_gradient_matrices.emplace_back(nullptr, rows, previous);
            m_total_n_params =
                metalrt::checked_add(m_total_n_params, metalrt::checked_mul(rows, previous));
            previous = rows;
        }
        if (m_path == network_kernels::Path::FullyFused)
            nn_check(m_total_n_params <= UINT32_MAX, "fused MLP parameters exceed 32-bit indexing");
    }

    struct ForwardContext : Context {
        const BaselineMLP* owner = nullptr;
        uint32_t batch_size = 0;
        std::vector<GPUMatrixDynamic<T>> pre, hidden;
        GPUMatrixDynamic<T> fused_pre, fused_hidden;
    };

    const ForwardContext& checked_context(const Context& ctx) const {
        const auto* forward = dynamic_cast<const ForwardContext*>(&ctx);
        nn_check(forward && forward->owner == this, "forward context belongs to another network");
        return *forward;
    }

    network_kernels::FusedShape fused_shape() const {
        return {m_input_width,     m_network_width,        m_padded_output_width,
                m_n_hidden_layers, uint32_t(m_activation), uint32_t(m_output_activation)};
    }

    uint32_t m_input_width, m_network_width, m_output_width, m_n_hidden_layers,
        m_padded_output_width = 0;
    Activation m_activation, m_output_activation;
    std::string m_otype;
    network_kernels::Path m_path;
    bool m_can_fuse_activation = true;
    size_t m_total_n_params = 0;
    std::vector<GPUMatrix<T, RM>> m_weight_matrices, m_weight_matrices_inference,
        m_gradient_matrices;
};
} // namespace detail
} // namespace tcnn
