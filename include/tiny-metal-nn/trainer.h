#pragma once
#include <tiny-metal-nn/loss.h>
#include <tiny-metal-nn/optimizer.h>
#include <tiny-metal-nn/optimizers/adam.h>
#include <tiny-metal-nn/metal_graph.h>
#include <tiny-metal-nn/gpu_memory_json.h>
#include <tiny-metal-nn/random.h>
#include <tiny-metal-nn/reduce_sum.h>
#include <tiny-metal-nn/detail/trainer_dispatch.h>
#include <random>

namespace tcnn {
// Own parameter storage, optimizer state and forward/loss/backward orchestration.
// Model/optimizer/loss interfaces and public argument names follow the uploaded ZIP.
template <typename T, typename PARAMS_T, typename COMPUTE_T = T>
class Trainer : public ObjectWithMutableHyperparams {
  public:
    Trainer(std::shared_ptr<DifferentiableObject<T, PARAMS_T, COMPUTE_T>> model,
            std::shared_ptr<Optimizer<PARAMS_T>> optimizer, std::shared_ptr<Loss<COMPUTE_T>> loss,
            uint32_t seed = 1337, float perturbation_sigma = 0)
        : m_model{std::move(model)}, m_optimizer{std::move(optimizer)}, m_loss{std::move(loss)},
          m_perturbation_sigma{perturbation_sigma} {
        nn_check(m_model && m_optimizer && m_loss,
                 "Trainer model, optimizer and loss must be non-null");
        std::seed_seq seq{seed};
        std::vector<uint32_t> seeds(2);
        seq.generate(seeds.begin(), seeds.end());
        m_rng = pcg32{seeds.front()};
        initialize_params();
    }

    virtual ~Trainer() = default;

    // Replace the objective while keeping current weights and optimizer state.
    void set_loss(std::shared_ptr<Loss<COMPUTE_T>> loss) {
        nn_check(static_cast<bool>(loss), "Trainer: may not set loss to nullptr");
        m_loss = std::move(loss);
        m_jit_fused_kernel.reset();
        m_jit_fused_key.clear();
    }

    // Reset optimizer buffers and initialize the exact FP32 + parameter + gradient layout.
    void initialize_params() {
        size_t n_params = m_model->n_params();
        m_optimizer->allocate(m_model);
        m_params_buffer.resize(
            metalrt::checked_mul(n_params, 2 * sizeof(PARAMS_T) + sizeof(float)));
        m_params_buffer.memset(0);
        reset_param_pointers();
        m_model->initialize_params(m_rng, m_params_full_precision);
        set_params_full_precision(m_params_full_precision, n_params, true);
        m_graph.reset();
    }

    struct ForwardContext : public Context {
        GPUMatrix<COMPUTE_T> perturbed_output;
        GPUMatrix<COMPUTE_T> output;
        GPUMatrix<COMPUTE_T> dL_doutput;
        GPUMatrix<float> L;
        std::unique_ptr<Context> model_ctx;
    };

    // Run the model and objective, retaining all buffers needed by backward.
    // An external derivative bypasses loss evaluation and target validation.
    std::unique_ptr<ForwardContext>
    forward(metalStream_t stream, const float loss_scale, const GPUMatrixDynamic<T>& input,
            const GPUMatrix<float>& target, const GPUMatrix<float>* data_pdf = nullptr,
            bool use_inference_params = false, bool prepare_input_gradients = false,
            const GPUMatrix<COMPUTE_T>* external_dL_dy = nullptr) {
        auto batch = detail::resolve_stream(stream)->batch();
        const uint32_t batch_size = input.n();
        auto forward = std::make_unique<ForwardContext>();
        forward->output = GPUMatrix<COMPUTE_T>{m_model->padded_output_width(), batch_size, stream};
        forward->model_ctx = m_model->forward(stream, input, &forward->output, use_inference_params,
                                              prepare_input_gradients);
        if (m_perturbation_sigma > 0) {
            GPUMatrix<float> perturbation{m_model->padded_output_width(), batch_size, stream};
            forward->perturbed_output =
                GPUMatrix<COMPUTE_T>{m_model->padded_output_width(), batch_size, stream};
            generate_random_logistic<float>(stream, m_rng, perturbation.n_elements(),
                                            perturbation.data(), 0.0f, m_perturbation_sigma);
            detail::add_perturbation(stream, forward->output, perturbation,
                                     forward->perturbed_output);
        }
        auto& loss_input = m_perturbation_sigma > 0 ? forward->perturbed_output : forward->output;
        forward->L = GPUMatrix<float>{m_model->padded_output_width(), batch_size, stream};
        if (external_dL_dy) {
            nn_check(external_dL_dy->m() == m_model->padded_output_width() &&
                         external_dL_dy->n() == batch_size,
                     "external derivative shape mismatch");
            forward->dL_doutput = GPUMatrix<COMPUTE_T>{external_dL_dy->data(),
                                                       m_model->padded_output_width(), batch_size};
            // The reference leaves this unused buffer unspecified; zeroing avoids
            // returning allocator contents if a caller nevertheless requests loss().
            forward->L.memset_async(stream, 0);
        } else {
            nn_check(input.n() == target.n() && m_model->output_width() == target.m(),
                     "training target shape mismatch");
            forward->dL_doutput =
                GPUMatrix<COMPUTE_T>{m_model->padded_output_width(), batch_size, stream};
            m_loss->evaluate(stream, loss_scale, loss_input, target, forward->L,
                             forward->dL_doutput, data_pdf);
        }
        return forward;
    }

    std::unique_ptr<ForwardContext>
    forward(const float loss_scale, const GPUMatrixDynamic<T>& input,
            const GPUMatrix<float>& target, const GPUMatrix<float>* data_pdf = nullptr,
            bool use_inference_params = false, bool prepare_input_gradients = false,
            const GPUMatrix<COMPUTE_T>* external_dL_dy = nullptr) {
        return forward(nullptr, loss_scale, input, target, data_pdf, use_inference_params,
                       prepare_input_gradients, external_dL_dy);
    }

    // Differentiate a matching forward context; parameter accumulation is explicit.
    void backward(metalStream_t stream, const ForwardContext& ctx, const GPUMatrixDynamic<T>& input,
                  GPUMatrixDynamic<T>* dL_dinput = nullptr, bool use_inference_params = false,
                  GradientMode param_gradients_mode = GradientMode::Overwrite) {
        m_model->backward(stream, *ctx.model_ctx, input, ctx.output, ctx.dL_doutput, dL_dinput,
                          use_inference_params, param_gradients_mode);
    }

    void backward(const ForwardContext& ctx, const GPUMatrixDynamic<T>& input,
                  GPUMatrixDynamic<T>* dL_dinput = nullptr, bool use_inference_params = false,
                  GradientMode param_gradients_mode = GradientMode::Overwrite) {
        backward(nullptr, ctx, input, dL_dinput, use_inference_params, param_gradients_mode);
    }

    // Advance the optimizer at the same host API boundary as the supplied reference.
    void optimizer_step(metalStream_t stream, float loss_scale) {
        m_optimizer->step(stream, loss_scale, m_params_full_precision, m_params, m_param_gradients);
    }

    void optimizer_step(float loss_scale) { optimizer_step(nullptr, loss_scale); }

    std::unique_ptr<MetalRtcKernel> generate_fused_fwd_bwd_kernel(const std::string& name,
                                                                  uint32_t n_threads,
                                                                  bool fwd_ctx_in_shmem,
                                                                  uint32_t non_fwd_ctx_shmem_bytes,
                                                                  bool has_external_dL_dy) {
        return detail::jit_compile_kernel(name,
                                          fused_training_source(name, n_threads, fwd_ctx_in_shmem,
                                                                non_fwd_ctx_shmem_bytes,
                                                                has_external_dL_dy),
                                          3);
    }

    // Capture forward/backward, then update outside the capture. Host optimizer
    // schedules advance per call, matching the uploaded implementation exactly.
    std::unique_ptr<ForwardContext>
    training_step(metalStream_t stream, const GPUMatrixDynamic<T>& input,
                  const GPUMatrix<float>& target, const GPUMatrix<float>* data_pdf = nullptr,
                  bool run_optimizer = true, GPUMatrixDynamic<T>* dL_dinput = nullptr,
                  bool use_inference_params = false,
                  GradientMode param_gradients_mode = GradientMode::Overwrite,
                  const GPUMatrix<COMPUTE_T>* external_dL_dy = nullptr) {
        const float loss_scale = default_loss_scale<PARAMS_T>();
        if (auto fused =
                try_jit_training(stream, input, target, data_pdf, run_optimizer, dL_dinput,
                                 use_inference_params, param_gradients_mode, external_dL_dy))
            return fused;
        std::unique_ptr<ForwardContext> ctx;
        {
            auto capture_guard = m_graph.capture_guard(stream);
            ctx = forward(stream, loss_scale, input, target, data_pdf, use_inference_params,
                          dL_dinput != nullptr, external_dL_dy);
            backward(stream, *ctx, input, dL_dinput, use_inference_params, param_gradients_mode);
            detail::GraphAccess::finish(m_graph);
        }
        if (run_optimizer)
            optimizer_step(stream, loss_scale);
        return ctx;
    }

    std::unique_ptr<ForwardContext>
    training_step(const GPUMatrixDynamic<T>& input, const GPUMatrix<float>& target,
                  const GPUMatrix<float>* data_pdf = nullptr, bool run_optimizer = true,
                  GPUMatrixDynamic<T>* dL_dinput = nullptr, bool use_inference_params = false,
                  GradientMode param_gradients_mode = GradientMode::Overwrite,
                  const GPUMatrix<COMPUTE_T>* external_dL_dy = nullptr) {
        return training_step(nullptr, input, target, data_pdf, run_optimizer, dL_dinput,
                             use_inference_params, param_gradients_mode, external_dL_dy);
    }

    // Sum the already normalized per-element losses and return one CPU scalar.
    float loss(metalStream_t stream, const ForwardContext& ctx) const {
        // A scalar read is already a CPU boundary. The null-stream overload must
        // also see work that produced this context on another explicit queue.
        if (!stream)
            metalrt::synchronize_device(ctx.L.device());
        return reduce_sum(ctx.L.data(), static_cast<uint32_t>(ctx.L.n_elements()), stream);
    }

    float loss(const ForwardContext& ctx) const { return loss(nullptr, ctx); }

    void update_hyperparams(const json& params) override {
        m_optimizer->update_hyperparams(params.value("optimizer", json::object()));
        m_loss->update_hyperparams(params.value("loss", json::object()));
    }

    json hyperparams() const override {
        return {{"otype", "Trainer"},
                {"optimizer", m_optimizer->hyperparams()},
                {"loss", m_loss->hyperparams()}};
    }

    float* params_full_precision() const { return m_params_full_precision; }

    PARAMS_T* params() const { return m_params; }

    PARAMS_T* params_inference() const { return m_params_inference; }

    PARAMS_T* param_gradients() const { return m_param_gradients; }

    // Import FP32 parameters, cast on the GPU, and keep inference/training copies coherent.
    void set_params_full_precision(const float* params, size_t n_params, bool device_ptr = false) {
        nn_check(n_params == m_model->n_params(),
                 "Can't set fp params because buffer has the wrong size");
        if (!n_params)
            return;
        nn_check(n_params <= UINT32_MAX, "parameter conversion exceeds 32-bit indexing");
        METAL_CHECK_THROW(
            metalMemcpy(m_params_full_precision, params, sizeof(float) * n_params,
                        device_ptr ? metalMemcpyDeviceToDevice : metalMemcpyHostToDevice));
        GPUMatrixDynamic<float> full{m_params_full_precision, 1, static_cast<uint32_t>(n_params)};
        GPUMatrixDynamic<PARAMS_T> inference{m_params_inference, 1,
                                             static_cast<uint32_t>(n_params)};
        detail::copy_matrix(nullptr, full, inference);
        METAL_CHECK_THROW(metalDeviceSynchronize());
        METAL_CHECK_THROW(metalMemcpy(m_params, m_params_inference, sizeof(PARAMS_T) * n_params,
                                      metalMemcpyDeviceToDevice));
    }

    // Import typed parameters and rebuild the FP32 master weights on the GPU.
    void set_params(const PARAMS_T* params, size_t n_params, bool device_ptr = false) {
        nn_check(n_params == m_model->n_params(),
                 "Can't set params because buffer has the wrong size");
        if (!n_params)
            return;
        nn_check(n_params <= UINT32_MAX, "parameter conversion exceeds 32-bit indexing");
        METAL_CHECK_THROW(
            metalMemcpy(m_params_inference, params, sizeof(PARAMS_T) * n_params,
                        device_ptr ? metalMemcpyDeviceToDevice : metalMemcpyHostToDevice));
        METAL_CHECK_THROW(metalMemcpy(m_params, m_params_inference, sizeof(PARAMS_T) * n_params,
                                      metalMemcpyDeviceToDevice));
        GPUMatrixDynamic<PARAMS_T> inference{m_params_inference, 1,
                                             static_cast<uint32_t>(n_params)};
        GPUMatrixDynamic<float> full{m_params_full_precision, 1, static_cast<uint32_t>(n_params)};
        detail::copy_matrix(nullptr, inference, full);
        METAL_CHECK_THROW(metalDeviceSynchronize());
    }

    std::shared_ptr<DifferentiableObject<T, PARAMS_T, COMPUTE_T>> model() { return m_model; }

    // Use the uploaded binary JSON checkpoint fields and original type spellings.
    json serialize(bool serialize_optimizer = false) {
        json data = {
            {"n_params", m_model->n_params()},
            {"params_type", type_to_string<PARAMS_T>()},
            {"params_binary", gpu_memory_to_json_binary(m_params_inference,
                                                        sizeof(PARAMS_T) * m_model->n_params())}};
        if (serialize_optimizer)
            data["optimizer"] = m_optimizer->serialize();
        return data;
    }

    void deserialize(const json& data) {
        const std::string type = data.value("params_type", type_to_string<PARAMS_T>());
        if (type == "float") {
            GPUMemory<float> params = data.at("params_binary");
            set_params_full_precision(params.data(), params.size(), true);
        } else if (type == "half") {
            GPUMemory<half> params_hp = data.at("params_binary");
            GPUMemory<PARAMS_T> params{params_hp.size()};
            GPUMatrixDynamic<half> src{params_hp.data(), 1,
                                       static_cast<uint32_t>(params_hp.size())};
            GPUMatrixDynamic<PARAMS_T> dst{params.data(), 1, static_cast<uint32_t>(params.size())};
            detail::copy_matrix(nullptr, src, dst);
            METAL_CHECK_THROW(metalDeviceSynchronize());
            set_params(params.data(), params.size(), true);
        } else
            throw std::runtime_error("Trainer: snapshot parameters must be of type float or half");
        if (data.contains("optimizer"))
            m_optimizer->deserialize(data.at("optimizer"));
        reset_param_pointers();
        METAL_CHECK_THROW(metalDeviceSynchronize());
        m_graph.reset();
    }

    // Override the model's gradient destination while preserving the Trainer's
    // own optimizer buffer, following the uploaded setter's precise semantics.
    void set_param_gradients_pointer(PARAMS_T* gradients) {
        reset_param_pointers();
        m_model->set_params(m_params, m_params_inference, gradients);
    }

    void reset_param_pointers() {
        size_t n_params = m_model->n_params();
        auto* base = m_params_buffer.data();
        m_params_full_precision = reinterpret_cast<float*>(base);
        m_params =
            n_params ? reinterpret_cast<PARAMS_T*>(base + sizeof(float) * n_params) : nullptr;
        m_param_gradients = n_params ? reinterpret_cast<PARAMS_T*>(base + sizeof(float) * n_params +
                                                                   sizeof(PARAMS_T) * n_params)
                                     : nullptr;
        m_params_inference = m_optimizer ? m_optimizer->custom_weights() : nullptr;
        if (!m_params_inference)
            m_params_inference = m_params;
        m_model->set_params(m_params, m_params_inference, m_param_gradients);
    }

    size_t n_params() const { return m_model->n_params(); }

  private:
    std::string fused_training_source(const std::string& name, uint32_t n_threads,
                                      bool in_threadgroup, uint32_t scratch_bytes, bool external,
                                      size_t gradient_parameters = 0,
                                      bool private_context = false) const {
        nn_check(n_threads > 0, "JIT training threadgroup must be nonempty");
        nn_check(scratch_bytes % 4 == 0, "JIT backward scratch must preserve context alignment");
        return detail::jit_training_kernel_source(
            name,
            private_context ? m_model->generate_training_device_function("eval_model")
                            : m_model->generate_device_function("eval_model"),
            private_context
                ? m_model->generate_training_backward_device_function("backward_eval_model",
                                                                      n_threads)
                : m_model->generate_backward_device_function("backward_eval_model", n_threads),
            external ? std::string{}
                     : m_loss->generate_device_function("eval_loss", m_model->output_width()),
            type_to_string<T>(), type_to_string<PARAMS_T>(), type_to_string<COMPUTE_T>(),
            m_model->input_width(), m_model->output_width(),
            private_context ? m_model->device_function_training_fwd_ctx_bytes()
                            : m_model->device_function_fwd_ctx_bytes(),
            in_threadgroup, scratch_bytes, external, gradient_parameters);
    }

    std::unique_ptr<ForwardContext>
    try_jit_training(metalStream_t stream, const GPUMatrixDynamic<T>& input,
                     const GPUMatrix<float>& target, const GPUMatrix<float>* pdf,
                     bool run_optimizer, GPUMatrixDynamic<T>* dx, bool inference_params,
                     GradientMode mode, const GPUMatrix<COMPUTE_T>* external) {
        if (!m_model->jit_fusion())
            return {};
        nn_check(static_cast<unsigned>(mode) < 3, "Invalid parameter gradient mode");
        nn_check(input.m() == m_model->input_width() && input.n() % BATCH_SIZE_GRANULARITY == 0,
                 "JIT training input shape mismatch");
        nn_check(!dx || (dx->m() == input.m() && dx->n() == input.n()),
                 "JIT training input gradient shape mismatch");
        nn_check(!m_model->n_params() ||
                     (inference_params ? m_model->inference_params() : m_model->params()),
                 "JIT training parameters have not been set");
        nn_check(mode == GradientMode::Ignore || !m_model->n_params() || m_model->gradients(),
                 "JIT training gradients have not been set");
        if (external)
            nn_check(external->m() == m_model->padded_output_width() && external->n() == input.n(),
                     "external derivative shape mismatch");
        else {
            nn_check(target.m() == m_model->output_width() && target.n() == input.n(),
                     "training target shape mismatch");
            nn_check(!pdf || (pdf->m() == target.m() && pdf->n() == target.n()),
                     "training PDF shape mismatch");
            nn_check(uint64_t(input.n()) * m_model->output_width() <= UINT32_MAX,
                     "JIT loss normalization exceeds 32-bit indexing");
        }
        uint32_t threads = 128, shared = 0, context_bytes = 0;
        bool context_in_threadgroup = false;
        try {
            if (!supports_jit_fusion())
                throw std::runtime_error(
                    "The selected runtime cannot compile fused Metal training");
            const auto device = metalrt::Device::current();
            const auto caps = device.capabilities();
            const bool wide_gradients = m_model->jit_allows_float_gradient_accumulation();
            context_bytes = m_model->device_function_training_fwd_ctx_bytes();
            const std::string key = json{{"model", m_model->jit_configuration_key()},
                                         {"loss", m_loss->hyperparams()},
                                         {"context_bytes", context_bytes},
                                         {"external", external != nullptr},
                                         {"gradient_mode", static_cast<unsigned>(mode)}}
                                        .dump();
            if (!m_jit_fused_kernel || m_jit_fused_device != device || m_jit_fused_key != key) {
                std::unique_ptr<MetalRtcKernel> selected;
                std::string source,
                    failure = "No native threadgroup configuration fits fused training";
                // Prefer full cooperative workgroups with device-backed saved
                // activations. Keeping the complete context in shared memory can
                // force a single-SIMD workgroup and reduce occupancy (notably for
                // HashGrid + width-64 MLPs). Retain shared-context candidates as
                // a fallback when the device-backed candidates cannot compile.
                std::vector<std::pair<uint32_t, bool>> candidates;
                for (bool in_group : {false, true})
                    for (uint32_t count = 128; count >= 32; count /= 2)
                        candidates.emplace_back(count, in_group);
                for (uint32_t count = 16; count; count /= 2)
                    for (bool in_group : {true, false})
                        candidates.emplace_back(count, in_group);
                for (const auto& configuration : candidates) {
                    threads = configuration.first;
                    const bool in_group = configuration.second;
                    if (caps.max_threads_per_threadgroup &&
                        threads > caps.max_threads_per_threadgroup)
                        continue;
                    uint32_t scratch;
                    try {
                        scratch = m_model->backward_device_function_shmem_bytes(threads, mode);
                    } catch (const std::runtime_error& e) {
                        failure = e.what();
                        continue;
                    }
                    {
                        const auto bytes = metalrt::checked_add(
                            scratch, in_group ? metalrt::checked_mul(context_bytes, threads) : 0);
                        if (bytes > UINT32_MAX - 15u)
                            continue;
                        auto rounded = std::max(16u, (static_cast<uint32_t>(bytes) + 15u) & ~15u);
                        if (caps.max_threadgroup_memory_bytes &&
                            rounded > caps.max_threadgroup_memory_bytes)
                            continue;
                        try {
                            auto candidate_source = fused_training_source(
                                "tcnn_jit_training", threads, in_group, scratch,
                                external != nullptr, wide_gradients ? m_model->n_params() : 0,
                                true);
                            auto candidate = detail::jit_compile_kernel("tcnn_jit_training",
                                                                        candidate_source, 3);
                            rounded = std::max(
                                rounded, candidate->required_threadgroup_memory_bytes(threads));
                            if (candidate->max_threads_per_threadgroup() &&
                                threads > candidate->max_threads_per_threadgroup())
                                continue;
                            if (caps.max_threadgroup_memory_bytes &&
                                uint64_t(rounded) + candidate->static_threadgroup_memory_bytes() >
                                    caps.max_threadgroup_memory_bytes)
                                continue;
                            source = std::move(candidate_source);
                            selected = std::move(candidate);
                            context_in_threadgroup = in_group;
                            shared = rounded;
                            break;
                        } catch (const std::runtime_error& e) {
                            failure = e.what();
                        }
                    }
                    if (selected)
                        break;
                }
                if (!selected)
                    throw std::runtime_error(failure);
                m_jit_fused_kernel = std::move(selected);
                m_jit_fused_device = device;
                m_jit_fused_source = std::move(source);
                m_jit_fused_wide_gradients = wide_gradients;
                m_jit_fused_gradient_plan =
                    wide_gradients
                        ? detail::jit_gradient_plan(m_jit_fused_source, m_model->n_params())
                        : detail::JitGradientPlan{};
                m_jit_fused_key = key;
                m_jit_fused_threads = threads;
                m_jit_fused_shared = shared;
                m_jit_fused_context_in_threadgroup = context_in_threadgroup;
            } else {
                threads = m_jit_fused_threads;
                shared = m_jit_fused_shared;
                context_in_threadgroup = m_jit_fused_context_in_threadgroup;
            }
        } catch (const std::runtime_error& e) {
            m_model->set_jit_fusion(false);
            log_warning("{}\nFailed to generate/compile fused Metal training. Disabling JIT for "
                        "this model.",
                        e.what());
            return {};
        }

        if (!m_model->jit_fusion())
            return {};
        auto batch = detail::resolve_stream(stream)->batch();
        auto ctx = std::make_unique<ForwardContext>();
        ctx->L = GPUMatrix<float>{m_model->output_width(), input.n(), stream};
        auto controls =
            detail::jit_collect_runtime_inputs(stream, m_model->jit_runtime_inputs(), input.n());
        auto fwd_context = allocate_workspace(
            stream, context_in_threadgroup
                        ? 0
                        : std::max(size_t{4}, metalrt::checked_mul(context_bytes, input.n())));
        const auto gradient_plan =
            mode == GradientMode::Ignore ? detail::JitGradientPlan{} : m_jit_fused_gradient_plan;
        const auto gradient_bytes =
            mode == GradientMode::Ignore
                ? 0
                : metalrt::checked_mul(m_model->n_params(), sizeof(PARAMS_T));
        // Only closed native model types promise canonical layout and widened
        // helper-only writes. Extensions, nested optimizers and redirected
        // gradient destinations keep the ordinary finalization boundary.
        const bool fused_adam =
            run_optimizer && mode != GradientMode::Ignore && m_jit_fused_wide_gradients &&
            m_model->jit_supports_fused_optimizer() && m_model->gradients() == m_param_gradients &&
            typeid(*m_optimizer) == typeid(AdamOptimizer<PARAMS_T>);
        auto gradients =
            fused_adam ? GPUMemoryArena::Allocation{}
                       : detail::jit_prepare_gradients(stream, m_model->gradients(), gradient_bytes,
                                                       mode == GradientMode::Accumulate);
        auto* gradient_data =
            fused_adam ? m_param_gradients : reinterpret_cast<PARAMS_T*>(gradients.data());
        auto wide_gradients = detail::jit_prepare_float_gradients(stream, gradient_plan);
        GPUMatrix<float> perturbation;
        if (m_perturbation_sigma > 0) {
            // Keep the existing PCG stream and logistic distribution. Only the
            // model/loss/backward chain is fused; the native optimizer epilogue
            // below preserves the same RNG and host schedule updates.
            perturbation = GPUMatrix<float>{m_model->padded_output_width(), input.n(), stream};
            generate_random_logistic<float>(stream, m_rng, perturbation.n_elements(),
                                            perturbation.data(), 0.0f, m_perturbation_sigma);
        }
        {
            m_jit_fused_kernel->launch(
                n_blocks_linear(input.n(), threads), threads, shared, stream, input.n(),
                default_loss_scale<PARAMS_T>(), input.view(),
                external ? MatrixView<float>{} : target.view(),
                pdf ? pdf->view() : MatrixView<float>{},
                external ? external->view() : MatrixView<COMPUTE_T>{},
                dx ? dx->view() : MatrixView<T>{}, ctx->L.view(),
                inference_params ? m_model->inference_params() : m_model->params(),
                fwd_context.data(), gradient_data, reinterpret_cast<const float*>(controls.data()),
                perturbation.view(), detail::jit_float_gradient_argument(22, wide_gradients),
                // Widened helpers need the exact gradient origin, including
                // an odd half view; the packed typed fallback uses aligned scratch.
                gradient_bytes ? metalrt::Argument::buffer(
                                     20, metalrt::resolve_pointer(gradient_data, gradient_bytes),
                                     metalrt::Access::ReadWrite)
                               : metalrt::Argument::null_buffer(20));
            if (m_jit_fused_wide_gradients && !fused_adam)
                detail::jit_finish_float_gradients(stream, m_model->gradients(), gradients,
                                                   wide_gradients, gradient_plan, sizeof(PARAMS_T));
            else if (!fused_adam)
                detail::jit_finish_gradients(stream, m_model->gradients(), gradients,
                                             gradient_bytes);
        }
        if (run_optimizer) {
            if (fused_adam) {
                detail::AdamJitGradientInput input_gradients{
                    mode == GradientMode::Accumulate ? m_param_gradients : nullptr,
                    reinterpret_cast<const float*>(wide_gradients.data()), gradient_plan.parameters,
                    gradient_plan.dense_parameters, gradient_plan.partitions};
                static_cast<AdamOptimizer<PARAMS_T>*>(m_optimizer.get())
                    ->step_with_float_gradients(stream, default_loss_scale<PARAMS_T>(),
                                                m_params_full_precision, m_params,
                                                m_param_gradients, input_gradients);
            } else
                optimizer_step(stream, default_loss_scale<PARAMS_T>());
        }
        return ctx;
    }

    std::shared_ptr<DifferentiableObject<T, PARAMS_T, COMPUTE_T>> m_model;
    std::shared_ptr<Optimizer<PARAMS_T>> m_optimizer;
    std::shared_ptr<Loss<COMPUTE_T>> m_loss;
    MetalGraph m_graph;
    GPUMemory<char> m_params_buffer;
    float* m_params_full_precision = nullptr;
    PARAMS_T* m_params_inference = nullptr;
    PARAMS_T* m_params = nullptr;
    PARAMS_T* m_param_gradients = nullptr;
    float m_perturbation_sigma;
    std::unique_ptr<Context> m_training_ctx;
    pcg32 m_rng;
    std::unique_ptr<MetalRtcKernel> m_jit_fused_kernel;
    std::string m_jit_fused_source, m_jit_fused_key;
    detail::JitGradientPlan m_jit_fused_gradient_plan;
    bool m_jit_fused_wide_gradients = false;
    uint32_t m_jit_fused_threads = 0, m_jit_fused_shared = 0;
    bool m_jit_fused_context_in_threadgroup = false;
    metalrt::Device m_jit_fused_device;
};
} // namespace tcnn
