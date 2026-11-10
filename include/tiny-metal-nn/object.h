#pragma once
#include <tiny-metal-nn/cpp_api.h>
#include <tiny-metal-nn/common_host.h>
#include <tiny-metal-nn/gpu_matrix.h>
#include <tiny-metal-nn/rtc_kernel.h>
#include <tiny-metal-nn/detail/jit.h>
#include <array>
#include <map>
#include <pcg32/pcg32.h>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

namespace tcnn {
using json = nlohmann::json;

inline void nn_check(bool condition, const char* message) {
    if (!condition)
        throw std::invalid_argument(message);
}

[[noreturn]] inline void unsupported_nn(const std::string& feature) {
    throw std::runtime_error("tiny-metal-nn: " + feature + " is unavailable on the Metal backend");
}

// Common configuration identity shared by encodings, networks, losses and optimizers.
class Object {
  public:
    virtual ~Object() = default;
    virtual json hyperparams() const = 0;

    std::string name() const { return hyperparams().value("otype", "<Unknown>"); }
};

class ObjectWithMutableHyperparams : public Object {
  public:
    virtual ~ObjectWithMutableHyperparams() = default;
    virtual void update_hyperparams(const json& params) = 0;
};

// Bind caller-owned training, inference and gradient arrays in a fixed flat layout.
template <typename PARAMS_T> class ParametricObject : public Object {
  public:
    virtual ~ParametricObject() = default;
    virtual void set_params_impl(PARAMS_T* params, PARAMS_T* inference_params,
                                 PARAMS_T* gradients) = 0;

    // Set all three pointers before invoking the concrete model's binding routine.
    void set_params(PARAMS_T* params, PARAMS_T* inference_params, PARAMS_T* gradients) {
        m_params = params;
        m_inference_params = inference_params;
        m_gradients = gradients;
        set_params_impl(params, inference_params, gradients);
    }

    virtual void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) = 0;
    virtual size_t n_params() const = 0;

    PARAMS_T* params() const { return m_params; }

    PARAMS_T* inference_params() const { return m_inference_params; }

    PARAMS_T* gradients() const { return m_gradients; }

    virtual std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const = 0;

  private:
    PARAMS_T* m_params = nullptr;
    PARAMS_T* m_inference_params = nullptr;
    PARAMS_T* m_gradients = nullptr;
};
enum class GradientMode { Ignore, Overwrite, Accumulate };

// Write the same scaled basis vector to every column of a batch.
template <typename T>
void one_hot_batched(metalStream_t stream, const uint32_t num_elements, const uint32_t width,
                     const uint32_t one_hot_dim, T* out, float scale);
// Scale an existing device vector without a host synchronization boundary.
template <typename T>
void mult(metalStream_t stream, const uint32_t num_elements, T* inout, float factor);
// Remove padded channels while converting an inference result to FP32.
template <typename T>
void trim_and_cast_from(metalStream_t stream, const MatrixLayout layout,
                        const uint32_t num_elements, const uint32_t input_width,
                        const uint32_t output_width, const T* in, float* out);

namespace detail {
template <typename IN, typename OUT>
void copy_matrix(metalStream_t stream, const GPUMatrixDynamic<IN>& src, GPUMatrixDynamic<OUT>& dst);
}

// Generate callable native MSL kernels. MatrixView arguments retain their host
// launch order and are expanded into a retained buffer and a stride descriptor.
std::unique_ptr<MetalRtcKernel> generate_kernel(const std::string& kernel_name,
                                                const std::string& device_function,
                                                const std::string& T, const std::string& PARAMS_T,
                                                const std::string& COMPUTE_T, uint32_t n_dims_in,
                                                uint32_t n_fwd_ctx_bytes = 0);
std::unique_ptr<MetalRtcKernel>
generate_backward_kernel(const std::string& kernel_name, const std::string& device_function,
                         const std::string& T, const std::string& PARAMS_T,
                         const std::string& COMPUTE_T, uint32_t n_dims_in, uint32_t n_dims_out,
                         uint32_t n_fwd_ctx_bytes);
std::unique_ptr<MetalRtcKernel> generate_backward_backward_input_kernel(
    const std::string& kernel_name, const std::string& device_function, const std::string& T,
    const std::string& PARAMS_T, const std::string& COMPUTE_T, uint32_t n_dims_in,
    uint32_t n_dims_out, uint32_t n_fwd_ctx_bytes);

// Validate shapes and parameter bindings around a concrete differentiable model.
// T is the input type, PARAMS_T the parameter type, COMPUTE_T the output type.
template <typename T, typename PARAMS_T, typename COMPUTE_T = T>
class DifferentiableObject : public ParametricObject<PARAMS_T> {
  public:
    virtual ~DifferentiableObject() = default;
    virtual void inference_mixed_precision_impl(metalStream_t stream,
                                                const GPUMatrixDynamic<T>& input,
                                                GPUMatrixDynamic<COMPUTE_T>& output,
                                                bool use_inference_params = true)
#ifdef TCNN_NO_FWD_BWD
    {
        (void)stream;
        (void)input;
        (void)output;
        (void)use_inference_params;
        throw std::runtime_error(
            "tiny-metal-nn was built without AOT forward/backward; enable jit_fusion on the model");
    }
#else
        = 0;
#endif

    // Evaluate into the padded, compute-precision output matrix.
    void inference_mixed_precision(metalStream_t stream, const GPUMatrixDynamic<T>& input,
                                   GPUMatrixDynamic<COMPUTE_T>& output,
                                   bool use_inference_params = true) {
        validate_forward(input, &output, use_inference_params);
        if (try_jit_inference(stream, input, output, use_inference_params, 0))
            return;
        inference_mixed_precision_impl(stream, input, output, use_inference_params);
    }

    void inference_mixed_precision(const GPUMatrixDynamic<T>& input,
                                   GPUMatrixDynamic<COMPUTE_T>& output,
                                   bool use_inference_params = true) {
        inference_mixed_precision(nullptr, input, output, use_inference_params);
    }

    // Return logical output channels as FP32, trimming padding on the GPU.
    void inference(metalStream_t stream, const GPUMatrixDynamic<T>& input,
                   GPUMatrixDynamic<float>& output, bool use_inference_params = true) {
        nn_check(output.m() == output_width() && output.n() == input.n(),
                 "inference output shape mismatch");
        validate_forward(input, static_cast<const GPUMatrixDynamic<COMPUTE_T>*>(nullptr),
                         use_inference_params);
        if (try_jit_inference(stream, input, output, use_inference_params, 1))
            return;
        if constexpr (std::is_same<COMPUTE_T, float>::value) {
            if (padded_output_width() == output_width()) {
                inference_mixed_precision(stream, input, output, use_inference_params);
                return;
            }
        }

        GPUMatrixDynamic<COMPUTE_T> inference_output_tmp{padded_output_width(), input.n(), stream,
                                                         output.layout()};
        inference_mixed_precision(stream, input, inference_output_tmp, use_inference_params);
        detail::copy_matrix(stream, inference_output_tmp, output);
    }

    void inference(const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<float>& output,
                   bool use_inference_params = true) {
        inference(nullptr, input, output, use_inference_params);
    }

    virtual std::unique_ptr<Context>
    forward_impl(metalStream_t stream, const GPUMatrixDynamic<T>& input,
                 GPUMatrixDynamic<COMPUTE_T>* output = nullptr, bool use_inference_params = false,
                 bool prepare_input_gradients = false)
#ifdef TCNN_NO_FWD_BWD
    {
        (void)stream;
        (void)input;
        (void)output;
        (void)use_inference_params;
        (void)prepare_input_gradients;
        throw std::runtime_error(
            "tiny-metal-nn was built without AOT forward/backward; enable jit_fusion on the model");
    }
#else
        = 0;
#endif

    // Save a model-specific context for a later backward pass with matching inputs.
    std::unique_ptr<Context> forward(metalStream_t stream, const GPUMatrixDynamic<T>& input,
                                     GPUMatrixDynamic<COMPUTE_T>* output = nullptr,
                                     bool use_inference_params = false,
                                     bool prepare_input_gradients = false) {
        validate_forward(input, output, use_inference_params);

        if (auto jit = try_jit_forward(stream, input, output, use_inference_params))
            return jit;

        return forward_impl(stream, input, output, use_inference_params, prepare_input_gradients);
    }

    std::unique_ptr<Context> forward(const GPUMatrixDynamic<T>& input,
                                     GPUMatrixDynamic<COMPUTE_T>* output = nullptr,
                                     bool use_inference_params = false,
                                     bool prepare_input_gradients = false) {
        return forward(nullptr, input, output, use_inference_params, prepare_input_gradients);
    }

    virtual void
    backward_impl(metalStream_t stream, const Context& ctx, const GPUMatrixDynamic<T>& input,
                  const GPUMatrixDynamic<COMPUTE_T>& output,
                  const GPUMatrixDynamic<COMPUTE_T>& dL_doutput,
                  GPUMatrixDynamic<T>* dL_dinput = nullptr, bool use_inference_params = false,
                  GradientMode param_gradients_mode = GradientMode::Overwrite)
#ifdef TCNN_NO_FWD_BWD
    {
        (void)stream;
        (void)ctx;
        (void)input;
        (void)output;
        (void)dL_doutput;
        (void)dL_dinput;
        (void)use_inference_params;
        (void)param_gradients_mode;
        throw std::runtime_error(
            "tiny-metal-nn was built without AOT forward/backward; enable jit_fusion on the model");
    }
#else
        = 0;
#endif

    // Apply the chain rule and choose whether parameter gradients overwrite or accumulate.
    void backward(metalStream_t stream, const Context& ctx, const GPUMatrixDynamic<T>& input,
                  const GPUMatrixDynamic<COMPUTE_T>& output,
                  const GPUMatrixDynamic<COMPUTE_T>& dL_doutput,
                  GPUMatrixDynamic<T>* dL_dinput = nullptr, bool use_inference_params = false,
                  GradientMode param_gradients_mode = GradientMode::Overwrite) {
        validate_forward(input, &output, use_inference_params);

        nn_check(dL_doutput.m() == padded_output_width() && dL_doutput.n() == input.n(),
                 "output gradient shape mismatch");
        nn_check(!dL_dinput || (dL_dinput->m() == input_width() && dL_dinput->n() == input.n()),
                 "input gradient shape mismatch");
        nn_check(param_gradients_mode == GradientMode::Ignore || !this->n_params() ||
                     this->gradients(),
                 "parameter gradients have not been set");

        if (try_jit_backward(stream, ctx, input, dL_doutput, dL_dinput, use_inference_params,
                             param_gradients_mode))
            return;

        backward_impl(stream, ctx, input, output, dL_doutput, dL_dinput, use_inference_params,
                      param_gradients_mode);
    }

    void backward(const Context& ctx, const GPUMatrixDynamic<T>& input,
                  const GPUMatrixDynamic<COMPUTE_T>& output,
                  const GPUMatrixDynamic<COMPUTE_T>& dL_doutput,
                  GPUMatrixDynamic<T>* dL_dinput = nullptr, bool use_inference_params = false,
                  GradientMode param_gradients_mode = GradientMode::Overwrite) {
        backward(nullptr, ctx, input, output, dL_doutput, dL_dinput, use_inference_params,
                 param_gradients_mode);
    }

    virtual void backward_backward_input_impl(
        metalStream_t stream, const Context& ctx, const GPUMatrixDynamic<T>& input,
        const GPUMatrixDynamic<T>& dL_ddLdinput, const GPUMatrixDynamic<COMPUTE_T>& dL_doutput,
        GPUMatrixDynamic<COMPUTE_T>* dL_ddLdoutput = nullptr,
        GPUMatrixDynamic<T>* dL_dinput = nullptr, bool use_inference_params = false,
        GradientMode param_gradients_mode = GradientMode::Overwrite) {
        (void)stream;
        (void)ctx;
        (void)input;
        (void)dL_ddLdinput;
        (void)dL_doutput;
        (void)dL_ddLdoutput;
        (void)dL_dinput;
        (void)use_inference_params;
        (void)param_gradients_mode;
        unsupported_nn(this->name() + " second derivatives");
    }

    void backward_backward_input(metalStream_t stream, const Context& ctx,
                                 const GPUMatrixDynamic<T>& input,
                                 const GPUMatrixDynamic<T>& dL_ddLdinput,
                                 const GPUMatrixDynamic<COMPUTE_T>& dL_doutput,
                                 GPUMatrixDynamic<COMPUTE_T>* dL_ddLdoutput = nullptr,
                                 GPUMatrixDynamic<T>* dL_dinput = nullptr,
                                 bool use_inference_params = false,
                                 GradientMode param_gradients_mode = GradientMode::Overwrite) {
        validate_forward(input, static_cast<const GPUMatrixDynamic<COMPUTE_T>*>(nullptr),
                         use_inference_params);

        nn_check(input.m() == dL_ddLdinput.m() && input.n() == dL_ddLdinput.n(),
                 "double backward input shape mismatch");
        nn_check(dL_doutput.m() == padded_output_width() && dL_doutput.n() == input.n(),
                 "double backward output shape mismatch");
        nn_check(!dL_ddLdoutput || (dL_ddLdoutput->m() == padded_output_width() &&
                                    dL_ddLdoutput->n() == input.n()),
                 "double backward output destination mismatch");
        nn_check(!dL_dinput || (dL_dinput->m() == input_width() && dL_dinput->n() == input.n()),
                 "double backward input destination mismatch");
        nn_check(param_gradients_mode == GradientMode::Ignore || !this->n_params() ||
                     this->gradients(),
                 "parameter gradients have not been set");

        if (try_jit_second(stream, ctx, input, dL_ddLdinput, dL_doutput, dL_ddLdoutput, dL_dinput,
                           use_inference_params, param_gradients_mode))
            return;

        backward_backward_input_impl(stream, ctx, input, dL_ddLdinput, dL_doutput, dL_ddLdoutput,
                                     dL_dinput, use_inference_params, param_gradients_mode);
    }

    void backward_backward_input(const Context& ctx, const GPUMatrixDynamic<T>& input,
                                 const GPUMatrixDynamic<T>& dL_ddLdinput,
                                 const GPUMatrixDynamic<COMPUTE_T>& dL_doutput,
                                 GPUMatrixDynamic<COMPUTE_T>* dL_ddLdoutput = nullptr,
                                 GPUMatrixDynamic<T>* dL_dinput = nullptr,
                                 bool use_inference_params = false,
                                 GradientMode param_gradients_mode = GradientMode::Overwrite) {
        backward_backward_input(nullptr, ctx, input, dL_ddLdinput, dL_doutput, dL_ddLdoutput,
                                dL_dinput, use_inference_params, param_gradients_mode);
    }

    // Compute one output channel's Jacobian row, using inference parameters.
    void input_gradient(metalStream_t stream, uint32_t dim, const GPUMatrix<T>& input,
                        GPUMatrix<T>& d_dinput,
                        float backprop_scale = default_loss_scale<PARAMS_T>()) {
        nn_check(dim < padded_output_width(), "invalid output dimension for input_gradient");
        nn_check(std::isfinite(backprop_scale) && backprop_scale != 0.0f,
                 "input_gradient scale must be finite and nonzero");
        nn_check(d_dinput.is_contiguous(),
                 "input_gradient helper requires a contiguous destination");
        nn_check(size_t(padded_output_width()) * input.n() <= UINT32_MAX &&
                     d_dinput.n_elements() <= UINT32_MAX,
                 "input_gradient size exceeds 32-bit indexing");
        GPUMatrix<COMPUTE_T> output{padded_output_width(), input.n(), stream};
        GPUMatrix<COMPUTE_T> d_doutput{padded_output_width(), input.n(), stream};
        one_hot_batched(stream, static_cast<uint32_t>(output.n_elements()), padded_output_width(),
                        dim, d_doutput.data(), backprop_scale);
        auto ctx = forward(stream, input, &output, true, true);
        backward(stream, *ctx, input, output, d_doutput, &d_dinput, true, GradientMode::Ignore);
        mult(stream, static_cast<uint32_t>(d_dinput.n_elements()), d_dinput.data(),
             1.0f / backprop_scale);
    }

    virtual uint32_t input_width() const = 0;
    virtual uint32_t padded_output_width() const = 0;
    virtual uint32_t output_width() const = 0;
    virtual uint32_t required_input_alignment() const = 0;

    virtual std::string generate_device_function(const std::string& name) const {
        (void)name;
        unsupported_nn("automatic MSL device-function fusion");
    }

    virtual std::string generate_backward_device_function(const std::string& name,
                                                          uint32_t n_threads) const {
        (void)name;
        (void)n_threads;
        unsupported_nn("automatic MSL backward fusion");
    }

    virtual std::string generate_backward_backward_input_device_function(const std::string& name,
                                                                         uint32_t n_threads) const {
        (void)name;
        (void)n_threads;
        unsupported_nn("automatic MSL second-backward fusion");
    }

    virtual uint32_t device_function_fwd_ctx_bytes() const {
        unsupported_nn("fused forward context layout");
    }

    virtual bool device_function_fwd_ctx_aligned_per_element() const { return true; }

    virtual uint32_t backward_device_function_shmem_bytes(uint32_t n_threads,
                                                          GradientMode param_gradients_mode) const {
        (void)n_threads;
        (void)param_gradients_mode;
        return 0;
    }

    std::string generate_vec_in() const {
        return "tcnn_jit_array<" + type_to_string<T>() + ", " + std::to_string(input_width()) + ">";
    }

    std::string generate_vec(uint32_t width) const {
        return "tcnn_jit_array<" + type_to_string<COMPUTE_T>() + ", " + std::to_string(width) + ">";
    }

    std::string generate_vec_padded_out() const { return generate_vec(padded_output_width()); }

    std::string generate_vec_out() const { return generate_vec(output_width()); }

    std::string generate_device_function_from_body(const std::string& name,
                                                   const std::string& body) const {
        return detail::jit_forward_function(
            name, generate_vec_in(), generate_vec_out(), type_to_string<PARAMS_T>(), body,
            device_function_fwd_ctx_bytes(), device_function_fwd_ctx_aligned_per_element());
    }

    std::string generate_backward_device_function_from_body(const std::string& name,
                                                            const std::string& body) const {
        return detail::jit_backward_function(
            name, generate_vec_in(), generate_vec_out(), type_to_string<PARAMS_T>(), body,
            device_function_fwd_ctx_bytes(), device_function_fwd_ctx_aligned_per_element());
    }

    std::string
    generate_backward_backward_input_device_function_from_body(const std::string& name,
                                                               const std::string body) const {
        return detail::jit_second_function(
            name, generate_vec_in(), generate_vec_out(), type_to_string<PARAMS_T>(), body,
            device_function_fwd_ctx_bytes(), device_function_fwd_ctx_aligned_per_element());
    }

    // Optional resources are an additive Metal customization point: a host
    // pointer cannot be embedded in generated MSL as an executable GPU address.
    virtual std::vector<const float*> jit_runtime_inputs() const { return {}; }
    // Custom generators whose source depends on state outside hyperparams()
    // append that state here. Parameter values and runtime-input pointers are
    // bound per launch and must not enter a source compilation key.
    virtual std::string jit_configuration_key() const {
        json key{{"hyperparams", this->hyperparams()},
                 {"input", input_width()},
                 {"output", output_width()},
                 {"padded_output", padded_output_width()},
                 {"params", this->n_params()},
                 {"per_element", device_function_fwd_ctx_aligned_per_element()},
                 {"float_gradient_accumulation", jit_allows_float_gradient_accumulation()},
                 {"compact_training_context", jit_supports_compact_training_context()},
                 {"runtime_slots", jit_runtime_inputs().size()},
                 {"revision", m_jit_revision}};
        try {
            key["context_bytes"] = device_function_fwd_ctx_bytes();
        } catch (const std::runtime_error&) {
            key["context_bytes"] = nullptr;
        }
        return key.dump();
    }
    void invalidate_jit_cache() {
        ++m_jit_revision;
        m_jit_sources.reset();
    }

    bool jit_fusion() const { return m_jit_fusion; }
    void set_jit_fusion(bool val) { m_jit_fusion = val; }

    // Widened/striped reduction is opt-in: a generated body that reads or
    // overwrites its typed gradient destination must retain immediate stores.
    // Builtins opt in for their exact dynamic types; existing subclasses keep
    // the original behavior unless they explicitly promise additive writes.
    // Append the customization point after the existing virtual declarations.
    virtual bool jit_allows_float_gradient_accumulation() const { return false; }

    // Opt-in for the native training epilogue: canonical parameter/gradient
    // layout and all generated gradient contributions routed through the
    // environment's widened helpers. Unrecognized extensions keep typed scratch.
    virtual bool jit_supports_fused_optimizer() const { return false; }

    // Automatic training owns a private context which never escapes as a model
    // forward context. Extensions keep their public representation by default.
    virtual bool jit_supports_compact_training_context() const { return false; }
    virtual uint32_t device_function_training_fwd_ctx_bytes() const {
        return device_function_fwd_ctx_bytes();
    }
    virtual std::string generate_training_device_function(const std::string& name) const {
        return generate_device_function(name);
    }
    virtual std::string generate_training_backward_device_function(const std::string& name,
                                                                   uint32_t threads) const {
        return generate_backward_device_function(name, threads);
    }

  protected:
    bool is_jit_context(const Context& ctx) const {
        auto* saved = dynamic_cast<const JitForwardContext*>(&ctx);
        if (saved)
            nn_check(saved->owner == this, "JIT context belongs to another model");
        return saved != nullptr;
    }

    const uint8_t* jit_forward_context_data(const Context& ctx) const {
        auto* saved = dynamic_cast<const JitForwardContext*>(&ctx);
        if (!saved)
            return nullptr;
        nn_check(saved->owner == this, "JIT context belongs to another model");
        return saved->data.data();
    }

    uint32_t jit_forward_context_batch(const Context& ctx) const {
        auto* saved = dynamic_cast<const JitForwardContext*>(&ctx);
        if (!saved)
            return 0;
        nn_check(saved->owner == this, "JIT context belongs to another model");
        return saved->batch;
    }

    std::pair<const COMPUTE_T*, MatrixLayout>
    jit_forward_activation(const Context& ctx, uint32_t byte_offset, uint32_t width) const {
        const auto& saved = checked_jit_context(ctx);
        nn_check(!saved.cooperative_width,
                 "Cooperative JIT contexts require a model-specific activation diagnostic");
        nn_check(byte_offset % sizeof(COMPUTE_T) == 0 &&
                     saved.context_bytes % sizeof(COMPUTE_T) == 0,
                 "JIT activation is not aligned to its scalar type");
        nn_check(uint64_t(byte_offset) + uint64_t(width) * sizeof(COMPUTE_T) <= saved.context_bytes,
                 "JIT activation exceeds its saved context");
        const auto key = std::make_pair(byte_offset, width);
        auto found = saved.activations.find(key);
        if (found == saved.activations.end()) {
            GPUMatrixDynamic<COMPUTE_T> source{
                reinterpret_cast<COMPUTE_T*>(const_cast<uint8_t*>(saved.data.data()) + byte_offset),
                width, saved.batch, CM,
                static_cast<uint32_t>(saved.context_bytes / sizeof(COMPUTE_T))};
            GPUMatrixDynamic<COMPUTE_T> output{width, saved.batch, saved.stream, CM};
            detail::copy_matrix(saved.stream, source, output);
            found = saved.activations.emplace(key, std::move(output)).first;
        }
        return {found->second.data(), CM};
    }

    void validate_forward(const GPUMatrixDynamic<T>& input,
                          const GPUMatrixDynamic<COMPUTE_T>* output, bool inference) const {
        nn_check(input.m() == input_width(), "model input width mismatch");
        nn_check(input.n() % BATCH_SIZE_GRANULARITY == 0,
                 "batch size must be a multiple of BATCH_SIZE_GRANULARITY");
        nn_check(!output || (output->m() == padded_output_width() && output->n() == input.n()),
                 "model output shape mismatch");
        nn_check(!this->n_params() || (inference ? this->inference_params() : this->params()),
                 "model parameters have not been set");
    }

  private:
    struct JitGeneratedSources {
        std::string key, forward, forward_kernel;
        std::array<std::string, 2> inference;
        uint32_t context_bytes = 0;
        size_t parameter_count = 0;
        bool wide_gradients = false;
        bool derivatives = false;
        std::array<std::shared_ptr<const std::string>, 8> backward, second;
        std::array<detail::JitGradientPlan, 8> backward_gradients, second_gradients;
        std::array<std::string, 8> backward_error, second_error;
        std::array<std::array<uint32_t, 3>, 8> scratch{};
        std::array<std::array<std::string, 3>, 8> scratch_error;
    };
    struct JitForwardContext : Context {
        const DifferentiableObject* owner = nullptr;
        metalrt::Device device;
        std::unique_ptr<metal::Stream> retained_stream;
        metalStream_t stream = nullptr;
        uint32_t batch = 0, context_bytes = 0, input_width = 0, output_width = 0,
                 cooperative_width = 0;
        GPUMemoryArena::Allocation data, runtime_inputs;
        std::shared_ptr<const JitGeneratedSources> sources;
        mutable std::map<std::pair<uint32_t, uint32_t>, GPUMatrixDynamic<COMPUTE_T>> activations;
    };

    std::shared_ptr<JitGeneratedSources> jit_sources(bool derivatives) {
        const auto key = jit_configuration_key();
        if (!m_jit_sources || m_jit_sources->key != key) {
            auto source = std::make_shared<JitGeneratedSources>();
            source->key = key;
            source->forward = generate_device_function("eval_model");
            source->parameter_count = this->n_params();
            source->wide_gradients = jit_allows_float_gradient_accumulation();
            m_jit_sources = std::move(source);
        }
        auto source = m_jit_sources;
        if (derivatives && !source->derivatives) {
            source->context_bytes = device_function_fwd_ctx_bytes();
            // Capture all launch specializations once per configuration. This
            // lets a resource-heavy pipeline select a smaller native group
            // without re-reading mutable model state after forward.
            for (size_t i = 0; i < 8; ++i) {
                const uint32_t threads = 128u >> i;
                for (size_t mode = 0; mode < 3; ++mode) {
                    try {
                        source->scratch[i][mode] = backward_device_function_shmem_bytes(
                            threads, static_cast<GradientMode>(mode));
                    } catch (const std::runtime_error& e) {
                        source->scratch_error[i][mode] = e.what();
                    }
                }
                try {
                    const auto function =
                        generate_backward_device_function("backward_eval_model", threads);
                    if (source->wide_gradients)
                        source->backward_gradients[i] =
                            detail::jit_gradient_plan(function, source->parameter_count);
                    auto code = detail::jit_backward_kernel_source(
                        "tcnn_jit_backward", function, type_to_string<T>(),
                        type_to_string<PARAMS_T>(), type_to_string<COMPUTE_T>(), input_width(),
                        output_width(), source->context_bytes, false,
                        source->wide_gradients ? source->parameter_count : 0);
                    source->backward[i] =
                        i && source->backward[i - 1] && *source->backward[i - 1] == code
                            ? source->backward[i - 1]
                            : std::make_shared<const std::string>(std::move(code));
                } catch (const std::runtime_error& e) {
                    source->backward_error[i] = e.what();
                }
                try {
                    const auto function = generate_backward_backward_input_device_function(
                        "backward_backward_input_eval_model", threads);
                    if (source->wide_gradients)
                        source->second_gradients[i] =
                            detail::jit_gradient_plan(function, source->parameter_count);
                    auto code = detail::jit_backward_kernel_source(
                        "tcnn_jit_second", function, type_to_string<T>(),
                        type_to_string<PARAMS_T>(), type_to_string<COMPUTE_T>(), input_width(),
                        output_width(), source->context_bytes, true,
                        source->wide_gradients ? source->parameter_count : 0);
                    source->second[i] = i && source->second[i - 1] && *source->second[i - 1] == code
                                            ? source->second[i - 1]
                                            : std::make_shared<const std::string>(std::move(code));
                } catch (const std::runtime_error& e) {
                    source->second_error[i] = e.what();
                }
            }
            source->derivatives = true;
        }
        return source;
    }

    static uint32_t jit_launch_threads(const MetalRtcKernel& kernel) {
        uint32_t result = 128;
        const auto limit = kernel.max_threads_per_threadgroup();
        while (limit && result > limit)
            result /= 2;
        return std::max(1u, result);
    }

    MetalRtcKernel* jit_derivative_kernel(const JitForwardContext& saved, bool second,
                                          GradientMode mode, uint32_t& threads, uint32_t& scratch,
                                          detail::JitGradientPlan& gradients) {
        nn_check(static_cast<unsigned>(mode) < 3, "Invalid parameter gradient mode");
        std::string failure = "No native launch configuration fits this generated derivative";
        const auto caps = saved.device.capabilities();
        for (size_t i = 0; i < 8; ++i) {
            threads = 128u >> i;
            if (!saved.sources->scratch_error[i][static_cast<size_t>(mode)].empty()) {
                failure = saved.sources->scratch_error[i][static_cast<size_t>(mode)];
                continue;
            }
            const auto& code = second ? saved.sources->second[i] : saved.sources->backward[i];
            if (!code) {
                failure =
                    second ? saved.sources->second_error[i] : saved.sources->backward_error[i];
                continue;
            }
            const auto requested = saved.sources->scratch[i][static_cast<size_t>(mode)];
            nn_check(requested <= UINT32_MAX - 15u, "JIT scratch exceeds uint32_t");
            scratch = std::max(16u, (requested + 15u) & ~15u);
            try {
                auto* kernel = m_jit_kernels[second ? 4 : 3].get(
                    second ? "tcnn_jit_second" : "tcnn_jit_backward", *code, second ? 2 : 1);
                scratch = std::max(scratch, kernel->required_threadgroup_memory_bytes(threads));
                if (kernel->max_threads_per_threadgroup() &&
                    threads > kernel->max_threads_per_threadgroup())
                    continue;
                if (caps.max_threadgroup_memory_bytes &&
                    uint64_t(scratch) + kernel->static_threadgroup_memory_bytes() >
                        caps.max_threadgroup_memory_bytes)
                    continue;
                if (saved.cooperative_width &&
                    std::min(threads, kernel->thread_execution_width()) !=
                        saved.cooperative_width) {
                    failure = "Saved cooperative context requires the same native SIMD-group width "
                              "for forward and backward";
                    continue;
                }
                gradients = mode == GradientMode::Ignore
                                ? detail::JitGradientPlan{}
                                : (second ? saved.sources->second_gradients[i]
                                          : saved.sources->backward_gradients[i]);
                return kernel;
            } catch (const std::runtime_error& e) {
                failure = e.what();
            }
        }
        throw std::runtime_error("Failed to compile/launch saved Metal JIT derivative: " + failure);
    }

    const JitForwardContext& checked_jit_context(const Context& ctx) const {
        auto* saved = dynamic_cast<const JitForwardContext*>(&ctx);
        nn_check(saved && saved->owner == this,
                 "JIT context belongs to another model or execution path");
        nn_check(saved->device == metalrt::Device::current(),
                 "JIT context belongs to another Metal device");
        return *saved;
    }

    void disable_jit(const std::string& reason) {
        m_jit_fusion = false;
        log_warning(
            "{}\nFailed to generate/compile Metal model fusion. Disabling JIT for this model.",
            reason);
    }

    template <class OUT>
    bool try_jit_inference(metalStream_t stream, const GPUMatrixDynamic<T>& input,
                           GPUMatrixDynamic<OUT>& output, bool inference_params, size_t slot) {
        if (!m_jit_fusion)
            return false;
        MetalRtcKernel* kernel;
        try {
            if (!supports_jit_fusion())
                throw std::runtime_error("The selected runtime cannot compile Metal model fusion");
            const std::string name = slot == 0 ? "tcnn_jit_inference_mp" : "tcnn_jit_inference";
            auto source = jit_sources(false);
            if (source->inference[slot].empty())
                source->inference[slot] = detail::jit_forward_kernel_source(
                    name, source->forward, type_to_string<T>(), type_to_string<PARAMS_T>(),
                    type_to_string<OUT>(), input_width(), 0);
            kernel = m_jit_kernels[slot].get(name, source->inference[slot], 0);
        } catch (const std::runtime_error& e) {
            disable_jit(e.what());
            return false;
        }
        if (!m_jit_fusion)
            return false;
        auto controls = detail::jit_collect_runtime_inputs(stream, jit_runtime_inputs(), input.n());
        const auto threads = jit_launch_threads(*kernel);
        kernel->launch(n_blocks_linear(input.n(), threads), threads, 0, stream, input.n(),
                       input.view(), output.view(),
                       inference_params ? this->inference_params() : this->params(),
                       static_cast<uint8_t*>(nullptr),
                       reinterpret_cast<const float*>(controls.data()), output.m());
        return true;
    }

    std::unique_ptr<Context> try_jit_forward(metalStream_t stream, const GPUMatrixDynamic<T>& input,
                                             GPUMatrixDynamic<COMPUTE_T>* output,
                                             bool inference_params) {
        if (!m_jit_fusion)
            return {};
        auto saved = std::make_unique<JitForwardContext>();
        MetalRtcKernel* kernel;
        try {
            if (!supports_jit_fusion())
                throw std::runtime_error("The selected runtime cannot compile Metal model fusion");
            auto sources = jit_sources(true);
            saved->sources = sources;
            saved->context_bytes = saved->sources->context_bytes;
            saved->input_width = input_width();
            saved->output_width = output_width();
            if (sources->forward_kernel.empty())
                sources->forward_kernel = detail::jit_forward_kernel_source(
                    "tcnn_jit_forward", saved->sources->forward, type_to_string<T>(),
                    type_to_string<PARAMS_T>(), type_to_string<COMPUTE_T>(), input_width(),
                    saved->context_bytes);
            kernel = m_jit_kernels[2].get("tcnn_jit_forward", sources->forward_kernel, 0);
        } catch (const std::runtime_error& e) {
            disable_jit(e.what());
            return {};
        }
        if (!m_jit_fusion)
            return {};
        saved->owner = this;
        saved->device = metalrt::Device::current();
        saved->batch = input.n();
        saved->retained_stream =
            std::make_unique<metal::Stream>(detail::resolve_stream(stream)->retain());
        saved->stream = saved->retained_stream->get();
        saved->data = allocate_workspace(
            stream, std::max(size_t{4}, metalrt::checked_mul(saved->context_bytes, input.n())));
        saved->runtime_inputs =
            detail::jit_collect_runtime_inputs(stream, jit_runtime_inputs(), input.n());
        const auto threads = jit_launch_threads(*kernel);
        if (!device_function_fwd_ctx_aligned_per_element())
            saved->cooperative_width = std::min(threads, kernel->thread_execution_width());
        kernel->launch(n_blocks_linear(input.n(), threads), threads, 0, stream, input.n(),
                       input.view(), output ? output->view() : MatrixView<COMPUTE_T>{},
                       inference_params ? this->inference_params() : this->params(),
                       saved->data.data(),
                       reinterpret_cast<const float*>(saved->runtime_inputs.data()),
                       output ? output->m() : 0u);
        return saved;
    }

    bool try_jit_backward(metalStream_t stream, const Context& ctx,
                          const GPUMatrixDynamic<T>& input, const GPUMatrixDynamic<COMPUTE_T>& dy,
                          GPUMatrixDynamic<T>* dx, bool inference_params, GradientMode mode) {
        // A saved context selects its execution path even if the user toggled
        // JIT after forward. An AOT context remains valid when JIT is enabled.
        if (!is_jit_context(ctx))
            return false;
        const auto& saved = checked_jit_context(ctx);
        nn_check(saved.batch == input.n() && saved.input_width == input_width() &&
                     saved.output_width == output_width(),
                 "JIT context shape changed after forward");
        uint32_t threads, scratch;
        detail::JitGradientPlan plan;
        auto* kernel = jit_derivative_kernel(saved, false, mode, threads, scratch, plan);
        nn_check(saved.sources->parameter_count == this->n_params(),
                 "JIT parameter count changed after forward");
        const auto bytes =
            mode == GradientMode::Ignore
                ? 0
                : metalrt::checked_mul(saved.sources->parameter_count, sizeof(PARAMS_T));
        auto gradients = detail::jit_prepare_gradients(stream, this->gradients(), bytes,
                                                       mode == GradientMode::Accumulate);
        auto wide_gradients = detail::jit_prepare_float_gradients(stream, plan);
        kernel->launch(n_blocks_linear(input.n(), threads), threads, scratch, stream, input.n(),
                       dy.view(), dx ? dx->view() : MatrixView<T>{},
                       inference_params ? this->inference_params() : this->params(),
                       saved.data.data(), reinterpret_cast<PARAMS_T*>(gradients.data()),
                       reinterpret_cast<const float*>(saved.runtime_inputs.data()),
                       detail::jit_float_gradient_argument(11, wide_gradients));
        if (saved.sources->wide_gradients)
            detail::jit_finish_float_gradients(stream, this->gradients(), gradients, wide_gradients,
                                               plan, sizeof(PARAMS_T));
        else
            detail::jit_finish_gradients(stream, this->gradients(), gradients, bytes);
        return true;
    }

    bool try_jit_second(metalStream_t stream, const Context& ctx, const GPUMatrixDynamic<T>& input,
                        const GPUMatrixDynamic<T>& ddx, const GPUMatrixDynamic<COMPUTE_T>& dy,
                        GPUMatrixDynamic<COMPUTE_T>* ddy, GPUMatrixDynamic<T>* dx,
                        bool inference_params, GradientMode mode) {
        if (!is_jit_context(ctx))
            return false;
        const auto& saved = checked_jit_context(ctx);
        nn_check(saved.batch == input.n() && saved.input_width == input_width() &&
                     saved.output_width == output_width(),
                 "JIT context shape changed after forward");
        uint32_t threads, scratch;
        detail::JitGradientPlan plan;
        auto* kernel = jit_derivative_kernel(saved, true, mode, threads, scratch, plan);
        nn_check(saved.sources->parameter_count == this->n_params(),
                 "JIT parameter count changed after forward");
        const auto bytes =
            mode == GradientMode::Ignore
                ? 0
                : metalrt::checked_mul(saved.sources->parameter_count, sizeof(PARAMS_T));
        auto gradients = detail::jit_prepare_gradients(stream, this->gradients(), bytes,
                                                       mode == GradientMode::Accumulate);
        auto wide_gradients = detail::jit_prepare_float_gradients(stream, plan);
        kernel->launch(n_blocks_linear(input.n(), threads), threads, scratch, stream, input.n(),
                       ddx.view(), dy.view(), dx ? dx->view() : MatrixView<T>{},
                       ddy ? ddy->view() : MatrixView<COMPUTE_T>{},
                       inference_params ? this->inference_params() : this->params(),
                       saved.data.data(), reinterpret_cast<PARAMS_T*>(gradients.data()),
                       reinterpret_cast<const float*>(saved.runtime_inputs.data()),
                       ddy ? ddy->m() : 0u,
                       detail::jit_float_gradient_argument(16, wide_gradients));
        if (saved.sources->wide_gradients)
            detail::jit_finish_float_gradients(stream, this->gradients(), gradients, wide_gradients,
                                               plan, sizeof(PARAMS_T));
        else
            detail::jit_finish_gradients(stream, this->gradients(), gradients, bytes);
        return true;
    }

    std::shared_ptr<JitGeneratedSources> m_jit_sources;
    uint64_t m_jit_revision = 0;
    std::array<detail::JitKernelCache, 5> m_jit_kernels;
    bool m_jit_fusion = false;
};
} // namespace tcnn
