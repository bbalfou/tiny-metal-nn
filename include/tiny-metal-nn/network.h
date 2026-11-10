#pragma once
#include <tiny-metal-nn/host_math.h>
#include <tiny-metal-nn/object.h>

namespace tcnn {
// Extract one activation as positive/negative visualization channels.
template <typename T>
void extract_dimension_pos_neg(metalStream_t stream, const uint32_t num_elements,
                               const uint32_t dim, const uint32_t fan_in, const uint32_t fan_out,
                               const T* encoded, MatrixLayout layout, float* output);

// Reference network interface: the input type and parameter/activation type may differ.
template <typename T, typename PARAMS_T = T>
class Network : public DifferentiableObject<T, PARAMS_T, PARAMS_T> {
  public:
    virtual ~Network() = default;

    // Run a forward pass and expose one hidden channel for diagnostic rendering.
    void visualize_activation(metalStream_t stream, uint32_t layer, uint32_t dimension,
                              const GPUMatrix<T>& input, GPUMatrix<float>& output) {
        nn_check(num_forward_activations() != 0, "network has no hidden activations to visualize");
        layer = std::min(layer, num_forward_activations() - 1);
        dimension = std::min(dimension, width(layer) - 1);
        auto ctx = this->forward(stream, input);
        auto vals = forward_activations(*ctx, layer);
        extract_dimension_pos_neg<PARAMS_T>(stream, static_cast<uint32_t>(output.n_elements()),
                                            dimension, width(layer), output.rows(), vals.first,
                                            vals.second, output.data());
    }

    virtual uint32_t width(uint32_t layer) const = 0;
    virtual uint32_t num_forward_activations() const = 0;
    virtual std::pair<const PARAMS_T*, MatrixLayout> forward_activations(const Context& ctx,
                                                                         uint32_t layer) const = 0;

    // Byte offset in one generated forward context, used to expose the same
    // activation diagnostics after a JIT forward pass. Custom networks can
    // implement this independently of their ordinary forward context type.
    virtual uint32_t forward_activation_device_offset(uint32_t layer) const {
        (void)layer;
        unsupported_nn(this->name() + " generated activation diagnostics");
    }
};

// Construct the named network; dimensions belong in the JSON as in the reference.
template <typename T> Network<T, T>* create_network(const json& network);

template <typename T>
std::unique_ptr<Network<T>> default_network(uint32_t n_input_dims, uint32_t n_output_dims,
                                            const std::string& name) {
    return std::unique_ptr<Network<T>>{create_network<T>(
        {{"otype", name}, {"n_input_dims", n_input_dims}, {"n_output_dims", n_output_dims}})};
}

std::vector<std::string> builtin_networks();
std::string select_network(const json& network);
uint32_t minimum_alignment(const json& network);

namespace detail {
// Runtime-generated source lives in the library implementation, keeping the
// installed host headers independent of Metal's shading-language syntax.
std::string generate_mlp_body(const std::string& type, uint32_t input_width, uint32_t width,
                              uint32_t padded_output_width, uint32_t output_width,
                              uint32_t n_hidden, Activation activation,
                              Activation output_activation, uint32_t derivative_order);
// Private automatic-training layout. The public callable overload above
// retains its saved-preactivation representation for forward contexts.
std::string generate_mlp_body(const std::string& type, uint32_t input_width, uint32_t width,
                              uint32_t padded_output_width, uint32_t output_width,
                              uint32_t n_hidden, Activation activation,
                              Activation output_activation, uint32_t derivative_order,
                              bool compact_context);
uint32_t mlp_jit_training_context_bytes(size_t element_size, uint32_t input_width, uint32_t width,
                                        uint32_t padded_output_width, uint32_t n_hidden,
                                        Activation activation, Activation output_activation);
uint32_t mlp_jit_context_bytes(size_t element_size, uint32_t input_width, uint32_t width,
                               uint32_t padded_output_width, uint32_t n_hidden);
uint32_t mlp_jit_simdgroup_scratch_bytes(size_t element_size, uint32_t input_width, uint32_t width,
                                         uint32_t padded_output_width);
} // namespace detail

template <typename T>
std::string generate_mlp_device_code(uint32_t input_width, uint32_t width,
                                     uint32_t padded_output_width, uint32_t output_width,
                                     uint32_t n_hidden, Activation activation,
                                     Activation output_activation) {
    return detail::generate_mlp_body(type_to_string<T>(), input_width, width, padded_output_width,
                                     output_width, n_hidden, activation, output_activation, 0);
}

template <typename T>
std::string generate_backward_mlp_device_code(uint32_t n_threads, uint32_t input_width,
                                              uint32_t width, uint32_t padded_output_width,
                                              uint32_t output_width, uint32_t n_hidden,
                                              Activation activation, Activation output_activation) {
    nn_check(n_threads != 0, "generated MLP backward requires a positive thread count");
    return detail::generate_mlp_body(type_to_string<T>(), input_width, width, padded_output_width,
                                     output_width, n_hidden, activation, output_activation, 1);
}

// The native port also exposes its existing second-order operation through
// generated code. This addition uses the same saved context as first backward.
template <typename T>
std::string generate_backward_backward_input_mlp_device_code(
    uint32_t n_threads, uint32_t input_width, uint32_t width, uint32_t padded_output_width,
    uint32_t output_width, uint32_t n_hidden, Activation activation, Activation output_activation) {
    nn_check(n_threads != 0, "generated MLP second backward requires a positive thread count");
    return detail::generate_mlp_body(type_to_string<T>(), input_width, width, padded_output_width,
                                     output_width, n_hidden, activation, output_activation, 2);
}

template <typename T>
uint32_t mlp_device_code_fwd_ctx_bytes(uint32_t input_width, uint32_t width,
                                       uint32_t padded_output_width, uint32_t output_width,
                                       uint32_t n_hidden, Activation activation,
                                       Activation output_activation) {
    (void)output_width;
    (void)activation;
    (void)output_activation;
    return detail::mlp_jit_context_bytes(sizeof(T), input_width, width, padded_output_width,
                                         n_hidden);
}

template <typename T>
uint32_t backward_mlp_device_code_shmem_bytes(uint32_t n_threads, GradientMode param_gradients_mode,
                                              uint32_t input_width, uint32_t width,
                                              uint32_t padded_output_width) {
    nn_check(static_cast<unsigned>(param_gradients_mode) < 3,
             "invalid generated MLP gradient mode");
    // Matrix products also use this plane when parameter gradients are ignored.
    // A final incomplete SIMD-group executes the scalar path and owns no slice.
    if (n_threads < 32)
        return 0;
    const uint64_t bytes =
        uint64_t(n_threads / 32u) *
        detail::mlp_jit_simdgroup_scratch_bytes(sizeof(T), input_width, width, padded_output_width);
    // Preserve a scalar option for a query whose complete group allocation
    // cannot be represented; never wrap or impose this limit on raw bodies.
    if (bytes > UINT32_MAX)
        return 0;
    return static_cast<uint32_t>(bytes);
}

namespace detail {
// Preserve explicit strides for matrix overloads; raw-pointer overloads remain
// contiguous-vector operations with the original argument order.
template <class T>
void activation_matrix(metalStream_t stream, const GPUMatrixDynamic<T>& values,
                       const GPUMatrixDynamic<T>& gradient, GPUMatrixDynamic<T>& result,
                       Activation activation, uint32_t mode);
} // namespace detail

// Apply an activation, or its derivative with respect to input/output values.
template <typename T>
void activation_gpu(metalStream_t stream, const uint32_t num_elements, const Activation act,
                    const T* in, T* out);

template <typename T>
void activation_gpu(metalStream_t stream, Activation activation, const GPUMatrixDynamic<T>& input,
                    GPUMatrixDynamic<T>& output) {
    nn_check(input.n() == output.n() && input.m() == output.m(), "activation shape mismatch");
    detail::activation_matrix(stream, input, input, output, activation, 0);
}

template <typename T>
void activation_backward_gpu(metalStream_t stream, const uint32_t num_elements,
                             const Activation act, const T* values, const T* gradients_out,
                             T* gradients_in);

template <typename T>
void activation_backward_gpu(metalStream_t stream, Activation activation,
                             const GPUMatrixDynamic<T>& values, GPUMatrixDynamic<T>& gradients) {
    nn_check(values.n() == gradients.n() && values.m() == gradients.m(),
             "activation gradient shape mismatch");
    detail::activation_matrix(stream, values, gradients, gradients, activation, 1);
}

template <typename T>
void activation_backward_output_gpu(metalStream_t stream, const uint32_t num_elements,
                                    const Activation act, const T* output_values,
                                    const T* gradients_out, T* gradients_in);
} // namespace tcnn
