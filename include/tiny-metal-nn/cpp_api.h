#pragma once

#include <nlohmann/json.hpp>

#include <tiny-metal-nn/metal_runtime.h>

#include <memory>
#include <string>
#include <cstdint>
#include <functional>

namespace tcnn {
struct Context {
    Context() = default;

    virtual ~Context() {}

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;
};
} // namespace tcnn

namespace tcnn {
namespace cpp {

enum class LogSeverity {
    Info,
    Debug,
    Warning,
    Error,
    Success,
};

using json = nlohmann::json;

// Return the batch multiple required by the uploaded reference's public API.
uint32_t batch_size_granularity();

// Select the active metalrt device through the Metal host API.
int metal_device();
// Select an available device by its Metal backend index.
void set_metal_device(int device);

// Release idle pooled temporary allocations while retaining in-flight resources.
void free_temporary_memory();

// Report whether this build includes the network factory implementation.
bool has_networks();

enum class Precision {
    Fp32,
    Fp16,
};

// Return the training loss scale associated with the requested storage precision.
float default_loss_scale(Precision p);

// Return the library's configured default network parameter/output precision.
Precision preferred_precision();

// Report automatic model-function fusion support, independently of MSL compilation.
bool supports_jit_fusion(int device = -1);

// Configure optional expanded-MSL source caching and quoted include resolution.
void rtc_set_cache_dir(const std::string& dir);
void rtc_set_include_dir(const std::string& dir);

// Install the process-wide callback used for host diagnostic messages.
void set_log_callback(const std::function<void(LogSeverity, const std::string&)>& callback);

struct Context {
    std::unique_ptr<tcnn::Context> ctx;
};

// Stable non-template facade. Virtual order, argument names, precision fields,
// defaults, and return types retain the reference operation structure with Metal types.
class Module {
  public:
    Module(Precision param_precision, Precision output_precision)
        : m_param_precision{param_precision}, m_output_precision{output_precision} {}

    virtual ~Module() {}

    // Evaluate a batch using caller-owned, backend-allocated input/parameter memory.
    virtual void inference(metalStream_t stream, uint32_t n_elements, const float* input,
                           void* output, void* params) = 0;
    // Save the intermediate values needed to differentiate this exact invocation.
    virtual Context forward(metalStream_t stream, uint32_t n_elements, const float* input,
                            void* output, void* params, bool prepare_input_gradients) = 0;
    // Propagate an output derivative to optional input and parameter destinations.
    virtual void backward(metalStream_t stream, const Context& ctx, uint32_t n_elements,
                          float* dL_dinput, const void* dL_doutput, void* dL_dparams,
                          const float* input, const void* output, const void* params) = 0;
    // Differentiate the input-gradient operation where the selected model supports it.
    virtual void backward_backward_input(metalStream_t stream, const Context& ctx,
                                         uint32_t n_elements, const float* dL_ddLdinput,
                                         const float* input, const void* dL_doutput,
                                         void* dL_dparams, void* dL_ddLdoutput, float* dL_dinput,
                                         const void* params) = 0;

    virtual uint32_t n_input_dims() const = 0;
    virtual uint32_t n_output_dims() const = 0;

    Precision output_precision() const { return m_output_precision; }

    virtual size_t n_params() const = 0;

    Precision param_precision() const { return m_param_precision; }

    virtual void initialize_params(size_t seed, float* params_full_precision,
                                   float scale = 1.0f) = 0;

    virtual json hyperparams() const = 0;
    virtual std::string name() const = 0;

    virtual bool jit_fusion() const = 0;
    virtual void set_jit_fusion(bool val) = 0;

  private:
    Precision m_param_precision;
    Precision m_output_precision;
};

// Create an encoding followed by a network, with network parameters stored first.
Module* create_network_with_input_encoding(uint32_t n_input_dims, uint32_t n_output_dims,
                                           const json& encoding, const json& network);
// Create a network using the configured default precision and caller-owned parameters.
Module* create_network(uint32_t n_input_dims, uint32_t n_output_dims, const json& network);
// Create an encoding with explicitly requested float or half parameter/output storage.
Module* create_encoding(uint32_t n_input_dims, const json& encoding, Precision requested_precision);

} // namespace cpp
} // namespace tcnn
