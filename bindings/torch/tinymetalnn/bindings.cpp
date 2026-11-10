#include <torch/extension.h>
#include <pybind11/functional.h>
#include <nlohmann/json.hpp>
#include <pybind11_json/pybind11_json.hpp>
#include <tiny-metal-nn/cpp_api.h>

#include "mps_bridge.h"
#include "native_log.h"

#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using metalrt::Access;
using tinymetalnn::torch_bindings::bind_tensor;
using tinymetalnn::torch_bindings::copy_back;
using tinymetalnn::torch_bindings::execute_mps;
using tinymetalnn::torch_bindings::MPSSession;
using tinymetalnn::torch_bindings::TensorBuffer;

// This state is owned only by Python function closures. Its std::function may
// contain Python objects, so it is accessed and destroyed with the GIL held.
// The native library receives only capture_native_log, never this callback.
struct PythonLogState {
    std::function<void(tcnn::cpp::LogSeverity, const std::string&)> callback;
};

template <class Function>
std::invoke_result_t<Function> invoke_with_logs(const std::shared_ptr<PythonLogState>& state,
                                                Function&& function) {
    using Result = std::invoke_result_t<Function>;
    using StoredResult = std::conditional_t<std::is_void_v<Result>, bool, Result>;
    tinymetalnn::torch_bindings::NativeLogCapture capture;
    std::optional<StoredResult> result;
    std::exception_ptr failure;
    {
        py::gil_scoped_release release;
        try {
            if constexpr (std::is_void_v<Result>) {
                function();
                result.emplace(true);
            } else {
                result.emplace(function());
            }
        } catch (...) {
            failure = std::current_exception();
        }
    }

    // All native/module/submission locks have unwound before calling Python.
    // Snapshot under the GIL so reentrant callbacks can replace the configured
    // callback without destroying the callable currently being invoked.
    auto pending = tinymetalnn::torch_bindings::pending_native_logs().take();
    auto records = capture.queue()->take();
    if (!pending.empty() || !records.empty()) {
        const auto callback = state->callback;
        for (const auto& record : pending)
            callback(record.severity, record.message);
        for (const auto& record : records)
            callback(record.severity, record.message);
    }
    // A callback exception propagates directly, including when native work had
    // already failed. Otherwise preserve the original native exception.
    if (failure)
        std::rethrow_exception(failure);
    if constexpr (!std::is_void_v<Result>)
        return std::move(*result);
}

// Keep the original callable signatures visible to pybind while centralizing
// explicit GIL release and delivery. No Python callbacks run from destructors.
template <class Result, class... Args>
auto with_logs(std::shared_ptr<PythonLogState> state, Result (*function)(Args...)) {
    return [state = std::move(state), function](Args... args) -> Result {
        return invoke_with_logs(state,
                                [&]() -> Result { return function(std::forward<Args>(args)...); });
    };
}

template <class Class, class Result, class... Args>
auto with_logs(std::shared_ptr<PythonLogState> state, Result (Class::*function)(Args...)) {
    return [state = std::move(state), function](Class& self, Args... args) -> Result {
        return invoke_with_logs(
            state, [&]() -> Result { return (self.*function)(std::forward<Args>(args)...); });
    };
}

template <class Class, class Result, class... Args>
auto with_logs(std::shared_ptr<PythonLogState> state, Result (Class::*function)(Args...) const) {
    return [state = std::move(state), function](const Class& self, Args... args) -> Result {
        return invoke_with_logs(
            state, [&]() -> Result { return (self.*function)(std::forward<Args>(args)...); });
    };
}

c10::ScalarType torch_type(tcnn::cpp::Precision precision) {
    switch (precision) {
    case tcnn::cpp::Precision::Fp32:
        return torch::kFloat32;
    case tcnn::cpp::Precision::Fp16:
        return torch::kHalf;
    default:
        throw std::runtime_error("Unknown precision tcnn->torch");
    }
}

void check_tensor(const torch::Tensor& tensor, const char* name, c10::ScalarType type,
                  int64_t rank) {
    TORCH_CHECK(tensor.defined(), name, " must be a defined tensor.");
    TORCH_CHECK(tensor.device().is_mps(), name, " must be on the MPS device.");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
    TORCH_CHECK(tensor.scalar_type() == type, name, " has the wrong dtype; expected ", type, ".");
    TORCH_CHECK(tensor.dim() == rank, name, " must have rank ", rank, ".");
}

struct ModuleState {
    explicit ModuleState(std::unique_ptr<tcnn::cpp::Module> value) : module{std::move(value)} {}
    std::unique_ptr<tcnn::cpp::Module> module;
    // The facade rebinds caller-owned parameters for each invocation. Protect
    // this mutable state when Python threads or autograd workers share a model.
    std::mutex mutex;
    bool backward_backward_input_no_input_grad = false;
};

// Python sees the same opaque Context API as upstream. The extra ownership
// keeps imported storage and private-buffer staging alive for saved native
// contexts, without retaining TensorImpl/grad_fn objects and creating cycles.
struct Context {
    tcnn::cpp::Context native;
    std::shared_ptr<ModuleState> owner;
    std::vector<TensorBuffer> resources;
    uint32_t batch = 0;
    bool training = false;
    bool prepared_input_gradients = false;
    mutable metalrt::EventPoint completion;
    mutable metalrt::Stream* producer = nullptr;
};

void wait_context(MPSSession& session, const Context& context) {
    if (context.completion && context.producer != session.queue.get())
        session.queue->wait(context.completion);
}

void record_context(MPSSession& session, const Context& context) {
    metalrt::TimelineEvent event{session.device};
    context.completion = session.queue->signal(event);
    context.producer = session.queue.get();
}

class Module {
  public:
    explicit Module(std::unique_ptr<tcnn::cpp::Module> module)
        : m_state{std::make_shared<ModuleState>(std::move(module))} {}

    std::tuple<Context, torch::Tensor> fwd(torch::Tensor input, torch::Tensor params) {
        check_input_params(input, params);
        const auto batch = static_cast<uint32_t>(input.size(0));
        auto output =
            torch::empty({input.size(0), static_cast<int64_t>(n_output_dims())},
                         input.options().dtype(c10_output_precision()).requires_grad(false));
        Context context;
        context.owner = m_state;
        context.batch = batch;
        context.training = input.requires_grad() || params.requires_grad();
        context.prepared_input_gradients = input.requires_grad();
        if (!batch)
            return {std::move(context), output};

        std::lock_guard<std::mutex> lock{m_state->mutex};
        execute_mps([&](MPSSession& session) {
            {
                auto recording = session.queue->batch();
                auto x = bind_tensor(session, input, Access::Read);
                auto p = bind_tensor(session, params, Access::Read);
                auto y = bind_tensor(session, output, Access::Write);
                if (context.training) {
                    context.native = m_state->module->forward(session.nn_stream->get(), batch,
                                                              x.data_as<float>(), y.data(),
                                                              p.data(), input.requires_grad());
                    context.resources = {x, p, y};
                } else {
                    m_state->module->inference(session.nn_stream->get(), batch, x.data_as<float>(),
                                               y.data(), p.data());
                }
                copy_back(session, y);
                recording.finish();
            }
            if (context.training)
                record_context(session, context);
        });
        return {std::move(context), output};
    }

    std::tuple<torch::Tensor, torch::Tensor> bwd(const Context& context, torch::Tensor input,
                                                 torch::Tensor params, torch::Tensor output,
                                                 torch::Tensor dL_doutput) {
        check_context(context, input);
        check_input_params(input, params);
        check_output(output, input, "output");
        check_output(dL_doutput, input, "dL_doutput");
        TORCH_CHECK(!input.requires_grad() || context.prepared_input_gradients,
                    "The forward context was not prepared for input gradients.");

        torch::Tensor dL_dinput, dL_dparams;
        if (input.requires_grad())
            dL_dinput = torch::zeros_like(input, input.options().requires_grad(false));
        if (params.requires_grad())
            dL_dparams = torch::zeros_like(params, params.options().requires_grad(false));
        if (!context.batch || (!dL_dinput.defined() && !dL_dparams.defined()))
            return {dL_dinput, dL_dparams};

        std::lock_guard<std::mutex> lock{m_state->mutex};
        execute_mps([&](MPSSession& session) {
            wait_context(session, context);
            {
                auto recording = session.queue->batch();
                auto x = bind_tensor(session, input, Access::Read);
                auto p = bind_tensor(session, params, Access::Read);
                auto y = bind_tensor(session, output, Access::Read);
                auto dy = bind_tensor(session, dL_doutput, Access::Read);
                TensorBuffer dx, dp;
                if (dL_dinput.defined())
                    dx = bind_tensor(session, dL_dinput, Access::Write);
                if (dL_dparams.defined())
                    dp = bind_tensor(session, dL_dparams, Access::Write);
                m_state->module->backward(session.nn_stream->get(), context.native, context.batch,
                                          dL_dinput.defined() ? dx.data_as<float>() : nullptr,
                                          dy.data(), dL_dparams.defined() ? dp.data() : nullptr,
                                          x.data_as<float>(), y.data(), p.data());
                copy_back(session, dx);
                copy_back(session, dp);
                recording.finish();
            }
            record_context(session, context);
        });
        return {dL_dinput, dL_dparams};
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    bwd_bwd_input(const Context& context, torch::Tensor input, torch::Tensor params,
                  torch::Tensor dL_ddLdinput, torch::Tensor dL_doutput) {
        check_context(context, input);
        check_input_params(input, params);
        check_output(dL_doutput, input, "dL_doutput");
        check_tensor(dL_ddLdinput, "dL_ddLdinput", torch::kFloat32, 2);
        TORCH_CHECK(dL_ddLdinput.sizes() == input.sizes(),
                    "dL_ddLdinput must have the input shape.");
        TORCH_CHECK(dL_ddLdinput.device() == input.device(),
                    "All tensors must be on the same MPS device.");
        TORCH_CHECK(context.prepared_input_gradients,
                    "The forward context was not prepared for input gradients.");

        std::lock_guard<std::mutex> lock{m_state->mutex};
        torch::Tensor dL_ddLdoutput, dL_dparams, dL_dinput;
        if (dL_doutput.requires_grad())
            dL_ddLdoutput =
                torch::zeros_like(dL_doutput, dL_doutput.options().requires_grad(false));
        if (params.requires_grad())
            dL_dparams = torch::zeros_like(params, params.options().requires_grad(false));
        if (input.requires_grad() && !m_state->backward_backward_input_no_input_grad)
            dL_dinput = torch::zeros_like(input, input.options().requires_grad(false));
        if (!context.batch ||
            (!dL_ddLdoutput.defined() && !dL_dparams.defined() && !dL_dinput.defined()))
            return {dL_ddLdoutput, dL_dparams, dL_dinput};

        execute_mps([&](MPSSession& session) {
            wait_context(session, context);
            {
                auto recording = session.queue->batch();
                auto x = bind_tensor(session, input, Access::Read);
                auto p = bind_tensor(session, params, Access::Read);
                auto v = bind_tensor(session, dL_ddLdinput, Access::Read);
                auto dy = bind_tensor(session, dL_doutput, Access::Read);
                TensorBuffer ddy, dp, dx;
                // Preserve upstream's zero-initialized accumulation destinations.
                if (dL_ddLdoutput.defined())
                    ddy = bind_tensor(session, dL_ddLdoutput, Access::ReadWrite);
                if (dL_dparams.defined())
                    dp = bind_tensor(session, dL_dparams, Access::ReadWrite);
                if (dL_dinput.defined())
                    dx = bind_tensor(session, dL_dinput, Access::ReadWrite);
                m_state->module->backward_backward_input(
                    session.nn_stream->get(), context.native, context.batch, v.data_as<float>(),
                    x.data_as<float>(), dy.data(), dL_dparams.defined() ? dp.data() : nullptr,
                    dL_ddLdoutput.defined() ? ddy.data() : nullptr,
                    dL_dinput.defined() ? dx.data_as<float>() : nullptr, p.data());
                copy_back(session, ddy);
                copy_back(session, dp);
                copy_back(session, dx);
                recording.finish();
            }
            record_context(session, context);
        });
        return {dL_ddLdoutput, dL_dparams, dL_dinput};
    }

    torch::Tensor initial_params(size_t seed) {
        auto output =
            torch::empty({static_cast<int64_t>(n_params())},
                         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS));
        if (!n_params())
            return output;
        std::lock_guard<std::mutex> lock{m_state->mutex};
        execute_mps([&](MPSSession& session) {
            // The existing stream-less facade initializes MLP matrices on the
            // host and lattice parameters on its default Metal stream. Use fresh
            // runtime storage so pending Torch writes cannot race those accesses.
            const auto bytes = metalrt::checked_mul(static_cast<size_t>(n_params()), sizeof(float));
            auto initialized = session.device.allocate(bytes);
            std::memset(initialized.contents(), 0, bytes);
            m_state->module->initialize_params(seed, initialized.data<float>());
            metalrt::ordered_default_stream(session.device).synchronize();
            auto recording = session.queue->batch();
            auto target = bind_tensor(session, output, Access::Write);
            session.queue->copy(initialized.view(), target.original);
            recording.finish();
        });
        return output;
    }

    uint32_t n_input_dims() const { return m_state->module->n_input_dims(); }
    uint32_t n_output_dims() const { return m_state->module->n_output_dims(); }
    uint32_t n_params() const {
        const auto count = m_state->module->n_params();
        TORCH_CHECK(count <= std::numeric_limits<uint32_t>::max(),
                    "Parameter count exceeds the native binding ABI.");
        return static_cast<uint32_t>(count);
    }
    tcnn::cpp::Precision param_precision() const { return m_state->module->param_precision(); }
    tcnn::cpp::Precision output_precision() const { return m_state->module->output_precision(); }
    c10::ScalarType c10_param_precision() const { return torch_type(param_precision()); }
    c10::ScalarType c10_output_precision() const { return torch_type(output_precision()); }
    nlohmann::json hyperparams() const { return m_state->module->hyperparams(); }
    std::string name() const { return m_state->module->name(); }
    bool jit_fusion() const {
        std::lock_guard<std::mutex> lock{m_state->mutex};
        return m_state->module->jit_fusion();
    }
    void set_jit_fusion(bool value) {
        std::lock_guard<std::mutex> lock{m_state->mutex};
        execute_mps([&](MPSSession&) { m_state->module->set_jit_fusion(value); });
    }
    bool backward_backward_input_no_input_grad() const {
        std::lock_guard<std::mutex> lock{m_state->mutex};
        return m_state->backward_backward_input_no_input_grad;
    }
    void set_backward_backward_input_no_input_grad(bool value) {
        std::lock_guard<std::mutex> lock{m_state->mutex};
        m_state->backward_backward_input_no_input_grad = value;
    }

  private:
    void check_input_params(const torch::Tensor& input, const torch::Tensor& params) const {
        check_tensor(input, "input", torch::kFloat32, 2);
        check_tensor(params, "params", c10_param_precision(), 1);
        TORCH_CHECK(input.size(1) == n_input_dims(), "input has the wrong feature width.");
        TORCH_CHECK(params.numel() == n_params(), "params has the wrong parameter count.");
        TORCH_CHECK(input.device() == params.device(),
                    "input and params must be on the same MPS device.");
        TORCH_CHECK(input.size(0) <= std::numeric_limits<uint32_t>::max(),
                    "Batch size exceeds uint32_t.");
        TORCH_CHECK(input.size(0) % tcnn::cpp::batch_size_granularity() == 0,
                    "Native batch size must be a multiple of batch_size_granularity().");
    }
    void check_output(const torch::Tensor& output, const torch::Tensor& input,
                      const char* name) const {
        check_tensor(output, name, c10_output_precision(), 2);
        TORCH_CHECK(output.size(0) == input.size(0) && output.size(1) == n_output_dims(), name,
                    " must have the padded native output shape.");
        TORCH_CHECK(output.device() == input.device(),
                    "All tensors must be on the same MPS device.");
    }
    void check_context(const Context& context, const torch::Tensor& input) const {
        TORCH_CHECK(context.owner == m_state && context.training &&
                        (context.native.ctx || context.batch == 0),
                    "Called with invalid context; fwd may have run in inference mode or on a "
                    "different module.");
        TORCH_CHECK(input.dim() == 2 && input.size(0) == context.batch,
                    "Forward context batch does not match input.");
    }
    std::shared_ptr<ModuleState> m_state;
};

#ifndef TCNN_NO_NETWORKS
Module create_network_with_input_encoding(uint32_t n_input_dims, uint32_t n_output_dims,
                                          const nlohmann::json& encoding,
                                          const nlohmann::json& network) {
    std::unique_ptr<tcnn::cpp::Module> result;
    execute_mps([&](MPSSession&) {
        result.reset(tcnn::cpp::create_network_with_input_encoding(n_input_dims, n_output_dims,
                                                                   encoding, network));
    });
    return Module{std::move(result)};
}

Module create_network(uint32_t n_input_dims, uint32_t n_output_dims,
                      const nlohmann::json& network) {
    std::unique_ptr<tcnn::cpp::Module> result;
    execute_mps([&](MPSSession&) {
        result.reset(tcnn::cpp::create_network(n_input_dims, n_output_dims, network));
    });
    return Module{std::move(result)};
}
#endif

Module create_encoding(uint32_t n_input_dims, const nlohmann::json& encoding,
                       tcnn::cpp::Precision requested_precision) {
    std::unique_ptr<tcnn::cpp::Module> result;
    execute_mps([&](MPSSession&) {
        result.reset(tcnn::cpp::create_encoding(n_input_dims, encoding, requested_precision));
    });
    return Module{std::move(result)};
}

bool supports_jit_fusion(int device = -1) {
    bool supported = false;
    execute_mps([&](MPSSession&) { supported = tcnn::cpp::supports_jit_fusion(device); });
    return supported;
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    auto log_state = std::make_shared<PythonLogState>();
    py::enum_<tcnn::cpp::LogSeverity>(m, "LogSeverity", py::module_local())
        .value("Info", tcnn::cpp::LogSeverity::Info)
        .value("Debug", tcnn::cpp::LogSeverity::Debug)
        .value("Warning", tcnn::cpp::LogSeverity::Warning)
        .value("Error", tcnn::cpp::LogSeverity::Error)
        .value("Success", tcnn::cpp::LogSeverity::Success)
        .export_values();
    py::enum_<tcnn::cpp::Precision>(m, "Precision", py::module_local())
        .value("Fp32", tcnn::cpp::Precision::Fp32)
        .value("Fp16", tcnn::cpp::Precision::Fp16)
        .export_values();

    m.def("batch_size_granularity", with_logs(log_state, &tcnn::cpp::batch_size_granularity));
    m.def("default_loss_scale", with_logs(log_state, &tcnn::cpp::default_loss_scale));
    m.def("free_temporary_memory", with_logs(log_state, &tcnn::cpp::free_temporary_memory));
    m.def("has_networks", with_logs(log_state, &tcnn::cpp::has_networks));
    m.def("preferred_precision", with_logs(log_state, &tcnn::cpp::preferred_precision));
    m.def("supports_jit_fusion", with_logs(log_state, &supports_jit_fusion),
          py::arg("device") = -1);
    m.def("rtc_set_cache_dir", with_logs(log_state, &tcnn::cpp::rtc_set_cache_dir));
    m.def("rtc_set_include_dir", with_logs(log_state, &tcnn::cpp::rtc_set_include_dir));
    m.def("set_log_callback",
          [log_state](
              const std::function<void(tcnn::cpp::LogSeverity, const std::string&)>& callback) {
              // Preserve the upstream empty-callback behavior; an empty callback will
              // throw bad_function_call when a subsequent log is delivered.
              log_state->callback = callback;
              tcnn::cpp::set_log_callback(tinymetalnn::torch_bindings::capture_native_log);
          });

    py::class_<Context>(m, "Context", py::module_local());
    py::class_<Module>(m, "Module", py::module_local())
        .def("fwd", with_logs(log_state, &Module::fwd))
        .def("bwd", with_logs(log_state, &Module::bwd))
        .def("bwd_bwd_input", with_logs(log_state, &Module::bwd_bwd_input))
        .def("initial_params", with_logs(log_state, &Module::initial_params))
        .def("n_input_dims", with_logs(log_state, &Module::n_input_dims))
        .def("n_output_dims", with_logs(log_state, &Module::n_output_dims))
        .def("n_params", with_logs(log_state, &Module::n_params))
        .def("param_precision", with_logs(log_state, &Module::param_precision))
        .def("output_precision", with_logs(log_state, &Module::output_precision))
        .def("hyperparams", with_logs(log_state, &Module::hyperparams))
        .def("name", with_logs(log_state, &Module::name))
        .def_property("jit_fusion", with_logs(log_state, &Module::jit_fusion),
                      with_logs(log_state, &Module::set_jit_fusion))
        .def_property("backward_backward_input_no_input_grad",
                      with_logs(log_state, &Module::backward_backward_input_no_input_grad),
                      with_logs(log_state, &Module::set_backward_backward_input_no_input_grad));
#ifndef TCNN_NO_NETWORKS
    m.def("create_network_with_input_encoding",
          with_logs(log_state, &create_network_with_input_encoding));
    m.def("create_network", with_logs(log_state, &create_network));
#endif
    m.def("create_encoding", with_logs(log_state, &create_encoding));
}
