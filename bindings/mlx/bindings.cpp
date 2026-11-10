// Python/MLX integration only. Native graph/NN code lives in src/module.cpp.
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "src/module.h"

#include <exception>
#include <limits>
#include <optional>
#include <type_traits>

namespace nb = nanobind;
namespace api = tinymetalnn::mlx_bindings;
using namespace nb::literals;

namespace {
using Json = tcnn::cpp::json;

Json from_python(nb::handle value) {
    if (value.is_none())
        return nullptr;
    if (nb::isinstance<nb::bool_>(value))
        return nb::cast<bool>(value);
    if (nb::isinstance<nb::int_>(value)) {
        int64_t signed_value;
        if (nb::try_cast(value, signed_value))
            return signed_value;
        uint64_t unsigned_value;
        if (nb::try_cast(value, unsigned_value))
            return unsigned_value;
        throw std::overflow_error("JSON integer is outside the signed/unsigned 64-bit range");
    }
    if (nb::isinstance<nb::float_>(value))
        return nb::cast<double>(value);
    if (nb::isinstance<nb::str>(value))
        return nb::cast<std::string>(value);
    if (nb::isinstance<nb::dict>(value)) {
        Json result = Json::object();
        for (auto entry : nb::borrow<nb::dict>(value)) {
            if (!nb::isinstance<nb::str>(entry.first))
                throw nb::type_error("Configuration dictionary keys must be strings");
            result[nb::cast<std::string>(entry.first)] = from_python(entry.second);
        }
        return result;
    }
    if (nb::isinstance<nb::list>(value) || nb::isinstance<nb::tuple>(value)) {
        Json result = Json::array();
        for (auto entry : nb::borrow<nb::iterable>(value))
            result.push_back(from_python(entry));
        return result;
    }
    throw nb::type_error("Configuration must contain JSON-compatible Python values");
}

nb::object to_python(const Json& value) {
    if (value.is_null())
        return nb::none();
    if (value.is_boolean())
        return nb::bool_(value.get<bool>());
    if (value.is_number_unsigned())
        return nb::int_(value.get<uint64_t>());
    if (value.is_number_integer())
        return nb::int_(value.get<int64_t>());
    if (value.is_number_float())
        return nb::float_(value.get<double>());
    if (value.is_string()) {
        const auto& text = value.get_ref<const std::string&>();
        return nb::str(text.data(), text.size());
    }
    if (value.is_array()) {
        nb::list result;
        for (const auto& item : value)
            result.append(to_python(item));
        return result;
    }
    if (value.is_object()) {
        nb::dict result;
        for (auto it = value.begin(); it != value.end(); ++it)
            result[nb::str(it.key().data(), it.key().size())] = to_python(it.value());
        return result;
    }
    throw nb::type_error("Native configuration cannot be represented as JSON");
}

struct PythonLogState {
    std::function<void(tcnn::cpp::LogSeverity, const std::string&)> callback;
};

void flush_logs(const std::shared_ptr<PythonLogState>& state) {
    auto records = api::take_logs();
    if (records.empty())
        return;
    auto callback = state->callback;
    for (const auto& record : records)
        callback(record.first, record.second);
}

template <class Function>
std::invoke_result_t<Function> invoke(const std::shared_ptr<PythonLogState>& state,
                                      Function&& function) {
    using Result = std::invoke_result_t<Function>;
    using Stored = std::conditional_t<std::is_void_v<Result>, bool, Result>;
    flush_logs(state);
    std::optional<Stored> result;
    std::exception_ptr failure;
    {
        nb::gil_scoped_release release;
        try {
            if constexpr (std::is_void_v<Result>) {
                function();
                result.emplace(true);
            } else
                result.emplace(function());
        } catch (...) {
            failure = std::current_exception();
        }
    }
    flush_logs(state);
    if (failure)
        std::rethrow_exception(failure);
    if constexpr (!std::is_void_v<Result>)
        return std::move(*result);
}

template <class Result, class... Args>
auto wrap(std::shared_ptr<PythonLogState> state, Result (*function)(Args...)) {
    return [state = std::move(state), function](Args... args) -> Result {
        return invoke(state, [&]() -> Result { return function(std::forward<Args>(args)...); });
    };
}
template <class Class, class Result, class... Args>
auto wrap(std::shared_ptr<PythonLogState> state, Result (Class::*function)(Args...)) {
    return [state = std::move(state), function](Class& self, Args... args) -> Result {
        return invoke(state,
                      [&]() -> Result { return (self.*function)(std::forward<Args>(args)...); });
    };
}
template <class Class, class Result, class... Args>
auto wrap(std::shared_ptr<PythonLogState> state, Result (Class::*function)(Args...) const) {
    return [state = std::move(state), function](const Class& self, Args... args) -> Result {
        return invoke(state,
                      [&]() -> Result { return (self.*function)(std::forward<Args>(args)...); });
    };
}
} // namespace

NB_MODULE(_C, m) {
    // Required by MLX's official extension contract and NB_DOMAIN mlx: arrays,
    // streams and devices use MLX's registered nanobind types.
    nb::module_::import_("mlx.core");
    m.doc() = "Native lazy MLX primitives for tiny-metal-nn";
    auto logs = std::make_shared<PythonLogState>();
    nb::enum_<tcnn::cpp::Precision>(m, "Precision")
        .value("Fp32", tcnn::cpp::Precision::Fp32)
        .value("Fp16", tcnn::cpp::Precision::Fp16)
        .export_values();
    nb::enum_<tcnn::cpp::LogSeverity>(m, "LogSeverity")
        .value("Info", tcnn::cpp::LogSeverity::Info)
        .value("Debug", tcnn::cpp::LogSeverity::Debug)
        .value("Warning", tcnn::cpp::LogSeverity::Warning)
        .value("Error", tcnn::cpp::LogSeverity::Error)
        .value("Success", tcnn::cpp::LogSeverity::Success)
        .export_values();

    m.def("batch_size_granularity", wrap(logs, &tcnn::cpp::batch_size_granularity));
    m.def("has_networks", wrap(logs, &tcnn::cpp::has_networks));
    m.def("preferred_precision", wrap(logs, &tcnn::cpp::preferred_precision));
    m.def("default_loss_scale", wrap(logs, &tcnn::cpp::default_loss_scale));
    m.def("supports_jit_fusion", wrap(logs, &api::supports_jit_fusion), "device"_a = -1);
    m.def("guard_shapes", wrap(logs, &api::guard_inputs), "input"_a, "params"_a, nb::kw_only(),
          "stream"_a = nb::none());
    m.def("free_temporary_memory", wrap(logs, &tcnn::cpp::free_temporary_memory));
    m.def("rtc_set_cache_dir", wrap(logs, &tcnn::cpp::rtc_set_cache_dir));
    m.def("rtc_set_include_dir", wrap(logs, &tcnn::cpp::rtc_set_include_dir));
    m.def("set_log_callback",
          [logs](const std::function<void(tcnn::cpp::LogSeverity, const std::string&)>& callback) {
              // Records made under an existing callback belong to that callback.
              // Drain them before changing registration, on the Python caller thread.
              if (logs->callback)
                  flush_logs(logs);
              logs->callback = callback;
              api::capture_logs();
          });
    m.def(
        "flush_logs", [logs]() { flush_logs(logs); },
        "Deliver lazy native diagnostics on this Python thread after mx.eval/synchronize.");

    nb::class_<api::Module>(m, "Module")
        .def("forward", wrap(logs, &api::Module::forward), "input"_a, "params"_a, nb::kw_only(),
             "stream"_a = nb::none())
        .def("initial_params", wrap(logs, &api::Module::initial_params), "seed"_a, nb::kw_only(),
             "stream"_a = nb::none())
        .def("n_input_dims", wrap(logs, &api::Module::n_input_dims))
        .def("n_output_dims", wrap(logs, &api::Module::n_output_dims))
        .def("n_params", wrap(logs, &api::Module::n_params))
        .def("param_precision", wrap(logs, &api::Module::param_precision))
        .def("output_precision", wrap(logs, &api::Module::output_precision))
        .def("name", wrap(logs, &api::Module::name))
        .def("hyperparams",
             [logs](const api::Module& self) {
                 auto result = invoke(logs, [&]() { return self.hyperparams(); });
                 return to_python(result);
             })
        .def_prop_rw("jit_fusion", wrap(logs, &api::Module::jit_fusion),
                     wrap(logs, &api::Module::set_jit_fusion))
        .def_prop_rw("backward_backward_input_no_input_grad",
                     wrap(logs, &api::Module::backward_backward_input_no_input_grad),
                     wrap(logs, &api::Module::set_backward_backward_input_no_input_grad))
        .def_prop_rw("loss_scale", wrap(logs, &api::Module::loss_scale),
                     wrap(logs, &api::Module::set_loss_scale));

    m.def("create_encoding",
          [logs](uint32_t inputs, nb::handle config, tcnn::cpp::Precision precision) {
              auto json = from_python(config);
              return invoke(logs, [&]() { return api::create_encoding(inputs, json, precision); });
          });
#ifndef TCNN_NO_NETWORKS
    m.def("create_network", [logs](uint32_t inputs, uint32_t outputs, nb::handle config) {
        auto json = from_python(config);
        return invoke(logs, [&]() { return api::create_network(inputs, outputs, json); });
    });
    m.def("create_network_with_input_encoding",
          [logs](uint32_t inputs, uint32_t outputs, nb::handle encoding, nb::handle network) {
              auto encoding_json = from_python(encoding);
              auto network_json = from_python(network);
              return invoke(logs, [&]() {
                  return api::create_network_with_input_encoding(inputs, outputs, encoding_json,
                                                                 network_json);
              });
          });
#endif
}
