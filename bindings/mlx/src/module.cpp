// Lazy MLX primitives over the native tiny-metal-nn facade. No NN arithmetic is
// evaluated by MLX's CPU backend. MLX graph operations implement transforms and
// layout adaptation; every model/derivative evaluation uses the native runtime.
#include "module.h"
#include "mlx_bridge.h"

#include <mlx/ops.h>
#include <mlx/primitives.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace mx = mlx::core;
namespace tinymetalnn {
namespace mlx_bindings {
namespace {

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::invalid_argument("tinymetalnn_mlx: " + message);
}

mx::Stream gpu_stream(mx::StreamOrDevice value = {}) {
    auto stream = mx::to_stream(value, mx::Device::gpu);
    require(stream.device.type == mx::Device::gpu,
            "native model execution requires a Metal GPU stream");
    return stream;
}

mx::Dtype dtype(tcnn::cpp::Precision precision) {
    if (precision == tcnn::cpp::Precision::Fp32)
        return mx::float32;
    if (precision == tcnn::cpp::Precision::Fp16)
        return mx::float16;
    throw std::invalid_argument("Unknown native precision");
}

int shape_dim(size_t value) {
    require(value <= static_cast<size_t>(std::numeric_limits<mx::ShapeElem>::max()),
            "dimension exceeds MLX's signed 32-bit shape representation");
    return static_cast<int>(value);
}

std::vector<mx::Shape> shapes(const std::vector<mx::array>& inputs) {
    std::vector<mx::Shape> result;
    for (const auto& input : inputs)
        result.push_back(input.shape());
    return result;
}

struct Options {
    std::shared_ptr<ModuleState> state;
    bool jit;
    bool no_second_input_grad;
    float loss_scale;
    bool operator==(const Options&) const = default;
};

Options snapshot(const std::shared_ptr<ModuleState>& state) {
    std::lock_guard<std::mutex> lock{state->mutex};
    return {state, state->jit, state->no_second_input_grad, state->loss_scale};
}

enum class Kind {
    Forward,
    InputBackward,
    ParameterBackward,
    InputJvp,
    SecondInput,
    SecondParams,
    Initialize
};

mx::Shape result_shape(Kind kind, const Options& options, const std::vector<mx::array>& inputs) {
    auto& native = *options.state->native;
    if (kind == Kind::Initialize || kind == Kind::ParameterBackward || kind == Kind::SecondParams)
        return {shape_dim(native.n_params())};
    if (kind == Kind::InputBackward || kind == Kind::SecondInput)
        return inputs[0].shape();
    return {inputs[0].shape(0), shape_dim(native.n_output_dims())};
}

mx::Dtype result_dtype(Kind kind, const Options& options) {
    if (kind == Kind::Initialize || kind == Kind::InputBackward || kind == Kind::SecondInput)
        return mx::float32;
    if (kind == Kind::ParameterBackward || kind == Kind::SecondParams)
        return dtype(options.state->native->param_precision());
    return dtype(options.state->native->output_precision());
}

void check_inputs(Kind kind, const Options& options, const std::vector<mx::array>& inputs) {
    if (kind == Kind::Initialize) {
        require(inputs.empty(), "initializer takes no array inputs");
        return;
    }
    const size_t count = kind == Kind::Forward                                       ? 2
                         : (kind == Kind::SecondInput || kind == Kind::SecondParams) ? 4
                                                                                     : 3;
    require(inputs.size() == count, "invalid native primitive input count");
    const auto& x = inputs[0];
    const auto& p = inputs[1];
    auto& native = *options.state->native;
    require(x.ndim() == 2 && x.dtype() == mx::float32, "input must be a rank-2 float32 array");
    require(x.shape(1) == shape_dim(native.n_input_dims()),
            "input width does not match the module");
    require(x.shape(0) % tcnn::cpp::batch_size_granularity() == 0,
            "native input batch must satisfy batch_size_granularity");
    require(p.ndim() == 1 && p.size() == native.n_params(),
            "parameters must be the exact flat native parameter vector");
    require(p.dtype() == dtype(native.param_precision()),
            "parameters have the wrong native execution dtype");
    if (kind == Kind::InputJvp) {
        require(inputs[2].shape() == x.shape() && inputs[2].dtype() == mx::float32,
                "input tangent must have the input shape and float32 dtype");
    } else if (count >= 3) {
        const mx::Shape expected{x.shape(0), shape_dim(native.n_output_dims())};
        require(inputs[2].shape() == expected &&
                    inputs[2].dtype() == dtype(native.output_precision()),
                "output cotangent must have the padded native output shape and dtype");
    }
    if (count == 4)
        require(inputs[3].shape() == x.shape() && inputs[3].dtype() == mx::float32,
                "input-gradient cotangent must have the input shape and float32 dtype");
}

// Trace-time batching/basis expansions must not silently retain stale loop
// extents in MLX's shape-changing shapeless compilation. Ordinary mx.compile
// specializes on shapes and remains supported for every valid shape.
class ShapeGuard : public mx::Primitive {
  public:
    ShapeGuard(mx::Stream stream, std::vector<mx::Shape> expected)
        : Primitive(stream), expected_{std::move(expected)} {}
    const char* name() const override { return "TinyMetalShapeGuard"; }
    void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
        for (size_t i = 0; i < inputs.size(); ++i)
            outputs[i].copy_shared_buffer(inputs[i]);
    }
    void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
        eval_cpu(inputs, outputs);
    }
    std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override {
        require(shapes(inputs) == expected_,
                "shape-changing shapeless compilation is not implemented for native binding "
                "transforms; use shape-specialized mx.compile");
        return expected_;
    }
    std::vector<mx::array> vjp(const std::vector<mx::array>&,
                               const std::vector<mx::array>& cotangents,
                               const std::vector<int>& argnums,
                               const std::vector<mx::array>&) override {
        std::vector<mx::array> result;
        for (int arg : argnums)
            result.push_back(cotangents.at(arg));
        return result;
    }
    std::vector<mx::array> jvp(const std::vector<mx::array>& primals,
                               const std::vector<mx::array>& tangents,
                               const std::vector<int>& argnums) override {
        std::vector<mx::array> result;
        for (size_t i = 0; i < primals.size(); ++i) {
            auto it = std::find(argnums.begin(), argnums.end(), static_cast<int>(i));
            result.push_back(it == argnums.end() ? mx::zeros_like(primals[i], stream())
                                                 : tangents[it - argnums.begin()]);
        }
        return result;
    }
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(const std::vector<mx::array>& inputs,
                                                             const std::vector<int>& axes) override;
    bool is_equivalent(const mx::Primitive& other) const override {
        const auto* guard = dynamic_cast<const ShapeGuard*>(&other);
        return guard && guard->expected_ == expected_;
    }

  private:
    std::vector<mx::Shape> expected_;
};

std::vector<mx::array> guard_shapes(const std::vector<mx::array>& inputs, mx::Stream stream) {
    // Independent guards preserve independent differentiation dependencies.
    // A joint multi-output identity would make constant parameters appear
    // dependent on a traced input, requesting an unrelated parameter Hessian.
    std::vector<mx::array> result;
    for (const auto& input : inputs) {
        std::vector<mx::array> single{input};
        result.emplace_back(input.shape(), input.dtype(),
                            std::make_shared<ShapeGuard>(stream, shapes(single)), single);
    }
    return result;
}

std::pair<std::vector<mx::array>, std::vector<int>>
ShapeGuard::vmap(const std::vector<mx::array>& inputs, const std::vector<int>& axes) {
    return {guard_shapes(inputs, stream()), axes};
}

mx::array with_shape_dependencies(const mx::array&, const std::vector<mx::array>&, mx::Stream);

// MLX 0.32.4's general Depends primitive does not implement shape inference,
// JVP, or vmap. This identity retains the control dependencies needed by empty
// and suppressed results while implementing all applicable transformations.
class ShapeDependency : public mx::Primitive {
  public:
    ShapeDependency(mx::Stream stream, const std::vector<mx::array>& inputs)
        : Primitive(stream), expected_{shapes(inputs)} {}
    const char* name() const override { return "TinyMetalShapeDependency"; }
    void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
        outputs[0].copy_shared_buffer(inputs[0]);
    }
    void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
        eval_cpu(inputs, outputs);
    }
    std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override {
        require(shapes(inputs) == expected_,
                "shape-changing shapeless compilation is not implemented for native binding "
                "transforms; use shape-specialized mx.compile");
        return {inputs[0].shape()};
    }
    std::vector<mx::array> vjp(const std::vector<mx::array>& primals,
                               const std::vector<mx::array>& cotangents,
                               const std::vector<int>& argnums,
                               const std::vector<mx::array>&) override {
        std::vector<mx::array> result;
        for (int arg : argnums)
            result.push_back(with_shape_dependencies(
                arg == 0 ? cotangents[0] : mx::zeros_like(primals.at(arg), stream()), primals,
                stream()));
        return result;
    }
    std::vector<mx::array> jvp(const std::vector<mx::array>& primals,
                               const std::vector<mx::array>& tangents,
                               const std::vector<int>& argnums) override {
        auto it = std::find(argnums.begin(), argnums.end(), 0);
        auto value = it == argnums.end() ? mx::zeros_like(primals[0], stream())
                                         : tangents[it - argnums.begin()];
        return {with_shape_dependencies(value, primals, stream())};
    }
    std::pair<std::vector<mx::array>, std::vector<int>>
    vmap(const std::vector<mx::array>& inputs, const std::vector<int>& axes) override {
        return {{with_shape_dependencies(inputs[0], {inputs.begin() + 1, inputs.end()}, stream())},
                {axes[0]}};
    }
    bool is_equivalent(const mx::Primitive& other) const override {
        const auto* dependency = dynamic_cast<const ShapeDependency*>(&other);
        return dependency && dependency->expected_ == expected_;
    }

  private:
    std::vector<mx::Shape> expected_;
};

mx::array with_shape_dependencies(const mx::array& value,
                                  const std::vector<mx::array>& dependencies, mx::Stream stream) {
    std::vector<mx::array> inputs{value};
    // Control dependencies carry shape/lifetime, not numerical derivatives.
    for (const auto& dependency : dependencies)
        inputs.push_back(mx::stop_gradient(dependency, stream));
    return mx::array(value.shape(), value.dtype(),
                     std::make_shared<ShapeDependency>(stream, inputs), inputs);
}

mx::array operation(Kind, const Options&, std::vector<mx::array>, mx::Stream, size_t seed = 0);

mx::array tangent_cast(const mx::array& value, mx::Dtype type, mx::Stream stream) {
    // MLX permits tangent/cotangent dtypes different from the primal. Float64
    // storage is CPU-only in MLX Metal, so its boundary conversion stays lazy
    // on the CPU before the native primitive consumes float32/float16 storage.
    if (value.dtype() == mx::float64 && type != mx::float64)
        return mx::astype(value, type, mx::default_stream(mx::Device::cpu));
    return mx::astype(value, type, stream);
}

mx::array scaled_backward(Kind kind, const Options& options, const std::vector<mx::array>& inputs,
                          const mx::array& cotangent, mx::Stream stream) {
    auto scaled = mx::multiply(cotangent, mx::array(options.loss_scale, cotangent.dtype()), stream);
    auto gradient = operation(kind, options, {inputs[0], inputs[1], scaled}, stream);
    return mx::divide(gradient, mx::array(options.loss_scale, gradient.dtype()), stream);
}

mx::array dot(const mx::array& a, const mx::array& b, mx::Stream stream) {
    return mx::sum(mx::multiply(mx::astype(a, mx::float32, stream),
                                mx::astype(b, mx::float32, stream), stream),
                   stream);
}

mx::array basis(const mx::Shape& shape, mx::Dtype type, size_t index, mx::Stream stream) {
    size_t count = 1;
    for (auto dim : shape)
        count = metalrt::checked_mul(count, static_cast<size_t>(dim));
    require(count <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
            "basis index range exceeds int64");
    auto positions = mx::arange(0.0, static_cast<double>(count), 1.0, mx::int64, stream);
    auto selected = mx::equal(positions, mx::array(static_cast<int64_t>(index), mx::int64), stream);
    return mx::reshape(mx::astype(selected, type, stream), shape, stream);
}

mx::array stack_values(std::vector<mx::array> values, const mx::Shape& shape, mx::Dtype type,
                       mx::Stream stream) {
    if (values.empty())
        return mx::zeros(shape, type, stream);
    return mx::astype(mx::reshape(mx::stack(values, 0, stream), shape, stream), type, stream);
}

// Exact native fallback for a direction in parameter space. The facade has no
// parameter-JVP entry point, so contract-equivalent basis VJPs are contracted
// with the tangent using lazy MLX GPU operations. No numerical differencing.
mx::array parameter_jvp(const Options& options, const mx::array& x, const mx::array& p,
                        const mx::array& tangent, mx::Stream stream) {
    auto guarded = guard_shapes({x, p, tangent}, stream);
    const auto output_shape = result_shape(Kind::Forward, options, {x, p});
    const auto output_type = result_dtype(Kind::Forward, options);
    const size_t count = metalrt::checked_mul(static_cast<size_t>(output_shape[0]),
                                              static_cast<size_t>(output_shape[1]));
    if (!p.size() || !count)
        return with_shape_dependencies(mx::zeros(output_shape, output_type, stream), guarded,
                                       stream);
    std::vector<mx::array> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        auto dy = basis(output_shape, output_type, i, stream);
        auto gradient =
            scaled_backward(Kind::ParameterBackward, options, {guarded[0], guarded[1]}, dy, stream);
        values.push_back(dot(gradient, guarded[2], stream));
    }
    return stack_values(std::move(values), output_shape, output_type, stream);
}

// Transpose the existing mixed input/parameter derivative exactly through an
// input basis. This covers B_p -> x without inventing a parameter Hessian API.
mx::array mixed_parameter_direction(const Options& options, const mx::array& x, const mx::array& p,
                                    const mx::array& dy, const mx::array& direction,
                                    mx::Stream stream) {
    auto guarded = guard_shapes({x, p, dy, direction}, stream);
    if (!x.size() || !p.size())
        return with_shape_dependencies(mx::zeros_like(x, stream), guarded, stream);
    std::vector<mx::array> values;
    values.reserve(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        auto v = basis(x.shape(), mx::float32, i, stream);
        auto mixed =
            operation(Kind::SecondParams, options, {guarded[0], guarded[1], guarded[2], v}, stream);
        values.push_back(dot(mixed, guarded[3], stream));
    }
    return stack_values(std::move(values), x.shape(), mx::float32, stream);
}

class NativePrimitive : public mx::Primitive {
  public:
    NativePrimitive(mx::Stream stream, Kind kind, Options options,
                    const std::vector<mx::array>& inputs, size_t seed)
        : Primitive(stream), kind_{kind}, options_{std::move(options)}, expected_{shapes(inputs)},
          seed_{seed} {}
    const char* name() const override {
        switch (kind_) {
        case Kind::Forward:
            return "TinyMetalForward";
        case Kind::InputBackward:
            return "TinyMetalInputBackward";
        case Kind::ParameterBackward:
            return "TinyMetalParameterBackward";
        case Kind::InputJvp:
            return "TinyMetalInputJvp";
        case Kind::SecondInput:
            return "TinyMetalSecondInput";
        case Kind::SecondParams:
            return "TinyMetalSecondParams";
        case Kind::Initialize:
            return "TinyMetalInitialize";
        }
        return "TinyMetalUnknown";
    }
    void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
        throw std::runtime_error("tinymetalnn_mlx primitives require native Metal execution; no "
                                 "CPU fallback is selected");
    }
    void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
    std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override {
        require(shapes(inputs) == expected_,
                "shape-changing shapeless compilation is not implemented for this binding; use "
                "shape-specialized mx.compile");
        check_inputs(kind_, options_, inputs);
        return {result_shape(kind_, options_, inputs)};
    }
    bool is_equivalent(const mx::Primitive& other) const override {
        const auto* native = dynamic_cast<const NativePrimitive*>(&other);
        return native && kind_ == native->kind_ && options_ == native->options_ &&
               seed_ == native->seed_ && expected_ == native->expected_;
    }
    std::vector<mx::array> vjp(const std::vector<mx::array>& primals,
                               const std::vector<mx::array>& cotangents,
                               const std::vector<int>& argnums,
                               const std::vector<mx::array>& outputs) override;
    std::vector<mx::array> jvp(const std::vector<mx::array>& primals,
                               const std::vector<mx::array>& tangents,
                               const std::vector<int>& argnums) override;
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(const std::vector<mx::array>& inputs,
                                                             const std::vector<int>& axes) override;

  private:
    Kind kind_;
    Options options_;
    std::vector<mx::Shape> expected_;
    size_t seed_;
};

mx::array operation(Kind kind, const Options& options, std::vector<mx::array> inputs,
                    mx::Stream stream, size_t seed) {
    check_inputs(kind, options, inputs);
    for (auto& input : inputs)
        input = mx::contiguous(input, false, stream);
    return mx::array(result_shape(kind, options, inputs), result_dtype(kind, options),
                     std::make_shared<NativePrimitive>(stream, kind, options, inputs, seed),
                     inputs);
}

void NativePrimitive::eval_gpu(const std::vector<mx::array>& inputs,
                               std::vector<mx::array>& outputs) {
    check_inputs(kind_, options_, inputs);
    std::lock_guard<std::mutex> lock{options_.state->mutex};
    execute_mlx(stream(), inputs, outputs, [&](MLXSession& session) {
        auto& native = *options_.state->native;
        if (kind_ == Kind::Initialize) {
            if (!native.n_params())
                return;
            const auto bytes = metalrt::checked_mul(native.n_params(), sizeof(float));
            // Fresh registered staging is safe for the core's PCG host writes;
            // its lattice initializer remains on the native default GPU queue.
            auto initialized = session.device.allocate(bytes);
            std::memset(initialized.contents(), 0, bytes);
            native.initialize_params(seed_, initialized.data<float>());
            metalrt::ordered_default_stream(session.device).synchronize();
            session.queue->copy(initialized.view(), session.outputs[0].view());
            return;
        }
        native.set_jit_fusion(options_.jit);
        const auto verify_jit = [&]() {
            if (options_.jit && !native.jit_fusion())
                throw std::runtime_error("tinymetalnn_mlx: requested native JIT execution failed; "
                                         "AOT fallback is not accepted for this graph");
        };
        const auto batch = static_cast<uint32_t>(inputs[0].shape(0));
        auto& destination = session.outputs[0];
        if (kind_ != Kind::Forward && destination.view().length)
            session.queue->fill(destination.view(), 0);
        if (!batch || !native.n_output_dims() ||
            (kind_ == Kind::SecondInput && options_.no_second_input_grad))
            return;
        auto* x = session.inputs[0].data_as<float>();
        auto* p = session.inputs[1].data();
        auto stream_handle = session.nn_stream->get();
        if (kind_ == Kind::Forward) {
            native.inference(stream_handle, batch, x, destination.data(), p);
            verify_jit();
            return;
        }
        const auto output_bytes = metalrt::checked_mul(
            metalrt::checked_mul(static_cast<size_t>(batch), native.n_output_dims()),
            native.output_precision() == tcnn::cpp::Precision::Fp32 ? sizeof(float)
                                                                    : sizeof(uint16_t));
        auto y = session.device.allocate(output_bytes);
        // Context belongs to this evaluation. It never escapes into a mutable
        // Primitive cache, so compiled replay and lazy parameter replacement
        // always differentiate the correct graph inputs.
        auto context = native.forward(stream_handle, batch, x, y.contents(), p, true);
        verify_jit();
        if (kind_ == Kind::InputBackward || kind_ == Kind::ParameterBackward) {
            native.backward(stream_handle, context, batch,
                            kind_ == Kind::InputBackward ? destination.data_as<float>() : nullptr,
                            session.inputs[2].data(),
                            kind_ == Kind::ParameterBackward ? destination.data() : nullptr, x,
                            y.contents(), p);
            verify_jit();
            return;
        }
        metalrt::Buffer zero_adjoint;
        const void* dy;
        const float* v;
        if (kind_ == Kind::InputJvp) {
            zero_adjoint = session.device.allocate(output_bytes);
            session.queue->fill(zero_adjoint.view(), 0);
            dy = zero_adjoint.contents();
            v = session.inputs[2].data_as<float>();
        } else {
            dy = session.inputs[2].data();
            v = session.inputs[3].data_as<float>();
        }
        native.backward_backward_input(
            stream_handle, context, batch, v, x, dy,
            kind_ == Kind::SecondParams ? destination.data() : nullptr,
            kind_ == Kind::InputJvp ? destination.data() : nullptr,
            kind_ == Kind::SecondInput ? destination.data_as<float>() : nullptr, p);
        verify_jit();
    });
}

std::vector<mx::array> NativePrimitive::vjp(const std::vector<mx::array>& a,
                                            const std::vector<mx::array>& cotangents,
                                            const std::vector<int>& argnums,
                                            const std::vector<mx::array>&) {
    if (kind_ == Kind::Initialize)
        return {};
    if (kind_ == Kind::SecondInput || kind_ == Kind::SecondParams)
        throw std::runtime_error(
            "tinymetalnn_mlx: third derivatives are not exposed by the native facade");
    const auto s = stream();
    const auto c = tangent_cast(cotangents.at(0), result_dtype(kind_, options_), s);
    std::vector<mx::array> result;
    for (int arg : argnums) {
        if (kind_ == Kind::Forward) {
            require(arg == 0 || arg == 1, "invalid forward differentiation argument");
            result.push_back(scaled_backward(
                arg == 0 ? Kind::InputBackward : Kind::ParameterBackward, options_, a, c, s));
        } else if (kind_ == Kind::InputBackward) {
            if (arg == 0)
                result.push_back(operation(Kind::SecondInput, options_, {a[0], a[1], a[2], c}, s));
            else if (arg == 1)
                result.push_back(operation(Kind::SecondParams, options_, {a[0], a[1], a[2], c}, s));
            else if (arg == 2)
                result.push_back(operation(Kind::InputJvp, options_, {a[0], a[1], c}, s));
            else
                throw std::invalid_argument("Invalid input-backward argument");
        } else if (kind_ == Kind::ParameterBackward) {
            if (arg == 0)
                result.push_back(mixed_parameter_direction(options_, a[0], a[1], a[2], c, s));
            else if (arg == 2)
                result.push_back(parameter_jvp(options_, a[0], a[1], c, s));
            else if (arg == 1)
                throw std::runtime_error(
                    "tinymetalnn_mlx: parameter Hessians are not exposed by the native facade");
            else
                throw std::invalid_argument("Invalid parameter-backward argument");
        } else if (kind_ == Kind::InputJvp) {
            if (arg == 0)
                result.push_back(operation(Kind::SecondInput, options_, {a[0], a[1], c, a[2]}, s));
            else if (arg == 1)
                result.push_back(operation(Kind::SecondParams, options_, {a[0], a[1], c, a[2]}, s));
            else if (arg == 2)
                result.push_back(operation(Kind::InputBackward, options_, {a[0], a[1], c}, s));
            else
                throw std::invalid_argument("Invalid input-JVP argument");
        }
    }
    return result;
}

std::vector<mx::array> NativePrimitive::jvp(const std::vector<mx::array>& a,
                                            const std::vector<mx::array>& tangents,
                                            const std::vector<int>& argnums) {
    const auto s = stream();
    if (kind_ == Kind::Initialize)
        return {mx::zeros(result_shape(kind_, options_, a), mx::float32, s)};
    if (kind_ == Kind::SecondInput || kind_ == Kind::SecondParams)
        throw std::runtime_error(
            "tinymetalnn_mlx: third derivatives are not exposed by the native facade");
    std::vector<mx::array> terms;
    for (size_t i = 0; i < argnums.size(); ++i) {
        const int arg = argnums[i];
        const auto t = tangent_cast(tangents[i], a.at(arg).dtype(), s);
        if (kind_ == Kind::Forward) {
            require(arg == 0 || arg == 1, "invalid forward JVP argument");
            terms.push_back(arg == 0 ? operation(Kind::InputJvp, options_, {a[0], a[1], t}, s)
                                     : parameter_jvp(options_, a[0], a[1], t, s));
        } else if (kind_ == Kind::InputBackward) {
            if (arg == 0)
                terms.push_back(operation(Kind::SecondInput, options_, {a[0], a[1], a[2], t}, s));
            else if (arg == 1)
                terms.push_back(mixed_parameter_direction(options_, a[0], a[1], a[2], t, s));
            else if (arg == 2)
                terms.push_back(operation(Kind::InputBackward, options_, {a[0], a[1], t}, s));
        } else if (kind_ == Kind::ParameterBackward) {
            if (arg == 0)
                terms.push_back(operation(Kind::SecondParams, options_, {a[0], a[1], a[2], t}, s));
            else if (arg == 2)
                terms.push_back(operation(Kind::ParameterBackward, options_, {a[0], a[1], t}, s));
            else if (arg == 1)
                throw std::runtime_error(
                    "tinymetalnn_mlx: parameter Hessians are not exposed by the native facade");
        } else if (kind_ == Kind::InputJvp) {
            if (arg == 2) {
                terms.push_back(operation(Kind::InputJvp, options_, {a[0], a[1], t}, s));
                continue;
            }
            require(arg == 0 || arg == 1, "invalid input-JVP tangent argument");
            const auto shape = result_shape(kind_, options_, a);
            const auto type = result_dtype(kind_, options_);
            auto guarded = guard_shapes({a[0], a[1], a[2], t}, s);
            const size_t count =
                metalrt::checked_mul(static_cast<size_t>(shape[0]), static_cast<size_t>(shape[1]));
            std::vector<mx::array> values;
            values.reserve(count);
            for (size_t j = 0; j < count; ++j) {
                auto c = basis(shape, type, j, s);
                auto second = operation(arg == 0 ? Kind::SecondInput : Kind::SecondParams, options_,
                                        {guarded[0], guarded[1], c, guarded[2]}, s);
                values.push_back(dot(second, guarded[3], s));
            }
            terms.push_back(with_shape_dependencies(stack_values(std::move(values), shape, type, s),
                                                    guarded, s));
        }
    }
    if (terms.empty())
        return {with_shape_dependencies(
            mx::zeros(result_shape(kind_, options_, a), result_dtype(kind_, options_), s),
            guard_shapes(a, s), s)};
    auto result = terms[0];
    for (size_t i = 1; i < terms.size(); ++i)
        result = mx::add(result, terms[i], s);
    return {result};
}

std::pair<std::vector<mx::array>, std::vector<int>>
NativePrimitive::vmap(const std::vector<mx::array>& inputs, const std::vector<int>& axes) {
    require(inputs.size() == axes.size(), "invalid vmap axis count");
    auto guarded = guard_shapes(inputs, stream());
    int count = -1;
    for (size_t i = 0; i < axes.size(); ++i) {
        if (axes[i] == -1)
            continue;
        require(axes[i] >= 0 && static_cast<size_t>(axes[i]) < inputs[i].ndim(),
                "invalid mapped axis");
        if (count < 0)
            count = inputs[i].shape(axes[i]);
        else
            require(count == inputs[i].shape(axes[i]), "mapped axes must have equal lengths");
    }
    if (count < 0)
        return {{operation(kind_, options_, inputs, stream(), seed_)}, {-1}};
    if (!count) {
        // Shape-only descriptors remove each mapped dimension without trying
        // to select an element from an empty axis.
        std::vector<mx::array> samples;
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto shape = inputs[i].shape();
            if (axes[i] >= 0)
                shape.erase(shape.begin() + axes[i]);
            samples.push_back(mx::zeros(shape, inputs[i].dtype(), stream()));
        }
        check_inputs(kind_, options_, samples);
        auto shape = result_shape(kind_, options_, samples);
        shape.insert(shape.begin(), 0);
        auto empty = mx::zeros(shape, result_dtype(kind_, options_), stream());
        return {{with_shape_dependencies(empty, guarded, stream())}, {0}};
    }
    std::vector<std::vector<mx::array>> slices;
    slices.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i)
        slices.push_back(axes[i] < 0 ? std::vector<mx::array>{}
                                     : mx::unstack(guarded[i], axes[i], stream()));
    std::vector<mx::array> outputs;
    outputs.reserve(count);
    for (int index = 0; index < count; ++index) {
        std::vector<mx::array> row;
        for (size_t j = 0; j < inputs.size(); ++j)
            row.push_back(axes[j] < 0 ? guarded[j] : slices[j][index]);
        outputs.push_back(operation(kind_, options_, std::move(row), stream(), seed_));
    }
    return {{mx::stack(outputs, 0, stream())}, {0}};
}

std::mutex log_mutex;
std::vector<LogRecord> log_records;
void record_log(tcnn::cpp::LogSeverity severity, const std::string& message) {
    std::lock_guard<std::mutex> lock{log_mutex};
    log_records.emplace_back(severity, message);
}

} // namespace

ModuleState::ModuleState(std::unique_ptr<tcnn::cpp::Module> module)
    : native{std::move(module)},
      loss_scale{tcnn::cpp::default_loss_scale(native->param_precision())} {}
Module::Module(std::unique_ptr<tcnn::cpp::Module> module)
    : state_{std::make_shared<ModuleState>(std::move(module))} {}

mx::array Module::forward(const mx::array& input, const mx::array& params,
                          mx::StreamOrDevice stream) {
    return operation(Kind::Forward, snapshot(state_), {input, params}, gpu_stream(stream));
}
mx::array Module::initial_params(size_t seed, mx::StreamOrDevice stream) {
    return operation(Kind::Initialize, snapshot(state_), {}, gpu_stream(stream), seed);
}
uint32_t Module::n_input_dims() const {
    return state_->native->n_input_dims();
}
uint32_t Module::n_output_dims() const {
    return state_->native->n_output_dims();
}
size_t Module::n_params() const {
    return state_->native->n_params();
}
tcnn::cpp::Precision Module::param_precision() const {
    return state_->native->param_precision();
}
tcnn::cpp::Precision Module::output_precision() const {
    return state_->native->output_precision();
}
tcnn::cpp::json Module::hyperparams() const {
    std::lock_guard<std::mutex> lock{state_->mutex};
    return state_->native->hyperparams();
}
std::string Module::name() const {
    std::lock_guard<std::mutex> lock{state_->mutex};
    return state_->native->name();
}
bool Module::jit_fusion() const {
    std::lock_guard<std::mutex> lock{state_->mutex};
    return state_->jit;
}
void Module::set_jit_fusion(bool enabled) {
    std::lock_guard<std::mutex> lock{state_->mutex};
    with_mlx_device(gpu_stream(),
                    [&](metalrt::Device) { state_->native->set_jit_fusion(enabled); });
    state_->jit = state_->native->jit_fusion();
}
bool Module::backward_backward_input_no_input_grad() const {
    std::lock_guard<std::mutex> lock{state_->mutex};
    return state_->no_second_input_grad;
}
void Module::set_backward_backward_input_no_input_grad(bool enabled) {
    std::lock_guard<std::mutex> lock{state_->mutex};
    state_->no_second_input_grad = enabled;
}
float Module::loss_scale() const {
    std::lock_guard<std::mutex> lock{state_->mutex};
    return state_->loss_scale;
}
void Module::set_loss_scale(float value) {
    require(std::isfinite(value) && value > 0, "loss_scale must be finite and positive");
    std::lock_guard<std::mutex> lock{state_->mutex};
    state_->loss_scale = value;
}

std::shared_ptr<Module> create_encoding(uint32_t inputs, const tcnn::cpp::json& encoding,
                                        tcnn::cpp::Precision precision) {
    std::unique_ptr<tcnn::cpp::Module> native;
    with_mlx_device(gpu_stream(), [&](metalrt::Device) {
        native.reset(tcnn::cpp::create_encoding(inputs, encoding, precision));
    });
    return std::make_shared<Module>(std::move(native));
}
std::shared_ptr<Module> create_network(uint32_t inputs, uint32_t outputs,
                                       const tcnn::cpp::json& network) {
    std::unique_ptr<tcnn::cpp::Module> native;
    with_mlx_device(gpu_stream(), [&](metalrt::Device) {
        native.reset(tcnn::cpp::create_network(inputs, outputs, network));
    });
    return std::make_shared<Module>(std::move(native));
}
std::shared_ptr<Module> create_network_with_input_encoding(uint32_t inputs, uint32_t outputs,
                                                           const tcnn::cpp::json& encoding,
                                                           const tcnn::cpp::json& network) {
    std::unique_ptr<tcnn::cpp::Module> native;
    with_mlx_device(gpu_stream(), [&](metalrt::Device) {
        native.reset(
            tcnn::cpp::create_network_with_input_encoding(inputs, outputs, encoding, network));
    });
    return std::make_shared<Module>(std::move(native));
}
bool supports_jit_fusion(int device) {
    bool supported = false;
    with_mlx_device(gpu_stream(),
                    [&](metalrt::Device) { supported = tcnn::cpp::supports_jit_fusion(device); });
    return supported;
}
std::pair<mx::array, mx::array> guard_inputs(const mx::array& input, const mx::array& params,
                                             mx::StreamOrDevice stream) {
    auto result = guard_shapes({input, params}, gpu_stream(stream));
    return {result[0], result[1]};
}
void capture_logs() {
    tcnn::cpp::set_log_callback(record_log);
}
std::vector<LogRecord> take_logs() {
    std::lock_guard<std::mutex> lock{log_mutex};
    std::vector<LogRecord> result;
    result.swap(log_records);
    return result;
}

} // namespace mlx_bindings
} // namespace tinymetalnn
