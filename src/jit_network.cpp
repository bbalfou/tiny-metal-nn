#include <tiny-metal-nn/network.h>
#include <algorithm>
#include <sstream>

namespace tcnn {
namespace detail {
namespace {

struct Layer {
    uint32_t inputs, outputs, activation;
    size_t weights, input, pre, output;
};

std::vector<Layer> layers(uint32_t input_width, uint32_t width, uint32_t padded_output_width,
                          uint32_t output_width, uint32_t n_hidden, Activation activation,
                          Activation output_activation) {
    nn_check(input_width && output_width && padded_output_width >= output_width,
             "invalid generated MLP input/output dimensions");
    nn_check(!n_hidden || width, "generated MLP hidden width must be positive");
    nn_check(uint32_t(activation) <= uint32_t(Activation::None) &&
                 uint32_t(output_activation) <= uint32_t(Activation::None),
             "invalid generated MLP activation");
    // Bound both source generation and all address arithmetic before allocating.
    mlp_jit_context_bytes(sizeof(float), input_width, width, padded_output_width, n_hidden);
    size_t parameter_count = 0;
    std::vector<Layer> result;
    uint32_t previous = input_width;
    size_t input = 0, context = input_width;
    for (uint64_t i = 0; i <= n_hidden; ++i) {
        const uint32_t next = i == n_hidden ? padded_output_width : width;
        const auto next_context = metalrt::checked_add(context, metalrt::checked_mul(2u, next));
        result.push_back({previous, next, uint32_t(i == n_hidden ? output_activation : activation),
                          parameter_count, input, context, context + next});
        parameter_count =
            metalrt::checked_add(parameter_count, metalrt::checked_mul(previous, next));
        nn_check(parameter_count <= UINT32_MAX,
                 "generated MLP parameter indexing exceeds uint32_t");
        input = context + next;
        context = next_context;
        previous = next;
    }
    return result;
}

void validate_type(const std::string& type) {
    nn_check(type == "float" || type == "half",
             "generated MLP supports native float and half storage");
}

// The generated scalar ABI remains one sample per lane. These helpers change
// only its execution: complete native SIMD-groups share one sample/feature
// plane and use 8x8 matrix instructions for both products and reductions.
uint32_t cooperative_stride(uint32_t max_width) {
    const auto result = metalrt::checked_add(size_t(max_width), 7u) & ~size_t{7};
    nn_check(result <= UINT32_MAX, "generated MLP cooperative stride exceeds uint32_t");
    return static_cast<uint32_t>(result);
}

void emit_cooperative_begin(std::ostringstream& code, const std::string& type, uint32_t max_width,
                            uint32_t derivative_order) {
    const uint32_t stride = cooperative_stride(max_width);
    const auto bytes = mlp_jit_simdgroup_scratch_bytes(type == "half" ? 2u : 4u, max_width, 0, 0);
    const char* environment = derivative_order ? "env" : "TCNN_JIT_MLP_CALLABLE_ENV";
    code << "// TCNN_JIT_SIMDGROUP_SCRATCH_BYTES: " << bytes << "\n"
         << "#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 230 && "
            "TCNN_JIT_HAS_SIMDGROUP_MATRIX";
    if (!derivative_order)
        code << " && defined(TCNN_JIT_MLP_CALLABLE_ENV)";
    code << "\n    if (tcnn_jit_mlp::available(" << environment << ", " << bytes << "u)";
    if (derivative_order)
        code << " && (simd_all(dL_dparams != nullptr) || simd_all(dL_dparams == nullptr))"
             << " && (simd_all(dL_dx != nullptr) || simd_all(dL_dx == nullptr))";
    code << ") {\n"
         << "        using Mma = tcnn_jit_mlp::NativeAtom<" << type << ">;\n"
         << "        const uint lane = " << environment << "->lane_index;\n"
         << "        auto state = reinterpret_cast<threadgroup " << type << "*>(" << environment
         << "->simd_shared_memory);\n"
         << "        auto panel = state + " << size_t(32) * stride << "u;\n";
    if (derivative_order)
        code << "        auto sums = reinterpret_cast<threadgroup float*>(panel + 256u);\n";
    code << "        auto saved = reinterpret_cast<" << (derivative_order ? "const " : "")
         << "TCNN_JIT_CTX " << type << "*>(fwd_ctx);\n";
}

void emit_cooperative_end(std::ostringstream& code) {
    code << "    }\n#endif\n";
}

void emit_product(std::ostringstream& code, const Layer& layer, uint32_t stride, bool backward) {
    code << "        tcnn_jit_mlp::Product<Mma, " << (backward ? layer.outputs : layer.inputs)
         << "u, " << (backward ? layer.inputs : layer.outputs) << "u, " << stride << "u, "
         << (backward ? "true" : "false") << ">::run(state, params + " << layer.weights
         << "u, panel, lane);\n";
}

void emit_gradient_panels(std::ostringstream& code, const std::string& type, const Layer& layer,
                          uint32_t stride, const char* source, const std::string& expression = {}) {
    const std::string value =
        expression.empty()
            ? std::string{source} + "[" + std::to_string(layer.input) + "u + first + k]"
            : expression;
    // Keep panel loads, matrix reduction and global publication in one bounded
    // iteration rather than expanding every input panel into the caller.
    code << "        if (dL_dparams) {\n"
         << "#pragma nounroll\n"
         << "            for (uint first = 0; first < " << layer.inputs << "u; first += 8u) {\n"
         << "                for (uint k = 0; k < 8u; ++k)\n"
         << "                    panel[k * 32u + lane] = first + k < " << layer.inputs << "u ? "
         << value << " : " << type << "(0);\n"
         << "                Mma::barrier();\n"
         << "                tcnn_jit_mlp::GradientPanel<Mma, " << layer.outputs << "u, "
         << layer.inputs << "u, " << stride << "u>::run(state, panel, sums, dL_dparams + "
         << layer.weights << "u, first, lane, env);\n"
         << "            }\n"
         << "        }\n";
}

// Keep the exact rounded values consumed by mlp.metal. Only Sine and SiLU
// derivatives/curvature use preactivation; all other activations use output.
bool output_suffices_for_derivatives(Activation activation) {
    return activation != Activation::Sine && activation != Activation::SiLU;
}

void emit_cooperative_forward(std::ostringstream& code, const std::string& type,
                              const std::vector<Layer>& shape, uint32_t input_width,
                              uint32_t output_width, uint32_t max_width) {
    if (!mlp_jit_simdgroup_scratch_bytes(type == "half" ? 2u : 4u, max_width, 0, 0))
        return;
    const auto stride = cooperative_stride(max_width);
    emit_cooperative_begin(code, type, max_width, 0);
    code << "        for (uint k = 0; k < " << stride << "u; ++k)\n"
         << "            state[k * 32u + lane] = k < " << input_width << "u ? input[k] : " << type
         << "(0);\n"
         << "        if (saved) for (uint k = 0; k < " << input_width
         << "u; ++k) saved[k] = input[k];\n"
         << "        Mma::barrier();\n";
    for (const auto& layer : shape) {
        emit_product(code, layer, stride, false);
        code << "#pragma nounroll\n"
             << "        for (uint row = 0; row < " << layer.outputs << "u; ++row) {\n"
             << "            const " << type << " pre = state[row * 32u + lane];\n"
             << "            const " << type << " value = tcnn_jit_mlp_activate(pre, "
             << layer.activation << "u);\n"
             << "            state[row * 32u + lane] = value;\n"
             << "            if (saved) {\n";
        if (layer.pre != layer.output)
            code << "                saved[" << layer.pre << "u + row] = pre;\n";
        code << "                saved[" << layer.output << "u + row] = value;\n            }\n"
             << "        }\n"
             << "        Mma::barrier();\n";
    }
    code << "        metal::array<" << type << ", " << output_width << "> result{};\n"
         << "        for (uint row = 0; row < " << output_width
         << "u; ++row) result[row] = state[row * 32u + lane];\n"
         << "        Mma::barrier();\n"
         << "        return result;\n";
    emit_cooperative_end(code);
}

void emit_cooperative_backward(std::ostringstream& code, const std::string& type,
                               const std::vector<Layer>& shape, uint32_t output_width,
                               uint32_t max_width) {
    if (!mlp_jit_simdgroup_scratch_bytes(type == "half" ? 2u : 4u, max_width, 0, 0))
        return;
    const auto stride = cooperative_stride(max_width);
    emit_cooperative_begin(code, type, max_width, 1);
    code << "#pragma nounroll\n"
         << "        for (uint row = 0; row < " << stride << "u; ++row)\n"
         << "            state[row * 32u + lane] = row < " << output_width
         << "u ? dL_dy[row] : " << type << "(0);\n"
         << "        Mma::barrier();\n";
    for (size_t i = shape.size(); i-- > 0;) {
        const auto& layer = shape[i];
        code << "#pragma nounroll\n"
             << "        for (uint row = 0; row < " << layer.outputs << "u; ++row) {\n"
             << "            const " << type << " first = tcnn_jit_mlp_derivative(saved["
             << layer.pre << "u + row], saved[" << layer.output << "u + row], " << layer.activation
             << "u);\n"
             << "            state[row * 32u + lane] = " << type
             << "(float(state[row * 32u + lane]) * float(first));\n"
             << "        }\n"
             << "        Mma::barrier();\n";
        emit_gradient_panels(code, type, layer, stride, "saved");
        if (i) {
            emit_product(code, layer, stride, true);
        } else {
            code << "        if (dL_dx) {\n";
            emit_product(code, layer, stride, true);
            code << "            for (uint k = 0; k < " << layer.inputs
                 << "u; ++k) (*dL_dx)[k] = state[k * 32u + lane];\n"
                 << "        }\n";
        }
    }
    code << "        Mma::barrier();\n        return;\n";
    emit_cooperative_end(code);
}

void emit_cooperative_second(std::ostringstream& code, const std::string& type,
                             const std::vector<Layer>& shape, uint32_t input_width,
                             uint32_t output_width, uint32_t max_width) {
    if (!mlp_jit_simdgroup_scratch_bytes(type == "half" ? 2u : 4u, max_width, 0, 0))
        return;
    const auto stride = cooperative_stride(max_width);
    // Only affine tangents must remain live. Activation tangents can be
    // reproduced with the exact stored-T derivative and rounding when dW needs
    // them. Keeping both copies would nearly double the private tangent state.
    std::vector<size_t> tangent_offsets;
    size_t tangent_elements = input_width;
    for (const auto& layer : shape) {
        tangent_offsets.push_back(tangent_elements);
        tangent_elements += layer.outputs;
    }
    emit_cooperative_begin(code, type, max_width, 2);
    code << "        " << type << " tangent[" << tangent_elements << "] = {};\n"
         << "        for (uint k = 0; k < " << stride << "u; ++k)\n"
         << "            state[k * 32u + lane] = k < " << input_width
         << "u ? dL_ddLdx[k] : " << type << "(0);\n"
         << "        for (uint k = 0; k < " << input_width << "u; ++k) tangent[k] = dL_ddLdx[k];\n"
         << "        Mma::barrier();\n";
    for (size_t i = 0; i < shape.size(); ++i) {
        const auto& layer = shape[i];
        emit_product(code, layer, stride, false);
        code << "#pragma nounroll\n"
             << "        for (uint row = 0; row < " << layer.outputs << "u; ++row) {\n"
             << "            const " << type << " first = tcnn_jit_mlp_derivative(saved["
             << layer.pre << "u + row], saved[" << layer.output << "u + row], " << layer.activation
             << "u);\n"
             << "            tangent[" << tangent_offsets[i]
             << "u + row] = state[row * 32u + lane];\n"
             << "            state[row * 32u + lane] = " << type
             << "(float(state[row * 32u + lane]) * float(first));\n"
             << "        }\n"
             << "        Mma::barrier();\n";
    }
    code << "        if (dL_ddLdy) for (uint row = 0; row < " << output_width
         << "u; ++row) (*dL_ddLdy)[row] = state[row * 32u + lane];\n"
         << "        Mma::barrier();\n"
         << "        if (!dL_dparams && !dL_dx) return;\n"
         << "        " << type << " adjoint[" << max_width << "] = {};\n"
         << "        " << type << " tangent_adjoint[" << max_width << "] = {};\n"
         << "        " << type << " tangent_delta[" << max_width << "] = {};\n"
         << "        for (uint row = 0; row < " << output_width
         << "u; ++row) tangent_adjoint[row] = dL_dy[row];\n";
    for (size_t i = shape.size(); i-- > 0;) {
        const auto& layer = shape[i];
        code << "#pragma nounroll\n"
             << "        for (uint row = 0; row < " << layer.outputs << "u; ++row) {\n"
             << "            const " << type << " first = tcnn_jit_mlp_derivative(saved["
             << layer.pre << "u + row], saved[" << layer.output << "u + row], " << layer.activation
             << "u);\n"
             << "            const " << type << " second = tcnn_jit_mlp_curvature(saved["
             << layer.pre << "u + row], saved[" << layer.output << "u + row], " << layer.activation
             << "u);\n"
             << "            state[row * 32u + lane] = " << type
             << "(float(adjoint[row]) * float(first) + float(tangent_adjoint[row]) * "
             << "float(second) * float(tangent[" << tangent_offsets[i] << "u + row]));\n"
             << "            tangent_delta[row] = " << type
             << "(float(tangent_adjoint[row]) * float(first));\n"
             << "        }\n"
             << "        for (uint row = " << layer.outputs << "u; row < "
             << cooperative_stride(layer.outputs) << "u; ++row)\n"
             << "            state[row * 32u + lane] = " << type << "(0);\n"
             << "        Mma::barrier();\n";
        emit_gradient_panels(code, type, layer, stride, "saved");
        code << "        if (" << (i ? "true" : "dL_dx != nullptr") << ") {\n";
        emit_product(code, layer, stride, true);
        code << "            for (uint k = 0; k < " << layer.inputs << "u; ++k) "
             << (i ? "adjoint[k]" : "(*dL_dx)[k]") << " = state[k * 32u + lane];\n"
             << "            Mma::barrier();\n"
             << "        }\n"
             << "        for (uint row = 0; row < " << cooperative_stride(layer.outputs)
             << "u; ++row)\n"
             << "            state[row * 32u + lane] = row < " << layer.outputs
             << "u ? tangent_delta[row] : " << type << "(0);\n"
             << "        Mma::barrier();\n";
        std::ostringstream tangent_input;
        if (i) {
            const auto& previous = shape[i - 1];
            tangent_input << type << "(float(tangent[" << tangent_offsets[i - 1]
                          << "u + first + k]) * "
                          << "float(tcnn_jit_mlp_derivative(saved[" << previous.pre
                          << "u + first + k], saved[" << previous.output << "u + first + k], "
                          << previous.activation << "u)))";
        } else {
            tangent_input << "tangent[first + k]";
        }
        emit_gradient_panels(code, type, layer, stride, "tangent", tangent_input.str());
        if (i) {
            emit_product(code, layer, stride, true);
            code << "        for (uint k = 0; k < " << layer.inputs
                 << "u; ++k) tangent_adjoint[k] = state[k * 32u + lane];\n"
                 << "        Mma::barrier();\n";
        }
    }
    code << "        Mma::barrier();\n        return;\n";
    emit_cooperative_end(code);
}

// All loops have configuration constants; Metal's compiler can specialize them
// without changing the public device-function or parameter-storage contract.
void emit_forward(std::ostringstream& code, const std::string& type,
                  const std::vector<Layer>& shape, uint32_t input_width, uint32_t output_width,
                  uint32_t max_width) {
    emit_cooperative_forward(code, type, shape, input_width, output_width, max_width);
    code << "    auto saved = reinterpret_cast<TCNN_JIT_CTX " << type << "*>(fwd_ctx);\n"
         << "    " << type << " current[" << max_width << "] = {};\n"
         << "    " << type << " next[" << max_width << "] = {};\n"
         << "    for (uint k = 0; k < " << input_width << "u; ++k) {\n"
         << "        current[k] = input[k];\n"
         << "        if (saved) saved[k] = input[k];\n"
         << "    }\n";
    for (const auto& layer : shape) {
        code << "    {\n"
             << "        for (uint row = 0; row < " << layer.outputs << "u; ++row) {\n"
             << "            float sum = 0.0f;\n"
             << "            for (uint k = 0; k < " << layer.inputs << "u; ++k)\n"
             << "                sum += float(params[" << layer.weights << "u + row * "
             << layer.inputs << "u + k]) * float(current[k]);\n"
             << "            const " << type << " pre = " << type << "(sum);\n"
             << "            next[row] = tcnn_jit_mlp_activate(pre, " << layer.activation << "u);\n"
             << "            if (saved) {\n";
        if (layer.pre != layer.output)
            code << "                saved[" << layer.pre << "u + row] = pre;\n";
        code << "                saved[" << layer.output << "u + row] = next[row];\n"
             << "            }\n"
             << "        }\n"
             << "        for (uint row = 0; row < " << layer.outputs
             << "u; ++row) current[row] = next[row];\n"
             << "    }\n";
    }
    code << "    metal::array<" << type << ", " << output_width << "> result{};\n"
         << "    for (uint row = 0; row < " << output_width
         << "u; ++row) result[row] = current[row];\n"
         << "    return result;\n";
}

void emit_backward(std::ostringstream& code, const std::string& type,
                   const std::vector<Layer>& shape, uint32_t output_width, uint32_t max_width) {
    emit_cooperative_backward(code, type, shape, output_width, max_width);
    code << "    const auto saved = reinterpret_cast<const TCNN_JIT_CTX " << type
         << "*>(fwd_ctx);\n"
         << "    " << type << " adjoint[" << max_width << "] = {};\n"
         << "    " << type << " delta[" << max_width << "] = {};\n"
         << "    " << type << " next[" << max_width << "] = {};\n"
         << "    for (uint row = 0; row < " << output_width
         << "u; ++row) adjoint[row] = dL_dy[row];\n";
    for (size_t i = shape.size(); i-- > 0;) {
        const auto& layer = shape[i];
        code << "    {\n"
             << "        for (uint row = 0; row < " << layer.outputs << "u; ++row) {\n"
             << "            const " << type << " first = tcnn_jit_mlp_derivative(saved["
             << layer.pre << "u + row], saved[" << layer.output << "u + row], " << layer.activation
             << "u);\n"
             << "            delta[row] = " << type << "(float(adjoint[row]) * float(first));\n"
             << "        }\n"
             << "        if (dL_dparams) {\n"
             << "            for (uint row = 0; row < " << layer.outputs << "u; ++row)\n"
             << "                for (uint k = 0; k < " << layer.inputs << "u; ++k)\n"
             << "                    tcnn_jit_atomic_add(dL_dparams, " << layer.weights
             << "u + row * " << layer.inputs << "u + k, float(delta[row]) * float(saved["
             << layer.input << "u + k]), env);\n"
             << "        }\n"
             << "        if (" << (i ? "true" : "dL_dx != nullptr") << ") {\n"
             << "            for (uint k = 0; k < " << layer.inputs << "u; ++k) {\n"
             << "                float sum = 0.0f;\n"
             << "                for (uint row = 0; row < " << layer.outputs << "u; ++row)\n"
             << "                    sum += float(params[" << layer.weights << "u + row * "
             << layer.inputs << "u + k]) * float(delta[row]);\n"
             << "                next[k] = " << type << "(sum);\n"
             << "            }\n";
        if (i)
            code << "            for (uint k = 0; k < " << layer.inputs
                 << "u; ++k) adjoint[k] = next[k];\n";
        else
            code << "            for (uint k = 0; k < " << layer.inputs
                 << "u; ++k) (*dL_dx)[k] = next[k];\n";
        code << "        }\n"
             << "    }\n";
    }
}

void emit_second_backward(std::ostringstream& code, const std::string& type,
                          const std::vector<Layer>& shape, uint32_t input_width,
                          uint32_t output_width, uint32_t max_width) {
    emit_cooperative_second(code, type, shape, input_width, output_width, max_width);
    // Tangents need BOTH affine and activated values, even when the saved
    // primal context aliases preactivation/output for output-based derivatives.
    // Keep their addresses independent of the packed saved-context offsets.
    auto tangent_shape = shape;
    size_t context_elements = input_width, previous = 0;
    for (auto& layer : tangent_shape) {
        layer.input = previous;
        layer.pre = context_elements;
        layer.output = context_elements + layer.outputs;
        previous = layer.output;
        context_elements += 2u * layer.outputs;
    }
    code << "    const auto saved = reinterpret_cast<const TCNN_JIT_CTX " << type
         << "*>(fwd_ctx);\n"
         << "    " << type << " tangent[" << context_elements << "] = {};\n"
         << "    for (uint k = 0; k < " << input_width << "u; ++k) tangent[k] = dL_ddLdx[k];\n";
    for (size_t i = 0; i < shape.size(); ++i) {
        const auto& layer = shape[i];
        const auto& direction = tangent_shape[i];
        code << "    {\n"
             << "        for (uint row = 0; row < " << layer.outputs << "u; ++row) {\n"
             << "            float sum = 0.0f;\n"
             << "            for (uint k = 0; k < " << layer.inputs << "u; ++k)\n"
             << "                sum += float(params[" << layer.weights << "u + row * "
             << layer.inputs << "u + k]) * float(tangent[" << direction.input << "u + k]);\n"
             << "            const " << type << " first = tcnn_jit_mlp_derivative(saved["
             << layer.pre << "u + row], saved[" << layer.output << "u + row], " << layer.activation
             << "u);\n"
             << "            tangent[" << direction.pre << "u + row] = " << type << "(sum);\n"
             << "            tangent[" << direction.output << "u + row] = " << type << "(float("
             << type << "(sum)) * float(first));\n"
             << "        }\n"
             << "    }\n";
    }
    code << "    if (dL_ddLdy)\n"
         << "        for (uint row = 0; row < " << output_width
         << "u; ++row) (*dL_ddLdy)[row] = tangent[" << tangent_shape.back().output << "u + row];\n"
         << "    " << type << " adjoint[" << max_width << "] = {};\n"
         << "    " << type << " tangent_adjoint[" << max_width << "] = {};\n"
         << "    " << type << " delta[" << max_width << "] = {};\n"
         << "    " << type << " tangent_delta[" << max_width << "] = {};\n"
         << "    " << type << " next[" << max_width << "] = {};\n"
         << "    " << type << " tangent_next[" << max_width << "] = {};\n"
         << "    for (uint row = 0; row < " << output_width
         << "u; ++row) tangent_adjoint[row] = dL_dy[row];\n";
    for (size_t i = shape.size(); i-- > 0;) {
        const auto& layer = shape[i];
        const auto& direction = tangent_shape[i];
        code << "    {\n"
             << "        for (uint row = 0; row < " << layer.outputs << "u; ++row) {\n"
             << "            const " << type << " first = tcnn_jit_mlp_derivative(saved["
             << layer.pre << "u + row], saved[" << layer.output << "u + row], " << layer.activation
             << "u);\n"
             << "            const " << type << " second = tcnn_jit_mlp_curvature(saved["
             << layer.pre << "u + row], saved[" << layer.output << "u + row], " << layer.activation
             << "u);\n"
             << "            delta[row] = " << type
             << "(float(adjoint[row]) * float(first) + float(tangent_adjoint[row]) * float(second) "
                "* float(tangent["
             << direction.pre << "u + row]));\n"
             << "            tangent_delta[row] = " << type
             << "(float(tangent_adjoint[row]) * float(first));\n"
             << "        }\n"
             << "        if (dL_dparams) {\n"
             << "            for (uint row = 0; row < " << layer.outputs << "u; ++row)\n"
             << "                for (uint k = 0; k < " << layer.inputs << "u; ++k) {\n"
             << "                    const float contribution = float(delta[row]) * float(saved["
             << layer.input << "u + k])"
             << " + float(tangent_delta[row]) * float(tangent[" << direction.input << "u + k]);\n"
             << "                    tcnn_jit_atomic_add(dL_dparams, " << layer.weights
             << "u + row * " << layer.inputs << "u + k, contribution, env);\n"
             << "                }\n"
             << "        }\n"
             << "        if (" << (i ? "true" : "dL_dx != nullptr") << ") {\n"
             << "            for (uint k = 0; k < " << layer.inputs << "u; ++k) {\n"
             << "                float sum = 0.0f, tangent_sum = 0.0f;\n"
             << "                for (uint row = 0; row < " << layer.outputs << "u; ++row) {\n"
             << "                    const float weight = float(params[" << layer.weights
             << "u + row * " << layer.inputs << "u + k]);\n"
             << "                    sum += weight * float(delta[row]);\n"
             << "                    tangent_sum += weight * float(tangent_delta[row]);\n"
             << "                }\n"
             << "                next[k] = " << type << "(sum);\n"
             << "                tangent_next[k] = " << type << "(tangent_sum);\n"
             << "            }\n";
        if (i)
            code << "            for (uint k = 0; k < " << layer.inputs
                 << "u; ++k) { adjoint[k] = next[k]; tangent_adjoint[k] = tangent_next[k]; }\n";
        else
            code << "            for (uint k = 0; k < " << layer.inputs
                 << "u; ++k) (*dL_dx)[k] = next[k];\n";
        code << "        }\n"
             << "    }\n";
    }
}
} // namespace

uint32_t mlp_jit_simdgroup_scratch_bytes(size_t element_size, uint32_t input_width, uint32_t width,
                                         uint32_t padded_output_width) {
    nn_check(element_size == 2 || element_size == 4,
             "generated MLP supports native float and half storage");
    // Workspace is an optimization, not an additional source-generation bound.
    // A valid scalar shape may need more than uint32_t bytes for 32 lanes. In
    // that case omit the collective while preserving the original scalar body.
    const uint64_t stride =
        (uint64_t(std::max({input_width, width, padded_output_width})) + 7u) & ~uint64_t{7};
    // Keep one FP32 8-column result panel per output tile. Publishing them
    // together avoids two SIMD barriers per tile in the gradient collective.
    const uint64_t bytes = (32u * stride + 256u) * element_size + 8u * stride * sizeof(float);
    const uint64_t aligned = (bytes + 15u) & ~uint64_t{15};
    if (stride > UINT32_MAX || aligned > UINT32_MAX)
        return 0;
    return static_cast<uint32_t>(aligned);
}

uint32_t mlp_jit_context_bytes(size_t element_size, uint32_t input_width, uint32_t width,
                               uint32_t padded_output_width, uint32_t n_hidden) {
    const size_t activations =
        metalrt::checked_add(metalrt::checked_mul(width, n_hidden), padded_output_width);
    const size_t elements =
        metalrt::checked_add(input_width, metalrt::checked_mul(2u, activations));
    const size_t bytes = metalrt::checked_mul(element_size, elements);
    const size_t aligned_bytes = metalrt::checked_add(bytes, 3u) & ~size_t{3};
    nn_check(aligned_bytes <= UINT32_MAX, "generated MLP forward context exceeds uint32_t");
    return static_cast<uint32_t>(aligned_bytes);
}

uint32_t mlp_jit_training_context_bytes(size_t element_size, uint32_t input_width, uint32_t width,
                                        uint32_t padded_output_width, uint32_t n_hidden,
                                        Activation activation, Activation output_activation) {
    // Validate the same dimensions and index bounds as the public layout.
    mlp_jit_context_bytes(element_size, input_width, width, padded_output_width, n_hidden);
    const auto hidden = metalrt::checked_mul(metalrt::checked_mul(width, n_hidden),
                                             output_suffices_for_derivatives(activation) ? 1u : 2u);
    const auto final = metalrt::checked_mul(
        padded_output_width, output_suffices_for_derivatives(output_activation) ? 1u : 2u);
    const auto elements = metalrt::checked_add(input_width, metalrt::checked_add(hidden, final));
    const auto bytes = (metalrt::checked_mul(elements, element_size) + 3u) & ~size_t{3};
    nn_check(bytes <= UINT32_MAX, "generated training context exceeds uint32_t");
    return static_cast<uint32_t>(bytes);
}

std::string generate_mlp_body(const std::string& type, uint32_t input_width, uint32_t width,
                              uint32_t padded_output_width, uint32_t output_width,
                              uint32_t n_hidden, Activation activation,
                              Activation output_activation, uint32_t derivative_order) {
    return generate_mlp_body(type, input_width, width, padded_output_width, output_width, n_hidden,
                             activation, output_activation, derivative_order, false);
}

std::string generate_mlp_body(const std::string& type, uint32_t input_width, uint32_t width,
                              uint32_t padded_output_width, uint32_t output_width,
                              uint32_t n_hidden, Activation activation,
                              Activation output_activation, uint32_t derivative_order,
                              bool compact_context) {
    validate_type(type);
    auto shape = layers(input_width, width, padded_output_width, output_width, n_hidden, activation,
                        output_activation);
    if (compact_context) {
        size_t cursor = input_width, previous = 0;
        for (auto& layer : shape) {
            layer.input = previous;
            layer.pre = cursor;
            layer.output = cursor + (output_suffices_for_derivatives(Activation(layer.activation))
                                         ? 0u
                                         : layer.outputs);
            previous = layer.output;
            cursor = layer.output + layer.outputs;
        }
    }
    const auto max_width = std::max({input_width, n_hidden ? width : 0u, padded_output_width});
    std::ostringstream code;
    code << "// TCNN_JIT_DENSE_GRADIENT_PARAMETERS: "
         << shape.back().weights + size_t(shape.back().inputs) * shape.back().outputs << "\n";
    switch (derivative_order) {
    case 0:
        emit_forward(code, type, shape, input_width, output_width, max_width);
        break;
    case 1:
        emit_backward(code, type, shape, output_width, max_width);
        break;
    case 2:
        emit_second_backward(code, type, shape, input_width, output_width, max_width);
        break;
    default:
        throw std::invalid_argument("unknown generated MLP derivative order");
    }
    return code.str();
}

} // namespace detail
} // namespace tcnn
