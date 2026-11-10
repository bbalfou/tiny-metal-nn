#include <tiny-metal-nn/encodings/internal/implementation.h>
#include <tiny-metal-nn/encodings/composite.h>
#include <tiny-metal-nn/encodings/multi_level_lod.h>
#include <tiny-metal-nn/detail/shaders.h>
#include <tiny-metal-nn/detail/jit.h>
#include <cstring>
#include <sstream>

namespace tcnn {
namespace encoding_detail {
namespace {
// Bit-exact literals also represent caller-supplied infinities and NaNs without
// depending on host locale or nonstandard MSL spelling of floating constants.
std::string literal(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return "as_type<float>(" + std::to_string(bits) + "u)";
}

std::string vector_type(const std::string& type, uint32_t size) {
    return "tcnn_jit_array<" + type + ", " + std::to_string(size) + ">";
}

std::string math_source() {
    // The ordinary function wrapper supplies jit_device_prelude. Put that
    // wrapper before helpers only at assembly time (see with_math below).
    return detail::shader_source("encodings/jit.metal");
}

// Body helpers return source with the shared prelude. Forward declarations are
// unnecessary when the math header precedes the function itself.
std::string with_math(const std::string& function) {
    return detail::jit_device_prelude() + math_source() + function;
}

std::string load_input(uint32_t dims) {
    std::ostringstream body;
    body << vector_type("float", dims) << " input{};\n"
         << "for (uint d = 0; d < " << dims
         << "u; ++d) input[d] = reinterpret_cast<const TCNN_JIT_CTX float*>(fwd_ctx)[d];\n";
    return body.str();
}

std::string store_input(uint32_t dims) {
    return "if (fwd_ctx) for (uint d = 0; d < " + std::to_string(dims) +
           "u; ++d) reinterpret_cast<TCNN_JIT_CTX float*>(fwd_ctx)[d] = input[d];\n";
}

std::string level_limit(float scalar, bool gpu) {
    const auto fallback = gpu ? "(env && env->runtime_inputs ? env->runtime_inputs[env->offset * "
                                "env->count + env->index] : " +
                                    literal(scalar) + ")"
                              : literal(scalar);
    return "(env && env->has_level_override ? env->level_override : " + fallback + ")";
}

std::string shifted_env(uint32_t offset, const std::string& variable) {
    std::ostringstream body;
    body << "tcnn_jit_environment " << variable << "{};\n"
         << "if (env) { " << variable << " = *env; " << variable << ".offset += " << offset
         << "u; }\n";
    return body.str();
}

std::string ptr_offset(const char* name, size_t offset) {
    return offset
               ? std::string(name) + " ? " + name + " + " + std::to_string(offset) + "u : nullptr"
               : name;
}

std::string child_context(uint32_t offset, bool nullable = true, bool grouped = false) {
    if (!offset)
        return "fwd_ctx";
    const auto advance = (grouped ? "(env && env->lane_count ? env->lane_count : 1u) * " : "") +
                         std::to_string(offset) + "u";
    return nullable ? "fwd_ctx ? fwd_ctx + " + advance + " : nullptr" : "fwd_ctx + " + advance;
}

std::string sample_tail(uint32_t offset, uint32_t bytes, bool grouped) {
    if (!grouped)
        return "fwd_ctx + " + std::to_string(offset) + "u";
    return "fwd_ctx + (env && env->lane_count ? env->lane_count : 1u) * " + std::to_string(offset) +
           "u + (env ? env->lane_index : 0u) * " + std::to_string(bytes) + "u";
}

std::string float_array(const std::vector<float>& table, uint32_t begin, uint32_t count) {
    std::ostringstream result;
    result << "{";
    for (uint32_t i = 0; i < count; ++i) {
        if (i)
            result << ", ";
        result << literal(table.at(begin + i));
    }
    result << "}";
    return result.str();
}
} // namespace

// Analytic encodings have a block-diagonal Jacobian except for SH. Emit
// that structure explicitly: one coordinate visits only its own features, and
// padded constants never enter derivative loops. Literal SH rows let Metal
// remove the polynomial switch and share its common powers across outputs.
namespace {
std::string analytic_call(uint32_t dims, uint32_t kind, uint32_t degree, uint32_t width,
                          uint32_t padded, float scale, float offset, const std::string& row,
                          const std::string& dimension, const std::string& direction = "") {
    return "tcnn_jit_encoding::analytic<" + std::to_string(dims) + ">(" + std::to_string(kind) +
           "u, " + std::to_string(degree) + "u, " + std::to_string(width) + "u, " +
           std::to_string(padded) + "u, " + literal(scale) + ", " + literal(offset) + ", input, " +
           row + ", " + dimension + direction + ")";
}
} // namespace

template <typename T>
std::string AnalyticEncoding<T>::generate_device_function(const std::string& name) const {
    const auto type = type_to_string<T>();
    std::ostringstream body;
    if (device_function_fwd_ctx_bytes())
        body << store_input(m_n_dims_to_encode);
    body << this->generate_vec_out() << " result{};\n";
    if (m_kind == 4u) {
        body << "for (uint d = 0; d < " << m_n_dims_to_encode << "u; ++d) {\n"
             << "auto left = tcnn_jit_encoding::blob_boundary<0>(-input[d], " << m_degree << "u);\n"
             << "for (uint k = 0; k < " << m_degree << "u; ++k) {\n"
             << "const auto right = tcnn_jit_encoding::blob_boundary<0>(float(k + 1u) / "
             << m_degree << ".0f - input[d], " << m_degree << "u);\n"
             << "result[d * " << m_degree << "u + k] = " << type
             << "(right.v - left.v); left = right;\n}\n}\n";
    } else if (m_kind == 5u) {
        const auto prefix = m_padded_output_width - m_n_output_dims;
        for (uint32_t r = 0; r < m_n_output_dims; ++r)
            body << "result[" << prefix + r << "u] = " << type << "("
                 << analytic_call(m_n_dims_to_encode, m_kind, m_degree, m_n_output_dims,
                                  m_padded_output_width, m_scale, m_offset,
                                  std::to_string(prefix + r) + "u", "0xffffffffu")
                 << ".v);\n";
        body << "for (uint r = 0; r < " << prefix << "u; ++r) result[r] = " << type << "(1.0f);\n";
    } else {
        body << "for (uint r = 0; r < " << m_n_output_dims << "u; ++r) result[r] = " << type << "("
             << analytic_call(m_n_dims_to_encode, m_kind, m_degree, m_n_output_dims,
                              m_padded_output_width, m_scale, m_offset, "r", "0xffffffffu")
             << ".v);\n";
    }
    if (m_kind != 5u)
        body << "for (uint r = " << m_n_output_dims << "u; r < " << m_padded_output_width
             << "u; ++r) result[r] = " << type << "(1.0f);\n";
    body << "return result;\n";
    return with_math(this->generate_device_function_from_body(name, body.str()));
}

template <typename T>
std::string AnalyticEncoding<T>::generate_backward_device_function(const std::string& name,
                                                                   uint32_t n_threads) const {
    (void)n_threads;
    std::ostringstream body;
    body << "if (!dL_dx) return;\n";
    if (device_function_fwd_ctx_bytes())
        body << load_input(m_n_dims_to_encode);
    else
        body << this->generate_vec_in() << " input{};\n";
    body << "for (uint d = 0; d < " << m_n_dims_to_encode << "u; ++d) {\nfloat value = 0.0f;\n";
    if (m_kind == 1u) {
        body << "value = float(dL_dy[d]) * " << literal(m_scale) << ";\n";
    } else if (m_kind == 4u) {
        body << "auto left = tcnn_jit_encoding::blob_boundary<1>(-input[d], " << m_degree << "u);\n"
             << "for (uint k = 0; k < " << m_degree << "u; ++k) {\n"
             << "const auto right = tcnn_jit_encoding::blob_boundary<1>(float(k + 1u) / "
             << m_degree << ".0f - input[d], " << m_degree << "u);\n"
             << "value += float(dL_dy[d * " << m_degree
             << "u + k]) * (left.d - right.d); left = right;\n}\n";
    } else if (m_kind == 5u) {
        const auto prefix = m_padded_output_width - m_n_output_dims;
        for (uint32_t r = 0; r < m_n_output_dims; ++r)
            body << "value += float(dL_dy[" << prefix + r << "u]) * "
                 << analytic_call(m_n_dims_to_encode, m_kind, m_degree, m_n_output_dims,
                                  m_padded_output_width, m_scale, m_offset,
                                  std::to_string(prefix + r) + "u", "d")
                 << ".d;\n";
    } else if (m_kind != 0u) {
        const uint32_t fan = m_degree * (m_kind == 2u ? 2u : 1u);
        body << "for (uint r = d * " << fan << "u; r < (d + 1u) * " << fan
             << "u; ++r) value += float(dL_dy[r]) * "
             << analytic_call(m_n_dims_to_encode, m_kind, m_degree, m_n_output_dims,
                              m_padded_output_width, m_scale, m_offset, "r", "d")
             << ".d;\n";
    }
    body << "(*dL_dx)[d] = "
         << (m_kind == 1 ? "float(" + type_to_string<T>() + "(value))" : "value") << ";\n}\n";
    return with_math(this->generate_backward_device_function_from_body(name, body.str()));
}

template <typename T>
std::string
AnalyticEncoding<T>::generate_backward_backward_input_device_function(const std::string& name,
                                                                      uint32_t n_threads) const {
    (void)n_threads;
    const auto type = type_to_string<T>();
    std::ostringstream body;
    body << "if (!dL_dx && !dL_ddLdy) return;\n";
    if (device_function_fwd_ctx_bytes())
        body << load_input(m_n_dims_to_encode);
    else
        body << this->generate_vec_in() << " input{};\n";
    body << "if (dL_ddLdy) for (uint r = 0; r < " << m_padded_output_width
         << "u; ++r) (*dL_ddLdy)[r] = " << type << "(0.0f);\n"
         << "for (uint d = 0; d < " << m_n_dims_to_encode << "u; ++d) {\nfloat value = 0.0f;\n";
    if (m_kind == 1u) {
        body << "if (dL_ddLdy) (*dL_ddLdy)[d] = " << type << "(dL_ddLdx[d] * " << literal(m_scale)
             << ");\n";
    } else if (m_kind == 4u) {
        body << "auto left = tcnn_jit_encoding::blob_boundary<2>(-input[d], " << m_degree << "u);\n"
             << "for (uint k = 0; k < " << m_degree << "u; ++k) {\n"
             << "const auto right = tcnn_jit_encoding::blob_boundary<2>(float(k + 1u) / "
             << m_degree << ".0f - input[d], " << m_degree << "u);\n"
             << "const uint r = d * " << m_degree << "u + k;\n"
             << "if (dL_dx) value += float(dL_dy[r]) * (right.dt - left.dt) * dL_ddLdx[d];\n"
             << "if (dL_ddLdy) (*dL_ddLdy)[r] = " << type
             << "((left.d - right.d) * dL_ddLdx[d]);\nleft = right;\n}\n";
    } else if (m_kind == 5u) {
        const auto prefix = m_padded_output_width - m_n_output_dims;
        for (uint32_t r = 0; r < m_n_output_dims; ++r)
            body << "if (dL_dx) value += float(dL_dy[" << prefix + r << "u]) * "
                 << analytic_call(m_n_dims_to_encode, m_kind, m_degree, m_n_output_dims,
                                  m_padded_output_width, m_scale, m_offset,
                                  std::to_string(prefix + r) + "u", "d", ", &dL_ddLdx")
                 << ".dt;\n";
    } else if (m_kind != 0u) {
        const uint32_t fan = m_degree * (m_kind == 2u ? 2u : 1u);
        body << "for (uint r = d * " << fan << "u; r < (d + 1u) * " << fan << "u; ++r) {\n"
             << "const auto derivative = "
             << analytic_call(m_n_dims_to_encode, m_kind, m_degree, m_n_output_dims,
                              m_padded_output_width, m_scale, m_offset, "r", "d", ", &dL_ddLdx")
             << ";\n"
             << "if (dL_dx) value += float(dL_dy[r]) * derivative.dt;\n"
             << "if (dL_ddLdy) (*dL_ddLdy)[r] = " << type << "(derivative.t);\n}\n";
    }
    body << "if (dL_dx) (*dL_dx)[d] = value;\n}\n";
    if (m_kind == 5u) {
        const auto prefix = m_padded_output_width - m_n_output_dims;
        for (uint32_t r = 0; r < m_n_output_dims; ++r)
            body << "if (dL_ddLdy) (*dL_ddLdy)[" << prefix + r << "u] = " << type << "("
                 << analytic_call(m_n_dims_to_encode, m_kind, m_degree, m_n_output_dims,
                                  m_padded_output_width, m_scale, m_offset,
                                  std::to_string(prefix + r) + "u", "0xffffffffu", ", &dL_ddLdx")
                 << ".t);\n";
    }
    return with_math(
        this->generate_backward_backward_input_device_function_from_body(name, body.str()));
}

namespace {
std::string grid_type(uint32_t dims, uint32_t features, uint32_t hash, uint32_t grid,
                      uint32_t interpolation, bool fixed) {
    return "tcnn_jit_encoding::Grid<" + std::to_string(dims) + ", " + std::to_string(features) +
           ", " + std::to_string(hash) + ", " + std::to_string(grid) + ", " +
           std::to_string(interpolation) + ", " + (fixed ? "true" : "false") + ">";
}
} // namespace

// Expose per-level scales, table sizes and optional derivatives to the Metal
// optimizer even when its ordinary inlining heuristic rejects a large caller.
template <typename T>
std::string LatticeEncoding<T>::generate_lookup_device_function(const std::string& name) const {
    detail::jit_validate_identifier(name);
    nn_check(!m_permuto, "Grid lookup generation requires a Cartesian lattice");
    const auto in = vector_type("float", m_n_dims_to_encode),
               out = vector_type(type_to_string<T>(), m_n_features_per_level);
    const auto policy = grid_type(m_n_dims_to_encode, m_n_features_per_level, m_hash_type,
                                  m_grid_type, m_interpolation, m_fixed_point_pos);
    return detail::jit_device_prelude() + math_source() + "__attribute__((always_inline)) inline " +
           out + " " + name + "(float scale, " + in + " pos, const device " + type_to_string<T>() +
           "* grid, uint hashmap_size) {\nreturn " + policy +
           "::forward(scale, pos, grid, hashmap_size);\n}\n";
}

template <typename T>
std::string
LatticeEncoding<T>::generate_backward_lookup_device_function(const std::string& name) const {
    detail::jit_validate_identifier(name);
    nn_check(!m_permuto, "Grid backward lookup generation requires a Cartesian lattice");
    const auto in = vector_type("float", m_n_dims_to_encode),
               out = vector_type(type_to_string<T>(), m_n_features_per_level);
    const auto policy = grid_type(m_n_dims_to_encode, m_n_features_per_level, m_hash_type,
                                  m_grid_type, m_interpolation, m_fixed_point_pos);
    return detail::jit_device_prelude() + math_source() +
           "__attribute__((always_inline)) inline void " + name + "(float scale, " + in +
           " pos, const device " + type_to_string<T>() +
           "* grid, uint hashmap_size, const thread " + out + "& dL_dy, device " +
           type_to_string<T>() + "* dL_dparams, thread " + in +
           "* dL_dpos, const thread tcnn_jit_environment* env = nullptr, uint level = 0u) {\n" +
           policy + "::backward(scale, pos, grid, hashmap_size, dL_dy, dL_dparams, dL_dpos, " +
           (m_stochastic_interpolation ? "true" : "false") + ", level, env);\n}\n";
}

template <typename T>
std::string LatticeEncoding<T>::generate_backward_backward_input_lookup_device_function(
    const std::string& name) const {
    detail::jit_validate_identifier(name);
    nn_check(!m_permuto, "Grid second-backward lookup generation requires a Cartesian lattice");
    const auto in = vector_type("float", m_n_dims_to_encode),
               out = vector_type(type_to_string<T>(), m_n_features_per_level);
    const auto policy = grid_type(m_n_dims_to_encode, m_n_features_per_level, m_hash_type,
                                  m_grid_type, m_interpolation, m_fixed_point_pos);
    return detail::jit_device_prelude() + math_source() +
           "__attribute__((always_inline)) inline void " + name + "(float scale, " + in +
           " pos, const device " + type_to_string<T>() +
           "* grid, uint hashmap_size, const thread " + in + "& dL_ddLdx, const thread " + out +
           "& dL_dy, device " + type_to_string<T>() + "* dL_dparams, thread " + in +
           "* dL_dpos, thread " + out +
           "* dL_ddLdy, const thread tcnn_jit_environment* env = nullptr) {\n" + policy +
           "::backward_backward_input(scale, pos, grid, hashmap_size, dL_ddLdx, dL_dy, dL_dparams, "
           "dL_dpos, dL_ddLdy, env);\n}\n";
}

template <typename T>
std::string LatticeEncoding<T>::generate_jit_device_function(const std::string& name,
                                                             uint32_t operation) const {
    const auto type = type_to_string<T>();
    const auto part = vector_type(type, m_n_features_per_level);
    const auto in = vector_type("float", m_n_dims_to_encode);
    const auto lookup = name + "_lookup";
    std::string preamble;
    if (!m_permuto) {
        preamble = operation == 0 ? generate_lookup_device_function(lookup)
                   : operation == 1
                       ? generate_backward_lookup_device_function(lookup)
                       : generate_backward_backward_input_lookup_device_function(lookup);
    }
    std::ostringstream body;
    if (operation == 0) {
        body << store_input(m_n_dims_to_encode) << this->generate_vec_out() << " result{};\n";
    } else {
        body << "if (!dL_dx && !dL_dparams" << (operation == 2 ? " && !dL_ddLdy" : "")
             << ") return;\n";
        body << load_input(m_n_dims_to_encode);
        body << "if (dL_dx) for (uint d = 0; d < " << m_n_dims_to_encode
             << "u; ++d) (*dL_dx)[d] = 0.0f;\n";
        if (operation == 2)
            body << "if (dL_ddLdy) for (uint r = 0; r < " << m_padded_output_width
                 << "u; ++r) (*dL_ddLdy)[r] = " << type << "(0.0f);\n";
    }
    body << "const float max_level = "
         << level_limit(this->m_max_level, this->m_max_level_gpu != nullptr) << ";\n";
    // Large unrolled derivative chains compose poorly with the MLP's matrix
    // state. Bound the live level work while preserving ascending accumulation
    // order and the exact per-level scale bits, offsets and stochastic seeds.
    // Forward keeps literal specialization and independent gathers; small grids
    // and the separate simplex policy retain their existing generated schedule.
    const bool loop_levels = !m_permuto && m_n_levels >= 8 && operation != 0;
    if (loop_levels) {
        body << "const float scales[" << m_n_levels
             << "] = " << float_array(m_scales, 0, m_n_levels) << ";\n"
             << "const uint offsets[" << m_n_levels << "] = {";
        for (uint32_t l = 0; l < m_n_levels; ++l)
            body << (l ? "," : "") << m_offset_table.data[l] * m_n_features_per_level << "u";
        body << "};\nconst uint sizes[" << m_n_levels << "] = {";
        for (uint32_t l = 0; l < m_n_levels; ++l)
            body << (l ? "," : "") << m_offset_table.data[l + 1] - m_offset_table.data[l] << "u";
        body << "};\n#pragma nounroll\nfor (uint level = 0; level < " << m_n_levels
             << "u; ++level) {\n";
    }
    for (uint32_t l = 0; l < (loop_levels ? 1u : m_n_levels); ++l) {
        const uint32_t offset = m_offset_table.data[l] * m_n_features_per_level;
        const uint32_t size = m_offset_table.data[l + 1] - m_offset_table.data[l];
        const auto level = loop_levels ? "level" : std::to_string(l) + "u";
        const auto feature = loop_levels ? "level * " + std::to_string(m_n_features_per_level) + "u"
                                         : std::to_string(l * m_n_features_per_level) + "u";
        body << "if (!(float(" << level << ") >= max_level * " << m_n_levels << ".0f + 1e-3f)) {\n";
        if (m_permuto) {
            body << "const " << in << " scales = "
                 << float_array(m_scales, l * m_n_dims_to_encode, m_n_dims_to_encode) << ";\n"
                 << "const " << in << " shifts = "
                 << float_array(m_shifts, l * m_n_dims_to_encode, m_n_dims_to_encode) << ";\n";
        }
        if (operation != 0) {
            body << part << " dy{};\n"
                 << "for (uint f = 0; f < " << m_n_features_per_level << "u; ++f) dy[f] = dL_dy["
                 << feature << " + f];\n";
        }
        std::ostringstream arguments;
        if (m_permuto)
            arguments << "input, params + " << offset << "u, " << size << "u, scales, shifts, "
                      << m_max_input_grad_dims << "u";
        else if (loop_levels)
            arguments << "scales[level], input, params + offsets[level], sizes[level]";
        else
            arguments << literal(m_scales[l]) << ", input, params + " << offset << "u, " << size
                      << "u";
        const auto gradient = loop_levels ? "dL_dparams ? dL_dparams + offsets[level] : nullptr"
                                          : ptr_offset("dL_dparams", offset);
        if (operation == 1)
            arguments << ", dy, " << gradient << ", dL_dx, env" << (m_permuto ? "" : ", " + level);
        else if (operation == 2)
            arguments << ", dL_ddLdx, dy, " << gradient
                      << ", dL_dx, dL_ddLdy ? &part : nullptr, env";
        const auto function = m_permuto ? "tcnn_jit_encoding::Permuto<" +
                                              std::to_string(m_n_dims_to_encode) + ", " +
                                              std::to_string(m_n_features_per_level) + ">::" +
                                              (operation == 0   ? "forward"
                                               : operation == 1 ? "backward"
                                                                : "backward_backward_input")
                                        : lookup;
        if (operation == 0)
            body << "const auto part = " << function << "(" << arguments.str() << ");\n"
                 << "for (uint f = 0; f < " << m_n_features_per_level << "u; ++f) result["
                 << feature << " + f] = part[f];\n";
        else if (operation == 1)
            body << function << "(" << arguments.str() << ");\n";
        else
            body << part << " part{};\n"
                 << function << "(" << arguments.str() << ");\n"
                 << "if (dL_ddLdy) for (uint f = 0; f < " << m_n_features_per_level
                 << "u; ++f) (*dL_ddLdy)[" << feature << " + f] = part[f];\n";
        body << "}\n";
    }
    if (loop_levels)
        body << "}\n";
    if (operation == 0)
        body << "return result;\n";
    const auto function =
        operation == 0 ? this->generate_device_function_from_body(name, body.str())
        : operation == 1
            ? this->generate_backward_device_function_from_body(name, body.str())
            : this->generate_backward_backward_input_device_function_from_body(name, body.str());
    return with_math(preamble + function);
}

template <typename T>
std::string LatticeEncoding<T>::generate_device_function(const std::string& name) const {
    return generate_jit_device_function(name, 0);
}

template <typename T>
std::string LatticeEncoding<T>::generate_backward_device_function(const std::string& name,
                                                                  uint32_t n_threads) const {
    (void)n_threads;
    return generate_jit_device_function(name, 1);
}

template <typename T>
std::string
LatticeEncoding<T>::generate_backward_backward_input_device_function(const std::string& name,
                                                                     uint32_t n_threads) const {
    (void)n_threads;
    return generate_jit_device_function(name, 2);
}
} // namespace encoding_detail

using encoding_detail::child_context;
using encoding_detail::level_limit;
using encoding_detail::literal;
using encoding_detail::ptr_offset;
using encoding_detail::sample_tail;
using encoding_detail::shifted_env;
using encoding_detail::vector_type;

namespace {
template <typename T> uint32_t child_context_size(const std::shared_ptr<Encoding<T>>& child) {
    const size_t bytes = child->device_function_fwd_ctx_bytes();
    nn_check(bytes <= UINT32_MAX - 3u, "Child JIT context exceeds uint32_t");
    return static_cast<uint32_t>((bytes + 3u) & ~size_t(3u));
}

template <typename T>
uint32_t product_values_offset(const std::vector<std::shared_ptr<Encoding<T>>>& children) {
    size_t result = 0;
    for (const auto& child : children)
        result = metalrt::checked_add(result, child_context_size(child));
    nn_check(result <= UINT32_MAX, "Composite JIT context exceeds uint32_t");
    return static_cast<uint32_t>(result);
}
} // namespace

template <typename T>
std::string CompositeEncoding<T>::generate_device_function(const std::string& name) const {
    const bool grouped = !device_function_fwd_ctx_aligned_per_element();
    const auto type = type_to_string<T>();
    std::ostringstream preamble, body;
    for (size_t i = 0; i < m_nested.size(); ++i)
        preamble << m_nested[i]->generate_device_function(generate_nested_name(name, i)) << '\n';
    body << this->generate_vec_out() << " result{};\n";
    if (m_reduction_type != ReductionType::Concatenation) {
        body << vector_type("float", padded_output_width()) << " reduction{};\n";
        if (m_reduction_type == ReductionType::Product)
            body << "for (uint r = 0; r < " << padded_output_width()
                 << "u; ++r) reduction[r] = 1.0f;\n";
    }
    const uint32_t values_offset = product_values_offset(m_nested);
    size_t param_offset = 0;
    uint32_t output_offset = 0, context_offset = 0, runtime_offset = 0;
    for (size_t i = 0; i < m_nested.size(); ++i) {
        const auto& child = m_nested[i];
        body << "{\n"
             << vector_type("float", child->input_width()) << " local_input{};\n"
             << "for (uint d = 0; d < " << child->input_width() << "u; ++d) local_input[d] = input["
             << m_dims_to_encode_begin[i] << "u + d];\n"
             << shifted_env(runtime_offset, "local_env")
             << "const auto value = " << generate_nested_name(name, i) << "(local_input, "
             << ptr_offset("params", param_offset) << ", "
             << child_context(context_offset, true, grouped) << ", env ? &local_env : nullptr);\n";
        if (m_reduction_type == ReductionType::Concatenation)
            body << "for (uint r = 0; r < " << child->output_width() << "u; ++r) result["
                 << output_offset << "u + r] = value[r];\n";
        else {
            body << "for (uint r = 0; r < " << child->output_width() << "u; ++r) reduction[r] "
                 << (m_reduction_type == ReductionType::Sum ? "+=" : "*=") << " float(value[r]);\n";
            if (m_reduction_type == ReductionType::Product)
                body << "if (fwd_ctx) for (uint r = 0; r < " << child->output_width() << "u; ++r) "
                     << "reinterpret_cast<TCNN_JIT_CTX float*>("
                     << sample_tail(values_offset,
                                    static_cast<uint32_t>(m_nested.size() * padded_output_width() *
                                                          sizeof(float)),
                                    grouped)
                     << ")[" << i * padded_output_width() << "u + r] = float(value[r]);\n";
        }
        body << "}\n";
        output_offset += child->output_width();
        param_offset += child->n_params();
        context_offset += child_context_size(child);
        runtime_offset += static_cast<uint32_t>(child->jit_runtime_inputs().size());
    }
    if (m_reduction_type != ReductionType::Concatenation)
        body << "for (uint r = 0; r < " << padded_output_width() << "u; ++r) result[r] = " << type
             << "(reduction[r]);\n";
    body << "return result;\n";
    return preamble.str() + this->generate_device_function_from_body(name, body.str());
}

template <typename T>
std::string CompositeEncoding<T>::generate_backward_device_function(const std::string& name,
                                                                    uint32_t n_threads) const {
    const bool grouped = !device_function_fwd_ctx_aligned_per_element();
    const auto type = type_to_string<T>();
    std::ostringstream preamble, body;
    for (size_t i = 0; i < m_nested.size(); ++i)
        preamble << m_nested[i]->generate_backward_device_function(generate_nested_name(name, i),
                                                                   n_threads)
                 << '\n';
    body << "if (!dL_dx && !dL_dparams) return;\n"
         << "if (dL_dx) for (uint d = 0; d < " << input_width() << "u; ++d) (*dL_dx)[d] = 0.0f;\n";
    if (m_reduction_type == ReductionType::Product)
        body << "const TCNN_JIT_CTX float* values = reinterpret_cast<const TCNN_JIT_CTX float*>("
             << sample_tail(
                    product_values_offset(m_nested),
                    static_cast<uint32_t>(m_nested.size() * padded_output_width() * sizeof(float)),
                    grouped)
             << ");\n";
    size_t param_offset = 0;
    uint32_t output_offset = 0, context_offset = 0, runtime_offset = 0;
    for (size_t i = 0; i < m_nested.size(); ++i) {
        const auto& child = m_nested[i];
        body << "{\n"
             << vector_type("float", child->input_width()) << " dx{};\n"
             << vector_type(type, child->output_width()) << " dy{};\n"
             << "for (uint r = 0; r < " << child->output_width()
             << "u; ++r) {\nfloat value = float(dL_dy["
             << (m_reduction_type == ReductionType::Concatenation ? output_offset : 0)
             << "u + r]);\n";
        if (m_reduction_type == ReductionType::Product)
            for (size_t k = 0; k < m_nested.size(); ++k)
                if (i != k)
                    body << "value *= values[" << k * padded_output_width() << "u + r];\n";
        body << "dy[r] = " << type << "(value);\n}\n"
             << shifted_env(runtime_offset, "local_env") << generate_nested_name(name, i) << "(dy, "
             << ptr_offset("params", param_offset) << ", "
             << child_context(context_offset, true, grouped) << ", "
             << ptr_offset("dL_dparams", param_offset)
             << ", dL_dx ? &dx : nullptr, env ? &local_env : nullptr);\n"
             << "if (dL_dx) for (uint d = 0; d < " << child->input_width() << "u; ++d) (*dL_dx)["
             << m_dims_to_encode_begin[i] << "u + d] += dx[d];\n}\n";
        output_offset += child->output_width();
        param_offset += child->n_params();
        context_offset += child_context_size(child);
        runtime_offset += static_cast<uint32_t>(child->jit_runtime_inputs().size());
    }
    return preamble.str() + this->generate_backward_device_function_from_body(name, body.str());
}

template <typename T>
std::string
CompositeEncoding<T>::generate_backward_backward_input_device_function(const std::string& name,
                                                                       uint32_t n_threads) const {
    const bool grouped = !device_function_fwd_ctx_aligned_per_element();
    const auto type = type_to_string<T>();
    const bool product = m_reduction_type == ReductionType::Product;
    std::ostringstream preamble, body;
    for (size_t i = 0; i < m_nested.size(); ++i) {
        preamble << m_nested[i]->generate_backward_backward_input_device_function(
                        generate_nested_name(name, i), n_threads)
                 << '\n';
        if (product)
            preamble << m_nested[i]->generate_backward_device_function(
                            generate_nested_name(name + "_first", i), n_threads)
                     << '\n';
    }
    body << "if (!dL_dx && !dL_dparams && !dL_ddLdy) return;\n"
         << "if (dL_dx) for (uint d = 0; d < " << input_width() << "u; ++d) (*dL_dx)[d] = 0.0f;\n"
         << "if (dL_ddLdy) for (uint r = 0; r < " << output_width()
         << "u; ++r) (*dL_ddLdy)[r] = " << type << "(0.0f);\n";
    if (product)
        body << "const TCNN_JIT_CTX float* values = reinterpret_cast<const TCNN_JIT_CTX float*>("
             << sample_tail(
                    product_values_offset(m_nested),
                    static_cast<uint32_t>(m_nested.size() * padded_output_width() * sizeof(float)),
                    grouped)
             << ");\n"
             << vector_type(type, static_cast<uint32_t>(m_nested.size() * padded_output_width()))
             << " tangents{};\n";
    size_t param_offset = 0;
    uint32_t output_offset = 0, context_offset = 0, runtime_offset = 0;
    for (size_t i = 0; i < m_nested.size(); ++i) {
        const auto& child = m_nested[i];
        body << "{\n"
             << vector_type("float", child->input_width()) << " dx{}, direction{};\n"
             << vector_type(type, child->output_width()) << " dy{}, ddy{};\n"
             << "for (uint d = 0; d < " << child->input_width()
             << "u; ++d) direction[d] = dL_ddLdx[" << m_dims_to_encode_begin[i] << "u + d];\n"
             << "for (uint r = 0; r < " << child->output_width()
             << "u; ++r) {\nfloat value = float(dL_dy["
             << (m_reduction_type == ReductionType::Concatenation ? output_offset : 0)
             << "u + r]);\n";
        if (product)
            for (size_t k = 0; k < m_nested.size(); ++k)
                if (i != k)
                    body << "value *= values[" << k * padded_output_width() << "u + r];\n";
        body << "dy[r] = " << type << "(value);\n}\n"
             << shifted_env(runtime_offset, "local_env") << generate_nested_name(name, i)
             << "(direction, dy, " << ptr_offset("params", param_offset) << ", "
             << child_context(context_offset, true, grouped) << ", "
             << ptr_offset("dL_dparams", param_offset) << ", dL_dx ? &dx : nullptr, "
             << (product ? "&ddy" : "dL_ddLdy ? &ddy : nullptr")
             << ", env ? &local_env : nullptr);\n"
             << "if (dL_dx) for (uint d = 0; d < " << child->input_width() << "u; ++d) (*dL_dx)["
             << m_dims_to_encode_begin[i] << "u + d] += dx[d];\n";
        if (product)
            body << "for (uint r = 0; r < " << child->output_width() << "u; ++r) tangents["
                 << i * padded_output_width() << "u + r] = ddy[r];\n";
        else
            body << "if (dL_ddLdy) for (uint r = 0; r < " << child->output_width()
                 << "u; ++r) (*dL_ddLdy)["
                 << (m_reduction_type == ReductionType::Concatenation ? output_offset : 0)
                 << "u + r] += ddy[r];\n";
        body << "}\n";
        output_offset += child->output_width();
        param_offset += child->n_params();
        context_offset += child_context_size(child);
        runtime_offset += static_cast<uint32_t>(child->jit_runtime_inputs().size());
    }
    if (product) {
        // Cross-child adjoints use products excluding the differentiated pair,
        // never division by a child output. Zero factors therefore stay valid.
        param_offset = 0;
        context_offset = 0;
        runtime_offset = 0;
        for (size_t i = 0; i < m_nested.size(); ++i) {
            const auto& child = m_nested[i];
            body << "{\n"
                 << vector_type("float", child->input_width()) << " dx{};\n"
                 << vector_type(type, child->output_width()) << " dy{};\n"
                 << "for (uint r = 0; r < " << child->output_width()
                 << "u; ++r) {\nfloat sum = 0.0f;\n";
            for (size_t k = 0; k < m_nested.size(); ++k)
                if (i != k) {
                    body << "{ float value = float(tangents[" << k * padded_output_width()
                         << "u + r]);\n";
                    for (size_t j = 0; j < m_nested.size(); ++j)
                        if (j != i && j != k)
                            body << "value *= values[" << j * padded_output_width() << "u + r];\n";
                    body << "sum += value; }\n";
                }
            body << "dy[r] = " << type << "(float(dL_dy[r]) * sum);\n}\n"
                 << shifted_env(runtime_offset, "local_env")
                 << generate_nested_name(name + "_first", i) << "(dy, "
                 << ptr_offset("params", param_offset) << ", "
                 << child_context(context_offset, true, grouped) << ", "
                 << ptr_offset("dL_dparams", param_offset)
                 << ", dL_dx ? &dx : nullptr, env ? &local_env : nullptr);\n"
                 << "if (dL_dx) for (uint d = 0; d < " << child->input_width()
                 << "u; ++d) (*dL_dx)[" << m_dims_to_encode_begin[i] << "u + d] += dx[d];\n}\n";
            param_offset += child->n_params();
            context_offset += child_context_size(child);
            runtime_offset += static_cast<uint32_t>(child->jit_runtime_inputs().size());
        }
        body << "if (dL_ddLdy) for (uint r = 0; r < " << padded_output_width()
             << "u; ++r) {\nfloat sum = 0.0f;\n";
        for (size_t i = 0; i < m_nested.size(); ++i) {
            body << "{ float value = float(tangents[" << i * padded_output_width() << "u + r]);\n";
            for (size_t k = 0; k < m_nested.size(); ++k)
                if (i != k)
                    body << "value *= values[" << k * padded_output_width() << "u + r];\n";
            body << "sum += value; }\n";
        }
        body << "(*dL_ddLdy)[r] = " << type << "(sum);\n}\n";
    }
    return preamble.str() +
           this->generate_backward_backward_input_device_function_from_body(name, body.str());
}

template <typename T>
std::string MultiLevelEncodingLoD<T>::generate_jit_device_function(const std::string& name,
                                                                   uint32_t n_threads,
                                                                   uint32_t operation) const {
    const bool grouped = !device_function_fwd_ctx_aligned_per_element();
    const auto type = type_to_string<T>(), base_name = name + "_base";
    const auto base_function =
        operation == 0 ? m_base->generate_device_function(base_name)
        : operation == 1
            ? m_base->generate_backward_device_function(base_name, n_threads)
            : m_base->generate_backward_backward_input_device_function(base_name, n_threads);
    const uint32_t dims = m_base->input_width(), context_bytes = child_context_size<T>(m_base);
    std::ostringstream body;
    body << "tcnn_jit_environment local_env{};\nlocal_env.count = 1u;\n"
         << "if (env) local_env = *env;\n"
         << "local_env.offset += " << (this->m_max_level_gpu ? 1u : 0u) << "u;\n";
    if (operation == 0) {
        body << "const float level = fmin(input[" << dims << "u], "
             << level_limit(this->m_max_level, this->m_max_level_gpu != nullptr) << ");\n"
             << "if (fwd_ctx) *reinterpret_cast<TCNN_JIT_CTX float*>("
             << sample_tail(context_bytes, sizeof(float), grouped) << ") = level;\n"
             << vector_type("float", dims) << " positions{};\n"
             << "for (uint d = 0; d < " << dims << "u; ++d) positions[d] = input[d];\n";
    } else {
        body << "const float level = *reinterpret_cast<const TCNN_JIT_CTX float*>("
             << sample_tail(context_bytes, sizeof(float), grouped) << ");\n"
             << vector_type(type, output_width()) << " dy{};\n"
             << vector_type("float", dims) << " dx{};\n"
             << "for (uint r = 0; r < " << output_width() << "u; ++r) dy[r] = dL_dy[r];\n";
        if (operation == 2)
            body << vector_type("float", dims) << " direction{};\n"
                 << vector_type(type, output_width()) << " ddy{};\n"
                 << "for (uint d = 0; d < " << dims << "u; ++d) direction[d] = dL_ddLdx[d];\n";
    }
    body << "local_env.has_level_override = 1u;\nlocal_env.level_override = level;\n";
    const auto mask = "fmax(0.0f, fmin(1.0f, level * " + std::to_string(n_levels()) +
                      ".0f + 1e-3f - float(r / " + std::to_string(n_features_per_level()) + "u)))";
    const uint32_t real_features = n_levels() * n_features_per_level();
    if (operation == 0) {
        body << "auto result = " << base_name << "(positions, params, fwd_ctx, &local_env);\n";
        if (m_is_soft)
            body << "for (uint r = 0; r < " << real_features
                 << "u; ++r) { const float weight = " << mask << ";\n"
                 << "result[r] = weight == 0.0f ? " << type << "(0.0f) : " << type
                 << "(weight * float(result[r])); }\n";
        body << "return result;\n";
    } else {
        if (m_is_soft)
            body << "for (uint r = 0; r < " << real_features
                 << "u; ++r) { const float weight = " << mask << ";\n"
                 << "dy[r] = weight == 0.0f ? " << type << "(0.0f) : " << type
                 << "(weight * float(dy[r])); }\n";
        body << base_name << "(" << (operation == 2 ? "direction, " : "")
             << "dy, params, fwd_ctx, dL_dparams, dL_dx ? &dx : nullptr, "
             << (operation == 2 ? "dL_ddLdy ? &ddy : nullptr, " : "") << "&local_env);\n"
             << "if (dL_dx) { for (uint d = 0; d < " << dims
             << "u; ++d) (*dL_dx)[d] = dx[d];\n(*dL_dx)[" << dims << "u] = 0.0f; }\n";
        if (operation == 2) {
            if (m_is_soft)
                body << "if (dL_ddLdy) for (uint r = 0; r < " << real_features
                     << "u; ++r) { const float weight = " << mask << ";\n"
                     << "ddy[r] = weight == 0.0f ? " << type << "(0.0f) : " << type
                     << "(weight * float(ddy[r])); }\n";
            body << "if (dL_ddLdy) *dL_ddLdy = ddy;\n";
        }
    }
    return base_function +
           (operation == 0   ? this->generate_device_function_from_body(name, body.str())
            : operation == 1 ? this->generate_backward_device_function_from_body(name, body.str())
                             : this->generate_backward_backward_input_device_function_from_body(
                                   name, body.str()));
}

template <typename T>
std::string MultiLevelEncodingLoD<T>::generate_device_function(const std::string& name) const {
    return generate_jit_device_function(name, 32, 0);
}

template <typename T>
std::string MultiLevelEncodingLoD<T>::generate_backward_device_function(const std::string& name,
                                                                        uint32_t n_threads) const {
    return generate_jit_device_function(name, n_threads, 1);
}

template <typename T>
std::string MultiLevelEncodingLoD<T>::generate_backward_backward_input_device_function(
    const std::string& name, uint32_t n_threads) const {
    return generate_jit_device_function(name, n_threads, 2);
}

#define TCNN_INSTANTIATE_ENCODING_JIT(T)                                                           \
    template std::string encoding_detail::AnalyticEncoding<T>::generate_device_function(           \
        const std::string&) const;                                                                 \
    template std::string encoding_detail::AnalyticEncoding<T>::generate_backward_device_function(  \
        const std::string&, uint32_t) const;                                                       \
    template std::string                                                                           \
    encoding_detail::AnalyticEncoding<T>::generate_backward_backward_input_device_function(        \
        const std::string&, uint32_t) const;                                                       \
    template std::string encoding_detail::LatticeEncoding<T>::generate_device_function(            \
        const std::string&) const;                                                                 \
    template std::string encoding_detail::LatticeEncoding<T>::generate_backward_device_function(   \
        const std::string&, uint32_t) const;                                                       \
    template std::string                                                                           \
    encoding_detail::LatticeEncoding<T>::generate_backward_backward_input_device_function(         \
        const std::string&, uint32_t) const;                                                       \
    template std::string encoding_detail::LatticeEncoding<T>::generate_lookup_device_function(     \
        const std::string&) const;                                                                 \
    template std::string                                                                           \
    encoding_detail::LatticeEncoding<T>::generate_backward_lookup_device_function(                 \
        const std::string&) const;                                                                 \
    template std::string                                                                           \
    encoding_detail::LatticeEncoding<T>::generate_backward_backward_input_lookup_device_function(  \
        const std::string&) const;                                                                 \
    template std::string CompositeEncoding<T>::generate_device_function(const std::string&) const; \
    template std::string CompositeEncoding<T>::generate_backward_device_function(                  \
        const std::string&, uint32_t) const;                                                       \
    template std::string CompositeEncoding<T>::generate_backward_backward_input_device_function(   \
        const std::string&, uint32_t) const;                                                       \
    template std::string MultiLevelEncodingLoD<T>::generate_device_function(const std::string&)    \
        const;                                                                                     \
    template std::string MultiLevelEncodingLoD<T>::generate_backward_device_function(              \
        const std::string&, uint32_t) const;                                                       \
    template std::string                                                                           \
    MultiLevelEncodingLoD<T>::generate_backward_backward_input_device_function(const std::string&, \
                                                                               uint32_t) const;
TCNN_INSTANTIATE_ENCODING_JIT(float)
TCNN_INSTANTIATE_ENCODING_JIT(half)
#undef TCNN_INSTANTIATE_ENCODING_JIT
} // namespace tcnn
