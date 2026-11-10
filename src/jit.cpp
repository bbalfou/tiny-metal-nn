// Model-independent native Metal code generation and workspace management.
#include <tiny-metal-nn/object.h>
#include <tiny-metal-nn/detail/jit.h>
#include <tiny-metal-nn/detail/shaders.h>
#include <cctype>
#include <initializer_list>
#include <sstream>
#include <utility>

namespace tcnn {
namespace detail {
void jit_validate_identifier(const std::string& name) {
    if (name.empty() || !(std::isalpha(static_cast<unsigned char>(name[0])) || name[0] == '_'))
        throw std::invalid_argument("JIT function name must be a Metal identifier");
    for (unsigned char c : name)
        if (!(std::isalnum(c) || c == '_'))
            throw std::invalid_argument("JIT function name must be a Metal identifier");
}

std::string jit_device_prelude() {
    return shader_source("jit/common.metal") + "\n" + shader_source("jit/mlp.metal") + "\n";
}

namespace {
uint32_t source_requirement(const std::string& source, const char* name, uint32_t alignment) {
    const std::string marker = std::string{"// "} + name + ": ";
    uint32_t maximum = 0;
    size_t at = 0;
    while ((at = source.find(marker, at)) != std::string::npos) {
        at += marker.size();
        size_t end = at;
        uint64_t value = 0;
        while (end < source.size() && source[end] >= '0' && source[end] <= '9') {
            value = value * 10 + static_cast<unsigned>(source[end++] - '0');
            if (value > UINT32_MAX)
                throw std::invalid_argument("Generated JIT workspace exceeds uint32_t");
        }
        if (end == at ||
            (end < source.size() && !std::isspace(static_cast<unsigned char>(source[end]))) ||
            value % alignment)
            throw std::invalid_argument("Invalid generated JIT workspace annotation");
        maximum = std::max(maximum, static_cast<uint32_t>(value));
        at = end;
    }
    return maximum;
}

std::string simd_workspace_prologue(uint32_t bytes) {
    if (!bytes)
        return {};
    return "env.simd_shared_bytes=(simd_width==32u && env.lane_count==32u) ? " +
           std::to_string(bytes) +
           "u : 0u;\n"
           "env.simd_shared_memory=env.simd_shared_bytes ? scratch+ulong(simd_group)*" +
           std::to_string(bytes) + "u : nullptr;\n";
}

std::string float_gradient_prologue(const JitGradientPlan& plan) {
    if (!plan.parameters)
        return {};
    return "env.float_gradients=wide_gradients;\n"
           "env.float_gradient_count=" +
           std::to_string(plan.parameters) +
           "u;\n"
           "env.dense_gradient_count=" +
           std::to_string(plan.dense_parameters) +
           "u;\n"
           "env.gradient_partitions=" +
           std::to_string(plan.partitions) + "u;\n";
}

// Metal requires a real resource for every active buffer argument. The runtime
// supplies a retained fallback for an absent binding and derives this mask
// from the semantic arguments. Restore optional-pointer behavior before any
// generated function sees the binding; a fallback is never model data.
std::string
optional_pointer_prologue(std::initializer_list<std::pair<uint32_t, const char*>> pointers) {
    std::ostringstream source;
    for (const auto& pointer : pointers)
        source << "if (!(present_buffers & (1u << " << pointer.first << "))) " << pointer.second
               << "=nullptr;\n";
    return source.str();
}

std::string context_prologue(uint32_t bytes, bool per_element) {
    if (!per_element)
        return {};
    return "tcnn_jit_environment context_env{};\n"
           "if (env) { context_env=*env;\n"
           "if (fwd_ctx && !env->context_is_per_element) fwd_ctx += ulong(env->lane_index)*" +
           std::to_string(bytes) +
           "u;\n"
           "context_env.context_is_per_element=1u; env=&context_env; }\n";
}
} // namespace

uint32_t jit_simdgroup_scratch_bytes(const std::string& source) {
    return source_requirement(source, "TCNN_JIT_SIMDGROUP_SCRATCH_BYTES", 16);
}

uint32_t jit_threadgroup_scratch_bytes(const std::string& source, uint32_t threads) {
    const auto bytes = metalrt::checked_mul(threads / 32u, jit_simdgroup_scratch_bytes(source));
    nn_check(bytes <= UINT32_MAX, "Generated JIT workspace exceeds uint32_t");
    return static_cast<uint32_t>(bytes);
}

// Inline across composition boundaries so optional destinations and native
// workspace metadata can specialize away unused derivative/scalar branches.
std::string jit_forward_function(const std::string& name, const std::string& in,
                                 const std::string& out, const std::string& params,
                                 const std::string& body, uint32_t bytes, bool per_element) {
    jit_validate_identifier(name);
    return jit_device_prelude() + "__attribute__((always_inline)) inline " + out + " " + name +
           "(const thread " + in + "& input, const device " + params +
           "* params, TCNN_JIT_CTX uchar* fwd_ctx = nullptr, "
           "const thread tcnn_jit_environment* env = nullptr) {\n" +
           context_prologue(bytes, per_element) +
           // The public raw MLP forward body historically needs no environment.
           // Callable wrappers opt into collectives without changing that contract.
           "#define TCNN_JIT_MLP_CALLABLE_ENV env\n" + body +
           "\n#undef TCNN_JIT_MLP_CALLABLE_ENV\n}\n";
}

std::string jit_backward_function(const std::string& name, const std::string& in,
                                  const std::string& out, const std::string& params,
                                  const std::string& body, uint32_t bytes, bool per_element) {
    jit_validate_identifier(name);
    return jit_device_prelude() + "__attribute__((always_inline)) inline void " + name +
           "(const thread " + out + "& dL_dy, const device " + params +
           "* params, const TCNN_JIT_CTX uchar* fwd_ctx, device " + params +
           "* dL_dparams = nullptr, thread " + in +
           "* dL_dx = nullptr, "
           "const thread tcnn_jit_environment* env = nullptr) {\n" +
           context_prologue(bytes, per_element) + body + "\n}\n";
}

std::string jit_second_function(const std::string& name, const std::string& in,
                                const std::string& out, const std::string& params,
                                const std::string& body, uint32_t bytes, bool per_element) {
    jit_validate_identifier(name);
    return jit_device_prelude() + "__attribute__((always_inline)) inline void " + name +
           "(const thread " + in + "& dL_ddLdx, const thread " + out + "& dL_dy, const device " +
           params + "* params, const TCNN_JIT_CTX uchar* fwd_ctx, device " + params +
           "* dL_dparams = nullptr, thread " + in + "* dL_dx = nullptr, thread " + out +
           "* dL_ddLdy = nullptr, const thread tcnn_jit_environment* env = nullptr) {\n" +
           context_prologue(bytes, per_element) + body + "\n}\n";
}

std::string jit_forward_kernel_source(const std::string& name, const std::string& function,
                                      const std::string& input_type,
                                      const std::string& parameter_type,
                                      const std::string& output_type, uint32_t input_width,
                                      uint32_t context_bytes) {
    jit_validate_identifier(name);
    const uint32_t simd_scratch = jit_simdgroup_scratch_bytes(function);
    std::ostringstream s;
    s << jit_device_prelude() << function << "\nkernel void " << name << "(\n"
      << "constant uint& n [[buffer(0)]], const device " << input_type << "* input [[buffer(1)]],\n"
      << "constant tcnn_jit_matrix_strides& is [[buffer(2)]], device " << output_type
      << "* output [[buffer(3)]],\n"
      << "constant tcnn_jit_matrix_strides& os [[buffer(4)]], const device " << parameter_type
      << "* params [[buffer(5)]],\n"
      << "device uchar* context [[buffer(6)]], const device float* controls [[buffer(7)]],\n"
      << "constant uint& output_rows [[buffer(8)]], constant uint& present_buffers [[buffer(9)]],\n"
      << (simd_scratch ? "threadgroup uchar* scratch [[threadgroup(0)]],\n" : "")
      << "uint3 global_tid [[thread_position_in_grid]],\n"
      << "uint ti [[thread_index_in_threadgroup]], uint3 group_extent "
         "[[threads_per_threadgroup]],\n"
      << "uint lane [[thread_index_in_simdgroup]], uint simd_width [[threads_per_simdgroup]], uint "
         "simd_group [[simdgroup_index_in_threadgroup]]) {\n"
      << optional_pointer_prologue(
             {{1, "input"}, {3, "output"}, {5, "params"}, {6, "context"}, {7, "controls"}})
      << "const uint i=global_tid.x-ti+simd_group*simd_width+lane;\nif (i >= n) "
         "return;\ntcnn_jit_array<"
      << input_type << "," << input_width << "> x{};\n"
      << "for (uint d=0; d<" << input_width
      << "; ++d) x[d]=input[ulong(i)*is.column+ulong(d)*is.row];\n"
      << "tcnn_jit_environment env{}; env.runtime_inputs=controls; env.index=i; env.count=n;\n"
      << "env.thread_index=ti; env.group_size=group_extent.x*group_extent.y*group_extent.z;\n"
      << "env.lane_index=lane; "
         "env.lane_count=min(simd_width,env.group_size-simd_group*simd_width);\n"
      << (simd_scratch ? "env.shared_memory=scratch;\n" : "")
      << simd_workspace_prologue(simd_scratch) << "auto y=eval_model(x,params,";
    if (context_bytes)
        s << "context ? context+ulong(i-lane)*" << context_bytes << " : nullptr";
    else
        s << "context";
    s << ",&env);\nif (output) {\n"
      << "for (uint d=0; d<y.size(); ++d) output[ulong(i)*os.column+ulong(d)*os.row]="
      << output_type << "(y[d]);\n"
      << "for (uint d=uint(y.size()); d<output_rows; ++d) "
         "output[ulong(i)*os.column+ulong(d)*os.row]="
      << output_type << "(0);\n"
      << "}\n}\n";
    return s.str();
}

std::string jit_backward_kernel_source(const std::string& name, const std::string& function,
                                       const std::string& input_type,
                                       const std::string& parameter_type,
                                       const std::string& output_type, uint32_t input_width,
                                       uint32_t output_width, uint32_t context_bytes, bool second,
                                       size_t gradient_parameters) {
    jit_validate_identifier(name);
    const uint32_t simd_scratch = jit_simdgroup_scratch_bytes(function);
    const auto gradients_plan = jit_gradient_plan(function, gradient_parameters);
    std::ostringstream s;
    if (gradient_parameters)
        s << "// TCNN_JIT_FLOAT_GRADIENT_BUFFER: " << (second ? 16 : 11) << "\n";
    s << jit_device_prelude() << function << "\nkernel void " << name
      << "(\nconstant uint& n [[buffer(0)]],\n";
    if (!second) {
        s << "const device " << output_type
          << "* dy [[buffer(1)]], constant tcnn_jit_matrix_strides& dys [[buffer(2)]],\n"
          << "device " << input_type
          << "* dx [[buffer(3)]], constant tcnn_jit_matrix_strides& dxs [[buffer(4)]],\n"
          << "const device " << parameter_type
          << "* params [[buffer(5)]], const device uchar* context [[buffer(6)]],\n"
          << "device " << parameter_type
          << "* gradients [[buffer(7)]], const device float* controls [[buffer(8)]],\n"
          << "device uchar* gradient_base [[buffer(9)]], constant uint& present_buffers "
             "[[buffer(10)]],\n";
    } else {
        s << "const device " << input_type
          << "* ddx [[buffer(1)]], constant tcnn_jit_matrix_strides& ddxs [[buffer(2)]],\n"
          << "const device " << output_type
          << "* dy [[buffer(3)]], constant tcnn_jit_matrix_strides& dys [[buffer(4)]],\n"
          << "device " << input_type
          << "* dx [[buffer(5)]], constant tcnn_jit_matrix_strides& dxs [[buffer(6)]],\n"
          << "device " << output_type
          << "* ddy [[buffer(7)]], constant tcnn_jit_matrix_strides& ddys [[buffer(8)]],\n"
          << "const device " << parameter_type
          << "* params [[buffer(9)]], const device uchar* context [[buffer(10)]],\n"
          << "device " << parameter_type
          << "* gradients [[buffer(11)]], const device float* controls [[buffer(12)]],\n"
          << "constant uint& output_rows [[buffer(13)]], device uchar* gradient_base "
             "[[buffer(14)]],\n"
          << "constant uint& present_buffers [[buffer(15)]],\n";
    }
    if (gradient_parameters)
        s << "device float* wide_gradients [[buffer(" << (second ? 16 : 11) << ")]],\n";
    s << "threadgroup uchar* scratch [[threadgroup(0)]], uint3 global_tid "
         "[[thread_position_in_grid]],\n"
      << "uint ti [[thread_index_in_threadgroup]], uint3 group_extent "
         "[[threads_per_threadgroup]],\n"
      << "uint lane [[thread_index_in_simdgroup]], uint simd_width [[threads_per_simdgroup]], uint "
         "simd_group [[simdgroup_index_in_threadgroup]]) {\n"
      << (second ? optional_pointer_prologue({{1, "ddx"},
                                              {3, "dy"},
                                              {5, "dx"},
                                              {7, "ddy"},
                                              {9, "params"},
                                              {10, "context"},
                                              {11, "gradients"},
                                              {12, "controls"},
                                              {14, "gradient_base"}})
                 : optional_pointer_prologue({{1, "dy"},
                                              {3, "dx"},
                                              {5, "params"},
                                              {6, "context"},
                                              {7, "gradients"},
                                              {8, "controls"},
                                              {9, "gradient_base"}}))
      << (gradient_parameters ? optional_pointer_prologue({{second ? 16u : 11u, "wide_gradients"}})
                              : "")
      << "const uint i=global_tid.x-ti+simd_group*simd_width+lane;\nif (i >= n) "
         "return;\ntcnn_jit_array<"
      << output_type << "," << output_width << "> g{};\n"
      << "tcnn_jit_array<" << input_type << "," << input_width << "> result{};\n"
      << "for (uint d=0; d<" << output_width
      << "; ++d) g[d]=dy[ulong(i)*dys.column+ulong(d)*dys.row];\n"
      << "tcnn_jit_environment env{}; env.runtime_inputs=controls; env.index=i; env.count=n;\n"
      << "env.gradient_base=gradient_base; env.shared_memory=scratch; env.thread_index=ti; "
         "env.group_size=group_extent.x*group_extent.y*group_extent.z;\n";
    s << "env.lane_index=lane; "
         "env.lane_count=min(simd_width,env.group_size-simd_group*simd_width);\n";
    s << simd_workspace_prologue(simd_scratch);
    s << float_gradient_prologue(gradients_plan);
    if (!second)
        s << "backward_eval_model(g,params,context ? context+ulong(i-lane)*" << context_bytes
          << " : nullptr,gradients,dx ? &result : nullptr,&env);\n";
    else {
        s << "tcnn_jit_array<" << input_type << "," << input_width << "> v{};\n"
          << "tcnn_jit_array<" << output_type << "," << output_width << "> jv{};\n"
          << "for (uint d=0; d<" << input_width
          << "; ++d) v[d]=ddx[ulong(i)*ddxs.column+ulong(d)*ddxs.row];\n"
          << "backward_backward_input_eval_model(v,g,params,context ? context+ulong(i-lane)*"
          << context_bytes
          << " : nullptr,gradients,dx ? &result : nullptr,ddy ? &jv : nullptr,&env);\n"
          << "if (ddy) { for (uint d=0; d<" << output_width
          << "; ++d) ddy[ulong(i)*ddys.column+ulong(d)*ddys.row]=jv[d];\n"
          << "for (uint d=" << output_width
          << "; d<output_rows; ++d) ddy[ulong(i)*ddys.column+ulong(d)*ddys.row]=" << output_type
          << "(0); }\n";
    }
    s << "if (dx) for (uint d=0; d<" << input_width
      << "; ++d) dx[ulong(i)*dxs.column+ulong(d)*dxs.row]=result[d];\n}\n";
    return s.str();
}

std::unique_ptr<MetalRtcKernel> jit_compile_kernel(const std::string& name,
                                                   const std::string& source, uint32_t kind) {
    auto kernel = std::make_unique<MetalRtcKernel>(name, source);
    const auto simd_scratch = jit_simdgroup_scratch_bytes(source);
    kernel->set_threadgroup_memory_layout(
        simd_scratch,
        source_requirement(source, "TCNN_JIT_THREADGROUP_CONTEXT_BYTES_PER_THREAD", 1),
        source_requirement(source, "TCNN_JIT_MINIMUM_SCRATCH_BYTES", 4));
    kernel->set_allow_null_arguments(true);
    std::vector<metalrt::Argument> defaults;
    if (kind == 0) {
        defaults = {metalrt::Argument::null_buffer(6), metalrt::Argument::null_buffer(7),
                    metalrt::Argument::scalar(8, uint32_t{0})};
        kernel->set_optional_buffer_mask_argument(9);
        if (simd_scratch)
            kernel->set_min_threadgroup_memory_bytes(16);
    } else if (kind == 1) {
        defaults = {metalrt::Argument::null_buffer(8)};
        kernel->set_pointer_base_argument(7, 9);
        kernel->set_optional_buffer_mask_argument(10);
        kernel->set_min_threadgroup_memory_bytes(16);
    } else if (kind == 2) {
        defaults = {metalrt::Argument::null_buffer(12), metalrt::Argument::scalar(13, uint32_t{0})};
        kernel->set_pointer_base_argument(11, 14);
        kernel->set_optional_buffer_mask_argument(15);
        kernel->set_min_threadgroup_memory_bytes(16);
    } else if (kind == 3) {
        struct Strides {
            uint32_t row, column;
        };
        defaults = {metalrt::Argument::null_buffer(17), metalrt::Argument::null_buffer(18),
                    metalrt::Argument::scalar(19, Strides{0, 0})};
        kernel->set_pointer_base_argument(16, 20);
        kernel->set_optional_buffer_mask_argument(21);
        kernel->set_min_threadgroup_memory_bytes(16);
    }
    const auto float_slot = source_requirement(source, "TCNN_JIT_FLOAT_GRADIENT_BUFFER", 1);
    if (float_slot)
        defaults.emplace_back(metalrt::Argument::null_buffer(float_slot));
    kernel->set_default_arguments(std::move(defaults));
    return kernel;
}

std::string jit_training_kernel_source(const std::string& name, const std::string& forward,
                                       const std::string& backward, const std::string& loss,
                                       const std::string& input_type,
                                       const std::string& parameter_type,
                                       const std::string& output_type, uint32_t input_width,
                                       uint32_t output_width, uint32_t context_bytes,
                                       bool context_in_threadgroup, uint32_t backward_scratch_bytes,
                                       bool external_gradient, size_t gradient_parameters) {
    jit_validate_identifier(name);
    const uint32_t simd_scratch =
        std::max(jit_simdgroup_scratch_bytes(forward), jit_simdgroup_scratch_bytes(backward));
    const auto gradients_plan = jit_gradient_plan(forward + backward, gradient_parameters);
    std::ostringstream s;
    if (context_in_threadgroup)
        s << "#define TCNN_JIT_CTX threadgroup\n";
    s << "// TCNN_JIT_MINIMUM_SCRATCH_BYTES: " << backward_scratch_bytes << "\n";
    if (context_in_threadgroup)
        s << "// TCNN_JIT_THREADGROUP_CONTEXT_BYTES_PER_THREAD: " << context_bytes << "\n";
    if (gradient_parameters)
        s << "// TCNN_JIT_FLOAT_GRADIENT_BUFFER: 22\n";
    s << jit_device_prelude() << forward << backward << loss << "\nkernel void " << name << "(\n"
      << "constant uint& n [[buffer(0)]], constant float& loss_scale [[buffer(1)]],\n"
      << "const device " << input_type
      << "* input [[buffer(2)]], constant tcnn_jit_matrix_strides& is [[buffer(3)]],\n"
      << "const device float* target [[buffer(4)]], constant tcnn_jit_matrix_strides& ts "
         "[[buffer(5)]],\n"
      << "const device float* pdf [[buffer(6)]], constant tcnn_jit_matrix_strides& ps "
         "[[buffer(7)]],\n"
      << "const device " << output_type
      << "* external [[buffer(8)]], constant tcnn_jit_matrix_strides& es [[buffer(9)]],\n"
      << "device " << input_type
      << "* dx [[buffer(10)]], constant tcnn_jit_matrix_strides& dxs [[buffer(11)]],\n"
      << "device float* values [[buffer(12)]], constant tcnn_jit_matrix_strides& vs "
         "[[buffer(13)]],\n"
      << "const device " << parameter_type
      << "* params [[buffer(14)]], device uchar* context [[buffer(15)]],\n"
      << "device " << parameter_type
      << "* gradients [[buffer(16)]], const device float* controls [[buffer(17)]],\n"
      << "const device float* perturbation [[buffer(18)]], constant tcnn_jit_matrix_strides& ns "
         "[[buffer(19)]],\n"
      << "device uchar* gradient_base [[buffer(20)]], constant uint& present_buffers "
         "[[buffer(21)]],\n"
      << (gradient_parameters ? "device float* wide_gradients [[buffer(22)]],\n" : "")
      << "threadgroup uchar* scratch [[threadgroup(0)]], uint3 global_tid "
         "[[thread_position_in_grid]],\n"
      << "uint ti [[thread_index_in_threadgroup]], uint3 group_extent "
         "[[threads_per_threadgroup]],\n"
      << "uint lane [[thread_index_in_simdgroup]], uint simd_width [[threads_per_simdgroup]], uint "
         "simd_group [[simdgroup_index_in_threadgroup]]) {\n"
      << optional_pointer_prologue({{2, "input"},
                                    {4, "target"},
                                    {6, "pdf"},
                                    {8, "external"},
                                    {10, "dx"},
                                    {12, "values"},
                                    {14, "params"},
                                    {15, "context"},
                                    {16, "gradients"},
                                    {17, "controls"},
                                    {18, "perturbation"},
                                    {20, "gradient_base"}})
      << (gradient_parameters ? optional_pointer_prologue({{22u, "wide_gradients"}}) : "")
      << "const uint i=global_tid.x-ti+simd_group*simd_width+lane;\nif (i>=n) "
         "return;\ntcnn_jit_environment env{}; env.runtime_inputs=controls; env.index=i; "
         "env.count=n;\n"
      << "env.gradient_base=gradient_base; env.shared_memory=scratch; env.thread_index=ti;\n"
      << "env.group_size=group_extent.x*group_extent.y*group_extent.z;\n"
      << "env.lane_index=lane; "
         "env.lane_count=min(simd_width,env.group_size-simd_group*simd_width);\n"
      << simd_workspace_prologue(simd_scratch) << float_gradient_prologue(gradients_plan)
      << "TCNN_JIT_CTX uchar* fwd_ctx=";
    if (context_in_threadgroup)
        s << "scratch+max(" << backward_scratch_bytes << "u,(env.group_size/32u)*" << simd_scratch
          << "u)+ulong(simd_group)*simd_width*" << context_bytes << ";\n";
    else
        s << "context ? context+ulong(i-lane)*" << context_bytes << " : nullptr;\n";
    s << "tcnn_jit_array<" << input_type << "," << input_width << "> x{};\n"
      << "for(uint d=0;d<" << input_width
      << ";++d) x[d]=input[ulong(i)*is.column+ulong(d)*is.row];\n"
      << "auto y=eval_model(x,params,fwd_ctx,&env);\n"
      << "tcnn_jit_array<" << output_type << "," << output_width << "> dy{};\n"
      << "tcnn_jit_array<float," << output_width << "> loss_value{};\n";
    if (external_gradient)
        s << "for(uint d=0;d<" << output_width
          << ";++d) dy[d]=external[ulong(i)*es.column+ulong(d)*es.row];\n";
    else {
        s << "tcnn_jit_array<float," << output_width << "> t{}, p{};\n"
          << "for(uint d=0;d<" << output_width << ";++d) {\n"
          << "t[d]=target[ulong(i)*ts.column+ulong(d)*ts.row]; p[d]=pdf ? "
             "pdf[ulong(i)*ps.column+ulong(d)*ps.row] : 1.0f;\n"
          << "if(perturbation) y[d]=" << output_type
          << "(float(y[d])+perturbation[ulong(i)*ns.column+ulong(d)*ns.row]);\n}\n"
          << "dy=eval_loss(n*" << output_width << ",loss_scale,y,t,p,&loss_value);\n";
    }
    s << "if(values) for(uint d=0;d<" << output_width
      << ";++d) values[ulong(i)*vs.column+ulong(d)*vs.row]=loss_value[d];\n"
      << "tcnn_jit_array<" << input_type << "," << input_width << "> derivative{};\n"
      << "backward_eval_model(dy,params,fwd_ctx,gradients,dx ? &derivative : nullptr,&env);\n"
      << "if(dx) for(uint d=0;d<" << input_width
      << ";++d) dx[ulong(i)*dxs.column+ulong(d)*dxs.row]=derivative[d];\n}\n";
    return s.str();
}

GPUMemoryArena::Allocation jit_collect_runtime_inputs(metalStream_t stream,
                                                      const std::vector<const float*>& inputs,
                                                      uint32_t batch) {
    const auto slot_bytes = metalrt::checked_mul(batch, sizeof(float));
    auto allocation = allocate_workspace(stream, metalrt::checked_mul(slot_bytes, inputs.size()));
    for (size_t j = 0; j < inputs.size(); ++j) {
        if (!inputs[j])
            throw std::invalid_argument("Null JIT runtime input");
        if (slot_bytes)
            METAL_CHECK_THROW(metalMemcpyAsync(allocation.data() + j * slot_bytes, inputs[j],
                                               slot_bytes, metalMemcpyDeviceToDevice, stream));
    }
    return allocation;
}

GPUMemoryArena::Allocation jit_prepare_gradients(metalStream_t stream, const void* gradients,
                                                 size_t bytes, bool accumulate) {
    if (!bytes)
        return {};
    const auto padded = metalrt::checked_add(bytes, size_t{3}) & ~size_t{3};
    auto scratch = allocate_workspace(stream, padded);
    METAL_CHECK_THROW(metalMemsetAsync(scratch.data(), 0, padded, stream));
    if (accumulate)
        METAL_CHECK_THROW(
            metalMemcpyAsync(scratch.data(), gradients, bytes, metalMemcpyDeviceToDevice, stream));
    return scratch;
}
void jit_finish_gradients(metalStream_t stream, void* gradients,
                          const GPUMemoryArena::Allocation& scratch, size_t bytes) {
    if (bytes)
        METAL_CHECK_THROW(
            metalMemcpyAsync(gradients, scratch.data(), bytes, metalMemcpyDeviceToDevice, stream));
}
} // namespace detail
} // namespace tcnn

namespace tcnn {
std::unique_ptr<MetalRtcKernel> generate_kernel(const std::string& name,
                                                const std::string& function, const std::string& t,
                                                const std::string& p, const std::string& c,
                                                uint32_t in, uint32_t bytes) {
    return detail::jit_compile_kernel(
        name, detail::jit_forward_kernel_source(name, function, t, p, c, in, bytes), 0);
}
std::unique_ptr<MetalRtcKernel> generate_backward_kernel(const std::string& name,
                                                         const std::string& function,
                                                         const std::string& t, const std::string& p,
                                                         const std::string& c, uint32_t in,
                                                         uint32_t out, uint32_t bytes) {
    return detail::jit_compile_kernel(
        name, detail::jit_backward_kernel_source(name, function, t, p, c, in, out, bytes), 1);
}
std::unique_ptr<MetalRtcKernel> generate_backward_backward_input_kernel(
    const std::string& name, const std::string& function, const std::string& t,
    const std::string& p, const std::string& c, uint32_t in, uint32_t out, uint32_t bytes) {
    return detail::jit_compile_kernel(
        name, detail::jit_backward_kernel_source(name, function, t, p, c, in, out, bytes, true), 2);
}
} // namespace tcnn
