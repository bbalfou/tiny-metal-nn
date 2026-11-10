#include <tiny-metal-nn/detail/fused_mlp_dispatch.h>
#include <tiny-metal-nn/detail/shaders.h>
#include <tiny-metal-nn/metal.h>
#include <algorithm>
#include <map>
#include <mutex>
#include <sstream>
#include <type_traits>

// The Reference backend uses the same scalar precision helpers as the existing
// layerwise C++ implementation. The native backend compiles the Metal helpers.
namespace tcnn {
namespace detail {
float read(const void*, size_t, uint32_t);
void write(void*, size_t, uint32_t, float);
float round_to_precision(float, uint32_t);
float activate(float, uint32_t, uint32_t);
float derivative(float, uint32_t);
float derivative_output(float, uint32_t, uint32_t);
float activation_gradient(float, float, uint32_t, uint32_t, bool);
float second_derivative(float, float, uint32_t, uint32_t);
} // namespace detail
} // namespace tcnn

#include "../kernels/networks/fused_mlp_math.metal"
#include "../kernels/networks/fused/training_traits.metal"
#include "../kernels/networks/fused/training_rectangular_traits.metal"
#include "../kernels/networks/fused/higher_order_traits.metal"
#include "../kernels/networks/fused/parameter_traits.metal"
#include "../kernels/networks/fused/inference_traits.metal"

namespace tcnn {
namespace detail {
namespace {
using network_kernels::FusedArgs;
using network_kernels::FusedShape;
using network_kernels::Mat;
using Traits = network_kernels::FullyFusedMLPTraits;
using ReferenceTraits = network_kernels::FusedReferenceTraits;
using InferenceTraits = network_kernels::FusedInferenceTraits;
using TrainingTraits = network_kernels::FusedTrainingTraits;
using RectangularTrainingTraits = network_kernels::FusedRectangularTrainingTraits;
using HigherOrderTraits = network_kernels::FusedHigherOrderTraits;
using ParameterTraits = network_kernels::FusedParameterGradientTraits;

bool fused_launch_fits(const metalrt::Device& device, const metalrt::Kernel& kernel,
                       uint32_t threads) {
    // Instrumentation and compiler staging can add storage beyond the tuning
    // traits' scratch arrays. Check the compiled pipeline before submitting.
    const auto caps = device.capabilities();
    return threads <= kernel.max_threads_per_threadgroup() &&
           kernel.static_threadgroup_memory_bytes() <= caps.max_threadgroup_memory_bytes;
}

template <uint32_t Width, uint32_t Precision, bool Backward, bool Rectangular = false>
FusedTrainingConfig training_config() {
    using Tuning =
        typename std::conditional<Rectangular,
                                  tcnn_fused::RectangularTuning<Width, Precision, Backward>,
                                  tcnn_fused::TrainingTuning<Width, Precision, Backward>>::type;
    using Shape = typename Tuning::Shape;
    using Scratch = typename Tuning::Scratch;
    return {Shape::Samples,   Shape::Threads,      Shape::KPanel,
            Shape::RowGroups, Shape::ColumnGroups, Scratch::Bytes};
}

template <uint32_t Precision, bool Backward, bool Rectangular = false>
FusedTrainingConfig training_config(uint32_t width) {
    switch (width) {
    case 16:
        return training_config<16, Precision, Backward, Rectangular>();
    case 32:
        return training_config<32, Precision, Backward, Rectangular>();
    case 64:
        return training_config<64, Precision, Backward, Rectangular>();
    case 128:
        return training_config<128, Precision, Backward, Rectangular>();
    default:
        return {};
    }
}

template <uint32_t Width, uint32_t Precision> FusedTrainingConfig higher_order_config() {
    using Tuning = tcnn_fused::HigherOrderTuning<Width, Precision>;
    return {Tuning::Samples,   Tuning::Threads,      Tuning::KPanel,
            Tuning::RowGroups, Tuning::ColumnGroups, Tuning::ScratchBytes};
}

template <uint32_t Precision> FusedTrainingConfig higher_order_config(uint32_t width) {
    switch (width) {
    case 16:
        return higher_order_config<16, Precision>();
    case 32:
        return higher_order_config<32, Precision>();
    case 64:
        return higher_order_config<64, Precision>();
    case 128:
        return higher_order_config<128, Precision>();
    default:
        return {};
    }
}

bool training_matrix_supported(const Mat& m, uint32_t width, uint32_t batch, uint32_t precision) {
    if (!width || !batch || m.rows != width || m.cols != batch || m.precision != precision ||
        !m.rs || !m.cs)
        return false;
    const uint64_t span = uint64_t(width - 1) * m.rs + uint64_t(batch - 1) * m.cs + 1;
    return span <= UINT32_MAX;
}

bool training_aligned(const void* pointer) {
    return pointer && reinterpret_cast<uintptr_t>(pointer) % 16 == 0;
}

metalrt::Kernel fused_training_kernel(const metalrt::Device& device, FusedShape shape,
                                      uint32_t precision, FusedTrainingPass pass,
                                      bool rectangular = false) {
    struct Cache {
        metalrt::Device device;
        uint32_t width, precision, activation, output_activation;
        uint32_t saved_inputs, saved_outputs;
        FusedTrainingPass pass;
        bool rectangular;
        metalrt::Kernel kernel;
    };
    const bool saved_forward = rectangular && pass == FusedTrainingPass::Forward &&
        precision == 2 && shape.hidden_layers && (shape.width == 64 || shape.width == 128) &&
        shape.inputs % 16 == 0 && shape.outputs % 16 == 0 && shape.outputs <= shape.width;
    const uint32_t saved_inputs = saved_forward ? shape.inputs : 0;
    const uint32_t saved_outputs = saved_forward ? shape.outputs : 0;
    static std::mutex mutex;
    static std::vector<Cache> caches;
    std::lock_guard<std::mutex> lock{mutex};
    const auto found = std::find_if(caches.begin(), caches.end(), [&](const auto& item) {
        return item.device == device && item.width == shape.width && item.precision == precision &&
               item.activation == shape.activation &&
               item.output_activation == shape.output_activation && item.pass == pass &&
               item.rectangular == rectangular && item.saved_inputs == saved_inputs &&
               item.saved_outputs == saved_outputs;
    });
    if (found != caches.end())
        return found->kernel;
    // Only policy/shape constants vary. All arithmetic and synchronization live
    // in reusable C++14 Metal templates; depth remains a runtime argument.
    std::ostringstream prefix;
    prefix << "#define TCNN_FUSED_TRAIN_WIDTH " << shape.width << "\n"
           << "#define TCNN_FUSED_TRAIN_SCALAR " << (precision == 2 ? "half" : "float") << "\n"
           << "#define TCNN_FUSED_TRAIN_ACTIVATION " << shape.activation << "\n"
           << "#define TCNN_FUSED_TRAIN_OUTPUT_ACTIVATION " << shape.output_activation << "\n";
    prefix << "#define TCNN_SAVED_TRAIN_INPUTS " << saved_inputs << "\n"
           << "#define TCNN_SAVED_TRAIN_OUTPUTS " << saved_outputs << "\n";
    const char* source = rectangular ? RectangularTrainingTraits::source : TrainingTraits::source;
    const char* entry =
        pass == FusedTrainingPass::Forward
            ? (rectangular ? RectangularTrainingTraits::forward : TrainingTraits::forward)
            : (rectangular ? RectangularTrainingTraits::backward : TrainingTraits::backward);
    if (pass == FusedTrainingPass::Tangent || pass == FusedTrainingPass::SecondBackward) {
        source = HigherOrderTraits::source;
        entry = pass == FusedTrainingPass::Tangent ? HigherOrderTraits::tangent
                                                   : HigherOrderTraits::second_backward;
    }
    auto kernel = device.kernel(prefix.str() + shader_source(source), entry);
    caches.push_back({device, shape.width, precision, shape.activation, shape.output_activation,
                      saved_inputs, saved_outputs, pass, rectangular, kernel});
    return kernel;
}

metalrt::Kernel fused_parameter_kernel(const metalrt::Device& device, uint32_t width,
                                       uint32_t precision, const char* entry,
                                       bool generic = false) {
    struct Cache {
        metalrt::Device device;
        uint32_t width, precision;
        std::string entry;
        bool generic;
        metalrt::Kernel kernel;
    };
    static std::mutex mutex;
    static std::vector<Cache> caches;
    std::lock_guard<std::mutex> lock{mutex};
    const auto found = std::find_if(caches.begin(), caches.end(), [&](const Cache& item) {
        return item.device == device && item.width == width && item.precision == precision &&
               item.entry == entry && item.generic == generic;
    });
    if (found != caches.end())
        return found->kernel;
    std::ostringstream prefix;
    prefix << "#define TCNN_FUSED_TRAIN_WIDTH " << width << "\n"
           << "#define TCNN_FUSED_TRAIN_SCALAR " << (precision == 2 ? "half" : "float") << "\n";
    auto kernel =
        device.kernel(prefix.str() + shader_source(generic ? ParameterTraits::generic_source
                                                           : ParameterTraits::source),
                      entry);
    caches.push_back({device, width, precision, entry, generic, kernel});
    return kernel;
}

bool ranges_overlap(uintptr_t a, size_t a_bytes, uintptr_t b, size_t b_bytes) {
    // Subtraction avoids address wraparound in this eligibility-only query.
    return a <= b ? b - a < a_bytes : a - b < b_bytes;
}

template <uint32_t Width, uint32_t Precision>
FusedInferenceConfig inference_config(FusedShape shape) {
    using P = tcnn_fused::InferenceProfile<Width, Precision>;
    return {P::Samples,
            P::Threads,
            P::KPanel,
            P::RowGroups,
            P::ColumnGroups,
            P::output_rows(shape.outputs),
            P::scratch_elements(shape.inputs) * Precision,
            P::Private ? uint32_t(std::min<uint64_t>(6, uint64_t(shape.hidden_layers) + 1)) : 0u,
            P::Private != 0,
            P::DirectWeights != 0,
            false};
}

template <uint32_t Precision> FusedInferenceConfig inference_config(FusedShape shape) {
    switch (shape.width) {
    case 16:
        return inference_config<16, Precision>(shape);
    case 32:
        return inference_config<32, Precision>(shape);
    case 64:
        return inference_config<64, Precision>(shape);
    case 128:
        return inference_config<128, Precision>(shape);
    default:
        return {};
    }
}

template <uint32_t Width, uint32_t Precision> FusedInferenceConfig boundary_inference_config() {
    using P = tcnn_fused::BoundaryInferenceTuning<Width, Precision>;
    return {P::Samples,   P::Threads,      P::KPanel, P::RowGroups, P::ColumnGroups,
            P::RowGroups, P::ScratchBytes, 0,         false,        false,
            true};
}

template <uint32_t Precision> FusedInferenceConfig boundary_inference_config(uint32_t width) {
    switch (width) {
    case 16:
        return boundary_inference_config<16, Precision>();
    case 32:
        return boundary_inference_config<32, Precision>();
    case 64:
        return boundary_inference_config<64, Precision>();
    case 128:
        return boundary_inference_config<128, Precision>();
    default:
        return {};
    }
}

bool inference_needs_boundary(FusedShape shape, const Mat& input, const Mat& output,
                              uintptr_t input_address, uintptr_t weights_address,
                              uintptr_t output_address) {
    const auto config = fused_inference_config(shape, input.precision);
    const auto contiguous_axis = [](const Mat& m) {
        return (m.rs == 1 && m.cs >= m.rows) || (m.cs == 1 && m.rs >= m.cols);
    };
    return config.boundary_tiles || !config.samples_per_threadgroup ||
           input.cols % config.samples_per_threadgroup || !contiguous_axis(input) ||
           !contiguous_axis(output) || input_address % 16 || weights_address % 16 ||
           output_address % 16;
}

metalrt::Kernel fused_inference_kernel(const metalrt::Device& device, FusedShape shape,
                                       uint32_t precision, bool boundary) {
    struct Cache {
        metalrt::Device device;
        uint32_t width, inputs, outputs, precision, activation, output_activation, preload;
        bool boundary;
        metalrt::Kernel kernel;
    };
    const auto config = fused_inference_config(shape, precision, boundary);
    const uint32_t preload = std::max(config.preloaded_layers, 1u);
    static std::mutex mutex;
    static std::vector<Cache> caches;
    std::lock_guard<std::mutex> lock{mutex};
    const auto found = std::find_if(caches.begin(), caches.end(), [&](const Cache& c) {
        return c.device == device && c.width == shape.width && c.inputs == shape.inputs &&
               c.outputs == shape.outputs && c.precision == precision &&
               c.activation == shape.activation && c.output_activation == shape.output_activation &&
               c.preload == preload && c.boundary == config.boundary_tiles;
    });
    if (found != caches.end())
        return found->kernel;
    std::ostringstream prefix;
    prefix << "#define TCNN_FUSED_INFERENCE_WIDTH " << shape.width << "\n"
           << "#define TCNN_FUSED_INFERENCE_INPUTS " << shape.inputs << "\n"
           << "#define TCNN_FUSED_INFERENCE_OUTPUTS " << shape.outputs << "\n"
           << "#define TCNN_FUSED_INFERENCE_SCALAR " << (precision == 2 ? "half" : "float") << "\n"
           << "#define TCNN_FUSED_INFERENCE_ACTIVATION " << shape.activation << "\n"
           << "#define TCNN_FUSED_INFERENCE_OUTPUT_ACTIVATION " << shape.output_activation << "\n"
           << "#define TCNN_FUSED_INFERENCE_PRELOAD " << preload << "\n"
           << "#define TCNN_FUSED_INFERENCE_THREADS " << config.threads << "\n"
           << "#define TCNN_FUSED_INFERENCE_BOUNDARY " << config.boundary_tiles << "\n";
    auto kernel = device.kernel(prefix.str() + shader_source(InferenceTraits::source),
                                InferenceTraits::entry);
    caches.push_back({device, shape.width, shape.inputs, shape.outputs, precision, shape.activation,
                      shape.output_activation, preload, config.boundary_tiles, kernel});
    return kernel;
}

template <class T> Mat fused_matrix(const GPUMatrixDynamic<T>& matrix) {
    const auto view = matrix.view();
    const uint64_t span = matrix.m() && matrix.n()
                              ? uint64_t(matrix.m() - 1) * view.stride_i +
                                    uint64_t(matrix.n() - 1) * view.stride_j + 1
                              : 0;
    nn_check(span <= UINT32_MAX && matrix.n_elements() <= UINT32_MAX,
             "fused MLP matrix exceeds 32-bit indexing");
    return {matrix.m(), matrix.n(), view.stride_i, view.stride_j, sizeof(T)};
}

template <class T>
metalrt::Argument fused_raw(uint32_t slot, const T* pointer, size_t elements,
                            metalrt::Access access) {
    nn_check(elements <= UINT32_MAX, "fused MLP allocation exceeds 32-bit indexing");
    return metalrt::Argument::buffer(
        slot, metalrt::resolve_pointer(pointer, metalrt::checked_mul(elements, sizeof(T))), access);
}

template <class T>
metalrt::Argument fused_matrix_arg(uint32_t slot, const GPUMatrixDynamic<T>& matrix,
                                   metalrt::Access access) {
    const auto m = fused_matrix(matrix);
    const size_t span =
        m.rows && m.cols ? size_t(m.rows - 1) * m.rs + size_t(m.cols - 1) * m.cs + 1 : 0;
    return fused_raw(slot, matrix.data(), span, access);
}

template <class T> size_t fused_matrix_span_bytes(const GPUMatrixDynamic<T>& matrix) {
    const auto m = fused_matrix(matrix);
    const size_t elements =
        m.rows && m.cols ? size_t(m.rows - 1) * m.rs + size_t(m.cols - 1) * m.cs + 1 : 0;
    return metalrt::checked_mul(elements, sizeof(T));
}

// Direct dX output is safe only when it cannot overwrite an operand still
// needed by backward propagation or the following parameter reduction. Use
// the complete strided bounding span, conservatively including padding.
template <class T>
bool fused_dx_needs_staging(const GPUMatrixDynamic<T>& dx, const GPUMatrixDynamic<T>& input,
                            const GPUMatrixDynamic<T>& output, const GPUMatrixDynamic<T>& incoming,
                            const T* weights, T* gradients, size_t n_params,
                            const GPUMatrixDynamic<T>& pre, const GPUMatrixDynamic<T>& hidden) {
    const auto address = reinterpret_cast<uintptr_t>(dx.data());
    const auto bytes = fused_matrix_span_bytes(dx);
    const auto overlaps = [&](const void* pointer, size_t other_bytes) {
        return bytes && pointer && other_bytes &&
               ranges_overlap(address, bytes, reinterpret_cast<uintptr_t>(pointer), other_bytes);
    };
    const auto overlaps_matrix = [&](const GPUMatrixDynamic<T>& matrix) {
        return overlaps(matrix.data(), fused_matrix_span_bytes(matrix));
    };
    const auto parameter_bytes = metalrt::checked_mul(n_params, sizeof(T));
    return overlaps_matrix(input) || overlaps_matrix(output) || overlaps_matrix(incoming) ||
           overlaps_matrix(pre) || overlaps_matrix(hidden) || overlaps(weights, parameter_bytes) ||
           overlaps(gradients, parameter_bytes);
}

metalrt::CpuKernel fused_reference(const std::string& name) {
    return [name](const metalrt::ComputeCommand& command) {
        using namespace fused_math;
        auto r = [&](uint32_t index) { return command.buffer<const uint8_t>(index); };
        auto w = [&](uint32_t index) { return command.buffer<uint8_t>(index); };
        const uint32_t constant_slot =
            name == Traits::forward || name == ReferenceTraits::parameter_gradient ? 4
            : name == Traits::second_backward                                      ? 8
                                                                                   : 7;
        const auto a = command.scalar<FusedArgs>(constant_slot);
        if (name == Traits::forward)
            for (uint32_t col = 0; col < a.x.cols; ++col)
                fused_forward_sample(a, r(0), r(1), w(2), w(3), col);
        else if (name == Traits::backward)
            for (uint32_t col = 0; col < a.x.cols; ++col)
                fused_backward_sample(a, r(0), r(1), r(2), r(3), r(4), w(5), w(6), col);
        else if (name == Traits::tangent)
            for (uint32_t col = 0; col < a.x.cols; ++col)
                fused_tangent_sample(a, r(0), r(1), r(2), r(3), w(4), w(5), w(6), col);
        else if (name == Traits::second_backward)
            for (uint32_t col = 0; col < a.x.cols; ++col)
                fused_second_backward_sample(a, r(0), r(1), r(2), r(3), r(4), w(5), w(6), w(7),
                                             col);
        else if (name == ReferenceTraits::parameter_gradient ||
                 name == ReferenceTraits::second_parameter_gradient) {
            const uint32_t count = fused_weight_offset(a.shape, a.shape.hidden_layers) +
                                   fused_rows(a.shape, a.shape.hidden_layers) *
                                       fused_inputs(a.shape, a.shape.hidden_layers);
            for (uint32_t p = 0; p < count; ++p)
                if (name == ReferenceTraits::parameter_gradient)
                    fused_parameter_reference(a, r(0), r(1), r(2), w(3), p);
                else
                    fused_second_parameter_reference(a, r(0), r(1), r(2), r(3), r(4), r(5), w(6),
                                                     p);
        } else
            throw std::logic_error("Unknown fused MLP kernel: " + name);
    };
}

metalrt::Kernel fused_kernel(metalStream_t stream, const std::string& name) {
    struct Cache {
        metalrt::Device device;
        std::map<std::string, metalrt::Kernel> kernels;
    };

    static std::mutex mutex;
    static std::vector<Cache> caches;
    auto device = resolve_stream(stream)->device();
    nn_check(device.capabilities().backend == metalrt::BackendKind::Reference ||
                 (name != ReferenceTraits::parameter_gradient &&
                  name != ReferenceTraits::second_parameter_gradient),
             "scalar parameter callbacks are Reference-only");
    std::lock_guard<std::mutex> lock{mutex};
    auto it = std::find_if(caches.begin(), caches.end(),
                           [&](const auto& cache) { return cache.device == device; });
    if (it == caches.end()) {
        caches.push_back({device, {}});
        it = std::prev(caches.end());
    }
    auto kernel = it->kernels.find(name);
    if (kernel == it->kernels.end())
        kernel = it->kernels
                     .emplace(name, device.kernel(shader_source(Traits::source), name,
                                                  fused_reference(name)))
                     .first;
    return kernel->second;
}

void fused_dispatch(metalStream_t stream, const char* entry, size_t count,
                    std::vector<metalrt::Argument> arguments) {
    nn_check(count <= UINT32_MAX, "fused MLP dispatch exceeds 32-bit indexing");
    if (count)
        metalrt::launch_linear(resolve_stream(stream), fused_kernel(stream, entry), count,
                               std::move(arguments));
}

void fused_training_dispatch(metalStream_t stream, FusedTrainingPass pass, const FusedArgs& a,
                             size_t n_params, bool aligned,
                             std::vector<metalrt::Argument> arguments) {
    auto* queue = resolve_stream(stream);
    const auto device = queue->device();
    const bool backward_matrices =
        pass == FusedTrainingPass::Forward ||
        (training_matrix_supported(a.output, a.shape.outputs, a.x.cols, a.x.precision) &&
         training_matrix_supported(a.incoming, a.shape.outputs, a.x.cols, a.x.precision) &&
         (!a.write_dx || training_matrix_supported(a.dx, a.shape.inputs, a.x.cols, a.x.precision)));
    const bool eligible =
        backward_matrices && fused_training_eligible(device.capabilities(), a.x, a.shape, n_params);
    if (eligible) {
        // The general boundary collective also handles scalar-aligned views.
        // It keeps MMA arithmetic even when vector-copy preconditions fail.
        const bool rectangular =
            !aligned || a.shape.inputs != a.shape.width || a.shape.outputs != a.shape.width;
        const auto config =
            (rectangular ? fused_rectangular_training_config(a.shape.width, a.x.precision, pass)
                         : fused_training_config(a.shape.width, a.x.precision, pass));
        const auto kernel =
            fused_training_kernel(device, a.shape, a.x.precision, pass, rectangular);
        if (fused_launch_fits(device, kernel, config.threads)) {
            const uint64_t groups = (uint64_t(a.x.cols) + config.samples_per_threadgroup - 1) /
                                    config.samples_per_threadgroup;
            const uint64_t threads = groups * config.threads;
            nn_check(threads <= UINT32_MAX, "fused training dispatch exceeds 32-bit indexing");
            queue->dispatch(kernel, {uint32_t(threads), 1, 1}, {config.threads, 1, 1},
                            std::move(arguments));
            return;
        }
    }
    fused_dispatch(stream, pass == FusedTrainingPass::Forward ? Traits::forward : Traits::backward,
                   a.x.cols, std::move(arguments));
}

void fused_higher_order_dispatch(metalStream_t stream, FusedTrainingPass pass, const FusedArgs& a,
                                 size_t n_params, std::vector<metalrt::Argument> arguments) {
    auto* queue = resolve_stream(stream);
    const auto device = queue->device();
    const bool matrices =
        training_matrix_supported(a.direction, a.shape.inputs, a.x.cols, a.x.precision) &&
        training_matrix_supported(a.incoming, a.shape.outputs, a.x.cols, a.x.precision) &&
        (!a.write_dx || training_matrix_supported(a.dx, a.shape.inputs, a.x.cols, a.x.precision)) &&
        (!a.write_ddy ||
         training_matrix_supported(a.ddy, a.shape.outputs, a.x.cols, a.x.precision));
    if (matrices && fused_training_eligible(device.capabilities(), a.x, a.shape, n_params)) {
        const auto config = fused_higher_order_config(a.shape.width, a.x.precision);
        const auto kernel = fused_training_kernel(device, a.shape, a.x.precision, pass);
        if (fused_launch_fits(device, kernel, config.threads)) {
            const uint64_t groups = (uint64_t(a.x.cols) + config.samples_per_threadgroup - 1) /
                                    config.samples_per_threadgroup;
            nn_check(groups * config.threads <= UINT32_MAX,
                     "fused higher-order dispatch exceeds 32-bit indexing");
            queue->dispatch(kernel, {uint32_t(groups * config.threads), 1, 1},
                            {config.threads, 1, 1}, std::move(arguments));
            return;
        }
    }
    fused_dispatch(stream,
                   pass == FusedTrainingPass::Tangent ? Traits::tangent : Traits::second_backward,
                   a.x.cols, std::move(arguments));
}

struct ParameterLaunch {
    uint64_t tiles;
    uint32_t threads, reduction_threads, reduction_vector, tile_rows, tile_columns, k_panel,
        scratch_bytes;
};

template <uint32_t Width, uint32_t Precision>
ParameterLaunch parameter_config_for_width(FusedShape shape) {
    using S = tcnn_fused::ParameterTuning<Width, Precision>;
    return {tcnn_fused::parameter_tiles<S>(shape.inputs, shape.width, shape.outputs,
                                           shape.hidden_layers),
            S::Threads,
            S::ReductionThreads,
            S::ReductionVector,
            S::TileRows,
            S::TileColumns,
            S::KPanel,
            S::ScratchBytes};
}

template <uint32_t Precision> ParameterLaunch parameter_config(FusedShape shape) {
    switch (shape.width) {
    case 16:
        return parameter_config_for_width<16, Precision>(shape);
    case 32:
        return parameter_config_for_width<32, Precision>(shape);
    case 64:
        return parameter_config_for_width<64, Precision>(shape);
    case 128:
        return parameter_config_for_width<128, Precision>(shape);
    default:
        return {};
    }
}

void fused_parameter_dispatch(metalStream_t stream, bool second, const FusedArgs& a,
                              size_t n_params, uint32_t vector_mask,
                              std::vector<metalrt::Argument> arguments) {
    auto* queue = resolve_stream(stream);
    const auto device = queue->device();
    const bool eligible = fused_training_eligible(device.capabilities(), a.x, a.shape, n_params) &&
                          (!second || training_matrix_supported(a.direction, a.shape.inputs,
                                                                a.x.cols, a.x.precision));
    if (device.capabilities().backend == metalrt::BackendKind::Reference) {
        fused_dispatch(stream,
                       second ? ReferenceTraits::second_parameter_gradient
                              : ReferenceTraits::parameter_gradient,
                       n_params, std::move(arguments));
        return;
    }
    bool generic = !eligible;
    metalrt::Kernel product;
    if (!generic) {
        product = fused_parameter_kernel(device, a.shape.width, a.x.precision,
                                         second ? ParameterTraits::second : ParameterTraits::first,
                                         false);
        const auto tuned =
            a.x.precision == 2 ? parameter_config<2>(a.shape) : parameter_config<4>(a.shape);
        generic = !fused_launch_fits(device, product, tuned.threads);
    }
    const auto config = generic              ? ParameterLaunch{n_params, 256, 256, 4, 1, 1, 32, 0}
                        : a.x.precision == 2 ? parameter_config<2>(a.shape)
                                             : parameter_config<4>(a.shape);
    nn_check(config.tiles && config.tiles <= UINT32_MAX,
             "fused parameter tile count exceeds 32-bit indexing");
    const uint32_t count = uint32_t(n_params);
    const uint32_t splits = tcnn_fused::parameter_splits(a.x.cols, count, second);
    const size_t elements = tcnn_fused::parameter_partial_elements(count, splits, second);
    // Buffer arguments retain this private scratch allocation through deferred
    // execution and graph replay. No host synchronization or persistent weight
    // packing is introduced, and the allocation uses the actual stream device.
    const auto partials = device.allocate(metalrt::checked_mul(elements, sizeof(float)),
                                          metalrt::StorageMode::Private);
    const uint32_t partial_slot =
        second ? ParameterTraits::second_partials : ParameterTraits::first_partials;
    const uint32_t metadata_slot =
        second ? ParameterTraits::second_dispatch : ParameterTraits::first_dispatch;
    const auto gradient =
        std::find_if(arguments.begin(), arguments.end(), [&](const metalrt::Argument& argument) {
            return argument.index == partial_slot;
        });
    nn_check(gradient != arguments.end(), "missing fused parameter-gradient binding");
    const auto destination = gradient->resource;
    *gradient = metalrt::Argument::buffer(partial_slot, partials.view(), metalrt::Access::Write);
    const char* product_entry =
        generic ? (second ? ParameterTraits::generic_second : ParameterTraits::generic_first)
                : (second ? ParameterTraits::second : ParameterTraits::first);
    if (generic)
        product = fused_parameter_kernel(device, a.shape.width, a.x.precision, product_entry, true);
    const uint32_t maximum_tiles =
        generic ? UINT32_MAX - (config.threads - 1) : UINT32_MAX / config.threads;
    for (uint64_t tile = 0; tile < config.tiles; tile += maximum_tiles) {
        const uint32_t tiles = uint32_t(std::min<uint64_t>(config.tiles - tile, maximum_tiles));
        tcnn_fused::ParameterDispatch metadata{count, splits, uint32_t(config.tiles), vector_mask,
                                               uint32_t(tile)};
        auto launch_arguments = arguments;
        launch_arguments.push_back(metalrt::Argument::scalar(metadata_slot, metadata));
        queue->dispatch(product, {generic ? tiles : tiles * config.threads, splits, 1},
                        {config.threads, 1, 1}, std::move(launch_arguments));
    }
    const uint32_t vectors =
        uint32_t((uint64_t(count) + config.reduction_vector - 1) / config.reduction_vector);
    const tcnn_fused::ParameterReduceArguments reduction{count, splits, a.mode, second ? 1u : 0u};
    queue->dispatch(
        fused_parameter_kernel(device, a.shape.width, a.x.precision,
                               generic ? ParameterTraits::generic_reduce : ParameterTraits::reduce,
                               generic),
        {vectors, 1, 1}, {config.reduction_threads, 1, 1},
        {metalrt::Argument::buffer(0, partials.view(), metalrt::Access::Read),
         metalrt::Argument::buffer(1, destination,
                                   a.mode == uint32_t(GradientMode::Accumulate)
                                       ? metalrt::Access::ReadWrite
                                       : metalrt::Access::Write),
         metalrt::Argument::scalar(2, reduction)});
}

template <class T>
bool empty_gradient(metalStream_t stream, uint32_t batch, T* gradients, size_t n_params,
                    GradientMode mode) {
    if (batch)
        return false;
    if (mode == GradientMode::Overwrite)
        resolve_stream(stream)->fill(
            metalrt::resolve_pointer(gradients, metalrt::checked_mul(n_params, sizeof(T))), 0);
    return true;
}
} // namespace

FusedTrainingConfig fused_training_config(uint32_t width, uint32_t precision,
                                          FusedTrainingPass pass) {
    if (pass == FusedTrainingPass::Forward) {
        if (precision == 2)
            return training_config<2, false>(width);
        if (precision == 4)
            return training_config<4, false>(width);
    } else if (pass == FusedTrainingPass::Backward) {
        if (precision == 2)
            return training_config<2, true>(width);
        if (precision == 4)
            return training_config<4, true>(width);
    }
    return {};
}

FusedTrainingConfig fused_rectangular_training_config(uint32_t width, uint32_t precision,
                                                      FusedTrainingPass pass) {
    if (pass == FusedTrainingPass::Forward) {
        if (precision == 2)
            return training_config<2, false, true>(width);
        if (precision == 4)
            return training_config<4, false, true>(width);
    } else if (pass == FusedTrainingPass::Backward) {
        if (precision == 2)
            return training_config<2, true, true>(width);
        if (precision == 4)
            return training_config<4, true, true>(width);
    }
    return {};
}

FusedParameterConfig fused_parameter_config(FusedShape shape, uint32_t precision, uint32_t batch) {
    if ((precision != 2 && precision != 4) || !Traits::valid_width(shape.width) || !shape.inputs ||
        !shape.outputs)
        return {};
    const auto config = precision == 2 ? parameter_config<2>(shape) : parameter_config<4>(shape);
    const uint64_t parameters =
        shape.hidden_layers ? uint64_t(shape.inputs) * shape.width +
                                  uint64_t(shape.hidden_layers - 1) * shape.width * shape.width +
                                  uint64_t(shape.outputs) * shape.width
                            : uint64_t(shape.inputs) * shape.outputs;
    if (parameters > UINT32_MAX)
        return {};
    return {config.tile_rows,
            config.tile_columns,
            config.k_panel,
            config.threads,
            tcnn_fused::parameter_splits(batch, uint32_t(parameters), false),
            config.scratch_bytes};
}

FusedTrainingConfig fused_higher_order_config(uint32_t width, uint32_t precision) {
    if (precision == 2)
        return higher_order_config<2>(width);
    if (precision == 4)
        return higher_order_config<4>(width);
    return {};
}

bool fused_training_eligible(const metalrt::Capabilities& capabilities,
                             const network_kernels::Mat& input, network_kernels::FusedShape shape,
                             size_t n_params) {
    if (capabilities.backend != metalrt::BackendKind::Metal || capabilities.apple_gpu_family < 9 ||
        !Traits::valid_width(shape.width) || !shape.inputs || !shape.outputs ||
        shape.activation > uint32_t(Activation::None) ||
        shape.output_activation > uint32_t(Activation::None) ||
        (input.precision != 2 && input.precision != 4) || !input.cols)
        return false;
    if (!training_matrix_supported(input, shape.inputs, input.cols, input.precision))
        return false;
    const uint64_t parameters =
        shape.hidden_layers ? uint64_t(shape.inputs) * shape.width +
                                  uint64_t(shape.hidden_layers - 1) * shape.width * shape.width +
                                  uint64_t(shape.outputs) * shape.width
                            : uint64_t(shape.inputs) * shape.outputs;
    // Check the product without overflow. Device indexing counts elements;
    // byte ranges are resolved independently using checked size_t arithmetic.
    const uint64_t context_width = uint64_t(shape.hidden_layers) * shape.width + shape.outputs;
    if (parameters > UINT32_MAX || n_params != parameters ||
        context_width > UINT32_MAX / input.cols)
        return false;
    for (const auto pass : {FusedTrainingPass::Forward, FusedTrainingPass::Backward}) {
        const auto config = fused_training_config(shape.width, input.precision, pass);
        const uint64_t groups = (uint64_t(input.cols) + config.samples_per_threadgroup - 1) /
                                config.samples_per_threadgroup;
        if (groups * config.threads > UINT32_MAX)
            return false;
    }
    return true;
}

FusedInferenceConfig fused_inference_config(FusedShape shape, uint32_t precision, bool boundary) {
    if (!Traits::valid_width(shape.width) || !shape.inputs || !shape.outputs ||
        shape.activation > uint32_t(Activation::None) ||
        shape.output_activation > uint32_t(Activation::None))
        return {};
    // The width16 private specialization preloads its first affine matrix.
    // Keep that compile-time register footprint bounded as input dimensions
    // grow; the panel specialization accepts all larger dimensions unchanged.
    boundary = boundary || !shape.hidden_layers || shape.inputs % 16 || shape.outputs % 16 ||
               shape.outputs > shape.width ||
               (shape.width == 16 &&
                shape.inputs > tcnn_fused::InferenceProfile<16, 2>::PrivateInputLimit);
    if (precision == 2)
        return boundary ? boundary_inference_config<2>(shape.width) : inference_config<2>(shape);
    if (precision == 4)
        return boundary ? boundary_inference_config<4>(shape.width) : inference_config<4>(shape);
    return {};
}

bool fused_inference_eligible(const metalrt::Capabilities& capabilities, const Mat& input,
                              const Mat& output, FusedShape shape, size_t n_params,
                              uintptr_t input_address, uintptr_t weights_address,
                              uintptr_t output_address) {
    if (capabilities.backend != metalrt::BackendKind::Metal || capabilities.apple_gpu_family < 9)
        return false;
    const bool boundary = inference_needs_boundary(shape, input, output, input_address,
                                                   weights_address, output_address);
    const auto config = fused_inference_config(shape, input.precision, boundary);
    if (!config.threads || !input.cols || output.precision != input.precision ||
        input.cols != output.cols || input.rows != shape.inputs || output.rows != shape.outputs ||
        config.threadgroup_bytes > 32768 || !input.rs || !input.cs || !output.rs || !output.cs)
        return false;
    const uint64_t parameters =
        shape.hidden_layers ? uint64_t(shape.width) * shape.inputs +
                                  uint64_t(shape.hidden_layers - 1) * shape.width * shape.width +
                                  uint64_t(shape.outputs) * shape.width
                            : uint64_t(shape.inputs) * shape.outputs;
    const uint64_t input_span =
        uint64_t(input.rows - 1) * input.rs + uint64_t(input.cols - 1) * input.cs + 1;
    const uint64_t output_span =
        uint64_t(output.rows - 1) * output.rs + uint64_t(output.cols - 1) * output.cs + 1;
    const uint64_t groups = (uint64_t(input.cols) + config.samples_per_threadgroup - 1) /
                            config.samples_per_threadgroup;
    if (parameters > UINT32_MAX || n_params != parameters || input_span > UINT32_MAX ||
        output_span > UINT32_MAX || groups * config.threads > UINT32_MAX || !input_address ||
        !weights_address || !output_address || input_address % input.precision ||
        weights_address % input.precision || output_address % input.precision)
        return false;
    const uint64_t input_bytes = input_span * input.precision,
                   output_bytes = output_span * input.precision,
                   weight_bytes = parameters * input.precision;
    if (input_bytes > UINTPTR_MAX - input_address || output_bytes > UINTPTR_MAX - output_address ||
        weight_bytes > UINTPTR_MAX - weights_address)
        return false;
    return !ranges_overlap(input_address, size_t(input_bytes), output_address,
                           size_t(output_bytes)) &&
           !ranges_overlap(weights_address, size_t(weight_bytes), output_address,
                           size_t(output_bytes));
}

template <class T>
bool try_fused_inference(metalStream_t stream, const GPUMatrixDynamic<T>& input, const T* weights,
                         size_t n_params, GPUMatrixDynamic<T>& output, FusedShape shape) {
    auto* queue = resolve_stream(stream);
    const auto device = queue->device();
    const auto x = fused_matrix(input), y = fused_matrix(output);
    const auto input_address = reinterpret_cast<uintptr_t>(input.data());
    const auto weight_address = reinterpret_cast<uintptr_t>(weights);
    const auto output_address = reinterpret_cast<uintptr_t>(output.data());
    if (!fused_inference_eligible(device.capabilities(), x, y, shape, n_params, input_address,
                                  weight_address, output_address))
        return false;
    const bool boundary =
        inference_needs_boundary(shape, x, y, input_address, weight_address, output_address);
    const auto config = fused_inference_config(shape, sizeof(T), boundary);
    const auto kernel = fused_inference_kernel(device, shape, sizeof(T), config.boundary_tiles);
    if (!fused_launch_fits(device, kernel, config.threads))
        return false;
    const tcnn_fused::InferenceArguments args{input.n(), shape.hidden_layers, x.rs, x.cs, y.rs,
                                              y.cs};
    const uint64_t groups =
        (uint64_t(input.n()) + config.samples_per_threadgroup - 1) / config.samples_per_threadgroup;
    queue->dispatch(kernel, {uint32_t(groups * config.threads), 1, 1}, {config.threads, 1, 1},
                    {fused_matrix_arg(0, input, metalrt::Access::Read),
                     fused_raw(1, weights, n_params, metalrt::Access::Read),
                     fused_matrix_arg(2, output, metalrt::Access::Write),
                     metalrt::Argument::scalar(3, args)});
    return true;
}

size_t fused_storage_count(network_kernels::FusedShape shape, uint32_t batch_size) {
    const size_t width =
        metalrt::checked_add(metalrt::checked_mul(shape.width, shape.hidden_layers), shape.outputs);
    const size_t count = metalrt::checked_mul(width, batch_size);
    nn_check(count <= UINT32_MAX, "fused MLP context exceeds 32-bit indexing");
    return count;
}

template <class T>
void fused_forward(metalStream_t stream, const GPUMatrixDynamic<T>& input, const T* weights,
                   size_t n_params, GPUMatrixDynamic<T>& pre, GPUMatrixDynamic<T>& hidden,
                   network_kernels::FusedShape shape) {
    if (!input.n())
        return;
    FusedArgs a{};
    a.x = fused_matrix(input);
    a.shape = shape;
    const size_t count = fused_storage_count(shape, input.n());
    fused_training_dispatch(stream, FusedTrainingPass::Forward, a, n_params,
                            training_aligned(input.data()) && training_aligned(weights) &&
                                training_aligned(pre.data()) && training_aligned(hidden.data()),
                            {fused_matrix_arg(0, input, metalrt::Access::Read),
                             fused_raw(1, weights, n_params, metalrt::Access::Read),
                             fused_raw(2, pre.data(), count, metalrt::Access::Write),
                             fused_raw(3, hidden.data(), count, metalrt::Access::ReadWrite),
                             metalrt::Argument::scalar(Traits::forward_constants, a)});
}

template <class T>
void fused_backward(metalStream_t stream, const GPUMatrixDynamic<T>& input,
                    const GPUMatrixDynamic<T>& output, const GPUMatrixDynamic<T>& incoming,
                    GPUMatrixDynamic<T>* dinput, const T* weights, T* gradients, size_t n_params,
                    const GPUMatrixDynamic<T>& pre, const GPUMatrixDynamic<T>& hidden,
                    network_kernels::FusedShape shape, GradientMode mode) {
    if (empty_gradient(stream, input.n(), gradients, n_params, mode))
        return;
    FusedArgs a{};
    a.x = fused_matrix(input);
    a.output = fused_matrix(output);
    a.incoming = fused_matrix(incoming);
    a.shape = shape;
    a.mode = uint32_t(mode);
    a.write_dx = dinput ? 1u : 0u;
    const size_t count = fused_storage_count(shape, input.n());
    GPUMatrixDynamic<T> delta{uint32_t(count), 1u, stream}, input_gradient;
    // A separate encoded-gradient buffer is the common composed-network case.
    // Write it directly, preserving its layout. Only aliased destinations need
    // the original staging allocation and post-reduction copy.
    auto* dx_target = dinput;
    bool stage_dx = false;
    if (dinput) {
        stage_dx = fused_dx_needs_staging(*dinput, input, output, incoming, weights, gradients,
                                          n_params, pre, hidden);
        if (stage_dx) {
            input_gradient = GPUMatrixDynamic<T>{input.m(), input.n(), stream};
            dx_target = &input_gradient;
        }
        a.dx = fused_matrix(*dx_target);
    }
    fused_training_dispatch(
        stream, FusedTrainingPass::Backward, a, n_params,
        training_aligned(weights) && training_aligned(pre.data()) &&
            training_aligned(hidden.data()) && training_aligned(output.data()) &&
            training_aligned(incoming.data()) && training_aligned(delta.data()) &&
            (!dx_target || training_aligned(dx_target->data())),
        {fused_raw(0, weights, n_params, metalrt::Access::Read),
         fused_raw(1, pre.data(), count, metalrt::Access::Read),
         fused_raw(2, hidden.data(), count, metalrt::Access::Read),
         fused_matrix_arg(3, output, metalrt::Access::Read),
         fused_matrix_arg(4, incoming, metalrt::Access::Read),
         fused_raw(5, delta.data(), count, metalrt::Access::ReadWrite),
         dx_target ? fused_matrix_arg(6, *dx_target, metalrt::Access::Write)
                   : fused_raw(6, delta.data(), count, metalrt::Access::ReadWrite),
         metalrt::Argument::scalar(Traits::backward_constants, a)});
    if (mode != GradientMode::Ignore)
        fused_parameter_dispatch(
            stream, false, a, n_params,
            (training_aligned(input.data()) ? 1u : 0u) |
                (training_aligned(hidden.data()) ? 2u : 0u) |
                (training_aligned(delta.data()) ? 4u : 0u),
            {fused_matrix_arg(0, input, metalrt::Access::Read),
             fused_raw(1, hidden.data(), count, metalrt::Access::Read),
             fused_raw(2, delta.data(), count, metalrt::Access::Read),
             fused_raw(3, gradients, n_params,
                       mode == GradientMode::Accumulate ? metalrt::Access::ReadWrite
                                                        : metalrt::Access::Write),
             metalrt::Argument::scalar(Traits::parameter_gradient_constants, a)});
    if (stage_dx)
        copy_matrix(stream, input_gradient, *dinput);
}

template <class T>
void fused_second_backward(metalStream_t stream, const GPUMatrixDynamic<T>& input,
                           const GPUMatrixDynamic<T>& direction,
                           const GPUMatrixDynamic<T>& incoming, GPUMatrixDynamic<T>* doutput,
                           GPUMatrixDynamic<T>* dinput, const T* weights, T* gradients,
                           size_t n_params, const GPUMatrixDynamic<T>& pre,
                           const GPUMatrixDynamic<T>& hidden, network_kernels::FusedShape shape,
                           GradientMode mode) {
    if (empty_gradient(stream, input.n(), gradients, n_params, mode))
        return;
    FusedArgs a{};
    a.x = fused_matrix(input);
    a.direction = fused_matrix(direction);
    a.incoming = fused_matrix(incoming);
    a.shape = shape;
    a.mode = uint32_t(mode);
    a.write_dx = dinput ? 1u : 0u;
    a.write_ddy = doutput ? 1u : 0u;
    GPUMatrixDynamic<T> input_gradient, output_gradient;
    if (dinput) {
        input_gradient = GPUMatrixDynamic<T>{input.m(), input.n(), stream};
        a.dx = fused_matrix(input_gradient);
    }
    if (doutput) {
        output_gradient = GPUMatrixDynamic<T>{incoming.m(), incoming.n(), stream};
        a.ddy = fused_matrix(output_gradient);
    }
    const size_t count = fused_storage_count(shape, input.n());
    GPUMatrixDynamic<T> tangent_pre{uint32_t(count), 1u, stream},
        tangent{uint32_t(count), 1u, stream}, delta{uint32_t(count), 1u, stream},
        tangent_delta{uint32_t(count), 1u, stream};
    fused_higher_order_dispatch(
        stream, FusedTrainingPass::Tangent, a, n_params,
        {fused_matrix_arg(0, direction, metalrt::Access::Read),
         fused_raw(1, weights, n_params, metalrt::Access::Read),
         fused_raw(2, pre.data(), count, metalrt::Access::Read),
         fused_raw(3, hidden.data(), count, metalrt::Access::Read),
         fused_raw(4, tangent_pre.data(), count, metalrt::Access::Write),
         fused_raw(5, tangent.data(), count, metalrt::Access::ReadWrite),
         doutput ? fused_matrix_arg(6, output_gradient, metalrt::Access::Write)
                 : fused_raw(6, tangent.data(), count, metalrt::Access::ReadWrite),
         metalrt::Argument::scalar(Traits::tangent_constants, a)});
    fused_higher_order_dispatch(
        stream, FusedTrainingPass::SecondBackward, a, n_params,
        {fused_raw(0, weights, n_params, metalrt::Access::Read),
         fused_raw(1, pre.data(), count, metalrt::Access::Read),
         fused_raw(2, hidden.data(), count, metalrt::Access::Read),
         fused_raw(3, tangent_pre.data(), count, metalrt::Access::Read),
         fused_matrix_arg(4, incoming, metalrt::Access::Read),
         fused_raw(5, delta.data(), count, metalrt::Access::ReadWrite),
         fused_raw(6, tangent_delta.data(), count, metalrt::Access::ReadWrite),
         dinput ? fused_matrix_arg(7, input_gradient, metalrt::Access::Write)
                : fused_raw(7, delta.data(), count, metalrt::Access::ReadWrite),
         metalrt::Argument::scalar(Traits::second_backward_constants, a)});
    if (mode != GradientMode::Ignore)
        fused_parameter_dispatch(
            stream, true, a, n_params,
            (training_aligned(input.data()) ? 1u : 0u) |
                (training_aligned(hidden.data()) ? 2u : 0u) |
                (training_aligned(delta.data()) ? 4u : 0u) |
                (training_aligned(direction.data()) ? 8u : 0u) |
                (training_aligned(tangent.data()) ? 16u : 0u) |
                (training_aligned(tangent_delta.data()) ? 32u : 0u),
            {fused_matrix_arg(0, input, metalrt::Access::Read),
             fused_matrix_arg(1, direction, metalrt::Access::Read),
             fused_raw(2, hidden.data(), count, metalrt::Access::Read),
             fused_raw(3, tangent.data(), count, metalrt::Access::Read),
             fused_raw(4, delta.data(), count, metalrt::Access::Read),
             fused_raw(5, tangent_delta.data(), count, metalrt::Access::Read),
             fused_raw(6, gradients, n_params,
                       mode == GradientMode::Accumulate ? metalrt::Access::ReadWrite
                                                        : metalrt::Access::Write),
             metalrt::Argument::scalar(Traits::second_parameter_gradient_constants, a)});
    if (doutput)
        copy_matrix(stream, output_gradient, *doutput);
    if (dinput)
        copy_matrix(stream, input_gradient, *dinput);
}

#define TCNN_FUSED_INSTANTIATE(T)                                                                  \
    template bool try_fused_inference<T>(metalStream_t, const GPUMatrixDynamic<T>&, const T*,      \
                                         size_t, GPUMatrixDynamic<T>&,                             \
                                         network_kernels::FusedShape);                             \
    template void fused_forward<T>(metalStream_t, const GPUMatrixDynamic<T>&, const T*, size_t,    \
                                   GPUMatrixDynamic<T>&, GPUMatrixDynamic<T>&,                     \
                                   network_kernels::FusedShape);                                   \
    template void fused_backward<T>(metalStream_t, const GPUMatrixDynamic<T>&,                     \
                                    const GPUMatrixDynamic<T>&, const GPUMatrixDynamic<T>&,        \
                                    GPUMatrixDynamic<T>*, const T*, T*, size_t,                    \
                                    const GPUMatrixDynamic<T>&, const GPUMatrixDynamic<T>&,        \
                                    network_kernels::FusedShape, GradientMode);                    \
    template void fused_second_backward<T>(                                                        \
        metalStream_t, const GPUMatrixDynamic<T>&, const GPUMatrixDynamic<T>&,                     \
        const GPUMatrixDynamic<T>&, GPUMatrixDynamic<T>*, GPUMatrixDynamic<T>*, const T*, T*,      \
        size_t, const GPUMatrixDynamic<T>&, const GPUMatrixDynamic<T>&,                            \
        network_kernels::FusedShape, GradientMode);
TCNN_FUSED_INSTANTIATE(float)
TCNN_FUSED_INSTANTIATE(half)
#undef TCNN_FUSED_INSTANTIATE
} // namespace detail
} // namespace tcnn
