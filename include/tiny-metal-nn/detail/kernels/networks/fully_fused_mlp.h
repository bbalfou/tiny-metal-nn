// Whole-model kernel contracts. The scalar entries remain the Reference and
// older-device fallback; tiled native entries preserve their argument records.
#pragma once
#include <tiny-metal-nn/detail/kernels/networks/abi.h>

namespace tcnn {
namespace detail {
namespace network_kernels {
struct FullyFusedMLPTraits {
    static constexpr Path path = Path::FullyFused;
    static constexpr uint32_t abi_version = ABI_VERSION;
    static constexpr const char* source = "networks/fully_fused_mlp.metal";
    static constexpr const char* forward = "fused_mlp_forward";
    static constexpr const char* backward = "fused_mlp_backward";
    static constexpr const char* tangent = "fused_mlp_tangent";
    static constexpr const char* second_backward = "fused_mlp_second_backward";
    using Args = FusedArgs;
    using Shape = FusedShape;

    struct ForwardBindings {
        static constexpr uint32_t input = 0, weights = 1, preactivation = 2, hidden = 3,
                                  arguments = 4;
    };

    struct BackwardBindings {
        static constexpr uint32_t weights = 0, preactivation = 1, hidden = 2, output = 3,
                                  output_gradient = 4, delta = 5, input_gradient = 6, arguments = 7;
    };

    struct ParameterGradientBindings {
        static constexpr uint32_t input = 0, hidden = 1, delta = 2, parameter_gradient = 3,
                                  arguments = 4;
    };

    struct TangentBindings {
        static constexpr uint32_t direction = 0, weights = 1, preactivation = 2, hidden = 3,
                                  tangent_preactivation = 4, tangent = 5, output_gradient = 6,
                                  arguments = 7;
    };

    struct SecondBackwardBindings {
        static constexpr uint32_t weights = 0, preactivation = 1, hidden = 2,
                                  tangent_preactivation = 3, output_gradient = 4, delta = 5,
                                  tangent_delta = 6, input_gradient = 7, arguments = 8;
    };

    struct SecondParameterGradientBindings {
        static constexpr uint32_t input = 0, direction = 1, hidden = 2, tangent = 3, delta = 4,
                                  tangent_delta = 5, parameter_gradient = 6, arguments = 7;
    };

    enum : uint32_t {
        forward_constants = 4,
        backward_constants = 7,
        parameter_gradient_constants = 4,
        tangent_constants = 7,
        second_backward_constants = 8,
        second_parameter_gradient_constants = 7,
        required_alignment = 16
    };

    static constexpr bool valid_width(uint32_t width) {
        return width == 16 || width == 32 || width == 64 || width == 128;
    }
};

// Reference-only callback identifiers. These are independent scalar oracles,
// not Metal entrypoints and cannot be requested on the native backend.
struct FusedReferenceTraits {
    static constexpr const char* parameter_gradient = "fused_mlp_parameter_reference";
    static constexpr const char* second_parameter_gradient = "fused_mlp_second_parameter_reference";
};

// Training uses exactly the established version-1 argument record and buffer
// slots. These are additive native entrypoints, not a new public network ABI.
// Geometry is shared with the shader's portable TrainingTuning traits.
struct FusedTrainingTraits {
    static constexpr uint32_t abi_version = ABI_VERSION;
    static constexpr const char* source = "networks/fused_training.metal";
    static constexpr const char* forward = "fused_mlp_forward_simd";
    static constexpr const char* backward = "fused_mlp_backward_simd";
    using Args = FusedArgs;
    using Shape = FusedShape;
    using ForwardBindings = FullyFusedMLPTraits::ForwardBindings;
    using BackwardBindings = FullyFusedMLPTraits::BackwardBindings;
};

// General boundary layers share the saved-state ABI with the square kernels.
// Dimensions and element strides remain runtime arguments, including zero
// hidden layers and matrices wider than the hidden network width.
struct FusedRectangularTrainingTraits {
    static constexpr uint32_t abi_version = ABI_VERSION;
    static constexpr const char* source = "networks/fused_rectangular_training.metal";
    static constexpr const char* forward = "fused_mlp_forward_rectangular_simd";
    static constexpr const char* backward = "fused_mlp_backward_rectangular_simd";
    using Args = FusedArgs;
    using Shape = FusedShape;
    using ForwardBindings = FullyFusedMLPTraits::ForwardBindings;
    using BackwardBindings = FullyFusedMLPTraits::BackwardBindings;
};

struct FusedHigherOrderTraits {
    static constexpr uint32_t abi_version = ABI_VERSION;
    static constexpr const char* source = "networks/fused_higher_order.metal";
    static constexpr const char* tangent = "fused_mlp_tangent_simd";
    static constexpr const char* second_backward = "fused_mlp_second_backward_simd";
    using Args = FusedArgs;
    using Shape = FusedShape;
    using TangentBindings = FullyFusedMLPTraits::TangentBindings;
    using SecondBackwardBindings = FullyFusedMLPTraits::SecondBackwardBindings;
};

// Parameter products use FP32 split-batch partials followed by a deterministic
// reduction into the original parameter array. The extra metadata is internal
// to these entries; the public version-1 FusedArgs record is not enlarged.
struct FusedParameterGradientTraits {
    static constexpr uint32_t abi_version = ABI_VERSION;
    static constexpr const char* source = "networks/fused_parameters.metal";
    static constexpr const char* first = "fused_mlp_parameter_partials";
    static constexpr const char* second = "fused_mlp_second_parameter_partials";
    static constexpr const char* reduce = "fused_mlp_parameter_reduce";
    static constexpr const char* generic_source = "networks/fused_parameter_generic.metal";
    static constexpr const char* generic_first = "fused_mlp_parameter_partials_generic";
    static constexpr const char* generic_second = "fused_mlp_second_parameter_partials_generic";
    static constexpr const char* generic_reduce = "fused_mlp_parameter_reduce_generic";
    static constexpr uint32_t first_partials = 3, first_dispatch = 5;
    static constexpr uint32_t second_partials = 6, second_dispatch = 8;
};

// Inference has a separate argument record; the saved training FusedArgs ABI
// is unchanged. Host and shader specialize geometry from inference_traits.metal.
struct FusedInferenceTraits {
    static constexpr uint32_t abi_version = ABI_VERSION;
    static constexpr const char* source = "networks/fused_inference.metal";
    static constexpr const char* entry = "fused_mlp_inference";
    struct Bindings {
        static constexpr uint32_t input = 0, weights = 1, output = 2, arguments = 3;
    };
};

} // namespace network_kernels
} // namespace detail
} // namespace tcnn
