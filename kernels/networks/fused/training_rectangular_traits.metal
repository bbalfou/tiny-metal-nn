#ifndef TCNN_FUSED_TRAINING_RECTANGULAR_TRAITS_METAL
#define TCNN_FUSED_TRAINING_RECTANGULAR_TRAITS_METAL
#include "training_traits.metal"

namespace tcnn_fused {

// Runtime boundary dimensions do not change the hidden-layer geometry. A
// separate export tile keeps the last hidden state alive while arbitrarily
// wide output/input boundaries are visited one Width-sized tile at a time.
template <class S, unsigned ScalarBytes>
struct RectangularScratch {
    enum : unsigned {
        InputOffset = 0,
        WeightOffset = S::Samples * S::KPanel,
        StateOffset = WeightOffset + S::Width * S::KPanel,
        TemporaryOffset = StateOffset + S::Samples * S::Width,
        Elements = TemporaryOffset + S::Samples * S::Width,
        Bytes = Elements * ScalarBytes
    };
    static_assert(Bytes <= 32768, "rectangular fused training exceeds the portable threadgroup budget");
};

template <unsigned Width, unsigned ScalarBytes, bool Backward>
struct RectangularTuning {
    // Smaller FP16 backward tiles balance register pressure against parallelism.
    // Four SIMD groups partition both samples and features.
    using DefaultShape = typename TrainingTuning<Width, ScalarBytes, Backward>::Shape;
    enum : unsigned { Fast = Backward && ScalarBytes == 2 && Width >= 64 };
    using Shape = TrainingShape<Width, Fast ? 16 : DefaultShape::Samples,
        Fast ? 2 : DefaultShape::RowGroups, Fast ? 2 : DefaultShape::ColumnGroups,
        Fast ? 16 : DefaultShape::KPanel>;
    using Scratch = RectangularScratch<Shape, ScalarBytes>;
};

struct RectangularArguments {
    TrainingArguments training;
    unsigned inputs, outputs;
    unsigned layer_inputs(unsigned layer, unsigned width) const { return layer == 0 ? inputs : width; }
    unsigned layer_rows(unsigned layer, unsigned width) const {
        return layer == training.hidden_layers ? outputs : width;
    }
    unsigned long weight_offset(unsigned layer, unsigned width) const {
        return layer == 0 ? 0 : static_cast<unsigned long>(inputs) * width +
            static_cast<unsigned long>(layer - 1) * width * width;
    }
    unsigned long state_offset(unsigned layer, unsigned width) const {
        return static_cast<unsigned long>(layer) * training.batch * width;
    }
};

} // namespace tcnn_fused
#endif
