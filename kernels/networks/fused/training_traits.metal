#ifndef TCNN_FUSED_TRAINING_TRAITS_METAL
#define TCNN_FUSED_TRAINING_TRAITS_METAL

// This header is deliberately portable C++14. Host launch selection includes
// exactly these traits, so a shader geometry change also changes its dispatch.
namespace tcnn_fused {

template <unsigned Width_, unsigned Samples_, unsigned RowGroups_, unsigned ColumnGroups_, unsigned KPanel_>
struct TrainingShape {
    static_assert(Width_ == 16 || Width_ == 32 || Width_ == 64 || Width_ == 128,
                  "unsupported fused training width");
    static_assert(RowGroups_ > 0 && ColumnGroups_ > 0, "empty SIMD-group partition");
    static_assert(Samples_ % (8 * RowGroups_) == 0, "sample tiles must be complete");
    static_assert(Width_ % (8 * ColumnGroups_) == 0, "column tiles must be complete");
    static_assert(KPanel_ > 0 && KPanel_ % 8 == 0 && Width_ % KPanel_ == 0,
                  "K panels must contain complete matrix tiles");
    enum : unsigned {
        Width = Width_, Samples = Samples_, RowGroups = RowGroups_, ColumnGroups = ColumnGroups_,
        KPanel = KPanel_, Tile = 8, Simdgroups = RowGroups * ColumnGroups, Threads = 32 * Simdgroups,
        RowsPerSimdgroup = Samples / RowGroups, ColumnsPerSimdgroup = Width / ColumnGroups,
        RowTiles = RowsPerSimdgroup / Tile, ColumnTiles = ColumnsPerSimdgroup / Tile,
        Fragments = RowTiles * ColumnTiles, KTiles = KPanel / Tile,
        ResidentState = ColumnGroups == 1
    };
    static constexpr unsigned first_row(unsigned sg) { return (sg % RowGroups) * RowsPerSimdgroup; }
    static constexpr unsigned first_column(unsigned sg) { return (sg / RowGroups) * ColumnsPerSimdgroup; }
    static constexpr unsigned fragment_row(unsigned index) { return (index / ColumnTiles) * Tile; }
    static constexpr unsigned fragment_column(unsigned index) { return (index % ColumnTiles) * Tile; }
};

template <class S, unsigned ScalarBytes, bool Backward>
struct TrainingScratch;

template <class S, unsigned ScalarBytes>
struct TrainingScratch<S, ScalarBytes, false> {
    // The supplied width-64 forward keeps its input panel separate. Width 128
    // reuses the initial input-panel storage for the complete hidden state.
    enum : unsigned {
        AliasInputState = S::Width == 128,
        WeightOffset = AliasInputState ? 0 : S::Samples * S::KPanel,
        StateOffset = WeightOffset + S::Width * S::KPanel,
        InputOffset = AliasInputState ? StateOffset : 0,
        Elements = StateOffset + S::Samples * S::Width,
        Bytes = Elements * ScalarBytes
    };
    static_assert(Bytes <= 32768, "fused training exceeds the portable threadgroup-memory budget");
};

template <class S, unsigned ScalarBytes>
struct TrainingScratch<S, ScalarBytes, true> {
    // The incoming K panel is dead before the first hidden delta is stored.
    enum : unsigned {
        AliasInputState = 1, WeightOffset = 0,
        StateOffset = S::Width * S::KPanel, InputOffset = StateOffset,
        Elements = StateOffset + S::Samples * S::Width,
        Bytes = Elements * ScalarBytes
    };
    static_assert(Bytes <= 32768, "fused backward exceeds the portable threadgroup-memory budget");
};

template <unsigned Width, unsigned ScalarBytes, bool Backward>
struct TrainingTuning {
    static_assert(ScalarBytes == 2 || ScalarBytes == 4, "fused training supports FP16 and FP32 storage");
    enum : unsigned {
        Samples = Backward ? (Width == 128 ? (ScalarBytes == 2 ? 32 : 16) : (Width == 64 ? 16 : 32))
                           : (Width == 128 ? (ScalarBytes == 2 ? 32 : 16)
                                          : (Width == 16 || ScalarBytes == 2 ? 64 : 32)),
        RowGroups = Width <= 32 ? 4 : (Backward && Width == 64 ? 2 : 1),
        ColumnGroups = Width <= 32 || (Backward && Width == 64) ? 1 : 4,
        KPanel = Backward || ScalarBytes == 4 || Width == 16 ? 16 : 32
    };
    using Shape = TrainingShape<Width, Samples, RowGroups, ColumnGroups, KPanel>;
    using Scratch = TrainingScratch<Shape, ScalarBytes, Backward>;
};

struct TrainingMatrixLayout {
    unsigned row_stride, sample_stride;
};

struct TrainingArguments {
    unsigned batch, hidden_layers;
    TrainingMatrixLayout input, output, incoming, dx;
    bool write_dx;
};

} // namespace tcnn_fused
#endif
