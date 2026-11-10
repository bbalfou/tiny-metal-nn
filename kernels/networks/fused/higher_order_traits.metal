#ifndef TCNN_FUSED_HIGHER_ORDER_TRAITS_METAL
#define TCNN_FUSED_HIGHER_ORDER_TRAITS_METAL

namespace tcnn_fused {

// A SIMD-group owns complete sample rows, as in inference. Its tangent/adjoint
// fragments remain in registers while the next layer's bounded B panel loads.
template <unsigned Width_, unsigned ScalarBytes>
struct HigherOrderTuning {
    static_assert(Width_ == 16 || Width_ == 32 || Width_ == 64 || Width_ == 128,
                  "unsupported fused derivative width");
    static_assert(ScalarBytes == 2 || ScalarBytes == 4, "unsupported fused derivative scalar");
    enum : unsigned {
        Width = Width_, Tile = 8, RowTiles = Width <= 32 ? 2 : 1,
        ColumnTiles = Width / Tile, Fragments = RowTiles * ColumnTiles,
        Simdgroups = 4, Threads = Simdgroups * 32,
        SamplesPerSimdgroup = RowTiles * Tile, Samples = Simdgroups * SamplesPerSimdgroup,
        RowGroups = Simdgroups, ColumnGroups = 1, KPanel = 16, KTiles = KPanel / Tile,
        WeightOffset = 0, InputOffset = Width * KPanel,
        SecondInputOffset = InputOffset + Samples * KPanel,
        ScratchElements = SecondInputOffset + Samples * KPanel,
        ScratchBytes = ScratchElements * ScalarBytes
    };
    using Shape = HigherOrderTuning;
    static_assert(ScratchBytes <= 32768, "fused derivatives exceed the threadgroup-memory budget");
};

struct HigherOrderLayout { unsigned row_stride, sample_stride; };

struct HigherOrderArguments {
    unsigned batch, inputs, outputs, hidden_layers;
    HigherOrderLayout direction, incoming, dx, ddy;
    bool write_dx, write_ddy;
};

} // namespace tcnn_fused
#endif
