#ifndef TCNN_FUSED_PARAMETER_TRAITS_METAL
#define TCNN_FUSED_PARAMETER_TRAITS_METAL

// Portable C++14 launch traits shared verbatim by host selection and Metal.
// Matrices use the public canonical [output,input] parameter order. Batch is
// the reduction dimension; partial sums never change the parameter layout.
namespace tcnn_fused {

template <unsigned Width, unsigned ScalarBytes>
struct ParameterTuning {
    static_assert(Width == 16 || Width == 32 || Width == 64 || Width == 128, "unsupported fused width");
    static_assert(ScalarBytes == 2 || ScalarBytes == 4, "unsupported fused scalar");
    enum : unsigned {
        TileRows = Width == 16 ? 16 : 32,
        TileColumns = TileRows,
        KPanel = Width >= 64 && ScalarBytes == 2 ? 64 : 32, Tile = 8,
        RowGroups = 2, ColumnGroups = 2, Simdgroups = 4, Threads = Simdgroups * 32,
        RowsPerGroup = TileRows / RowGroups, ColumnsPerGroup = TileColumns / ColumnGroups,
        RowTiles = RowsPerGroup / Tile, ColumnTiles = ColumnsPerGroup / Tile,
        Fragments = RowTiles * ColumnTiles, KTiles = KPanel / Tile,
        LeftElements = KPanel * TileRows, RightElements = KPanel * TileColumns,
        ScratchElements = LeftElements + RightElements,
        OutputElements = TileRows * TileColumns,
        ScratchBytes = ScratchElements * ScalarBytes + OutputElements * sizeof(float),
        ReductionThreads = 256, ReductionVector = 4
    };
    static_assert(ScratchBytes <= 32768, "parameter staging exceeds threadgroup budget");
    static constexpr unsigned first_row(unsigned sg) { return (sg % RowGroups) * RowsPerGroup; }
    static constexpr unsigned first_column(unsigned sg) { return (sg / RowGroups) * ColumnsPerGroup; }
};

struct ParameterMatrixLayout { unsigned row_stride, sample_stride; };

// Internal dispatch metadata; the public FusedArgs ABI remains unchanged.
// vector_mask bits: input, hidden, delta, direction, tangent, tangent_delta.
// A bit permits a 16-byte transfer only when its element offset is aligned too.
struct ParameterDispatch {
    unsigned parameters, splits, tiles_per_split, vector_mask, tile_offset;
};
static_assert(sizeof(ParameterDispatch) == 20, "parameter dispatch ABI mismatch");

struct ParameterReduceArguments {
    unsigned parameters, splits, mode, second;
};
static_assert(sizeof(ParameterReduceArguments) == 16, "parameter reduction ABI mismatch");

struct ParameterArguments {
    unsigned batch, inputs, width, outputs, hidden_layers;
    ParameterMatrixLayout input, direction;
    ParameterDispatch dispatch;
};

inline unsigned parameter_rows(ParameterArguments a, unsigned layer) {
    return layer == a.hidden_layers ? a.outputs : a.width;
}
inline unsigned parameter_inputs(ParameterArguments a, unsigned layer) {
    return layer == 0 ? a.inputs : a.width;
}
inline unsigned long parameter_weight_offset(ParameterArguments a, unsigned layer) {
    return layer == 0 ? 0 : static_cast<unsigned long>(a.inputs) * a.width +
                                  static_cast<unsigned long>(layer - 1) * a.width * a.width;
}

template <class S>
inline unsigned long parameter_tiles(unsigned inputs, unsigned width, unsigned outputs, unsigned hidden_layers) {
    const unsigned long input_tiles = (static_cast<unsigned long>(inputs) + S::TileColumns - 1) / S::TileColumns;
    const unsigned long output_tiles = (static_cast<unsigned long>(outputs) + S::TileRows - 1) / S::TileRows;
    const unsigned long hidden_rows = (static_cast<unsigned long>(width) + S::TileRows - 1) / S::TileRows;
    const unsigned long hidden_columns = (static_cast<unsigned long>(width) + S::TileColumns - 1) / S::TileColumns;
    if (!hidden_layers)
        return output_tiles * input_tiles;
    return hidden_rows * input_tiles +
           static_cast<unsigned long>(hidden_layers - 1) * hidden_rows * hidden_columns +
           output_tiles * hidden_columns;
}

// Bound concurrency and workspace while exposing parallelism for small MLPs.
// The 64 MiB target can be exceeded by the unavoidable one-split FP32 buffer
// for very large parameter arrays; extra splits never push it over that target.
inline unsigned parameter_splits(unsigned batch, unsigned parameters, bool second) {
    if (!batch || !parameters)
        return 1;
    unsigned wanted = unsigned((static_cast<unsigned long>(batch) + 1023) / 1024);
    if (wanted > 64) wanted = 64;
    const unsigned long one_split = static_cast<unsigned long>(parameters) * sizeof(float) * (second ? 2 : 1);
    const unsigned long affordable = (64ul * 1024 * 1024) / one_split;
    if (affordable < wanted) wanted = affordable ? unsigned(affordable) : 1;
    return wanted ? wanted : 1;
}

inline unsigned long parameter_partial_elements(unsigned parameters, unsigned splits, bool second) {
    return static_cast<unsigned long>(parameters) * splits * (second ? 2 : 1);
}

} // namespace tcnn_fused
#endif
