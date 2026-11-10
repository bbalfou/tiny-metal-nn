#ifndef TCNN_FUSED_INFERENCE_TRAITS
#define TCNN_FUSED_INFERENCE_TRAITS

// Canonical inference geometry. The retained private/shared-state schedules
// share these traits with host launch inspection and CPU matrix-atom checks.
namespace tcnn_fused {

constexpr unsigned inference_gcd(unsigned a, unsigned b) {
    return b ? inference_gcd(b, a % b) : a;
}

template <unsigned M, unsigned N, unsigned K, unsigned R, unsigned C>
struct InferenceShape {
    static_assert(M % (8 * R) == 0 && N % (8 * C) == 0 && K % 8 == 0,
                  "inference partitions require complete 8x8 matrices");
    enum : unsigned {
        Samples = M, Columns = N, KPanel = K, RowGroups = R, ColumnGroups = C,
        Simdgroups = R * C, Threads = Simdgroups * 32,
        RowTiles = M / (8 * R), ColumnTiles = N / (8 * C),
        Fragments = RowTiles * ColumnTiles, KTiles = K / 8
    };
    static constexpr unsigned first_row(unsigned sg) { return (sg % R) * (M / R); }
    static constexpr unsigned first_column(unsigned sg) { return (sg / R) * (N / C); }
};

template <unsigned W, unsigned Bytes>
struct InferenceProfile {
    static_assert(W == 16 || W == 32 || W == 64 || W == 128, "unsupported inference width");
    static_assert(Bytes == 2 || Bytes == 4, "inference requires half or float storage");
    enum : unsigned {
        Width = W, ScalarBytes = Bytes,
        Samples = W == 16 ? (Bytes == 2 ? 64 : 32) : W == 32 ? (Bytes == 2 ? 32 : 64) : W == 64 ? 64 : 32,
        KPanel = W == 16 ? 16 : W == 32 ? 32 : W == 64 ? (Bytes == 2 ? 16 : 64) : 32,
        Threads = W == 16 ? 32 : W == 32 && Bytes == 4 ? 64 : 128,
        Simdgroups = Threads / 32,
        RowGroups = W == 32 || (W == 64 && Bytes == 2) ? 2 : 1,
        ColumnGroups = Simdgroups / RowGroups,
        Private = W == 16, PrivateInputLimit = 128, DirectWeights = Bytes == 4,
        MergedWeights = W == 32 && Bytes == 2
    };
    static constexpr bool output_partition(unsigned n, unsigned rows) {
        return Simdgroups % rows == 0 && Samples % (8 * rows) == 0 && n % (8 * (Simdgroups / rows)) == 0;
    }
    // FP32 wide sources prefer two row groups for their final projection.
    // The FP16 adapters choose the first valid partition at/above the hidden one.
    static constexpr unsigned output_rows(unsigned n) {
        return W == 16 ? 1 : Bytes == 4 && W >= 64
            ? (output_partition(n, 2) ? 2 : output_partition(n, 1) ? 1 : 4)
            : (RowGroups <= 1 && output_partition(n, 1) ? 1 :
               RowGroups <= 2 && output_partition(n, 2) ? 2 : 4);
    }
    static constexpr unsigned scratch_elements(unsigned inputs) {
        return Private ? 0 : Samples * W + (Bytes == 2 ? W * KPanel :
            inputs == W ? 0 : Samples * inference_gcd(inputs, KPanel));
    }
};

struct InferenceArguments {
    unsigned batch, hidden_layers, input_row_stride, input_sample_stride,
        output_row_stride, output_sample_stride;
};

template <unsigned W, unsigned Bytes, unsigned I, unsigned O, unsigned Preload = 6>
struct InferenceTuning {
    using Profile = InferenceProfile<W, Bytes>;
    static_assert(I > 0 && I % 16 == 0 && O > 0 && O <= W && O % 16 == 0,
                  "aligned inference requires aligned input and bounded output dimensions");
    static_assert(Preload > 0 && Preload <= 6, "private inference preloads at most six affine layers");
    enum : unsigned {
        Width = W, Inputs = I, Outputs = O, ScalarBytes = Bytes,
        Samples = Profile::Samples, KPanel = Profile::KPanel,
        FirstK = inference_gcd(I, KPanel), Threads = Profile::Threads,
        Simdgroups = Profile::Simdgroups, RowGroups = Profile::RowGroups,
        ColumnGroups = Profile::ColumnGroups, OutputRowGroups = Profile::output_rows(O),
        Private = Profile::Private, DirectWeights = Profile::DirectWeights,
        MergedWeights = Profile::MergedWeights, PreloadedLayers = Preload,
        FirstOffset = Bytes == 4 && I != W ? Samples * W : 0,
        WeightOffset = Samples * W,
        ScratchElements = Profile::scratch_elements(I), ScratchBytes = ScratchElements * Bytes
    };
    using Hidden = InferenceShape<Samples, W, KPanel, RowGroups, ColumnGroups>;
    using First = InferenceShape<Samples, W, FirstK, RowGroups, ColumnGroups>;
    using Output = InferenceShape<Samples, O, KPanel, OutputRowGroups, Simdgroups / OutputRowGroups>;
    static_assert(ScratchBytes <= 32768, "aligned inference exceeds 32 KiB shared storage");
};

// Boundary specialization of the same panel/MMA algorithm. State is bounded
// by the hidden width; arbitrary input/output dimensions are visited in tiles.
// The input-panel storage is reused for masked output publication after MMA.
template <unsigned W, unsigned Bytes, unsigned I = W, unsigned O = W>
struct BoundaryInferenceTuning {
    static_assert(W == 16 || W == 32 || W == 64 || W == 128, "unsupported inference width");
    static_assert(Bytes == 2 || Bytes == 4, "inference requires half or float storage");
    static_assert(I > 0 && O > 0, "inference dimensions must be positive");
    enum : unsigned {
        Width = W, Inputs = I, Outputs = O, ScalarBytes = Bytes,
        Samples = 32, KPanel = 16, Threads = 128, Simdgroups = 4, Private = 0,
        RowGroups = 2, ColumnGroups = 2, Projection = W == 16 ? 16 : 32,
        PanelOffset = Samples * W,
        PanelElements = Samples * Projection,
        WeightOffset = PanelOffset + PanelElements,
        ScratchElements = WeightOffset + W * KPanel,
        ScratchBytes = ScratchElements * Bytes
    };
    using Hidden = InferenceShape<Samples, W, KPanel, RowGroups, ColumnGroups>;
    using Output = InferenceShape<Samples, Projection, KPanel, RowGroups, ColumnGroups>;
    static_assert(ScratchBytes <= 32768, "boundary inference exceeds 32 KiB shared storage");
};

} // namespace tcnn_fused
#endif
