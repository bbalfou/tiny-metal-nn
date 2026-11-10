#ifndef TCNN_FUSED_PARAMETER_KERNEL_METAL
#define TCNN_FUSED_PARAMETER_KERNEL_METAL
#include "parameter_traits.metal"
#include "parameter_reduction.metal"
#include "traits.metal"
#ifdef __METAL_VERSION__
#include "parameter_atoms.metal"
#endif

namespace tcnn_fused {
namespace parameter_detail {

struct TileProblem {
    unsigned layer, rows, inputs, row, column;
    unsigned long weight_base, saved_base, input_base;
};

template <class S>
TCNN_FUSED_INLINE TileProblem tile_problem(ParameterArguments a, unsigned tile) {
    unsigned layer = 0;
    for (; layer < a.hidden_layers; ++layer) {
        const unsigned rows = parameter_rows(a, layer), inputs = parameter_inputs(a, layer);
        const unsigned long count = ((static_cast<unsigned long>(rows) + S::TileRows - 1) / S::TileRows) *
                                    ((static_cast<unsigned long>(inputs) + S::TileColumns - 1) / S::TileColumns);
        if (tile < count) break;
        tile -= unsigned(count);
    }
    const unsigned rows = parameter_rows(a, layer), inputs = parameter_inputs(a, layer);
    const unsigned columns = unsigned((static_cast<unsigned long>(inputs) + S::TileColumns - 1) / S::TileColumns);
    return {layer, rows, inputs, (tile / columns) * S::TileRows, (tile % columns) * S::TileColumns,
            parameter_weight_offset(a, layer), static_cast<unsigned long>(layer) * a.batch * a.width,
            layer ? static_cast<unsigned long>(layer - 1) * a.batch * a.width : 0};
}

template <class S, class Atom, class Copy, bool Second>
struct Workspace {
    using Scalar = typename Atom::Scalar;
    typename Atom::Accumulator primal[S::Fragments], tangent[Second ? unsigned(S::Fragments) : 1u];
    threadgroup Scalar* left;
    threadgroup Scalar* right;
    threadgroup float* output;
    device const Scalar* input;
    device const Scalar* hidden;
    device const Scalar* delta;
    device const Scalar* direction;
    device const Scalar* tangent_input;
    device const Scalar* tangent_delta;
    device float* partials;
    ParameterArguments args;
    TileProblem tile;
    unsigned tid, row, column, split;
};

template <class S, class Atom, class Copy, bool Second>
struct Zero {
    template <unsigned I>
    static TCNN_FUSED_INLINE void apply(thread Workspace<S, Atom, Copy, Second>& w) {
        Atom::zero(w.primal[I]);
        if (Second) Atom::zero(w.tangent[I]);
    }
};

template <class S, class Atom, class Copy, bool Second>
struct LoadPanels {
    using Scalar = typename Atom::Scalar;
    enum : unsigned { V = Copy::VectorElements };

    template <unsigned Columns>
    static TCNN_FUSED_INLINE void load(thread Workspace<S, Atom, Copy, Second>& w, threadgroup Scalar* target,
                                        device const Scalar* source, unsigned long source_base,
                                        ParameterMatrixLayout layout, unsigned source_rows,
                                        unsigned first_row, unsigned long first_sample, bool aligned) {
        enum : unsigned { VectorsPerSample = Columns / V, Vectors = S::KPanel * VectorsPerSample };
        if (layout.sample_stride == 1 && layout.row_stride != 1) {
            // First-layer encoding values (and second-order directions) can
            // be SoA. Coalesce across the reduction's sample axis, then
            // transpose into the unchanged [sample, feature] MMA panel.
            for (unsigned i = w.tid; i < S::KPanel * Columns; i += S::Threads) {
                const unsigned k = i % S::KPanel, column = i / S::KPanel;
                const unsigned long sample = first_sample + k;
                const unsigned row = first_row + column;
                target[k * Columns + column] = sample < w.args.batch && row < source_rows
                    ? source[source_base + sample + static_cast<unsigned long>(row) * layout.row_stride] : Scalar(0);
            }
            return;
        }
        for (unsigned v = w.tid; v < Vectors; v += S::Threads) {
            const unsigned long sample = first_sample + v / VectorsPerSample;
            const unsigned row = first_row + (v % VectorsPerSample) * V;
            const unsigned long index = source_base + sample * layout.sample_stride + static_cast<unsigned long>(row) * layout.row_stride;
            if (aligned && sample < w.args.batch && row < source_rows && source_rows - row >= V &&
                layout.row_stride == 1 && index % V == 0) {
                Copy::to_shared(target, v * V, source, index);
            } else {
                for (unsigned i = 0; i < V; ++i)
                    target[v * V + i] = sample < w.args.batch && row + i < source_rows
                        ? source[index + static_cast<unsigned long>(i) * layout.row_stride] : Scalar(0);
            }
        }
    }

    template <bool Tangent>
    static TCNN_FUSED_INLINE void run(thread Workspace<S, Atom, Copy, Second>& w, unsigned long sample) {
        const unsigned mask = w.args.dispatch.vector_mask;
        const bool first = w.tile.layer == 0;
        load<S::TileRows>(w, w.left, Tangent ? w.tangent_delta : w.delta, w.tile.saved_base,
                          {1, w.tile.rows}, w.tile.rows, w.tile.row, sample,
                          (mask & (1u << (Tangent ? 5 : 2))) != 0);
        load<S::TileColumns>(w, w.right,
                             Tangent ? (first ? w.direction : w.tangent_input) : (first ? w.input : w.hidden),
                             w.tile.input_base,
                             first ? (Tangent ? w.args.direction : w.args.input) : ParameterMatrixLayout{1, w.args.width},
                             w.tile.inputs, w.tile.column, sample,
                             (mask & (1u << (Tangent ? (first ? 3 : 4) : (first ? 0 : 1)))) != 0);
    }
};

template <class S, class Atom, class Copy, bool Second>
struct FragmentContext {
    thread Workspace<S, Atom, Copy, Second>* workspace;
    thread typename Atom::Operand* a;
    thread typename Atom::Operand* b;
};

template <class S, class Atom, class Copy, bool Second, unsigned K>
struct LoadLeft {
    template <unsigned Row>
    static TCNN_FUSED_INLINE void apply(thread FragmentContext<S, Atom, Copy, Second>& c) {
        thread Workspace<S, Atom, Copy, Second>& w = *c.workspace;
        Atom::load_shared(c.a[Row], w.left, K * S::Tile * S::TileRows + w.row + Row * S::Tile,
                          S::TileRows, true);
    }
};
template <class S, class Atom, class Copy, bool Second, unsigned K>
struct LoadRight {
    template <unsigned Column>
    static TCNN_FUSED_INLINE void apply(thread FragmentContext<S, Atom, Copy, Second>& c) {
        thread Workspace<S, Atom, Copy, Second>& w = *c.workspace;
        Atom::load_shared(c.b[Column], w.right, K * S::Tile * S::TileColumns + w.column + Column * S::Tile,
                          S::TileColumns, false);
    }
};
template <class S, class Atom, class Copy, bool Second, bool Tangent>
struct Multiply {
    template <unsigned I>
    static TCNN_FUSED_INLINE void apply(thread FragmentContext<S, Atom, Copy, Second>& c) {
        Atom::mma(Tangent ? c.workspace->tangent[I] : c.workspace->primal[I],
                  c.a[I / S::ColumnTiles], c.b[I % S::ColumnTiles]);
    }
};
template <class S, class Atom, class Copy, bool Second, bool Tangent>
struct KTile {
    template <unsigned K>
    static TCNN_FUSED_INLINE void apply(thread Workspace<S, Atom, Copy, Second>& w) {
        typename Atom::Operand a[S::RowTiles], b[S::ColumnTiles];
        FragmentContext<S, Atom, Copy, Second> context{&w, a, b};
        StaticFor<0, S::RowTiles>::template run<LoadLeft<S, Atom, Copy, Second, K>>(context);
        StaticFor<0, S::ColumnTiles>::template run<LoadRight<S, Atom, Copy, Second, K>>(context);
        StaticFor<0, S::Fragments>::template run<Multiply<S, Atom, Copy, Second, Tangent>>(context);
    }
};

template <class S, class Atom, class Copy, bool Second, bool Tangent>
struct StoreFragment {
    template <unsigned I>
    static TCNN_FUSED_INLINE void apply(thread Workspace<S, Atom, Copy, Second>& w) {
        const unsigned offset = (w.row + (I / S::ColumnTiles) * S::Tile) * S::TileColumns +
                                 w.column + (I % S::ColumnTiles) * S::Tile;
        Atom::store_accumulator(Tangent ? w.tangent[I] : w.primal[I], w.output, offset, S::TileColumns);
    }
};
template <class S, class Atom, class Copy, bool Second, bool Tangent>
struct StorePartial {
    static TCNN_FUSED_INLINE void run(thread Workspace<S, Atom, Copy, Second>& w) {
        StaticFor<0, S::Fragments>::template run<StoreFragment<S, Atom, Copy, Second, Tangent>>(w);
        Copy::barrier();
        const unsigned long partial_base = (static_cast<unsigned long>(Tangent ? w.args.dispatch.splits : 0) + w.split) *
                                             w.args.dispatch.parameters + w.tile.weight_base;
        for (unsigned i = w.tid; i < S::OutputElements; i += S::Threads) {
            const unsigned row = w.tile.row + i / S::TileColumns, column = w.tile.column + i % S::TileColumns;
            if (row < w.tile.rows && column < w.tile.inputs)
                w.partials[partial_base + static_cast<unsigned long>(row) * w.tile.inputs + column] = w.output[i];
        }
        Copy::barrier();
    }
};
} // namespace parameter_detail

// Matrix-multiply reduction over batch. Each threadgroup owns one parameter
// tile and one contiguous batch partition. Only this epilogue exports FP32;
// conversion/Accumulate are delayed until every partition has been reduced.
template <class S, class Atom, class Copy, bool Second>
struct ParameterGradient {
    using Scalar = typename Atom::Scalar;
    static TCNN_FUSED_INLINE void run(
        device const Scalar* input, device const Scalar* hidden, device const Scalar* delta,
        device const Scalar* direction, device const Scalar* tangent, device const Scalar* tangent_delta,
        device float* partials, threadgroup Scalar* scratch, threadgroup float* output,
        ParameterArguments args, unsigned group, unsigned split, unsigned tid, unsigned sg
    ) {
        const unsigned long tile_index = static_cast<unsigned long>(group) + args.dispatch.tile_offset;
        if (tile_index >= args.dispatch.tiles_per_split || split >= args.dispatch.splits)
            return;
        parameter_detail::Workspace<S, Atom, Copy, Second> w;
        w.left = scratch; w.right = scratch + S::LeftElements; w.output = output;
        w.input = input; w.hidden = hidden; w.delta = delta; w.direction = direction;
        w.tangent_input = tangent; w.tangent_delta = tangent_delta; w.partials = partials;
        w.args = args; w.tile = parameter_detail::tile_problem<S>(args, unsigned(tile_index));
        w.tid = tid; w.row = S::first_row(sg); w.column = S::first_column(sg); w.split = split;
        StaticFor<0, S::Fragments>::template run<parameter_detail::Zero<S, Atom, Copy, Second>>(w);
        const unsigned long panels = (static_cast<unsigned long>(args.batch) + S::KPanel - 1) / S::KPanel;
        const unsigned long per_split = (panels + args.dispatch.splits - 1) / args.dispatch.splits;
        const unsigned long begin = static_cast<unsigned long>(split) * per_split * S::KPanel;
        const unsigned long end = begin + per_split * S::KPanel;
#pragma clang loop unroll(disable)
        for (unsigned long sample = begin; sample < end && sample < args.batch; sample += S::KPanel) {
            parameter_detail::LoadPanels<S, Atom, Copy, Second>::template run<false>(w, sample);
            Copy::barrier();
            StaticFor<0, S::KTiles>::template run<parameter_detail::KTile<S, Atom, Copy, Second, false>>(w);
            Copy::barrier();
            if (Second) {
                parameter_detail::LoadPanels<S, Atom, Copy, Second>::template run<true>(w, sample);
                Copy::barrier();
                StaticFor<0, S::KTiles>::template run<parameter_detail::KTile<S, Atom, Copy, Second, true>>(w);
                Copy::barrier();
            }
        }
        parameter_detail::StorePartial<S, Atom, Copy, Second, false>::run(w);
        if (Second) parameter_detail::StorePartial<S, Atom, Copy, Second, true>::run(w);
    }
};

} // namespace tcnn_fused
#endif
