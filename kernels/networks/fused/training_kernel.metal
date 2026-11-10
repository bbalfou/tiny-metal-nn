#ifndef TCNN_FUSED_TRAINING_KERNEL_METAL
#define TCNN_FUSED_TRAINING_KERNEL_METAL
#include "training_traits.metal"
#include "traits.metal"
#ifdef __METAL_VERSION__
#include "training_atoms.metal"
#endif

namespace tcnn_fused {

template <class S, class Atom, class Copy>
struct TrainingWorkspace {
    using Scalar = typename Atom::Scalar;
    typename Atom::Accumulator accumulators[S::Fragments];
    // The inference schedule retains complete sample rows in registers.
    // Training can do the same exactly when a SIMD-group owns all columns;
    // column-partitioned reference geometries continue to exchange shared state.
    typename Atom::Operand resident[S::ResidentState ? unsigned(S::Fragments) : 1u];
    threadgroup Scalar* input_panel;
    threadgroup Scalar* weight_panel;
    threadgroup Scalar* states;
    device const Scalar* weights;
    TrainingArguments args;
    unsigned sample, tid, row, column;
    bool full;
};

namespace training_detail {

template <class S, class Atom, class Copy>
struct Zero {
    template <unsigned I>
    static TCNN_FUSED_INLINE void apply(thread TrainingWorkspace<S, Atom, Copy>& w) {
        Atom::zero(w.accumulators[I]);
    }
};

template <class S, class Atom, class Copy>
struct PanelContext {
    thread TrainingWorkspace<S, Atom, Copy>* workspace;
    unsigned k;
};

template <class S, class Atom, class Copy>
struct FragmentContext {
    thread TrainingWorkspace<S, Atom, Copy>* workspace;
    thread typename Atom::Operand* a;
    thread typename Atom::Operand* b;
    unsigned k;
};

template <class S, class Atom, class Copy, bool First, unsigned K>
struct LoadA {
    template <unsigned Row>
    static TCNN_FUSED_INLINE void apply(thread FragmentContext<S, Atom, Copy>& c) {
        thread TrainingWorkspace<S, Atom, Copy>& w = *c.workspace;
        const unsigned stride = First ? S::KPanel : S::Width;
        const unsigned column = (First ? 0 : c.k) + K * S::Tile;
        if (!First && S::ResidentState)
            c.a[Row] = w.resident[Row * S::ColumnTiles + c.k / S::Tile + K];
        else
            Atom::load_shared(c.a[Row], First ? w.input_panel : w.states,
                              (w.row + Row * S::Tile) * stride + column, stride);
    }
};

template <class S, class Atom, class Copy, bool Backward, unsigned K>
struct LoadB {
    template <unsigned Column>
    static TCNN_FUSED_INLINE void apply(thread FragmentContext<S, Atom, Copy>& c) {
        thread TrainingWorkspace<S, Atom, Copy>& w = *c.workspace;
        const unsigned column = w.column + Column * S::Tile;
        const unsigned offset = Backward ? K * S::Tile * S::Width + column
                                         : column * S::KPanel + K * S::Tile;
        Atom::load_shared(c.b[Column], w.weight_panel, offset, Backward ? S::Width : S::KPanel, !Backward);
    }
};

template <class S, class Atom, class Copy>
struct Multiply {
    template <unsigned I>
    static TCNN_FUSED_INLINE void apply(thread FragmentContext<S, Atom, Copy>& c) {
        Atom::mma(c.workspace->accumulators[I], c.a[I / S::ColumnTiles], c.b[I % S::ColumnTiles]);
    }
};

template <class S, class Atom, class Copy, bool Backward, bool First>
struct KTile {
    template <unsigned K>
    static TCNN_FUSED_INLINE void apply(thread PanelContext<S, Atom, Copy>& c) {
        typename Atom::Operand a[S::RowTiles], b[S::ColumnTiles];
        FragmentContext<S, Atom, Copy> fragments{c.workspace, a, b, c.k};
        StaticFor<0, S::RowTiles>::template run<LoadA<S, Atom, Copy, First, K>>(fragments);
        StaticFor<0, S::ColumnTiles>::template run<LoadB<S, Atom, Copy, Backward, K>>(fragments);
        StaticFor<0, S::Fragments>::template run<Multiply<S, Atom, Copy>>(fragments);
    }
};

template <class S, class Atom, class Copy, bool Backward>
struct WeightPanel {
    static TCNN_FUSED_INLINE void load(thread TrainingWorkspace<S, Atom, Copy>& w, unsigned layer, unsigned k) {
        enum : unsigned { V = Copy::VectorElements, Vectors = S::Width * S::KPanel / V };
        const unsigned long base = static_cast<unsigned long>(layer) * S::Width * S::Width;
        for (unsigned v = w.tid; v < Vectors; v += S::Threads) {
            // Forward stages [output,K], then transposes the native B load.
            // Backward stages [K,input] and consumes canonical weights directly.
            const unsigned source = Backward ? k * S::Width + v * V
                                              : (v / (S::KPanel / V)) * S::Width + k + (v % (S::KPanel / V)) * V;
            Copy::to_shared(w.weight_panel, v * V, w.weights, base + source);
        }
    }
};

template <class S, class Atom, class Copy>
struct InputSource {
    using Scalar = typename Atom::Scalar;
    enum : unsigned { First = 1 };
    device const Scalar* input;
    static TCNN_FUSED_INLINE bool vector_layout(TrainingMatrixLayout layout) {
        return layout.row_stride == 1 && (layout.sample_stride * sizeof(Scalar)) % 16 == 0;
    }
    TCNN_FUSED_INLINE void load(thread TrainingWorkspace<S, Atom, Copy>& w, unsigned k) {
        enum : unsigned { V = Copy::VectorElements, VectorsPerRow = S::KPanel / V };
        if (w.args.input.sample_stride == 1 && w.args.input.row_stride != 1) {
            // Encodings may produce SoA: consecutive samples, not features,
            // occupy consecutive device addresses. Transpose cooperatively
            // into the same [sample, feature] panel consumed by the MMA.
            for (unsigned i = w.tid; i < S::Samples * S::KPanel; i += S::Threads) {
                const unsigned row = i % S::Samples, column = i / S::Samples;
                const unsigned sample = w.sample + row;
                w.input_panel[row * S::KPanel + column] = sample < w.args.batch
                    ? input[static_cast<unsigned long>(sample) +
                            static_cast<unsigned long>(k + column) * w.args.input.row_stride]
                    : Scalar(0);
            }
        } else if (w.full && vector_layout(w.args.input)) {
            for (unsigned v = w.tid; v < S::Samples * VectorsPerRow; v += S::Threads) {
                const unsigned row = v / VectorsPerRow, column = (v % VectorsPerRow) * V;
                Copy::to_shared(w.input_panel, v * V, input,
                                static_cast<unsigned long>(w.sample + row) * w.args.input.sample_stride + k + column);
            }
        } else {
            for (unsigned i = w.tid; i < S::Samples * S::KPanel; i += S::Threads) {
                const unsigned sample = w.sample + i / S::KPanel, column = k + i % S::KPanel;
                w.input_panel[i] = sample < w.args.batch
                    ? input[static_cast<unsigned long>(sample) * w.args.input.sample_stride + column * w.args.input.row_stride]
                    : Scalar(0);
            }
        }
    }
};

struct HiddenSource {
    enum : unsigned { First = 0 };
    template <class Workspace>
    TCNN_FUSED_INLINE void load(thread Workspace&, unsigned) {}
};

template <class S, class Atom, class Copy, class Output>
struct IncomingSource {
    using Scalar = typename Atom::Scalar;
    enum : unsigned { First = 1, InputForm = Output::Kind == 2 || Output::Kind == 4 };
    device const Scalar* pre;
    device const Scalar* output;
    device const Scalar* incoming;
    device Scalar* delta;
    TCNN_FUSED_INLINE void load(thread TrainingWorkspace<S, Atom, Copy>& w, unsigned k) {
        enum : unsigned { V = Copy::IncomingElements, VectorsPerRow = S::KPanel / V };
        const unsigned long base = static_cast<unsigned long>(w.args.hidden_layers) * w.args.batch * S::Width;
        if (Output::Kind == 9 && w.full && w.args.incoming.row_stride == 1 &&
            (w.args.incoming.sample_stride * sizeof(Scalar)) % (V * sizeof(Scalar)) == 0) {
            for (unsigned v = w.tid; v < S::Samples * VectorsPerRow; v += S::Threads) {
                const unsigned sample = w.sample + v / VectorsPerRow, column = k + (v % VectorsPerRow) * V;
                Copy::incoming_panel(w.input_panel, v * V, incoming,
                                     static_cast<unsigned long>(sample) * w.args.incoming.sample_stride + column,
                                     delta, base + static_cast<unsigned long>(sample) * S::Width + column);
            }
        } else {
            for (unsigned i = w.tid; i < S::Samples * S::KPanel; i += S::Threads) {
                const unsigned sample = w.sample + i / S::KPanel, column = k + i % S::KPanel;
                Scalar result = Scalar(0);
                if (sample < w.args.batch) {
                    const unsigned long saved = base + static_cast<unsigned long>(sample) * S::Width + column;
                    const Scalar value = InputForm ? pre[saved]
                        : output[static_cast<unsigned long>(sample) * w.args.output.sample_stride + column * w.args.output.row_stride];
                    result = Atom::template gradient<Output>(value, float(incoming[
                        static_cast<unsigned long>(sample) * w.args.incoming.sample_stride + column * w.args.incoming.row_stride]));
                    delta[saved] = result;
                }
                w.input_panel[i] = result;
            }
        }
    }
};

// Every shared-memory writer is followed by a publication barrier. The second
// barrier retires all readers before a panel refill or in-place state epilogue.
template <class S, class Atom, class Copy, bool Backward>
struct Mainloop {
    // A compile-time K index is essential for register residency: a dynamic
    // fragment-array index can force an otherwise resident state to spill.
    struct ResidentContext { thread TrainingWorkspace<S, Atom, Copy>* workspace; unsigned layer; };
    struct RunResidentPanel {
        template <unsigned Panel>
        static TCNN_FUSED_INLINE void apply(thread ResidentContext& c) {
            thread TrainingWorkspace<S, Atom, Copy>& w = *c.workspace;
            enum : unsigned { K = Panel * S::KPanel };
            WeightPanel<S, Atom, Copy, Backward>::load(w, c.layer, K);
            Copy::barrier();
            PanelContext<S, Atom, Copy> panel{&w, K};
            StaticFor<0, S::KTiles>::template run<KTile<S, Atom, Copy, Backward, false>>(panel);
            Copy::barrier();
        }
    };
    template <class Source>
    static TCNN_FUSED_INLINE void run(thread TrainingWorkspace<S, Atom, Copy>& w,
                                       thread Source& source, unsigned layer) {
        StaticFor<0, S::Fragments>::template run<Zero<S, Atom, Copy>>(w);
        if (!Source::First && S::ResidentState) {
            ResidentContext context{&w, layer};
            StaticFor<0, S::Width / S::KPanel>::template run<RunResidentPanel>(context);
            return;
        }
        // Keep K panels bounded in scope (notably width-128 backward). Matrix
        // fragments within each K panel are completely specialized/unrolled.
#pragma clang loop unroll(disable)
        for (unsigned k = 0; k < S::Width; k += S::KPanel) {
            source.load(w, k);
            WeightPanel<S, Atom, Copy, Backward>::load(w, layer, k);
            Copy::barrier();
            PanelContext<S, Atom, Copy> panel{&w, k};
            StaticFor<0, S::KTiles>::template run<KTile<S, Atom, Copy, Backward, Source::First != 0>>(panel);
            Copy::barrier();
        }
    }
};

template <class S, class Atom, class Copy>
struct TileStore {
    using Scalar = typename Atom::Scalar;
    static TCNN_FUSED_INLINE void run(thread TrainingWorkspace<S, Atom, Copy>& w, device Scalar* destination,
                                       unsigned long base, TrainingMatrixLayout layout) {
        enum : unsigned { V = Copy::VectorElements, VectorsPerRow = S::Width / V };
        if (layout.sample_stride == 1 && layout.row_stride != 1) {
            // Symmetric SoA export, including direct encoded-input gradients.
            // The caller has already published the shared panel.
            for (unsigned i = w.tid; i < S::Samples * S::Width; i += S::Threads) {
                const unsigned row = i % S::Samples, column = i / S::Samples;
                const unsigned sample = w.sample + row;
                if (sample < w.args.batch)
                    destination[base + static_cast<unsigned long>(sample) +
                                static_cast<unsigned long>(column) * layout.row_stride] = w.states[row * S::Width + column];
            }
        } else if (w.full && layout.row_stride == 1 && (layout.sample_stride * sizeof(Scalar)) % 16 == 0) {
            for (unsigned v = w.tid; v < S::Samples * VectorsPerRow; v += S::Threads) {
                const unsigned sample = w.sample + v / VectorsPerRow, column = (v % VectorsPerRow) * V;
                Copy::to_device(destination, base + static_cast<unsigned long>(sample) * layout.sample_stride + column,
                                w.states, v * V);
            }
        } else {
            for (unsigned i = w.tid; i < S::Samples * S::Width; i += S::Threads) {
                const unsigned sample = w.sample + i / S::Width, column = i % S::Width;
                if (sample < w.args.batch)
                    destination[base + static_cast<unsigned long>(sample) * layout.sample_stride + column * layout.row_stride] = w.states[i];
            }
        }
    }
};

template <class S, class Atom, class Copy>
struct SaveContext {
    thread TrainingWorkspace<S, Atom, Copy>* workspace;
    device typename Atom::Scalar* destination;
    device const typename Atom::Scalar* saved;
    unsigned long base;
};

template <class S, class Atom, class Copy, class Policy, bool Full>
struct SaveActivated {
    template <unsigned I>
    static TCNN_FUSED_INLINE void apply(thread SaveContext<S, Atom, Copy>& c) {
        thread TrainingWorkspace<S, Atom, Copy>& w = *c.workspace;
        typename Atom::Operand value;
        Atom::template convert<Policy>(value, w.accumulators[I]);
        const unsigned row = w.row + S::fragment_row(I), column = w.column + S::fragment_column(I);
        if (S::ResidentState)
            w.resident[I] = value;
        if (!Full || !S::ResidentState)
            Atom::store_shared(value, w.states, row * S::Width + column, S::Width);
        if (Full)
            Atom::store_device(value, c.destination,
                               c.base + static_cast<unsigned long>(w.sample + row) * S::Width + column, S::Width);
    }
};

template <class S, class Atom, class Copy, class Policy, bool Shared>
struct ConvertStore {
    template <unsigned I>
    static TCNN_FUSED_INLINE void apply(thread SaveContext<S, Atom, Copy>& c) {
        thread TrainingWorkspace<S, Atom, Copy>& w = *c.workspace;
        typename Atom::Operand value;
        Atom::template convert<Policy>(value, w.accumulators[I]);
        const unsigned row = w.row + S::fragment_row(I), column = w.column + S::fragment_column(I);
        if (Shared)
            Atom::store_shared(value, w.states, row * S::Width + column, S::Width);
        else
            Atom::store_device(value, c.destination,
                               c.base + static_cast<unsigned long>(w.sample + row) * S::Width + column, S::Width);
    }
};

template <class S, class Atom, class Copy, class Policy, bool Final = false>
struct SaveForward {
    using Scalar = typename Atom::Scalar;
    static TCNN_FUSED_INLINE void run(thread TrainingWorkspace<S, Atom, Copy>& w, device Scalar* pre,
                                       device Scalar* hidden, unsigned layer) {
        const unsigned long base = static_cast<unsigned long>(layer) * w.args.batch * S::Width;
        SaveContext<S, Atom, Copy> context{&w, pre, nullptr, base};
        if (w.full) {
            StaticFor<0, S::Fragments>::template run<ConvertStore<S, Atom, Copy, Identity, false>>(context);
            if (Final) {
                // No next layer consumes a threadgroup state. Preserve the
                // references' direct final-fragment stores on complete tiles.
                context.destination = hidden;
                StaticFor<0, S::Fragments>::template run<ConvertStore<S, Atom, Copy, Policy, false>>(context);
                return;
            }
            context.destination = hidden;
            StaticFor<0, S::Fragments>::template run<SaveActivated<S, Atom, Copy, Policy, true>>(context);
            if (!S::ResidentState)
                Copy::barrier();
            return;
        } else {
            // Native matrix stores cannot mask a partial sample tile. Only
            // boundary groups pay this extra shared-memory round trip.
            StaticFor<0, S::Fragments>::template run<ConvertStore<S, Atom, Copy, Identity, true>>(context);
            Copy::barrier();
            TileStore<S, Atom, Copy>::run(w, pre, base, TrainingMatrixLayout{1, S::Width});
            Copy::barrier();
        }
        context.destination = hidden;
        StaticFor<0, S::Fragments>::template run<SaveActivated<S, Atom, Copy, Policy, false>>(context);
        Copy::barrier();
        TileStore<S, Atom, Copy>::run(w, hidden, base, TrainingMatrixLayout{1, S::Width});
    }
};

template <class S, class Atom, class Copy, class Policy, bool SharedSaved, bool Direct = false>
struct DifferentiateStore {
    template <unsigned I>
    static TCNN_FUSED_INLINE void apply(thread SaveContext<S, Atom, Copy>& c) {
        thread TrainingWorkspace<S, Atom, Copy>& w = *c.workspace;
        const unsigned row = w.row + S::fragment_row(I), column = w.column + S::fragment_column(I);
        typename Atom::Operand saved, value;
        if (SharedSaved)
            Atom::load_shared(saved, w.states, row * S::Width + column, S::Width);
        else
            Atom::load_device(saved, c.saved, c.base + static_cast<unsigned long>(w.sample + row) * S::Width + column, S::Width);
        Atom::template differentiate<Policy>(value, w.accumulators[I], saved);
        if (S::ResidentState)
            w.resident[I] = value;
        if (!Direct || !S::ResidentState)
            Atom::store_shared(value, w.states, row * S::Width + column, S::Width);
        if (Direct)
            Atom::store_device(value, c.destination,
                               c.base + static_cast<unsigned long>(w.sample + row) * S::Width + column, S::Width);
    }
};

template <class S, class Atom, class Copy, class Policy>
struct SaveHiddenDelta {
    using Scalar = typename Atom::Scalar;
    static TCNN_FUSED_INLINE void run(thread TrainingWorkspace<S, Atom, Copy>& w, device const Scalar* pre,
                                       device const Scalar* hidden, device Scalar* delta, unsigned layer) {
        const unsigned long base = static_cast<unsigned long>(layer) * w.args.batch * S::Width;
        device const Scalar* saved = Policy::Kind == 2 || Policy::Kind == 4 ? pre : hidden;
        SaveContext<S, Atom, Copy> context{&w, delta, saved, base};
        if (w.full) {
            StaticFor<0, S::Fragments>::template run<DifferentiateStore<S, Atom, Copy, Policy, false, true>>(context);
            if (!S::ResidentState)
                Copy::barrier();
            return;
        } else {
            for (unsigned i = w.tid; i < S::Samples * S::Width; i += S::Threads) {
                const unsigned sample = w.sample + i / S::Width;
                w.states[i] = sample < w.args.batch
                    ? saved[base + static_cast<unsigned long>(sample) * S::Width + i % S::Width] : Scalar(0);
            }
            Copy::barrier();
            // Each SIMD-group reads/writes its own disjoint fragments here.
            StaticFor<0, S::Fragments>::template run<DifferentiateStore<S, Atom, Copy, Policy, true>>(context);
        }
        Copy::barrier();
        // The standalone references use reverse layer order. The established
        // port ABI uses forward order, consumed by its parameter-gradient pass.
        TileStore<S, Atom, Copy>::run(w, delta, base, TrainingMatrixLayout{1, S::Width});
    }
};

template <class S, class Atom, class Copy>
struct SaveInputDelta {
    using Scalar = typename Atom::Scalar;
    static TCNN_FUSED_INLINE void run(thread TrainingWorkspace<S, Atom, Copy>& w, device Scalar* dx) {
        SaveContext<S, Atom, Copy> context{&w, dx, nullptr, 0};
        if (w.full && w.args.dx.row_stride == 1 && w.args.dx.sample_stride == S::Width) {
            StaticFor<0, S::Fragments>::template run<ConvertStore<S, Atom, Copy, Identity, false>>(context);
        } else {
            StaticFor<0, S::Fragments>::template run<ConvertStore<S, Atom, Copy, Identity, true>>(context);
            Copy::barrier();
            TileStore<S, Atom, Copy>::run(w, dx, 0, w.args.dx);
        }
    }
};

template <class S, class Atom, class Copy, bool Backward>
TCNN_FUSED_INLINE void initialize(thread TrainingWorkspace<S, Atom, Copy>& w,
                                   threadgroup typename Atom::Scalar* scratch,
                                   device const typename Atom::Scalar* weights,
                                   TrainingArguments args, unsigned group, unsigned tid, unsigned sg) {
    using Memory = TrainingScratch<S, sizeof(typename Atom::Scalar), Backward>;
    w.input_panel = scratch + Memory::InputOffset;
    w.weight_panel = scratch + Memory::WeightOffset;
    w.states = scratch + Memory::StateOffset;
    w.weights = weights;
    w.args = args;
    w.sample = group * S::Samples;
    w.tid = tid;
    w.row = S::first_row(sg);
    w.column = S::first_column(sg);
    w.full = w.sample < args.batch && args.batch - w.sample >= S::Samples;
}

} // namespace training_detail

template <class S, class Atom, class Copy, class Hidden, class Output>
struct TrainingForward {
    using Scalar = typename Atom::Scalar;
    static TCNN_FUSED_INLINE void run(device const Scalar* input, device const Scalar* weights,
                                       device Scalar* pre, device Scalar* hidden, threadgroup Scalar* scratch,
                                       TrainingArguments args, unsigned group, unsigned tid, unsigned sg) {
        TrainingWorkspace<S, Atom, Copy> w;
        training_detail::initialize<S, Atom, Copy, false>(w, scratch, weights, args, group, tid, sg);
        training_detail::InputSource<S, Atom, Copy> first{input};
        training_detail::HiddenSource resident;
        for (unsigned layer = 0;; ++layer) {
            if (layer == 0)
                training_detail::Mainloop<S, Atom, Copy, false>::run(w, first, layer);
            else
                training_detail::Mainloop<S, Atom, Copy, false>::run(w, resident, layer);
            if (layer == args.hidden_layers) {
                training_detail::SaveForward<S, Atom, Copy, Output, true>::run(w, pre, hidden, layer);
                break;
            }
            training_detail::SaveForward<S, Atom, Copy, Hidden>::run(w, pre, hidden, layer);
        }
    }
};

template <class S, class Atom, class Copy, class Hidden, class Output>
struct TrainingBackward {
    using Scalar = typename Atom::Scalar;
    static TCNN_FUSED_INLINE void run(device const Scalar* weights, device const Scalar* pre,
                                       device const Scalar* hidden, device const Scalar* output,
                                       device const Scalar* incoming, device Scalar* delta, device Scalar* dx,
                                       threadgroup Scalar* scratch, TrainingArguments args,
                                       unsigned group, unsigned tid, unsigned sg) {
        TrainingWorkspace<S, Atom, Copy> w;
        training_detail::initialize<S, Atom, Copy, true>(w, scratch, weights, args, group, tid, sg);
        training_detail::IncomingSource<S, Atom, Copy, Output> first{pre, output, incoming, delta};
        training_detail::HiddenSource resident;
        if (args.hidden_layers == 0 && !args.write_dx) {
            // Weight-gradient-only differentiation of a network with no hidden
            // layers still needs the output delta, but no matrix product.
            for (unsigned k = 0; k < S::Width; k += S::KPanel)
                first.load(w, k);
            return;
        }
        for (unsigned layer = args.hidden_layers;; --layer) {
            if (layer == args.hidden_layers)
                training_detail::Mainloop<S, Atom, Copy, true>::run(w, first, layer);
            else
                training_detail::Mainloop<S, Atom, Copy, true>::run(w, resident, layer);
            if (layer == 0) {
                training_detail::SaveInputDelta<S, Atom, Copy>::run(w, dx);
                break;
            }
            training_detail::SaveHiddenDelta<S, Atom, Copy, Hidden>::run(w, pre, hidden, delta, layer - 1);
            if (layer == 1 && !args.write_dx)
                break;
        }
    }
};

} // namespace tcnn_fused
#endif
