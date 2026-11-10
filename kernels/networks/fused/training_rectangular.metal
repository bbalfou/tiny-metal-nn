#ifndef TCNN_FUSED_TRAINING_RECTANGULAR_METAL
#define TCNN_FUSED_TRAINING_RECTANGULAR_METAL
#include "training_kernel.metal"
#include "training_rectangular_traits.metal"

namespace tcnn_fused {
namespace rectangular_detail {

// The matrix work is always an 8x8 SIMD-group MMA. Packed cooperative vectors
// cover contiguous ranges even for scalar-aligned views; scalar gathers are
// confined to arbitrary element strides and vector/tile boundaries.
template <class S, class Atom, class Copy, bool Backward>
struct WeightPanel {
    static TCNN_FUSED_INLINE void load(thread TrainingWorkspace<S, Atom, Copy>& w,
        unsigned long base, unsigned rows, unsigned columns, unsigned output_feature, unsigned k) {
        enum : unsigned { V = Copy::VectorElements, Vectors = S::Width * S::KPanel / V };
        for (unsigned v = w.tid; v < Vectors; v += S::Threads) {
            const unsigned i = v * V;
            const unsigned long row = Backward ? static_cast<unsigned long>(k) + i / S::Width
                                               : static_cast<unsigned long>(output_feature) + i / S::KPanel;
            const unsigned long column = Backward ? static_cast<unsigned long>(output_feature) + i % S::Width
                                                  : static_cast<unsigned long>(k) + i % S::KPanel;
            const unsigned long index = base + row * columns + column;
            if (row < rows && column + V <= columns)
                Copy::to_shared(w.weight_panel, i, w.weights, index);
            else
                for (unsigned j = 0; j < V; ++j)
                    w.weight_panel[i + j] = row < rows && column + j < columns
                        ? w.weights[index + j] : typename Atom::Scalar(0);
        }
    }
};

template <class S, class Atom, class Copy, bool Backward>
struct Mainloop {
    struct ResidentContext {
        thread TrainingWorkspace<S, Atom, Copy>* workspace;
        unsigned long base;
        unsigned rows, columns, output_feature;
    };
    struct ResidentPanel {
        template <unsigned Panel>
        static TCNN_FUSED_INLINE void apply(thread ResidentContext& c) {
            thread TrainingWorkspace<S, Atom, Copy>& w = *c.workspace;
            enum : unsigned { K = Panel * S::KPanel };
            WeightPanel<S, Atom, Copy, Backward>::load(w, c.base, c.rows, c.columns, c.output_feature, K);
            Copy::barrier();
            training_detail::PanelContext<S, Atom, Copy> panel{&w, K};
            StaticFor<0, S::KTiles>::template run<training_detail::KTile<S, Atom, Copy, Backward, false>>(panel);
            Copy::barrier();
        }
    };
    template <class Source>
    static TCNN_FUSED_INLINE void run(thread TrainingWorkspace<S, Atom, Copy>& w, thread Source& source,
        unsigned long base, unsigned rows, unsigned columns, unsigned output_feature) {
        StaticFor<0, S::Fragments>::template run<training_detail::Zero<S, Atom, Copy>>(w);
        if (!Source::First) {
            // Every resident source is a complete Width-column hidden state.
            // Static indices preserve the same register state as inference.
            ResidentContext c{&w, base, rows, columns, output_feature};
            StaticFor<0, S::Width / S::KPanel>::template run<ResidentPanel>(c);
            return;
        }
        const unsigned extent = Backward ? rows : columns;
#pragma clang loop unroll(disable)
        for (unsigned long panel_k = 0; panel_k < extent; panel_k += S::KPanel) {
            const unsigned k = static_cast<unsigned>(panel_k);
            source.load(w, k);
            WeightPanel<S, Atom, Copy, Backward>::load(w, base, rows, columns, output_feature, k);
            Copy::barrier();
            training_detail::PanelContext<S, Atom, Copy> panel{&w, k};
            StaticFor<0, S::KTiles>::template run<training_detail::KTile<S, Atom, Copy, Backward, true>>(panel);
            Copy::barrier();
        }
    }
};

template <class S, class Atom, class Copy>
struct MatrixSource {
    using Scalar = typename Atom::Scalar;
    enum : unsigned { First = 1 };
    device const Scalar* source;
    unsigned features;
    TrainingMatrixLayout layout;
    TCNN_FUSED_INLINE void load(thread TrainingWorkspace<S, Atom, Copy>& w, unsigned k) {
        enum : unsigned { V = Copy::VectorElements, VectorsPerRow = S::KPanel / V };
        if (layout.sample_stride == 1 && layout.row_stride != 1) {
            // Read adjacent samples from SoA and transpose into the existing
            // shared panel; feature and batch tails retain zero padding.
            for (unsigned i = w.tid; i < S::Samples * S::KPanel; i += S::Threads) {
                const unsigned row = i % S::Samples, column = i / S::Samples;
                const unsigned sample = w.sample + row;
                const unsigned long feature = static_cast<unsigned long>(k) + column;
                w.input_panel[row * S::KPanel + column] = sample < w.args.batch && feature < features
                    ? source[static_cast<unsigned long>(sample) + feature * layout.row_stride] : Scalar(0);
            }
            return;
        }
        for (unsigned v = w.tid; v < S::Samples * VectorsPerRow; v += S::Threads) {
            const unsigned sample = w.sample + v / VectorsPerRow;
            const unsigned long feature = static_cast<unsigned long>(k) + (v % VectorsPerRow) * V;
            const unsigned long index = static_cast<unsigned long>(sample) * layout.sample_stride + feature * layout.row_stride;
            if (sample < w.args.batch && layout.row_stride == 1 && feature + V <= features)
                Copy::to_shared(w.input_panel, v * V, source, index);
            else
                for (unsigned j = 0; j < V; ++j)
                    w.input_panel[v * V + j] = sample < w.args.batch && feature + j < features
                        ? source[index + static_cast<unsigned long>(j) * layout.row_stride] : Scalar(0);
        }
    }
};

template <class S, class Atom, class Copy, class Output>
struct IncomingSource {
    using Scalar = typename Atom::Scalar;
    enum : unsigned { First = 1, InputForm = Output::Kind == 2 || Output::Kind == 4 };
    device const Scalar* pre;
    device const Scalar* output;
    device const Scalar* incoming;
    device Scalar* delta;
    unsigned outputs;
    unsigned long base;
    TCNN_FUSED_INLINE void load(thread TrainingWorkspace<S, Atom, Copy>& w, unsigned k) {
        for (unsigned i = w.tid; i < S::Samples * S::KPanel; i += S::Threads) {
            const unsigned sample = w.sample + i / S::KPanel;
            const unsigned long feature = static_cast<unsigned long>(k) + i % S::KPanel;
            Scalar value = Scalar(0);
            if (sample < w.args.batch && feature < outputs) {
                const unsigned long saved = base + static_cast<unsigned long>(sample) * outputs + feature;
                const Scalar activation = InputForm ? pre[saved] : output[
                    static_cast<unsigned long>(sample) * w.args.output.sample_stride +
                    static_cast<unsigned long>(feature) * w.args.output.row_stride];
                value = Atom::template gradient<Output>(activation, float(incoming[
                    static_cast<unsigned long>(sample) * w.args.incoming.sample_stride +
                    static_cast<unsigned long>(feature) * w.args.incoming.row_stride]));
                delta[saved] = value;
            }
            w.input_panel[i] = value;
        }
    }
};

template <class S, class Atom, class Copy>
struct Export {
    using Scalar = typename Atom::Scalar;
    static TCNN_FUSED_INLINE void run(thread TrainingWorkspace<S, Atom, Copy>& w,
        threadgroup const Scalar* source, device Scalar* destination, unsigned long base,
        unsigned features, unsigned feature_base, TrainingMatrixLayout layout) {
        enum : unsigned { V = Copy::VectorElements, VectorsPerRow = S::Width / V };
        if (layout.sample_stride == 1 && layout.row_stride != 1) {
            for (unsigned i = w.tid; i < S::Samples * S::Width; i += S::Threads) {
                const unsigned row = i % S::Samples, column = i / S::Samples;
                const unsigned sample = w.sample + row;
                const unsigned long feature = static_cast<unsigned long>(feature_base) + column;
                if (sample < w.args.batch && feature < features)
                    destination[base + static_cast<unsigned long>(sample) + feature * layout.row_stride] =
                        source[row * S::Width + column];
            }
            return;
        }
        for (unsigned v = w.tid; v < S::Samples * VectorsPerRow; v += S::Threads) {
            const unsigned sample = w.sample + v / VectorsPerRow;
            const unsigned long feature = static_cast<unsigned long>(feature_base) + (v % VectorsPerRow) * V;
            const unsigned long index = base + static_cast<unsigned long>(sample) * layout.sample_stride + feature * layout.row_stride;
            if (sample < w.args.batch && layout.row_stride == 1 && feature + V <= features)
                Copy::to_device(destination, index, source, v * V);
            else if (sample < w.args.batch)
                for (unsigned j = 0; j < V; ++j)
                    if (feature + j < features)
                        destination[index + static_cast<unsigned long>(j) * layout.row_stride] = source[v * V + j];
        }
    }
};

template <class S, class Atom, class Copy>
struct StoreContext {
    thread TrainingWorkspace<S, Atom, Copy>* workspace;
    threadgroup typename Atom::Scalar* target;
};

// Full matrix fragments can write saved state directly to device memory.
// Included inside rectangular_detail, after StoreContext. Full 8x8 fragments
// have disjoint sample/feature ranges; tails keep the original masked path.
template <class S, class Atom, class Copy>
struct DirectForwardContext {
    thread TrainingWorkspace<S, Atom, Copy>* workspace;
    device typename Atom::Scalar* pre;
    device typename Atom::Scalar* hidden;
    unsigned long base;
    unsigned features, feature_base;
};
template <class S, class Atom, class Copy, class Policy, bool Final>
struct DirectForwardStore {
    template <unsigned I>
    static TCNN_FUSED_INLINE void apply(thread DirectForwardContext<S, Atom, Copy>& c) {
        thread TrainingWorkspace<S, Atom, Copy>& w=*c.workspace;
        const unsigned row=w.row+S::fragment_row(I), local=w.column+S::fragment_column(I);
        const unsigned feature=c.feature_base+local;
        typename Atom::Operand value;
        if (feature + 8 <= c.features) {
            const unsigned long offset=c.base+static_cast<unsigned long>(w.sample+row)*c.features+feature;
            Atom::template convert<Identity>(value,w.accumulators[I]);
            Atom::store_device(value,c.pre,offset,c.features);
            Atom::template convert<Policy>(value,w.accumulators[I]);
            Atom::store_device(value,c.hidden,offset,c.features);
        } else {
            Atom::template convert<Policy>(value,w.accumulators[I]);
        }
        if (!Final) {
            if (S::ResidentState) w.resident[I]=value;
            else Atom::store_shared(value,w.states,row*S::Width+local,S::Width);
        }
    }
};

template <class S, class Atom, class Copy, class Policy, bool Resident>
struct Convert {
    template <unsigned I>
    static TCNN_FUSED_INLINE void apply(thread StoreContext<S, Atom, Copy>& c) {
        thread TrainingWorkspace<S, Atom, Copy>& w = *c.workspace;
        typename Atom::Operand value;
        Atom::template convert<Policy>(value, w.accumulators[I]);
        if (Resident && S::ResidentState)
            w.resident[I] = value;
        Atom::store_shared(value, c.target,
            (w.row + S::fragment_row(I)) * S::Width + w.column + S::fragment_column(I), S::Width);
    }
};

template <class S, class Atom, class Copy, class Policy, bool Final>
struct SaveForward {
    using Scalar = typename Atom::Scalar;
    static TCNN_FUSED_INLINE void run(thread TrainingWorkspace<S, Atom, Copy>& w,
        threadgroup Scalar* temporary, device Scalar* pre, device Scalar* hidden,
        unsigned long base, unsigned features, unsigned feature_base) {
        if (S::Width >= 64 && sizeof(Scalar) == 2 &&
            w.full && features % 8 == 0) {
            DirectForwardContext<S, Atom, Copy> direct{&w, pre, hidden, base, features, feature_base};
            StaticFor<0, S::Fragments>::template run<DirectForwardStore<S, Atom, Copy, Policy, Final>>(direct);
            if (!Final && !S::ResidentState) Copy::barrier();
            return;
        }
        StoreContext<S, Atom, Copy> c{&w, temporary};
        StaticFor<0, S::Fragments>::template run<Convert<S, Atom, Copy, Identity, false>>(c);
        Copy::barrier();
        Export<S, Atom, Copy>::run(w, temporary, pre, base, features, feature_base, {1, features});
        Copy::barrier();
        c.target = Final ? temporary : w.states;
        StaticFor<0, S::Fragments>::template run<Convert<S, Atom, Copy, Policy, !Final>>(c);
        Copy::barrier();
        Export<S, Atom, Copy>::run(w, c.target, hidden, base, features, feature_base, {1, features});
        // A later output tile may immediately reuse temporary or input panels.
        Copy::barrier();
    }
};

template <class S, class Atom, class Copy, class Policy>
struct Differentiate {
    template <unsigned I>
    static TCNN_FUSED_INLINE void apply(thread StoreContext<S, Atom, Copy>& c) {
        thread TrainingWorkspace<S, Atom, Copy>& w = *c.workspace;
        const unsigned offset = (w.row + S::fragment_row(I)) * S::Width + w.column + S::fragment_column(I);
        typename Atom::Operand saved, value;
        Atom::load_shared(saved, c.target, offset, S::Width);
        Atom::template differentiate<Policy>(value, w.accumulators[I], saved);
        if (S::ResidentState)
            w.resident[I] = value;
        Atom::store_shared(value, w.states, offset, S::Width);
    }
};

template <class S, class Atom, class Copy, class Policy>
struct SaveHiddenDelta {
    using Scalar = typename Atom::Scalar;
    static TCNN_FUSED_INLINE void run(thread TrainingWorkspace<S, Atom, Copy>& w,
        threadgroup Scalar* temporary, device const Scalar* pre, device const Scalar* hidden,
        device Scalar* delta, unsigned long base) {
        device const Scalar* saved = Policy::Kind == 2 || Policy::Kind == 4 ? pre : hidden;
        if (S::Width >= 64 && sizeof(Scalar) == 2 && w.full) {
            training_detail::SaveContext<S, Atom, Copy> direct{&w, delta, saved, base};
            StaticFor<0, S::Fragments>::template run<
                training_detail::DifferentiateStore<S, Atom, Copy, Policy, false, true>>(direct);
            if (!S::ResidentState) Copy::barrier();
            return;
        }


        for (unsigned i = w.tid; i < S::Samples * S::Width; i += S::Threads) {
            const unsigned sample = w.sample + i / S::Width;
            temporary[i] = sample < w.args.batch
                ? saved[base + static_cast<unsigned long>(sample) * S::Width + i % S::Width] : Scalar(0);
        }
        Copy::barrier();
        StoreContext<S, Atom, Copy> c{&w, temporary};
        StaticFor<0, S::Fragments>::template run<Differentiate<S, Atom, Copy, Policy>>(c);
        Copy::barrier();
        Export<S, Atom, Copy>::run(w, w.states, delta, base, S::Width, 0, {1, S::Width});
        Copy::barrier();
    }
};

template <class S, class Atom, class Copy>
TCNN_FUSED_INLINE void initialize(thread TrainingWorkspace<S, Atom, Copy>& w,
    threadgroup typename Atom::Scalar* scratch, device const typename Atom::Scalar* weights,
    TrainingArguments args, unsigned group, unsigned tid, unsigned sg) {
    using Memory = RectangularScratch<S, sizeof(typename Atom::Scalar)>;
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

} // namespace rectangular_detail

template <class S, class Atom, class Copy, class Hidden, class Output>
struct RectangularForward {
    using Scalar = typename Atom::Scalar;
    static TCNN_FUSED_INLINE void run(device const Scalar* input, device const Scalar* weights,
        device Scalar* pre, device Scalar* hidden, threadgroup Scalar* scratch,
        RectangularArguments args, unsigned group, unsigned tid, unsigned sg) {
        TrainingWorkspace<S, Atom, Copy> w;
        rectangular_detail::initialize(w, scratch, weights, args.training, group, tid, sg);
        using Memory = RectangularScratch<S, sizeof(Scalar)>;
        threadgroup Scalar* temporary = scratch + Memory::TemporaryOffset;
        rectangular_detail::MatrixSource<S, Atom, Copy> first{input, args.inputs, args.training.input};
        training_detail::HiddenSource resident;
        for (unsigned layer = 0;; ++layer) {
            const bool final = layer == args.training.hidden_layers;
            const unsigned rows = args.layer_rows(layer, S::Width), columns = args.layer_inputs(layer, S::Width);
            const unsigned long weight_base = args.weight_offset(layer, S::Width);
            const unsigned long state_base = args.state_offset(layer, S::Width);
            for (unsigned long feature_tile = 0; feature_tile < rows; feature_tile += S::Width) {
                const unsigned feature = static_cast<unsigned>(feature_tile);
                if (layer == 0)
                    rectangular_detail::Mainloop<S, Atom, Copy, false>::run(w, first, weight_base, rows, columns, feature);
                else
                    rectangular_detail::Mainloop<S, Atom, Copy, false>::run(w, resident, weight_base, rows, columns, feature);
                if (final)
                    rectangular_detail::SaveForward<S, Atom, Copy, Output, true>::run(
                        w, temporary, pre, hidden, state_base, rows, feature);
                else
                    rectangular_detail::SaveForward<S, Atom, Copy, Hidden, false>::run(
                        w, temporary, pre, hidden, state_base, rows, feature);
            }
            if (final) break;
        }
    }
};

template <class S, class Atom, class Copy, class Hidden, class Output>
struct RectangularBackward {
    using Scalar = typename Atom::Scalar;
    static TCNN_FUSED_INLINE void run(device const Scalar* weights, device const Scalar* pre,
        device const Scalar* hidden, device const Scalar* output, device const Scalar* incoming,
        device Scalar* delta, device Scalar* dx, threadgroup Scalar* scratch,
        RectangularArguments args, unsigned group, unsigned tid, unsigned sg) {
        TrainingWorkspace<S, Atom, Copy> w;
        rectangular_detail::initialize(w, scratch, weights, args.training, group, tid, sg);
        using Memory = RectangularScratch<S, sizeof(Scalar)>;
        threadgroup Scalar* temporary = scratch + Memory::TemporaryOffset;
        rectangular_detail::IncomingSource<S, Atom, Copy, Output> first{
            pre, output, incoming, delta, args.outputs, args.state_offset(args.training.hidden_layers, S::Width)};
        training_detail::HiddenSource resident;
        if (args.training.hidden_layers == 0 && !args.training.write_dx) {
            for (unsigned long k = 0; k < args.outputs; k += S::KPanel)
                first.load(w, static_cast<unsigned>(k));
            return;
        }
        for (unsigned layer = args.training.hidden_layers;; --layer) {
            const unsigned rows = args.layer_rows(layer, S::Width), columns = args.layer_inputs(layer, S::Width);
            const unsigned long weight_base = args.weight_offset(layer, S::Width);
            for (unsigned long feature_tile = 0; feature_tile < columns; feature_tile += S::Width) {
                const unsigned feature = static_cast<unsigned>(feature_tile);
                if (layer == args.training.hidden_layers)
                    rectangular_detail::Mainloop<S, Atom, Copy, true>::run(w, first, weight_base, rows, columns, feature);
                else
                    rectangular_detail::Mainloop<S, Atom, Copy, true>::run(w, resident, weight_base, rows, columns, feature);
                if (layer == 0) {
                    rectangular_detail::StoreContext<S, Atom, Copy> c{&w, temporary};
                    StaticFor<0, S::Fragments>::template run<rectangular_detail::Convert<S, Atom, Copy, Identity, false>>(c);
                    Copy::barrier();
                    rectangular_detail::Export<S, Atom, Copy>::run(w, temporary, dx, 0, args.inputs, feature, args.training.dx);
                    Copy::barrier();
                } else {
                    rectangular_detail::SaveHiddenDelta<S, Atom, Copy, Hidden>::run(
                        w, temporary, pre, hidden, delta, args.state_offset(layer - 1, S::Width));
                }
            }
            if (layer == 0 || (layer == 1 && !args.training.write_dx)) break;
        }
    }
};

} // namespace tcnn_fused
#endif
