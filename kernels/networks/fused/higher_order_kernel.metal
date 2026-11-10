#ifndef TCNN_FUSED_HIGHER_ORDER_KERNEL_METAL
#define TCNN_FUSED_HIGHER_ORDER_KERNEL_METAL
#include "higher_order_traits.metal"
#include "traits.metal"
#ifdef __METAL_VERSION__
#include "higher_order_atoms.metal"
#endif

namespace tcnn_fused {

template <class S, class Atom>
struct HigherOrderWorkspace {
    using Scalar = typename Atom::Scalar;
    typename Atom::Operand state[S::Fragments], second_state[S::Fragments];
    typename Atom::Accumulator accumulators[S::Fragments], second_accumulators[S::Fragments];
    threadgroup Scalar* weights;
    threadgroup Scalar* panel;
    threadgroup Scalar* second_panel;
    device const Scalar* parameters;
    HigherOrderArguments args;
    unsigned long sample;
    unsigned tid, sg;
};

namespace higher_order_detail {

inline unsigned long saved_base(HigherOrderArguments a, unsigned width, unsigned layer) {
    return static_cast<unsigned long>(layer) * width * a.batch;
}
inline unsigned long weight_base(HigherOrderArguments a, unsigned width, unsigned layer) {
    return layer == 0 ? 0 : static_cast<unsigned long>(a.inputs) * width +
        static_cast<unsigned long>(layer - 1) * width * width;
}

template <class S, class Atom, class Copy>
struct DirectionSource {
    using Scalar = typename Atom::Scalar;
    device const Scalar* direction;
    TCNN_FUSED_INLINE void load(thread HigherOrderWorkspace<S, Atom>& w, unsigned long k) {
        enum : unsigned { V = Copy::VectorElements, VectorsPerRow = S::KPanel / V };
        for (unsigned v = w.tid; v < S::Samples * VectorsPerRow; v += S::Threads) {
            const unsigned row = v / VectorsPerRow, column = (v % VectorsPerRow) * V;
            const unsigned long sample = w.sample + row;
            if (sample < w.args.batch && w.args.direction.row_stride == 1 && k + column + V <= w.args.inputs) {
                Copy::to_shared(w.panel, v * V, direction,
                    static_cast<unsigned long>(sample) * w.args.direction.sample_stride + k + column);
            } else {
                for (unsigned j = 0; j < V; ++j)
                    w.panel[v * V + j] = sample < w.args.batch && k + column + j < w.args.inputs
                        ? direction[static_cast<unsigned long>(sample) * w.args.direction.sample_stride +
                                    static_cast<unsigned long>(k + column + j) * w.args.direction.row_stride]
                        : Scalar(0);
            }
        }
    }
};

struct RegisterSource {
    template <class Workspace>
    TCNN_FUSED_INLINE void load(thread Workspace&, unsigned long) {}
};

template <class S, class Atom, class Math, class Output>
struct SeedSource {
    using Scalar = typename Atom::Scalar;
    device const Scalar* pre;
    device const Scalar* hidden;
    device const Scalar* tangent_pre;
    device const Scalar* incoming;
    device Scalar* delta;
    device Scalar* tangent_delta;
    TCNN_FUSED_INLINE void load(thread HigherOrderWorkspace<S, Atom>& w, unsigned long k) {
        const unsigned long base = saved_base(w.args, S::Width, w.args.hidden_layers);
        for (unsigned i = w.tid; i < S::Samples * S::KPanel; i += S::Threads) {
            const unsigned long sample = w.sample + i / S::KPanel, feature = k + i % S::KPanel;
            Scalar primal = Scalar(0), tangent = Scalar(0);
            if (sample < w.args.batch && feature < w.args.outputs) {
                const unsigned long saved = base + static_cast<unsigned long>(sample) * w.args.outputs + feature;
                const Scalar first = Math::template first<Output>(pre[saved], hidden[saved]);
                const Scalar second = Math::template second<Output>(pre[saved], hidden[saved]);
                const Scalar adjoint = incoming[static_cast<unsigned long>(sample) * w.args.incoming.sample_stride +
                                                 static_cast<unsigned long>(feature) * w.args.incoming.row_stride];
                primal = Scalar(0.0f * float(first) + float(adjoint) * float(second) * float(tangent_pre[saved]));
                tangent = Scalar(float(adjoint) * float(first));
                delta[saved] = primal;
                tangent_delta[saved] = tangent;
            }
            w.panel[i] = primal;
            w.second_panel[i] = tangent;
        }
    }
};

template <class S, class Atom, class Copy, bool Backward>
struct WeightPanel {
    static TCNN_FUSED_INLINE void load(thread HigherOrderWorkspace<S, Atom>& w, unsigned long base,
                                       unsigned rows, unsigned columns, unsigned long feature_base, unsigned long k) {
        using Scalar = typename Atom::Scalar;
        enum : unsigned { V = Copy::VectorElements, Vectors = S::Width * S::KPanel / V };
        for (unsigned v = w.tid; v < Vectors; v += S::Threads) {
            const unsigned major = v * V / (Backward ? S::Width : S::KPanel);
            const unsigned minor = v * V % (Backward ? S::Width : S::KPanel);
            const unsigned long row = Backward ? k + major : feature_base + major;
            const unsigned long column = Backward ? feature_base + minor : k + minor;
            if (row < rows && column + V <= columns) {
                Copy::to_shared(w.weights, v * V, w.parameters,
                    base + static_cast<unsigned long>(row) * columns + column);
            } else {
                for (unsigned j = 0; j < V; ++j)
                    w.weights[v * V + j] = row < rows && column + j < columns
                        ? w.parameters[base + static_cast<unsigned long>(row) * columns + column + j] : Scalar(0);
            }
        }
    }
};

template <class S, class Atom, bool Pair>
struct Zero {
    template <unsigned I>
    static TCNN_FUSED_INLINE void apply(thread HigherOrderWorkspace<S, Atom>& w) {
        Atom::zero(w.accumulators[I]);
        if (Pair) Atom::zero(w.second_accumulators[I]);
    }
};

template <class S, class Atom>
struct FragmentContext {
    thread HigherOrderWorkspace<S, Atom>* w;
    thread typename Atom::Operand* a;
    thread typename Atom::Operand* second_a;
    thread typename Atom::Operand* b;
    unsigned long k;
};

template <class S, class Atom, bool Pair, unsigned K>
struct LoadA {
    template <unsigned Row>
    static TCNN_FUSED_INLINE void apply(thread FragmentContext<S, Atom>& c) {
        const unsigned sample = c.w->sg * S::SamplesPerSimdgroup + Row * S::Tile;
        Atom::load_shared(c.a[Row], c.w->panel, sample * S::KPanel + K * S::Tile, S::KPanel);
        if (Pair) Atom::load_shared(c.second_a[Row], c.w->second_panel,
                                   sample * S::KPanel + K * S::Tile, S::KPanel);
    }
};

template <class S, class Atom, bool Backward, unsigned K>
struct LoadB {
    template <unsigned Column>
    static TCNN_FUSED_INLINE void apply(thread FragmentContext<S, Atom>& c) {
        Atom::load_shared(c.b[Column], c.w->weights,
            Backward ? K * S::Tile * S::Width + Column * S::Tile
                     : Column * S::Tile * S::KPanel + K * S::Tile,
            Backward ? S::Width : S::KPanel, !Backward);
    }
};

template <class S, class Atom, bool Pair, bool Registers, unsigned K>
struct Multiply {
    template <unsigned I>
    static TCNN_FUSED_INLINE void apply(thread FragmentContext<S, Atom>& c) {
        if (Registers) {
            const unsigned input = I / S::ColumnTiles * S::ColumnTiles + unsigned(c.k / S::Tile) + K;
            Atom::mma(c.w->accumulators[I], c.w->state[input], c.b[I % S::ColumnTiles]);
            if (Pair) Atom::mma(c.w->second_accumulators[I], c.w->second_state[input], c.b[I % S::ColumnTiles]);
        } else {
            Atom::mma(c.w->accumulators[I], c.a[I / S::ColumnTiles], c.b[I % S::ColumnTiles]);
            if (Pair) Atom::mma(c.w->second_accumulators[I], c.second_a[I / S::ColumnTiles], c.b[I % S::ColumnTiles]);
        }
    }
};

template <class S, class Atom>
struct TileContext { thread HigherOrderWorkspace<S, Atom>* w; unsigned long k; };

template <class S, class Atom, bool Backward, bool Pair, bool Registers>
struct KTile {
    template <unsigned K>
    static TCNN_FUSED_INLINE void apply(thread TileContext<S, Atom>& c) {
        typename Atom::Operand a[S::RowTiles], second_a[Pair ? unsigned(S::RowTiles) : 1u], b[S::ColumnTiles];
        FragmentContext<S, Atom> f{c.w, a, second_a, b, c.k};
        if (!Registers) StaticFor<0, S::RowTiles>::template run<LoadA<S, Atom, Pair, K>>(f);
        StaticFor<0, S::ColumnTiles>::template run<LoadB<S, Atom, Backward, K>>(f);
        StaticFor<0, S::Fragments>::template run<Multiply<S, Atom, Pair, Registers, K>>(f);
    }
};

template <class S, class Atom, class Source>
struct PanelContext {
    thread HigherOrderWorkspace<S, Atom>* w;
    thread Source* source;
    unsigned long base;
    unsigned rows, columns;
    unsigned long feature;
};

template <class S, class Atom, class Copy, class Source, bool Backward, bool Pair, bool Registers>
struct Panel {
    static TCNN_FUSED_INLINE void run(thread PanelContext<S, Atom, Source>& c, unsigned long k) {
        if (!Registers) c.source->load(*c.w, k);
        WeightPanel<S, Atom, Copy, Backward>::load(*c.w, c.base, c.rows, c.columns, c.feature, k);
        Copy::barrier();
        TileContext<S, Atom> tile{c.w, k};
        StaticFor<0, S::KTiles>::template run<KTile<S, Atom, Backward, Pair, Registers>>(tile);
        // Retire every B-panel reader before the cooperative refill.
        Copy::barrier();
    }
    template <unsigned K>
    static TCNN_FUSED_INLINE void apply(thread PanelContext<S, Atom, Source>& c) { run(c, K * S::KPanel); }
};

template <class S, class Atom, class Copy, bool Backward, bool Pair, bool Registers>
struct Mainloop {
    template <class Source>
    static TCNN_FUSED_INLINE void run(thread HigherOrderWorkspace<S, Atom>& w, thread Source& source,
                                       unsigned long base, unsigned rows, unsigned columns, unsigned long feature = 0) {
        StaticFor<0, S::Fragments>::template run<Zero<S, Atom, Pair>>(w);
        PanelContext<S, Atom, Source> context{&w, &source, base, rows, columns, feature};
        using PanelType = Panel<S, Atom, Copy, Source, Backward, Pair, Registers>;
        if (Registers) {
            // All hidden-state indices become compile-time constants. The A
            // fragments survive both layer changes and final-feature tiles.
            StaticFor<0, S::Width / S::KPanel>::template run<PanelType>(context);
        } else {
#pragma clang loop unroll(disable)
            for (unsigned long k = 0; k < (Backward ? rows : columns); k += S::KPanel)
                PanelType::run(context, k);
        }
    }
};

template <class S, class Atom>
struct EpilogueContext {
    thread HigherOrderWorkspace<S, Atom>* w;
    device const typename Atom::Scalar* pre;
    device const typename Atom::Scalar* hidden;
    device const typename Atom::Scalar* tangent_pre_input;
    device typename Atom::Scalar* first;
    device typename Atom::Scalar* second;
    device typename Atom::Scalar* external;
    unsigned layer, features;
    unsigned long feature_base;
};

template <class S, unsigned I>
struct EpilogueIndex {
    static TCNN_FUSED_INLINE unsigned long sample(unsigned long group_sample, unsigned i) {
        return group_sample + i / 64 * S::SamplesPerSimdgroup + I / S::ColumnTiles * 8 + (i % 64) / 8;
    }
    static TCNN_FUSED_INLINE unsigned long feature(unsigned long feature_base, unsigned i) {
        return feature_base + I % S::ColumnTiles * 8 + i % 8;
    }
};

template <class S, class Atom, class Copy, class Math, class Policy, bool Final>
struct TangentEpilogue {
    template <unsigned I>
    static TCNN_FUSED_INLINE void apply(thread EpilogueContext<S, Atom>& c) {
        using Scalar = typename Atom::Scalar;
        thread HigherOrderWorkspace<S, Atom>& w = *c.w;
        typename Atom::Operand result;
        Atom::template convert<Identity>(result, w.accumulators[I]);
        const unsigned long base = saved_base(w.args, S::Width, c.layer);
        const unsigned long sample = w.sample + w.sg * S::SamplesPerSimdgroup + I / S::ColumnTiles * 8;
        const unsigned long feature = c.feature_base + I % S::ColumnTiles * 8;
        if (w.args.batch - w.sample >= S::Samples && feature + 8 <= c.features &&
            c.features % Copy::VectorElements == 0 && (!Final || !w.args.write_ddy || w.args.ddy.row_stride == 1)) {
            const unsigned long saved = base + static_cast<unsigned long>(sample) * c.features + feature;
            typename Atom::Operand pre, hidden, tangent;
            Atom::load_device(pre, c.pre, saved, c.features);
            Atom::load_device(hidden, c.hidden, saved, c.features);
            Atom::template tangent<Math, Policy>(tangent, result, pre, hidden);
            Atom::store_device(result, c.first, saved, c.features);
            Atom::store_device(tangent, c.second, saved, c.features);
            if (!Final) w.state[I] = tangent;
            if (Final && w.args.write_ddy)
                Atom::store_device(tangent, c.external,
                    static_cast<unsigned long>(sample) * w.args.ddy.sample_stride + feature, w.args.ddy.sample_stride);
            return;
        }
        Atom::store_shared(result, w.panel, w.sg * 64, 8);
        Copy::barrier();
        for (unsigned i = w.tid; i < S::Simdgroups * 64; i += S::Threads) {
            const unsigned long sample = EpilogueIndex<S, I>::sample(w.sample, i);
            const unsigned long feature = EpilogueIndex<S, I>::feature(c.feature_base, i);
            Scalar tangent = Scalar(0);
            if (sample < w.args.batch && feature < c.features) {
                const unsigned long saved = base + static_cast<unsigned long>(sample) * c.features + feature;
                const Scalar t = w.panel[i];
                c.first[saved] = t;
                tangent = Scalar(float(t) * float(Math::template first<Policy>(c.pre[saved], c.hidden[saved])));
                c.second[saved] = tangent;
                if (Final && w.args.write_ddy)
                    c.external[static_cast<unsigned long>(sample) * w.args.ddy.sample_stride +
                               static_cast<unsigned long>(feature) * w.args.ddy.row_stride] = tangent;
            }
            w.panel[i] = tangent;
        }
        Copy::barrier();
        if (!Final) Atom::load_shared(w.state[I], w.panel, w.sg * 64, 8);
        Copy::barrier();
    }
};

template <class S, class Atom, class Copy, class Math, class Policy>
struct SecondEpilogue {
    template <unsigned I>
    static TCNN_FUSED_INLINE void apply(thread EpilogueContext<S, Atom>& c) {
        using Scalar = typename Atom::Scalar;
        thread HigherOrderWorkspace<S, Atom>& w = *c.w;
        typename Atom::Operand result, second_result;
        Atom::template convert<Identity>(result, w.accumulators[I]);
        Atom::template convert<Identity>(second_result, w.second_accumulators[I]);
        const unsigned long base = saved_base(w.args, S::Width, c.layer);
        if (w.args.batch - w.sample >= S::Samples) {
            const unsigned long sample = w.sample + w.sg * S::SamplesPerSimdgroup + I / S::ColumnTiles * 8;
            const unsigned feature = I % S::ColumnTiles * 8;
            const unsigned long saved = base + static_cast<unsigned long>(sample) * S::Width + feature;
            typename Atom::Operand pre, hidden, tangent_pre, delta, tangent_delta;
            Atom::load_device(pre, c.pre, saved, S::Width);
            Atom::load_device(hidden, c.hidden, saved, S::Width);
            Atom::load_device(tangent_pre, c.tangent_pre_input, saved, S::Width);
            Atom::template second<Math, Policy>(delta, tangent_delta, result, second_result, pre, hidden, tangent_pre);
            Atom::store_device(delta, c.first, saved, S::Width);
            Atom::store_device(tangent_delta, c.second, saved, S::Width);
            w.state[I] = delta;
            w.second_state[I] = tangent_delta;
            return;
        }
        Atom::store_shared(result, w.panel, w.sg * 64, 8);
        Atom::store_shared(second_result, w.second_panel, w.sg * 64, 8);
        Copy::barrier();
        for (unsigned i = w.tid; i < S::Simdgroups * 64; i += S::Threads) {
            const unsigned long sample = EpilogueIndex<S, I>::sample(w.sample, i);
            const unsigned feature = EpilogueIndex<S, I>::feature(0, i);
            Scalar delta = Scalar(0), tangent_delta = Scalar(0);
            if (sample < w.args.batch) {
                const unsigned long saved = base + static_cast<unsigned long>(sample) * S::Width + feature;
                const Scalar first = Math::template first<Policy>(c.pre[saved], c.hidden[saved]);
                const Scalar second = Math::template second<Policy>(c.pre[saved], c.hidden[saved]);
                const Scalar adjoint = w.panel[i], tangent_adjoint = w.second_panel[i];
                delta = Scalar(float(adjoint) * float(first) +
                    float(tangent_adjoint) * float(second) * float(c.tangent_pre_input[saved]));
                tangent_delta = Scalar(float(tangent_adjoint) * float(first));
                c.first[saved] = delta;
                c.second[saved] = tangent_delta;
            }
            w.panel[i] = delta;
            w.second_panel[i] = tangent_delta;
        }
        Copy::barrier();
        Atom::load_shared(w.state[I], w.panel, w.sg * 64, 8);
        Atom::load_shared(w.second_state[I], w.second_panel, w.sg * 64, 8);
        Copy::barrier();
    }
};

template <class S, class Atom, class Copy>
struct InputEpilogue {
    template <unsigned I>
    static TCNN_FUSED_INLINE void apply(thread EpilogueContext<S, Atom>& c) {
        thread HigherOrderWorkspace<S, Atom>& w = *c.w;
        typename Atom::Operand result;
        Atom::template convert<Identity>(result, w.accumulators[I]);
        const unsigned long feature = c.feature_base + I % S::ColumnTiles * 8;
        if (w.args.batch - w.sample >= S::Samples && feature + 8 <= w.args.inputs && w.args.dx.row_stride == 1) {
            const unsigned long sample = w.sample + w.sg * S::SamplesPerSimdgroup + I / S::ColumnTiles * 8;
            Atom::store_device(result, c.external,
                static_cast<unsigned long>(sample) * w.args.dx.sample_stride + feature, w.args.dx.sample_stride);
            return;
        }
        Atom::store_shared(result, w.panel, w.sg * 64, 8);
        Copy::barrier();
        for (unsigned i = w.tid; i < S::Simdgroups * 64; i += S::Threads) {
            const unsigned long sample = EpilogueIndex<S, I>::sample(w.sample, i);
            const unsigned long feature = EpilogueIndex<S, I>::feature(c.feature_base, i);
            if (sample < w.args.batch && feature < w.args.inputs)
                c.external[static_cast<unsigned long>(sample) * w.args.dx.sample_stride +
                           static_cast<unsigned long>(feature) * w.args.dx.row_stride] = w.panel[i];
        }
        Copy::barrier();
    }
};

template <class S, class Atom>
TCNN_FUSED_INLINE void initialize(thread HigherOrderWorkspace<S, Atom>& w,
                                   device const typename Atom::Scalar* weights,
                                   threadgroup typename Atom::Scalar* scratch,
                                   HigherOrderArguments args, unsigned group, unsigned tid, unsigned sg) {
    w.weights = scratch + S::WeightOffset;
    w.panel = scratch + S::InputOffset;
    w.second_panel = scratch + S::SecondInputOffset;
    w.parameters = weights;
    w.args = args;
    w.sample = static_cast<unsigned long>(group) * S::Samples;
    w.tid = tid;
    w.sg = sg;
}

} // namespace higher_order_detail

template <class S, class Atom, class Copy, class Math, class Hidden, class Output>
struct HigherOrderTangent {
    using Scalar = typename Atom::Scalar;
    static TCNN_FUSED_INLINE void run(device const Scalar* direction, device const Scalar* weights,
        device const Scalar* pre, device const Scalar* hidden, device Scalar* tangent_pre,
        device Scalar* tangent, device Scalar* ddy, threadgroup Scalar* scratch,
        HigherOrderArguments args, unsigned group, unsigned tid, unsigned sg) {
        using namespace higher_order_detail;
        HigherOrderWorkspace<S, Atom> w;
        initialize(w, weights, scratch, args, group, tid, sg);
        DirectionSource<S, Atom, Copy> input{direction};
        RegisterSource state;
        EpilogueContext<S, Atom> epilogue{&w, pre, hidden, nullptr, tangent_pre, tangent, ddy, 0, S::Width, 0};
        if (args.hidden_layers) {
            Mainloop<S, Atom, Copy, false, false, false>::run(w, input, 0, S::Width, args.inputs);
            StaticFor<0, S::Fragments>::template run<TangentEpilogue<S, Atom, Copy, Math, Hidden, false>>(epilogue);
            for (unsigned layer = 1; layer < args.hidden_layers; ++layer) {
                Mainloop<S, Atom, Copy, false, false, true>::run(w, state,
                    weight_base(args, S::Width, layer), S::Width, S::Width);
                epilogue.layer = layer;
                StaticFor<0, S::Fragments>::template run<TangentEpilogue<S, Atom, Copy, Math, Hidden, false>>(epilogue);
            }
        }
        epilogue.layer = args.hidden_layers;
        epilogue.features = args.outputs;
        for (unsigned long feature = 0; feature < args.outputs; feature += S::Width) {
            if (args.hidden_layers)
                Mainloop<S, Atom, Copy, false, false, true>::run(w, state,
                    weight_base(args, S::Width, args.hidden_layers), args.outputs, S::Width, feature);
            else
                Mainloop<S, Atom, Copy, false, false, false>::run(w, input, 0, args.outputs, args.inputs, feature);
            epilogue.feature_base = feature;
            StaticFor<0, S::Fragments>::template run<TangentEpilogue<S, Atom, Copy, Math, Output, true>>(epilogue);
        }
    }
};

template <class S, class Atom, class Copy, class Math, class Hidden, class Output>
struct HigherOrderBackward {
    using Scalar = typename Atom::Scalar;
    static TCNN_FUSED_INLINE void run(device const Scalar* weights, device const Scalar* pre,
        device const Scalar* hidden, device const Scalar* tangent_pre, device const Scalar* incoming,
        device Scalar* delta, device Scalar* tangent_delta, device Scalar* dx, threadgroup Scalar* scratch,
        HigherOrderArguments args, unsigned group, unsigned tid, unsigned sg) {
        using namespace higher_order_detail;
        HigherOrderWorkspace<S, Atom> w;
        initialize(w, weights, scratch, args, group, tid, sg);
        SeedSource<S, Atom, Math, Output> seed{pre, hidden, tangent_pre, incoming, delta, tangent_delta};
        RegisterSource state;
        EpilogueContext<S, Atom> epilogue{&w, pre, hidden, tangent_pre, delta, tangent_delta, dx, 0, S::Width, 0};
        if (args.hidden_layers) {
            Mainloop<S, Atom, Copy, true, true, false>::run(w, seed,
                weight_base(args, S::Width, args.hidden_layers), args.outputs, S::Width);
            epilogue.layer = args.hidden_layers - 1;
            StaticFor<0, S::Fragments>::template run<SecondEpilogue<S, Atom, Copy, Math, Hidden>>(epilogue);
            for (unsigned layer = args.hidden_layers - 1; layer > 0; --layer) {
                Mainloop<S, Atom, Copy, true, true, true>::run(w, state,
                    weight_base(args, S::Width, layer), S::Width, S::Width);
                epilogue.layer = layer - 1;
                StaticFor<0, S::Fragments>::template run<SecondEpilogue<S, Atom, Copy, Math, Hidden>>(epilogue);
            }
        } else if (!args.write_dx) {
            // Parameter-only second backward still produces both output seeds.
            for (unsigned long k = 0; k < args.outputs; k += S::KPanel) {
                seed.load(w, k);
                Copy::barrier();
            }
        }
        if (args.write_dx) {
            for (unsigned long feature = 0; feature < args.inputs; feature += S::Width) {
                if (args.hidden_layers)
                    Mainloop<S, Atom, Copy, true, false, true>::run(w, state, 0, S::Width, args.inputs, feature);
                else
                    Mainloop<S, Atom, Copy, true, false, false>::run(w, seed, 0, args.outputs, args.inputs, feature);
                epilogue.feature_base = feature;
                StaticFor<0, S::Fragments>::template run<InputEpilogue<S, Atom, Copy>>(epilogue);
            }
        }
    }
};

} // namespace tcnn_fused
#endif
