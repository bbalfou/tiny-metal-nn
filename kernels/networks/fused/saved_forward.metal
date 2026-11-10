// Aligned training forward schedule: use inference matrix tiling while retaining
// both rounded preactivations and activations for backward and higher derivatives.
#ifndef TCNN_FUSED_SAVED_FORWARD_METAL
#define TCNN_FUSED_SAVED_FORWARD_METAL
#include "inference_kernel.metal"
namespace tcnn_fused { namespace inference_detail {
template <class Tuning, class Atom, class Copy, class Activation, class OutputActivation = Identity>
struct SavedForward {
    using T = typename Atom::Scalar;
    using H = typename Tuning::Hidden;
    using F = typename Tuning::First;
    using O = typename Tuning::Output;
    struct Workspace {
        typename Atom::Accumulator acc[H::Fragments];
        device const T* input;
        device const T* params;
        device T* output;
        device T* pre;
        device T* hidden;
        unsigned layer;
        threadgroup T* state;
        threadgroup T* first;
        threadgroup T* weights;
        InferenceArguments args;
        unsigned sample, tid, sg;
    };

    template <unsigned K>
    static TCNN_FUSED_INLINE void input_panel(thread Workspace& w, threadgroup T* destination, unsigned offset) {
        if (w.args.batch - w.sample >= Tuning::Samples &&
            w.args.input_row_stride == 1 && w.args.input_sample_stride % Copy::VectorElements == 0) {
            for (unsigned i = w.tid; i < Tuning::Samples * K / Copy::VectorElements; i += Tuning::Threads) {
                const unsigned index = i * Copy::VectorElements;
                Copy::to_shared(destination, index, w.input,
                    (static_cast<unsigned long>(w.sample) + index / K) * w.args.input_sample_stride + offset + index % K);
            }
        } else {
            const bool soa = w.args.input_sample_stride == 1;
            for (unsigned i = w.tid; i < Tuning::Samples * K; i += Tuning::Threads) {
                const unsigned row = soa ? i % Tuning::Samples : i / K;
                const unsigned column = soa ? i / Tuning::Samples : i % K;
                destination[row * K + column] = w.sample + row < w.args.batch ? w.input[
                    (static_cast<unsigned long>(w.sample) + row) * w.args.input_sample_stride +
                    static_cast<unsigned long>(offset + column) * w.args.input_row_stride] : T(0);
            }
        }
    }

    template <unsigned N, unsigned K>
    static TCNN_FUSED_INLINE void weight_panel(thread Workspace& w, unsigned long offset, unsigned stride, unsigned kb) {
        for (unsigned i = w.tid; i < N * K / Copy::VectorElements; i += Tuning::Threads) {
            const unsigned index = i * Copy::VectorElements;
            Copy::to_shared(w.weights, index, w.params,
                offset + static_cast<unsigned long>(index / K) * stride + kb + index % K);
        }
    }

    template <class S>
    static TCNN_FUSED_INLINE MatrixContext<S, Atom> context(thread Workspace& w, threadgroup const T* a,
        unsigned a_stride, unsigned long offset, unsigned b_stride) {
        return {w.acc, a, w.weights, w.params + offset, a_stride, b_stride, w.sg};
    }
    template <class S>
    static TCNN_FUSED_INLINE void zero(thread Workspace& w) {
        auto c = context<S>(w, w.state, Tuning::Width, 0, Tuning::Width);
        StaticFor<0, S::Fragments>::template run<Zero<S, Atom>>(c);
    }
    template <class S, class Policy, bool Final>
    struct Save {
        template<unsigned I> static TCNN_FUSED_INLINE void apply(thread Workspace& w) {
            const unsigned row = S::first_row(w.sg) + (I / S::ColumnTiles)*8;
            const unsigned column = S::first_column(w.sg) + (I % S::ColumnTiles)*8;
            const unsigned features = Final ? Tuning::Outputs : Tuning::Width;
            const unsigned long offset = static_cast<unsigned long>(w.layer)*w.args.batch*Tuning::Width +
                static_cast<unsigned long>(w.sample+row)*features+column;
            typename Atom::Operand v;
            Atom::template convert<Identity>(v,w.acc[I]);
            Atom::store_device(v,w.pre,offset,features);
            if (!Final) {
                Atom::template convert<Policy>(v,w.acc[I]);
                Atom::store_device(v,w.hidden,offset,features);
            }
        }
    };
    // A partial sample tile uses shared state for masked scalar publication.
    // It follows the identical MMA/conversion order and never writes guard rows.
    template <class S, class Policy> struct StageTail {
        template <unsigned I> static TCNN_FUSED_INLINE void apply(thread Workspace& w) {
            typename Atom::Operand value;
            Atom::template convert<Policy>(value, w.acc[I]);
            const unsigned row = S::first_row(w.sg) + (I / S::ColumnTiles) * 8;
            const unsigned column = S::first_column(w.sg) + (I % S::ColumnTiles) * 8;
            Atom::store_shared(value, w.state, row * S::Columns + column, S::Columns);
        }
    };
    template <class S> static TCNN_FUSED_INLINE void export_tail(
        thread Workspace& w, device T* destination) {
        const unsigned long base = static_cast<unsigned long>(w.layer) * w.args.batch * Tuning::Width;
        for (unsigned i = w.tid; i < Tuning::Samples * S::Columns; i += Tuning::Threads) {
            const unsigned sample = w.sample + i / S::Columns;
            if (sample < w.args.batch)
                destination[base + static_cast<unsigned long>(sample) * S::Columns + i % S::Columns] = w.state[i];
        }
    }
    template <class S, class Policy, bool Final>
    static TCNN_FUSED_INLINE void publish(thread Workspace& w) {
        if (w.args.batch - w.sample < Tuning::Samples) {
            StaticFor<0, S::Fragments>::template run<StageTail<S, Identity>>(w);
            Copy::barrier();
            export_tail<S>(w, w.pre);
            Copy::barrier();
            StaticFor<0, S::Fragments>::template run<StageTail<S, Policy>>(w);
            Copy::barrier();
            export_tail<S>(w, w.hidden);
            Copy::barrier();
            return;
        }
        StaticFor<0,S::Fragments>::template run<Save<S,Policy,Final>>(w);
        PublishContext<S, Atom> c{w.acc, w.state, w.output, w.args, w.sample, w.sg};
        StaticFor<0, S::Fragments>::template run<Publish<S, Atom, Policy, Final>>(c);
    }
    static TCNN_FUSED_INLINE unsigned long layer_offset(unsigned layer) {
        return layer ? static_cast<unsigned long>(Tuning::Width) * Tuning::Inputs +
            static_cast<unsigned long>(layer - 1) * Tuning::Width * Tuning::Width : 0;
    }

    static TCNN_FUSED_INLINE void first_product(thread Workspace& w) {
        w.layer=0;
        zero<F>(w);
        for (unsigned kb = 0; kb < Tuning::Inputs; kb += Tuning::FirstK) {
            input_panel<Tuning::FirstK>(w, w.first, kb);
            if (!Tuning::DirectWeights)
                weight_panel<Tuning::Width, Tuning::FirstK>(w, 0, Tuning::Inputs, kb);
            Copy::barrier();
            auto c = context<F>(w, w.first, Tuning::FirstK, kb,
                                Tuning::DirectWeights ? unsigned(Tuning::Inputs) : unsigned(Tuning::FirstK));
            matrix<F, Atom, Tuning::DirectWeights>(c);
            Copy::barrier();
        }
    }

    template <class S>
    static TCNN_FUSED_INLINE void state_product(thread Workspace& w, unsigned layer) {
        w.layer=layer;
        zero<S>(w);
        for (unsigned kb = 0; kb < Tuning::Width; kb += S::KPanel) {
            if (!Tuning::DirectWeights) {
                weight_panel<S::Columns, S::KPanel>(w, layer_offset(layer), Tuning::Width, kb);
                Copy::barrier();
            }
            auto c = context<S>(w, w.state + kb, Tuning::Width, layer_offset(layer) + kb,
                                Tuning::DirectWeights ? unsigned(Tuning::Width) : unsigned(S::KPanel));
            matrix<S, Atom, Tuning::DirectWeights>(c);
            if (!Tuning::DirectWeights) Copy::barrier();
        }
    }
    static TCNN_FUSED_INLINE void hidden_step(thread Workspace& w, unsigned layer) {
        state_product<H>(w, layer);
        if (Tuning::DirectWeights) Copy::barrier();
        publish<H, Activation, false>(w);
        Copy::barrier();
    }

    static TCNN_FUSED_INLINE void run(device const T* input, device const T* params, device T* pre, device T* hidden,
        threadgroup T* scratch, RectangularArguments r, unsigned group, unsigned tid, unsigned sg) {
        InferenceArguments args{r.training.batch,r.training.hidden_layers,r.training.input.row_stride,
            r.training.input.sample_stride,1,Tuning::Outputs};
        device T* output=hidden+static_cast<unsigned long>(args.hidden_layers)*args.batch*Tuning::Width;
        Workspace w;
        w.pre=pre;w.hidden=hidden;w.layer=0;
        w.input = input; w.params = params; w.output = output;
        w.state = scratch; w.first = scratch + Tuning::FirstOffset;
        // FP32 never stages B; this one-past pointer is never dereferenced.
        w.weights = scratch + Tuning::WeightOffset;
        w.args = args; w.sample = group * Tuning::Samples; w.tid = tid; w.sg = sg;

        unsigned first_hidden = 0;
        if (Tuning::DirectWeights && Tuning::Inputs == Tuning::Width) {
            input_panel<Tuning::Width>(w, w.state, 0);
            Copy::barrier();
        } else {
            first_product(w);
            publish<H, Activation, false>(w);
            Copy::barrier();
            first_hidden = 1;
        }
        unsigned layer = first_hidden;
        for (; layer < args.hidden_layers; ++layer) hidden_step(w, layer);
        state_product<O>(w, args.hidden_layers);
        publish<O, OutputActivation, true>(w);
    }
};


}}

#endif
