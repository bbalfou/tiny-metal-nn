#ifndef TCNN_FUSED_INFERENCE_KERNEL
#define TCNN_FUSED_INFERENCE_KERNEL
#include "traits.metal"
#include "inference_traits.metal"
#ifdef __METAL_VERSION__
#include "inference_atoms.metal"
#endif

namespace tcnn_fused {
namespace inference_detail {

template <class S, class Atom>
struct MatrixContext {
    thread typename Atom::Accumulator* acc;
    threadgroup const typename Atom::Scalar* a;
    threadgroup const typename Atom::Scalar* shared_b;
    device const typename Atom::Scalar* device_b;
    unsigned a_stride, b_stride, sg;
};
template <class S, class Atom>
struct Zero {
    template <unsigned I> static TCNN_FUSED_INLINE void apply(thread MatrixContext<S, Atom>& c) {
        Atom::zero(c.acc[I]);
    }
};
template <class S, class Atom>
struct MultiplyContext {
    thread MatrixContext<S, Atom>* matrix;
    typename Atom::Operand a[S::RowTiles], b[S::ColumnTiles];
};
template <class S, class Atom, unsigned K>
struct LoadA {
    template <unsigned R> static TCNN_FUSED_INLINE void apply(thread MultiplyContext<S, Atom>& c) {
        Atom::load_shared(c.a[R], c.matrix->a,
            (S::first_row(c.matrix->sg) + R * 8) * c.matrix->a_stride + K * 8, c.matrix->a_stride);
    }
};
template <class S, class Atom, unsigned K, bool Direct>
struct LoadB {
    template <unsigned C> static TCNN_FUSED_INLINE void apply(thread MultiplyContext<S, Atom>& c) {
        const unsigned offset = (S::first_column(c.matrix->sg) + C * 8) * c.matrix->b_stride + K * 8;
        if (Direct)
            Atom::load_device(c.b[C], c.matrix->device_b, offset, c.matrix->b_stride, true);
        else
            Atom::load_shared(c.b[C], c.matrix->shared_b, offset, c.matrix->b_stride, true);
    }
};
template <class S, class Atom>
struct Multiply {
    template <unsigned I> static TCNN_FUSED_INLINE void apply(thread MultiplyContext<S, Atom>& c) {
        Atom::mma(c.matrix->acc[I], c.a[I / S::ColumnTiles], c.b[I % S::ColumnTiles]);
    }
};
template <class S, class Atom, bool Direct>
struct KTile {
    template <unsigned K> static TCNN_FUSED_INLINE void apply(thread MatrixContext<S, Atom>& c) {
        MultiplyContext<S, Atom> operation;
        operation.matrix = &c;
        StaticFor<0, S::RowTiles>::template run<LoadA<S, Atom, K>>(operation);
        StaticFor<0, S::ColumnTiles>::template run<LoadB<S, Atom, K, Direct>>(operation);
        StaticFor<0, S::Fragments>::template run<Multiply<S, Atom>>(operation);
    }
};
template <class S, class Atom, bool Direct>
TCNN_FUSED_INLINE void matrix(thread MatrixContext<S, Atom>& c) {
    // Literal matrix indices reproduce the source's operand reuse and K8 order.
    StaticFor<0, S::KTiles>::template run<KTile<S, Atom, Direct>>(c);
}

template <class S, class Atom>
struct PublishContext {
    thread typename Atom::Accumulator* acc;
    threadgroup typename Atom::Scalar* state;
    device typename Atom::Scalar* output;
    InferenceArguments args;
    unsigned sample, sg;
};
template <class S, class Atom, class Policy, bool Final>
struct Publish {
    template <unsigned I> static TCNN_FUSED_INLINE void apply(thread PublishContext<S, Atom>& c) {
        typename Atom::Operand value;
        Atom::template convert<Policy>(value, c.acc[I]);
        const unsigned row = S::first_row(c.sg) + (I / S::ColumnTiles) * 8;
        const unsigned column = S::first_column(c.sg) + (I % S::ColumnTiles) * 8;
        if (Final) {
            const bool soa = c.args.output_sample_stride == 1;
            Atom::store_device(value, c.output,
                (static_cast<unsigned long>(c.sample) + row) * c.args.output_sample_stride +
                    static_cast<unsigned long>(column) * c.args.output_row_stride,
                soa ? c.args.output_row_stride : c.args.output_sample_stride, soa);
        } else {
            Atom::store_shared(value, c.state, row * S::Columns + column, S::Columns);
        }
    }
};

template <class Tuning, class Atom, class Copy, class Activation, class OutputActivation = Identity>
struct Shared {
    using T = typename Atom::Scalar;
    using H = typename Tuning::Hidden;
    using F = typename Tuning::First;
    using O = typename Tuning::Output;
    struct Workspace {
        typename Atom::Accumulator acc[H::Fragments];
        device const T* input;
        device const T* params;
        device T* output;
        threadgroup T* state;
        threadgroup T* first;
        threadgroup T* weights;
        InferenceArguments args;
        unsigned sample, tid, sg;
    };

    template <unsigned K>
    static TCNN_FUSED_INLINE void input_panel(thread Workspace& w, threadgroup T* destination, unsigned offset) {
        if (w.args.input_row_stride == 1 && w.args.input_sample_stride % Copy::VectorElements == 0) {
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
                destination[row * K + column] = w.input[
                    (static_cast<unsigned long>(w.sample) + row) * w.args.input_sample_stride +
                    static_cast<unsigned long>(offset + column) * w.args.input_row_stride];
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
    static TCNN_FUSED_INLINE void publish(thread Workspace& w) {
        PublishContext<S, Atom> c{w.acc, w.state, w.output, w.args, w.sample, w.sg};
        StaticFor<0, S::Fragments>::template run<Publish<S, Atom, Policy, Final>>(c);
    }
    static TCNN_FUSED_INLINE unsigned long layer_offset(unsigned layer) {
        return layer ? static_cast<unsigned long>(Tuning::Width) * Tuning::Inputs +
            static_cast<unsigned long>(layer - 1) * Tuning::Width * Tuning::Width : 0;
    }

    static TCNN_FUSED_INLINE void first_product(thread Workspace& w) {
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

    static TCNN_FUSED_INLINE void next_merged_weights(thread Workspace& w, unsigned next) {
        if (next < w.args.hidden_layers)
            weight_panel<Tuning::Width, Tuning::KPanel>(w, layer_offset(next), Tuning::Width, 0);
        else
            weight_panel<Tuning::Outputs, Tuning::KPanel>(w, layer_offset(next), Tuning::Width, 0);
    }
    static TCNN_FUSED_INLINE void merged_step(thread Workspace& w, unsigned layer) {
        zero<H>(w);
        auto c = context<H>(w, w.state, Tuning::Width, 0, Tuning::KPanel);
        matrix<H, Atom, false>(c);
        Copy::barrier();
        // FP16 width32 merges state publication and the following weight load
        // between the same two barriers (_inference32.py merged_sync).
        publish<H, Activation, false>(w);
        next_merged_weights(w, layer + 1);
        Copy::barrier();
    }

    static TCNN_FUSED_INLINE void run(device const T* input, device const T* params, device T* output,
        threadgroup T* scratch, InferenceArguments args, unsigned group, unsigned tid, unsigned sg) {
        Workspace w;
        w.input = input; w.params = params; w.output = output;
        w.state = scratch; w.first = scratch + Tuning::FirstOffset;
        // FP32 never stages B; this one-past pointer is never dereferenced.
        w.weights = scratch + Tuning::WeightOffset;
        w.args = args; w.sample = group * Tuning::Samples; w.tid = tid; w.sg = sg;

        if (Tuning::MergedWeights) {
            if (Tuning::Inputs == Tuning::Width) {
                input_panel<Tuning::Width>(w, w.state, 0);
                weight_panel<Tuning::Width, Tuning::KPanel>(w, 0, Tuning::Width, 0);
                Copy::barrier();
                merged_step(w, 0);
            } else {
                first_product(w);
                publish<H, Activation, false>(w);
                next_merged_weights(w, 1);
                Copy::barrier();
            }
            unsigned layer = 1;
            for (; layer + 3 < args.hidden_layers; layer += 4) {
                merged_step(w, layer); merged_step(w, layer + 1);
                merged_step(w, layer + 2); merged_step(w, layer + 3);
            }
            for (; layer < args.hidden_layers; ++layer) merged_step(w, layer);
            zero<O>(w);
            auto c = context<O>(w, w.state, Tuning::Width, 0, Tuning::KPanel);
            matrix<O, Atom, false>(c);
            Copy::barrier();
            publish<O, OutputActivation, true>(w);
            return;
        }

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
        if (Tuning::Width == 32 && Tuning::DirectWeights) {
            for (; layer + 3 < args.hidden_layers; layer += 4) {
                hidden_step(w, layer); hidden_step(w, layer + 1);
                hidden_step(w, layer + 2); hidden_step(w, layer + 3);
            }
        }
        for (; layer < args.hidden_layers; ++layer) hidden_step(w, layer);
        state_product<O>(w, args.hidden_layers);
        publish<O, OutputActivation, true>(w);
    }
};

// The width16 source owns four eight-row strips in private fragments. FP16
// expands two such blocks per group; FP32 uses one. Up to six affine matrices
// are preloaded once and reused across those blocks, followed by a bounded loop.
template <class Tuning, class Atom, class Activation, class OutputActivation = Identity>
struct Private {
    using T = typename Atom::Scalar;
    enum : unsigned { FirstMatrices = 2 * Tuning::Inputs / 8,
        WeightMatrices = FirstMatrices + (Tuning::PreloadedLayers - 1) * 4 };
    struct Workspace {
        typename Atom::Operand weights[WeightMatrices], state[8];
        typename Atom::Accumulator acc[8];
        device const T* input;
        device const T* params;
        device T* output;
        InferenceArguments args;
        unsigned sample;
    };
    struct LoadWeights {
        template <unsigned I> static TCNN_FUSED_INLINE void apply(thread Workspace& w) {
            const unsigned layer = I < FirstMatrices ? 0 : 1 + (I - FirstMatrices) / 4;
            const unsigned local = I < FirstMatrices ? I : (I - FirstMatrices) % 4;
            const unsigned columns = I < FirstMatrices ? unsigned(Tuning::Inputs) : 16;
            const unsigned long offset = layer ? 16ul * Tuning::Inputs + (layer - 1ul) * 256ul : 0;
            Atom::load_device(w.weights[I], w.params,
                offset + (local / (columns / 8)) * 8ul * columns + (local % (columns / 8)) * 8ul, columns, true);
        }
    };
    struct Clear {
        template <unsigned I> static TCNN_FUSED_INLINE void apply(thread Workspace& w) { Atom::zero(w.acc[I]); }
    };
    struct Save {
        template <unsigned I> static TCNN_FUSED_INLINE void apply(thread Workspace& w) {
            typename Atom::Operand value;
            Atom::template convert<OutputActivation>(value, w.acc[I]);
            const bool soa = w.args.output_sample_stride == 1;
            Atom::store_device(value, w.output,
                (static_cast<unsigned long>(w.sample) + (I / 2) * 8) * w.args.output_sample_stride +
                    (I % 2) * 8ul * w.args.output_row_stride,
                soa ? w.args.output_row_stride : w.args.output_sample_stride, soa);
        }
    };
    struct Activate {
        template <unsigned I> static TCNN_FUSED_INLINE void apply(thread Workspace& w) {
            Atom::template convert<Activation>(w.state[I], w.acc[I]);
        }
    };
    template <unsigned Layer, unsigned K>
    struct PreloadedMultiply {
        template <unsigned R> static TCNN_FUSED_INLINE void apply(thread Workspace& w) {
            typename Atom::Operand a;
            if (Layer == 0) {
                const bool soa = w.args.input_sample_stride == 1;
                Atom::load_device(a, w.input,
                    (static_cast<unsigned long>(w.sample) + R * 8) * w.args.input_sample_stride +
                        K * 8ul * w.args.input_row_stride,
                    soa ? w.args.input_row_stride : w.args.input_sample_stride, soa);
            } else {
                a = w.state[R * 2 + K];
            }
            const unsigned base = Layer == 0 ? 0 : FirstMatrices + (Layer - 1) * 4;
            const unsigned ks = Layer == 0 ? Tuning::Inputs / 8 : 2;
            Atom::mma(w.acc[R * 2], a, w.weights[base + K]);
            Atom::mma(w.acc[R * 2 + 1], a, w.weights[base + ks + K]);
        }
    };
    template <unsigned Layer>
    struct PreloadedK {
        template <unsigned K> static TCNN_FUSED_INLINE void apply(thread Workspace& w) {
            StaticFor<0, 4>::template run<PreloadedMultiply<Layer, K>>(w);
        }
    };
    struct PreloadedLayer {
        template <unsigned Layer> static TCNN_FUSED_INLINE void apply(thread Workspace& w) {
            StaticFor<0, 8>::template run<Clear>(w);
            StaticFor<0, Layer == 0 ? Tuning::Inputs / 8 : 2>::template run<PreloadedK<Layer>>(w);
            if (Layer == w.args.hidden_layers) StaticFor<0, 8>::template run<Save>(w);
            else StaticFor<0, 8>::template run<Activate>(w);
        }
    };
    struct RuntimeContext {
        thread Workspace* workspace;
        typename Atom::Operand b[2];
        unsigned layer;
    };
    template <unsigned K>
    struct RuntimeMultiply {
        template <unsigned R> static TCNN_FUSED_INLINE void apply(thread RuntimeContext& c) {
            Atom::mma(c.workspace->acc[R * 2], c.workspace->state[R * 2 + K], c.b[0]);
            Atom::mma(c.workspace->acc[R * 2 + 1], c.workspace->state[R * 2 + K], c.b[1]);
        }
    };
    struct RuntimeK {
        template <unsigned K> static TCNN_FUSED_INLINE void apply(thread RuntimeContext& c) {
            const unsigned long offset = 16ul * Tuning::Inputs + (c.layer - 1ul) * 256ul + K * 8ul;
            Atom::load_device(c.b[0], c.workspace->params, offset, 16, true);
            Atom::load_device(c.b[1], c.workspace->params, offset + 128, 16, true);
            StaticFor<0, 4>::template run<RuntimeMultiply<K>>(c);
        }
    };
    struct Block {
        template <unsigned B> static TCNN_FUSED_INLINE void apply(thread Workspace& w) {
            w.sample += B * 32;
            StaticFor<0, Tuning::PreloadedLayers>::template run<PreloadedLayer>(w);
            for (unsigned layer = Tuning::PreloadedLayers; layer <= w.args.hidden_layers; ++layer) {
                StaticFor<0, 8>::template run<Clear>(w);
                RuntimeContext c;
                c.workspace = &w; c.layer = layer;
                StaticFor<0, 2>::template run<RuntimeK>(c);
                if (layer == w.args.hidden_layers) StaticFor<0, 8>::template run<Save>(w);
                else StaticFor<0, 8>::template run<Activate>(w);
            }
            w.sample -= B * 32;
        }
    };
    static TCNN_FUSED_INLINE void run(device const T* input, device const T* params, device T* output,
        InferenceArguments args, unsigned group) {
        Workspace w;
        w.input = input; w.params = params; w.output = output;
        w.args = args; w.sample = group * Tuning::Samples;
        StaticFor<0, WeightMatrices>::template run<LoadWeights>(w);
        StaticFor<0, Tuning::Samples / 32>::template run<Block>(w);
    }
};

// Masked boundary tiles use the same FP32 MMA accumulation and storage-rounding
// policies as Shared. No per-layer global inference context is allocated.
template <class Tuning, class Atom, class Copy, class Activation, class OutputActivation = Identity>
struct Boundary {
    using T = typename Atom::Scalar;
    using H = typename Tuning::Hidden;
    using O = typename Tuning::Output;
    struct Workspace {
        typename Atom::Accumulator acc[H::Fragments];
        device const T* input;
        device const T* params;
        device T* output;
        threadgroup T* state;
        threadgroup T* panel;
        threadgroup T* weights;
        InferenceArguments args;
        unsigned sample, tid, sg;
    };

    template <class S>
    static TCNN_FUSED_INLINE void product(thread Workspace& w, unsigned layer, unsigned column) {
        MatrixContext<S, Atom> c{w.acc, w.panel, w.weights, w.params,
                                Tuning::KPanel, Tuning::KPanel, w.sg};
        StaticFor<0, S::Fragments>::template run<Zero<S, Atom>>(c);
        const unsigned inputs = layer ? unsigned(Tuning::Width) : unsigned(Tuning::Inputs);
        const unsigned outputs = layer == w.args.hidden_layers ? unsigned(Tuning::Outputs) : unsigned(Tuning::Width);
        const unsigned long offset = layer ? static_cast<unsigned long>(Tuning::Width) * Tuning::Inputs +
            static_cast<unsigned long>(layer - 1) * Tuning::Width * Tuning::Width : 0;
        for (unsigned long kb = 0; kb < inputs; kb += Tuning::KPanel) {
            for (unsigned i = w.tid; i < Tuning::Samples * Tuning::KPanel; i += Tuning::Threads) {
                const unsigned row = i / Tuning::KPanel, k = i % Tuning::KPanel;
                const unsigned sample = w.sample + row;
                T value = T(0);
                if (sample < w.args.batch && kb + k < inputs) {
                    value = layer ? w.state[row * Tuning::Width + kb + k]
                        : w.input[static_cast<unsigned long>(sample) * w.args.input_sample_stride +
                                  (kb + k) * w.args.input_row_stride];
                }
                w.panel[i] = value;
            }
            for (unsigned i = w.tid; i < S::Columns * Tuning::KPanel; i += Tuning::Threads) {
                const unsigned row = column + i / Tuning::KPanel, k = i % Tuning::KPanel;
                w.weights[i] = row < outputs && kb + k < inputs
                    ? w.params[offset + static_cast<unsigned long>(row) * inputs + kb + k] : T(0);
            }
            Copy::barrier();
            matrix<S, Atom, false>(c);
            // All groups finish reading both panels before their next refill.
            Copy::barrier();
        }
    }

    static TCNN_FUSED_INLINE void run(device const T* input, device const T* params, device T* output,
        threadgroup T* scratch, InferenceArguments args, unsigned group, unsigned tid, unsigned sg) {
        Workspace w;
        w.input = input; w.params = params; w.output = output;
        w.state = scratch; w.panel = scratch + Tuning::PanelOffset; w.weights = scratch + Tuning::WeightOffset;
        w.args = args; w.sample = group * Tuning::Samples; w.tid = tid; w.sg = sg;
        for (unsigned layer = 0; layer < args.hidden_layers; ++layer) {
            product<H>(w, layer, 0);
            PublishContext<H, Atom> c{w.acc, w.state, output, args, w.sample, sg};
            StaticFor<0, H::Fragments>::template run<Publish<H, Atom, Activation, false>>(c);
            Copy::barrier();
        }
        for (unsigned long column = 0; column < Tuning::Outputs; column += Tuning::Projection) {
            product<O>(w, args.hidden_layers, unsigned(column));
            PublishContext<O, Atom> c{w.acc, w.panel, output, args, w.sample, sg};
            StaticFor<0, O::Fragments>::template run<Publish<O, Atom, OutputActivation, false>>(c);
            Copy::barrier();
            for (unsigned i = tid; i < Tuning::Samples * Tuning::Projection; i += Tuning::Threads) {
                const unsigned sample = w.sample + i / Tuning::Projection;
                const unsigned long channel = column + i % Tuning::Projection;
                if (sample < args.batch && channel < Tuning::Outputs)
                    output[static_cast<unsigned long>(sample) * args.output_sample_stride +
                           channel * args.output_row_stride] = w.panel[i];
            }
            Copy::barrier();
        }
    }
};

} // namespace inference_detail
} // namespace tcnn_fused
#endif
