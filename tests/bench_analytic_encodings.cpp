// Reproduce: cmake -S . -B build/analytic-tests -DCMAKE_BUILD_TYPE=Release -DMETALNN_BUILD_TESTS=ON
// cmake --build build/analytic-tests --target bench_analytic_encodings -j 8
// build/analytic-tests/bench_analytic_encodings [batch-size]
// Run before/after binaries sequentially on an idle GPU. Each JSON line retains
// five rounds of 128 prepared graph launches, synchronized every 16 launches.
// Compilation and ten warmups are excluded. Buffers are reused (warm-cache test).
// Native FP16, three coordinates, width-32/two-hidden-layer MLP; 12 spectral
// frequencies, 64 OneBlob bins, degree-4 SH. Encoder/backward/MLP use SoA;
// aos measures AoS, combined uses the encoder's preferred layout. Backward
// measures input gradients; parameter-free training without input gradients
// only runs the forward kernel. Identity is an unchanged control.
#include <tiny-metal-nn/network_with_input_encoding.h>
#include <tiny-metal-nn/random.h>
#include <chrono>
#include <iostream>
using namespace tcnn;
using T = network_precision_t;
int main(int argc, char** argv) try {
    const uint32_t n = argc > 1 ? std::stoul(argv[1]) : 65536;
    auto device = metalrt::Device::system_default();
    metal::set_default_device(device);
    metal::Stream ns{device};
    auto& s = ns.runtime();
    for (auto kind : {"Identity", "Frequency", "TriangleWave", "OneBlob", "SphericalHarmonics"}) {
        json cfg = {{"otype", kind}, {"n_frequencies", 12}, {"n_bins", 64}, {"degree", 4}};
        auto enc = std::shared_ptr<Encoding<T>>(create_encoding<T>(3, cfg, 16));
        auto mlp = std::shared_ptr<Network<T>>(
            create_network<T>({{"otype", "FullyFusedMLP"},
                               {"n_input_dims", enc->padded_output_width()},
                               {"n_output_dims", 3},
                               {"n_neurons", 32},
                               {"n_hidden_layers", 2},
                               {"activation", "ReLU"},
                               {"output_activation", "None"}}));
        NetworkWithInputEncoding<T> combined{enc, mlp};
        GPUMemory<float> master(combined.n_params());
        GPUMemory<T> weights(combined.n_params());
        pcg32 rng{42};
        combined.initialize_params(rng, master.data());
        metal::runtime_stream().synchronize();
        for (size_t i = 0; i < weights.size(); ++i)
            weights.data()[i] = T(master.data()[i]);
        combined.set_params(weights.data(), weights.data(), nullptr);
        GPUMatrixDynamic<float> x(3, n, CM), dx(3, n, CM);
        GPUMatrixDynamic<T> y(enc->padded_output_width(), n, SoA),
            aos(enc->padded_output_width(), n, CM), dy(enc->padded_output_width(), n, SoA),
            out(mlp->padded_output_width(), n, CM);
        generate_random_uniform(ns.get(), rng, x.n_elements(), x.data());
        s.synchronize();
        std::fill(dy.data(), dy.data() + dy.n_elements(), T(.01f));
        enc->inference_mixed_precision(ns.get(), x, y);
        s.synchronize();
        for (auto part : {"encoder", "aos", "backward", "mlp", "combined"}) {
            auto ctx = enc->forward(ns.get(), x, &y, false, true);
            s.synchronize();
            metalrt::Graph g;
            auto rec = g.record_guard(&s);
            if (std::string(part) == "encoder")
                enc->inference_mixed_precision(ns.get(), x, y);
            if (std::string(part) == "aos")
                enc->inference_mixed_precision(ns.get(), x, aos);
            if (std::string(part) == "backward")
                enc->backward(ns.get(), *ctx, x, y, dy, &dx, false, GradientMode::Overwrite);
            if (std::string(part) == "mlp")
                mlp->inference_mixed_precision(ns.get(), y, out);
            if (std::string(part) == "combined")
                combined.inference_mixed_precision(ns.get(), x, out);
            rec.finish();
            g.prepare(&s);
            for (int i = 0; i < 10; ++i)
                g.launch(&s);
            s.synchronize();
            std::vector<double> rounds;
            for (int r = 0; r < 5; ++r) {
                auto start = std::chrono::steady_clock::now();
                for (int i = 0; i < 128; ++i) {
                    g.launch(&s);
                    if (i % 16 == 15)
                        s.synchronize();
                }
                s.synchronize();
                rounds.push_back(std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - start)
                                     .count() /
                                 128);
            }
            std::sort(rounds.begin(), rounds.end());
            std::cout << json{{"encoder", kind},     {"part", part},
                              {"batch", n},          {"ms", rounds[2]},
                              {"rounds_ms", rounds}, {"device", device.capabilities().name}}
                             .dump()
                      << std::endl;
        }
    }
} catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
}
