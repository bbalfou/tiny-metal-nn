// Actual Trainer steps: encoder + MLP + L2 loss + backward + Adam updates.
// Usage: bench_encoder_training [batch=65536] [width=32] [layout=default|soa]
// Optional fourth argument selects an encoder by name; fifth sets input dimensions (2 or 3),
// sixth may be "profile" to sample kernels AFTER uninstrumented measurements. No data sampling in
// timing. Fixed seeded coordinates/targets; 20 warmup steps, five rounds of 64 actual Trainer
// calls, synchronizing every four steps. Forward, L2, parameter backward, Adam and host submission
// are timed; coordinate gradients and JIT are disabled. Compare separately built before/after
// executables sequentially on an idle GPU. Every case checks decreasing finite loss and finite
// updated FP32 master weights.
#include <tiny-metal-nn/network_with_input_encoding.h>
#include <tiny-metal-nn/encodings/oneblob.h>
#include <tiny-metal-nn/trainer.h>
#include <chrono>
#include <cctype>
#include <iostream>
using namespace tcnn;
using T = network_precision_t;
struct SoABlob : OneBlobEncoding<T> {
    SoABlob(uint32_t bins, uint32_t dims) : OneBlobEncoding<T>(bins, dims) {}
    MatrixLayout preferred_output_layout() const override { return SoA; }
};
int main(int argc, char** argv) try {
    const uint32_t n = argc > 1 ? std::stoul(argv[1]) : 65536;
    const uint32_t width = argc > 2 ? std::stoul(argv[2]) : 32;
    const bool soa = argc > 3 && std::string(argv[3]) == "soa";
    const uint32_t dims = argc > 5 ? std::stoul(argv[5]) : 3;
    if (dims != 2 && dims != 3)
        throw std::invalid_argument("input dimensions must be 2 or 3");
    const bool profile = argc > 6 && std::string(argv[6]) == "profile";
    auto device = metalrt::Device::system_default();
    metal::set_default_device(device);
    metal::Stream ns{device};
    auto& s = ns.runtime();
    for (std::string kind :
         {"Identity", "Frequency", "TriangleWave", "OneBlob", "SphericalHarmonics", "HashGrid"}) {
        if (argc > 4) {
            auto selected = std::string(argv[4]), name = kind;
            for (char& c : selected)
                c = char(std::tolower(static_cast<unsigned char>(c)));
            for (char& c : name)
                c = char(std::tolower(static_cast<unsigned char>(c)));
            if (selected != name)
                continue;
        }
        if (kind == "SphericalHarmonics" && dims != 3)
            continue;
        for (uint32_t bins : {16u, 64u}) {
            if (kind != "OneBlob" && bins != 16)
                continue;
            json config = {{"otype", kind}, {"n_bins", bins}, {"n_frequencies", 12}, {"degree", 4}};
            if (kind == "HashGrid")
                config = {
                    {"otype", kind},           {"n_levels", 16},        {"n_features_per_level", 2},
                    {"log2_hashmap_size", 19}, {"base_resolution", 16}, {"per_level_scale", 2.0}};
            std::shared_ptr<Encoding<T>> enc;
            if (kind == "OneBlob" && soa) {
                enc = std::make_shared<SoABlob>(bins, dims);
                enc->set_alignment(16);
            } else
                enc.reset(create_encoding<T>(dims, config, 16));
            auto mlp = std::shared_ptr<Network<T>>(
                create_network<T>({{"otype", "FullyFusedMLP"},
                                   {"n_input_dims", enc->padded_output_width()},
                                   {"n_output_dims", 3},
                                   {"n_neurons", width},
                                   {"n_hidden_layers", 2},
                                   {"activation", "ReLU"},
                                   {"output_activation", "None"}}));
            auto model = std::make_shared<NetworkWithInputEncoding<T>>(enc, mlp);
            model->set_jit_fusion(false);
            auto opt = std::shared_ptr<Optimizer<T>>(
                create_optimizer<T>({{"otype", "Adam"}, {"learning_rate", .001f}}));
            auto loss = std::shared_ptr<Loss<T>>(create_loss<T>({{"otype", "L2"}}));
            Trainer<float, T, T> trainer{model, opt, loss, 42};
            GPUMatrixDynamic<float> x(dims, n, CM);
            GPUMatrix<float> target(3, n);
            GPUMatrixDynamic<T> out(model->padded_output_width(), n, CM);
            pcg32 rng{123};
            generate_random_uniform(ns.get(), rng, x.n_elements(), x.data());
            s.synchronize();
            for (uint32_t j = 0; j < n; ++j)
                for (uint32_t d = 0; d < 3; ++d)
                    target.data()[j * 3 + d] = .1f + .5f * x.data()[j * dims + d % dims];
            std::unique_ptr<Trainer<float, T, T>::ForwardContext> ctx;
            auto call = [&] {
                auto b = s.batch();
                ctx = trainer.training_step(ns.get(), x, target);
                b.finish();
            };
            call();
            s.synchronize();
            const float initial = trainer.loss(ns.get(), *ctx);
            for (int i = 0; i < 19; ++i) {
                call();
                // Match measured queue depth: unbounded large-batch warmup can
                // retain enough activation storage to trigger memory pressure.
                if (i % 4 == 3)
                    s.synchronize();
            }
            s.synchronize();
            std::vector<double> rounds;
            for (int r = 0; r < 5; ++r) {
                auto start = std::chrono::steady_clock::now();
                for (int i = 0; i < 64; ++i) {
                    call();
                    if (i % 4 == 3)
                        s.synchronize();
                }
                s.synchronize();
                rounds.push_back(std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - start)
                                     .count() /
                                 64);
            }
            const float final = trainer.loss(ns.get(), *ctx);
            double checksum = 0;
            for (size_t i = 0; i < model->n_params(); ++i) {
                const float v = trainer.params_full_precision()[i];
                if (!std::isfinite(v))
                    throw std::runtime_error("nonfinite weight");
                checksum += v * (1 + i % 13);
            }
            if (!std::isfinite(final) || final >= initial)
                throw std::runtime_error("loss did not decrease");
            json kernel_samples = json::array();
            if (profile) {
                // Time individual recorded commands using command-buffer timestamps.
                // Counter sampling an entire 3D step exceeds the device's sample-buffer
                // capacity. Serial diagnostic times are separate from throughput above.
                metalrt::Graph graph;
                auto recording = graph.record_guard(&s);
                auto probe = trainer.forward(ns.get(), default_loss_scale<T>(), x, target);
                trainer.backward(ns.get(), *probe, x);
                trainer.optimizer_step(ns.get(), default_loss_scale<T>());
                recording.finish();
                const auto commands = graph.commands();
                for (int repeat = 0; repeat < 3; ++repeat) {
                    for (const auto& command : commands) {
                        const auto name = command.kind == metalrt::Command::Kind::Compute
                                              ? command.compute.kernel.name()
                                              : std::string("transfer");
                        s.begin_gpu_timing();
                        s.enqueue(command);
                        auto timing = s.end_gpu_timing();
                        kernel_samples.push_back({{"round", repeat},
                                                  {"kernel", name},
                                                  {"valid", timing.valid()},
                                                  {"ms", timing.occupied_seconds * 1000}});
                    }
                }
            }
            auto sorted = rounds;
            std::sort(sorted.begin(), sorted.end());
            std::cout << json{{"encoder", kind},
                              {"bins", kind == "OneBlob" ? bins : 0},
                              {"batch", n},
                              {"width", width},
                              {"layout", soa ? "soa" : "default"},
                              {"training_ms", sorted[2]},
                              {"rounds_ms", rounds},
                              {"initial_loss", initial},
                              {"final_loss", final},
                              {"weight_checksum", checksum},
                              {"warmup_steps", 20},
                              {"measured_steps", 320},
                              {"precision_bytes", sizeof(T)},
                              {"input_gradients", false},
                              {"jit", false},
                              {"parameters", model->n_params()},
                              {"mlp_parameters", mlp->n_params()},
                              {"encoding_parameters", enc->n_params()},
                              {"encoding_config", enc->hyperparams()},
                              {"input_dims", dims},
                              {"kernel_samples", kernel_samples},
                              {"device", device.capabilities().name}}
                             .dump()
                      << std::endl;
        }
    }
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
