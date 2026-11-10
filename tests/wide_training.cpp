// Check wide training across depths, layouts and partial tiles. An optional
// output path also records forward/dX/dW arrays for cross-build comparisons.
#include <tiny-metal-nn/trainer.h>
#include <tiny-metal-nn/network.h>
#include <tiny-metal-nn/detail/mlp.h>
#include <fstream>
#include <iostream>
using namespace tcnn;
#ifdef WIDE_TEST_FP32
using P = float;
#else
using P = half;
#endif
int main(int argc, char** argv) {
    try {
        if (argc > 2)
            return 2;
        const bool record = argc == 2;
        auto device = metalrt::Device::system_default();
        if (device.capabilities().apple_gpu_family < 9)
            return 77;
        metal::set_default_device(device);
        metal::Stream stream{device};
        std::ofstream out(record ? argv[1] : "/dev/null", std::ios::binary);
        json manifest = json::array();
        size_t offset = 0;
        for (unsigned w : {64u, 128u})
            for (unsigned depth : {0u, 1u, 2u, 3u, 5u, 8u, 16u})
                for (unsigned shape : {0u, 1u, 2u, 3u}) {
                    unsigned inputs = shape == 0 ? 128 : shape == 1 ? w : shape == 2 ? 32 : 48;
                    unsigned outputs = shape == 0 ? 3 : shape == 1 ? w : shape == 2 ? 32 : w + 16;
                    auto net = std::shared_ptr<Network<P>>(create_network<P>(
                        {{"otype", "FullyFusedMLP"},
                         {"n_input_dims", inputs},
                         {"n_output_dims", outputs},
                         {"n_neurons", w},
                         {"n_hidden_layers", depth},
                         {"activation", shape == 2 ? "Sine" : "ReLU"},
                         {"output_activation", shape == 2 ? "Sigmoid" : "None"}}));
                    net->set_jit_fusion(false);
                    auto opt =
                        std::shared_ptr<Optimizer<P>>(create_optimizer<P>({{"otype", "Adam"}}));
                    auto loss = std::shared_ptr<Loss<P>>(create_loss<P>({{"otype", "L2"}}));
                    Trainer<P, P, P> trainer{net, opt, loss, 1337};
                    for (unsigned batch : {1u, 7u, 16u, 31u, 32u, 63u, 64u, 65u, 255u, 256u, 513u})
                        for (auto layout : {AoS, SoA}) {
                            GPUMatrixDynamic<P> x{inputs, batch, stream, layout},
                                dx{inputs, batch, stream, layout};
                            GPUMatrix<P> y{net->padded_output_width(), batch},
                                dy{net->padded_output_width(), batch};
                            for (size_t i = 0; i < x.n_elements(); ++i)
                                x.data()[i] = P(float(int((i * 37) % 251) - 125) / 256);
                            for (size_t i = 0; i < dy.n_elements(); ++i)
                                dy.data()[i] = P(float(int((i * 17) % 127) - 63) / 1024);
                            auto* impl = dynamic_cast<detail::BaselineMLP<P>*>(net.get());
                            auto ctx = impl->forward_impl(stream, x, &y, false, true);
                            impl->backward_impl(stream, *ctx, x, y, dy, &dx, false,
                                                GradientMode::Overwrite);
                            stream.runtime().synchronize();
                            auto save = [&](const P* p, size_t n) {
                                std::vector<float> v(n);
                                for (size_t i = 0; i < n; ++i) {
                                    v[i] = float(p[i]);
                                    if (!std::isfinite(v[i]))
                                        throw std::runtime_error("nonfinite");
                                }
                                out.write((char*)v.data(), n * sizeof(float));
                                json j = {{"offset", offset}, {"count", n}};
                                offset += n;
                                return j;
                            };
                            manifest.push_back(
                                {{"width", w},
                                 {"depth", depth},
                                 {"inputs", inputs},
                                 {"outputs", outputs},
                                 {"batch", batch},
                                 {"layout", int(layout)},
                                 {"y", save(y.data(), y.n_elements())},
                                 {"dx", save(dx.data(), dx.n_elements())},
                                 {"dw", save(trainer.param_gradients(), net->n_params())}});
                            // Compare an unaligned tile against an equivalent zero-gradient-padded
                            // batch. This checks masked publication, dX, dW and both physical
                            // layouts.
                            const unsigned padded = (batch + 255) / 256 * 256;
                            if (padded != batch) {
                                std::vector<float> gradients(net->n_params());
                                for (size_t i = 0; i < gradients.size(); ++i)
                                    gradients[i] = float(trainer.param_gradients()[i]);
                                GPUMatrixDynamic<P> px{inputs, padded, stream, layout},
                                    pdx{inputs, padded, stream, layout};
                                GPUMatrix<P> py{net->padded_output_width(), padded},
                                    pdy{net->padded_output_width(), padded};
                                std::fill(px.data(), px.data() + px.n_elements(), P(0));
                                std::fill(pdy.data(), pdy.data() + pdy.n_elements(), P(0));
                                auto index = [&](unsigned row, unsigned sample, unsigned rows,
                                                 unsigned samples) {
                                    return layout == AoS ? size_t(sample) * rows + row
                                                         : size_t(row) * samples + sample;
                                };
                                for (unsigned n = 0; n < batch; ++n) {
                                    for (unsigned k = 0; k < inputs; ++k)
                                        px.data()[index(k, n, inputs, padded)] =
                                            x.data()[index(k, n, inputs, batch)];
                                    for (unsigned k = 0; k < py.m(); ++k)
                                        pdy.data()[size_t(n) * py.m() + k] =
                                            dy.data()[size_t(n) * py.m() + k];
                                }
                                auto pc = impl->forward_impl(stream, px, &py, false, true);
                                impl->backward_impl(stream, *pc, px, py, pdy, &pdx, false,
                                                    GradientMode::Overwrite);
                                stream.runtime().synchronize();
                                auto close = [](float a, float b) {
                                    if (std::abs(a - b) > 0.001f + 0.003f * std::abs(a))
                                        throw std::runtime_error("padded-batch numerical mismatch");
                                };
                                for (unsigned n = 0; n < batch; ++n) {
                                    for (unsigned k = 0; k < py.m(); ++k)
                                        close(float(y.data()[size_t(n) * py.m() + k]),
                                              float(py.data()[size_t(n) * py.m() + k]));
                                    for (unsigned k = 0; k < inputs; ++k)
                                        close(float(dx.data()[index(k, n, inputs, batch)]),
                                              float(pdx.data()[index(k, n, inputs, padded)]));
                                }
                                for (size_t i = 0; i < gradients.size(); ++i)
                                    close(gradients[i], float(trainer.param_gradients()[i]));
                            }
                        }
                    std::cout << "checked " << w << " depth " << depth << " shape " << shape
                              << std::endl;
                }
        if (record)
            std::ofstream(std::string(argv[1]) + ".json") << manifest.dump(2);
        return out ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
