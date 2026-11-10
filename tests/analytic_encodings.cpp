// Independent scalar formulas validate native traversal, padding and view guards.
#include <tiny-metal-nn/encoding.h>
#include "../kernels/encodings/math.metal"
#include <iostream>
using namespace tcnn;
namespace scalar = tcnn_encoding_math;

template <class T> void run(metal::Stream& stream) {
    for (uint32_t kind = 0; kind <= 5; ++kind)
        for (uint32_t degree : {1u, 2u, 4u, 8u, 16u, 32u, 64u}) {
            if (kind == 5 && degree > 8)
                continue;
            const char* names[] = {"Empty",        "Identity", "Frequency",
                                   "TriangleWave", "OneBlob",  "SphericalHarmonics"};
            auto enc =
                std::unique_ptr<Encoding<T>>(create_encoding<T>(3, {{"otype", names[kind]},
                                                                    {"n_frequencies", degree},
                                                                    {"n_bins", degree},
                                                                    {"degree", degree},
                                                                    {"scale", 1.3f},
                                                                    {"offset", -.2f}}));
            const uint32_t width = kind == 0   ? 0
                                   : kind == 1 ? 3
                                   : kind == 5 ? degree * degree
                                               : 3 * degree * (kind == 2 ? 2 : 1);
            for (uint32_t batch : {1u, 31u, 129u, 256u})
                for (auto layout : {CM, RM})
                    for (uint32_t padding : {0u, 1u, 5u, 32u})
                        for (uint32_t gap : {0u, 3u}) {
                            const uint32_t rows = width + padding;
                            if (!rows)
                                continue;
                            enc->set_padded_output_width(rows);
                            const uint32_t stride = (layout == CM ? rows : batch) + gap;
                            GPUMemory<T> memory(size_t(stride) * (layout == CM ? batch : rows) +
                                                16);
                            GPUMatrixDynamic<T> output(memory.data() + 4, rows, batch, layout,
                                                       stride);
                            GPUMatrixDynamic<float> input(3, batch, layout);
                            for (size_t i = 0; i < input.n_elements(); ++i)
                                input.data()[i] = float(int(i * 17 % 211) - 31) / 149;
                            std::fill(memory.data(), memory.data() + memory.size(), T(-17));
                            std::vector<T> expected(memory.size(), T(-17));
                            auto xv = input.view();
                            auto yv = output.view();
                            scalar::Args a{};
                            a.x = {3, batch, xv.stride_i, xv.stride_j};
                            a.y = {rows, batch, yv.stride_i, yv.stride_j};
                            a.kind = kind;
                            a.degree = degree;
                            a.dims = 3;
                            a.width = width;
                            a.resolution = padding;
                            a.padding = 1;
                            a.scale = 1.3f;
                            a.offset = -.2f;
                            for (uint32_t i = 0; i < rows * batch; ++i)
                                scalar::forward<T>(a, input.data(), nullptr, nullptr,
                                                   expected.data() + 4, i);
                            auto context =
                                enc->forward_impl(stream.get(), input, &output, false, true);
                            if (batch % BATCH_SIZE_GRANULARITY)
                                enc->forward_impl(stream.get(), input, &output);
                            else
                                enc->inference_mixed_precision(stream.get(), input, output);
                            stream.runtime().synchronize();
                            for (size_t i = 0; i < memory.size(); ++i) {
                                float actual = float(memory.data()[i]), ref = float(expected[i]);
                                if (!std::isfinite(actual) ||
                                    std::abs(actual - ref) >
                                        (sizeof(T) == 2 ? 3e-3f : 5e-5f) * (1 + std::abs(ref)))
                                    throw std::runtime_error(
                                        std::string(names[kind]) +
                                        " forward/guard mismatch degree=" + std::to_string(degree));
                            }
                            if (kind >= 2 && kind <= 4 && degree <= 16) {
                                GPUMatrixDynamic<float> gradient(3, batch, layout);
                                std::vector<float> reference(gradient.n_elements());
                                a.dx = a.x;
                                GPUMatrixDynamic<T> upstream(rows, batch, layout);
                                for (size_t i = 0; i < upstream.n_elements(); ++i)
                                    upstream.data()[i] = T(.01f * float(int(i % 17) - 8));
                                auto uv = upstream.view();
                                a.dy = {rows, batch, uv.stride_i, uv.stride_j};
                                for (uint32_t i = 0; i < 3 * batch; ++i)
                                    scalar::backward_input<T>(a, input.data(), nullptr, nullptr,
                                                              upstream.data(), reference.data(),
                                                              nullptr, i);
                                enc->backward_impl(stream.get(), *context, input, output, upstream,
                                                   &gradient);
                                stream.runtime().synchronize();
                                for (size_t i = 0; i < reference.size(); ++i)
                                    if (!std::isfinite(gradient.data()[i]) ||
                                        std::abs(gradient.data()[i] - reference[i]) >
                                            3e-4f * (1 + std::abs(reference[i])))
                                        throw std::runtime_error(
                                            std::string(names[kind]) +
                                            " input gradient mismatch degree=" +
                                            std::to_string(degree) +
                                            " actual=" + std::to_string(gradient.data()[i]) +
                                            " reference=" + std::to_string(reference[i]));
                            }
                        }
        }
}
int main() try {
    auto device = metalrt::Device::system_default();
    metal::set_default_device(device);
    metal::Stream stream{device};
    run<float>(stream);
    run<half>(stream);
    std::cout << "Analytic FP16/FP32 layouts, partial batches, padding and guards passed\n";
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
