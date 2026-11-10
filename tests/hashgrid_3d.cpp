// Check production dispatch against independent scalar geometry and CPU accumulation.
#include <tiny-metal-nn/encoding.h>
#include <tiny-metal-nn/encodings/multi_level_interface.h>
#include <tiny-metal-nn/detail/kernels/encodings/hash_grid_parameter_gradient.h>
#include "../kernels/encodings/math.metal"
#include <iostream>
namespace scalar = tcnn_encoding_math;
using namespace tcnn;
using T = half;
void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
void run(metal::Stream& ns, uint32_t dims) {
    json config = {{"otype", "HashGrid"},     {"n_levels", 16},        {"n_features_per_level", 2},
                   {"log2_hashmap_size", 19}, {"base_resolution", 16}, {"per_level_scale", 2.0}};
    // Old checkpoint keys must not reactivate comparison-only implementations.
    config["grouped_parameter_gradients"] = false;
    config["grouped_inference"] = false;
    auto fast = std::unique_ptr<Encoding<T>>(create_encoding<T>(dims, config, 16));
    require(!fast->hyperparams().contains("grouped_parameter_gradients") &&
                !fast->hyperparams().contains("grouped_inference"),
            "obsolete dispatch switches serialized");
    auto* lattice = dynamic_cast<MultiLevelEncoding<T>*>(fast.get());
    const size_t count = fast->n_params();
    GPUMemory<T> weights(count), actual(count + 8), expected(count + 8);
    for (size_t i = 0; i < count; ++i)
        weights.data()[i] = T(.001f * float(int(i % 17) - 8));
    fast->set_params(weights.data(), weights.data(), actual.data() + 4);
    auto& s = ns.runtime();
    // Mix layouts, out-of-domain coordinates, coherent cells and SIMD tails.
    for (uint32_t batch : {0u, 1u, 255u, 256u, 4096u})
        for (auto layout : {CM, RM})
            for (bool coherent : {false, true}) {
                const uint32_t xs = (layout == CM ? dims : batch) + 3,
                               ys = (layout == CM ? 32 : batch) + 3;
                GPUMemory<float> xb(size_t(xs) * (layout == CM ? batch : dims) + 8), limits(batch);
                GPUMemory<T> yb(size_t(ys) * (layout == CM ? batch : 32) + 8), db(yb.size());
                GPUMatrixDynamic<float> x(xb.data() + 4, dims, batch, layout, xs);
                GPUMatrixDynamic<T> y(yb.data() + 4, 32, batch, layout, ys),
                    dy(db.data() + 4, 32, batch, layout, ys);
                auto xv = x.view();
                auto dv = dy.view();
                for (uint32_t j = 0; j < batch; ++j) {
                    limits.data()[j] = j % 7 == 0 ? .4f : 1.f;
                    for (uint32_t d = 0; d < dims; ++d)
                        x.data()[size_t(d) * xv.stride_i + size_t(j) * xv.stride_j] =
                            coherent ? .313f : float(int((j * 137 + d * 29) % 997) - 99) / 797.f;
                    for (uint32_t r = 0; r < 32; ++r)
                        dy.data()[size_t(r) * dv.stride_i + size_t(j) * dv.stride_j] =
                            T(j % 11 == 0 ? 0.f : .0003f * float(int((j * 7 + r) % 19) - 9));
                }
                dynamic_cast<MultiLevelEncoding<T>*>(fast.get())->set_max_level_gpu(limits.data());
                std::fill(yb.data(), yb.data() + yb.size(), T(-17));
                auto fc = fast->forward_impl(ns.get(), x, &y, false, true);
                s.synchronize();
                GPUMatrixDynamic<float> fast_dx(dims, batch, layout), slow_dx(dims, batch, layout);
                const bool need_dx = batch == 256 && !coherent;
                std::vector<float> reference(count, 0.f);
                std::vector<T> reference_y(yb.size(), T(-17));
                auto dxv = slow_dx.view();
                auto yv = y.view();
                for (uint32_t level = 0; level < 16; ++level) {
                    scalar::Args a{};
                    a.kind = 6;
                    a.dims = dims;
                    a.features = 2;
                    a.levels = 16;
                    a.level = level;
                    a.x = {dims, batch, xv.stride_i, xv.stride_j};
                    a.y = {32, batch, yv.stride_i, yv.stride_j};
                    a.dy = {32, batch, dv.stride_i, dv.stride_j};
                    a.dx = {dims, batch, dxv.stride_i, dxv.stride_j};
                    a.interpolation = 1;
                    a.hash_type = 1;
                    a.grid_type = 0;
                    a.resolution = 16u << level;
                    a.scale = float(a.resolution) - 1.f;
                    a.max_level = 1000.f;
                    a.has_limits = 1;
                    a.max_input_grad_dims = dims;
                    a.accumulate = level != 0;
                    const auto& offsets = lattice->params_offset_table();
                    a.table_size = offsets.data[level + 1] - offsets.data[level];
                    const size_t offset = size_t(offsets.data[level]) * 2;
                    for (uint32_t j = 0; j < batch; ++j) {
                        if (scalar::inactive(a, limits.data(), j))
                            continue;
                        uint32_t ids[25];
                        float w[25], g[25], h[25];
                        const auto corners =
                            scalar::geometry(a, x.data(), j, UINT32_MAX, UINT32_MAX, ids, w, g, h);
                        for (uint32_t c = 0; c < corners; ++c)
                            for (uint32_t f = 0; f < 2; ++f)
                                reference[offset + ids[c] * 2 + f] +=
                                    w[c] * float(dy.data()[scalar::at(a.dy, level * 2 + f, j)]);
                    }
                    for (uint32_t i = 0; i < 2 * batch; ++i)
                        scalar::forward<T>(a, x.data(), weights.data() + offset, limits.data(),
                                           reference_y.data() + 4, i);
                    if (need_dx)
                        for (uint32_t i = 0; i < dims * batch; ++i)
                            scalar::backward_input<T>(a, x.data(), weights.data() + offset,
                                                      limits.data(), dy.data(), slow_dx.data(),
                                                      nullptr, i);
                }
                // The independent oracle also verifies grouped inference and masks.
                fast->inference_mixed_precision_impl(ns.get(), x, y);
                s.synchronize();
                for (size_t i = 0; i < yb.size(); ++i)
                    require(std::isfinite(float(yb.data()[i])) &&
                                std::abs(float(yb.data()[i]) - float(reference_y[i])) <=
                                    1e-5f + 3e-3f * std::abs(float(reference_y[i])),
                            "inference differs from scalar reference");
                for (auto mode :
                     {GradientMode::Overwrite, GradientMode::Accumulate, GradientMode::Ignore}) {
                    std::fill(actual.data(), actual.data() + actual.size(), T(.025f));
                    std::fill(expected.data(), expected.data() + expected.size(), T(.025f));
                    metalrt::Graph graph;
                    auto rec = graph.record_guard(&s);
                    fast->backward_impl(ns.get(), *fc, x, y, dy, need_dx ? &fast_dx : nullptr,
                                        false, mode);
                    rec.finish();
                    if (batch && mode != GradientMode::Ignore) {
                        bool found = false;
                        for (const auto& c : graph.commands())
                            if (c.kind == metalrt::Command::Kind::Compute)
                                found |= c.compute.kernel.name() == "encoding_hash_grid_classify";
                        require(found, "eligible case did not use grouped classifier");
                    }
                    graph.prepare(&s);
                    graph.launch(&s).wait();
                    if (mode != GradientMode::Ignore)
                        for (size_t i = 0; i < count; ++i)
                            expected.data()[i + 4] =
                                T(reference[i] +
                                  (mode == GradientMode::Accumulate ? float(T(.025f)) : 0.f));
                    if (need_dx)
                        for (size_t i = 0; i < fast_dx.n_elements(); ++i)
                            require(std::isfinite(fast_dx.data()[i]) &&
                                        std::abs(fast_dx.data()[i] - slow_dx.data()[i]) <=
                                            1e-5f * (1 + std::abs(slow_dx.data()[i])),
                                    "input gradient changed");
                    for (size_t i = 0; i < actual.size(); ++i) {
                        const float a = float(actual.data()[i]), e = float(expected.data()[i]);
                        const bool guard = i < 4 || i >= count + 4;
                        if (!std::isfinite(a) ||
                            std::abs(a - e) > (guard || mode == GradientMode::Ignore
                                                   ? 0.f
                                                   : 1e-5f + 3e-3f * std::abs(e)))
                            throw std::runtime_error(
                                "gradient mismatch dims=" + std::to_string(dims) +
                                " batch=" + std::to_string(batch) + " index=" + std::to_string(i) +
                                " actual=" + std::to_string(a) + " expected=" + std::to_string(e));
                    }
                }
            }
}
int main() try {
    using Traits = detail::kernels::encodings::HashGridParameterGradientTraits;
    require(Traits::supports_configuration(3, 16, 2, 19, 16, 2.f, 1, 1, false, false, true),
            "3D eligibility");
    require(!Traits::supports_configuration(3, 16, 2, 19, 16, 2.f, 1, 1, false, false, false),
            "FP32 fallback");
    require(!Traits::supports_configuration(3, 16, 2, 15, 16, 1.5f, 1, 1, false, false, true),
            "small table fallback");
    require(!Traits::supports_configuration(3, 16, 2, 19, 16, 2.f, 2, 1, false, false, true),
            "smoothstep fallback");
    require(!Traits::supports_dispatch(9, UINT32_MAX, true), "expanded grid overflow fallback");
    auto device = metalrt::Device::system_default();
    metal::set_default_device(device);
    for (auto policy : {metalrt::DispatchPolicy::Direct, metalrt::DispatchPolicy::Indirect}) {
        metal::Stream ns{device, policy};
        detail::set_indirect_dispatch_enabled(ns.get(),
                                              policy == metalrt::DispatchPolicy::Indirect);
        run(ns, 3);
        run(ns, 2);
    }
    std::cout
        << "3D and 2D grouped gradients match the scalar CPU reference; layouts, tails, masks, "
           "modes, replay and dispatch policies passed\n";
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
