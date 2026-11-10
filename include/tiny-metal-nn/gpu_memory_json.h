#pragma once
#include <tiny-metal-nn/gpu_memory.h>
#include <nlohmann/json.hpp>

namespace tcnn {
// Synchronize GPU data into the binary representation used by model snapshots.
inline nlohmann::json::binary_t gpu_memory_to_json_binary(const void* gpu_data, size_t n_bytes) {
    nlohmann::json::binary_t data_cpu;
    data_cpu.resize(n_bytes);
    METAL_CHECK_THROW(metalMemcpy(data_cpu.data(), gpu_data, n_bytes, metalMemcpyDeviceToHost));
    return data_cpu;
}

// Restore an exact byte span from a snapshot after checking its size.
inline void json_binary_to_gpu_memory(const nlohmann::json::binary_t& cpu_data, void* gpu_data,
                                      size_t n_bytes) {
    if (cpu_data.size() < n_bytes)
        throw std::invalid_argument("snapshot byte span is too small");
    METAL_CHECK_THROW(metalMemcpy(gpu_data, cpu_data.data(), n_bytes, metalMemcpyHostToDevice));
}

// Serialize the complete owned allocation as JSON binary (CBOR preserves it directly).
template <class T> void to_json(nlohmann::json& j, const GPUMemory<T>& gpu_data) {
    j = gpu_memory_to_json_binary(gpu_data.data(), gpu_data.get_bytes());
}

// Accept both native binary JSON and its text JSON {bytes, subtype} representation.
template <class T> void from_json(const nlohmann::json& j, GPUMemory<T>& gpu_data) {
    nlohmann::json::binary_t cpu_data;
    if (j.is_binary())
        cpu_data = j.get_binary();
    else if (j.is_object()) {
        const auto& bytes = j.at("bytes");
        if (!bytes.is_array())
            throw std::invalid_argument("snapshot bytes must be an array");
        cpu_data.resize(bytes.size());
        for (size_t i = 0; i < bytes.size(); ++i) {
            const auto v = bytes.at(i).template get<int>();
            if (v < 0 || v > 255)
                throw std::invalid_argument("snapshot byte is out of range");
            cpu_data[i] = static_cast<uint8_t>(v);
        }
    } else
        throw std::invalid_argument("Invalid json type: must be either binary or object");
    if (cpu_data.size() % sizeof(T))
        throw std::invalid_argument("snapshot byte count is not an element multiple");
    GPUMemory<T> replacement{cpu_data.size() / sizeof(T)};
    json_binary_to_gpu_memory(cpu_data, replacement.data(), replacement.get_bytes());
    gpu_data = std::move(replacement);
}
} // namespace tcnn
