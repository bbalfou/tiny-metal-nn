#pragma once

#include <tiny-metal-nn/common.h>
#include <tiny-metal-nn/object.h>

#include <tiny-metal-nn/encoding.h>
#include <tiny-metal-nn/network.h>
#include <tiny-metal-nn/network_with_input_encoding.h>
#include <tiny-metal-nn/optimizer.h>
#include <tiny-metal-nn/trainer.h>

#include <type_traits>

namespace tcnn {

struct TrainableModel {
    std::shared_ptr<Loss<network_precision_t>> loss;
    std::shared_ptr<Optimizer<network_precision_t>> optimizer;
    std::shared_ptr<NetworkWithInputEncoding<network_precision_t>> network;
    std::shared_ptr<Trainer<float, network_precision_t, network_precision_t>> trainer;
};

inline TrainableModel create_from_config(uint32_t n_input_dims, uint32_t n_output_dims,
                                         json config) {
    std::shared_ptr<Loss<network_precision_t>> loss{
        create_loss<network_precision_t>(config.value("loss", json::object()))};
    std::shared_ptr<Optimizer<network_precision_t>> optimizer{
        create_optimizer<network_precision_t>(config.value("optimizer", json::object()))};
    auto network = std::make_shared<NetworkWithInputEncoding<network_precision_t>>(
        n_input_dims, n_output_dims, config.value("encoding", json::object()),
        config.value("network", json::object()));
    auto trainer = std::make_shared<Trainer<float, network_precision_t, network_precision_t>>(
        network, optimizer, loss);
    return {loss, optimizer, network, trainer};
}

} // namespace tcnn
