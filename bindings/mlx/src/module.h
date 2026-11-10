// Native MLX operations over the unchanged non-template NN facade.
#pragma once

#include <mlx/array.h>
#include <mlx/utils.h>
#include <tiny-metal-nn/cpp_api.h>

#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace tinymetalnn {
namespace mlx_bindings {

struct ModuleState {
    explicit ModuleState(std::unique_ptr<tcnn::cpp::Module> module);
    std::unique_ptr<tcnn::cpp::Module> native;
    std::mutex mutex;
    bool jit = false;
    bool no_second_input_grad = false;
    float loss_scale = 1.0f;
};

class Module {
  public:
    explicit Module(std::unique_ptr<tcnn::cpp::Module> module);
    mlx::core::array forward(const mlx::core::array& input, const mlx::core::array& params,
                             mlx::core::StreamOrDevice stream = {});
    mlx::core::array initial_params(size_t seed, mlx::core::StreamOrDevice stream = {});
    uint32_t n_input_dims() const;
    uint32_t n_output_dims() const;
    size_t n_params() const;
    tcnn::cpp::Precision param_precision() const;
    tcnn::cpp::Precision output_precision() const;
    tcnn::cpp::json hyperparams() const;
    std::string name() const;
    bool jit_fusion() const;
    void set_jit_fusion(bool enabled);
    bool backward_backward_input_no_input_grad() const;
    void set_backward_backward_input_no_input_grad(bool enabled);
    float loss_scale() const;
    void set_loss_scale(float value);

  private:
    std::shared_ptr<ModuleState> state_;
};

std::shared_ptr<Module> create_encoding(uint32_t n_input_dims, const tcnn::cpp::json& encoding,
                                        tcnn::cpp::Precision precision);
std::shared_ptr<Module> create_network(uint32_t n_input_dims, uint32_t n_output_dims,
                                       const tcnn::cpp::json& network);
std::shared_ptr<Module> create_network_with_input_encoding(uint32_t n_input_dims,
                                                           uint32_t n_output_dims,
                                                           const tcnn::cpp::json& encoding,
                                                           const tcnn::cpp::json& network);
bool supports_jit_fusion(int device = -1);
// Keep the caller's logical dimensions in the graph before native row padding
// and output slicing specialize them. This is an identity/layout operation.
std::pair<mlx::core::array, mlx::core::array> guard_inputs(const mlx::core::array& input,
                                                           const mlx::core::array& params,
                                                           mlx::core::StreamOrDevice stream = {});

// MLX executes primitives lazily, outside the Python binding call. Python
// callbacks never run on the scheduler/encoding thread. They are delivered at
// the next binding entry or explicit flush_logs() after mx.eval/synchronize.
using LogRecord = std::pair<tcnn::cpp::LogSeverity, std::string>;
void capture_logs();
std::vector<LogRecord> take_logs();

} // namespace mlx_bindings
} // namespace tinymetalnn
