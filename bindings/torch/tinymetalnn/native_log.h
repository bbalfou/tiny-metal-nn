// Native diagnostic transport for the Python adapter. This header deliberately
// contains no Python state: GCD blocks and native loggers only retain records.
#pragma once

#include <tiny-metal-nn/cpp_api.h>

#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace tinymetalnn {
namespace torch_bindings {

struct NativeLogRecord {
    tcnn::cpp::LogSeverity severity;
    std::string message;
};

class NativeLogQueue {
  public:
    void push(tcnn::cpp::LogSeverity severity, const std::string& message) {
        std::lock_guard<std::mutex> lock{mutex_};
        records_.push_back({severity, message});
    }

    std::vector<NativeLogRecord> take() {
        std::lock_guard<std::mutex> lock{mutex_};
        std::vector<NativeLogRecord> result;
        result.swap(records_);
        return result;
    }

  private:
    std::mutex mutex_;
    std::vector<NativeLogRecord> records_;
};

inline std::shared_ptr<NativeLogQueue>& current_native_logs() {
    static thread_local std::shared_ptr<NativeLogQueue> queue;
    return queue;
}

// Current native diagnostics are synchronous. Retain any future out-of-call
// diagnostic for the next Python call instead of invoking Python on its thread.
inline NativeLogQueue& pending_native_logs() {
    static NativeLogQueue queue;
    return queue;
}

class NativeLogCapture {
  public:
    NativeLogCapture() : NativeLogCapture{std::make_shared<NativeLogQueue>()} {}
    explicit NativeLogCapture(std::shared_ptr<NativeLogQueue> queue)
        : previous_{std::move(current_native_logs())} {
        current_native_logs() = std::move(queue);
    }
    NativeLogCapture(const NativeLogCapture&) = delete;
    NativeLogCapture& operator=(const NativeLogCapture&) = delete;
    ~NativeLogCapture() { current_native_logs() = std::move(previous_); }

    std::shared_ptr<NativeLogQueue> queue() const { return current_native_logs(); }

  private:
    std::shared_ptr<NativeLogQueue> previous_;
};

inline void capture_native_log(tcnn::cpp::LogSeverity severity, const std::string& message) {
    if (auto queue = current_native_logs())
        queue->push(severity, message);
    else
        pending_native_logs().push(severity, message);
}

} // namespace torch_bindings
} // namespace tinymetalnn
