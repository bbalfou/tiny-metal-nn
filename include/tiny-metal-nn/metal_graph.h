#pragma once
#include <tiny-metal-nn/common_host.h>
#include <tiny-metal-nn/metal.h>
#include <algorithm>
#include <deque>
#include <exception>
#include <memory>
#include <utility>

namespace tcnn {
class MetalGraph;

namespace detail {
struct GraphAccess;
}

// Track active NN captures on this host thread, as the reference API does.
inline std::deque<MetalGraph*>& current_captures() {
    static thread_local std::deque<MetalGraph*> s_current_captures;
    return s_current_captures;
}

inline MetalGraph* current_capture() {
    return current_captures().empty() ? nullptr : current_captures().front();
}

// Keep the original graph API while delegating versioned replay to metalrt.
// Runtime graph extensions remain in metalrt instead of changing this facade.
class MetalGraph {
  public:
    // Cancel an unfinished recording before its owner disappears; an outstanding
    // guard then becomes a harmless no-op rather than referring to a dead owner.
    ~MetalGraph() {
        if (m_capture && !m_capture->finished) {
            m_capture->finished = true;
            m_capture->scope.cancel();
            erase_capture();
        }
    }

    // Capture and submit at scope exit. A null stream or nested recording joins
    // ordinary execution; exceptional scope exit cancels partial recording.
    ScopeGuard capture_guard(metalStream_t stream) {
        throw_deferred_error();
        if (!stream || detail::resolve_stream(stream)->is_recording())
            return {};
        if (m_capture && !m_capture->finished)
            throw std::runtime_error("MetalGraph is already capturing");
        auto state =
            std::make_shared<Capture>(m_graph.capture_guard(detail::resolve_stream(stream)));
        m_capture = state;
        try {
            current_captures().push_back(this);
            return ScopeGuard{[this, state] {
                if (!state->finished)
                    complete(state, false);
            }};
        } catch (...) {
            state->finished = true;
            state->scope.cancel();
            erase_capture();
            throw;
        }
    }

    // Forget the cached plan. A previous implicit-completion failure is reported
    // here or on the next capture instead of being silently discarded.
    void reset() {
        throw_deferred_error();
        if (m_capture && !m_capture->finished)
            throw std::runtime_error("Cannot reset an active MetalGraph capture");
        m_graph.reset();
    }

    // Request the same end-of-capture synchronization used by reference allocators.
    void schedule_synchronize() { m_graph.schedule_synchronize(); }

  private:
    struct Capture {
        explicit Capture(metalrt::Scope value)
            : scope{std::move(value)}, exceptions{std::uncaught_exceptions()} {}

        metalrt::Scope scope;
        int exceptions;
        bool finished = false;
    };

    // Remove only this graph's bookkeeping, including on out-of-order cancellation.
    void erase_capture() {
        auto& captures = current_captures();
        auto found = std::find(captures.begin(), captures.end(), this);
        if (found != captures.end())
            captures.erase(found);
    }

    // Trainer uses explicit completion internally so it cannot update weights
    // after a failed submission. The public ScopeGuard signature stays unchanged.
    void complete(const std::shared_ptr<Capture>& state, bool explicit_completion) {
        if (state->finished)
            return;
        state->finished = true;
        const bool unwinding = std::uncaught_exceptions() > state->exceptions;
        const bool ordered = !current_captures().empty() && current_captures().back() == this;
        erase_capture();
        if (unwinding) {
            state->scope.cancel();
            return;
        }
        try {
            if (!ordered) {
                state->scope.cancel();
                throw std::runtime_error("MetalGraph: captures must finish in reverse order");
            }
            state->scope.finish();
        } catch (...) {
            m_error = std::current_exception();
            if (explicit_completion)
                throw_deferred_error();
        }
    }

    void throw_deferred_error() {
        if (m_error) {
            auto error = std::exchange(m_error, {});
            std::rethrow_exception(error);
        }
    }

    metalrt::Graph m_graph;
    std::shared_ptr<Capture> m_capture;
    std::exception_ptr m_error;
    friend struct detail::GraphAccess;
};

namespace detail {
// Internal access avoids adding a nonreference finish() method to ScopeGuard or MetalGraph.
struct GraphAccess {
    static void finish(MetalGraph& graph) {
        if (graph.m_capture)
            graph.complete(graph.m_capture, true);
        graph.throw_deferred_error();
    }
};
} // namespace detail
} // namespace tcnn
