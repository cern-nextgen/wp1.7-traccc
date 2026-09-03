#pragma once

// Project include(s).
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/examples/utils/threadpool.hpp"

// Vecmem include(s).
#include <vecmem/utils/abstract_event.hpp>

namespace traccc::cuda {

/// Await CUDA stream completion with a callback using Boost.Fiber
void boost_fiber_await_callback(const traccc::cuda::stream& stream,
                                vecmem::abstract_event& event);

/// Await CUDA event completion by polling in a service threadpool using
/// Boost.Fiber
struct boost_fiber_await_poll {
    traccc::threadpool& threadpool;
    void operator()(const traccc::cuda::stream& stream,
                    vecmem::abstract_event& event) const;
};

/// Await CUDA event completion by deferring synchronization to a service
/// threadpool using Boost.Fiber
struct boost_fiber_await_defer_sync_event {
    traccc::threadpool& threadpool;
    void operator()(const traccc::cuda::stream& stream,
                    vecmem::abstract_event& event) const;
};

/// Await CUDA stream completion by deferring synchronization to a service
/// threadpool using Boost.Fiber
struct boost_fiber_await_defer_sync_stream {
    traccc::threadpool& threadpool;
    void operator()(const traccc::cuda::stream& stream,
                    vecmem::abstract_event& event) const;
};

}  // namespace traccc::cuda
