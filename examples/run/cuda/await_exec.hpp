#pragma once

// Project include(s).
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/examples/utils/threadpool.hpp"
#include "traccc/execution/task.hpp"

// Vecmem include(s).
#include <vecmem/utils/abstract_event.hpp>

namespace traccc::cuda {

/// Await CUDA steam completion by registering a callback on the stream
task<void> await_callback(const cuda::stream& stream,
                          vecmem::abstract_event& event);

/// Await CUDA event completion by polling in a service threadpool
struct await_poll {
    traccc::threadpool& threadpool;
    task<void> operator()(const traccc::cuda::stream& stream,
                          vecmem::abstract_event& event) const;
};

/// Await CUDA event completion by deferring synchronization to a service
/// threadpool
struct await_defer_event_sync {
    traccc::threadpool& threadpool;
    task<void> operator()(const traccc::cuda::stream& stream,
                          vecmem::abstract_event& event) const;
};

/// Await CUDA stream completion by deferring synchronization to a service
/// threadpool
struct await_defer_stream_sync {
    traccc::threadpool& threadpool;
    task<void> operator()(const traccc::cuda::stream& stream,
                          vecmem::abstract_event& event) const;
};

}  // namespace traccc::cuda
