#pragma once

// Project include(s).
#include "traccc/cuda/utils/stream.hpp"

// Vecmem include(s).
#include <vecmem/utils/abstract_event.hpp>

// Stdexec include(s).
#include <exec/task.hpp>

namespace traccc::cuda {

/// Await coroutine that returns a task which suspends execution with callback
/// until all asynchronous operations on the given stream are complete.
///
exec::task<void> await_callback(const cuda::stream& stream,
                                vecmem::abstract_event& event);

}  // namespace traccc::cuda
