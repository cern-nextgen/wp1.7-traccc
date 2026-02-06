#pragma once

// Project include(s).
#include "traccc/cuda/utils/stream.hpp"

// Stdexec include(s).
#include <exec/task.hpp>

namespace traccc::cuda {

/// Await function that returns a task which suspends execution until all
/// asynchronous operations on the given stream are complete.
///
exec::task<void> suspend_exec(const cuda::stream& stream);

}  // namespace traccc::cuda
