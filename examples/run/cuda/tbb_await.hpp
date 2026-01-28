#pragma once

// Project include(s).
#include "traccc/cuda/utils/stream.hpp"

// Vecmem include(s).
#include <vecmem/utils/abstract_event.hpp>

namespace traccc::cuda {

/// Await CUDA stream completion with callback using TBB task suspension
void tbb_await_callback(const traccc::cuda::stream& stream,
                        vecmem::abstract_event& event);
}  // namespace traccc::cuda
