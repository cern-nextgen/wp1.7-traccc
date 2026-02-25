/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/cuda/utils/algorithm_base.hpp"

#include "../utils/utils.hpp"
#include "traccc/execution/task.hpp"

namespace traccc::cuda {

algorithm_base::algorithm_base(cuda::stream& str)
    : m_stream(str), m_warp_size(details::get_warp_size(str.device())) {}

cuda::stream& algorithm_base::stream() const {

    return m_stream.get();
}

unsigned int algorithm_base::warp_size() const {

    return m_warp_size;
}

task<void> default_await_function(const cuda::stream&,
                                  vecmem::abstract_event& event) {
    event.wait();
    co_return;
}

task<void> await_stream_sync(const cuda::stream& stream,
                             vecmem::abstract_event&) {
    stream.synchronize();
    co_return;
}

task<void> await_event_sync(const cuda::stream&,
                            vecmem::abstract_event& event) {
    event.wait();
    co_return;
}

}  // namespace traccc::cuda
