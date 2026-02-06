/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/cuda/seeding/silicon_pixel_spacepoint_formation_algorithm.hpp"

#include "../utils/utils.hpp"
#include "silicon_pixel_spacepoint_formation_kernel.hpp"

namespace traccc::cuda {

silicon_pixel_spacepoint_formation_algorithm::
    silicon_pixel_spacepoint_formation_algorithm(
        const traccc::memory_resource& mr, vecmem::copy& copy,
        cuda::stream& str, std::unique_ptr<const Logger> logger,
        await_function_t await_func)
    : device::silicon_pixel_spacepoint_formation_algorithm(mr, copy,
                                                           std::move(logger)),
      cuda::algorithm_base(str),
      m_await_function(await_func) {}

void silicon_pixel_spacepoint_formation_algorithm::form_spacepoints_kernel(
    const form_spacepoints_kernel_payload& payload) const {

    launch_form_spacepoints_kernel(payload, details::get_stream(stream()),
                                   warp_size());
}

exec::task<void> silicon_pixel_spacepoint_formation_algorithm::await() const {
    co_await m_await_function(stream());
}

}  // namespace traccc::cuda
