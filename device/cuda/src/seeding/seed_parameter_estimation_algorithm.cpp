/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/cuda/seeding/seed_parameter_estimation_algorithm.hpp"

#include "../utils/utils.hpp"
#include "seed_parameter_estimation_kernel.hpp"

// Project include(s).
#include "traccc/seeding/device/estimate_track_params.hpp"

namespace traccc::cuda {

seed_parameter_estimation_algorithm::seed_parameter_estimation_algorithm(
    const track_params_estimation_config& config,
    const traccc::memory_resource& mr, vecmem::copy& copy, cuda::stream& str,
    std::unique_ptr<const Logger> logger, await_function_t await_func)
    : device::seed_parameter_estimation_algorithm(config, mr, copy,
                                                  std::move(logger)),
      cuda::algorithm_base(str),
      m_await_function(await_func) {}

void seed_parameter_estimation_algorithm::estimate_seed_params_kernel(
    const estimate_seed_params_kernel_payload& payload) const {

    launch_estimate_track_params_kernel(payload, details::get_stream(stream()),
                                        warp_size());
}

void seed_parameter_estimation_algorithm::await() const {
    m_await_function(stream());
}
}  // namespace traccc::cuda
