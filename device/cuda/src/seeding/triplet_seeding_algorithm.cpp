/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/cuda/seeding/triplet_seeding_algorithm.hpp"

#include "../utils/utils.hpp"
#include "triplet_seeding_kernel.hpp"

namespace traccc::cuda {

triplet_seeding_algorithm::triplet_seeding_algorithm(
    const seedfinder_config& finder_config,
    const spacepoint_grid_config& grid_config,
    const seedfilter_config& filter_config, const traccc::memory_resource& mr,
    vecmem::copy& copy, cuda::stream& str, std::unique_ptr<const Logger> logger,
    await_function_t await_func)
    : device::triplet_seeding_algorithm(finder_config, grid_config,
                                        filter_config, mr, copy,
                                        std::move(logger)),
      cuda::algorithm_base{str},
      m_await_function(await_func) {}

void triplet_seeding_algorithm::count_grid_capacities_kernel(
    const count_grid_capacities_kernel_payload& payload) const {

    launch_count_grid_capacities_kernel(payload, details::get_stream(stream()),
                                        warp_size());
}

void triplet_seeding_algorithm::populate_grid_kernel(
    const populate_grid_kernel_payload& payload) const {

    launch_populate_grid_kernel(payload, details::get_stream(stream()),
                                warp_size());
}

void triplet_seeding_algorithm::count_doublets_kernel(
    const count_doublets_kernel_payload& payload) const {

    launch_count_doublets_kernel(payload, details::get_stream(stream()),
                                 warp_size());
}

void triplet_seeding_algorithm::find_doublets_kernel(
    const find_doublets_kernel_payload& payload) const {

    launch_find_doublets_kernel(payload, details::get_stream(stream()),
                                warp_size());
}

void triplet_seeding_algorithm::count_triplets_kernel(
    const count_triplets_kernel_payload& payload) const {

    launch_count_triplets_kernel(payload, details::get_stream(stream()),
                                 warp_size());
}

void triplet_seeding_algorithm::triplet_counts_reduction_kernel(
    const triplet_counts_reduction_kernel_payload& payload) const {

    launch_triplet_counts_reduction_kernel(
        payload, details::get_stream(stream()), warp_size());
}

void triplet_seeding_algorithm::find_triplets_kernel(
    const find_triplets_kernel_payload& payload) const {

    launch_find_triplets_kernel(payload, details::get_stream(stream()),
                                warp_size());
}

void triplet_seeding_algorithm::update_triplet_weights_kernel(
    const update_triplet_weights_kernel_payload& payload) const {

    launch_update_triplet_weights_kernel(payload, details::get_stream(stream()),
                                         warp_size());
}

void triplet_seeding_algorithm::select_seeds_kernel(
    const select_seeds_kernel_payload& payload) const {

    launch_select_seeds_kernel(payload, details::get_stream(stream()),
                               warp_size());
}

exec::task<void> triplet_seeding_algorithm::await() const {
    co_await m_await_function(stream());
}

}  // namespace traccc::cuda
