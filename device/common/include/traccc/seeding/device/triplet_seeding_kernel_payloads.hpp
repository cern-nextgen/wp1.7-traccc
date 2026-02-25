/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/prefix_sum_element.hpp"
#include "traccc/edm/device/device_doublet.hpp"
#include "traccc/edm/device/device_triplet.hpp"
#include "traccc/edm/device/doublet_counter.hpp"
#include "traccc/edm/device/triplet_counter.hpp"

// Project include(s).
#include "traccc/edm/seed_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

struct triplet_seeding_count_grid_capacities_kernel_payload {
    edm::spacepoint_collection::const_view::size_type n_spacepoints;
    const seedfinder_config& config;
    const traccc::details::spacepoint_grid_types::host::axis_p0_type& phi_axis;
    const traccc::details::spacepoint_grid_types::host::axis_p1_type& z_axis;
    const edm::spacepoint_collection::const_view& spacepoints;
    vecmem::data::vector_view<unsigned int>& grid_capacities;
};

struct triplet_seeding_populate_grid_kernel_payload {
    edm::spacepoint_collection::const_view::size_type n_spacepoints;
    const seedfinder_config& config;
    const edm::spacepoint_collection::const_view& spacepoints;
    traccc::details::spacepoint_grid_types::view& grid;
    vecmem::data::vector_view<prefix_sum_element_t>& grid_prefix_sum;
};

struct triplet_seeding_count_doublets_kernel_payload {
    edm::spacepoint_collection::const_view::size_type n_spacepoints;
    const seedfinder_config& config;
    const edm::spacepoint_collection::const_view& spacepoints;
    const traccc::details::spacepoint_grid_types::const_view& grid;
    const vecmem::data::vector_view<const prefix_sum_element_t>&
        grid_prefix_sum;
    doublet_counter_collection_types::view& doublet_counter;
    unsigned int& nMidBot;
    unsigned int& nMidTop;
};

struct triplet_seeding_find_doublets_kernel_payload {
    device::doublet_counter_collection_types::const_view::size_type n_doublets;
    const seedfinder_config& config;
    const edm::spacepoint_collection::const_view& spacepoints;
    const traccc::details::spacepoint_grid_types::const_view& grid;
    const doublet_counter_collection_types::const_view& doublet_counter;
    device_doublet_collection_types::view& mb_doublets;
    device_doublet_collection_types::view& mt_doublets;
};

struct triplet_seeding_count_triplets_kernel_payload {
    unsigned int nMidBot;
    const seedfinder_config& config;
    const edm::spacepoint_collection::const_view& spacepoints;
    const traccc::details::spacepoint_grid_types::const_view& grid;
    const doublet_counter_collection_types::const_view& doublet_counter;
    const device_doublet_collection_types::const_view& mb_doublets;
    const device_doublet_collection_types::const_view& mt_doublets;
    triplet_counter_spM_collection_types::view& spM_counter;
    triplet_counter_collection_types::view& midBot_counter;
};

struct triplet_seeding_triplet_counts_reduction_kernel_payload {
    device::doublet_counter_collection_types::const_view::size_type n_doublets;
    const doublet_counter_collection_types::const_view& doublet_counter;
    triplet_counter_spM_collection_types::view& spM_counter;
    unsigned int& nTriplets;
};

struct triplet_seeding_find_triplets_kernel_payload {
    unsigned int nMidBot;
    const seedfinder_config& finding_config;
    const seedfilter_config& filter_config;
    const edm::spacepoint_collection::const_view& spacepoints;
    const traccc::details::spacepoint_grid_types::const_view& grid;
    const doublet_counter_collection_types::const_view& doublet_counter;
    const device_doublet_collection_types::const_view& mt_doublets;
    const triplet_counter_spM_collection_types::const_view& spM_tc;
    const triplet_counter_collection_types::const_view& midBot_tc;
    device_triplet_collection_types::view& triplets;
};

struct triplet_seeding_update_triplet_weights_kernel_payload {
    device_triplet_collection_types::const_view::size_type n_triplets;
    const seedfilter_config& config;
    const edm::spacepoint_collection::const_view& spacepoints;
    const triplet_counter_spM_collection_types::const_view& spM_tc;
    const triplet_counter_collection_types::const_view& midBot_tc;
    device_triplet_collection_types::view& triplets;
};

struct triplet_seeding_select_seeds_kernel_payload {
    device::doublet_counter_collection_types::const_view::size_type n_doublets;
    const seedfinder_config& finder_config;
    const seedfilter_config& filter_config;
    const edm::spacepoint_collection::const_view& spacepoints;
    const traccc::details::spacepoint_grid_types::const_view& grid;
    const triplet_counter_spM_collection_types::const_view& spM_tc;
    const triplet_counter_collection_types::const_view& midBot_tc;
    const device_triplet_collection_types::const_view& triplets;
    edm::seed_collection::view& seeds;
};

}  // namespace traccc::device
