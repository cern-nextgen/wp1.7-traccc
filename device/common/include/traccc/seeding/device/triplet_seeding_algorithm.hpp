/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/algorithm_base.hpp"
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
#include "traccc/seeding/device/triplet_seeding_kernel_payloads.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/messaging.hpp"

// Stdexec include(s).
#include <exec/task.hpp>

// System include(s).
#include <memory>

namespace traccc::device {

/// Main algorithm for performing triplet track seeding
///
/// This algorithm returns a buffer which is not necessarily filled yet. A
/// synchronisation statement is required before destroying this buffer.
///
class triplet_seeding_algorithm
    : public algorithm<edm::seed_collection::buffer(
          const edm::spacepoint_collection::const_view&)>,
      public messaging,
      public algorithm_base {

    public:
    /// Constructor for the seed finding algorithm
    ///
    /// @param finder_config The seed finding configuration
    /// @param grid_config   The spacepoint grid forming configuration
    /// @param filter_config The seed filtering configuration
    /// @param mr The memory resource(s) to use in the algorithm
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param logger The logger instance to use
    ///
    triplet_seeding_algorithm(
        const seedfinder_config& finder_config,
        const spacepoint_grid_config& grid_config,
        const seedfilter_config& filter_config, const memory_resource& mr,
        vecmem::copy& copy,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());
    /// Destructor
    virtual ~triplet_seeding_algorithm();

    /// Operator executing the algorithm.
    ///
    /// @param spacepoints is a view of all spacepoints in the event
    /// @return the buffer of track seeds reconstructed from the spacepoints
    ///
    output_type operator()(const edm::spacepoint_collection::const_view&
                               spacepoints) const override;

    protected:
    /// @name Function(s) to be implemented by derived classes
    /// @{

    /// Payload for the @c count_grid_capacities_kernel function
    using count_grid_capacities_kernel_payload =
        triplet_seeding_count_grid_capacities_kernel_payload;

    /// Spacepoint grid capacity counting kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void count_grid_capacities_kernel(
        const count_grid_capacities_kernel_payload& payload) const = 0;

    /// Payload for the @c populate_grid_kernel function
    using populate_grid_kernel_payload =
        triplet_seeding_populate_grid_kernel_payload;

    /// Spacepoint grid population kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void populate_grid_kernel(
        const populate_grid_kernel_payload& payload) const = 0;

    /// Payload for the @c count_doublets_kernel function
    using count_doublets_kernel_payload =
        triplet_seeding_count_doublets_kernel_payload;

    /// Doublet counting kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void count_doublets_kernel(
        const count_doublets_kernel_payload& payload) const = 0;

    /// Payload for the @c find_doublets_kernel function
    using find_doublets_kernel_payload =
        triplet_seeding_find_doublets_kernel_payload;

    /// Doublet finding kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void find_doublets_kernel(
        const find_doublets_kernel_payload& payload) const = 0;

    /// Payload for the @c count_triplets_kernel function
    using count_triplets_kernel_payload =
        triplet_seeding_count_triplets_kernel_payload;

    /// Triplet counting kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void count_triplets_kernel(
        const count_triplets_kernel_payload& payload) const = 0;

    /// Payload for the @c triplet_counts_reduction_kernel function
    using triplet_counts_reduction_kernel_payload =
        triplet_seeding_triplet_counts_reduction_kernel_payload;

    /// Triplet count reduction kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void triplet_counts_reduction_kernel(
        const triplet_counts_reduction_kernel_payload& payload) const = 0;

    /// Payload for the @c find_triplets_kernel function
    using find_triplets_kernel_payload =
        triplet_seeding_find_triplets_kernel_payload;

    /// Triplet finding kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void find_triplets_kernel(
        const find_triplets_kernel_payload& payload) const = 0;

    /// Payload for the @c update_triplet_weights_kernel function
    using update_triplet_weights_kernel_payload =
        triplet_seeding_update_triplet_weights_kernel_payload;

    /// Triplet weight updater/filler kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void update_triplet_weights_kernel(
        const update_triplet_weights_kernel_payload& payload) const = 0;

    /// Payload for the @c select_seeds_kernel function
    using select_seeds_kernel_payload =
        triplet_seeding_select_seeds_kernel_payload;

    /// Seed selection/filling kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void select_seeds_kernel(
        const select_seeds_kernel_payload& payload) const = 0;

    /// @}

    /// Possibly suspend execution until all asynchronous operations are done
    virtual void await() const = 0;

    private:
    /// Internal data type
    struct data;
    /// Pointer to internal data
    std::unique_ptr<data> m_data;

};  // class triplet_seeding_algorithm

}  // namespace traccc::device
