/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/clusterization/device/ccl_kernel_definitions.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// Payload for the @c traccc::device::clusterization_algorithm::ccl_kernel
/// function.
struct clusterization_ccl_kernel_payload {
    /// Number of cells in the event
    unsigned int n_cells;
    /// The clustering configuration
    const clustering_config& config;
    /// All cells in an event
    const edm::silicon_cell_collection::const_view& cells;
    /// The detector description
    const silicon_detector_description::const_view& det_descr;
    /// The measurement collection to fill
    edm::measurement_collection<default_algebra>::view& measurements;
    /// Buffer for linking cells to measurements
    vecmem::data::vector_view<unsigned int>& cell_links;
    /// Buffer for backup of the first element links
    vecmem::data::vector_view<details::index_t>& f_backup;
    /// Buffer for backup of the group first element links
    vecmem::data::vector_view<details::index_t>& gf_backup;
    /// Buffer for backup of the adjacency matrix (counts)
    vecmem::data::vector_view<unsigned char>& adjc_backup;
    /// Buffer for backup of the adjacency matrix (values)
    vecmem::data::vector_view<details::index_t>& adjv_backup;
    /// Mutex for the backup structures
    unsigned int* backup_mutex;
    /// Buffer for the disjoint set data structure
    vecmem::data::vector_view<unsigned int>& disjoint_set;
    /// Buffer for the sizes of the clusters
    vecmem::data::vector_view<unsigned int>& cluster_sizes;
};

}  // namespace traccc::device
