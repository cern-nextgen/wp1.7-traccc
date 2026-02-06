/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/clusterization/device/clusterization_kernel_payload.hpp"
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/edm/silicon_cluster_collection.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

// CUDA include(s).
#include <cuda_runtime_api.h>

namespace traccc::cuda {

/// Host-side interface for the sanity check whether the input cells are valid
/// for clusterization.
bool input_is_valid_on_device(
    vecmem::memory_resource& mr, const vecmem::copy& copy, stream& stream,
    const edm::silicon_cell_collection::const_view& cells);

/// Host-side interface for the main CCL kernel.
void launch_ccl_kernel(
    const traccc::device::clusterization_ccl_kernel_payload& payload,
    cudaStream_t stream);

/// Host-side interface for the cluster reification kernel.
void launch_reify_cluster_data_kernel(
    unsigned int num_cells,
    const vecmem::data::vector_view<unsigned int>& disjoint_set,
    edm::silicon_cluster_collection::view& cluster_data, cudaStream_t stream,
    unsigned int warp_size);

}  // namespace traccc::cuda
