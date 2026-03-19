/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../sanity/contiguous_on.cuh"
#include "../sanity/ordered_on.cuh"
#include "../utils/cuda_error_handling.hpp"
#include "./kernels/ccl_kernel.cuh"
#include "./kernels/reify_cluster_data.cuh"
#include "clusterization_kernel.hpp"

// Project include(s).
#include "traccc/utils/projections.hpp"
#include "traccc/utils/relations.hpp"

namespace traccc::cuda {

bool input_is_valid_on_device(
    vecmem::memory_resource& mr, const vecmem::copy& copy, stream& stream,
    const edm::silicon_cell_collection::const_view& cells) {

    return (is_contiguous_on<edm::silicon_cell_collection::const_device>(
                cell_module_projection(), mr, copy, stream, cells) &&
            is_ordered_on<edm::silicon_cell_collection::const_device>(
                channel0_major_cell_order_relation(), mr, copy, stream, cells));
}

void launch_ccl_kernel(
    const traccc::device::clusterization_ccl_kernel_payload& payload,
    cudaStream_t stream) {

    const unsigned int num_blocks =
        (payload.n_cells + (payload.config.target_partition_size()) - 1) /
        payload.config.target_partition_size();
    kernels::ccl_kernel<<<num_blocks, payload.config.threads_per_partition,
                          2 * payload.config.max_partition_size() *
                              sizeof(device::details::index_t),
                          stream>>>(
        payload.config, payload.cells, payload.det_descr, payload.measurements,
        payload.cell_links, payload.f_backup, payload.gf_backup,
        payload.adjc_backup, payload.adjv_backup, payload.backup_mutex,
        payload.disjoint_set, payload.cluster_sizes);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void launch_reify_cluster_data_kernel(
    unsigned int num_cells,
    const vecmem::data::vector_view<unsigned int>& disjoint_set,
    edm::silicon_cluster_collection::view& cluster_data, cudaStream_t stream,
    unsigned int warp_size) {

    const unsigned int num_threads = warp_size * 16u;
    const unsigned int num_blocks = (num_cells + num_threads - 1) / num_threads;
    kernels::reify_cluster_data<<<num_blocks, num_threads, 0, stream>>>(
        disjoint_set, cluster_data);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

}  // namespace traccc::cuda
