/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"

#include "../utils/utils.hpp"
#include "clusterization_kernel.hpp"

namespace traccc::cuda {

clusterization_algorithm::clusterization_algorithm(
    const traccc::memory_resource& mr, vecmem::copy& copy, cuda::stream& str,
    const config_type& config, std::unique_ptr<const Logger> logger,
    await_function_t await_func)
    : device::clusterization_algorithm(mr, copy, config, std::move(logger)),
      cuda::algorithm_base(str),
      m_await_function(await_func) {}

bool clusterization_algorithm::input_is_valid(
    const edm::silicon_cell_collection::const_view& cells) const {
    return input_is_valid_on_device(mr().main, copy(), stream(), cells);
}

void clusterization_algorithm::ccl_kernel(
    const ccl_kernel_payload& payload) const {
    launch_ccl_kernel(payload, details::get_stream(stream()));
}

void clusterization_algorithm::cluster_maker_kernel(
    unsigned int num_cells,
    const vecmem::data::vector_view<unsigned int>& disjoint_set,
    edm::silicon_cluster_collection::view& cluster_data) const {
    launch_reify_cluster_data_kernel(num_cells, disjoint_set, cluster_data,
                                     details::get_stream(stream()),
                                     warp_size());
}

exec::task<void> clusterization_algorithm::await() const {
    co_await m_await_function(stream());
}

}  // namespace traccc::cuda
