/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../utils/cuda_error_handling.hpp"
#include "../utils/global_index.hpp"
#include "silicon_pixel_spacepoint_formation_kernel.hpp"

// Project include(s).
#include "traccc/geometry/detector.hpp"
#include "traccc/seeding/device/form_spacepoints.hpp"

namespace traccc::cuda {
namespace kernels {

/// Kernel wrapping @c device::form_spacepoints
template <typename detector_t>
__global__ void __launch_bounds__(1024, 1) form_spacepoints(
    typename detector_t::view detector,
    typename edm::measurement_collection<
        typename detector_t::device::algebra_type>::const_view measurements,
    edm::spacepoint_collection::view spacepoints)
    requires(traccc::is_detector_traits<detector_t>)
{
    device::form_spacepoints<detector_t>(details::global_index1(), detector,
                                         measurements, spacepoints);
}

}  // namespace kernels

void launch_form_spacepoints_kernel(
    const traccc::device::silicon_pixel_spacepoint_formation_kernel_payload&
        payload,
    cudaStream_t stream, unsigned int warp_size) {

    const unsigned int n_threads = warp_size * 8;
    const unsigned int n_blocks =
        (payload.n_measurements + n_threads - 1) / n_threads;
    detector_buffer_visitor<detector_type_list>(
        payload.detector, [&]<typename detector_traits_t>(
                              const typename detector_traits_t::view& det) {
            kernels::form_spacepoints<detector_traits_t>
                <<<n_blocks, n_threads, 0, stream>>>(det, payload.measurements,
                                                     payload.spacepoints);
        });
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

}  // namespace traccc::cuda
