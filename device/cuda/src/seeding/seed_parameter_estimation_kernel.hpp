/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).

// Project include(s).
#include "traccc/seeding/device/estimate_track_params.hpp"
#include "traccc/seeding/device/seed_parameter_estimation_kernel_payload.hpp"

// CUDA include(s).
#include <cuda_runtime_api.h>

namespace traccc::cuda {

/// Host-side interface for seed parameter estimation kernel
///
/// @param payload The payload for the kernel
/// @param stream The CUDA stream to launch the kernel in
/// @param warp_size The warp size of the GPU being used
///
void launch_estimate_track_params_kernel(
    const traccc::device::estimate_seed_params_kernel_payload& payload,
    cudaStream_t stream, unsigned int warp_size);

}  // namespace traccc::cuda
