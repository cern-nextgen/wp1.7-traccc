/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/seeding/device/silicon_pixel_spacepoint_formation_kernel_payload.hpp"

// CUDA include(s).
#include <cuda_runtime_api.h>

namespace traccc::cuda {

/// Host-side interface for silicon pixel spacepoint formation kernel.
///
/// @param payload The payload for the kernel
/// @param stream The CUDA stream to launch the kernel in
/// @param warp_size The warp size of the GPU being used
///
void launch_form_spacepoints_kernel(
    const traccc::device::silicon_pixel_spacepoint_formation_kernel_payload&
        payload,
    cudaStream_t stream, unsigned int warp_size);

}  // namespace traccc::cuda
