/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/seeding/device/triplet_seeding_kernel_payloads.hpp"

// CUDA include(s).
#include <cuda_runtime_api.h>

namespace traccc::cuda {

void launch_count_grid_capacities_kernel(
    const traccc::device::triplet_seeding_count_grid_capacities_kernel_payload&
        payload,
    cudaStream_t stream, unsigned int warp_size);

void launch_populate_grid_kernel(
    const traccc::device::triplet_seeding_populate_grid_kernel_payload& payload,
    cudaStream_t stream, unsigned int warp_size);

void launch_count_doublets_kernel(
    const traccc::device::triplet_seeding_count_doublets_kernel_payload&
        payload,
    cudaStream_t stream, unsigned int warp_size);

void launch_find_doublets_kernel(
    const traccc::device::triplet_seeding_find_doublets_kernel_payload& payload,
    cudaStream_t stream, unsigned int warp_size);

void launch_count_triplets_kernel(
    const traccc::device::triplet_seeding_count_triplets_kernel_payload&
        payload,
    cudaStream_t stream, unsigned int warp_size);

void launch_triplet_counts_reduction_kernel(
    const traccc::device::
        triplet_seeding_triplet_counts_reduction_kernel_payload& payload,
    cudaStream_t stream, unsigned int warp_size);

void launch_find_triplets_kernel(
    const traccc::device::triplet_seeding_find_triplets_kernel_payload& payload,
    cudaStream_t stream, unsigned int warp_size);

void launch_update_triplet_weights_kernel(
    const traccc::device::triplet_seeding_update_triplet_weights_kernel_payload&
        payload,
    cudaStream_t stream, unsigned int warp_size);

void launch_select_seeds_kernel(
    const traccc::device::triplet_seeding_select_seeds_kernel_payload& payload,
    cudaStream_t stream, unsigned int warp_size);

}  // namespace traccc::cuda
