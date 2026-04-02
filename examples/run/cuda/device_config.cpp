// Local include(s).
#include "device_config.hpp"

// CUDA include(s).
#include <cuda_runtime_api.h>

// System include(s).
#include <stdexcept>
#include <string>

/// Helper macro for checking the return value of CUDA function calls
#define CUDA_ERROR_CHECK(EXP)                                                  \
    do {                                                                       \
        const cudaError_t errorCode = EXP;                                     \
        if (errorCode != cudaSuccess) {                                        \
            throw std::runtime_error(std::string("Failed to run " #EXP " (") + \
                                     cudaGetErrorString(errorCode) + ")");     \
        }                                                                      \
    } while (false)

namespace traccc::cuda {
void device_config::apply(
    const traccc::opts::device::device_sync_strategy strategy) {
    switch (strategy) {
        case traccc::opts::device::device_sync_strategy::automatic:
            CUDA_ERROR_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleAuto));
            break;
        case traccc::opts::device::device_sync_strategy::spin:
            CUDA_ERROR_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
            break;
        case traccc::opts::device::device_sync_strategy::yield:
            CUDA_ERROR_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleYield));
            break;
        case traccc::opts::device::device_sync_strategy::block:
            CUDA_ERROR_CHECK(
                cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
            break;
        default:
            throw std::invalid_argument{
                "Unknown device synchronization strategy"};
            break;
    }
}
}  // namespace traccc::cuda
