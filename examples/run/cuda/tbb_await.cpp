// Local include(s).
#include "tbb_await.hpp"

// Project include(s).
#include "traccc/cuda/utils/stream.hpp"

// TBB include(s).
#include <tbb/task.h>

// CUDA include(s).
#include <cuda_runtime_api.h>

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
namespace {
void suspend_stream_callback(void* tag) {
    tbb::task::resume(*static_cast<tbb::task::suspend_point*>(tag));
}
}  // namespace

void tbb_await_callback(const traccc::cuda::stream& stream,
                        vecmem::abstract_event&) {
    tbb::task::suspend_point
        suspend_point;  // suspension point address must remain valid when
                        // resumption callback is called
    tbb::task::suspend([&stream, &suspend_point](auto tag) {
        suspend_point = tag;
        auto cuda_stream = reinterpret_cast<cudaStream_t>(stream.cudaStream());
        CUDA_ERROR_CHECK(cudaLaunchHostFunc(
            cuda_stream, suspend_stream_callback, &suspend_point));
    });
    CUDA_ERROR_CHECK(cudaGetLastError());
}

}  // namespace traccc::cuda
