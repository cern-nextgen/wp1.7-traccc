// Local include(s).
#include "suspend_exec.hpp"

// Project include(s).
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/execution/task.hpp"

// CUDA includes(s).
#include <cuda_runtime_api.h>

#define CUDA_ERROR_CHECK(EXP)                                                  \
    do {                                                                       \
        const cudaError_t errorCode = EXP;                                     \
        if (errorCode != cudaSuccess) {                                        \
            throw std::runtime_error(std::string("Failed to run " #EXP " (") + \
                                     cudaGetErrorString(errorCode) + ")");     \
        }                                                                      \
    } while (false)

namespace traccc::cuda {

/// Wrapper sender suspending execution until all operations on a CUDA
/// stream are complete.
class StreamAwaitable {
    public:
    explicit StreamAwaitable(cudaStream_t stream) : m_stream(stream) {}
    bool await_ready() const noexcept { return false; }
    template <typename T>
    void await_suspend(std::coroutine_handle<T> handle) {
        auto error =
            cudaLaunchHostFunc(m_stream, callback<T>, handle.address());
        if (error != cudaSuccess) {
            // resume immediately if the callback could not be registered
            m_error = error;
            handle.promise().reschedule();
        }
    }
    cudaError_t await_resume() const noexcept { return m_error; }

    private:
    cudaStream_t m_stream;
    cudaError_t m_error = cudaSuccess;

    template <typename T>
    static void callback(void* context) {
        auto handle = std::coroutine_handle<T>::from_address(context);
        handle.promise().reschedule();
    }
};

task<void> suspend_exec(const cuda::stream& stream) {
    auto cuda_stream = static_cast<cudaStream_t>(stream.cudaStream());
    CUDA_ERROR_CHECK(co_await StreamAwaitable{cuda_stream});
    co_return;
}

}  // namespace traccc::cuda
