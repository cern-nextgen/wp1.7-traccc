// Local include(s).
#include "await_exec.hpp"

// Project include(s).
#include "traccc/cuda/utils/algorithm_base.hpp"
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/execution/schedule_on.hpp"
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
class StreamCallbackAwaitable {
    public:
    explicit StreamCallbackAwaitable(cudaStream_t stream) : m_stream(stream) {}
    bool await_ready() const noexcept { return false; }
    template <typename T>
    void await_suspend(std::coroutine_handle<T> handle) {
        auto error = cudaLaunchHostFunc(m_stream, resumption_callback<T>,
                                        handle.address());
        if (error != cudaSuccess) {
            m_error = error;
            handle.promise().reschedule();
        }
    }
    cudaError_t await_resume() const noexcept { return m_error; }

    private:
    cudaStream_t m_stream;
    cudaError_t m_error = cudaSuccess;

    template <typename T>
    static void CUDART_CB resumption_callback(void* context) {
        auto handle = std::coroutine_handle<T>::from_address(context);
        handle.promise().reschedule();
    }
};

struct Retry {
    bool await_ready() const noexcept { return false; }
    template <typename Promise>
    void await_suspend(std::coroutine_handle<Promise> handle) const {
        handle.promise().reschedule();
    }
    void await_resume() const noexcept {}
};

task<void> poll(vecmem::abstract_event& event) {
    while (!event.is_ready()) {
        co_await Retry{};
    }
}

task<void> await_callback(const cuda::stream& stream, vecmem::abstract_event&) {
    auto cuda_stream = static_cast<cudaStream_t>(stream.cudaStream());
    CUDA_ERROR_CHECK(co_await StreamCallbackAwaitable{cuda_stream});
    co_return;
}

task<void> await_poll::operator()(const cuda::stream&,
                                  vecmem::abstract_event& event) const {
    auto threadpool_scheduler = [this](std::coroutine_handle<> handle) {
        threadpool.enqueue([handle]() { handle.resume(); });
    };

    co_await schedule_on(std::move(threadpool_scheduler), poll(event));
}

task<void> await_defer_event_sync::operator()(
    const cuda::stream& stream, vecmem::abstract_event& event) const {
    auto threadpool_scheduler = [this](std::coroutine_handle<> handle) {
        threadpool.enqueue([handle]() { handle.resume(); });
    };
    co_await schedule_on(std::move(threadpool_scheduler),
                         await_event_sync(stream, event));
}

task<void> await_defer_stream_sync::operator()(
    const cuda::stream& stream, vecmem::abstract_event& event) const {
    auto threadpool_scheduler = [this](std::coroutine_handle<> handle) {
        threadpool.enqueue([handle]() { handle.resume(); });
    };
    co_await schedule_on(std::move(threadpool_scheduler),
                         await_stream_sync(stream, event));
}
}  // namespace traccc::cuda
