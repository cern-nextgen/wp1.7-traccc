// Local include(s).
#include "await_exec.hpp"

// Project include(s).
#include "traccc/cuda/utils/algorithm_base.hpp"
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/execution/task.hpp"

// CUDA includes(s).
#include <cuda_runtime_api.h>

// Boost.Capy include(s).
#include <boost/capy.hpp>

// Standard include(s).
#include <coroutine>
#include <stdexcept>
#include <string>

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

class stream_awaiter {
    public:
    stream_awaiter(cudaStream_t stream) : m_stream(stream) {}

    bool await_ready() const noexcept { return false; }

    void await_suspend(std::coroutine_handle<> handle,
                       boost::capy::io_env const* env) noexcept {
        m_context.handle = handle;
        m_context.env = env;
        m_error = cudaLaunchHostFunc(m_stream, resumption_callback, &m_context);
        // If the callback couldn't be registered, we need to reschedule the
        // coroutine immediately to avoid deadlock.
        if (m_error != cudaSuccess) {
            resumption_callback(&m_context);
        }
    }
    cudaError_t await_resume() const noexcept { return m_error; }

    private:
    struct context {
        std::coroutine_handle<> handle;
        boost::capy::io_env const* env;
    };
    cudaStream_t m_stream;
    cudaError_t m_error = cudaSuccess;
    context m_context;

    static void CUDART_CB resumption_callback(void* userData) {
        auto* ctx = static_cast<context*>(userData);
        ctx->env->executor.post(ctx->handle);
    }
};

struct retry {
    bool await_ready() const noexcept { return false; }
    void await_suspend(std::coroutine_handle<> handle,
                       boost::capy::io_env const* env) const noexcept {
        env->executor.post(handle);
    }
    void await_resume() const noexcept {}
};

task<void> poll(vecmem::abstract_event& event) {
    while (!event.is_ready()) {
        co_await retry{};
    }
}

task<void> await_callback(const cuda::stream& stream, vecmem::abstract_event&) {
    auto cuda_stream = static_cast<cudaStream_t>(stream.cudaStream());
    CUDA_ERROR_CHECK(co_await stream_awaiter{cuda_stream});
    co_return;
}

task<void> await_poll::operator()(const cuda::stream&,
                                  vecmem::abstract_event& event) const {
    auto context = threadpool_executor::threadpool_context{threadpool};
    auto executor = threadpool_executor{context};
    co_await boost::capy::run(executor)(poll(event));
}

task<void> await_defer_event_sync::operator()(
    const cuda::stream& stream, vecmem::abstract_event& event) const {
    auto context = threadpool_executor::threadpool_context{threadpool};
    auto executor = threadpool_executor{context};
    co_await boost::capy::run(executor)(await_event_sync(stream, event));
}

task<void> await_defer_stream_sync::operator()(
    const cuda::stream& stream, vecmem::abstract_event& event) const {
    auto context = threadpool_executor::threadpool_context{threadpool};
    auto executor = threadpool_executor{context};
    co_await boost::capy::run(executor)(await_stream_sync(stream, event));
}
}  // namespace traccc::cuda
