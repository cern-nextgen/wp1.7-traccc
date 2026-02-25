// Local include(s).
#include "suspend_exec.hpp"

// Project include(s).
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/execution/task.hpp"

// CUDA includes(s).
#include <cuda_runtime_api.h>
#include <driver_types.h>

// Stdexec include(s).
#include <stdexec/execution.hpp>

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
class stream_await_sender {
    public:
    // associated operation state
    template <stdexec::receiver Receiver>
    class stream_await_operation;

    using sender_concept = stdexec::sender_t;
    using completion_signatures = stdexec::completion_signatures<
        stdexec::set_value_t(void), stdexec::set_error_t(std::exception_ptr)>;

    stream_await_sender(const cudaStream_t stream) : m_stream(stream) {}
    stdexec::env<> get_env() const noexcept { return {}; }

    template <stdexec::receiver Receiver>
    auto connect(Receiver&& receiver) const {
        return stream_await_operation<std::remove_cvref_t<Receiver>>(
            std::forward<Receiver>(receiver), m_stream);
    }

    private:
    cudaStream_t m_stream;
};

/// Operation state associated with @c stream_await_sender
///
template <stdexec::receiver Receiver>
class stream_await_sender::stream_await_operation {
    public:
    using operation_state_concept = stdexec::operation_state_t;

    stream_await_operation(Receiver&& recv, const cudaStream_t stream)
        : m_receiver(std::forward<Receiver>(recv)), m_stream(stream) {}

    void start() & noexcept {
        try {
            CUDA_ERROR_CHECK(
                cudaLaunchHostFunc(m_stream, callback, &m_receiver));
        } catch (...) {
            stdexec::set_error(std::move(m_receiver), std::current_exception());
        }
    }

    private:
    std::remove_cvref_t<Receiver> m_receiver;
    cudaStream_t m_stream;

    static void callback(void* userData) {
        auto& recv = *static_cast<Receiver*>(userData);
        try {
            CUDA_ERROR_CHECK(cudaGetLastError());
            stdexec::set_value(std::move(recv));
        } catch (...) {
            stdexec::set_error(std::move(recv), std::current_exception());
            return;
        }
    }
};

static_assert(stdexec::sender<stream_await_sender>);

task<void> suspend_exec(const cuda::stream& stream) {
    auto cuda_stream = static_cast<cudaStream_t>(stream.cudaStream());
    co_await stream_await_sender{cuda_stream};
    co_return;
}

}  // namespace traccc::cuda
