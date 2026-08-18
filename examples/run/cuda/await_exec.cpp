// Local include(s).
#include "await_exec.hpp"

// Project include(s).
#include "traccc/cuda/utils/algorithm_base.hpp"
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/execution/task.hpp"

// CUDA includes(s).
#include <cuda_runtime_api.h>

// Beman.execution include(s).
#include <beman/execution/execution.hpp>

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
    template <beman::execution::receiver Receiver>
    class stream_await_operation;

    struct env {};

    using sender_concept = beman::execution::sender_t;
    using completion_signatures =
        beman::execution::completion_signatures<beman::execution::set_value_t(
            cudaError_t)>;

    stream_await_sender(const cudaStream_t stream) : m_stream(stream) {}
    env get_env() const noexcept { return {}; }

    template <beman::execution::receiver Receiver>
    auto connect(Receiver&& receiver) const {
        return stream_await_operation<std::remove_cvref_t<Receiver>>(
            std::forward<Receiver>(receiver), m_stream);
    }

    private:
    cudaStream_t m_stream;
};

/// Operation state associated with @c stream_await_sender
///
template <beman::execution::receiver Receiver>
class stream_await_sender::stream_await_operation {
    public:
    using operation_state_concept = beman::execution::operation_state_t;

    stream_await_operation(Receiver&& recv, const cudaStream_t stream)
        : m_receiver(std::forward<Receiver>(recv)), m_stream(stream) {}

    void start() & noexcept {

        auto error = cudaLaunchHostFunc(m_stream, callback, &m_receiver);
        // resume immediately if the callback could not be registered
        if (error != cudaSuccess) {
            beman::execution::set_value(std::move(m_receiver), error);
        }
    }

    private:
    std::remove_cvref_t<Receiver> m_receiver;
    cudaStream_t m_stream;

    static void CUDART_CB callback(void* userData) noexcept {
        auto& recv = *static_cast<Receiver*>(userData);
        beman::execution::set_value(std::move(recv), cudaSuccess);
    }
};

static_assert(beman::execution::sender<stream_await_sender>);

task<void> await_callback(const cuda::stream& stream, vecmem::abstract_event&) {
    auto cuda_stream = static_cast<cudaStream_t>(stream.cudaStream());
    CUDA_ERROR_CHECK(co_await stream_await_sender{cuda_stream});
    co_return;
}

task<void> await_defer_event_sync::operator()(
    const cuda::stream& stream, vecmem::abstract_event& event) const {
    co_await beman::execution::starts_on(threadpool_scheduler{threadpool},
                                         await_event_sync(stream, event));
}

task<void> await_defer_stream_sync::operator()(
    const cuda::stream& stream, vecmem::abstract_event& event) const {
    co_await beman::execution::starts_on(threadpool_scheduler{threadpool},
                                         await_stream_sync(stream, event));
}
}  // namespace traccc::cuda
