// Local include(s).
#include "tbb_await.hpp"

// Project include(s).
#include "traccc/cuda/utils/stream.hpp"

// TBB include(s).
#include <tbb/task.h>

// CUDA include(s).
#include <cuda_runtime_api.h>

// System include(s).
#include <exception>

/// Helper macro for checking the return value of CUDA function calls
#define CUDA_ERROR_CHECK(EXP)                                                  \
    do {                                                                       \
        const cudaError_t errorCode = EXP;                                     \
        if (errorCode != cudaSuccess) {                                        \
            throw std::runtime_error(std::string("Failed to run " #EXP " (") + \
                                     cudaGetErrorString(errorCode) + ")");     \
        }                                                                      \
    } while (false)

namespace {
struct polling_task {
    traccc::threadpool& pool;
    vecmem::abstract_event& event;
    std::exception_ptr& exception;
    tbb::task::suspend_point suspend_point;

    void operator()() {
        try {
            if (event.is_ready()) {
                tbb::task::resume(suspend_point);
            } else {
                pool.enqueue(*this);
            }
        } catch (...) {
            exception = std::current_exception();
            tbb::task::resume(suspend_point);
        }
    }
};
}  // namespace

namespace traccc::cuda {
namespace {
void CUDART_CB suspend_stream_callback(void* tag) {
    tbb::task::resume(*static_cast<tbb::task::suspend_point*>(tag));
}
}  // namespace

void tbb_await_callback(const traccc::cuda::stream& stream,
                        vecmem::abstract_event&) {
    cudaError_t err = cudaSuccess;
    tbb::task::suspend_point
        suspend_point;  // suspension point address must remain valid when
                        // resumption callback is called
    tbb::task::suspend([&err, &stream, &suspend_point](auto tag) {
        suspend_point = tag;
        auto cuda_stream = reinterpret_cast<cudaStream_t>(stream.cudaStream());
        err = cudaLaunchHostFunc(cuda_stream, suspend_stream_callback,
                                 &suspend_point);
        // resume immediately if the callback could not be registered
        if (err != cudaSuccess) {
            tbb::task::resume(suspend_point);
        }
    });
    CUDA_ERROR_CHECK(err);
}

void tbb_await_poll::operator()(const traccc::cuda::stream&,
                                vecmem::abstract_event& event) const {
    std::exception_ptr exception = nullptr;
    tbb::task::suspend([&event, &exception, this](auto tag) {
        auto task = ::polling_task{threadpool, event, exception, tag};
        task();  // eagerly execute the task once, to avoid queuing if the event
                 // is already done, then continue on the threadpool
        // alternatively, skip the eager execution and always enqueue the task
        // pool.enqueue(std::move(task));
    });
    if (exception) {
        std::rethrow_exception(exception);
    }
}

void tbb_await_defer_sync_event::operator()(
    const traccc::cuda::stream&, vecmem::abstract_event& event) const {
    std::exception_ptr exception = nullptr;
    tbb::task::suspend([&event, &exception, this](auto tag) {
        threadpool.enqueue([&event, &exception, tag]() {
            try {
                event.wait();
            } catch (...) {
                exception = std::current_exception();
            }
            tbb::task::resume(tag);
        });
    });
    if (exception) {
        std::rethrow_exception(exception);
    }
}

void tbb_await_defer_sync_stream::operator()(const traccc::cuda::stream& stream,
                                             vecmem::abstract_event&) const {
    std::exception_ptr exception = nullptr;
    tbb::task::suspend([&stream, &exception, this](auto tag) {
        threadpool.enqueue([&stream, &exception, tag]() {
            try {
                stream.synchronize();
            } catch (...) {
                exception = std::current_exception();
            }
            tbb::task::resume(tag);
        });
    });
    if (exception) {
        std::rethrow_exception(exception);
    }
}

}  // namespace traccc::cuda
