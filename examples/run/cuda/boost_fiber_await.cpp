// Local include(s).
#include "boost_fiber_await.hpp"

// Project include(s).
#include "traccc/cuda/utils/stream.hpp"

// CUDA include(s).
#include <cuda_runtime_api.h>  // boost/fiber/cuda/waitfor.hpp includes by mistake driver header so this must be placed before as a workaround

// Boost include(s).
#include <boost/fiber/condition_variable.hpp>
#include <boost/fiber/cuda/waitfor.hpp>

// System include(s).
#include <exception>
#include <mutex>

/// Helper macro for checking the return value of CUDA function calls
#define CUDA_ERROR_CHECK(EXP)                                                  \
    do {                                                                       \
        const cudaError_t errorCode = EXP;                                     \
        if (errorCode != cudaSuccess) {                                        \
            throw std::runtime_error(std::string("Failed to run " #EXP " (") + \
                                     cudaGetErrorString(errorCode) + ")");     \
        }                                                                      \
    } while (false)

struct polling_task {

    traccc::threadpool& pool;
    vecmem::abstract_event& event;
    std::exception_ptr& exception;
    boost::fibers::condition_variable& cv;
    boost::fibers::mutex& mutex;
    bool& done;

    void operator()() {
        try {
            if (event.is_ready()) {
                {
                    std::lock_guard lock(mutex);
                    done = true;
                }
                cv.notify_one();
            } else {
                pool.enqueue(*this);
            }
        } catch (...) {
            {
                std::lock_guard lock(mutex);
                exception = std::current_exception();
                done = true;
            }
            cv.notify_one();
        }
    }
};

namespace traccc::cuda {

void boost_fiber_await_callback(const traccc::cuda::stream& stream,
                                vecmem::abstract_event&) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream.cudaStream());
    auto result = boost::fibers::cuda::waitfor_all(cuda_stream);
    CUDA_ERROR_CHECK(std::get<1>(result));
}

void boost_fiber_await_poll::operator()(const traccc::cuda::stream&,
                                        vecmem::abstract_event& event) const {
    std::exception_ptr exception = nullptr;
    boost::fibers::condition_variable cv;
    boost::fibers::mutex mutex;
    auto done = false;

    auto task = ::polling_task{threadpool, event, exception, cv, mutex, done};
    task();
    std::unique_lock lock(mutex);
    cv.wait(lock, [&done] { return done; });
    if (exception) {
        std::rethrow_exception(exception);
    }
}

void boost_fiber_await_defer_sync_event::operator()(
    const traccc::cuda::stream&, vecmem::abstract_event& event) const {
    std::exception_ptr exception = nullptr;
    auto done = false;
    boost::fibers::condition_variable_any cv;
    boost::fibers::mutex mutex;
    threadpool.enqueue([&exception, &done, &cv, &mutex, &event]() {
        auto exception_local = std::exception_ptr{};
        try {
            event.wait();
        } catch (...) {
            exception_local = std::current_exception();
        }
        {
            std::lock_guard lock(mutex);
            exception = exception_local;
            done = true;
        }
        cv.notify_one();
    });
    {
        std::unique_lock lock(mutex);
        cv.wait(lock, [&done] { return done; });
    }
    if (exception) {
        std::rethrow_exception(exception);
    }
}

void boost_fiber_await_defer_sync_stream::operator()(
    const traccc::cuda::stream& stream, vecmem::abstract_event&) const {
    std::exception_ptr exception = nullptr;
    auto done = false;
    boost::fibers::condition_variable_any cv;
    boost::fibers::mutex mutex;
    threadpool.enqueue([&exception, &done, &cv, &mutex, &stream]() {
        auto exception_local = std::exception_ptr{};
        try {
            stream.synchronize();
        } catch (...) {
            exception_local = std::current_exception();
        }
        {
            std::lock_guard lock(mutex);
            exception = exception_local;
            done = true;
        }
        cv.notify_one();
    });
    {
        std::unique_lock lock(mutex);
        cv.wait(lock, [&done] { return done; });
    }
    if (exception) {
        std::rethrow_exception(exception);
    }
}
}  // namespace traccc::cuda
