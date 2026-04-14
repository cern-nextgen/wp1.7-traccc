/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Tbb include(s).
#include <tbb/concurrent_queue.h>

// System include(s).
#include <functional>
#include <ostream>
#include <stop_token>
#include <thread>
#include <vector>

namespace traccc {
class threadpool {
    public:
    /// Wait policy for the threads when there are no tasks to execute
    enum class wait_policy {
        spin,   ///< Threads will spin while waiting for tasks, consuming CPU
                ///< resources
        yield,  ///< Threads will yield their time slice while waiting for tasks
        block   ///< Threads will sleep while waiting for tasks
    };

    /// Construct a thread pool with the given number of threads and
    /// wait policy.
    /// @param num_threads Number of threads in the pool, must be greater than 0
    /// @param policy Wait policy for the threads when there are no tasks to
    /// execute
    ///
    threadpool(size_t num_threads, wait_policy policy = wait_policy::yield);
    threadpool(const threadpool&) = delete;
    threadpool& operator=(const threadpool&) = delete;
    threadpool(threadpool&&) = default;
    threadpool& operator=(threadpool&&) = default;
    /// Destructor that requests all threads to stop and waits for them to
    /// finish.
    ~threadpool();

    /// Enqueue a task to be executed by the thread pool.
    /// @tparam T Type of the task, must be callable with no arguments and
    /// return void
    /// @param task The task to be executed
    ///
    template <typename T>
    void enqueue(T&& task) {
        m_queue.push(std::forward<T>(task));
    }

    private:
    tbb::concurrent_bounded_queue<std::function<void()>> m_queue;
    std::vector<std::jthread> m_workers;
    wait_policy m_policy;

    void run_spinning(std::stop_token token);  /// Run loop for threads with
                                               /// spinning wait policy
    void run_yielding(std::stop_token token);  /// Run loop for threads with
                                               /// yielding wait policy
    void run_blocking(std::stop_token token);  /// Run loop for threads with
                                               /// blocking wait policy
};

std::ostream& operator<<(std::ostream& os, threadpool::wait_policy policy);
}  // namespace traccc
