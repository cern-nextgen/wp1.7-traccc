/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Tbb include(s).
#include <tbb/concurrent_queue.h>

// beman.task include(s).
#include <beman/task/task.hpp>

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

/// Wrapper around a threadpool to be used as a scheduler for beman.execution.
class threadpool_scheduler {
    public:
    using scheduler_concept = beman::execution::scheduler_t;

    /// Construct a threadpool_scheduler that uses the given threadpool.
    /// @param threadpool The threadpool to use for scheduling.
    ///
    /// @note The threadpool_scheduler does not take ownership of the
    /// threadpool, the threadpool should remain valid for the lifetime of the
    /// scheduler.
    ///
    threadpool_scheduler(threadpool& pool);

    class env {
        public:
        env(threadpool* pool) noexcept;

        template <typename T>
        auto query(const beman::execution::get_completion_scheduler_t<T>&)
            const noexcept {
            return threadpool_scheduler{*m_threadpool};
        }

        private:
        threadpool* m_threadpool;  /// non-owning pointer to the threadpool
    };

    template <beman::execution::receiver Receiver>
    class operation {
        public:
        using operation_state_concept = beman::execution::operation_state_t;

        operation(Receiver&& receiver, threadpool* pool) noexcept
            : m_receiver(std::forward<Receiver>(receiver)),
              m_threadpool(pool) {}

        void start() & noexcept {
            m_threadpool->enqueue([this]() {
                beman::execution::set_value(std::move(m_receiver));
            });
        }

        private:
        std::remove_cvref_t<Receiver> m_receiver;
        threadpool* m_threadpool;
    };

    class sender {
        public:
        using sender_concept = beman::execution::sender_t;
        using completion_signatures = beman::execution::completion_signatures<
            beman::execution::set_value_t()>;

        sender(threadpool* pool) noexcept;
        env get_env() const noexcept;

        template <beman::execution::receiver Receiver>
        auto connect(Receiver&& receiver) {
            return operation<Receiver>(std::forward<Receiver>(receiver),
                                       m_threadpool);
        }

        private:
        threadpool* m_threadpool;  /// non-owning pointer to the threadpool
    };

    sender schedule() const noexcept;
    bool operator==(const threadpool_scheduler& other) const = default;

    private:
    threadpool* m_threadpool = nullptr;
};

}  // namespace traccc
