// Local include(s).
#include "traccc/examples/utils/threadpool.hpp"

// System include(s).
#include <stdexcept>

namespace traccc {

threadpool::threadpool(size_t num_threads, wait_policy policy)
    : m_policy(policy) {
    if (num_threads == 0) {
        throw std::invalid_argument("Number of threads must be greater than 0");
    }
    m_workers.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        switch (policy) {
            case wait_policy::spin:
                m_workers.emplace_back(
                    [this](std::stop_token token) { run_spinning(token); });
                break;
            case wait_policy::yield:
                m_workers.emplace_back(
                    [this](std::stop_token token) { run_yielding(token); });
                break;
            case wait_policy::block:
                m_workers.emplace_back(
                    [this](std::stop_token token) { run_blocking(token); });
                break;
            default:
                throw std::invalid_argument("Invalid wait policy");
        }
    }
}

threadpool::~threadpool() {
    // Special handling of blocking policy to ensure that threads are not stuck
    // waiting on an empty queue
    if (m_policy == wait_policy::block) {
        for (auto& worker : m_workers) {
            worker.request_stop();
        }
        // Push empty tasks to wake up blocked threads
        for (size_t i = 0; i < m_workers.size(); ++i) {
            m_queue.push([]() {});
        }
    }
}

void threadpool::run_spinning(std::stop_token token) {
    while (!token.stop_requested()) {
        auto task = std::function<void()>{};
        if (m_queue.try_pop(task)) {
            task();
        }
    }
}

void threadpool::run_yielding(std::stop_token token) {
    while (!token.stop_requested()) {
        auto task = std::function<void()>{};
        if (m_queue.try_pop(task)) {
            task();
        } else {
            std::this_thread::yield();
        }
    }
}

void threadpool::run_blocking(std::stop_token token) {
    while (!token.stop_requested()) {
        auto task = std::function<void()>{};
        m_queue.pop(task);
        task();
    }
}

std::ostream& operator<<(std::ostream& os, threadpool::wait_policy policy) {
    switch (policy) {
        case threadpool::wait_policy::spin:
            os << "spin";
            break;
        case threadpool::wait_policy::yield:
            os << "yield";
            break;
        case threadpool::wait_policy::block:
            os << "block";
            break;
        default:
            os << "unknown";
            break;
    }
    return os;
}

}  // namespace traccc
