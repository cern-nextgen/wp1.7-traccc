/**
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/examples/utils/threadpool.hpp"

// GTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <atomic>
#include <condition_variable>
#include <mutex>

// Helper for recursive lambdas in C++20
template <typename T>
struct recursion_wrapper {
    T f;
    template <class... Args>
    decltype(auto) operator()(Args&&... args) const {
        return f(*this, std::forward<Args>(args)...);
    }
};

class ThreadpoolTest
    : public ::testing::TestWithParam<traccc::threadpool::wait_policy> {};

TEST_P(ThreadpoolTest, Basics) {

    auto policy = GetParam();

    {
        auto pool = traccc::threadpool(4, policy);

        auto result = std::atomic<int>{0};
        auto task = [&result]() {
            result = 42;
            result.notify_all();
        };
        EXPECT_EQ(result, 0);
        pool.enqueue(task);
        result.wait(0);

        EXPECT_EQ(result, 42);
    }
}

TEST_P(ThreadpoolTest, SelfResubmitTasks) {

    auto policy = GetParam();

    {
        auto pool = traccc::threadpool(4, policy);
        auto counter = 0;
        auto mutex = std::mutex{};
        auto cv = std::condition_variable{};
        auto task = recursion_wrapper([&](auto self) {
            if (counter < 5) {
                std::lock_guard<std::mutex> lock(mutex);
                ++counter;
                pool.enqueue(self);
            } else {
                cv.notify_all();
            }
        });
        EXPECT_EQ(counter, 0);
        pool.enqueue(task);
        {
            std::unique_lock lock(mutex);
            cv.wait(lock, [&counter] { return counter >= 5; });
        }

        EXPECT_EQ(counter, 5);
    }
}

// Instantiate test suite with both policies
INSTANTIATE_TEST_SUITE_P(
    WaitPolicies, ThreadpoolTest,
    ::testing::Values(traccc::threadpool::wait_policy::spin,
                      traccc::threadpool::wait_policy::yield,
                      traccc::threadpool::wait_policy::block));
