#pragma once

#include <tbb/task_arena.h>

#include <boost/capy.hpp>
#include <coroutine>

namespace traccc {
class task_arena_executor {

    public:
    class task_arena_context : public boost::capy::execution_context {
        public:
        task_arena_context(tbb::task_arena& arena) : m_arena(&arena) {}

        void schedule(std::coroutine_handle<> h) const {
            m_arena->enqueue([h]() { h.resume(); });
        }
        bool operator==(const task_arena_context& other) const noexcept {
            return m_arena == other.m_arena;
        }

        private:
        tbb::task_arena* m_arena;
    };

    task_arena_executor(task_arena_context& context) noexcept
        : m_context(&context) {
        static_assert(boost::capy::Executor<task_arena_executor>,
                      "task_arena_executor should be a valid capy Executor");
    }

    task_arena_executor(task_arena_executor const&) noexcept = default;

    std::coroutine_handle<> dispatch(std::coroutine_handle<> h) const {
        m_context->schedule(h);
        return std::noop_coroutine();
    }
    void post(std::coroutine_handle<> h) const { m_context->schedule(h); }
    task_arena_context& context() const noexcept { return *m_context; }
    void on_work_started() const noexcept {}
    void on_work_finished() const noexcept {}
    bool operator==(const task_arena_executor& other) const noexcept {
        return m_context == other.m_context;
    }

    private:
    task_arena_context* m_context;
};

}  // namespace traccc
