#pragma once

// TBB include(s).
#include <tbb/task_arena.h>

// System include(s).
#include <coroutine>
#include <iostream>

namespace traccc {

/// Wrapper around a TBB task arena to be used as a scheduler for "alien"
/// coroutine.

class task_arena_scheduler {
    public:
    task_arena_scheduler(tbb::task_arena& arena) : m_arena(&arena) {}

    void operator()(std::coroutine_handle<> handle) {
        m_arena->enqueue([handle]() { handle.resume(); });
    }

    private:
    tbb::task_arena* m_arena;
};
}  // namespace traccc
