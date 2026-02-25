#pragma once

// TBB include(s).
#include <tbb/task_arena.h>

// System include(s).
#include <coroutine>

namespace traccc {

/// Minimalistic scheduler for "alien" coroutines that executes the tasks on the
/// current thread.

struct inline_scheduler {
    public:
    void operator()(std::coroutine_handle<> handle) { handle.resume(); }
};
}  // namespace traccc
