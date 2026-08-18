// Local include(s).
#include "task_arena_scheduler.hpp"

// TBB include(s).
#include <tbb/task_arena.h>

namespace traccc {

task_arena_scheduler::task_arena_scheduler(tbb::task_arena& task_arena)
    : m_arena(&task_arena) {}

task_arena_scheduler::sender task_arena_scheduler::schedule() const noexcept {
    return sender{m_arena};
}

task_arena_scheduler::env::env(tbb::task_arena* arena) noexcept
    : m_arena(arena) {}

task_arena_scheduler::sender::sender(tbb::task_arena* arena) noexcept
    : m_arena(arena) {}

task_arena_scheduler::env task_arena_scheduler::sender::get_env()
    const noexcept {
    return env{m_arena};
}

static_assert(stdexec::scheduler<task_arena_scheduler>,
              "task_arena_scheduler should model scheduler");

}  // namespace traccc
