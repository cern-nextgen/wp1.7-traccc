#pragma once

// TBB include(s).
#include <tbb/task_arena.h>

// Stdexec include(s).
#include <stdexec/execution.hpp>

namespace traccc {

/// Wrapper around a TBB task arena to be used as a scheduler for stdexec.
class task_arena_scheduler {
    public:
    using scheduler_concept = stdexec::scheduler_t;

    /// Construct a task_arena_scheduler that uses the given TBB task arena.
    /// @param task_arena The TBB task arena to use for scheduling.
    ///
    /// @note The task_arena_scheduler does not take ownership of the task
    /// arena, the task arena should remain valid for the lifetime of the
    /// scheduler.
    ///
    task_arena_scheduler(tbb::task_arena& task_arena);

    class env {
        public:
        env(tbb::task_arena* arena) noexcept;

        template <typename T>
        auto query(
            const stdexec::get_completion_scheduler_t<T>&) const noexcept {
            return task_arena_scheduler{*m_arena};
        }

        private:
        tbb::task_arena* m_arena;  /// non-owning pointer to the task arena
    };

    template <stdexec::receiver Receiver>
    class operation {
        public:
        using operation_state_concept = stdexec::operation_state_t;

        operation(Receiver&& receiver, tbb::task_arena* arena) noexcept
            : m_receiver(std::forward<Receiver>(receiver)), m_arena(arena) {}

        void start() & noexcept {
            m_arena->enqueue(
                [this]() { stdexec::set_value(std::move(m_receiver)); });
        }

        private:
        std::remove_cvref_t<Receiver> m_receiver;
        tbb::task_arena* m_arena;
    };

    class sender {
        public:
        using sender_concept = stdexec::sender_t;
        using completion_signatures =
            stdexec::completion_signatures<stdexec::set_value_t()>;

        sender(tbb::task_arena* arena) noexcept;
        env get_env() const noexcept;

        template <stdexec::receiver Receiver>
        auto connect(Receiver&& receiver) {
            return operation<Receiver>(std::forward<Receiver>(receiver),
                                       m_arena);
        }

        private:
        tbb::task_arena* m_arena;
    };

    sender schedule() const noexcept;
    bool operator==(const task_arena_scheduler& other) const = default;

    private:
    tbb::task_arena* m_arena = nullptr;
};

}  // namespace traccc
