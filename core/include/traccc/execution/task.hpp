#pragma once

#include <concepts>
#include <coroutine>
#include <exception>
#include <functional>
#include <optional>

namespace traccc {

namespace concepts {
template <typename T>
concept HasScheduler = requires(T t) {
    {
        t.get_scheduler()
    } -> std::convertible_to<std::function<void(std::coroutine_handle<>)>>;
};
}  // namespace concepts

// Nestable coroutine, can be co_awaited by other coroutines.
// Returns a single value of templated type via co_return if it isn't void
// Doesn't co_yield.
template <typename ResultType>
class [[nodiscard]] task {

    static_assert(std::movable<ResultType> || std::same_as<ResultType, void>,
                  "task<ResultType> requires ResultType to be movable or void");

    public:
    using result_type = ResultType;

    struct promise_type;  // typedef required by coroutines
    using handle_type =
        std::coroutine_handle<promise_type>;  // not required but useful

    // Constructor from coroutine handle
    task(handle_type coroutine_handle) : m_coroutine(coroutine_handle) {}
    ~task() {
        if (m_coroutine) {
            m_coroutine.destroy();
        }
    }
    task() = default;
    task(const task&) = delete;
    task& operator=(const task&) = delete;
    task(task&& other) noexcept : m_coroutine{other.m_coroutine} {
        other.m_coroutine = {};
    }
    task& operator=(task&& other) noexcept {
        if (this != &other) {
            if (m_coroutine) {
                m_coroutine.destroy();
            }
            m_coroutine = other.m_coroutine;
            other.m_coroutine = {};
        }
        return *this;
    }

    // Awaitable interface: always suspend to allow async execution
    bool await_ready() const noexcept { return false; }
    // Awaitable interface: setup parent/scheduler relationship and transfer
    // control to this coroutine
    template <concepts::HasScheduler T>
    inline handle_type await_suspend(std::coroutine_handle<T> handle) noexcept;
    // Awaitable interface: return result or rethrow exception on resume
    result_type await_resume() const;

    private:
    handle_type m_coroutine = nullptr;
};

namespace detail::task {
// Helper for handling co_return in promise_type, default implementation for
// non-void ResultType
template <typename ResultType>
struct ReturnHelper {
    // Storage for the co_return result value
    std::optional<ResultType> m_value;

    // Required by coroutines, mutually exclusive with return_void
    // Store the co_return value
    template <typename T>
        requires std::constructible_from<ResultType, T&&>
    void return_value(T&& value) {
        m_value.emplace(std::forward<T>(value));
    }
    // Overload to resolve ambiguity
    void return_value(ResultType value) { m_value.emplace(std::move(value)); }
};

// Specialization for void return type
template <>
struct ReturnHelper<void> {
    // Required by coroutines, mutually exclusive with return_value
    // Handle co_return without value
    void return_void() {}
};

}  // namespace detail::task

template <typename ResultType>
struct task<ResultType>::promise_type
    : public detail::task::ReturnHelper<
          typename task<ResultType>::result_type> {
    // Storage for exceptions thrown in the coroutine body
    std::exception_ptr m_exception;
    // Handle to the parent coroutine that co_awaited this task
    std::coroutine_handle<> m_parent;
    // Handle to scheduler to resume this coroutine and propagate to children
    std::function<void(std::coroutine_handle<>)> m_scheduler;

    // Accessor for scheduler used by child coroutines
    const auto& get_scheduler() const { return m_scheduler; }
    // Schedule resumption of this task
    void reschedule() { m_scheduler(handle_type::from_promise(*this)); }

    // Required by coroutines: create the object
    task get_return_object() { return {handle_type::from_promise(*this)}; }
    // Required by coroutines: suspend immediately on start (lazy execution)
    std::suspend_always initial_suspend() const { return {}; }
    // Required by coroutines: handle completion and resume parent
    auto final_suspend() const noexcept {
        struct final_awaiter {
            // Don't skip final suspension
            bool await_ready() const noexcept { return false; }
            // Resume parent coroutine with symmetric transfer or return to
            // caller
            std::coroutine_handle<> await_suspend(handle_type handle) noexcept {
                auto parent = handle.promise().m_parent;
                if (parent) {
                    return parent;
                }
                return std::noop_coroutine();
            }
            // No action needed on resume
            void await_resume() const noexcept {}
        };
        return final_awaiter{};
    }
    // Required by coroutines: capture exceptions for later rethrowing
    void unhandled_exception() { m_exception = std::current_exception(); }
};

template <typename ResultType>
template <concepts::HasScheduler T>
inline task<ResultType>::handle_type task<ResultType>::await_suspend(
    std::coroutine_handle<T> handle) noexcept {
    m_coroutine.promise().m_parent = handle;
    m_coroutine.promise().m_scheduler = handle.promise().get_scheduler();
    return m_coroutine;
}

template <typename ResultType>
inline typename task<ResultType>::result_type task<ResultType>::await_resume()
    const {
    if (m_coroutine.promise().m_exception) {
        std::rethrow_exception(m_coroutine.promise().m_exception);
    }
    if constexpr (std::same_as<result_type, void>) {
        return;
    } else {
        return std::move(m_coroutine.promise().m_value).value();
    }
}
}  // namespace traccc
