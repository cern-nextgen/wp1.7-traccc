#pragma once

namespace traccc {
/// Enumeration of await strategies for synchronous or suspending operations

enum class await_strategy {
    sync_event,    ///< Synchronous waiting on an event
    sync_stream,   ///< Synchronous waiting on a stream
    tbb_callback,  ///< Suspend TBB task, use callback
    tbb_poll,  ///< Suspend TBB task, poll on an event in a service threadpool
    tbb_defer_sync_event,  ///< Suspend TBB task, defer event synchronization to
                           ///< service threadpool
    tbb_defer_sync_stream,  ///< Suspend TBB task, defer stream synchronization
                            ///< to service threadpool
    boost_fiber_callback,   ///< Suspend Boost.Fiber, use callback
    boost_fiber_poll,  ///< Suspend Boost.Fiber, poll on an event in a service
                       ///< threadpool
    boost_fiber_defer_sync_event,  ///< Suspend Boost.Fiber, defer event
                                   ///< synchronization to service threadpool
    boost_fiber_defer_sync_stream  ///< Suspend Boost.Fiber, defer stream
                                   ///< synchronization to service threadpool
};

}  // namespace traccc
