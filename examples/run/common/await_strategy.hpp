#pragma once

namespace traccc {
/// Enumeration of await strategies for synchronous or suspending operations

enum class await_strategy {
    sync_event,        ///< Synchronous waiting on an event
    sync_stream,       ///< Synchronous waiting on a stream
    callback,          ///< Suspending on a stream with a callback
    poll,              ///< Suspending with polling on an event
    defer_sync_event,  ///< Suspending and deferring event synchronization to a
                       ///< service threadpool
    defer_sync_stream  ///< Suspending and deferring stream synchronization to a
                       ///< service
};

}  // namespace traccc
