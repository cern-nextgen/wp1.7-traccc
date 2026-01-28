#pragma once

namespace traccc {
/// Enumeration of await strategies for synchronous or suspending operations

enum class await_strategy {
    sync_event,           ///< Synchronous waiting on an event
    sync_stream,          ///< Synchronous waiting on a stream
    tbb_callback  ///< Suspend TBB task, use callback
};

}  // namespace traccc
