#pragma once

namespace traccc {
/// Enumeration of await strategies for synchronous or suspending operations

enum class await_strategy {
    sync,    ///< Synchronous waiting
    suspend  ///< Suspend execution until the operation is complete
};

}  // namespace traccc
