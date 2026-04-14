/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/options/details/interface.hpp"

// System include(s).
#include <cstddef>
#include <ostream>

namespace traccc::opts {

/// Option(s) for multi-threaded code execution
class threading : public interface {

    public:
    /// @name Options
    /// @{

    enum class await_strategy {
        sync_event,        ///< Synchronous waiting on an event
        sync_stream,       ///< Synchronous waiting on a stream
        callback,          ///< Suspending on a stream with a callback
        poll,              ///< Suspending with polling on an event
        defer_sync_event,  ///< Suspending and deferring event synchronization
                           ///< to a service threadpool
        defer_sync_stream  ///< Suspending and deferring stream synchronization
                           ///< to a service threadpool
    };

    enum class service_threads_strategy {
        spin,   ///< Service threads will spin while waiting for work
        yield,  ///< Service threads will yield while waiting for work
        block   ///< Service threads will block while waiting for work
    };

    service_threads_strategy service_threads_mode =
        service_threads_strategy::spin;

    await_strategy await_mode = await_strategy::sync_event;

    /// The number of threads to use for the data processing
    std::size_t threads = 1;

    /// The number of events that can  be processed concurrently
    std::size_t concurrent_slots = 1;

    /// The number of threads to use for service tasks (e.g. event polling)
    std::size_t service_threads = 0;

    /// @}

    /// Constructor
    threading();

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm) override;

    std::unique_ptr<configuration_printable> as_printable() const override;
};  // struct threading

std::ostream& operator<<(std::ostream& os,
                         const threading::await_strategy& opts);
std::ostream& operator<<(std::ostream& os,
                         const threading::service_threads_strategy& opts);

}  // namespace traccc::opts
