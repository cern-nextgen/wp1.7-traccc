/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/options/details/interface.hpp"

namespace traccc::opts {

/// Option(s) for device configuration
class device : public interface {

    public:
    /// @name Options
    /// @{

    enum class event_sync_strategy {
        spin,  ///< Calling thread spins while synchronizing events
        block  ///< Calling thread blocks while synchronizing events
    };

    event_sync_strategy event_sync_mode = event_sync_strategy::spin;

    enum device_sync_strategy {
        automatic,  ///< Use heuristic to choose between yield and spin
        spin,       ///< Calling thread spins while waiting for the device
        yield,      ///< Calling thread yields while waiting for the device
        block       ///< Calling thread blocks while waiting for the device
    };

    device_sync_strategy device_sync_mode = device_sync_strategy::automatic;

    /// @}

    /// Constructor
    device();

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm) override;

    std::unique_ptr<configuration_printable> as_printable() const override;
};  // struct device

}  // namespace traccc::opts
