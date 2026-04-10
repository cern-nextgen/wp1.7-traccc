/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc {
/// Enumeration of strategies for waiting on vecmem events in the algorithms

enum class event_sync_strategy {
    spin,  ///<  Calling thread spins while synchronizing events
    block  ///< Calling thread blocks while synchronizing events
};

}  // namespace traccc
