/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/geometry/detector_buffer.hpp"

namespace traccc::device {

/// Payload for the
/// @c
/// traccc::device::silicon_pixel_spacepoint_formation_algorithm::form_spacepoints_kernel
/// function.
struct silicon_pixel_spacepoint_formation_kernel_payload {
    /// The number of measurements in the event
    edm::measurement_collection<default_algebra>::const_view::size_type
        n_measurements;
    /// The detector object
    const detector_buffer& detector;
    /// The input measurements
    const edm::measurement_collection<default_algebra>::const_view&
        measurements;
    /// The output spacepoints
    edm::spacepoint_collection::view& spacepoints;
};

}  // namespace traccc::device
