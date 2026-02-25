/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "traccc/bfield/magnetic_field.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/seed_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/seeding/detail/track_params_estimation_config.hpp"

namespace traccc::device {

/// Payload for the
/// @c seed_parameter_estimation_algorithm::estimate_seed_params_kernel function
struct estimate_seed_params_kernel_payload {
    /// The number of seeds
    edm::seed_collection::const_view::size_type n_seeds;
    /// The track parameter estimation configuration
    const track_params_estimation_config& config;
    /// The magnetic field object
    const magnetic_field& bfield;
    /// All measurements of the event
    const edm::measurement_collection<default_algebra>::const_view&
        measurements;
    /// All spacepoints of the event
    const edm::spacepoint_collection::const_view& spacepoints;
    /// The reconstructed track seeds of the event
    const edm::seed_collection::const_view& seeds;
    /// The output buffer for the bound track parameters
    bound_track_parameters_collection_types::view& params;
};

}  // namespace traccc::device
