/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/options/device.hpp"

namespace traccc::cuda {
struct device_config {
    static void apply(
        const traccc::opts::device::device_sync_strategy strategy);
};
}  // namespace traccc::cuda
