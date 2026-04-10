/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/options/device.hpp"

#include "traccc/examples/utils/printable.hpp"

// System include(s).
#include <stdexcept>

namespace traccc::opts {

/// Type alias for the event sync strategy enumeration
using event_sync_strategy_type = std::string;
/// Name of the sync strategy option
static const char* event_sync_strategy_option = "event-sync";

device::device() : interface("Device Options") {

    m_desc.add_options()(
        event_sync_strategy_option,
        boost::program_options::value<std::string>()->default_value("spin"),
        "The event synchronization strategy to use (\"spin\" or \"block\")");
}

void device::read(const boost::program_options::variables_map& vm) {

    if (vm.count(event_sync_strategy_option)) {
        const std::string event_sync_string =
            vm[event_sync_strategy_option].as<event_sync_strategy_type>();
        if (event_sync_string == "spin") {
            event_sync_mode = event_sync_strategy::spin;
        } else if (event_sync_string == "block") {
            event_sync_mode = event_sync_strategy::block;
        } else {
            throw std::invalid_argument{
                "Unknown event synchronization strategy: " + event_sync_string};
        }
    }
}

std::unique_ptr<configuration_printable> device::as_printable() const {
    auto cat = std::make_unique<configuration_category>(m_description);

    std::string event_sync_string;
    switch (event_sync_mode) {
        case event_sync_strategy::spin:
            event_sync_string = "spin";
            break;
        case event_sync_strategy::block:
            event_sync_string = "block";
            break;
        default:
            event_sync_string = "unknown";
            break;
    }

    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Event synchronization strategy", event_sync_string));
    return cat;
}

}  // namespace traccc::opts
