/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/options/threading.hpp"

#include "traccc/examples/utils/printable.hpp"

// System include(s).
#include <ostream>
#include <stdexcept>

namespace traccc::opts {

/// Type alias for concurrent slots option
using concurrent_slots_type = std::size_t;
/// Name of the concurrent slots option
static const char* concurrent_slots_option = "concurrent-slots";

/// Type alias for the await strategy enumeration
using await_strategy_type = std::string;
/// Name of the await strategy option
static const char* await_strategy_option = "await-strategy";

/// Type alias for the service thread strategy enumeration
using service_threads_strategy_type = std::string;
/// Name of the service thread strategy option
static const char* service_threads_strategy_option = "service-threads-strategy";

threading::threading() : interface("Multi-Threading Options") {

    m_desc.add_options()(
        "cpu-threads",
        boost::program_options::value(&threads)->default_value(threads),
        "The number of CPU threads to use")(
        concurrent_slots_option,
        boost::program_options::value<concurrent_slots_type>(),
        "The number of events that can be "
        "processed concurrently, be default equal to cpu-threads")(
        await_strategy_option,
        boost::program_options::value<std::string>()->default_value(
            "sync-event"),
        "The await strategy to use (\"sync-event\", \"sync-stream\", "
        "\"callback\", \"poll\", \"defer-sync-event\", \"defer-sync-stream\")")(
        "service-threads",
        boost::program_options::value(&service_threads)
            ->default_value(service_threads),
        "The number of threads to use for service tasks (e.g. event polling)")(
        service_threads_strategy_option,
        boost::program_options::value<std::string>()->default_value("spin"),
        "The strategy to use for service threads (\"spin\", \"yield\", or "
        "\"block\")");
}

void threading::read(const boost::program_options::variables_map& vm) {

    if (threads == 0) {
        throw std::invalid_argument{"Must use threads>0"};
    }
    if (!vm.count(concurrent_slots_option)) {
        concurrent_slots = threads;
    } else {
        concurrent_slots =
            vm[concurrent_slots_option].as<concurrent_slots_type>();
        if (concurrent_slots == 0) {
            throw std::invalid_argument{"Must use concurrent-slots>0"};
        }
    }
    if (vm.count(await_strategy_option)) {
        const std::string await_string =
            vm[await_strategy_option].as<await_strategy_type>();
        if (await_string == "sync-event") {
            await_mode = await_strategy::sync_event;
        } else if (await_string == "sync-stream") {
            await_mode = await_strategy::sync_stream;
        } else if (await_string == "callback") {
            await_mode = await_strategy::callback;
        } else if (await_string == "poll") {
            await_mode = await_strategy::poll;
        } else if (await_string == "defer-sync-event") {
            await_mode = await_strategy::defer_sync_event;
        } else if (await_string == "defer-sync-stream") {
            await_mode = await_strategy::defer_sync_stream;
        } else {
            throw std::invalid_argument{"Unknown await strategy: " +
                                        await_string};
        }
    }
    if (vm.count(service_threads_strategy_option)) {
        const std::string service_threads_strategy_string =
            vm[service_threads_strategy_option]
                .as<service_threads_strategy_type>();
        if (service_threads_strategy_string == "spin") {
            service_threads_mode = service_threads_strategy::spin;
        } else if (service_threads_strategy_string == "yield") {
            service_threads_mode = service_threads_strategy::yield;
        } else if (service_threads_strategy_string == "block") {
            service_threads_mode = service_threads_strategy::block;
        } else {
            throw std::invalid_argument{"Unknown service thread strategy: " +
                                        service_threads_strategy_string};
        }
    }
}

std::unique_ptr<configuration_printable> threading::as_printable() const {
    auto cat = std::make_unique<configuration_category>(m_description);

    std::string await_string;
    switch (await_mode) {
        case await_strategy::sync_event:
            await_string = "synchronous (event)";
            break;
        case await_strategy::sync_stream:
            await_string = "synchronous (stream)";
            break;
        case await_strategy::callback:
            await_string = "suspending (stream callback)";
            break;
        case await_strategy::poll:
            await_string = "suspending (event polling)";
            break;
        case await_strategy::defer_sync_event:
            await_string = "suspending (deferred event synchronization)";
            break;
        case await_strategy::defer_sync_stream:
            await_string = "suspending (deferred stream synchronization)";
            break;
        default:
            await_string = "unknown";
            break;
    }

    std::string service_threads_strategy_string;
    switch (service_threads_mode) {
        case service_threads_strategy::spin:
            service_threads_strategy_string = "spin";
            break;
        case service_threads_strategy::yield:
            service_threads_strategy_string = "yield";
            break;
        case service_threads_strategy::block:
            service_threads_strategy_string = "block";
            break;
        default:
            service_threads_strategy_string = "unknown";
            break;
    }

    cat->add_child(std::make_unique<configuration_kv_pair>("Await strategy",
                                                           await_string));
    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Number of CPU thread", std::to_string(threads)));
    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Number of concurrent slots", std::to_string(concurrent_slots)));
    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Number of service threads", std::to_string(service_threads)));
    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Service threads policy", service_threads_strategy_string));

    return cat;
}

std::ostream& operator<<(std::ostream& os,
                         const threading::await_strategy& opts) {
    switch (opts) {
        case threading::await_strategy::sync_event:
            return os << "sync-event";
        case threading::await_strategy::sync_stream:
            return os << "sync-stream";
        case threading::await_strategy::callback:
            return os << "callback";
        case threading::await_strategy::poll:
            return os << "poll";
        case threading::await_strategy::defer_sync_event:
            return os << "defer-sync-event";
        case threading::await_strategy::defer_sync_stream:
            return os << "defer-sync-stream";
        default:
            return os << "unknown";
    }
}

std::ostream& operator<<(std::ostream& os,
                         const threading::service_threads_strategy& opts) {
    switch (opts) {
        case threading::service_threads_strategy::spin:
            return os << "spin";
        case threading::service_threads_strategy::yield:
            return os << "yield";
        case threading::service_threads_strategy::block:
            return os << "block";
        default:
            return os << "unknown";
    }
}

}  // namespace traccc::opts
