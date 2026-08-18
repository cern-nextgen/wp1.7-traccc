#pragma once

// beman.task include(s).
#include <beman/task/task.hpp>

namespace traccc {

template <typename T>
using task = beman::execution::task<T>;
}
