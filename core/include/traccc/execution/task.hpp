#pragma once

// Stdexec include(s).
#include <exec/task.hpp>

namespace traccc {

template <typename T>
using task = exec::task<T>;
}
