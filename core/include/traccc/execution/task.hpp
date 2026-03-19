#pragma once

// Boost.Capy include(s).
#include <boost/capy.hpp>

namespace traccc {

template <typename T>
using task = boost::capy::task<T>;
}
