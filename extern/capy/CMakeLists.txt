# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022-2026 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

cmake_minimum_required(VERSION 3.25)
include(FetchContent)

# Tell the user what's happening.
message(STATUS "Building Boost.Capy as part of the traccc project")

# Declare where to get stdexec from.
set(TRACCC_CAPY_SOURCE
    "URL;https://github.com/cppalliance/capy/archive/df92ffbb0670c06ebf0f201da73f06d1385d5d9b.zip;URL_MD5;d6b6e1e25684986d17ae92c13f992cfc"
    CACHE STRING "Source for Boost.Capy, when built as part of this project")

mark_as_advanced(TRACCC_CAPY_SOURCE)
FetchContent_Declare(capy SYSTEM ${TRACCC_CAPY_SOURCE})

# Turn off build tests, examples, and docs.
set(BOOST_CAPY_BUILD_TESTS FALSE)
set(BOOST_CAPY_BUILD_EXAMPLES FALSE)
set(BOOST_CAPY_BUILD_BENCH FALSE)
set(BOOST_CAPY_MRDOCS_BUILD FALSE)

# Get it into the current directory.
FetchContent_MakeAvailable(capy)
