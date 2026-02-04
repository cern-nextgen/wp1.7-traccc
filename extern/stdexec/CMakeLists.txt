# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022-2026 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

cmake_minimum_required( VERSION 3.25 )
include( FetchContent )

# Tell the user what's happening.
message( STATUS "Building stdexec as part of the TRACCC project" )

# Declare where to get stdexec from.
set( TRACCC_STDEXEC_SOURCE
    "URL;https://github.com/NVIDIA/stdexec/archive/refs/tags/nvhpc-25.09.tar.gz;URL_MD5;e385824bb560a3ac36ce859a2e3ded1f"
    CACHE STRING "Source for stdexec, when built as part of this project" )

mark_as_advanced( TRACCC_STDEXEC_SOURCE )
FetchContent_Declare( stdexec SYSTEM ${TRACCC_STDEXEC_SOURCE} )

# Turn off build tests, examples, and docs.
set( BUILD_TESTING FALSE )
set( STDEXEC_BUILD_EXAMPLES FALSE )
set( STDEXEC_BUILD_DOCS FALSE )
set( STDEXEC_ENABLE_TBB FALSE )
set( STDEXEC_ENABLE_TASKFLOW FALSE )
set( STDEXEC_ENABLE_CUDA FALSE ) # This option adds nvexec target with nvidia specific extensions and wrappers.
                                 # Some parts of the stdexec still can't be compiled with nvcc.

# Get it into the current directory.
FetchContent_MakeAvailable( stdexec )
