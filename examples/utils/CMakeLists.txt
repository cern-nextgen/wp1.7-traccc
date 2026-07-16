# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

traccc_add_library( traccc_utils utils TYPE SHARED
  "src/printable.cpp"
  "src/threadpool.cpp"
)

target_link_libraries(traccc_utils PUBLIC TBB::tbb beman::task)
