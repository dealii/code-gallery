cmake_minimum_required(VERSION 3.17)

# deal.II >= 9.5
find_package(deal.II 9.8.0 REQUIRED HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

set(TARGET "topology_optimization")
deal_ii_initialize_cached_variables()

file(GLOB_RECURSE TARGET_SRC  "src/*.cpp")
file(GLOB_RECURSE TARGET_INC  "include/*.h")

set(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC})
project(${TARGET})
deal_ii_invoke_autopilot()