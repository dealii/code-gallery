CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

set (CMAKE_CXX_STANDARD 14)

# Set the name of the project and target:
SET(TARGET "parallel_in_time")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")


set(PROJECT_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
set(SRC_PATH "${PROJECT_PATH}/src")
set(TEST_SRC_PATH "${PROJECT_PATH}/test")
# set(BIN_PATH  "${PROJECT_BINARY_DIR}/bin")
set(LIB_NAME "${TARGET}_lib")
set(BIN_NAME "${TARGET}")
set(TEST_PATH "${PROJECT_BINARY_DIR}/test")
set(TEST_NAME "test_${TARGET}")

SET(LIB_SRC
  ${SRC_PATH}/BraidFuncs.cc
  ${SRC_PATH}/Utilities.cc
  )

SET(TARGET_SRC
  ${SRC_PATH}/${TARGET}.cc
  )

SET(TEST_SRC
  ${TEST_SRC_PATH}/test_braid.cc
  )

OPTION(DO_MFG "Do the manufactured solution run" OFF)

# Tell CMake to find MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
add_definitions(-DUSE_MPI)

if(DO_MFG)
  # If we are doing the manufactured solution, tell
  # the C++ processor so it can enable MFG solutions
  add_definitions(-DDO_MFG)
endif(DO_MFG)

FIND_PACKAGE(deal.II 9.0.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

# Find braid details
if (NOT "$ENV{BRAID_DIR}" STREQUAL "")
    set(BRAID_DIR "$ENV{BRAID_DIR}" CACHE INTERNAL "Copied BRAID_DIR from environment variable")
endif()
# Include the braid paths and libraries
include_directories(${BRAID_DIR})

# Build the library
ADD_LIBRARY(${LIB_NAME} SHARED ${LIB_SRC})
DEAL_II_SETUP_TARGET(${LIB_NAME})

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()
target_link_libraries(${TARGET} ${BRAID_DIR}libbraid.a) # link braid
target_link_libraries(${TARGET} ${LIB_NAME})
set_target_properties(${TARGET} PROPERTIES
  LINK_FLAGS "${MPI_LINK_FLAGS}") # Link MPI
set_target_properties(${MAIN_NAME} PROPERTIES
  COMPILE_FLAGS "${MPI_COMPILE_FLAGS}") # Use MPI compile flags

# Lastly build the tests
enable_testing()
add_subdirectory(test)



