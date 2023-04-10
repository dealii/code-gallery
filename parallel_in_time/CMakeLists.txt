CMAKE_MINIMUM_REQUIRED(VERSION 3.3.0)

# Set the name of the project and target:
SET(TARGET "parallel_in_time")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")

SET(PROJECT_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
SET(SRC_PATH "${PROJECT_PATH}/src")
SET(TEST_SRC_PATH "${PROJECT_PATH}/test")
# SET(BIN_PATH  "${PROJECT_BINARY_DIR}/bin")
SET(LIB_NAME "${TARGET}_lib")
SET(BIN_NAME "${TARGET}")
SET(TEST_PATH "${PROJECT_BINARY_DIR}/test")
SET(TEST_NAME "test_${TARGET}")

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

IF(DO_MFG)
  # If we are doing the manufactured solution, tell
  # the C++ processor so it can enable MFG solutions
  MESSAGE(STATUS "Manufactured solution enabled")
  ADD_DEFINITIONS(-DDO_MFG)
ENDIF(DO_MFG)

FIND_PACKAGE(deal.II 9.4.0 QUIET
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
IF(NOT "$ENV{BRAID_DIR}" STREQUAL "")
    SET(BRAID_DIR "$ENV{BRAID_DIR}" CACHE INTERNAL "Copied BRAID_DIR from environment variable")
ENDIF()

IF("${CMAKE_FIND_LIBRARY_PREFIXES}" STREQUAL "")
    SET(CMAKE_FIND_LIBRARY_PREFIXES "lib")
ENDIF()

FIND_PATH(BRAID_INCLUDE_DIR
  NAMES braid.h
  HINTS ${BRAID_DIR}
  PATH_SUFFIXES include
  )

FIND_LIBRARY(BRAID_LIBRARY
  NAMES braid.a
  HINTS ${BRAID_DIR}
  PATH_SUFFIXES lib64 lib
  )

MESSAGE(STATUS "Braid include directory: ${BRAID_INCLUDE_DIR}")
MESSAGE(STATUS "Braid library:           ${BRAID_LIBRARY}")

# Include the braid paths and libraries
INCLUDE_DIRECTORIES(${BRAID_INCLUDE_DIR})

# Build the library
ADD_LIBRARY(${LIB_NAME} SHARED ${LIB_SRC})
DEAL_II_SETUP_TARGET(${LIB_NAME})

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})

# Tell CMake to find MPI
FIND_PACKAGE(MPI REQUIRED)
ADD_DEFINITIONS(-DUSE_MPI)
INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})

DEAL_II_INVOKE_AUTOPILOT()
TARGET_LINK_LIBRARIES(${TARGET} ${BRAID_LIBRARY}) # link braid
TARGET_LINK_LIBRARIES(${TARGET} ${LIB_NAME})
SET_TARGET_PROPERTIES(${TARGET} PROPERTIES
  LINK_FLAGS "${MPI_LINK_FLAGS}") # Link MPI
SET_TARGET_PROPERTIES(${MAIN_NAME} PROPERTIES
  COMPILE_FLAGS "${MPI_COMPILE_FLAGS}") # Use MPI compile flags

# Lastly build the tests
ENABLE_TESTING()
ADD_SUBDIRECTORY(test)



