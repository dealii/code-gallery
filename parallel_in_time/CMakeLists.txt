CMAKE_MINIMUM_REQUIRED(VERSION 3.13.4)

PROJECT("parallel_in_time")

#
# setup deal.II
#
FIND_PACKAGE(deal.II 9.5.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()
DEAL_II_INITIALIZE_CACHED_VARIABLES()

IF(NOT ${DEAL_II_WITH_MPI})
  MESSAGE(FATAL_ERROR "parallel_in_time requires that deal.II be set up with MPI.")
ENDIF()

#
# setup braid
#
SET(BRAID_DIR "$ENV{BRAID_DIR}" CACHE INTERNAL
  "Copied BRAID_DIR from environment variable")

FIND_PATH(BRAID_INCLUDE_DIR
  NAMES braid.h
  HINTS ${BRAID_DIR}
  )

FIND_LIBRARY(BRAID_LIBRARY
  NAMES libbraid.a
  HINTS ${BRAID_DIR}
  )

MESSAGE(STATUS "Braid include directory: ${BRAID_INCLUDE_DIR}")
MESSAGE(STATUS "Braid library:           ${BRAID_LIBRARY}")

INCLUDE_DIRECTORIES(${BRAID_INCLUDE_DIR})

#
# setup parallel_in_time
#
SET(SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")
SET(LIB_NAME "parallel_in_time_lib")

SET(LIB_SRC
  ${SRC_PATH}/BraidFuncs.cc
  ${SRC_PATH}/Utilities.cc
  )

OPTION(DO_MFG "Do the manufactured solution run" OFF)

IF(DO_MFG)
  # If we are doing the manufactured solution, tell
  # the C++ processor so it can enable MFG solutions
  MESSAGE(STATUS "Manufactured solution enabled")
  ADD_DEFINITIONS(-DDO_MFG)
ENDIF(DO_MFG)

ADD_LIBRARY(${LIB_NAME} SHARED ${LIB_SRC})
DEAL_II_SETUP_TARGET(${LIB_NAME})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${BRAID_LIBRARY})

ADD_EXECUTABLE(parallel_in_time)
TARGET_SOURCES(parallel_in_time PRIVATE src/parallel_in_time.cc)
TARGET_LINK_LIBRARIES(parallel_in_time ${LIB_NAME})
DEAL_II_SETUP_TARGET(parallel_in_time)

#
# setup tests
#
ENABLE_TESTING()
ADD_SUBDIRECTORY(test)
