##
#  CMake script for the CeresFE program:
##

CMAKE_MINIMUM_REQUIRED(VERSION 3.0.2)

FIND_PACKAGE(deal.II 9.0 REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT("ceres")

#
# Find libconfig:
#

FIND_PATH(LIBCONFIG_INCLUDE_DIR libconfig.h++
  HINTS ${LIBCONFIG_DIR} $ENV{LIBCONFIG_DIR}
  PATH_SUFFIXES include
  )

FIND_LIBRARY(LIBCONFIG_LIBRARY NAMES config++
  HINTS ${LIBCONFIG_DIR} $ENV{LIBCONFIG_DIR}
  PATH_SUFFIXES lib${LIB_SUFFIX} lib64 lib
  )

#
# Find armadillo
#

FIND_PATH(ARMADILLO_INCLUDE_DIR armadillo
  HINTS ${ARMADILLO_DIR} $ENV{ARMADILLO_DIR}
  PATH_SUFFIXES include
  )

FIND_LIBRARY(ARMADILLO_LIBRARY NAMES armadillo
  HINTS ${ARMADILLO_DIR} $ENV{ARMADILLO_DIR}
  PATH_SUFFIXES lib${LIB_SUFFIX} lib64 lib
  )

#
# Are all dependencies fullfilled?
#

IF(NOT DEAL_II_WITH_UMFPACK)
  MESSAGE(FATAL_ERROR "
Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_UMFPACK = ON
One or all of these are OFF in your installation but are required for this tutorial step."
    )
ENDIF()


IF(${LIBCONFIG_INCLUDE_DIR} MATCHES "-NOTFOUND" OR ${LIBCONFIG_LIBRARY} MATCHES "-NOTFOUND")
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate libconfig ***\n\n"
    "This code gallery program requires libconfig, "
    "http://www.hyperrealm.com/libconfig/libconfig.html.\n"
    "Please install it either by hand, or with your favorite package manager"
    )
ENDIF()


IF(${ARMADILLO_INCLUDE_DIR} MATCHES "-NOTFOUND" OR ${ARMADILLO_LIBRARY} MATCHES "-NOTFOUND")
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate armadillo ***\n\n"
    "This code gallery program requires armadillo, "
    "http://arma.sourceforge.net.\n"
    "Please install it either by hand, or with your favorite package manager"
    )
ENDIF()

#
# Set up program:
#

SET(TARGET "ceres")
SET(TARGET_SRC src/${TARGET}.cc)
DEAL_II_INVOKE_AUTOPILOT()

TARGET_INCLUDE_DIRECTORIES(${TARGET}
  PUBLIC ${LIBCONFIG_INCLUDE_DIR} ${ARMADILLO_INCLUDE_DIR}
  )
TARGET_LINK_LIBRARIES(${TARGET}
  ${LIBCONFIG_LIBRARY} ${ARMADILLO_LIBRARY}
  )

