##
#  CMake script for the CeresFE program:
##

CMAKE_MINIMUM_REQUIRED(VERSION 3.0.2)

FIND_PACKAGE(deal.II 9.0 QUIET
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
PROJECT("mcmc-laplace")

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


#
# Set up program:
#

SET(TARGET "mcmc-laplace")
SET(TARGET_SRC ${TARGET}.cc)
DEAL_II_INVOKE_AUTOPILOT()

