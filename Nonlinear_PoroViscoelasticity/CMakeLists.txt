##
#  CMake script for the nonlinear-poro-viscoelasticity problem
##

# Set the name of the project and target:
SET(TARGET "nonlinear-poro-viscoelasticity")

# Declare all source files the target consists of:
SET(TARGET_SRC
  ${TARGET}.cc
  )

#Search for required deal.II version
CMAKE_MINIMUM_REQUIRED(VERSION 3.3.0)

FIND_PACKAGE(deal.II 9.4 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()


# Are all dependencies fullfilled?
IF(NOT DEAL_II_WITH_MPI       OR
   NOT DEAL_II_WITH_TRILINOS  OR
   NOT DEAL_II_TRILINOS_WITH_SACADO)
  MESSAGE(FATAL_ERROR "
Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_MPI = ON
    DEAL_II_WITH_TRILINOS = ON
    DEAL_II_TRILINOS_WITH_SACADO = ON
One or all of these are OFF in your installation but are required for this code gallery example."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()
