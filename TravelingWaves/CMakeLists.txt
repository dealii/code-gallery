##
#  CMake script for the TravelingWaves program:
##

# The name of the project and target:
SET(TARGET "main")

SET(TARGET_SRC
  ${TARGET}.cc calculate_profile.cc Solution.cc TravelingWaveSolver.cc Parameters.cc LimitSolution.cc 
  )

CMAKE_MINIMUM_REQUIRED(VERSION 3.13.4)

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

#
# Are all dependencies fulfilled?
#
IF(NOT (DEAL_II_WITH_UMFPACK AND DEAL_II_WITH_SUNDIALS)) # keep in one line
  MESSAGE(FATAL_ERROR "
    Error! This program requires a deal.II library that was configured with the following options:
        DEAL_II_WITH_UMFPACK  = ON
        DEAL_II_WITH_SUNDIALS = ON
    However, the deal.II library found at ${DEAL_II_PATH} was configured with these options:
        DEAL_II_WITH_UMFPACK  = ${DEAL_II_WITH_UMFPACK}
        DEAL_II_WITH_SUNDIALS = ${DEAL_II_WITH_SUNDIALS}
    This conflicts with the requirements."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})

DEAL_II_INVOKE_AUTOPILOT()
