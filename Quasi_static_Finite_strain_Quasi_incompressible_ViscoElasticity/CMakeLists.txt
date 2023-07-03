##
#  CMake script for the step-<XYZ> tutorial program:
##

# Set the name of the project and target:
SET(TARGET "viscoelastic_strip_with_hole")

# Declare all source files the targest consists of:
SET(TARGET_SRC
  ${TARGET}.cc
)

SET(CLEAN_UP_FILES
  # a custom list of globs, e.g. *.log *.vtk
  *.vtu *.pvtu *.pvd
)

# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 3.3.0)

FIND_PACKAGE(deal.II 9.5
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

#
# Are all dependencies fullfilled?
#
IF(NOT DEAL_II_WITH_CXX11     OR
   NOT DEAL_II_WITH_MPI       OR
   NOT DEAL_II_WITH_METIS     OR
   NOT DEAL_II_WITH_TRILINOS)
  MESSAGE(FATAL_ERROR "
Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_CXX11 = ON
    DEAL_II_WITH_MPI = ON
    DEAL_II_WITH_METIS = ON
    DEAL_II_WITH_TRILINOS = ON
One or all of these are OFF in your installation but are required for this code gallery example."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()
