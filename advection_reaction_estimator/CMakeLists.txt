# Set the name of the project and target:
SET(TARGET "DG_advection_reaction_estimator")

# Declare all source files the target consists of:
SET(TARGET_SRC
  source/DG_advection_reaction.cc
  main.cc
)

PROJECT(${TARGET} CXX)


CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)

FIND_PACKAGE(deal.II 9.3.0
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
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()
