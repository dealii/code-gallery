# Set the name of the project and target:
SET(TARGET "agglomeration_poisson")

# Declare all source files the target consists of.
FILE(GLOB TARGET_SRC CONFIGURE_DEPENDS ${TARGET}.cc ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cc)

CMAKE_MINIMUM_REQUIRED(VERSION 3.13.4)

FIND_PACKAGE(deal.II 9.6.0
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

TARGET_INCLUDE_DIRECTORIES(${TARGET} PRIVATE include)
TARGET_COMPILE_DEFINITIONS(${TARGET} PRIVATE MESH_DIR="${CMAKE_SOURCE_DIR}/mesh")



# -------------------------------------------------------------------
# Coverage target: "make coverage"
# -------------------------------------------------------------------
target_compile_options(${TARGET} PRIVATE
    -O0
    -g
    --coverage
    -fprofile-arcs
    -fprofile-update=atomic
    -ftest-coverage
)
target_link_options(${TARGET} PRIVATE "--coverage")

find_program(LCOV_EXEC lcov)
find_program(GENHTML_EXEC genhtml)

if(LCOV_EXEC AND GENHTML_EXEC)
    add_custom_target(coverage
        COMMAND ${CMAKE_COMMAND} -E echo "Generating coverage report..."
        COMMAND ${CMAKE_COMMAND} -E rm -rf coverage_report coverage.info
        COMMAND ${LCOV_EXEC} --capture --directory . --output-file coverage.info
        COMMAND ${LCOV_EXEC} --extract coverage.info "*${CMAKE_SOURCE_DIR}*" --output-file coverage.info
        COMMAND ${GENHTML_EXEC} --function-coverage --output-directory coverage_report coverage.info
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running LCOV + genhtml"
        VERBATIM
    )
else()
    message(WARNING "lcov/genhtml not found - coverage target will not be available.")
endif()
