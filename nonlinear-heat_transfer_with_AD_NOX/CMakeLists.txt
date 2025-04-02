CMAKE_MINIMUM_REQUIRED(VERSION 3.13.4)
set(CMAKE_CXX_STANDARD_REQUIRED True)
FIND_PACKAGE(deal.II 9.5.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

# Set the variable SOURCE_DIR as ./source, which can be used in the ADD_LIBRARY command
SET(SOURCE_DIR "./source")
# Specify the directory from where it should get the header files (.h files)
INCLUDE_DIRECTORIES(./include)

#Set a variable OUTPUT_DIR and use it to create a new directory in the file command below
SET(OUTPUT_DIR "./output")
#In the source code ./source/output_results.cc we use this folder to write the output
# This will create the directory
file(MAKE_DIRECTORY ${OUTPUT_DIR})

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(project)

# Add all source files
ADD_LIBRARY(lib 
        ${SOURCE_DIR}/compute_jacobian.cc # the jacobian
        ${SOURCE_DIR}/compute_residual.cc # the residual
        ${SOURCE_DIR}/initial_conditions.cc # the initial conditions
        ${SOURCE_DIR}/set_boundary_conditions.cc # apply boundary conditions
        ${SOURCE_DIR}/boundary_values.cc # Define boundary values
        ${SOURCE_DIR}/nonlinear_heat_cons_des.cc #Constructor/Destructor
        ${SOURCE_DIR}/output_results.cc # Output the results
        ${SOURCE_DIR}/setup_system.cc # Set up the system
        ${SOURCE_DIR}/solve_and_run.cc # Solve and run
        )

#Set up the Target for this library (You can call lib as anything)
DEAL_II_SETUP_TARGET(lib)

ADD_EXECUTABLE(nonlinear_heat nonlinear_heat.cc)
DEAL_II_SETUP_TARGET(nonlinear_heat)
#Here the main executable and the lib are linked
TARGET_LINK_LIBRARIES(nonlinear_heat lib)

# To make "make run" run the executable file plane_stress, use this
ADD_CUSTOM_TARGET (run COMMAND nonlinear_heat COMMENT "Run with ${COMMAND_BUILD_TYPE} configuration")

