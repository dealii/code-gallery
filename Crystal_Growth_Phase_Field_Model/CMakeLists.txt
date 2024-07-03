cmake_minimum_required(VERSION 3.16)
# Set the name of the project and target:
SET(TARGET "Kobayashi_Parallel")

#set(CMAKE_CXX_STANDARD 14)

SET(TARGET_SRC
        main.cpp
        PhaseFieldSolver.cpp
        applying_bc.cpp
        assemble_system.cpp
        grid_dof.cpp
        output_results.cpp
        run.cpp
        solve.cpp
        InitialValues.cpp
        get_random_number.cpp
        )

FIND_PACKAGE(deal.II 9.2.0 QUIET
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
