#include <iostream>
#include "PhaseFieldSolver.h"

int main(int argc, char **argv) {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    PhaseFieldSolver phasefieldsolver;
    phasefieldsolver.run();
    return 0;
}
