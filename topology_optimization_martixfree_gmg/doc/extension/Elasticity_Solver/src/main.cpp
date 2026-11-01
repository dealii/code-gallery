#include "../include/Elasticity_Matrix_Free.h"
int main(int argc, char *argv[])
{
  using namespace Elasticity_MF;
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  Elasticity_Solver<2,1> solver;
  solver.run();
  return 0;
}