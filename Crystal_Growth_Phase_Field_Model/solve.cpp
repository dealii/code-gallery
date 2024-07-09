#include "PhaseFieldSolver.h"

void PhaseFieldSolver::solve(){
    //Using a direct parallel solver
    SolverControl cn;
    PETScWrappers::SparseDirectMUMPS A_direct(cn, mpi_communicator);
    A_direct.solve(jacobian_matrix, solution_update, system_rhs);
    //Updating the solution by adding the delta solution
    conv_solution.add(1, solution_update);
    conv_solution.compress(VectorOperation::add);
}
