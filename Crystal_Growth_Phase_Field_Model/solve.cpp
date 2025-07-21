/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
 * Copyright (C) 2024 by Umair Hussain
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 */

#include "PhaseFieldSolver.h"

void PhaseFieldSolver::solve(){
    //Using a direct parallel solver
    SolverControl cn;
    PETScWrappers::SparseDirectMUMPS A_direct(cn);
    A_direct.solve(jacobian_matrix, solution_update, system_rhs);
    //Updating the solution by adding the delta solution
    conv_solution.add(1, solution_update);
    conv_solution.compress(VectorOperation::add);
}
