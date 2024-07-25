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

void PhaseFieldSolver::make_grid_and_dofs() {
    //Reading mesh
    gridin.attach_triangulation(triangulation);
    std::ifstream f("mesh/Kobayashi_mesh100x400.msh");
    gridin.read_msh(f);


    GridTools::partition_triangulation(n_mpi_processes, triangulation);
    dof_handler.distribute_dofs(fe);
    DoFRenumbering::subdomain_wise(dof_handler);

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);

    const std::vector<IndexSet> locally_owned_dofs_per_proc =
            DoFTools::locally_owned_dofs_per_subdomain(dof_handler);
    const IndexSet locally_owned_dofs =
            locally_owned_dofs_per_proc[this_mpi_process];
    jacobian_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);
    old_solution.reinit(locally_owned_dofs, mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);
    conv_solution.reinit(locally_owned_dofs, mpi_communicator);
    solution_update.reinit(locally_owned_dofs, mpi_communicator);

    conv_solution_np.reinit(dof_handler.n_dofs());
    old_solution_np.reinit(dof_handler.n_dofs());
}
