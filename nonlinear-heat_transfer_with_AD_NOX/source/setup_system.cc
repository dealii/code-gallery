/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
 * Copyright (C) 2024 by Narasimhan Swaminathan
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 */

#include "allheaders.h"
#include "nonlinear_heat.h"

/**
 * Sets up the system.
 * @param time_step
 */
void nonlinear_heat::setup_system(unsigned int time_step)
{
     if (time_step ==0) {
        dof_handler.distribute_dofs(fe);
         converged_solution.reinit(dof_handler.n_dofs());
         present_solution.reinit(dof_handler.n_dofs());
     }

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
    matrix_factorization.reset();
}
