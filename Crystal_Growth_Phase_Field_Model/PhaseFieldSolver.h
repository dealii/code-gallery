
/* -----------------------------------------------------------------------------
*
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
* Copyright (C) 2024 by Umair Hussain
*
* This file is part of the deal.II code gallery.
*
* -----------------------------------------------------------------------------
*/

#ifndef KOBAYASHI_PARALLEL_PHASEFIELDSOLVER_H
#define KOBAYASHI_PARALLEL_PHASEFIELDSOLVER_H

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/grid_in.h>

//For Parallel Computation
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/distributed/solution_transfer.h>

#include <fstream>
#include <iostream>

using namespace dealii;

class PhaseFieldSolver {
public:
    PhaseFieldSolver();
    void run();

private:
    void          make_grid_and_dofs();
    void          assemble_system();
    void          solve();
    void          output_results(const unsigned int timestep_number) const;
    double        compute_residual();
    void          applying_bc();
    float         get_random_number();

    MPI_Comm mpi_communicator;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;
    ConditionalOStream pcout;

    Triangulation<2>     triangulation;
    FESystem<2>          fe;
    DoFHandler<2>        dof_handler;
    GridIn<2>            gridin;

    PETScWrappers::MPI::SparseMatrix jacobian_matrix;

    double       time;
    const double final_time, time_step;
    const double theta;
    const double epsilon, tau, gamma, latent_heat, alpha, t_eq, a; //as given in Ref. [1]

    PETScWrappers::MPI::Vector conv_solution; //solution vector at last newton-raphson iteration
    PETScWrappers::MPI::Vector old_solution; //solution vector at last time step
    PETScWrappers::MPI::Vector solution_update; //increment in solution or delta solution
    PETScWrappers::MPI::Vector system_rhs; //to store residual
    Vector<double> conv_solution_np, old_solution_np; //creating non parallel vectors to store data for easy access of old solution values by all processes

};

// Initial values class
class InitialValues : public Function<2>
{
public:
    InitialValues(): Function<2>(2)
    {}
    virtual void vector_value(const Point<2> &p,
                              Vector<double> & value) const override;
};


#endif //KOBAYASHI_PARALLEL_PHASEFIELDSOLVER_H
