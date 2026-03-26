/*
 * NewtonStepSystem.h
 *
 *  Created on: 05 May 2015
 *      Author: maien
 */

#ifndef NEWTONSTEPSYSTEM_H_
#define NEWTONSTEPSYSTEM_H_


#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/sparsity_tools.h>

#include "DoFSystem.h"
#include "Constants.h"

using namespace dealii;

namespace PlasticityLab {

  class NewtonStepSystem {
   public:

    template<class DoFSystemType>
    void setup(const DoFSystemType &dof_system) {
      TrilinosWrappers::SparsityPattern sparsity_pattern(
        dof_system.locally_owned_dofs,
        mpi_communicator);

      DoFTools::make_sparsity_pattern(
        dof_system.dof_handler, sparsity_pattern,
        dof_system.nodal_constraints,
        false,
        Utilities::MPI::this_mpi_process(mpi_communicator));

      sparsity_pattern.compress();
      Newton_step_matrix.reinit(sparsity_pattern);
      Newton_step_solution.reinit(
        dof_system.locally_owned_dofs,
        mpi_communicator);
      current_increment.reinit(
        dof_system.locally_relevant_dofs,
        mpi_communicator);
      Newton_step_residual.reinit(
        dof_system.locally_owned_dofs,
        mpi_communicator);
      previous_deformation.reinit(
        dof_system.locally_relevant_dofs,
        mpi_communicator);

      previous_time_derivative.reinit(
        dof_system.locally_relevant_dofs,
        mpi_communicator);
      previous_second_time_derivative.reinit(
        dof_system.locally_relevant_dofs,
        mpi_communicator);

      _locally_owned_previous_time_derivative.reinit(
        dof_system.locally_owned_dofs,
        mpi_communicator);
      _locally_owned_previous_second_time_derivative.reinit(
        dof_system.locally_owned_dofs,
        mpi_communicator);
    }

    template<class DoFSystemType>
    void update_matrix_constraints(const DoFSystemType &dof_system) {
      TrilinosWrappers::SparsityPattern sparsity_pattern(
        dof_system.locally_owned_dofs,
        mpi_communicator);

      DoFTools::make_sparsity_pattern(
        dof_system.dof_handler, sparsity_pattern,
        dof_system.nodal_constraints,
        false,
        Utilities::MPI::this_mpi_process(mpi_communicator));

      sparsity_pattern.compress();
      Newton_step_matrix.reinit(sparsity_pattern);
    }

    void advance_time(double delta_t, double rho_infty, const bool reset_increment=true) {
      {
        double alpha_m, alpha_f, gamma, beta;
        get_generalized_alpha_method_params(
            &alpha_m, &alpha_f, &gamma, &beta, rho_infty);

        _locally_owned_previous_time_derivative = previous_time_derivative;
        _locally_owned_previous_second_time_derivative = previous_second_time_derivative;

        const TrilinosWrappers::MPI::Vector previous_second_time_derivative_backup(_locally_owned_previous_second_time_derivative);

        _locally_owned_previous_second_time_derivative = current_increment;
        _locally_owned_previous_second_time_derivative.add(
              -delta_t,
              _locally_owned_previous_time_derivative,
              -delta_t*delta_t*(0.5-beta),
              previous_second_time_derivative_backup);
        _locally_owned_previous_second_time_derivative *= (1./(beta*delta_t*delta_t));
        _locally_owned_previous_time_derivative.add(
              delta_t*(1.-gamma),
              previous_second_time_derivative_backup,
              delta_t*gamma,
              _locally_owned_previous_second_time_derivative);


        previous_time_derivative = _locally_owned_previous_time_derivative;
        previous_second_time_derivative = _locally_owned_previous_second_time_derivative;

      }
      previous_deformation += current_increment;
      if(reset_increment) {
        current_increment = 0;
      }
    }

    TrilinosWrappers::SparseMatrix  Newton_step_matrix;
    TrilinosWrappers::MPI::Vector   previous_deformation;
    TrilinosWrappers::MPI::Vector   current_increment;
    TrilinosWrappers::MPI::Vector   Newton_step_solution;
    TrilinosWrappers::MPI::Vector   Newton_step_residual;

    TrilinosWrappers::MPI::Vector   previous_time_derivative;
    TrilinosWrappers::MPI::Vector   previous_second_time_derivative;

    TrilinosWrappers::MPI::Vector   _locally_owned_previous_time_derivative;
    TrilinosWrappers::MPI::Vector   _locally_owned_previous_second_time_derivative;

  };

} /* namespace PlasticityLab */

#endif /* NEWTONSTEPSYSTEM_H_ */
