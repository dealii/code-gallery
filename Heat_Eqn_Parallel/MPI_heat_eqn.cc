/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2024 by Wasim Niyaz Munshi
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 */
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/generic_linear_algebra.h>
#define FORCE_USE_OF_TRILINOS
namespace LA {
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) &&     \
    !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
using namespace dealii::LinearAlgebraPETSc;
#define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
using namespace dealii::LinearAlgebraTrilinos;
#else
#error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

namespace MPIHeatEquation {
using namespace dealii;

template <int dim> class HeatEquation {
public:
  HeatEquation();
  void run();

private:
  void setup_system();
  void assemble_system(const double &time);
  void solve_time_step();
  void output_results() const;
  void refine_mesh(const unsigned int min_grid_level,
                   const unsigned int max_grid_level);

  MPI_Comm mpi_communicator;
  ConditionalOStream pcout;
  TimerOutput computing_timer;

  parallel::distributed::Triangulation<dim> triangulation;
  FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  AffineConstraints<double> constraints;

  LA::MPI::SparseMatrix system_matrix;

  LA::MPI::Vector locally_relevant_solution;
  LA::MPI::Vector old_locally_relevant_solution;
  LA::MPI::Vector system_rhs;

  double time;
  double time_step;
  unsigned int timestep_number;
  const double theta;
};

template <int dim> class RightHandSide : public Function<dim> {
public:
  RightHandSide() : Function<dim>(), period(0.2) {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;

private:
  const double period;
};

template <int dim>
double RightHandSide<dim>::value(const Point<dim> &p,
                                 const unsigned int component) const {
  (void)component;
  AssertIndexRange(component, 1);
  Assert(dim == 2, ExcNotImplemented());

  const double time = this->get_time();
  const double point_within_period =
      (time / period - std::floor(time / period));

  if ((point_within_period >= 0.0) && (point_within_period <= 0.2)) {
    if ((p[0] > 0.5) && (p[1] > -0.5))
      return 1;
    else
      return 0;
  } else if ((point_within_period >= 0.5) && (point_within_period <= 0.7)) {
    if ((p[0] > -0.5) && (p[1] > 0.5))
      return 1;
    else
      return 0;
  } else
    return 0;
}

template <int dim> class BoundaryValues : public Function<dim> {
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
};

template <int dim>
double BoundaryValues<dim>::value(const Point<dim> & /*p*/,
                                  const unsigned int component) const {
  (void)component;
  Assert(component == 0, ExcIndexRange(component, 0, 1));
  return 0;
}

template <int dim>

HeatEquation<dim>::HeatEquation()

    : mpi_communicator(MPI_COMM_WORLD),
      pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
      computing_timer(mpi_communicator, pcout, TimerOutput::never,
                      TimerOutput::wall_times),
      triangulation(mpi_communicator), fe(1), dof_handler(triangulation),
      time_step(1. / 500), theta(0.5) {}

template <int dim> void HeatEquation<dim>::setup_system() {
  TimerOutput::Scope t(computing_timer, "setup");

  dof_handler.distribute_dofs(fe);

  pcout << std::endl
        << "===========================================" << std::endl
        << "   Number of active cells:       "
        << triangulation.n_global_active_cells() << std::endl
        << "   Number of degrees of freedom: " << dof_handler.n_dofs()
        << std::endl;

  locally_owned_dofs = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
  locally_relevant_solution.reinit(locally_owned_dofs, locally_relevant_dofs,
                                   mpi_communicator);
  old_locally_relevant_solution.reinit(locally_owned_dofs,
                                       locally_relevant_dofs, mpi_communicator);
  system_rhs.reinit(locally_owned_dofs, mpi_communicator);

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(
      dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
  constraints.close();

  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, true);
  SparsityTools::distribute_sparsity_pattern(
      dsp, dof_handler.locally_owned_dofs(), mpi_communicator,
      locally_relevant_dofs);

  system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp,
                       mpi_communicator);
}

template <int dim> void HeatEquation<dim>::assemble_system(const double &time) {
  TimerOutput::Scope t(computing_timer, "assemble_system");

  const QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_laplace_matrix(dofs_per_cell, dofs_per_cell);

  Vector<double> cell_rhs(dofs_per_cell);
  Vector<double> cell_tmp(dofs_per_cell);
  Vector<double> cell_forcing_terms(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<double> rhs_values(n_q_points);
  std::vector<double> rhs_values_old(n_q_points);
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())

    {
      cell_matrix = 0;
      cell_mass_matrix = 0;
      cell_laplace_matrix = 0;
      cell_rhs = 0;
      cell_forcing_terms = 0;
      cell_tmp = 0;

      fe_values.reinit(cell);

      for (const unsigned int q_point : fe_values.quadrature_point_indices()) {
        for (const unsigned int i : fe_values.dof_indices()) {

          for (const unsigned int j : fe_values.dof_indices()) {

            cell_mass_matrix(i, j) += fe_values.shape_value(i, q_point) *
                                      fe_values.shape_value(j, q_point) *
                                      fe_values.JxW(q_point);
            cell_laplace_matrix(i, j) += fe_values.shape_grad(i, q_point) *
                                         fe_values.shape_grad(j, q_point) *
                                         fe_values.JxW(q_point);

            cell_matrix(i, j) +=
                ((fe_values.shape_value(i, q_point) *
                  fe_values.shape_value(j, q_point)) +
                 (theta * time_step * fe_values.shape_grad(i, q_point) *
                  fe_values.shape_grad(j, q_point))) *
                fe_values.JxW(q_point);
          }
        }
      }

      // First compute M*U_old -(1-theta)*k*A*U_old
      cell->get_dof_indices(local_dof_indices);
      Vector<double> old_cell_solution(dofs_per_cell);
      cell->get_dof_values(old_locally_relevant_solution, old_cell_solution);
      cell_mass_matrix.vmult(cell_rhs, old_cell_solution);
      cell_laplace_matrix.vmult(cell_tmp, old_cell_solution);
      cell_rhs.add(-(1 - theta) * time_step, cell_tmp);

      // Add the rhs terms
      RightHandSide<dim> rhs_function;
      // k*theta*F^n
      rhs_function.set_time(time);
      rhs_function.value_list(fe_values.get_quadrature_points(), rhs_values);
      for (const unsigned int q_point : fe_values.quadrature_point_indices())
        for (const unsigned int i : fe_values.dof_indices())
          cell_rhs(i) += time_step * theta *
                         (fe_values.shape_value(i, q_point) *
                          rhs_values[q_point] * fe_values.JxW(q_point));

      // Adding:k*(1-theta)*F^{n-1}
      rhs_function.set_time(time - time_step);
      rhs_function.value_list(fe_values.get_quadrature_points(),
                              rhs_values_old);
      for (const unsigned int q_point : fe_values.quadrature_point_indices())
        for (const unsigned int i : fe_values.dof_indices())
          cell_rhs(i) += time_step * (1 - theta) *
                         (fe_values.shape_value(i, q_point) *
                          rhs_values_old[q_point] * fe_values.JxW(q_point));

      constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}

template <int dim> void HeatEquation<dim>::solve_time_step() {
  TimerOutput::Scope t(computing_timer, "solve");

  LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                  mpi_communicator);

  SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());

  SolverCG<LA::MPI::Vector> solver(solver_control);

  LA::MPI::PreconditionAMG::AdditionalData data;
#ifdef USE_PETSC_LA
  data.symmetric_operator = true;
#else
  /* Trilinos defaults are good */
#endif
  LA::MPI::PreconditionAMG preconditioner;
  preconditioner.initialize(system_matrix, data);

  solver.solve(system_matrix, completely_distributed_solution, system_rhs,
               preconditioner);

  pcout << "   Solved in " << solver_control.last_step() << " iterations."
        << std::endl;

  constraints.distribute(completely_distributed_solution);

  locally_relevant_solution = completely_distributed_solution;
}

template <int dim> void HeatEquation<dim>::output_results() const {
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(locally_relevant_solution, "T");

  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();
  data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));

  data_out.write_vtu_with_pvtu_record("./", "solution", timestep_number,
                                      mpi_communicator, 2, 8);
}

template <int dim>
void HeatEquation<dim>::refine_mesh(const unsigned int min_grid_level,
                                    const unsigned int max_grid_level)

{
  TimerOutput::Scope t(computing_timer, "refine");
  Vector<float> estimated_error_per_cell(
      triangulation.n_locally_owned_active_cells());

  KellyErrorEstimator<dim>::estimate(
      dof_handler, QGauss<dim - 1>(fe.degree + 1),
      std::map<types::boundary_id, const Function<dim> *>(),
      locally_relevant_solution, estimated_error_per_cell);

  parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
      triangulation, estimated_error_per_cell, 0.6, 0.4);

  if (triangulation.n_levels() > max_grid_level) {
    for (const auto &cell :
         triangulation.active_cell_iterators_on_level(max_grid_level))
      if (cell->is_locally_owned())
        cell->clear_refine_flag();
  }
  for (const auto &cell :
       triangulation.active_cell_iterators_on_level(min_grid_level)) {
    if (cell->is_locally_owned())
      cell->clear_coarsen_flag();
  }

  parallel::distributed::SolutionTransfer<dim, LA::MPI::Vector> solution_trans(
      dof_handler);

  LA::MPI::Vector previous_locally_relevant_solution(
      locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  previous_locally_relevant_solution = locally_relevant_solution;
  triangulation.prepare_coarsening_and_refinement();
  solution_trans.prepare_for_coarsening_and_refinement(
      previous_locally_relevant_solution);
  triangulation.execute_coarsening_and_refinement();
  setup_system();
  LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                  mpi_communicator);

  solution_trans.interpolate(completely_distributed_solution);
  constraints.distribute(completely_distributed_solution);
  locally_relevant_solution = completely_distributed_solution;
}

template <int dim> void HeatEquation<dim>::run() {
  pcout << "Running with "
#ifdef USE_PETSC_LA
        << "PETSc"
#else
        << "Trilinos"
#endif
        << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
        << " MPI rank(s)..." << std::endl;

  const unsigned int initial_global_refinement = 12;//2;
  const unsigned int n_adaptive_pre_refinement_steps = 0;//4;
  GridGenerator::hyper_L(triangulation);
  triangulation.refine_global(initial_global_refinement);

  setup_system();

  unsigned int pre_refinement_step = 0;

start_time_iteration:

  time = 0.0;
  timestep_number = 0;

  LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                  mpi_communicator);
  VectorTools::interpolate(dof_handler, Functions::ZeroFunction<dim>(),
                           completely_distributed_solution);
  old_locally_relevant_solution = completely_distributed_solution;

  locally_relevant_solution = completely_distributed_solution;

  {
    TimerOutput::Scope t(computing_timer, "output");
    output_results();
  }

  const double end_time = 1./500;//0.5;
  while (time < end_time) {
  //while (time <= end_time) {
    time += time_step;
    ++timestep_number;

    pcout << "Time step " << timestep_number << " at t=" << time << std::endl;

    system_rhs.reinit(locally_owned_dofs, mpi_communicator);
    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, true);
    SparsityTools::distribute_sparsity_pattern(
        dsp, dof_handler.locally_owned_dofs(), mpi_communicator,
        locally_relevant_dofs);
    system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp,
                         mpi_communicator);

    assemble_system(time);
    solve_time_step();

    {
      TimerOutput::Scope t(computing_timer, "output");
      output_results();
    }
    computing_timer.print_summary();
    computing_timer.reset();
    pcout << std::endl;

    if ((timestep_number == 1) &&
        (pre_refinement_step < n_adaptive_pre_refinement_steps)) {
      refine_mesh(initial_global_refinement,
                  initial_global_refinement + n_adaptive_pre_refinement_steps);
      ++pre_refinement_step;

      pcout << std::endl;

      goto start_time_iteration;
    } else if ((timestep_number > 0) && (timestep_number % 5 == 0)) {
      refine_mesh(initial_global_refinement,
                  initial_global_refinement + n_adaptive_pre_refinement_steps);
    }

    old_locally_relevant_solution = locally_relevant_solution;
  }
}
} // namespace MPIHeatEquation

int main(int argc, char *argv[]) {
  try {
    // Added:
    using namespace dealii;

    using namespace MPIHeatEquation;

    // Added:
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    HeatEquation<2> heat_equation_solver;
    heat_equation_solver.run();
  } catch (std::exception &exc) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  } catch (...) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}
