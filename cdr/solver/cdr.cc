#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/numerics/error_estimator.h>

// These headers are for distributed computations:
#include <deal.II/base/utilities.h>
#include <deal.II/base/index_set.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_vector.h>

#include <chrono>
#include <functional>
#include <iostream>

#include <deal.II-cdr/system_matrix.h>
#include <deal.II-cdr/system_rhs.h>
#include <deal.II-cdr/parameters.h>
#include <deal.II-cdr/write_pvtu_output.h>

using namespace dealii;

constexpr int manifold_id {0};

// This is the actual solver class which performs time iteration and calls the
// appropriate library functions to do it.
template<int dim>
class CDRProblem
{
public:
  CDRProblem(const CDR::Parameters &parameters);
  void run();
private:
  const CDR::Parameters parameters;
  const double time_step;
  double current_time;

  MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;

  FE_Q<dim> fe;
  QGauss<dim> quad;
  const SphericalManifold<dim> boundary_description;
  parallel::distributed::Triangulation<dim> triangulation;
  DoFHandler<dim> dof_handler;

  const std::function<Tensor<1, dim>(const Point<dim>)> convection_function;
  const std::function<double(double, const Point<dim>)> forcing_function;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  AffineConstraints<double> constraints;
  bool first_run;

  // As is usual in parallel programs, I keep two copies of parts of the
  // complete solution: <code>locally_relevant_solution</code> contains both
  // the locally calculated solution as well as the layer of cells at its
  // boundary (the @ref GlossGhostCells "ghost cells") while
  // <code>completely_distributed_solution</code> only contains the parts of
  // the solution computed on the current @ref GlossMPIProcess "MPI process".
  TrilinosWrappers::MPI::Vector locally_relevant_solution;
  TrilinosWrappers::MPI::Vector completely_distributed_solution;
  TrilinosWrappers::MPI::Vector system_rhs;
  TrilinosWrappers::SparseMatrix system_matrix;
  TrilinosWrappers::PreconditionAMG preconditioner;

  ConditionalOStream pcout;

  void setup_geometry();
  void setup_system();
  void setup_dofs();
  void refine_mesh();
  void time_iterate();
};


template<int dim>
CDRProblem<dim>::CDRProblem(const CDR::Parameters &parameters) :
  parameters(parameters),
  time_step {(parameters.stop_time - parameters.start_time)
  /parameters.n_time_steps
},
current_time {parameters.start_time},
mpi_communicator (MPI_COMM_WORLD),
n_mpi_processes {Utilities::MPI::n_mpi_processes(mpi_communicator)},
this_mpi_process {Utilities::MPI::this_mpi_process(mpi_communicator)},
fe(parameters.fe_order),
quad(parameters.fe_order + 2),
boundary_description(Point<dim>()),
triangulation(mpi_communicator, typename Triangulation<dim>::MeshSmoothing
              (Triangulation<dim>::smoothing_on_refinement |
               Triangulation<dim>::smoothing_on_coarsening)),
dof_handler(triangulation),
convection_function
{
  [](const Point<dim> p) -> Tensor<1, dim>
  {Tensor<1, dim> v; v[0] = -p[1]; v[1] = p[0]; return v;}
},
forcing_function
{
  [](double t, const Point<dim> p) -> double
  {
    return std::exp(-8*t)*std::exp(-40*Utilities::fixed_power<6>(p[0] - 1.5))
    *std::exp(-40*Utilities::fixed_power<6>(p[1]));
  }
},
first_run {true},
pcout (std::cout, this_mpi_process == 0)
{
  Assert(dim == 2, ExcNotImplemented());
}


template<int dim>
void CDRProblem<dim>::setup_geometry()
{
  const Point<dim> center;
  GridGenerator::hyper_shell(triangulation, center, parameters.inner_radius,
                             parameters.outer_radius);
  triangulation.set_manifold(manifold_id, boundary_description);
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      cell->set_all_manifold_ids(manifold_id);
    }
  triangulation.refine_global(parameters.initial_refinement_level);
}


template<int dim>
void CDRProblem<dim>::setup_dofs()
{
  dof_handler.distribute_dofs(fe);
  pcout << "Number of degrees of freedom: "
        << dof_handler.n_dofs()
        << std::endl;
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  constraints.clear();
  constraints.reinit(locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  DoFTools::make_zero_boundary_constraints(dof_handler, manifold_id, constraints);
  constraints.close();

  completely_distributed_solution.reinit
  (locally_owned_dofs, mpi_communicator);

  locally_relevant_solution.reinit(locally_owned_dofs, locally_relevant_dofs,
                                   mpi_communicator);
}


template<int dim>
void CDRProblem<dim>::setup_system()
{
  DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern,
                                  constraints, /*keep_constrained_dofs*/true);
  SparsityTools::distribute_sparsity_pattern
  (dynamic_sparsity_pattern, dof_handler.n_locally_owned_dofs_per_processor(),
   mpi_communicator, locally_relevant_dofs);

  system_rhs.reinit(locally_owned_dofs, mpi_communicator);
  system_matrix.reinit(locally_owned_dofs, dynamic_sparsity_pattern,
                       mpi_communicator);

  CDR::create_system_matrix<dim>
  (dof_handler, quad, convection_function, parameters, time_step, constraints,
   system_matrix);
  system_matrix.compress(VectorOperation::add);
  preconditioner.initialize(system_matrix);
}


template<int dim>
void CDRProblem<dim>::time_iterate()
{
  double current_time = parameters.start_time;
  CDR::WritePVTUOutput pvtu_output(parameters.patch_level);
  for (unsigned int time_step_n = 0; time_step_n < parameters.n_time_steps;
       ++time_step_n)
    {
      current_time += time_step;

      system_rhs = 0.0;
      CDR::create_system_rhs<dim>
      (dof_handler, quad, convection_function, forcing_function, parameters,
       locally_relevant_solution, constraints, current_time, system_rhs);
      system_rhs.compress(VectorOperation::add);

      SolverControl solver_control(dof_handler.n_dofs(),
                                   1e-6*system_rhs.l2_norm(),
                                   /*log_history = */ false,
                                   /*log_result = */ false);
      TrilinosWrappers::SolverGMRES solver(solver_control);
      solver.solve(system_matrix, completely_distributed_solution, system_rhs,
                   preconditioner);
      constraints.distribute(completely_distributed_solution);
      locally_relevant_solution = completely_distributed_solution;

      if (time_step_n % parameters.save_interval == 0)
        {
          pvtu_output.write_output(dof_handler, locally_relevant_solution,
                                   time_step_n, current_time);
        }

      refine_mesh();
    }
}


template<int dim>
void CDRProblem<dim>::refine_mesh()
{
  using FunctionMap =
    std::map<types::boundary_id, const Function<dim> *>;

  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
  KellyErrorEstimator<dim>::estimate
  (dof_handler, QGauss<dim - 1>(fe.degree + 1), FunctionMap(),
   locally_relevant_solution, estimated_error_per_cell);

  // This solver uses a crude refinement strategy where cells with relatively
  // high errors are refined and cells with relatively low errors are
  // coarsened. The maximum refinement level is capped to prevent run-away
  // refinement.
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      if (std::abs(estimated_error_per_cell[cell->active_cell_index()]) >= 1e-3)
        {
          cell->set_refine_flag();
        }
      else if (std::abs(estimated_error_per_cell[cell->active_cell_index()]) <= 1e-5)
        {
          cell->set_coarsen_flag();
        }
    }

  if (triangulation.n_levels() > parameters.max_refinement_level)
    {
      for (const auto &cell :
           triangulation.cell_iterators_on_level(parameters.max_refinement_level))
        {
          cell->clear_refine_flag();
        }
    }

  // Transferring the solution between different grids is ultimately just a
  // few function calls but they must be made in exactly the right order.
  parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::Vector>
  solution_transfer(dof_handler);

  triangulation.prepare_coarsening_and_refinement();
  solution_transfer.prepare_for_coarsening_and_refinement
  (locally_relevant_solution);
  triangulation.execute_coarsening_and_refinement();

  setup_dofs();

  // The <code>solution_transfer</code> object stores a pointer to
  // <code>locally_relevant_solution</code>, so when
  // parallel::distributed::SolutionTransfer::interpolate is called it uses
  // those values to populate <code>temporary</code>.
  TrilinosWrappers::MPI::Vector temporary
  (locally_owned_dofs, mpi_communicator);
  solution_transfer.interpolate(temporary);
  // After <code>temporary</code> has the correct value, this call correctly
  // populates <code>completely_distributed_solution</code>, which had its
  // index set updated above with the call to <code>setup_dofs</code>.
  completely_distributed_solution = temporary;
  // Constraints cannot be applied to
  // @ref GlossGhostedVector "vectors with ghost entries" since the ghost
  // entries are write only, so this first goes through the completely
  // distributed vector.
  constraints.distribute(completely_distributed_solution);
  locally_relevant_solution = completely_distributed_solution;
  setup_system();
}


template<int dim>
void CDRProblem<dim>::run()
{
  setup_geometry();
  setup_dofs();
  setup_system();
  time_iterate();
}


constexpr int dim {2};


int main(int argc, char *argv[])
{
  // One of the new features in C++11 is the <code>chrono</code> component of
  // the standard library. This gives us an easy way to time the output.
  auto t0 = std::chrono::high_resolution_clock::now();

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  CDR::Parameters parameters;
  parameters.read_parameter_file("parameters.prm");
  CDRProblem<dim> cdr_problem(parameters);
  cdr_problem.run();

  auto t1 = std::chrono::high_resolution_clock::now();
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << "time elapsed: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                << " milliseconds."
                << std::endl;
    }

  return 0;
}
