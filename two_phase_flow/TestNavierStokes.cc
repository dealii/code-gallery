#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/compressed_simple_sparsity_pattern.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

#include "NavierStokesSolver.cc"
#include "utilities_test_NS.cc"

///////////////////////////////////////////////////////
///////////////////// MAIN CLASS //////////////////////
///////////////////////////////////////////////////////
template <int dim>
class TestNavierStokes
{
public:
  TestNavierStokes(const unsigned int degree_LS, const unsigned int degree_U);
  ~TestNavierStokes();
  void
  run();

private:
  void
  get_boundary_values_U(double t);
  void
  fix_pressure();
  void
  output_results();
  void
  process_solution(const unsigned int cycle);
  void
  setup();
  void
  initial_condition();
  void
  init_constraints();

  PETScWrappers::MPI::Vector locally_relevant_solution_rho;
  PETScWrappers::MPI::Vector locally_relevant_solution_u;
  PETScWrappers::MPI::Vector locally_relevant_solution_v;
  PETScWrappers::MPI::Vector locally_relevant_solution_w;
  PETScWrappers::MPI::Vector locally_relevant_solution_p;
  PETScWrappers::MPI::Vector completely_distributed_solution_rho;
  PETScWrappers::MPI::Vector completely_distributed_solution_u;
  PETScWrappers::MPI::Vector completely_distributed_solution_v;
  PETScWrappers::MPI::Vector completely_distributed_solution_w;
  PETScWrappers::MPI::Vector completely_distributed_solution_p;

  std::vector<unsigned int> boundary_values_id_u;
  std::vector<unsigned int> boundary_values_id_v;
  std::vector<unsigned int> boundary_values_id_w;
  std::vector<double>       boundary_values_u;
  std::vector<double>       boundary_values_v;
  std::vector<double>       boundary_values_w;

  double rho_fluid;
  double nu_fluid;
  double rho_air;
  double nu_air;

  MPI_Comm                                  mpi_communicator;
  parallel::distributed::Triangulation<dim> triangulation;

  int             degree_LS;
  DoFHandler<dim> dof_handler_LS;
  FE_Q<dim>       fe_LS;
  IndexSet        locally_owned_dofs_LS;
  IndexSet        locally_relevant_dofs_LS;

  int             degree_U;
  DoFHandler<dim> dof_handler_U;
  FE_Q<dim>       fe_U;
  IndexSet        locally_owned_dofs_U;
  IndexSet        locally_relevant_dofs_U;

  DoFHandler<dim> dof_handler_P;
  FE_Q<dim>       fe_P;
  IndexSet        locally_owned_dofs_P;
  IndexSet        locally_relevant_dofs_P;

  ConstraintMatrix constraints;

  // TimerOutput timer;

  double       time;
  double       time_step;
  double       final_time;
  unsigned int timestep_number;
  double       cfl;

  double min_h;

  unsigned int n_cycles;
  unsigned int n_refinement;
  unsigned int output_number;
  double       output_time;
  bool         get_output;

  double h;
  double umax;

  bool verbose;

  ConditionalOStream pcout;
  ConvergenceTable   convergence_table;

  double nu;
};

template <int dim>
TestNavierStokes<dim>::TestNavierStokes(const unsigned int degree_LS,
                                        const unsigned int degree_U)
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator,
                  typename Triangulation<dim>::MeshSmoothing(
                    Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening))
  , degree_LS(degree_LS)
  , dof_handler_LS(triangulation)
  , fe_LS(degree_LS)
  , degree_U(degree_U)
  , dof_handler_U(triangulation)
  , fe_U(degree_U)
  , dof_handler_P(triangulation)
  , fe_P(degree_U - 1)
  , // TODO: change this to be degree_Q-1
  // timer(std::cout, TimerOutput::summary, TimerOutput::wall_times),
  pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
{}

template <int dim>
TestNavierStokes<dim>::~TestNavierStokes()
{
  dof_handler_LS.clear();
  dof_handler_U.clear();
  dof_handler_P.clear();
}

/////////////////////////////////////////
///////////////// SETUP /////////////////
/////////////////////////////////////////
template <int dim>
void
TestNavierStokes<dim>::setup()
{
  // setup system LS
  dof_handler_LS.distribute_dofs(fe_LS);
  locally_owned_dofs_LS = dof_handler_LS.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler_LS,
                                          locally_relevant_dofs_LS);
  // setup system U
  dof_handler_U.distribute_dofs(fe_U);
  locally_owned_dofs_U = dof_handler_U.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler_U,
                                          locally_relevant_dofs_U);
  // setup system P //
  dof_handler_P.distribute_dofs(fe_P);
  locally_owned_dofs_P = dof_handler_P.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler_P,
                                          locally_relevant_dofs_P);
  init_constraints();
  // init vectors for rho
  locally_relevant_solution_rho.reinit(locally_owned_dofs_LS,
                                       locally_relevant_dofs_LS,
                                       mpi_communicator);
  locally_relevant_solution_rho = 0;
  completely_distributed_solution_rho.reinit(locally_owned_dofs_LS,
                                             mpi_communicator);
  // init vectors for u
  locally_relevant_solution_u.reinit(locally_owned_dofs_U,
                                     locally_relevant_dofs_U,
                                     mpi_communicator);
  locally_relevant_solution_u = 0;
  completely_distributed_solution_u.reinit(locally_owned_dofs_U,
                                           mpi_communicator);
  // init vectors for v
  locally_relevant_solution_v.reinit(locally_owned_dofs_U,
                                     locally_relevant_dofs_U,
                                     mpi_communicator);
  locally_relevant_solution_v = 0;
  completely_distributed_solution_v.reinit(locally_owned_dofs_U,
                                           mpi_communicator);
  // init vectors for w
  locally_relevant_solution_w.reinit(locally_owned_dofs_U,
                                     locally_relevant_dofs_U,
                                     mpi_communicator);
  locally_relevant_solution_w = 0;
  completely_distributed_solution_w.reinit(locally_owned_dofs_U,
                                           mpi_communicator);
  // init vectors for p
  locally_relevant_solution_p.reinit(locally_owned_dofs_P,
                                     locally_relevant_dofs_P,
                                     mpi_communicator);
  locally_relevant_solution_p = 0;
  completely_distributed_solution_p.reinit(locally_owned_dofs_P,
                                           mpi_communicator);
}

template <int dim>
void
TestNavierStokes<dim>::initial_condition()
{
  time = 0;
  // Initial conditions //
  // init condition for rho
  completely_distributed_solution_rho = 0;
  VectorTools::interpolate(dof_handler_LS,
                           RhoFunction<dim>(0),
                           completely_distributed_solution_rho);
  constraints.distribute(completely_distributed_solution_rho);
  locally_relevant_solution_rho = completely_distributed_solution_rho;
  // init condition for u
  completely_distributed_solution_u = 0;
  VectorTools::interpolate(dof_handler_U,
                           ExactSolution_and_BC_U<dim>(0, 0),
                           completely_distributed_solution_u);
  constraints.distribute(completely_distributed_solution_u);
  locally_relevant_solution_u = completely_distributed_solution_u;
  // init condition for v
  completely_distributed_solution_v = 0;
  VectorTools::interpolate(dof_handler_U,
                           ExactSolution_and_BC_U<dim>(0, 1),
                           completely_distributed_solution_v);
  constraints.distribute(completely_distributed_solution_v);
  locally_relevant_solution_v = completely_distributed_solution_v;
  // init condition for w
  if (dim == 3)
    {
      completely_distributed_solution_w = 0;
      VectorTools::interpolate(dof_handler_U,
                               ExactSolution_and_BC_U<dim>(0, 2),
                               completely_distributed_solution_w);
      constraints.distribute(completely_distributed_solution_w);
      locally_relevant_solution_w = completely_distributed_solution_w;
    }
  // init condition for p
  completely_distributed_solution_p = 0;
  VectorTools::interpolate(dof_handler_P,
                           ExactSolution_p<dim>(0),
                           completely_distributed_solution_p);
  constraints.distribute(completely_distributed_solution_p);
  locally_relevant_solution_p = completely_distributed_solution_p;
}

template <int dim>
void
TestNavierStokes<dim>::init_constraints()
{
  constraints.clear();
  constraints.reinit(locally_relevant_dofs_LS);
  DoFTools::make_hanging_node_constraints(dof_handler_LS, constraints);
  constraints.close();
}

template <int dim>
void
TestNavierStokes<dim>::fix_pressure()
{
  // fix the constant in the pressure
  completely_distributed_solution_p = locally_relevant_solution_p;
  double mean_value                 = VectorTools::compute_mean_value(
    dof_handler_P, QGauss<dim>(3), locally_relevant_solution_p, 0);
  if (dim == 2)
    completely_distributed_solution_p.add(
      -mean_value + std::sin(1) * (std::cos(time) - cos(1 + time)));
  else
    completely_distributed_solution_p.add(
      -mean_value + 8 * std::pow(std::sin(0.5), 3) * std::sin(1.5 + time));
  locally_relevant_solution_p = completely_distributed_solution_p;
}

template <int dim>
void
TestNavierStokes<dim>::output_results()
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler_U);
  data_out.add_data_vector(locally_relevant_solution_u, "u");
  data_out.add_data_vector(locally_relevant_solution_v, "v");
  if (dim == 3)
    data_out.add_data_vector(locally_relevant_solution_w, "w");

  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  const std::string filename =
    ("solution-" + Utilities::int_to_string(output_number, 3) + "." +
     Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4));
  std::ofstream output((filename + ".vtu").c_str());
  data_out.write_vtu(output);

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::vector<std::string> filenames;
      for (unsigned int i = 0;
           i < Utilities::MPI::n_mpi_processes(mpi_communicator);
           ++i)
        filenames.push_back("solution-" +
                            Utilities::int_to_string(output_number, 3) + "." +
                            Utilities::int_to_string(i, 4) + ".vtu");

      std::ofstream master_output((filename + ".pvtu").c_str());
      data_out.write_pvtu_record(master_output, filenames);
    }
  output_number++;
}

template <int dim>
void
TestNavierStokes<dim>::process_solution(const unsigned int cycle)
{
  Vector<double> difference_per_cell(triangulation.n_active_cells());
  // error for u
  VectorTools::integrate_difference(dof_handler_U,
                                    locally_relevant_solution_u,
                                    ExactSolution_and_BC_U<dim>(time, 0),
                                    difference_per_cell,
                                    QGauss<dim>(degree_U + 1),
                                    VectorTools::L2_norm);
  double u_L2_error = difference_per_cell.l2_norm();
  u_L2_error =
    std::sqrt(Utilities::MPI::sum(u_L2_error * u_L2_error, mpi_communicator));
  VectorTools::integrate_difference(dof_handler_U,
                                    locally_relevant_solution_u,
                                    ExactSolution_and_BC_U<dim>(time, 0),
                                    difference_per_cell,
                                    QGauss<dim>(degree_U + 1),
                                    VectorTools::H1_norm);
  double u_H1_error = difference_per_cell.l2_norm();
  u_H1_error =
    std::sqrt(Utilities::MPI::sum(u_H1_error * u_H1_error, mpi_communicator));
  // error for v
  VectorTools::integrate_difference(dof_handler_U,
                                    locally_relevant_solution_v,
                                    ExactSolution_and_BC_U<dim>(time, 1),
                                    difference_per_cell,
                                    QGauss<dim>(degree_U + 1),
                                    VectorTools::L2_norm);
  double v_L2_error = difference_per_cell.l2_norm();
  v_L2_error =
    std::sqrt(Utilities::MPI::sum(v_L2_error * v_L2_error, mpi_communicator));
  VectorTools::integrate_difference(dof_handler_U,
                                    locally_relevant_solution_v,
                                    ExactSolution_and_BC_U<dim>(time, 1),
                                    difference_per_cell,
                                    QGauss<dim>(degree_U + 1),
                                    VectorTools::H1_norm);
  double v_H1_error = difference_per_cell.l2_norm();
  v_H1_error =
    std::sqrt(Utilities::MPI::sum(v_H1_error * v_H1_error, mpi_communicator));
  // error for w
  double w_L2_error = 0;
  double w_H1_error = 0;
  if (dim == 3)
    {
      VectorTools::integrate_difference(dof_handler_U,
                                        locally_relevant_solution_w,
                                        ExactSolution_and_BC_U<dim>(time, 2),
                                        difference_per_cell,
                                        QGauss<dim>(degree_U + 1),
                                        VectorTools::L2_norm);
      w_L2_error = difference_per_cell.l2_norm();
      w_L2_error = std::sqrt(
        Utilities::MPI::sum(w_L2_error * w_L2_error, mpi_communicator));
      VectorTools::integrate_difference(dof_handler_U,
                                        locally_relevant_solution_w,
                                        ExactSolution_and_BC_U<dim>(time, 2),
                                        difference_per_cell,
                                        QGauss<dim>(degree_U + 1),
                                        VectorTools::H1_norm);
      w_H1_error = difference_per_cell.l2_norm();
      w_H1_error = std::sqrt(
        Utilities::MPI::sum(w_H1_error * w_H1_error, mpi_communicator));
    }
  // error for p
  VectorTools::integrate_difference(dof_handler_P,
                                    locally_relevant_solution_p,
                                    ExactSolution_p<dim>(time),
                                    difference_per_cell,
                                    QGauss<dim>(degree_U + 1),
                                    VectorTools::L2_norm);
  double p_L2_error = difference_per_cell.l2_norm();
  p_L2_error =
    std::sqrt(Utilities::MPI::sum(p_L2_error * p_L2_error, mpi_communicator));
  VectorTools::integrate_difference(dof_handler_P,
                                    locally_relevant_solution_p,
                                    ExactSolution_p<dim>(time),
                                    difference_per_cell,
                                    QGauss<dim>(degree_U + 1),
                                    VectorTools::H1_norm);
  double p_H1_error = difference_per_cell.l2_norm();
  p_H1_error =
    std::sqrt(Utilities::MPI::sum(p_H1_error * p_H1_error, mpi_communicator));

  const unsigned int n_active_cells = triangulation.n_active_cells();
  const unsigned int n_dofs_U       = dof_handler_U.n_dofs();
  const unsigned int n_dofs_P       = dof_handler_P.n_dofs();

  convergence_table.add_value("cycle", cycle);
  convergence_table.add_value("cells", n_active_cells);
  convergence_table.add_value("dofs_U", n_dofs_U);
  convergence_table.add_value("dofs_P", n_dofs_P);
  convergence_table.add_value("dt", time_step);
  convergence_table.add_value("u L2", u_L2_error);
  convergence_table.add_value("u H1", u_H1_error);
  convergence_table.add_value("v L2", v_L2_error);
  convergence_table.add_value("v H1", v_H1_error);
  if (dim == 3)
    {
      convergence_table.add_value("w L2", w_L2_error);
      convergence_table.add_value("w H1", w_H1_error);
    }
  convergence_table.add_value("p L2", p_L2_error);
  convergence_table.add_value("p H1", p_H1_error);
}

template <int dim>
void
TestNavierStokes<dim>::get_boundary_values_U(double t)
{
  std::map<unsigned int, double> map_boundary_values_u;
  std::map<unsigned int, double> map_boundary_values_v;

  VectorTools::interpolate_boundary_values(dof_handler_U,
                                           0,
                                           ExactSolution_and_BC_U<dim>(t, 0),
                                           map_boundary_values_u);
  VectorTools::interpolate_boundary_values(dof_handler_U,
                                           0,
                                           ExactSolution_and_BC_U<dim>(t, 1),
                                           map_boundary_values_v);

  boundary_values_id_u.resize(map_boundary_values_u.size());
  boundary_values_id_v.resize(map_boundary_values_v.size());
  boundary_values_u.resize(map_boundary_values_u.size());
  boundary_values_v.resize(map_boundary_values_v.size());
  std::map<unsigned int, double>::const_iterator boundary_value_u =
    map_boundary_values_u.begin();
  std::map<unsigned int, double>::const_iterator boundary_value_v =
    map_boundary_values_v.begin();
  if (dim == 3)
    {
      std::map<unsigned int, double> map_boundary_values_w;
      VectorTools::interpolate_boundary_values(dof_handler_U,
                                               0,
                                               ExactSolution_and_BC_U<dim>(t,
                                                                           2),
                                               map_boundary_values_w);
      boundary_values_id_w.resize(map_boundary_values_w.size());
      boundary_values_w.resize(map_boundary_values_w.size());
      std::map<unsigned int, double>::const_iterator boundary_value_w =
        map_boundary_values_w.begin();
      for (int i = 0; boundary_value_w != map_boundary_values_w.end();
           ++boundary_value_w, ++i)
        {
          boundary_values_id_w[i] = boundary_value_w->first;
          boundary_values_w[i]    = boundary_value_w->second;
        }
    }
  for (int i = 0; boundary_value_u != map_boundary_values_u.end();
       ++boundary_value_u, ++i)
    {
      boundary_values_id_u[i] = boundary_value_u->first;
      boundary_values_u[i]    = boundary_value_u->second;
    }
  for (int i = 0; boundary_value_v != map_boundary_values_v.end();
       ++boundary_value_v, ++i)
    {
      boundary_values_id_v[i] = boundary_value_v->first;
      boundary_values_v[i]    = boundary_value_v->second;
    }
}

template <int dim>
void
TestNavierStokes<dim>::run()
{
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::cout << "***** CONVERGENCE TEST FOR NS *****" << std::endl;
      std::cout << "DEGREE LS: " << degree_LS << std::endl;
      std::cout << "DEGREE U:  " << degree_U << std::endl;
    }
  // PARAMETERS FOR THE NAVIER STOKES PROBLEM
  final_time   = 1.0;
  time_step    = 0.1;
  n_cycles     = 6;
  n_refinement = 6;
  ForceTerms<dim>  force_function;
  RhoFunction<dim> rho_function;
  NuFunction<dim>  nu_function;

  output_time     = 0.1;
  output_number   = 0;
  bool get_output = false;
  bool get_error  = true;
  verbose         = true;

  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
    {
      if (cycle == 0)
        {
          GridGenerator::hyper_cube(triangulation);
          triangulation.refine_global(n_refinement);
          setup();
          initial_condition();
        }
      else
        {
          triangulation.refine_global(1);
          setup();
          initial_condition();
          time_step *= 0.5;
        }

      output_results();
      //      if (cycle==0)
      NavierStokesSolver<dim> navier_stokes(degree_LS,
                                            degree_U,
                                            time_step,
                                            force_function,
                                            rho_function,
                                            nu_function,
                                            verbose,
                                            triangulation,
                                            mpi_communicator);
      // set INITIAL CONDITION within TRANSPORT PROBLEM
      if (dim == 2)
        navier_stokes.initial_condition(locally_relevant_solution_rho,
                                        locally_relevant_solution_u,
                                        locally_relevant_solution_v,
                                        locally_relevant_solution_p);
      else // dim=3
        navier_stokes.initial_condition(locally_relevant_solution_rho,
                                        locally_relevant_solution_u,
                                        locally_relevant_solution_v,
                                        locally_relevant_solution_w,
                                        locally_relevant_solution_p);

      pcout << "Cycle " << cycle << ':' << std::endl;
      pcout << "   Cycle   " << cycle << "   Number of active cells:       "
            << triangulation.n_global_active_cells() << std::endl
            << "   Number of degrees of freedom (velocity): "
            << dof_handler_U.n_dofs() << std::endl
            << "   min h="
            << GridTools::minimal_cell_diameter(triangulation) / std::sqrt(2) /
                 degree_U
            << std::endl;

      // TIME STEPPING
      timestep_number         = 0;
      time                    = 0;
      double time_step_backup = time_step;
      while (time < final_time)
        {
          timestep_number++;
          ///////////////////
          // GET TIME_STEP //
          ///////////////////
          if (time + time_step > final_time - 1E-10)
            {
              pcout << "FINAL TIME STEP..." << std::endl;
              time_step_backup = time_step;
              time_step        = final_time - time;
            }
          pcout << "Time step " << timestep_number << "\twith dt=" << time_step
                << "\tat tn=" << time << std::endl;
          /////////////////
          // FORCE TERMS //
          /////////////////
          force_function.set_time(time + time_step);
          /////////////////////////////////
          // DENSITY AND VISCOSITY FIELD //
          /////////////////////////////////
          rho_function.set_time(time + time_step);
          nu_function.set_time(time + time_step);
          /////////////////////////
          // BOUNDARY CONDITIONS //
          /////////////////////////
          get_boundary_values_U(time + time_step);
          if (dim == 2)
            navier_stokes.set_boundary_conditions(boundary_values_id_u,
                                                  boundary_values_id_v,
                                                  boundary_values_u,
                                                  boundary_values_v);
          else
            navier_stokes.set_boundary_conditions(boundary_values_id_u,
                                                  boundary_values_id_v,
                                                  boundary_values_id_w,
                                                  boundary_values_u,
                                                  boundary_values_v,
                                                  boundary_values_w);
          //////////////////
          // GET SOLUTION //
          //////////////////
          navier_stokes.nth_time_step();
          if (dim == 2)
            navier_stokes.get_velocity(locally_relevant_solution_u,
                                       locally_relevant_solution_v);
          else
            navier_stokes.get_velocity(locally_relevant_solution_u,
                                       locally_relevant_solution_v,
                                       locally_relevant_solution_w);
          navier_stokes.get_pressure(locally_relevant_solution_p);

          //////////////////
          // FIX PRESSURE //
          //////////////////
          fix_pressure();

          /////////////////
          // UPDATE TIME //
          /////////////////
          time += time_step;

          ////////////
          // OUTPUT //
          ////////////
          if (get_output && time - (output_number)*output_time >= 1E-10)
            output_results();
        }
      pcout << "FINAL TIME: " << time << std::endl;
      time_step = time_step_backup;
      if (get_error)
        process_solution(cycle);

      if (get_error)
        {
          convergence_table.set_precision("u L2", 2);
          convergence_table.set_precision("u H1", 2);
          convergence_table.set_scientific("u L2", true);
          convergence_table.set_scientific("u H1", true);

          convergence_table.set_precision("v L2", 2);
          convergence_table.set_precision("v H1", 2);
          convergence_table.set_scientific("v L2", true);
          convergence_table.set_scientific("v H1", true);

          if (dim == 3)
            {
              convergence_table.set_precision("w L2", 2);
              convergence_table.set_precision("w H1", 2);
              convergence_table.set_scientific("w L2", true);
              convergence_table.set_scientific("w H1", true);
            }

          convergence_table.set_precision("p L2", 2);
          convergence_table.set_precision("p H1", 2);
          convergence_table.set_scientific("p L2", true);
          convergence_table.set_scientific("p H1", true);

          convergence_table.set_tex_format("cells", "r");
          convergence_table.set_tex_format("dofs_U", "r");
          convergence_table.set_tex_format("dofs_P", "r");
          convergence_table.set_tex_format("dt", "r");

          if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
            {
              std::cout << std::endl;
              convergence_table.write_text(std::cout);
            }
        }
    }
}

int
main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
      deallog.depth_console(0);

      {
        unsigned int        degree_LS = 1;
        unsigned int        degree_U  = 2;
        TestNavierStokes<2> test_navier_stokes(degree_LS, degree_U);
        test_navier_stokes.run();
      }

      PetscFinalize();
    }

  catch (std::exception &exc)
    {
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
    }
  catch (...)
    {
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
