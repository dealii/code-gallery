/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2016 Manuel Quezada de Luna
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/fe_system.h>

#include <fstream>
#include <iostream>
#include <memory>

using namespace dealii;

///////////////////////////
// FOR TRANSPORT PROBLEM //
///////////////////////////
// TIME_INTEGRATION
#define FORWARD_EULER 0
#define SSP33 1
// PROBLEM
#define CIRCULAR_ROTATION 0
#define DIAGONAL_ADVECTION 1
// OTHER FLAGS
#define VARIABLE_VELOCITY 0

#include "utilities_test_LS.cc"
#include "LevelSetSolver.cc"

///////////////////////////////////////////////////////
///////////////////// MAIN CLASS //////////////////////
///////////////////////////////////////////////////////
template <int dim>
class TestLevelSet
{
public:
  TestLevelSet (const unsigned int degree_LS,
                const unsigned int degree_U);
  ~TestLevelSet ();
  void run ();

private:
  // BOUNDARY //
  void set_boundary_inlet();
  void get_boundary_values_phi(std::vector<unsigned int> &boundary_values_id_phi,
                               std::vector<double> &boundary_values_phi);
  // VELOCITY //
  void get_interpolated_velocity();
  // SETUP AND INIT CONDITIONS //
  void setup();
  void initial_condition();
  void init_constraints();
  // POST PROCESSING //
  void process_solution(parallel::distributed::Triangulation<dim> &triangulation,
                        DoFHandler<dim> &dof_handler_LS,
                        PETScWrappers::MPI::Vector &solution);
  void output_results();
  void output_solution();

  // SOLUTION VECTORS
  PETScWrappers::MPI::Vector locally_relevant_solution_phi;
  PETScWrappers::MPI::Vector locally_relevant_solution_u;
  PETScWrappers::MPI::Vector locally_relevant_solution_v;
  PETScWrappers::MPI::Vector locally_relevant_solution_w;
  PETScWrappers::MPI::Vector completely_distributed_solution_phi;
  PETScWrappers::MPI::Vector completely_distributed_solution_u;
  PETScWrappers::MPI::Vector completely_distributed_solution_v;
  PETScWrappers::MPI::Vector completely_distributed_solution_w;
  // BOUNDARY VECTORS
  std::vector<unsigned int> boundary_values_id_phi;
  std::vector<double> boundary_values_phi;

  // GENERAL
  MPI_Comm mpi_communicator;
  parallel::distributed::Triangulation<dim>   triangulation;

  int                  degree;
  int                  degree_LS;
  DoFHandler<dim>      dof_handler_LS;
  FE_Q<dim>            fe_LS;
  IndexSet             locally_owned_dofs_LS;
  IndexSet             locally_relevant_dofs_LS;

  int                  degree_U;
  DoFHandler<dim>      dof_handler_U;
  FE_Q<dim>            fe_U;
  IndexSet             locally_owned_dofs_U;
  IndexSet             locally_relevant_dofs_U;

  DoFHandler<dim>      dof_handler_U_disp_field;
  FESystem<dim>        fe_U_disp_field;
  IndexSet             locally_owned_dofs_U_disp_field;
  IndexSet             locally_relevant_dofs_U_disp_field;

  AffineConstraints<double> constraints;
  AffineConstraints<double> constraints_disp_field;

  double time;
  double time_step;
  double final_time;
  unsigned int timestep_number;
  double cfl;
  double min_h;

  double sharpness;
  int sharpness_integer;

  unsigned int n_refinement;
  unsigned int output_number;
  double output_time;
  bool get_output;

  bool verbose;
  ConditionalOStream pcout;

  //FOR TRANSPORT
  double cK; //compression coeff
  double cE; //entropy-visc coeff
  unsigned int TRANSPORT_TIME_INTEGRATION;
  std::string ALGORITHM;
  unsigned int PROBLEM;

  //FOR RECONSTRUCTION OF MATERIAL FIELDS
  double eps, rho_air, rho_fluid;

  // MASS MATRIX
  PETScWrappers::MPI::SparseMatrix matrix_MC, matrix_MC_tnm1;
  std::shared_ptr<PETScWrappers::PreconditionBoomerAMG> preconditioner_MC;

};

template <int dim>
TestLevelSet<dim>::TestLevelSet (const unsigned int degree_LS,
                                 const unsigned int degree_U)
  :
  mpi_communicator (MPI_COMM_WORLD),
  triangulation (mpi_communicator,
                 typename Triangulation<dim>::MeshSmoothing
                 (Triangulation<dim>::smoothing_on_refinement |
                  Triangulation<dim>::smoothing_on_coarsening)),
  degree_LS(degree_LS),
  dof_handler_LS (triangulation),
  fe_LS (degree_LS),
  degree_U(degree_U),
  dof_handler_U (triangulation),
  fe_U (degree_U),
  dof_handler_U_disp_field(triangulation),
  fe_U_disp_field(FE_Q<dim>(degree_U),dim),
  pcout (std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator)== 0))
{}

template <int dim>
TestLevelSet<dim>::~TestLevelSet ()
{
  dof_handler_U_disp_field.clear();
  dof_handler_LS.clear ();
  dof_handler_U.clear ();
}

// VELOCITY //
//////////////
template <int dim>
void TestLevelSet<dim>::get_interpolated_velocity()
{
  // velocity in x
  completely_distributed_solution_u = 0;
  VectorTools::interpolate(dof_handler_U,
                           ExactU<dim>(PROBLEM,time),
                           completely_distributed_solution_u);
  constraints.distribute (completely_distributed_solution_u);
  locally_relevant_solution_u = completely_distributed_solution_u;
  // velocity in y
  completely_distributed_solution_v = 0;
  VectorTools::interpolate(dof_handler_U,
                           ExactV<dim>(PROBLEM,time),
                           completely_distributed_solution_v);
  constraints.distribute (completely_distributed_solution_v);
  locally_relevant_solution_v = completely_distributed_solution_v;
  if (dim==3)
    {
      completely_distributed_solution_w = 0;
      VectorTools::interpolate(dof_handler_U,
                               ExactW<dim>(PROBLEM,time),
                               completely_distributed_solution_w);
      constraints.distribute (completely_distributed_solution_w);
      locally_relevant_solution_w = completely_distributed_solution_w;
    }
}

//////////////
// BOUNDARY //
//////////////
template <int dim>
void TestLevelSet<dim>::set_boundary_inlet()
{
  const QGauss<dim-1>  face_quadrature_formula(1); // center of the face
  FEFaceValues<dim> fe_face_values (fe_U,face_quadrature_formula,
                                    update_values | update_quadrature_points |
                                    update_normal_vectors);
  const unsigned int n_face_q_points = face_quadrature_formula.size();
  std::vector<double>  u_value (n_face_q_points);
  std::vector<double>  v_value (n_face_q_points);
  std::vector<double>  w_value (n_face_q_points);

  typename DoFHandler<dim>::active_cell_iterator
  cell_U = dof_handler_U.begin_active(),
  endc_U = dof_handler_U.end();
  Tensor<1,dim> u;

  for (; cell_U!=endc_U; ++cell_U)
    if (cell_U->is_locally_owned())
      for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
        if (cell_U->face(face)->at_boundary())
          {
            fe_face_values.reinit(cell_U,face);
            fe_face_values.get_function_values(locally_relevant_solution_u,u_value);
            fe_face_values.get_function_values(locally_relevant_solution_v,v_value);
            if (dim==3)
              fe_face_values.get_function_values(locally_relevant_solution_w,w_value);
            u[0]=u_value[0];
            u[1]=v_value[0];
            if (dim==3)
              u[2]=w_value[0];
            if (fe_face_values.normal_vector(0)*u < -1e-14)
              cell_U->face(face)->set_boundary_id(10);
          }
}

template <int dim>
void TestLevelSet<dim>::get_boundary_values_phi(std::vector<unsigned int> &boundary_values_id_phi,
                                                std::vector<double> &boundary_values_phi)
{
  std::map<unsigned int, double> map_boundary_values_phi;
  unsigned int boundary_id=0;

  set_boundary_inlet();
  boundary_id=10; // inlet
  VectorTools::interpolate_boundary_values (dof_handler_LS,
                                            boundary_id,BoundaryPhi<dim>(),
                                            map_boundary_values_phi);

  boundary_values_id_phi.resize(map_boundary_values_phi.size());
  boundary_values_phi.resize(map_boundary_values_phi.size());
  std::map<unsigned int,double>::const_iterator boundary_value_phi = map_boundary_values_phi.begin();
  for (int i=0; boundary_value_phi !=map_boundary_values_phi.end(); ++boundary_value_phi, ++i)
    {
      boundary_values_id_phi[i]=boundary_value_phi->first;
      boundary_values_phi[i]=boundary_value_phi->second;
    }
}

///////////////////////////////////
// SETUP AND INITIAL CONDITIONS //
//////////////////////////////////
template <int dim>
void TestLevelSet<dim>::setup()
{
  degree = std::max(degree_LS,degree_U);
  // setup system LS
  dof_handler_LS.distribute_dofs (fe_LS);
  locally_owned_dofs_LS    = dof_handler_LS.locally_owned_dofs ();
  locally_relevant_dofs_LS = DoFTools::extract_locally_relevant_dofs (dof_handler_LS);
  // setup system U
  dof_handler_U.distribute_dofs (fe_U);
  locally_owned_dofs_U    = dof_handler_U.locally_owned_dofs ();
  locally_relevant_dofs_U = DoFTools::extract_locally_relevant_dofs (dof_handler_U);
  // setup system U for disp field
  dof_handler_U_disp_field.distribute_dofs (fe_U_disp_field);
  locally_owned_dofs_U_disp_field    = dof_handler_U_disp_field.locally_owned_dofs ();
  locally_relevant_dofs_U_disp_field = DoFTools::extract_locally_relevant_dofs (dof_handler_U_disp_field);
  // init vectors for phi
  locally_relevant_solution_phi.reinit(locally_owned_dofs_LS,
                                       locally_relevant_dofs_LS,
                                       mpi_communicator);
  locally_relevant_solution_phi = 0;
  completely_distributed_solution_phi.reinit(mpi_communicator,
                                             dof_handler_LS.n_dofs(),
                                             dof_handler_LS.n_locally_owned_dofs());
  //init vectors for u
  locally_relevant_solution_u.reinit(locally_owned_dofs_U,
                                     locally_relevant_dofs_U,
                                     mpi_communicator);
  locally_relevant_solution_u = 0;
  completely_distributed_solution_u.reinit(mpi_communicator,
                                           dof_handler_U.n_dofs(),
                                           dof_handler_U.n_locally_owned_dofs());
  //init vectors for v
  locally_relevant_solution_v.reinit(locally_owned_dofs_U,
                                     locally_relevant_dofs_U,
                                     mpi_communicator);
  locally_relevant_solution_v = 0;
  completely_distributed_solution_v.reinit(mpi_communicator,
                                           dof_handler_U.n_dofs(),
                                           dof_handler_U.n_locally_owned_dofs());
  // init vectors for w
  locally_relevant_solution_w.reinit(locally_owned_dofs_U,
                                     locally_relevant_dofs_U,
                                     mpi_communicator);
  locally_relevant_solution_w = 0;
  completely_distributed_solution_w.reinit(mpi_communicator,
                                           dof_handler_U.n_dofs(),
                                           dof_handler_U.n_locally_owned_dofs());
  init_constraints();
  // MASS MATRIX
  DynamicSparsityPattern dsp (locally_relevant_dofs_LS);
  DoFTools::make_sparsity_pattern (dof_handler_LS,dsp,constraints,false);
  SparsityTools::distribute_sparsity_pattern (dsp,
                                              dof_handler_LS.n_locally_owned_dofs_per_processor(),
                                              mpi_communicator,
                                              locally_relevant_dofs_LS);
  matrix_MC.reinit (mpi_communicator,
                    dsp,
                    dof_handler_LS.n_locally_owned_dofs_per_processor(),
                    dof_handler_LS.n_locally_owned_dofs_per_processor(),
                    Utilities::MPI::this_mpi_process(mpi_communicator));
  matrix_MC_tnm1.reinit (mpi_communicator,
                         dsp,
                         dof_handler_LS.n_locally_owned_dofs_per_processor(),
                         dof_handler_LS.n_locally_owned_dofs_per_processor(),
                         Utilities::MPI::this_mpi_process(mpi_communicator));
}

template <int dim>
void TestLevelSet<dim>::initial_condition()
{
  time=0;
  // Initial conditions //
  // init condition for phi
  completely_distributed_solution_phi = 0;
  VectorTools::interpolate(dof_handler_LS,
                           InitialPhi<dim>(PROBLEM, sharpness),
                           //Functions::ZeroFunction<dim>(),
                           completely_distributed_solution_phi);
  constraints.distribute (completely_distributed_solution_phi);
  locally_relevant_solution_phi = completely_distributed_solution_phi;
  // init condition for u=0
  completely_distributed_solution_u = 0;
  VectorTools::interpolate(dof_handler_U,
                           ExactU<dim>(PROBLEM,time),
                           completely_distributed_solution_u);
  constraints.distribute (completely_distributed_solution_u);
  locally_relevant_solution_u = completely_distributed_solution_u;
  // init condition for v
  completely_distributed_solution_v = 0;
  VectorTools::interpolate(dof_handler_U,
                           ExactV<dim>(PROBLEM,time),
                           completely_distributed_solution_v);
  constraints.distribute (completely_distributed_solution_v);
  locally_relevant_solution_v = completely_distributed_solution_v;
}

template <int dim>
void TestLevelSet<dim>::init_constraints()
{
  constraints.clear ();
#if DEAL_II_VERSION_GTE(9, 6, 0)
  constraints.reinit (locally_owned_dofs_LS, locally_relevant_dofs_LS);
#else
  constraints.reinit (locally_relevant_dofs_LS);
#endif
  DoFTools::make_hanging_node_constraints (dof_handler_LS, constraints);
  constraints.close ();
  constraints_disp_field.clear ();
#if DEAL_II_VERSION_GTE(9, 6, 0)
  constraints_disp_field.reinit (locally_owned_dofs_LS, locally_relevant_dofs_LS);
#else
  constraints_disp_field.reinit (locally_relevant_dofs_LS);
#endif
  DoFTools::make_hanging_node_constraints (dof_handler_LS, constraints_disp_field);
  constraints_disp_field.close ();
}

/////////////////////
// POST PROCESSING //
/////////////////////
template <int dim>
void TestLevelSet<dim>::process_solution(parallel::distributed::Triangulation<dim> &triangulation,
                                         DoFHandler<dim> &dof_handler_LS,
                                         PETScWrappers::MPI::Vector &solution)
{
  Vector<double> difference_per_cell (triangulation.n_active_cells());
  // error for phi
  VectorTools::integrate_difference (dof_handler_LS,
                                     solution,
                                     InitialPhi<dim>(PROBLEM,sharpness),
                                     difference_per_cell,
                                     QGauss<dim>(degree_LS+3),
                                     VectorTools::L1_norm);

  double u_L1_error = difference_per_cell.l1_norm();
  u_L1_error = std::sqrt(Utilities::MPI::sum(u_L1_error * u_L1_error, mpi_communicator));

  VectorTools::integrate_difference (dof_handler_LS,
                                     solution,
                                     InitialPhi<dim>(PROBLEM,sharpness),
                                     difference_per_cell,
                                     QGauss<dim>(degree_LS+3),
                                     VectorTools::L2_norm);
  double u_L2_error = difference_per_cell.l2_norm();
  u_L2_error = std::sqrt(Utilities::MPI::sum(u_L2_error * u_L2_error, mpi_communicator));

  pcout << "L1 error: " << u_L1_error << std::endl;
  pcout << "L2 error: " << u_L2_error << std::endl;
}

template<int dim>
void TestLevelSet<dim>::output_results()
{
  output_solution();
  output_number++;
}

template <int dim>
void TestLevelSet<dim>::output_solution()
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler_LS);
  data_out.add_data_vector (locally_relevant_solution_phi, "phi");
  data_out.build_patches();

  const std::string filename = ("solution-" +
                                Utilities::int_to_string (output_number, 3) +
                                "." +
                                Utilities::int_to_string
                                (triangulation.locally_owned_subdomain(), 4));
  std::ofstream output ((filename + ".vtu").c_str());
  data_out.write_vtu (output);

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::vector<std::string> filenames;
      for (unsigned int i=0;
           i<Utilities::MPI::n_mpi_processes(mpi_communicator);
           ++i)
        filenames.push_back ("solution-" +
                             Utilities::int_to_string (output_number, 3) +
                             "." +
                             Utilities::int_to_string (i, 4) +
                             ".vtu");

      std::ofstream master_output ((filename + ".pvtu").c_str());
      data_out.write_pvtu_record (master_output, filenames);
    }
}

template <int dim>
void TestLevelSet<dim>::run()
{
  ////////////////////////
  // GENERAL PARAMETERS //
  ////////////////////////
  cfl=0.1;
  verbose = false;
  get_output = true;
  output_number = 0;
  Timer t;
  n_refinement=6;
  output_time = 0.1;
  final_time = 1.0;
  PROBLEM=CIRCULAR_ROTATION;
  //PROBLEM=DIAGONAL_ADVECTION;
  double umax = 0;
  if (PROBLEM==CIRCULAR_ROTATION)
    umax = std::sqrt(2)*numbers::PI;
  else
    umax = std::sqrt(2);

  //////////////////////////////////////
  // PARAMETERS FOR TRANSPORT PROBLEM //
  //////////////////////////////////////
  cK = 1.0; // compression constant
  cE = 1.0; // entropy viscosity constant
  sharpness_integer=1; //this will be multiplied by min_h
  //TRANSPORT_TIME_INTEGRATION=FORWARD_EULER;
  TRANSPORT_TIME_INTEGRATION=SSP33;
  //ALGORITHM = "MPP_u1";
  ALGORITHM = "NMPP_uH";
  //ALGORITHM = "MPP_uH";

  //////////////
  // GEOMETRY //
  //////////////
  if (PROBLEM==CIRCULAR_ROTATION || PROBLEM==DIAGONAL_ADVECTION)
    GridGenerator::hyper_cube(triangulation);
  //GridGenerator::hyper_rectangle(triangulation, Point<dim>(0.0,0.0), Point<dim>(1.0,1.0), true);
  triangulation.refine_global (n_refinement);

  ///////////
  // SETUP //
  ///////////
  setup();

  // for Reconstruction of MATERIAL FIELDS
  min_h = GridTools::minimal_cell_diameter(triangulation)/std::sqrt(dim)/degree;
  eps=1*min_h; //For reconstruction of density in Navier Stokes
  sharpness=sharpness_integer*min_h; //adjust value of sharpness (for init cond of phi)
  rho_fluid = 1000;
  rho_air = 1;

  // GET TIME STEP //
  time_step = cfl*min_h/umax;

  //////////////////////
  // TRANSPORT SOLVER //
  //////////////////////
  LevelSetSolver<dim> level_set (degree_LS,degree_U,
                                 time_step,cK,cE,
                                 verbose,
                                 ALGORITHM,
                                 TRANSPORT_TIME_INTEGRATION,
                                 triangulation,
                                 mpi_communicator);

  ///////////////////////
  // INITIAL CONDITION //
  ///////////////////////
  initial_condition();
  output_results();
  if (dim==2)
    level_set.initial_condition(locally_relevant_solution_phi,
                                locally_relevant_solution_u,locally_relevant_solution_v);
  else //dim=3
    level_set.initial_condition(locally_relevant_solution_phi,
                                locally_relevant_solution_u,locally_relevant_solution_v,locally_relevant_solution_w);

  /////////////////////////////////
  // BOUNDARY CONDITIONS FOR PHI //
  /////////////////////////////////
  get_boundary_values_phi(boundary_values_id_phi,boundary_values_phi);
  level_set.set_boundary_conditions(boundary_values_id_phi,boundary_values_phi);

  // OUTPUT DATA REGARDING TIME STEPPING AND MESH //
  int dofs_LS = dof_handler_LS.n_dofs();
  pcout << "Cfl: " << cfl << std::endl;
  pcout << "   Number of active cells:       "
        << triangulation.n_global_active_cells() << std::endl
        << "   Number of degrees of freedom: " << std::endl
        << "      LS: " << dofs_LS << std::endl;

  // TIME STEPPING
  timestep_number=0;
  time=0;
  while (time<final_time)
    {
      timestep_number++;
      if (time+time_step > final_time)
        {
          pcout << "FINAL TIME STEP... " << std::endl;
          time_step = final_time-time;
        }
      pcout << "Time step " << timestep_number
            << "\twith dt=" << time_step
            << "\tat tn=" << time << std::endl;

      //////////////////
      // GET VELOCITY // (NS or interpolate from a function) at current time tn
      //////////////////
      if (VARIABLE_VELOCITY)
        {
          get_interpolated_velocity();
          // SET VELOCITY TO LEVEL SET SOLVER
          level_set.set_velocity(locally_relevant_solution_u,locally_relevant_solution_v);
        }
      ////////////////////////////
      // GET LEVEL SET SOLUTION // (at tnp1)
      ////////////////////////////
      level_set.nth_time_step();

      /////////////////
      // UPDATE TIME //
      /////////////////
      time+=time_step; // time tnp1

      ////////////
      // OUTPUT //
      ////////////
      if (get_output && time-(output_number)*output_time>=0)
        {
          level_set.get_unp1(locally_relevant_solution_phi);
          output_results();
        }
    }
  pcout << "FINAL TIME T=" << time << std::endl;
}

int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
      deallog.depth_console (0);
      {
        unsigned int degree = 1;
        TestLevelSet<2> multiphase(degree, degree);
        multiphase.run();
      }
      PetscFinalize();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
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
      std::cerr << std::endl << std::endl
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


