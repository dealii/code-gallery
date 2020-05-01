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

#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <mpi.h>

#include <fstream>
#include <iostream>

using namespace dealii;

// FLAGS
#define NUM_ITER 1
#define CHECK_MAX_PRINCIPLE 0

// LOG FOR LEVEL SET FROM -1 to 1
#define ENTROPY(phi) std::log(std::abs(1 - phi * phi) + 1E-14)
#define ENTROPY_GRAD(phi, phix)                    \
  2 * phi *phix *((1 - phi * phi >= 0) ? -1 : 1) / \
    (std::abs(1 - phi * phi) + 1E-14)

//////////////////////////////////////////////////////////
//////////////////// TRANSPORT SOLVER ////////////////////
//////////////////////////////////////////////////////////
// This is a solver for the transpor solver.
// We assume the velocity is divergence free
// and solve the equation in conservation form.
///////////////////////////////////
//---------- NOTATION ---------- //
///////////////////////////////////
// We use notation popular in the literature of conservation laws.
// For this reason the solution is denoted as u, unm1, unp1, etc.
// and the velocity is treated as vx, vy and vz.
template <int dim>
class LevelSetSolver
{
public:
  ////////////////////////
  // INITIAL CONDITIONS //
  ////////////////////////
  void
  initial_condition(PETScWrappers::MPI::Vector locally_relevant_solution_u,
                    PETScWrappers::MPI::Vector locally_relevant_solution_vx,
                    PETScWrappers::MPI::Vector locally_relevant_solution_vy);
  void
  initial_condition(PETScWrappers::MPI::Vector locally_relevant_solution_u,
                    PETScWrappers::MPI::Vector locally_relevant_solution_vx,
                    PETScWrappers::MPI::Vector locally_relevant_solution_vy,
                    PETScWrappers::MPI::Vector locally_relevant_solution_vz);
  /////////////////////////
  // BOUNDARY CONDITIONS //
  /////////////////////////
  void
  set_boundary_conditions(
    std::vector<types::global_dof_index> &boundary_values_id_u,
    std::vector<double>                   boundary_values_u);
  //////////////////
  // SET VELOCITY //
  //////////////////
  void
  set_velocity(PETScWrappers::MPI::Vector locally_relevant_solution_vx,
               PETScWrappers::MPI::Vector locally_relevant_solution_vy);
  void
  set_velocity(PETScWrappers::MPI::Vector locally_relevant_solution_vx,
               PETScWrappers::MPI::Vector locally_relevant_solution_vy,
               PETScWrappers::MPI::Vector locally_relevant_solution_vz);
  ///////////////////////
  // SET AND GET ALPHA //
  ///////////////////////
  void
  get_unp1(PETScWrappers::MPI::Vector &locally_relevant_solution_u);
  ///////////////////
  // NTH TIME STEP //
  ///////////////////
  void
  nth_time_step();
  ///////////
  // SETUP //
  ///////////
  void
  setup();

  LevelSetSolver(const unsigned int                         degree_LS,
                 const unsigned int                         degree_U,
                 const double                               time_step,
                 const double                               cK,
                 const double                               cE,
                 const bool                                 verbose,
                 std::string                                ALGORITHM,
                 const unsigned int                         TIME_INTEGRATION,
                 parallel::distributed::Triangulation<dim> &triangulation,
                 MPI_Comm &                                 mpi_communicator);
  ~LevelSetSolver();

private:
  ////////////////////////////////////////
  // ASSEMBLE MASS (and other) MATRICES //
  ////////////////////////////////////////
  void
  assemble_ML();
  void
  invert_ML();
  void
  assemble_MC();
  //////////////////////////////////////
  // LOW ORDER METHOD (DiJ Viscosity) //
  //////////////////////////////////////
  void
  assemble_C_Matrix();
  void
  assemble_K_times_vector(PETScWrappers::MPI::Vector &solution);
  void
  assemble_K_DL_DH_times_vector(PETScWrappers::MPI::Vector &solution);
  ///////////////////////
  // ENTROPY VISCOSITY //
  ///////////////////////
  void
  assemble_EntRes_Matrix();
  ///////////////////////////
  // FOR MAXIMUM PRINCIPLE //
  ///////////////////////////
  void
  compute_bounds(PETScWrappers::MPI::Vector &un_solution);
  void
  check_max_principle(PETScWrappers::MPI::Vector &unp1_solution);
  ///////////////////////
  // COMPUTE SOLUTIONS //
  ///////////////////////
  void
  compute_MPP_uL_and_NMPP_uH(PETScWrappers::MPI::Vector &MPP_uL_solution,
                             PETScWrappers::MPI::Vector &NMPP_uH_solution,
                             PETScWrappers::MPI::Vector &un_solution);
  void
  compute_MPP_uH(PETScWrappers::MPI::Vector &MPP_uH_solution,
                 PETScWrappers::MPI::Vector &MPP_uL_solution_ghosted,
                 PETScWrappers::MPI::Vector &NMPP_uH_solution_ghosted,
                 PETScWrappers::MPI::Vector &un_solution);
  void
  compute_MPP_uH_with_iterated_FCT(
    PETScWrappers::MPI::Vector &MPP_uH_solution,
    PETScWrappers::MPI::Vector &MPP_uL_solution_ghosted,
    PETScWrappers::MPI::Vector &NMPP_uH_solution_ghosted,
    PETScWrappers::MPI::Vector &un_solution);
  void
  compute_solution(PETScWrappers::MPI::Vector &unp1,
                   PETScWrappers::MPI::Vector &un,
                   std::string                 algorithm);
  void
  compute_solution_SSP33(PETScWrappers::MPI::Vector &unp1,
                         PETScWrappers::MPI::Vector &un,
                         std::string                 algorithm);
  ///////////////
  // UTILITIES //
  ///////////////
  void
  get_sparsity_pattern();
  void
  get_map_from_Q1_to_Q2();
  void
  solve(
    const ConstraintMatrix &                                    constraints,
    PETScWrappers::MPI::SparseMatrix &                          Matrix,
    std_cxx1x::shared_ptr<PETScWrappers::PreconditionBoomerAMG> preconditioner,
    PETScWrappers::MPI::Vector &      completely_distributed_solution,
    const PETScWrappers::MPI::Vector &rhs);
  void
  save_old_solution();
  void
  save_old_vel_solution();
  ///////////////////////
  // MY PETSC WRAPPERS //
  ///////////////////////
  void
  get_vector_values(PETScWrappers::VectorBase &                 vector,
                    const std::vector<types::global_dof_index> &indices,
                    std::vector<PetscScalar> &                  values);
  void
  get_vector_values(PETScWrappers::VectorBase &                 vector,
                    const std::vector<types::global_dof_index> &indices,
                    std::map<types::global_dof_index, types::global_dof_index>
                      &                       map_from_Q1_to_Q2,
                    std::vector<PetscScalar> &values);

  MPI_Comm mpi_communicator;

  // FINITE ELEMENT SPACE
  int             degree_MAX;
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

  // OPERATORS times SOLUTION VECTOR //
  PETScWrappers::MPI::Vector K_times_solution;
  PETScWrappers::MPI::Vector DL_times_solution;
  PETScWrappers::MPI::Vector DH_times_solution;

  // MASS MATRIX
  PETScWrappers::MPI::SparseMatrix                            MC_matrix;
  std_cxx1x::shared_ptr<PETScWrappers::PreconditionBoomerAMG> MC_preconditioner;

  // BOUNDARIES
  std::vector<types::global_dof_index> boundary_values_id_u;
  std::vector<double>                  boundary_values_u;

  //////////////
  // MATRICES //
  //////////////
  // FOR FIRST ORDER VISCOSITY
  PETScWrappers::MPI::SparseMatrix Cx_matrix, CTx_matrix, Cy_matrix, CTy_matrix,
    Cz_matrix, CTz_matrix;
  PETScWrappers::MPI::SparseMatrix dLij_matrix;
  // FOR ENTROPY VISCOSITY
  PETScWrappers::MPI::SparseMatrix EntRes_matrix, SuppSize_matrix, dCij_matrix;
  // FOR FCT (flux and limited flux)
  PETScWrappers::MPI::SparseMatrix A_matrix, LxA_matrix;
  // FOR ITERATIVE FCT
  PETScWrappers::MPI::SparseMatrix Akp1_matrix, LxAkp1_matrix;

  // GHOSTED VECTORS
  PETScWrappers::MPI::Vector uStage1, uStage2;
  PETScWrappers::MPI::Vector unm1, un;
  PETScWrappers::MPI::Vector R_pos_vector, R_neg_vector;
  PETScWrappers::MPI::Vector MPP_uL_solution_ghosted,
    MPP_uLkp1_solution_ghosted, NMPP_uH_solution_ghosted;
  PETScWrappers::MPI::Vector locally_relevant_solution_vx;
  PETScWrappers::MPI::Vector locally_relevant_solution_vy;
  PETScWrappers::MPI::Vector locally_relevant_solution_vz;
  PETScWrappers::MPI::Vector locally_relevant_solution_vx_old;
  PETScWrappers::MPI::Vector locally_relevant_solution_vy_old;
  PETScWrappers::MPI::Vector locally_relevant_solution_vz_old;

  // NON-GHOSTED VECTORS
  PETScWrappers::MPI::Vector uStage1_nonGhosted, uStage2_nonGhosted;
  PETScWrappers::MPI::Vector unp1;
  PETScWrappers::MPI::Vector R_pos_vector_nonGhosted, R_neg_vector_nonGhosted;
  PETScWrappers::MPI::Vector umin_vector, umax_vector;
  PETScWrappers::MPI::Vector MPP_uL_solution, NMPP_uH_solution, MPP_uH_solution;
  PETScWrappers::MPI::Vector RHS;

  // LUMPED MASS MATRIX
  PETScWrappers::MPI::Vector ML_vector, ones_vector;
  PETScWrappers::MPI::Vector inverse_ML_vector;

  // CONSTRAINTS
  ConstraintMatrix constraints;

  // TIME STEPPING
  double time_step;

  // SOME PARAMETERS
  double cE, cK;
  double solver_tolerance;
  double entropy_normalization_factor;

  // UTILITIES
  bool         verbose;
  std::string  ALGORITHM;
  unsigned int TIME_INTEGRATION;

  ConditionalOStream pcout;

  std::map<types::global_dof_index, types::global_dof_index> map_from_Q1_to_Q2;
  std::map<types::global_dof_index, std::vector<types::global_dof_index>>
    sparsity_pattern;
};

template <int dim>
LevelSetSolver<dim>::LevelSetSolver(
  const unsigned int                         degree_LS,
  const unsigned int                         degree_U,
  const double                               time_step,
  const double                               cK,
  const double                               cE,
  const bool                                 verbose,
  std::string                                ALGORITHM,
  const unsigned int                         TIME_INTEGRATION,
  parallel::distributed::Triangulation<dim> &triangulation,
  MPI_Comm &                                 mpi_communicator)
  : mpi_communicator(mpi_communicator)
  , degree_LS(degree_LS)
  , dof_handler_LS(triangulation)
  , fe_LS(degree_LS)
  , degree_U(degree_U)
  , dof_handler_U(triangulation)
  , fe_U(degree_U)
  , time_step(time_step)
  , cE(cE)
  , cK(cK)
  , verbose(verbose)
  , ALGORITHM(ALGORITHM)
  , TIME_INTEGRATION(TIME_INTEGRATION)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
{
  pcout << "********** LEVEL SET SETUP **********" << std::endl;
  setup();
}

template <int dim>
LevelSetSolver<dim>::~LevelSetSolver()
{
  dof_handler_LS.clear();
  dof_handler_U.clear();
}

///////////////////////////////////////////////////////////
///////////////////// PUBLIC FUNCTIONS ////////////////////
///////////////////////////////////////////////////////////
////////////////////////////////////////
////////// INITIAL CONDITIONS //////////
////////////////////////////////////////
template <int dim>
void
LevelSetSolver<dim>::initial_condition(
  PETScWrappers::MPI::Vector un,
  PETScWrappers::MPI::Vector locally_relevant_solution_vx,
  PETScWrappers::MPI::Vector locally_relevant_solution_vy)
{
  this->un                           = un;
  this->locally_relevant_solution_vx = locally_relevant_solution_vx;
  this->locally_relevant_solution_vy = locally_relevant_solution_vy;
  // initialize old vectors with current solution, this just happens the first
  // time
  unm1                             = un;
  locally_relevant_solution_vx_old = locally_relevant_solution_vx;
  locally_relevant_solution_vy_old = locally_relevant_solution_vy;
}

template <int dim>
void
LevelSetSolver<dim>::initial_condition(
  PETScWrappers::MPI::Vector un,
  PETScWrappers::MPI::Vector locally_relevant_solution_vx,
  PETScWrappers::MPI::Vector locally_relevant_solution_vy,
  PETScWrappers::MPI::Vector locally_relevant_solution_vz)
{
  this->un                           = un;
  this->locally_relevant_solution_vx = locally_relevant_solution_vx;
  this->locally_relevant_solution_vy = locally_relevant_solution_vy;
  this->locally_relevant_solution_vz = locally_relevant_solution_vz;
  // initialize old vectors with current solution, this just happens the first
  // time
  unm1                             = un;
  locally_relevant_solution_vx_old = locally_relevant_solution_vx;
  locally_relevant_solution_vy_old = locally_relevant_solution_vy;
  locally_relevant_solution_vz_old = locally_relevant_solution_vz;
}

/////////////////////////////////////////
////////// BOUNDARY CONDITIONS //////////
/////////////////////////////////////////
template <int dim>
void
LevelSetSolver<dim>::set_boundary_conditions(
  std::vector<types::global_dof_index> &boundary_values_id_u,
  std::vector<double>                   boundary_values_u)
{
  this->boundary_values_id_u = boundary_values_id_u;
  this->boundary_values_u    = boundary_values_u;
}

//////////////////////////////////
////////// SET VELOCITY //////////
//////////////////////////////////
template <int dim>
void
LevelSetSolver<dim>::set_velocity(
  PETScWrappers::MPI::Vector locally_relevant_solution_vx,
  PETScWrappers::MPI::Vector locally_relevant_solution_vy)
{
  // SAVE OLD SOLUTION
  save_old_vel_solution();
  // update velocity
  this->locally_relevant_solution_vx = locally_relevant_solution_vx;
  this->locally_relevant_solution_vy = locally_relevant_solution_vy;
}

template <int dim>
void
LevelSetSolver<dim>::set_velocity(
  PETScWrappers::MPI::Vector locally_relevant_solution_vx,
  PETScWrappers::MPI::Vector locally_relevant_solution_vy,
  PETScWrappers::MPI::Vector locally_relevant_solution_vz)
{
  // SAVE OLD SOLUTION
  save_old_vel_solution();
  // update velocity
  this->locally_relevant_solution_vx = locally_relevant_solution_vx;
  this->locally_relevant_solution_vy = locally_relevant_solution_vy;
  this->locally_relevant_solution_vz = locally_relevant_solution_vz;
}

///////////////////////////////////
////////// SET AND GET U //////////
///////////////////////////////////
template <int dim>
void
LevelSetSolver<dim>::get_unp1(PETScWrappers::MPI::Vector &unp1)
{
  unp1 = this->unp1;
}

// -------------------------------------------------------------------------------
// //
// ------------------------------ COMPUTE SOLUTIONS
// ------------------------------ //
// -------------------------------------------------------------------------------
// //
template <int dim>
void
LevelSetSolver<dim>::nth_time_step()
{
  assemble_EntRes_Matrix();
  // COMPUTE SOLUTION //
  if (TIME_INTEGRATION == FORWARD_EULER)
    compute_solution(unp1, un, ALGORITHM);
  else
    compute_solution_SSP33(unp1, un, ALGORITHM);
  // BOUNDARY CONDITIONS
  unp1.set(boundary_values_id_u, boundary_values_u);
  unp1.compress(VectorOperation::insert);
  // CHECK MAXIMUM PRINCIPLE
  if (CHECK_MAX_PRINCIPLE)
    {
      compute_bounds(un);
      check_max_principle(unp1);
    }
  // pcout <<
  // "*********************************************************************... "
  //  << unp1.min() << ", " << unp1.max() << std::endl;
  save_old_solution();
}

// --------------------------------------------------------------------//
// ------------------------------ SETUP ------------------------------ //
// --------------------------------------------------------------------//
template <int dim>
void
LevelSetSolver<dim>::setup()
{
  solver_tolerance = 1E-6;
  degree_MAX       = std::max(degree_LS, degree_U);
  ////////////////////////////
  // SETUP FOR DOF HANDLERS //
  ////////////////////////////
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
  //////////////////////
  // INIT CONSTRAINTS //
  //////////////////////
  constraints.clear();
  constraints.reinit(locally_relevant_dofs_LS);
  DoFTools::make_hanging_node_constraints(dof_handler_LS, constraints);
  constraints.close();
  /////////////////////////
  // NON-GHOSTED VECTORS //
  /////////////////////////
  MPP_uL_solution.reinit(locally_owned_dofs_LS, mpi_communicator);
  NMPP_uH_solution.reinit(locally_owned_dofs_LS, mpi_communicator);
  RHS.reinit(locally_owned_dofs_LS, mpi_communicator);
  uStage1_nonGhosted.reinit(locally_owned_dofs_LS, mpi_communicator);
  uStage2_nonGhosted.reinit(locally_owned_dofs_LS, mpi_communicator);
  unp1.reinit(locally_owned_dofs_LS, mpi_communicator);
  MPP_uH_solution.reinit(locally_owned_dofs_LS, mpi_communicator);
  // vectors for lumped mass matrix
  ML_vector.reinit(locally_owned_dofs_LS, mpi_communicator);
  inverse_ML_vector.reinit(locally_owned_dofs_LS, mpi_communicator);
  ones_vector.reinit(locally_owned_dofs_LS, mpi_communicator);
  ones_vector = 1.;
  // operators times solution
  K_times_solution.reinit(locally_owned_dofs_LS, mpi_communicator);
  DL_times_solution.reinit(locally_owned_dofs_LS, mpi_communicator);
  DH_times_solution.reinit(locally_owned_dofs_LS, mpi_communicator);
  // LIMITERS (FCT)
  R_pos_vector_nonGhosted.reinit(locally_owned_dofs_LS, mpi_communicator);
  R_neg_vector_nonGhosted.reinit(locally_owned_dofs_LS, mpi_communicator);
  umin_vector.reinit(locally_owned_dofs_LS, mpi_communicator);
  umax_vector.reinit(locally_owned_dofs_LS, mpi_communicator);
  /////////////////////////////////////////////////////////
  // GHOSTED VECTORS (used within some assemble process) //
  /////////////////////////////////////////////////////////
  uStage1.reinit(locally_owned_dofs_LS,
                 locally_relevant_dofs_LS,
                 mpi_communicator);
  uStage2.reinit(locally_owned_dofs_LS,
                 locally_relevant_dofs_LS,
                 mpi_communicator);
  unm1.reinit(locally_owned_dofs_LS,
              locally_relevant_dofs_LS,
              mpi_communicator);
  un.reinit(locally_owned_dofs_LS, locally_relevant_dofs_LS, mpi_communicator);
  MPP_uL_solution_ghosted.reinit(locally_owned_dofs_LS,
                                 locally_relevant_dofs_LS,
                                 mpi_communicator);
  MPP_uLkp1_solution_ghosted.reinit(locally_owned_dofs_LS,
                                    locally_relevant_dofs_LS,
                                    mpi_communicator);
  NMPP_uH_solution_ghosted.reinit(locally_owned_dofs_LS,
                                  locally_relevant_dofs_LS,
                                  mpi_communicator);
  // init vectors for vx
  locally_relevant_solution_vx.reinit(locally_owned_dofs_U,
                                      locally_relevant_dofs_U,
                                      mpi_communicator);
  locally_relevant_solution_vx_old.reinit(locally_owned_dofs_U,
                                          locally_relevant_dofs_U,
                                          mpi_communicator);
  // init vectors for vy
  locally_relevant_solution_vy.reinit(locally_owned_dofs_U,
                                      locally_relevant_dofs_U,
                                      mpi_communicator);
  locally_relevant_solution_vy_old.reinit(locally_owned_dofs_U,
                                          locally_relevant_dofs_U,
                                          mpi_communicator);
  // init vectors for vz
  locally_relevant_solution_vz.reinit(locally_owned_dofs_U,
                                      locally_relevant_dofs_U,
                                      mpi_communicator);
  locally_relevant_solution_vz_old.reinit(locally_owned_dofs_U,
                                          locally_relevant_dofs_U,
                                          mpi_communicator);
  // LIMITERS (FCT)
  R_pos_vector.reinit(locally_owned_dofs_LS,
                      locally_relevant_dofs_LS,
                      mpi_communicator);
  R_neg_vector.reinit(locally_owned_dofs_LS,
                      locally_relevant_dofs_LS,
                      mpi_communicator);
  ////////////////////
  // SETUP MATRICES //
  ////////////////////
  // MATRICES
  DynamicSparsityPattern dsp(locally_relevant_dofs_LS);
  DoFTools::make_sparsity_pattern(dof_handler_LS, dsp, constraints, false);
  SparsityTools::distribute_sparsity_pattern(
    dsp,
    dof_handler_LS.n_locally_owned_dofs_per_processor(),
    mpi_communicator,
    locally_relevant_dofs_LS);
  MC_matrix.reinit(mpi_communicator,
                   dsp,
                   dof_handler_LS.n_locally_owned_dofs_per_processor(),
                   dof_handler_LS.n_locally_owned_dofs_per_processor(),
                   Utilities::MPI::this_mpi_process(mpi_communicator));
  Cx_matrix.reinit(mpi_communicator,
                   dsp,
                   dof_handler_LS.n_locally_owned_dofs_per_processor(),
                   dof_handler_LS.n_locally_owned_dofs_per_processor(),
                   Utilities::MPI::this_mpi_process(mpi_communicator));
  CTx_matrix.reinit(mpi_communicator,
                    dsp,
                    dof_handler_LS.n_locally_owned_dofs_per_processor(),
                    dof_handler_LS.n_locally_owned_dofs_per_processor(),
                    Utilities::MPI::this_mpi_process(mpi_communicator));
  Cy_matrix.reinit(mpi_communicator,
                   dsp,
                   dof_handler_LS.n_locally_owned_dofs_per_processor(),
                   dof_handler_LS.n_locally_owned_dofs_per_processor(),
                   Utilities::MPI::this_mpi_process(mpi_communicator));
  CTy_matrix.reinit(mpi_communicator,
                    dsp,
                    dof_handler_LS.n_locally_owned_dofs_per_processor(),
                    dof_handler_LS.n_locally_owned_dofs_per_processor(),
                    Utilities::MPI::this_mpi_process(mpi_communicator));
  if (dim == 3)
    {
      Cz_matrix.reinit(mpi_communicator,
                       dsp,
                       dof_handler_LS.n_locally_owned_dofs_per_processor(),
                       dof_handler_LS.n_locally_owned_dofs_per_processor(),
                       Utilities::MPI::this_mpi_process(mpi_communicator));
      CTz_matrix.reinit(mpi_communicator,
                        dsp,
                        dof_handler_LS.n_locally_owned_dofs_per_processor(),
                        dof_handler_LS.n_locally_owned_dofs_per_processor(),
                        Utilities::MPI::this_mpi_process(mpi_communicator));
    }
  dLij_matrix.reinit(mpi_communicator,
                     dsp,
                     dof_handler_LS.n_locally_owned_dofs_per_processor(),
                     dof_handler_LS.n_locally_owned_dofs_per_processor(),
                     Utilities::MPI::this_mpi_process(mpi_communicator));
  EntRes_matrix.reinit(mpi_communicator,
                       dsp,
                       dof_handler_LS.n_locally_owned_dofs_per_processor(),
                       dof_handler_LS.n_locally_owned_dofs_per_processor(),
                       Utilities::MPI::this_mpi_process(mpi_communicator));
  SuppSize_matrix.reinit(mpi_communicator,
                         dsp,
                         dof_handler_LS.n_locally_owned_dofs_per_processor(),
                         dof_handler_LS.n_locally_owned_dofs_per_processor(),
                         Utilities::MPI::this_mpi_process(mpi_communicator));
  dCij_matrix.reinit(mpi_communicator,
                     dsp,
                     dof_handler_LS.n_locally_owned_dofs_per_processor(),
                     dof_handler_LS.n_locally_owned_dofs_per_processor(),
                     Utilities::MPI::this_mpi_process(mpi_communicator));
  A_matrix.reinit(mpi_communicator,
                  dsp,
                  dof_handler_LS.n_locally_owned_dofs_per_processor(),
                  dof_handler_LS.n_locally_owned_dofs_per_processor(),
                  Utilities::MPI::this_mpi_process(mpi_communicator));
  LxA_matrix.reinit(mpi_communicator,
                    dsp,
                    dof_handler_LS.n_locally_owned_dofs_per_processor(),
                    dof_handler_LS.n_locally_owned_dofs_per_processor(),
                    Utilities::MPI::this_mpi_process(mpi_communicator));
  Akp1_matrix.reinit(mpi_communicator,
                     dsp,
                     dof_handler_LS.n_locally_owned_dofs_per_processor(),
                     dof_handler_LS.n_locally_owned_dofs_per_processor(),
                     Utilities::MPI::this_mpi_process(mpi_communicator));
  LxAkp1_matrix.reinit(mpi_communicator,
                       dsp,
                       dof_handler_LS.n_locally_owned_dofs_per_processor(),
                       dof_handler_LS.n_locally_owned_dofs_per_processor(),
                       Utilities::MPI::this_mpi_process(mpi_communicator));
  // COMPUTE MASS MATRICES (AND OTHERS) FOR FIRST TIME STEP //
  assemble_ML();
  invert_ML();
  assemble_MC();
  assemble_C_Matrix();
  // get mat for DOFs between Q1 and Q2
  get_map_from_Q1_to_Q2();
  get_sparsity_pattern();
}

// ----------------------------------------------------------------------------//
// ------------------------------ MASS MATRICES ------------------------------
// //
// ----------------------------------------------------------------------------//
template <int dim>
void
LevelSetSolver<dim>::assemble_ML()
{
  ML_vector = 0;

  const QGauss<dim> quadrature_formula(degree_MAX + 1);
  FEValues<dim>     fe_values_LS(fe_LS,
                             quadrature_formula,
                             update_values | update_gradients |
                               update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe_LS.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  Vector<double>                       cell_ML(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator cell_LS = dof_handler_LS
                                                             .begin_active(),
                                                 endc_LS = dof_handler_LS.end();

  for (; cell_LS != endc_LS; ++cell_LS)
    if (cell_LS->is_locally_owned())
      {
        cell_ML = 0;
        fe_values_LS.reinit(cell_LS);
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            const double JxW = fe_values_LS.JxW(q_point);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              cell_ML(i) += fe_values_LS.shape_value(i, q_point) * JxW;
          }
        // distribute
        cell_LS->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_ML,
                                               local_dof_indices,
                                               ML_vector);
      }
  // compress
  ML_vector.compress(VectorOperation::add);
}

template <int dim>
void
LevelSetSolver<dim>::invert_ML()
{
  // loop on locally owned i-DOFs (rows)
  IndexSet::ElementIterator idofs_iter = locally_owned_dofs_LS.begin();
  for (; idofs_iter != locally_owned_dofs_LS.end(); idofs_iter++)
    {
      int gi                = *idofs_iter;
      inverse_ML_vector(gi) = 1. / ML_vector(gi);
    }
  inverse_ML_vector.compress(VectorOperation::insert);
}

template <int dim>
void
LevelSetSolver<dim>::assemble_MC()
{
  MC_matrix = 0;

  const QGauss<dim> quadrature_formula(degree_MAX + 1);
  FEValues<dim>     fe_values_LS(fe_LS,
                             quadrature_formula,
                             update_values | update_gradients |
                               update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe_LS.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double>                   cell_MC(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<double>                  shape_values(dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator cell_LS = dof_handler_LS
                                                             .begin_active(),
                                                 endc_LS = dof_handler_LS.end();

  for (; cell_LS != endc_LS; ++cell_LS)
    if (cell_LS->is_locally_owned())
      {
        cell_MC = 0;
        fe_values_LS.reinit(cell_LS);
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            const double JxW = fe_values_LS.JxW(q_point);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              shape_values[i] = fe_values_LS.shape_value(i, q_point);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                cell_MC(i, j) += shape_values[i] * shape_values[j] * JxW;
          }
        // distribute
        cell_LS->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_MC,
                                               local_dof_indices,
                                               MC_matrix);
      }
  // compress
  MC_matrix.compress(VectorOperation::add);
  MC_preconditioner.reset(new PETScWrappers::PreconditionBoomerAMG(
    MC_matrix, PETScWrappers::PreconditionBoomerAMG::AdditionalData(true)));
}

// ---------------------------------------------------------------------------------------
// //
// ------------------------------ LO METHOD (Dij Viscosity)
// ------------------------------ //
// ---------------------------------------------------------------------------------------
// //
template <int dim>
void
LevelSetSolver<dim>::assemble_C_Matrix()
{
  Cx_matrix  = 0;
  CTx_matrix = 0;
  Cy_matrix  = 0;
  CTy_matrix = 0;
  Cz_matrix  = 0;
  CTz_matrix = 0;

  const QGauss<dim> quadrature_formula(degree_MAX + 1);
  FEValues<dim>     fe_values_LS(fe_LS,
                             quadrature_formula,
                             update_values | update_gradients |
                               update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell_LS = fe_LS.dofs_per_cell;
  const unsigned int n_q_points       = quadrature_formula.size();

  FullMatrix<double> cell_Cij_x(dofs_per_cell_LS, dofs_per_cell_LS);
  FullMatrix<double> cell_Cij_y(dofs_per_cell_LS, dofs_per_cell_LS);
  FullMatrix<double> cell_Cij_z(dofs_per_cell_LS, dofs_per_cell_LS);
  FullMatrix<double> cell_Cji_x(dofs_per_cell_LS, dofs_per_cell_LS);
  FullMatrix<double> cell_Cji_y(dofs_per_cell_LS, dofs_per_cell_LS);
  FullMatrix<double> cell_Cji_z(dofs_per_cell_LS, dofs_per_cell_LS);

  std::vector<Tensor<1, dim>> shape_grads_LS(dofs_per_cell_LS);
  std::vector<double>         shape_values_LS(dofs_per_cell_LS);

  std::vector<types::global_dof_index> local_dof_indices_LS(dofs_per_cell_LS);

  typename DoFHandler<dim>::active_cell_iterator cell_LS, endc_LS;
  cell_LS = dof_handler_LS.begin_active();
  endc_LS = dof_handler_LS.end();

  for (; cell_LS != endc_LS; ++cell_LS)
    if (cell_LS->is_locally_owned())
      {
        cell_Cij_x = 0;
        cell_Cij_y = 0;
        cell_Cji_x = 0;
        cell_Cji_y = 0;
        if (dim == 3)
          {
            cell_Cij_z = 0;
            cell_Cji_z = 0;
          }

        fe_values_LS.reinit(cell_LS);
        cell_LS->get_dof_indices(local_dof_indices_LS);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            const double JxW = fe_values_LS.JxW(q_point);
            for (unsigned int i = 0; i < dofs_per_cell_LS; ++i)
              {
                shape_values_LS[i] = fe_values_LS.shape_value(i, q_point);
                shape_grads_LS[i]  = fe_values_LS.shape_grad(i, q_point);
              }

            for (unsigned int i = 0; i < dofs_per_cell_LS; ++i)
              for (unsigned int j = 0; j < dofs_per_cell_LS; j++)
                {
                  cell_Cij_x(i, j) +=
                    (shape_grads_LS[j][0]) * shape_values_LS[i] * JxW;
                  cell_Cij_y(i, j) +=
                    (shape_grads_LS[j][1]) * shape_values_LS[i] * JxW;
                  cell_Cji_x(i, j) +=
                    (shape_grads_LS[i][0]) * shape_values_LS[j] * JxW;
                  cell_Cji_y(i, j) +=
                    (shape_grads_LS[i][1]) * shape_values_LS[j] * JxW;
                  if (dim == 3)
                    {
                      cell_Cij_z(i, j) +=
                        (shape_grads_LS[j][2]) * shape_values_LS[i] * JxW;
                      cell_Cji_z(i, j) +=
                        (shape_grads_LS[i][2]) * shape_values_LS[j] * JxW;
                    }
                }
          }
        // Distribute
        constraints.distribute_local_to_global(cell_Cij_x,
                                               local_dof_indices_LS,
                                               Cx_matrix);
        constraints.distribute_local_to_global(cell_Cji_x,
                                               local_dof_indices_LS,
                                               CTx_matrix);
        constraints.distribute_local_to_global(cell_Cij_y,
                                               local_dof_indices_LS,
                                               Cy_matrix);
        constraints.distribute_local_to_global(cell_Cji_y,
                                               local_dof_indices_LS,
                                               CTy_matrix);
        if (dim == 3)
          {
            constraints.distribute_local_to_global(cell_Cij_z,
                                                   local_dof_indices_LS,
                                                   Cz_matrix);
            constraints.distribute_local_to_global(cell_Cji_z,
                                                   local_dof_indices_LS,
                                                   CTz_matrix);
          }
      }
  // COMPRESS
  Cx_matrix.compress(VectorOperation::add);
  CTx_matrix.compress(VectorOperation::add);
  Cy_matrix.compress(VectorOperation::add);
  CTy_matrix.compress(VectorOperation::add);
  if (dim == 3)
    {
      Cz_matrix.compress(VectorOperation::add);
      CTz_matrix.compress(VectorOperation::add);
    }
}

template <int dim>
void
LevelSetSolver<dim>::assemble_K_times_vector(
  PETScWrappers::MPI::Vector &solution)
{
  K_times_solution = 0;

  const QGauss<dim>  quadrature_formula(degree_MAX + 1);
  FEValues<dim>      fe_values_LS(fe_LS,
                             quadrature_formula,
                             update_values | update_gradients |
                               update_quadrature_points | update_JxW_values);
  FEValues<dim>      fe_values_U(fe_U,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
  const unsigned int dofs_per_cell = fe_LS.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  Vector<double> cell_K_times_solution(dofs_per_cell);

  std::vector<Tensor<1, dim>> un_grads(n_q_points);
  std::vector<double>         old_vx_values(n_q_points);
  std::vector<double>         old_vy_values(n_q_points);
  std::vector<double>         old_vz_values(n_q_points);

  std::vector<double>         shape_values(dofs_per_cell);
  std::vector<Tensor<1, dim>> shape_grads(dofs_per_cell);

  Vector<double> un_dofs(dofs_per_cell);

  std::vector<types::global_dof_index> indices_LS(dofs_per_cell);

  // loop on cells
  typename DoFHandler<dim>::active_cell_iterator cell_LS = dof_handler_LS
                                                             .begin_active(),
                                                 endc_LS = dof_handler_LS.end();
  typename DoFHandler<dim>::active_cell_iterator cell_U =
    dof_handler_U.begin_active();

  Tensor<1, dim> v;
  for (; cell_LS != endc_LS; ++cell_U, ++cell_LS)
    if (cell_LS->is_locally_owned())
      {
        cell_K_times_solution = 0;

        fe_values_LS.reinit(cell_LS);
        cell_LS->get_dof_indices(indices_LS);
        fe_values_LS.get_function_gradients(solution, un_grads);

        fe_values_U.reinit(cell_U);
        fe_values_U.get_function_values(locally_relevant_solution_vx,
                                        old_vx_values);
        fe_values_U.get_function_values(locally_relevant_solution_vy,
                                        old_vy_values);
        if (dim == 3)
          fe_values_U.get_function_values(locally_relevant_solution_vz,
                                          old_vz_values);

        // compute cell_K_times_solution
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            v[0] = old_vx_values[q_point];
            v[1] = old_vy_values[q_point];
            if (dim == 3)
              v[2] = old_vz_values[q_point]; // dim=3

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              cell_K_times_solution(i) += (v * un_grads[q_point]) *
                                          fe_values_LS.shape_value(i, q_point) *
                                          fe_values_LS.JxW(q_point);
          }
        // distribute
        constraints.distribute_local_to_global(cell_K_times_solution,
                                               indices_LS,
                                               K_times_solution);
      }
  K_times_solution.compress(VectorOperation::add);
}

template <int dim>
void
LevelSetSolver<dim>::assemble_K_DL_DH_times_vector(
  PETScWrappers::MPI::Vector &solution)
{
  // K_times_solution=0;
  DL_times_solution = 0;
  DH_times_solution = 0;
  dLij_matrix       = 0;
  dCij_matrix       = 0;

  PetscInt           ncolumns;
  const PetscInt *   gj;
  const PetscScalar *Cxi, *Cyi, *Czi, *CTxi, *CTyi, *CTzi;
  const PetscScalar *EntResi, *SuppSizei, *MCi;
  double             solni;

  Tensor<1, dim> vi, vj;
  Tensor<1, dim> C, CT;
  // loop on locally owned i-DOFs (rows)
  IndexSet::ElementIterator idofs_iter = locally_owned_dofs_LS.begin();

  for (; idofs_iter != locally_owned_dofs_LS.end(); idofs_iter++)
    {
      PetscInt gi = *idofs_iter;
      // double ith_K_times_solution = 0;

      // read velocity of i-th DOF
      vi[0] = locally_relevant_solution_vx(map_from_Q1_to_Q2[gi]);
      vi[1] = locally_relevant_solution_vy(map_from_Q1_to_Q2[gi]);
      if (dim == 3)
        vi[2] = locally_relevant_solution_vz(map_from_Q1_to_Q2[gi]);
      solni = solution(gi);

      // get i-th row of C matrices
      MatGetRow(Cx_matrix, gi, &ncolumns, &gj, &Cxi);
      MatGetRow(Cy_matrix, gi, &ncolumns, &gj, &Cyi);
      MatGetRow(CTx_matrix, gi, &ncolumns, &gj, &CTxi);
      MatGetRow(CTy_matrix, gi, &ncolumns, &gj, &CTyi);
      if (dim == 3)
        {
          MatGetRow(Cz_matrix, gi, &ncolumns, &gj, &Czi);
          MatGetRow(CTz_matrix, gi, &ncolumns, &gj, &CTzi);
        }
      MatGetRow(EntRes_matrix, gi, &ncolumns, &gj, &EntResi);
      MatGetRow(SuppSize_matrix, gi, &ncolumns, &gj, &SuppSizei);
      MatGetRow(MC_matrix, gi, &ncolumns, &gj, &MCi);

      // get vector values for column indices
      const std::vector<types::global_dof_index> gj_indices(gj, gj + ncolumns);
      std::vector<double>                        soln(ncolumns);
      std::vector<double>                        vx(ncolumns);
      std::vector<double>                        vy(ncolumns);
      std::vector<double>                        vz(ncolumns);
      get_vector_values(solution, gj_indices, soln);
      get_vector_values(locally_relevant_solution_vx,
                        gj_indices,
                        map_from_Q1_to_Q2,
                        vx);
      get_vector_values(locally_relevant_solution_vy,
                        gj_indices,
                        map_from_Q1_to_Q2,
                        vy);
      if (dim == 3)
        get_vector_values(locally_relevant_solution_vz,
                          gj_indices,
                          map_from_Q1_to_Q2,
                          vz);

      // Array for i-th row of matrices
      std::vector<double> dLi(ncolumns), dCi(ncolumns);
      double              dLii = 0, dCii = 0;
      // loop on sparsity pattern of i-th DOF
      for (int j = 0; j < ncolumns; j++)
        {
          C[0]  = Cxi[j];
          C[1]  = Cyi[j];
          CT[0] = CTxi[j];
          CT[1] = CTyi[j];
          vj[0] = vx[j];
          vj[1] = vy[j];
          if (dim == 3)
            {
              C[2]  = Czi[j];
              CT[2] = CTzi[j];
              vj[2] = vz[j];
            }

          // ith_K_times_solution += soln[j]*(vj*C);
          if (gi != gj[j])
            {
              // low order dissipative matrix
              dLi[j] = -std::max(std::abs(vi * C), std::abs(vj * CT));
              dLii -= dLi[j];
              // high order dissipative matrix (entropy viscosity)
              double dEij = -std::min(-dLi[j],
                                      cE * std::abs(EntResi[j]) /
                                        (entropy_normalization_factor * MCi[j] /
                                         SuppSizei[j]));
              // high order compression matrix
              double Compij =
                cK * std::max(1 - std::pow(0.5 * (solni + soln[j]), 2), 0.0) /
                (std::abs(solni - soln[j]) + 1E-14);
              dCi[j] = dEij * std::max(1 - Compij, 0.0);
              dCii -= dCi[j];
            }
        }
      // save K times solution vector
      // K_times_solution(gi)=ith_K_times_solution;
      // save i-th row of matrices on global matrices
      MatSetValuesRow(dLij_matrix,
                      gi,
                      &dLi[0]); // BTW: there is a dealii wrapper for this
      dLij_matrix.set(gi, gi, dLii);
      MatSetValuesRow(dCij_matrix,
                      gi,
                      &dCi[0]); // BTW: there is a dealii wrapper for this
      dCij_matrix.set(gi, gi, dCii);

      // Restore matrices after reading rows
      MatRestoreRow(Cx_matrix, gi, &ncolumns, &gj, &Cxi);
      MatRestoreRow(Cy_matrix, gi, &ncolumns, &gj, &Cyi);
      MatRestoreRow(CTx_matrix, gi, &ncolumns, &gj, &CTxi);
      MatRestoreRow(CTy_matrix, gi, &ncolumns, &gj, &CTyi);
      if (dim == 3)
        {
          MatRestoreRow(Cz_matrix, gi, &ncolumns, &gj, &Czi);
          MatRestoreRow(CTz_matrix, gi, &ncolumns, &gj, &CTzi);
        }
      MatRestoreRow(EntRes_matrix, gi, &ncolumns, &gj, &EntResi);
      MatRestoreRow(SuppSize_matrix, gi, &ncolumns, &gj, &SuppSizei);
      MatRestoreRow(MC_matrix, gi, &ncolumns, &gj, &MCi);
    }
  // compress
  // K_times_solution.compress(VectorOperation::insert);
  dLij_matrix.compress(VectorOperation::insert);
  dCij_matrix.compress(VectorOperation::insert);
  // get matrices times vector
  dLij_matrix.vmult(DL_times_solution, solution);
  dCij_matrix.vmult(DH_times_solution, solution);
}

// --------------------------------------------------------------------------------------
// //
// ------------------------------ ENTROPY VISCOSITY
// ------------------------------ //
// --------------------------------------------------------------------------------------
// //
template <int dim>
void
LevelSetSolver<dim>::assemble_EntRes_Matrix()
{
  EntRes_matrix                = 0;
  entropy_normalization_factor = 0;
  SuppSize_matrix              = 0;

  const QGauss<dim> quadrature_formula(degree_MAX + 1);
  FEValues<dim>     fe_values_U(fe_U,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
  FEValues<dim>     fe_values_LS(fe_LS,
                             quadrature_formula,
                             update_values | update_gradients |
                               update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell_LS = fe_LS.dofs_per_cell;
  const unsigned int n_q_points       = quadrature_formula.size();

  std::vector<double>         uqn(n_q_points); // un at q point
  std::vector<double>         uqnm1(n_q_points);
  std::vector<Tensor<1, dim>> guqn(n_q_points); // grad of uqn
  std::vector<Tensor<1, dim>> guqnm1(n_q_points);

  std::vector<double> vxqn(n_q_points);
  std::vector<double> vyqn(n_q_points);
  std::vector<double> vzqn(n_q_points);
  std::vector<double> vxqnm1(n_q_points);
  std::vector<double> vyqnm1(n_q_points);
  std::vector<double> vzqnm1(n_q_points);

  FullMatrix<double> cell_EntRes(dofs_per_cell_LS, dofs_per_cell_LS);
  FullMatrix<double> cell_volume(dofs_per_cell_LS, dofs_per_cell_LS);

  std::vector<Tensor<1, dim>> shape_grads_LS(dofs_per_cell_LS);
  std::vector<double>         shape_values_LS(dofs_per_cell_LS);

  std::vector<types::global_dof_index> local_dof_indices_LS(dofs_per_cell_LS);

  typename DoFHandler<dim>::active_cell_iterator cell_LS, endc_LS;
  cell_LS = dof_handler_LS.begin_active();
  endc_LS = dof_handler_LS.end();
  typename DoFHandler<dim>::active_cell_iterator cell_U =
    dof_handler_U.begin_active();

  double Rk;
  double max_entropy = -1E10, min_entropy = 1E10;
  double cell_max_entropy, cell_min_entropy;
  double cell_entropy_mass, entropy_mass = 0;
  double cell_volume_double, volume      = 0;

  for (; cell_LS != endc_LS; ++cell_LS, ++cell_U)
    if (cell_LS->is_locally_owned())
      {
        cell_entropy_mass  = 0;
        cell_volume_double = 0;
        cell_max_entropy   = -1E10;
        cell_min_entropy   = 1E10;
        cell_EntRes        = 0;
        cell_volume        = 0;

        // get solutions at quadrature points
        fe_values_LS.reinit(cell_LS);
        cell_LS->get_dof_indices(local_dof_indices_LS);
        fe_values_LS.get_function_values(un, uqn);
        fe_values_LS.get_function_values(unm1, uqnm1);
        fe_values_LS.get_function_gradients(un, guqn);
        fe_values_LS.get_function_gradients(unm1, guqnm1);

        fe_values_U.reinit(cell_U);
        fe_values_U.get_function_values(locally_relevant_solution_vx, vxqn);
        fe_values_U.get_function_values(locally_relevant_solution_vy, vyqn);
        if (dim == 3)
          fe_values_U.get_function_values(locally_relevant_solution_vz, vzqn);
        fe_values_U.get_function_values(locally_relevant_solution_vx_old,
                                        vxqnm1);
        fe_values_U.get_function_values(locally_relevant_solution_vy_old,
                                        vyqnm1);
        if (dim == 3)
          fe_values_U.get_function_values(locally_relevant_solution_vz_old,
                                          vzqnm1);

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            Rk = 1. / time_step * (ENTROPY(uqn[q]) - ENTROPY(uqnm1[q])) +
                 (vxqn[q] * ENTROPY_GRAD(uqn[q], guqn[q][0]) +
                  vyqn[q] * ENTROPY_GRAD(uqn[q], guqn[q][1])) /
                   2. +
                 (vxqnm1[q] * ENTROPY_GRAD(uqnm1[q], guqnm1[q][0]) +
                  vyqnm1[q] * ENTROPY_GRAD(uqnm1[q], guqnm1[q][1])) /
                   2.;
            if (dim == 3)
              Rk += 0.5 * (vzqn[q] * ENTROPY_GRAD(uqn[q], guqn[q][2]) +
                           vzqnm1[q] * ENTROPY_GRAD(uqnm1[q], guqnm1[q][2]));

            const double JxW = fe_values_LS.JxW(q);
            for (unsigned int i = 0; i < dofs_per_cell_LS; ++i)
              {
                shape_values_LS[i] = fe_values_LS.shape_value(i, q);
                shape_grads_LS[i]  = fe_values_LS.shape_grad(i, q);
              }

            for (unsigned int i = 0; i < dofs_per_cell_LS; ++i)
              for (unsigned int j = 0; j < dofs_per_cell_LS; j++)
                {
                  cell_EntRes(i, j) +=
                    Rk * shape_values_LS[i] * shape_values_LS[j] * JxW;
                  cell_volume(i, j) += JxW;
                }
            cell_entropy_mass += ENTROPY(uqn[q]) * JxW;
            cell_volume_double += JxW;

            cell_min_entropy = std::min(cell_min_entropy, ENTROPY(uqn[q]));
            cell_max_entropy = std::max(cell_max_entropy, ENTROPY(uqn[q]));
          }
        entropy_mass += cell_entropy_mass;
        volume += cell_volume_double;

        min_entropy = std::min(min_entropy, cell_min_entropy);
        max_entropy = std::max(max_entropy, cell_max_entropy);
        // Distribute
        constraints.distribute_local_to_global(cell_EntRes,
                                               local_dof_indices_LS,
                                               EntRes_matrix);
        constraints.distribute_local_to_global(cell_volume,
                                               local_dof_indices_LS,
                                               SuppSize_matrix);
      }
  EntRes_matrix.compress(VectorOperation::add);
  SuppSize_matrix.compress(VectorOperation::add);
  // ENTROPY NORM FACTOR
  volume       = Utilities::MPI::sum(volume, mpi_communicator);
  entropy_mass = Utilities::MPI::sum(entropy_mass, mpi_communicator) / volume;
  min_entropy  = Utilities::MPI::min(min_entropy, mpi_communicator);
  max_entropy  = Utilities::MPI::max(max_entropy, mpi_communicator);
  entropy_normalization_factor = std::max(std::abs(max_entropy - entropy_mass),
                                          std::abs(min_entropy - entropy_mass));
}

// ------------------------------------------------------------------------------------
// //
// ------------------------------ TO CHECK MAX PRINCIPLE
// ------------------------------ //
// ------------------------------------------------------------------------------------
// //
template <int dim>
void
LevelSetSolver<dim>::compute_bounds(PETScWrappers::MPI::Vector &un_solution)
{
  umin_vector = 0;
  umax_vector = 0;
  // loop on locally owned i-DOFs (rows)
  IndexSet::ElementIterator idofs_iter = locally_owned_dofs_LS.begin();
  for (; idofs_iter != locally_owned_dofs_LS.end(); idofs_iter++)
    {
      int gi = *idofs_iter;

      // get solution at DOFs on the sparsity pattern of i-th DOF
      std::vector<types::global_dof_index> gj_indices = sparsity_pattern[gi];
      std::vector<double>                  soln(gj_indices.size());
      get_vector_values(un_solution, gj_indices, soln);
      // compute bounds, ith row of flux matrix, P vectors
      double mini = 1E10, maxi = -1E10;
      for (unsigned int j = 0; j < gj_indices.size(); j++)
        {
          // bounds
          mini = std::min(mini, soln[j]);
          maxi = std::max(maxi, soln[j]);
        }
      umin_vector(gi) = mini;
      umax_vector(gi) = maxi;
    }
  umin_vector.compress(VectorOperation::insert);
  umax_vector.compress(VectorOperation::insert);
}

template <int dim>
void
LevelSetSolver<dim>::check_max_principle(
  PETScWrappers::MPI::Vector &unp1_solution)
{
  // compute min and max vectors
  const unsigned int                   dofs_per_cell = fe_LS.dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  double                                         tol     = 1e-10;
  typename DoFHandler<dim>::active_cell_iterator cell_LS = dof_handler_LS
                                                             .begin_active(),
                                                 endc_LS = dof_handler_LS.end();

  for (; cell_LS != endc_LS; ++cell_LS)
    if (cell_LS->is_locally_owned() && !cell_LS->at_boundary())
      {
        cell_LS->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; i++)
          if (locally_owned_dofs_LS.is_element(local_dof_indices[i]))
            {
              double solni = unp1_solution(local_dof_indices[i]);
              if (solni - umin_vector(local_dof_indices[i]) < -tol ||
                  umax_vector(local_dof_indices[i]) - solni < -tol)
                {
                  pcout << "MAX Principle violated" << std::endl;
                  abort();
                }
            }
      }
}

// -------------------------------------------------------------------------------
// //
// ------------------------------ COMPUTE SOLUTIONS
// ------------------------------ //
// -------------------------------------------------------------------------------
// //
template <int dim>
void
LevelSetSolver<dim>::compute_MPP_uL_and_NMPP_uH(
  PETScWrappers::MPI::Vector &MPP_uL_solution,
  PETScWrappers::MPI::Vector &NMPP_uH_solution,
  PETScWrappers::MPI::Vector &un_solution)
{
  // NON-GHOSTED VECTORS: MPP_uL_solution, NMPP_uH_solution
  // GHOSTED VECTORS: un_solution
  MPP_uL_solution = un_solution;
  NMPP_uH_solution =
    un_solution; // to start iterative solver at un_solution (instead of zero)
  // assemble RHS VECTORS
  assemble_K_times_vector(un_solution);
  assemble_K_DL_DH_times_vector(un_solution);
  /////////////////////////////
  // COMPUTE MPP u1 solution //
  /////////////////////////////
  MPP_uL_solution.scale(ML_vector);
  MPP_uL_solution.add(-time_step, K_times_solution);
  MPP_uL_solution.add(-time_step, DL_times_solution);
  MPP_uL_solution.scale(inverse_ML_vector);
  //////////////////////////////////
  // COMPUTE GALERKIN u2 solution //
  //////////////////////////////////
  MC_matrix.vmult(RHS, un_solution);
  RHS.add(-time_step, K_times_solution, -time_step, DH_times_solution);
  solve(constraints, MC_matrix, MC_preconditioner, NMPP_uH_solution, RHS);
}

template <int dim>
void
LevelSetSolver<dim>::compute_MPP_uH(
  PETScWrappers::MPI::Vector &MPP_uH_solution,
  PETScWrappers::MPI::Vector &MPP_uL_solution_ghosted,
  PETScWrappers::MPI::Vector &NMPP_uH_solution_ghosted,
  PETScWrappers::MPI::Vector &solution)
{
  MPP_uH_solution = 0;
  // loop on locally owned i-DOFs (rows)
  IndexSet::ElementIterator idofs_iter = locally_owned_dofs_LS.begin();

  PetscInt           ncolumns;
  const PetscInt *   gj;
  const PetscScalar *MCi, *dLi, *dCi;
  double             solni, mi, solLi, solHi;

  for (; idofs_iter != locally_owned_dofs_LS.end(); idofs_iter++)
    {
      int gi = *idofs_iter;
      // read vectors at i-th DOF
      solni = solution(gi);
      solHi = NMPP_uH_solution_ghosted(gi);
      solLi = MPP_uL_solution_ghosted(gi);
      mi    = ML_vector(gi);

      // get i-th row of matrices
      MatGetRow(MC_matrix, gi, &ncolumns, &gj, &MCi);
      MatGetRow(dLij_matrix, gi, &ncolumns, &gj, &dLi);
      MatGetRow(dCij_matrix, gi, &ncolumns, &gj, &dCi);

      // get vector values for support of i-th DOF
      const std::vector<types::global_dof_index> gj_indices(gj, gj + ncolumns);
      std::vector<double>                        soln(ncolumns);
      std::vector<double>                        solH(ncolumns);
      get_vector_values(solution, gj_indices, soln);
      get_vector_values(NMPP_uH_solution_ghosted, gj_indices, solH);

      // Array for i-th row of matrices
      std::vector<double> Ai(ncolumns);
      // compute bounds, ith row of flux matrix, P vectors
      double mini = 1E10, maxi = -1E10;
      double Pposi = 0, Pnegi = 0;
      for (int j = 0; j < ncolumns; j++)
        {
          // bounds
          mini = std::min(mini, soln[j]);
          maxi = std::max(maxi, soln[j]);

          // i-th row of flux matrix A
          Ai[j] = (((gi == gj[j]) ? 1 : 0) * mi - MCi[j]) *
                    (solH[j] - soln[j] - (solHi - solni)) +
                  time_step * (dLi[j] - dCi[j]) * (soln[j] - solni);

          // compute P vectors
          Pposi += Ai[j] * ((Ai[j] > 0) ? 1. : 0.);
          Pnegi += Ai[j] * ((Ai[j] < 0) ? 1. : 0.);
        }
      // save i-th row of flux matrix A
      MatSetValuesRow(A_matrix, gi, &Ai[0]);

      // compute Q vectors
      double Qposi = mi * (maxi - solLi);
      double Qnegi = mi * (mini - solLi);

      // compute R vectors
      R_pos_vector_nonGhosted(gi) =
        ((Pposi == 0) ? 1. : std::min(1.0, Qposi / Pposi));
      R_neg_vector_nonGhosted(gi) =
        ((Pnegi == 0) ? 1. : std::min(1.0, Qnegi / Pnegi));

      // Restore matrices after reading rows
      MatRestoreRow(MC_matrix, gi, &ncolumns, &gj, &MCi);
      MatRestoreRow(dLij_matrix, gi, &ncolumns, &gj, &dLi);
      MatRestoreRow(dCij_matrix, gi, &ncolumns, &gj, &dCi);
    }
  // compress A matrix
  A_matrix.compress(VectorOperation::insert);
  // compress R vectors
  R_pos_vector_nonGhosted.compress(VectorOperation::insert);
  R_neg_vector_nonGhosted.compress(VectorOperation::insert);
  // update ghost values for R vectors
  R_pos_vector = R_pos_vector_nonGhosted;
  R_neg_vector = R_neg_vector_nonGhosted;

  // compute limiters. NOTE: this is a different loop due to need of i- and j-th
  // entries of R vectors
  const double *Ai;
  double        Rposi, Rnegi;
  idofs_iter = locally_owned_dofs_LS.begin();
  for (; idofs_iter != locally_owned_dofs_LS.end(); idofs_iter++)
    {
      int gi = *idofs_iter;
      Rposi  = R_pos_vector(gi);
      Rnegi  = R_neg_vector(gi);

      // get i-th row of A matrix
      MatGetRow(A_matrix, gi, &ncolumns, &gj, &Ai);

      // get vector values for column indices
      const std::vector<types::global_dof_index> gj_indices(gj, gj + ncolumns);
      std::vector<double>                        Rpos(ncolumns);
      std::vector<double>                        Rneg(ncolumns);
      get_vector_values(R_pos_vector, gj_indices, Rpos);
      get_vector_values(R_neg_vector, gj_indices, Rneg);

      // Array for i-th row of A_times_L matrix
      std::vector<double> LxAi(ncolumns);
      // loop in sparsity pattern of i-th DOF
      for (int j = 0; j < ncolumns; j++)
        LxAi[j] = Ai[j] * ((Ai[j] > 0) ? std::min(Rposi, Rneg[j]) :
                                         std::min(Rnegi, Rpos[j]));

      // save i-th row of LxA
      MatSetValuesRow(LxA_matrix,
                      gi,
                      &LxAi[0]); // BTW: there is a dealii wrapper for this
      // restore A matrix after reading it
      MatRestoreRow(A_matrix, gi, &ncolumns, &gj, &Ai);
    }
  LxA_matrix.compress(VectorOperation::insert);
  LxA_matrix.vmult(MPP_uH_solution, ones_vector);
  MPP_uH_solution.scale(inverse_ML_vector);
  MPP_uH_solution.add(1.0, MPP_uL_solution_ghosted);
}

template <int dim>
void
LevelSetSolver<dim>::compute_MPP_uH_with_iterated_FCT(
  PETScWrappers::MPI::Vector &MPP_uH_solution,
  PETScWrappers::MPI::Vector &MPP_uL_solution_ghosted,
  PETScWrappers::MPI::Vector &NMPP_uH_solution_ghosted,
  PETScWrappers::MPI::Vector &un_solution)
{
  MPP_uH_solution = 0;
  compute_MPP_uH(MPP_uH_solution,
                 MPP_uL_solution_ghosted,
                 NMPP_uH_solution_ghosted,
                 un_solution);

  if (NUM_ITER > 0)
    {
      Akp1_matrix.copy_from(A_matrix);
      LxAkp1_matrix.copy_from(LxA_matrix);

      // loop in num of FCT iterations
      PetscInt           ncolumns;
      const PetscInt *   gj;
      const PetscScalar *Akp1i;
      double             mi;
      for (int iter = 0; iter < NUM_ITER; iter++)
        {
          MPP_uLkp1_solution_ghosted = MPP_uH_solution;
          Akp1_matrix.add(-1.0, LxAkp1_matrix); // new matrix to limit: A-LxA

          // loop on locally owned i-DOFs (rows)
          IndexSet::ElementIterator idofs_iter = locally_owned_dofs_LS.begin();
          for (; idofs_iter != locally_owned_dofs_LS.end(); idofs_iter++)
            {
              int gi = *idofs_iter;

              // read vectors at i-th DOF
              mi           = ML_vector(gi);
              double solLi = MPP_uLkp1_solution_ghosted(gi);

              // get i-th row of matrices
              MatGetRow(Akp1_matrix, gi, &ncolumns, &gj, &Akp1i);
              // get vector values for support of i-th DOF
              const std::vector<types::global_dof_index> gj_indices(gj,
                                                                    gj +
                                                                      ncolumns);
              std::vector<double>                        soln(ncolumns);
              get_vector_values(un_solution, gj_indices, soln);

              // compute bounds, ith row of flux matrix, P vectors
              double mini = 1E10, maxi = -1E10;
              double Pposi = 0, Pnegi = 0;
              for (int j = 0; j < ncolumns; j++)
                {
                  // bounds
                  mini = std::min(mini, soln[j]);
                  maxi = std::max(maxi, soln[j]);

                  // compute P vectors
                  Pposi += Akp1i[j] * ((Akp1i[j] > 0) ? 1. : 0.);
                  Pnegi += Akp1i[j] * ((Akp1i[j] < 0) ? 1. : 0.);
                }
              // compute Q vectors
              double Qposi = mi * (maxi - solLi);
              double Qnegi = mi * (mini - solLi);

              // compute R vectors
              R_pos_vector_nonGhosted(gi) =
                ((Pposi == 0) ? 1. : std::min(1.0, Qposi / Pposi));
              R_neg_vector_nonGhosted(gi) =
                ((Pnegi == 0) ? 1. : std::min(1.0, Qnegi / Pnegi));

              // Restore matrices after reading rows
              MatRestoreRow(Akp1_matrix, gi, &ncolumns, &gj, &Akp1i);
            }
          // compress R vectors
          R_pos_vector_nonGhosted.compress(VectorOperation::insert);
          R_neg_vector_nonGhosted.compress(VectorOperation::insert);
          // update ghost values for R vectors
          R_pos_vector = R_pos_vector_nonGhosted;
          R_neg_vector = R_neg_vector_nonGhosted;

          // compute limiters. NOTE: this is a different loop due to need of i-
          // and j-th entries of R vectors
          double Rposi, Rnegi;
          idofs_iter = locally_owned_dofs_LS.begin();
          for (; idofs_iter != locally_owned_dofs_LS.end(); idofs_iter++)
            {
              int gi = *idofs_iter;
              Rposi  = R_pos_vector(gi);
              Rnegi  = R_neg_vector(gi);

              // get i-th row of Akp1 matrix
              MatGetRow(Akp1_matrix, gi, &ncolumns, &gj, &Akp1i);

              // get vector values for column indices
              const std::vector<types::global_dof_index> gj_indices(gj,
                                                                    gj +
                                                                      ncolumns);
              std::vector<double>                        Rpos(ncolumns);
              std::vector<double>                        Rneg(ncolumns);
              get_vector_values(R_pos_vector, gj_indices, Rpos);
              get_vector_values(R_neg_vector, gj_indices, Rneg);

              // Array for i-th row of LxAkp1 matrix
              std::vector<double> LxAkp1i(ncolumns);
              for (int j = 0; j < ncolumns; j++)
                LxAkp1i[j] =
                  Akp1i[j] * ((Akp1i[j] > 0) ? std::min(Rposi, Rneg[j]) :
                                               std::min(Rnegi, Rpos[j]));

              // save i-th row of LxA
              MatSetValuesRow(
                LxAkp1_matrix,
                gi,
                &LxAkp1i[0]); // BTW: there is a dealii wrapper for this
              // restore A matrix after reading it
              MatRestoreRow(Akp1_matrix, gi, &ncolumns, &gj, &Akp1i);
            }
          LxAkp1_matrix.compress(VectorOperation::insert);
          LxAkp1_matrix.vmult(MPP_uH_solution, ones_vector);
          MPP_uH_solution.scale(inverse_ML_vector);
          MPP_uH_solution.add(1.0, MPP_uLkp1_solution_ghosted);
        }
    }
}

template <int dim>
void
LevelSetSolver<dim>::compute_solution(PETScWrappers::MPI::Vector &unp1,
                                      PETScWrappers::MPI::Vector &un,
                                      std::string                 algorithm)
{
  unp1 = 0;
  // COMPUTE MPP LOW-ORDER SOLN and NMPP HIGH-ORDER SOLN
  compute_MPP_uL_and_NMPP_uH(MPP_uL_solution, NMPP_uH_solution, un);

  if (algorithm.compare("MPP_u1") == 0)
    unp1 = MPP_uL_solution;
  else if (algorithm.compare("NMPP_uH") == 0)
    unp1 = NMPP_uH_solution;
  else if (algorithm.compare("MPP_uH") == 0)
    {
      MPP_uL_solution_ghosted  = MPP_uL_solution;
      NMPP_uH_solution_ghosted = NMPP_uH_solution;
      compute_MPP_uH_with_iterated_FCT(MPP_uH_solution,
                                       MPP_uL_solution_ghosted,
                                       NMPP_uH_solution_ghosted,
                                       un);
      unp1 = MPP_uH_solution;
    }
  else
    {
      pcout << "Error in algorithm" << std::endl;
      abort();
    }
}

template <int dim>
void
LevelSetSolver<dim>::compute_solution_SSP33(PETScWrappers::MPI::Vector &unp1,
                                            PETScWrappers::MPI::Vector &un,
                                            std::string algorithm)
{
  // GHOSTED VECTORS: un
  // NON-GHOSTED VECTORS: unp1
  unp1    = 0;
  uStage1 = 0., uStage2 = 0.;
  uStage1_nonGhosted = 0., uStage2_nonGhosted = 0.;
  /////////////////
  // FIRST STAGE //
  /////////////////
  // u1=un-dt*RH*un
  compute_solution(uStage1_nonGhosted, un, algorithm);
  uStage1 = uStage1_nonGhosted;
  //////////////////
  // SECOND STAGE //
  //////////////////
  // u2=3/4*un+1/4*(u1-dt*RH*u1)
  compute_solution(uStage2_nonGhosted, uStage1, algorithm);
  uStage2_nonGhosted *= 1. / 4;
  uStage2_nonGhosted.add(3. / 4, un);
  uStage2 = uStage2_nonGhosted;
  /////////////////
  // THIRD STAGE //
  /////////////////
  // unp1=1/3*un+2/3*(u2-dt*RH*u2)
  compute_solution(unp1, uStage2, algorithm);
  unp1 *= 2. / 3;
  unp1.add(1. / 3, un);
}

// ----------------------------------------------------------------------- //
// ------------------------------ UTILITIES ------------------------------ //
// ----------------------------------------------------------------------- //
template <int dim>
void
LevelSetSolver<dim>::get_sparsity_pattern()
{
  // loop on DOFs
  IndexSet::ElementIterator idofs_iter = locally_owned_dofs_LS.begin();
  PetscInt                  ncolumns;
  const PetscInt *          gj;
  const PetscScalar *       MCi;

  for (; idofs_iter != locally_owned_dofs_LS.end(); idofs_iter++)
    {
      PetscInt gi = *idofs_iter;
      // get i-th row of mass matrix (dummy, I just need the indices gj)
      MatGetRow(MC_matrix, gi, &ncolumns, &gj, &MCi);
      sparsity_pattern[gi] =
        std::vector<types::global_dof_index>(gj, gj + ncolumns);
      MatRestoreRow(MC_matrix, gi, &ncolumns, &gj, &MCi);
    }
}

template <int dim>
void
LevelSetSolver<dim>::get_map_from_Q1_to_Q2()
{
  map_from_Q1_to_Q2.clear();
  const unsigned int                   dofs_per_cell_LS = fe_LS.dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices_LS(dofs_per_cell_LS);
  const unsigned int                   dofs_per_cell_U = fe_U.dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices_U(dofs_per_cell_U);

  typename DoFHandler<dim>::active_cell_iterator cell_LS = dof_handler_LS
                                                             .begin_active(),
                                                 endc_LS = dof_handler_LS.end();
  typename DoFHandler<dim>::active_cell_iterator cell_U =
    dof_handler_U.begin_active();

  for (; cell_LS != endc_LS; ++cell_LS, ++cell_U)
    if (!cell_LS->is_artificial()) // loop on ghost cells as well
      {
        cell_LS->get_dof_indices(local_dof_indices_LS);
        cell_U->get_dof_indices(local_dof_indices_U);
        for (unsigned int i = 0; i < dofs_per_cell_LS; ++i)
          map_from_Q1_to_Q2[local_dof_indices_LS[i]] = local_dof_indices_U[i];
      }
}

template <int dim>
void
LevelSetSolver<dim>::solve(
  const ConstraintMatrix &                                    constraints,
  PETScWrappers::MPI::SparseMatrix &                          Matrix,
  std_cxx1x::shared_ptr<PETScWrappers::PreconditionBoomerAMG> preconditioner,
  PETScWrappers::MPI::Vector &      completely_distributed_solution,
  const PETScWrappers::MPI::Vector &rhs)
{
  // all vectors are NON-GHOSTED
  SolverControl solver_control(dof_handler_LS.n_dofs(), solver_tolerance);
  PETScWrappers::SolverCG solver(solver_control, mpi_communicator);
  constraints.distribute(completely_distributed_solution);
  solver.solve(Matrix, completely_distributed_solution, rhs, *preconditioner);
  constraints.distribute(completely_distributed_solution);
  if (verbose == true)
    pcout << "   Solved in " << solver_control.last_step() << " iterations."
          << std::endl;
}

template <int dim>
void
LevelSetSolver<dim>::save_old_solution()
{
  unm1 = un;
  un   = unp1;
}

template <int dim>
void
LevelSetSolver<dim>::save_old_vel_solution()
{
  locally_relevant_solution_vx_old = locally_relevant_solution_vx;
  locally_relevant_solution_vy_old = locally_relevant_solution_vy;
  if (dim == 3)
    locally_relevant_solution_vz_old = locally_relevant_solution_vz;
}

// -------------------------------------------------------------------------------
// //
// ------------------------------ MY PETSC WRAPPERS
// ------------------------------ //
// -------------------------------------------------------------------------------
// //
template <int dim>
void
LevelSetSolver<dim>::get_vector_values(
  PETScWrappers::VectorBase &                 vector,
  const std::vector<types::global_dof_index> &indices,
  std::vector<PetscScalar> &                  values)
{
  // PETSc wrapper to get sets of values from a petsc vector.
  // we assume the vector is ghosted
  // We need to figure out which elements we
  // own locally. Then get a pointer to the
  // elements that are stored here (both the
  // ones we own as well as the ghost elements).
  // In this array, the locally owned elements
  // come first followed by the ghost elements whose
  // position we can get from an index set

  IndexSet ghost_indices = locally_relevant_dofs_LS;
  ghost_indices.subtract_set(locally_owned_dofs_LS);

  PetscInt n_idx, begin, end, i;
  n_idx = indices.size();

  VecGetOwnershipRange(vector, &begin, &end);

  Vec solution_in_local_form = PETSC_NULL;
  VecGhostGetLocalForm(vector, &solution_in_local_form);

  PetscScalar *soln;
  VecGetArray(solution_in_local_form, &soln);

  for (i = 0; i < n_idx; i++)
    {
      int index = indices[i];
      if (index >= begin && index < end)
        values[i] = *(soln + index - begin);
      else // ghost
        {
          const unsigned int ghostidx = ghost_indices.index_within_set(index);
          values[i]                   = *(soln + ghostidx + end - begin);
        }
    }
  VecRestoreArray(solution_in_local_form, &soln);
  VecGhostRestoreLocalForm(vector, &solution_in_local_form);
}

template <int dim>
void
LevelSetSolver<dim>::get_vector_values(
  PETScWrappers::VectorBase &                                 vector,
  const std::vector<types::global_dof_index> &                indices,
  std::map<types::global_dof_index, types::global_dof_index> &map_from_Q1_to_Q2,
  std::vector<PetscScalar> &                                  values)
{
  // THIS IS MEANT TO BE USED WITH VELOCITY VECTORS
  // PETSc wrapper to get sets of values from a petsc vector.
  // we assume the vector is ghosted
  // We need to figure out which elements we
  // own locally. Then get a pointer to the
  // elements that are stored here (both the
  // ones we own as well as the ghost elements).
  // In this array, the locally owned elements
  // come first followed by the ghost elements whose
  // position we can get from an index set

  IndexSet ghost_indices = locally_relevant_dofs_U;
  ghost_indices.subtract_set(locally_owned_dofs_U);

  PetscInt n_idx, begin, end, i;
  n_idx = indices.size();

  VecGetOwnershipRange(vector, &begin, &end);

  Vec solution_in_local_form = PETSC_NULL;
  VecGhostGetLocalForm(vector, &solution_in_local_form);

  PetscScalar *soln;
  VecGetArray(solution_in_local_form, &soln);

  for (i = 0; i < n_idx; i++)
    {
      int index = map_from_Q1_to_Q2[indices[i]];
      if (index >= begin && index < end)
        values[i] = *(soln + index - begin);
      else // ghost
        {
          const unsigned int ghostidx = ghost_indices.index_within_set(index);
          values[i]                   = *(soln + ghostidx + end - begin);
        }
    }
  VecRestoreArray(solution_in_local_form, &soln);
  VecGhostRestoreLocalForm(vector, &solution_in_local_form);
}
