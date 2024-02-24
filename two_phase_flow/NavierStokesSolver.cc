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

#include <fstream>
#include <iostream>
#include <memory>

using namespace dealii;

#define MAX_NUM_ITER_TO_RECOMPUTE_PRECONDITIONER 10

/////////////////////////////////////////////////////////////////
///////////////////// NAVIER STOKES SOLVER //////////////////////
/////////////////////////////////////////////////////////////////
template<int dim>
class NavierStokesSolver
{
public:
  // constructor for using LEVEL SET
  NavierStokesSolver(const unsigned int degree_LS,
                     const unsigned int degree_U,
                     const double time_step,
                     const double eps,
                     const double rho_air,
                     const double nu_air,
                     const double rho_fluid,
                     const double nu_fluid,
                     Function<dim> &force_function,
                     const bool verbose,
                     parallel::distributed::Triangulation<dim> &triangulation,
                     MPI_Comm &mpi_communicator);
  // constructor for NOT LEVEL SET
  NavierStokesSolver(const unsigned int degree_LS,
                     const unsigned int degree_U,
                     const double time_step,
                     Function<dim> &force_function,
                     Function<dim> &rho_function,
                     Function<dim> &nu_function,
                     const bool verbose,
                     parallel::distributed::Triangulation<dim> &triangulation,
                     MPI_Comm &mpi_communicator);

  // rho and nu functions
  void set_rho_and_nu_functions(const Function<dim> &rho_function,
                                const Function<dim> &nu_function);
  //initial conditions
  void initial_condition(PETScWrappers::MPI::Vector locally_relevant_solution_rho,
                         PETScWrappers::MPI::Vector locally_relevant_solution_u,
                         PETScWrappers::MPI::Vector locally_relevant_solution_v,
                         PETScWrappers::MPI::Vector locally_relevant_solution_p);
  void initial_condition(PETScWrappers::MPI::Vector locally_relevant_solution_rho,
                         PETScWrappers::MPI::Vector locally_relevant_solution_u,
                         PETScWrappers::MPI::Vector locally_relevant_solution_v,
                         PETScWrappers::MPI::Vector locally_relevant_solution_w,
                         PETScWrappers::MPI::Vector locally_relevant_solution_p);
  //boundary conditions
  void set_boundary_conditions(std::vector<types::global_dof_index> boundary_values_id_u,
                               std::vector<types::global_dof_index> boundary_values_id_v, std::vector<double> boundary_values_u,
                               std::vector<double> boundary_values_v);
  void set_boundary_conditions(std::vector<types::global_dof_index> boundary_values_id_u,
                               std::vector<types::global_dof_index> boundary_values_id_v,
                               std::vector<types::global_dof_index> boundary_values_id_w, std::vector<double> boundary_values_u,
                               std::vector<double> boundary_values_v, std::vector<double> boundary_values_w);
  void set_velocity(PETScWrappers::MPI::Vector locally_relevant_solution_u,
                    PETScWrappers::MPI::Vector locally_relevant_solution_v);
  void set_velocity(PETScWrappers::MPI::Vector locally_relevant_solution_u,
                    PETScWrappers::MPI::Vector locally_relevant_solution_v,
                    PETScWrappers::MPI::Vector locally_relevant_solution_w);
  void set_phi(PETScWrappers::MPI::Vector locally_relevant_solution_phi);
  void get_pressure(PETScWrappers::MPI::Vector &locally_relevant_solution_p);
  void get_velocity(PETScWrappers::MPI::Vector &locally_relevant_solution_u,
                    PETScWrappers::MPI::Vector &locally_relevant_solution_v);
  void get_velocity(PETScWrappers::MPI::Vector &locally_relevant_solution_u,
                    PETScWrappers::MPI::Vector &locally_relevant_solution_v,
                    PETScWrappers::MPI::Vector &locally_relevant_solution_w);
  // DO STEPS //
  void nth_time_step();
  // SETUP //
  void setup();

  ~NavierStokesSolver();

private:
  // SETUP AND INITIAL CONDITION //
  void setup_DOF();
  void setup_VECTORS();
  void init_constraints();
  // ASSEMBLE SYSTEMS //
  void assemble_system_U();
  void assemble_system_dpsi_q();
  // SOLVERS //
  void solve_U(const AffineConstraints<double> &constraints, PETScWrappers::MPI::SparseMatrix &Matrix,
               std::shared_ptr<PETScWrappers::PreconditionBoomerAMG> preconditioner,
               PETScWrappers::MPI::Vector &completely_distributed_solution,
               const PETScWrappers::MPI::Vector &rhs);
  void solve_P(const AffineConstraints<double> &constraints, PETScWrappers::MPI::SparseMatrix &Matrix,
               std::shared_ptr<PETScWrappers::PreconditionBoomerAMG> preconditioner,
               PETScWrappers::MPI::Vector &completely_distributed_solution,
               const PETScWrappers::MPI::Vector &rhs);
  // GET DIFFERENT FIELDS //
  void get_rho_and_nu(double phi);
  void get_velocity();
  void get_pressure();
  // OTHERS //
  void save_old_solution();

  MPI_Comm &mpi_communicator;
  parallel::distributed::Triangulation<dim> &triangulation;

  int degree_LS;
  DoFHandler<dim> dof_handler_LS;
  FE_Q<dim> fe_LS;
  IndexSet locally_owned_dofs_LS;
  IndexSet locally_relevant_dofs_LS;

  int degree_U;
  DoFHandler<dim> dof_handler_U;
  FE_Q<dim> fe_U;
  IndexSet locally_owned_dofs_U;
  IndexSet locally_relevant_dofs_U;

  DoFHandler<dim> dof_handler_P;
  FE_Q<dim> fe_P;
  IndexSet locally_owned_dofs_P;
  IndexSet locally_relevant_dofs_P;

  Function<dim> &force_function;
  Function<dim> &rho_function;
  Function<dim> &nu_function;

  double rho_air;
  double nu_air;
  double rho_fluid;
  double nu_fluid;

  double time_step;
  double eps;

  bool verbose;
  unsigned int LEVEL_SET;
  unsigned int RHO_TIMES_RHS;

  ConditionalOStream pcout;

  double rho_min;
  double rho_value;
  double nu_value;

  double h;
  double umax;

  int degree_MAX;

  AffineConstraints<double> constraints;
  AffineConstraints<double> constraints_psi;

  std::vector<types::global_dof_index> boundary_values_id_u;
  std::vector<types::global_dof_index> boundary_values_id_v;
  std::vector<types::global_dof_index> boundary_values_id_w;
  std::vector<double> boundary_values_u;
  std::vector<double> boundary_values_v;
  std::vector<double> boundary_values_w;

  PETScWrappers::MPI::SparseMatrix system_Matrix_u;
  PETScWrappers::MPI::SparseMatrix system_Matrix_v;
  PETScWrappers::MPI::SparseMatrix system_Matrix_w;
  bool rebuild_Matrix_U;
  std::shared_ptr<PETScWrappers::PreconditionBoomerAMG> preconditioner_Matrix_u;
  std::shared_ptr<PETScWrappers::PreconditionBoomerAMG> preconditioner_Matrix_v;
  std::shared_ptr<PETScWrappers::PreconditionBoomerAMG> preconditioner_Matrix_w;
  PETScWrappers::MPI::SparseMatrix system_S;
  std::shared_ptr<PETScWrappers::PreconditionBoomerAMG> preconditioner_S;
  PETScWrappers::MPI::SparseMatrix system_M;
  std::shared_ptr<PETScWrappers::PreconditionBoomerAMG> preconditioner_M;
  bool rebuild_S_M;
  bool rebuild_Matrix_U_preconditioners;
  bool rebuild_S_M_preconditioners;
  PETScWrappers::MPI::Vector system_rhs_u;
  PETScWrappers::MPI::Vector system_rhs_v;
  PETScWrappers::MPI::Vector system_rhs_w;
  PETScWrappers::MPI::Vector system_rhs_psi;
  PETScWrappers::MPI::Vector system_rhs_q;
  PETScWrappers::MPI::Vector locally_relevant_solution_phi;
  PETScWrappers::MPI::Vector locally_relevant_solution_u;
  PETScWrappers::MPI::Vector locally_relevant_solution_v;
  PETScWrappers::MPI::Vector locally_relevant_solution_w;
  PETScWrappers::MPI::Vector locally_relevant_solution_u_old;
  PETScWrappers::MPI::Vector locally_relevant_solution_v_old;
  PETScWrappers::MPI::Vector locally_relevant_solution_w_old;

  PETScWrappers::MPI::Vector locally_relevant_solution_psi;
  PETScWrappers::MPI::Vector locally_relevant_solution_psi_old;
  PETScWrappers::MPI::Vector locally_relevant_solution_p;

  PETScWrappers::MPI::Vector completely_distributed_solution_u;
  PETScWrappers::MPI::Vector completely_distributed_solution_v;
  PETScWrappers::MPI::Vector completely_distributed_solution_w;
  PETScWrappers::MPI::Vector completely_distributed_solution_psi;
  PETScWrappers::MPI::Vector completely_distributed_solution_q;
  PETScWrappers::MPI::Vector completely_distributed_solution_p;
};

// CONSTRUCTOR FOR LEVEL SET
template<int dim>
NavierStokesSolver<dim>::NavierStokesSolver(const unsigned int degree_LS,
                                            const unsigned int degree_U,
                                            const double time_step,
                                            const double eps,
                                            const double rho_air,
                                            const double nu_air,
                                            const double rho_fluid,
                                            const double nu_fluid,
                                            Function<dim> &force_function,
                                            const bool verbose,
                                            parallel::distributed::Triangulation<dim> &triangulation,
                                            MPI_Comm &mpi_communicator)
  :
  mpi_communicator(mpi_communicator),
  triangulation(triangulation),
  degree_LS(degree_LS),
  dof_handler_LS(triangulation),
  fe_LS(degree_LS),
  degree_U(degree_U),
  dof_handler_U(triangulation),
  fe_U(degree_U),
  dof_handler_P(triangulation),
  fe_P(degree_U-1),
  force_function(force_function),
  //This is dummy since rho and nu functions won't be used
  rho_function(force_function),
  nu_function(force_function),
  rho_air(rho_air),
  nu_air(nu_air),
  rho_fluid(rho_fluid),
  nu_fluid(nu_fluid),
  time_step(time_step),
  eps(eps),
  verbose(verbose),
  LEVEL_SET(1),
  RHO_TIMES_RHS(1),
  pcout(std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator)==0)),
  rebuild_Matrix_U(true),
  rebuild_S_M(true),
  rebuild_Matrix_U_preconditioners(true),
  rebuild_S_M_preconditioners(true)
{setup();}

// CONSTRUCTOR NOT FOR LEVEL SET
template<int dim>
NavierStokesSolver<dim>::NavierStokesSolver(const unsigned int degree_LS,
                                            const unsigned int degree_U,
                                            const double time_step,
                                            Function<dim> &force_function,
                                            Function<dim> &rho_function,
                                            Function<dim> &nu_function,
                                            const bool verbose,
                                            parallel::distributed::Triangulation<dim> &triangulation,
                                            MPI_Comm &mpi_communicator) :
  mpi_communicator(mpi_communicator),
  triangulation(triangulation),
  degree_LS(degree_LS),
  dof_handler_LS(triangulation),
  fe_LS(degree_LS),
  degree_U(degree_U),
  dof_handler_U(triangulation),
  fe_U(degree_U),
  dof_handler_P(triangulation),
  fe_P(degree_U-1),
  force_function(force_function),
  rho_function(rho_function),
  nu_function(nu_function),
  time_step(time_step),
  verbose(verbose),
  LEVEL_SET(0),
  RHO_TIMES_RHS(0),
  pcout(std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator)==0)),
  rebuild_Matrix_U(true),
  rebuild_S_M(true),
  rebuild_Matrix_U_preconditioners(true),
  rebuild_S_M_preconditioners(true)
{setup();}

template<int dim>
NavierStokesSolver<dim>::~NavierStokesSolver()
{
  dof_handler_LS.clear();
  dof_handler_U.clear();
  dof_handler_P.clear();
}

/////////////////////////////////////////////////////////////
//////////////////// SETTERS AND GETTERS ////////////////////
/////////////////////////////////////////////////////////////
template<int dim>
void NavierStokesSolver<dim>::set_rho_and_nu_functions(const Function<dim> &rho_function,
                                                       const Function<dim> &nu_function)
{
  this->rho_function=rho_function;
  this->nu_function=nu_function;
}

template<int dim>
void NavierStokesSolver<dim>::initial_condition(PETScWrappers::MPI::Vector locally_relevant_solution_phi,
                                                PETScWrappers::MPI::Vector locally_relevant_solution_u,
                                                PETScWrappers::MPI::Vector locally_relevant_solution_v,
                                                PETScWrappers::MPI::Vector locally_relevant_solution_p)
{
  this->locally_relevant_solution_phi=locally_relevant_solution_phi;
  this->locally_relevant_solution_u=locally_relevant_solution_u;
  this->locally_relevant_solution_v=locally_relevant_solution_v;
  this->locally_relevant_solution_p=locally_relevant_solution_p;
  // set old vectors to the initial condition (just for first time step)
  save_old_solution();
}

template<int dim>
void NavierStokesSolver<dim>::initial_condition(PETScWrappers::MPI::Vector locally_relevant_solution_phi,
                                                PETScWrappers::MPI::Vector locally_relevant_solution_u,
                                                PETScWrappers::MPI::Vector locally_relevant_solution_v,
                                                PETScWrappers::MPI::Vector locally_relevant_solution_w,
                                                PETScWrappers::MPI::Vector locally_relevant_solution_p)
{
  this->locally_relevant_solution_phi=locally_relevant_solution_phi;
  this->locally_relevant_solution_u=locally_relevant_solution_u;
  this->locally_relevant_solution_v=locally_relevant_solution_v;
  this->locally_relevant_solution_w=locally_relevant_solution_w;
  this->locally_relevant_solution_p=locally_relevant_solution_p;
  // set old vectors to the initial condition (just for first time step)
  save_old_solution();
}

template<int dim>
void NavierStokesSolver<dim>::set_boundary_conditions(std::vector<types::global_dof_index> boundary_values_id_u,
                                                      std::vector<types::global_dof_index> boundary_values_id_v,
                                                      std::vector<double> boundary_values_u,
                                                      std::vector<double> boundary_values_v)
{
  this->boundary_values_id_u=boundary_values_id_u;
  this->boundary_values_id_v=boundary_values_id_v;
  this->boundary_values_u=boundary_values_u;
  this->boundary_values_v=boundary_values_v;
}

template<int dim>
void NavierStokesSolver<dim>::set_boundary_conditions(std::vector<types::global_dof_index> boundary_values_id_u,
                                                      std::vector<types::global_dof_index> boundary_values_id_v,
                                                      std::vector<types::global_dof_index> boundary_values_id_w,
                                                      std::vector<double> boundary_values_u,
                                                      std::vector<double> boundary_values_v,
                                                      std::vector<double> boundary_values_w)
{
  this->boundary_values_id_u=boundary_values_id_u;
  this->boundary_values_id_v=boundary_values_id_v;
  this->boundary_values_id_w=boundary_values_id_w;
  this->boundary_values_u=boundary_values_u;
  this->boundary_values_v=boundary_values_v;
  this->boundary_values_w=boundary_values_w;
}

template<int dim>
void NavierStokesSolver<dim>::set_velocity(PETScWrappers::MPI::Vector locally_relevant_solution_u,
                                           PETScWrappers::MPI::Vector locally_relevant_solution_v)
{
  this->locally_relevant_solution_u=locally_relevant_solution_u;
  this->locally_relevant_solution_v=locally_relevant_solution_v;
}

template<int dim>
void NavierStokesSolver<dim>::set_velocity(PETScWrappers::MPI::Vector locally_relevant_solution_u,
                                           PETScWrappers::MPI::Vector locally_relevant_solution_v,
                                           PETScWrappers::MPI::Vector locally_relevant_solution_w)
{
  this->locally_relevant_solution_u=locally_relevant_solution_u;
  this->locally_relevant_solution_v=locally_relevant_solution_v;
  this->locally_relevant_solution_w=locally_relevant_solution_w;
}

template<int dim>
void NavierStokesSolver<dim>::set_phi(PETScWrappers::MPI::Vector locally_relevant_solution_phi)
{
  this->locally_relevant_solution_phi=locally_relevant_solution_phi;
}

template<int dim>
void NavierStokesSolver<dim>::get_rho_and_nu(double phi)
{
  double H=0;
  // get rho, nu
  if (phi>eps)
    H=1;
  else if (phi<-eps)
    H=-1;
  else
    H=phi/eps;
  rho_value=rho_fluid*(1+H)/2.+rho_air*(1-H)/2.;
  nu_value=nu_fluid*(1+H)/2.+nu_air*(1-H)/2.;
  //rho_value=rho_fluid*(1+phi)/2.+rho_air*(1-phi)/2.;
  //nu_value=nu_fluid*(1+phi)/2.+nu_air*(1-phi)/2.;
}

template<int dim>
void NavierStokesSolver<dim>::get_pressure(PETScWrappers::MPI::Vector &locally_relevant_solution_p)
{
  locally_relevant_solution_p=this->locally_relevant_solution_p;
}

template<int dim>
void NavierStokesSolver<dim>::get_velocity(PETScWrappers::MPI::Vector &locally_relevant_solution_u,
                                           PETScWrappers::MPI::Vector &locally_relevant_solution_v)
{
  locally_relevant_solution_u=this->locally_relevant_solution_u;
  locally_relevant_solution_v=this->locally_relevant_solution_v;
}

template<int dim>
void NavierStokesSolver<dim>::get_velocity(PETScWrappers::MPI::Vector &locally_relevant_solution_u,
                                           PETScWrappers::MPI::Vector &locally_relevant_solution_v,
                                           PETScWrappers::MPI::Vector &locally_relevant_solution_w)
{
  locally_relevant_solution_u=this->locally_relevant_solution_u;
  locally_relevant_solution_v=this->locally_relevant_solution_v;
  locally_relevant_solution_w=this->locally_relevant_solution_w;
}

///////////////////////////////////////////////////////
///////////// SETUP AND INITIAL CONDITION /////////////
///////////////////////////////////////////////////////
template<int dim>
void NavierStokesSolver<dim>::setup()
{
  pcout<<"***** SETUP IN NAVIER STOKES SOLVER *****"<<std::endl;
  setup_DOF();
  init_constraints();
  setup_VECTORS();
}

template<int dim>
void NavierStokesSolver<dim>::setup_DOF()
{
  rho_min = 1.;
  degree_MAX=std::max(degree_LS,degree_U);
  // setup system LS
  dof_handler_LS.distribute_dofs(fe_LS);
  locally_owned_dofs_LS    = dof_handler_LS.locally_owned_dofs();
  locally_relevant_dofs_LS = DoFTools::extract_locally_relevant_dofs(dof_handler_LS);
  // setup system U
  dof_handler_U.distribute_dofs(fe_U);
  locally_owned_dofs_U    = dof_handler_U.locally_owned_dofs();
  locally_relevant_dofs_U = DoFTools::extract_locally_relevant_dofs(dof_handler_U);
  // setup system P
  dof_handler_P.distribute_dofs(fe_P);
  locally_owned_dofs_P    = dof_handler_P.locally_owned_dofs();
  locally_relevant_dofs_P = DoFTools::extract_locally_relevant_dofs(dof_handler_P);
}

template<int dim>
void NavierStokesSolver<dim>::setup_VECTORS()
{
  // init vectors for phi
  locally_relevant_solution_phi.reinit(locally_owned_dofs_LS,locally_relevant_dofs_LS,
                                       mpi_communicator);
  locally_relevant_solution_phi=0;
  //init vectors for u
  locally_relevant_solution_u.reinit(locally_owned_dofs_U,locally_relevant_dofs_U,
                                     mpi_communicator);
  locally_relevant_solution_u=0;
  completely_distributed_solution_u.reinit(locally_owned_dofs_U,mpi_communicator);
  system_rhs_u.reinit(locally_owned_dofs_U,mpi_communicator);
  //init vectors for u_old
  locally_relevant_solution_u_old.reinit(locally_owned_dofs_U,locally_relevant_dofs_U,
                                         mpi_communicator);
  locally_relevant_solution_u_old=0;
  //init vectors for v
  locally_relevant_solution_v.reinit(locally_owned_dofs_U,locally_relevant_dofs_U,
                                     mpi_communicator);
  locally_relevant_solution_v=0;
  completely_distributed_solution_v.reinit(locally_owned_dofs_U,mpi_communicator);
  system_rhs_v.reinit(locally_owned_dofs_U,mpi_communicator);
  //init vectors for v_old
  locally_relevant_solution_v_old.reinit(locally_owned_dofs_U,locally_relevant_dofs_U,
                                         mpi_communicator);
  locally_relevant_solution_v_old=0;
  //init vectors for w
  locally_relevant_solution_w.reinit(locally_owned_dofs_U,locally_relevant_dofs_U,
                                     mpi_communicator);
  locally_relevant_solution_w=0;
  completely_distributed_solution_w.reinit(locally_owned_dofs_U,mpi_communicator);
  system_rhs_w.reinit(locally_owned_dofs_U,mpi_communicator);
  //init vectors for w_old
  locally_relevant_solution_w_old.reinit(locally_owned_dofs_U,locally_relevant_dofs_U,
                                         mpi_communicator);
  locally_relevant_solution_w_old=0;
  //init vectors for dpsi
  locally_relevant_solution_psi.reinit(locally_owned_dofs_P,locally_relevant_dofs_P,
                                       mpi_communicator);
  locally_relevant_solution_psi=0;
  system_rhs_psi.reinit(locally_owned_dofs_P,mpi_communicator);
  //init vectors for dpsi old
  locally_relevant_solution_psi_old.reinit(locally_owned_dofs_P,locally_relevant_dofs_P,
                                           mpi_communicator);
  locally_relevant_solution_psi_old=0;
  //init vectors for q
  completely_distributed_solution_q.reinit(locally_owned_dofs_P,mpi_communicator);
  system_rhs_q.reinit(locally_owned_dofs_P,mpi_communicator);
  //init vectors for psi
  completely_distributed_solution_psi.reinit(locally_owned_dofs_P,mpi_communicator);
  //init vectors for p
  locally_relevant_solution_p.reinit(locally_owned_dofs_P,locally_relevant_dofs_P,
                                     mpi_communicator);
  locally_relevant_solution_p=0;
  completely_distributed_solution_p.reinit(locally_owned_dofs_P,mpi_communicator);
  ////////////////////////////
  // Initialize constraints //
  ////////////////////////////
  init_constraints();
  //////////////////////
  // Sparsity pattern //
  //////////////////////
  // sparsity pattern for A
  DynamicSparsityPattern dsp_Matrix(locally_relevant_dofs_U);
  DoFTools::make_sparsity_pattern(dof_handler_U,dsp_Matrix,constraints,false);
  SparsityTools::distribute_sparsity_pattern(dsp_Matrix,
                                             dof_handler_U.locally_owned_dofs(),
                                             mpi_communicator,
                                             locally_relevant_dofs_U);
  system_Matrix_u.reinit(dof_handler_U.locally_owned_dofs(),
                         dof_handler_U.locally_owned_dofs(),
                         dsp_Matrix,
                         mpi_communicator);
  system_Matrix_v.reinit(dof_handler_U.locally_owned_dofs(),
                         dof_handler_U.locally_owned_dofs(),
                         dsp_Matrix,
                         mpi_communicator);
  system_Matrix_w.reinit(dof_handler_U.locally_owned_dofs(),
                         dof_handler_U.locally_owned_dofs(),
                         dsp_Matrix,
                         mpi_communicator);
  rebuild_Matrix_U=true;
  // sparsity pattern for S
  DynamicSparsityPattern dsp_S(locally_relevant_dofs_P);
  DoFTools::make_sparsity_pattern(dof_handler_P,dsp_S,constraints_psi,false);
  SparsityTools::distribute_sparsity_pattern(dsp_S,
                                             dof_handler_P.locally_owned_dofs(),
                                             mpi_communicator,
                                             locally_relevant_dofs_P);
  system_S.reinit(dof_handler_P.locally_owned_dofs(),
                  dof_handler_P.locally_owned_dofs(),
                  dsp_S,
                  mpi_communicator);
  // sparsity pattern for M
  DynamicSparsityPattern dsp_M(locally_relevant_dofs_P);
  DoFTools::make_sparsity_pattern(dof_handler_P,dsp_M,constraints_psi,false);
  SparsityTools::distribute_sparsity_pattern(dsp_M,
                                             dof_handler_P.locally_owned_dofs(),
                                             mpi_communicator,
                                             locally_relevant_dofs_P);
  system_M.reinit(dof_handler_P.locally_owned_dofs(),
                  dof_handler_P.locally_owned_dofs(),
                  dsp_M,
                  mpi_communicator);
  rebuild_S_M=true;
}

template<int dim>
void NavierStokesSolver<dim>::init_constraints()
{
  //grl constraints
  constraints.clear();
  constraints.reinit(locally_relevant_dofs_U);
  DoFTools::make_hanging_node_constraints(dof_handler_U,constraints);
  constraints.close();
  //constraints for dpsi
  constraints_psi.clear();
  constraints_psi.reinit(locally_relevant_dofs_P);
  DoFTools::make_hanging_node_constraints(dof_handler_P,constraints_psi);
  //if (constraints_psi.can_store_line(0))
  //constraints_psi.add_line(0); //constraint u0 = 0
  constraints_psi.close();
}

///////////////////////////////////////////////////////
////////////////// ASSEMBLE SYSTEMS ///////////////////
///////////////////////////////////////////////////////
template<int dim>
void NavierStokesSolver<dim>::assemble_system_U()
{
  if (rebuild_Matrix_U==true)
    {
      system_Matrix_u=0;
      system_Matrix_v=0;
      system_Matrix_w=0;
    }
  system_rhs_u=0;
  system_rhs_v=0;
  system_rhs_w=0;

  const QGauss<dim> quadrature_formula(degree_MAX+1);
  FEValues<dim> fe_values_LS(fe_LS,quadrature_formula,
                             update_values|update_gradients|update_quadrature_points|update_JxW_values);
  FEValues<dim> fe_values_U(fe_U,quadrature_formula,
                            update_values|update_gradients|update_quadrature_points|update_JxW_values);
  FEValues<dim> fe_values_P(fe_P,quadrature_formula,
                            update_values|update_gradients|update_quadrature_points|update_JxW_values);

  const unsigned int dofs_per_cell=fe_U.dofs_per_cell;
  const unsigned int n_q_points=quadrature_formula.size();

  FullMatrix<double> cell_A_u(dofs_per_cell,dofs_per_cell);
  Vector<double> cell_rhs_u(dofs_per_cell);
  Vector<double> cell_rhs_v(dofs_per_cell);
  Vector<double> cell_rhs_w(dofs_per_cell);

  std::vector<double> phiqnp1(n_q_points);

  std::vector<double> uqn(n_q_points);
  std::vector<double> uqnm1(n_q_points);
  std::vector<double> vqn(n_q_points);
  std::vector<double> vqnm1(n_q_points);
  std::vector<double> wqn(n_q_points);
  std::vector<double> wqnm1(n_q_points);

  // FOR Explicit nonlinearity
  //std::vector<Tensor<1, dim> > grad_un(n_q_points);
  //std::vector<Tensor<1, dim> > grad_vn(n_q_points);
  //std::vector<Tensor<1, dim> > grad_wn(n_q_points);
  //Tensor<1, dim> Un;

  std::vector<Tensor<1, dim> > grad_pqn(n_q_points);
  std::vector<Tensor<1, dim> > grad_psiqn(n_q_points);
  std::vector<Tensor<1, dim> > grad_psiqnm1(n_q_points);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<Tensor<1, dim> > shape_grad(dofs_per_cell);
  std::vector<double> shape_value(dofs_per_cell);

  double force_u;
  double force_v;
  double force_w;
  double pressure_grad_u;
  double pressure_grad_v;
  double pressure_grad_w;
  double u_star=0;
  double v_star=0;
  double w_star=0;
  double rho_star;
  double rho;
  Vector<double> force_terms(dim);

  typename DoFHandler<dim>::active_cell_iterator
  cell_U=dof_handler_U.begin_active(), endc_U=dof_handler_U.end();
  typename DoFHandler<dim>::active_cell_iterator cell_P=dof_handler_P.begin_active();
  typename DoFHandler<dim>::active_cell_iterator cell_LS=dof_handler_LS.begin_active();

  for (; cell_U!=endc_U; ++cell_U,++cell_P,++cell_LS)
    if (cell_U->is_locally_owned())
      {
        cell_A_u=0;
        cell_rhs_u=0;
        cell_rhs_v=0;
        cell_rhs_w=0;

        fe_values_LS.reinit(cell_LS);
        fe_values_U.reinit(cell_U);
        fe_values_P.reinit(cell_P);

        // get function values for LS
        fe_values_LS.get_function_values(locally_relevant_solution_phi,phiqnp1);
        // get function values for U
        fe_values_U.get_function_values(locally_relevant_solution_u,uqn);
        fe_values_U.get_function_values(locally_relevant_solution_u_old,uqnm1);
        fe_values_U.get_function_values(locally_relevant_solution_v,vqn);
        fe_values_U.get_function_values(locally_relevant_solution_v_old,vqnm1);
        if (dim==3)
          {
            fe_values_U.get_function_values(locally_relevant_solution_w,wqn);
            fe_values_U.get_function_values(locally_relevant_solution_w_old,wqnm1);
          }
        // For explicit nonlinearity
        // get gradient values for U
        //fe_values_U.get_function_gradients(locally_relevant_solution_u,grad_un);
        //fe_values_U.get_function_gradients(locally_relevant_solution_v,grad_vn);
        //if (dim==3)
        //fe_values_U.get_function_gradients(locally_relevant_solution_w,grad_wn);

        // get values and gradients for p and dpsi
        fe_values_P.get_function_gradients(locally_relevant_solution_p,grad_pqn);
        fe_values_P.get_function_gradients(locally_relevant_solution_psi,grad_psiqn);
        fe_values_P.get_function_gradients(locally_relevant_solution_psi_old,grad_psiqnm1);

        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
          {
            const double JxW=fe_values_U.JxW(q_point);
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                shape_grad[i]=fe_values_U.shape_grad(i,q_point);
                shape_value[i]=fe_values_U.shape_value(i,q_point);
              }

            pressure_grad_u=(grad_pqn[q_point][0]+4./3*grad_psiqn[q_point][0]-1./3*grad_psiqnm1[q_point][0]);
            pressure_grad_v=(grad_pqn[q_point][1]+4./3*grad_psiqn[q_point][1]-1./3*grad_psiqnm1[q_point][1]);
            if (dim==3)
              pressure_grad_w=(grad_pqn[q_point][2]+4./3*grad_psiqn[q_point][2]-1./3*grad_psiqnm1[q_point][2]);

            if (LEVEL_SET==1) // use level set to define rho and nu
              get_rho_and_nu(phiqnp1[q_point]);
            else // rho and nu are defined through functions
              {
                rho_value=rho_function.value(fe_values_U.quadrature_point(q_point));
                nu_value=nu_function.value(fe_values_U.quadrature_point(q_point));
              }

            // Non-linearity: for semi-implicit
            u_star=2*uqn[q_point]-uqnm1[q_point];
            v_star=2*vqn[q_point]-vqnm1[q_point];
            if (dim==3)
              w_star=2*wqn[q_point]-wqnm1[q_point];

            // for explicit nonlinearity
            //Un[0] = uqn[q_point];
            //Un[1] = vqn[q_point];
            //if (dim==3)
            //Un[2] = wqn[q_point];

            //double nonlinearity_u = Un*grad_un[q_point];
            //double nonlinearity_v = Un*grad_vn[q_point];
            //double nonlinearity_w = 0;
            //if (dim==3)
            //nonlinearity_w = Un*grad_wn[q_point];

            rho_star=rho_value; // This is because we consider rho*u_t instead of (rho*u)_t
            rho=rho_value;

            // FORCE TERMS
            force_function.vector_value(fe_values_U.quadrature_point(q_point),force_terms);
            force_u=force_terms[0];
            force_v=force_terms[1];
            if (dim==3)
              force_w=force_terms[2];
            if (RHO_TIMES_RHS==1)
              {
                force_u*=rho;
                force_v*=rho;
                if (dim==3)
                  force_w*=rho;
              }

            for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                cell_rhs_u(i)+=((4./3*rho*uqn[q_point]-1./3*rho*uqnm1[q_point]
                                 +2./3*time_step*(force_u-pressure_grad_u)
                                 //-2./3*time_step*rho*nonlinearity_u
                                )*shape_value[i])*JxW;
                cell_rhs_v(i)+=((4./3*rho*vqn[q_point]-1./3*rho*vqnm1[q_point]
                                 +2./3*time_step*(force_v-pressure_grad_v)
                                 //-2./3*time_step*rho*nonlinearity_v
                                )*shape_value[i])*JxW;
                if (dim==3)
                  cell_rhs_w(i)+=((4./3*rho*wqn[q_point]-1./3*rho*wqnm1[q_point]
                                   +2./3*time_step*(force_w-pressure_grad_w)
                                   //-2./3*time_step*rho*nonlinearity_w
                                  )*shape_value[i])*JxW;
                if (rebuild_Matrix_U==true)
                  for (unsigned int j=0; j<dofs_per_cell; ++j)
                    {
                      if (dim==2)
                        cell_A_u(i,j)+=(rho_star*shape_value[i]*shape_value[j]
                                        +2./3*time_step*nu_value*(shape_grad[i]*shape_grad[j])
                                        +2./3*time_step*rho*shape_value[i]
                                        *(u_star*shape_grad[j][0]+v_star*shape_grad[j][1]) // semi-implicit NL
                                       )*JxW;
                      else //dim==3
                        cell_A_u(i,j)+=(rho_star*shape_value[i]*shape_value[j]
                                        +2./3*time_step*nu_value*(shape_grad[i]*shape_grad[j])
                                        +2./3*time_step*rho*shape_value[i]
                                        *(u_star*shape_grad[j][0]+v_star*shape_grad[j][1]+w_star*shape_grad[j][2]) // semi-implicit NL
                                       )*JxW;
                    }
              }
          }
        cell_U->get_dof_indices(local_dof_indices);
        // distribute
        if (rebuild_Matrix_U==true)
          constraints.distribute_local_to_global(cell_A_u,local_dof_indices,system_Matrix_u);
        constraints.distribute_local_to_global(cell_rhs_u,local_dof_indices,system_rhs_u);
        constraints.distribute_local_to_global(cell_rhs_v,local_dof_indices,system_rhs_v);
        if (dim==3)
          constraints.distribute_local_to_global(cell_rhs_w,local_dof_indices,system_rhs_w);
      }
  system_rhs_u.compress(VectorOperation::add);
  system_rhs_v.compress(VectorOperation::add);
  if (dim==3) system_rhs_w.compress(VectorOperation::add);
  if (rebuild_Matrix_U==true)
    {
      system_Matrix_u.compress(VectorOperation::add);
      system_Matrix_v.copy_from(system_Matrix_u);
      if (dim==3)
        system_Matrix_w.copy_from(system_Matrix_u);
    }
  // BOUNDARY CONDITIONS
  system_rhs_u.set(boundary_values_id_u,boundary_values_u);
  system_rhs_u.compress(VectorOperation::insert);
  system_rhs_v.set(boundary_values_id_v,boundary_values_v);
  system_rhs_v.compress(VectorOperation::insert);
  if (dim==3)
    {
      system_rhs_w.set(boundary_values_id_w,boundary_values_w);
      system_rhs_w.compress(VectorOperation::insert);
    }
  if (rebuild_Matrix_U)
    {
      system_Matrix_u.clear_rows(boundary_values_id_u,1);
      system_Matrix_v.clear_rows(boundary_values_id_v,1);
      if (dim==3)
        system_Matrix_w.clear_rows(boundary_values_id_w,1);
      if (rebuild_Matrix_U_preconditioners)
        {
          // PRECONDITIONERS
          rebuild_Matrix_U_preconditioners=false;
          preconditioner_Matrix_u.reset(new PETScWrappers::PreconditionBoomerAMG
                                        (system_Matrix_u,PETScWrappers::PreconditionBoomerAMG::AdditionalData(false)));
          preconditioner_Matrix_v.reset( new PETScWrappers::PreconditionBoomerAMG
                                         (system_Matrix_v,PETScWrappers::PreconditionBoomerAMG::AdditionalData(false)));
          if (dim==3)
            preconditioner_Matrix_w.reset(new PETScWrappers::PreconditionBoomerAMG
                                          (system_Matrix_w,PETScWrappers::PreconditionBoomerAMG::AdditionalData(false)));
        }
    }
  rebuild_Matrix_U=true;
}

template<int dim>
void NavierStokesSolver<dim>::assemble_system_dpsi_q()
{
  if (rebuild_S_M==true)
    {
      system_S=0;
      system_M=0;
    }
  system_rhs_psi=0;
  system_rhs_q=0;

  const QGauss<dim> quadrature_formula(degree_MAX+1);

  FEValues<dim> fe_values_U(fe_U,quadrature_formula,
                            update_values|update_gradients|update_quadrature_points|update_JxW_values);
  FEValues<dim> fe_values_P(fe_P,quadrature_formula,
                            update_values|update_gradients|update_quadrature_points|update_JxW_values);
  FEValues<dim> fe_values_LS(fe_LS,quadrature_formula,
                             update_values|update_gradients|update_quadrature_points|update_JxW_values);

  const unsigned int dofs_per_cell=fe_P.dofs_per_cell;
  const unsigned int n_q_points=quadrature_formula.size();

  FullMatrix<double> cell_S(dofs_per_cell,dofs_per_cell);
  FullMatrix<double> cell_M(dofs_per_cell,dofs_per_cell);
  Vector<double> cell_rhs_psi(dofs_per_cell);
  Vector<double> cell_rhs_q(dofs_per_cell);

  std::vector<double> phiqnp1(n_q_points);
  std::vector<Tensor<1, dim> > gunp1(n_q_points);
  std::vector<Tensor<1, dim> > gvnp1(n_q_points);
  std::vector<Tensor<1, dim> > gwnp1(n_q_points);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<double> shape_value(dofs_per_cell);
  std::vector<Tensor<1, dim> > shape_grad(dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
  cell_P=dof_handler_P.begin_active(), endc_P=dof_handler_P.end();
  typename DoFHandler<dim>::active_cell_iterator cell_U=dof_handler_U.begin_active();
  typename DoFHandler<dim>::active_cell_iterator cell_LS=dof_handler_LS.begin_active();

  for (; cell_P!=endc_P; ++cell_P,++cell_U,++cell_LS)
    if (cell_P->is_locally_owned())
      {
        cell_S=0;
        cell_M=0;
        cell_rhs_psi=0;
        cell_rhs_q=0;

        fe_values_P.reinit(cell_P);
        fe_values_U.reinit(cell_U);
        fe_values_LS.reinit(cell_LS);

        // get function values for LS
        fe_values_LS.get_function_values(locally_relevant_solution_phi,phiqnp1);

        // get function grads for u and v
        fe_values_U.get_function_gradients(locally_relevant_solution_u,gunp1);
        fe_values_U.get_function_gradients(locally_relevant_solution_v,gvnp1);
        if (dim==3)
          fe_values_U.get_function_gradients(locally_relevant_solution_w,gwnp1);

        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
          {
            const double JxW=fe_values_P.JxW(q_point);
            double divU = gunp1[q_point][0]+gvnp1[q_point][1];
            if (dim==3) divU += gwnp1[q_point][2];
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                shape_value[i]=fe_values_P.shape_value(i,q_point);
                shape_grad[i]=fe_values_P.shape_grad(i,q_point);
              }
            if (LEVEL_SET==1) // use level set to define rho and nu
              get_rho_and_nu (phiqnp1[q_point]);
            else // rho and nu are defined through functions
              nu_value=nu_function.value(fe_values_U.quadrature_point(q_point));

            for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                cell_rhs_psi(i)+=-3./2./time_step*rho_min*divU*shape_value[i]*JxW;
                cell_rhs_q(i)-=nu_value*divU*shape_value[i]*JxW;
                if (rebuild_S_M==true)
                  {
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                      if (i==j)
                        {
                          cell_S(i,j)+=shape_grad[i]*shape_grad[j]*JxW+1E-10;
                          cell_M(i,j)+=shape_value[i]*shape_value[j]*JxW;
                        }
                      else
                        {
                          cell_S(i,j)+=shape_grad[i]*shape_grad[j]*JxW;
                          cell_M(i,j)+=shape_value[i]*shape_value[j]*JxW;
                        }
                  }
              }
          }
        cell_P->get_dof_indices(local_dof_indices);
        // Distribute
        if (rebuild_S_M==true)
          {
            constraints_psi.distribute_local_to_global(cell_S,local_dof_indices,system_S);
            constraints_psi.distribute_local_to_global(cell_M,local_dof_indices,system_M);
          }
        constraints_psi.distribute_local_to_global(cell_rhs_q,local_dof_indices,system_rhs_q);
        constraints_psi.distribute_local_to_global(cell_rhs_psi,local_dof_indices,system_rhs_psi);
      }
  if (rebuild_S_M==true)
    {
      system_M.compress(VectorOperation::add);
      system_S.compress(VectorOperation::add);
      if (rebuild_S_M_preconditioners)
        {
          rebuild_S_M_preconditioners=false;
          preconditioner_S.reset(new PETScWrappers::PreconditionBoomerAMG
                                 (system_S,PETScWrappers::PreconditionBoomerAMG::AdditionalData(true)));
          preconditioner_M.reset(new PETScWrappers::PreconditionBoomerAMG
                                 (system_M,PETScWrappers::PreconditionBoomerAMG::AdditionalData(true)));
        }
    }
  system_rhs_psi.compress(VectorOperation::add);
  system_rhs_q.compress(VectorOperation::add);
  rebuild_S_M=false;
}

///////////////////////////////////////////////////////
/////////////////////// SOLVERS ///////////////////////
///////////////////////////////////////////////////////
template<int dim>
void NavierStokesSolver<dim>::solve_U(const AffineConstraints<double> &constraints,
                                      PETScWrappers::MPI::SparseMatrix &Matrix,
                                      std::shared_ptr<PETScWrappers::PreconditionBoomerAMG> preconditioner,
                                      PETScWrappers::MPI::Vector &completely_distributed_solution,
                                      const PETScWrappers::MPI::Vector &rhs)
{
  SolverControl solver_control(dof_handler_U.n_dofs(),1e-6);
  //PETScWrappers::SolverCG solver(solver_control, mpi_communicator);
  //PETScWrappers::SolverGMRES solver(solver_control, mpi_communicator);
  //PETScWrappers::SolverChebychev solver(solver_control, mpi_communicator);
  PETScWrappers::SolverBicgstab solver(solver_control,mpi_communicator);
  constraints.distribute(completely_distributed_solution);
  solver.solve(Matrix,completely_distributed_solution,rhs,*preconditioner);
  constraints.distribute(completely_distributed_solution);
  if (solver_control.last_step() > MAX_NUM_ITER_TO_RECOMPUTE_PRECONDITIONER)
    rebuild_Matrix_U_preconditioners=true;
  if (verbose==true)
    pcout<<"   Solved U in "<<solver_control.last_step()<<" iterations."<<std::endl;
}

template<int dim>
void NavierStokesSolver<dim>::solve_P(const AffineConstraints<double> &constraints,
                                      PETScWrappers::MPI::SparseMatrix &Matrix,
                                      std::shared_ptr<PETScWrappers::PreconditionBoomerAMG> preconditioner,
                                      PETScWrappers::MPI::Vector &completely_distributed_solution,
                                      const PETScWrappers::MPI::Vector &rhs)
{
  SolverControl solver_control(dof_handler_P.n_dofs(),1e-6);
  PETScWrappers::SolverCG solver(solver_control,mpi_communicator);
  //PETScWrappers::SolverGMRES solver(solver_control, mpi_communicator);
  constraints.distribute(completely_distributed_solution);
  solver.solve(Matrix,completely_distributed_solution,rhs,*preconditioner);
  constraints.distribute(completely_distributed_solution);
  if (solver_control.last_step() > MAX_NUM_ITER_TO_RECOMPUTE_PRECONDITIONER)
    rebuild_S_M_preconditioners=true;
  if (verbose==true)
    pcout<<"   Solved P in "<<solver_control.last_step()<<" iterations."<<std::endl;
}

///////////////////////////////////////////////////////
//////////////// get different fields /////////////////
///////////////////////////////////////////////////////
template<int dim>
void NavierStokesSolver<dim>::get_velocity()
{
  assemble_system_U();
  save_old_solution();
  solve_U(constraints,system_Matrix_u,preconditioner_Matrix_u,completely_distributed_solution_u,system_rhs_u);
  locally_relevant_solution_u=completely_distributed_solution_u;
  solve_U(constraints,system_Matrix_v,preconditioner_Matrix_v,completely_distributed_solution_v,system_rhs_v);
  locally_relevant_solution_v=completely_distributed_solution_v;
  if (dim==3)
    {
      solve_U(constraints,system_Matrix_w,preconditioner_Matrix_w,completely_distributed_solution_w,system_rhs_w);
      locally_relevant_solution_w=completely_distributed_solution_w;
    }
}

template<int dim>
void NavierStokesSolver<dim>::get_pressure()
{
  // GET DPSI
  assemble_system_dpsi_q();
  solve_P(constraints_psi,system_S,preconditioner_S,completely_distributed_solution_psi,system_rhs_psi);
  locally_relevant_solution_psi=completely_distributed_solution_psi;
  // SOLVE Q
  solve_P(constraints,system_M,preconditioner_M,completely_distributed_solution_q,system_rhs_q);
  // UPDATE THE PRESSURE
  completely_distributed_solution_p.add(1,completely_distributed_solution_psi);
  completely_distributed_solution_p.add(1,completely_distributed_solution_q);
  locally_relevant_solution_p = completely_distributed_solution_p;
}

///////////////////////////////////////////////////////
/////////////////////// DO STEPS //////////////////////
///////////////////////////////////////////////////////
template<int dim>
void NavierStokesSolver<dim>::nth_time_step()
{
  get_velocity();
  get_pressure();
}

///////////////////////////////////////////////////////
//////////////////////// OTHERS ///////////////////////
///////////////////////////////////////////////////////
template<int dim>
void NavierStokesSolver<dim>::save_old_solution()
{
  locally_relevant_solution_u_old=locally_relevant_solution_u;
  locally_relevant_solution_v_old=locally_relevant_solution_v;
  locally_relevant_solution_w_old=locally_relevant_solution_w;
  locally_relevant_solution_psi_old=locally_relevant_solution_psi;
}

