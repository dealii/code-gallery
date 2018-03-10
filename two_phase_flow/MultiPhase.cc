#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
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
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/base/parameter_handler.h>
#include <fstream>
#include <iostream>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/mapping_q.h>

using namespace dealii;

///////////////////////////
// FOR TRANSPORT PROBLEM //
///////////////////////////
// TIME_INTEGRATION
#define FORWARD_EULER 0
#define SSP33 1
// PROBLEM 
#define FILLING_TANK 0
#define BREAKING_DAM 1 
#define FALLING_DROP 2
#define SMALL_WAVE_PERTURBATION 3

#include "NavierStokesSolver.cc"
#include "LevelSetSolver.cc"
#include "utilities.cc"

///////////////////////////////////////////////////////
///////////////////// MAIN CLASS //////////////////////
///////////////////////////////////////////////////////
template <int dim>
class MultiPhase
{
public:
  MultiPhase (const unsigned int degree_LS,
	      const unsigned int degree_U);
  ~MultiPhase ();
  void run ();

private:
  void set_boundary_inlet();
  void get_boundary_values_U();
  void get_boundary_values_phi(std::vector<unsigned int> &boundary_values_id_phi,
			       std::vector<double> &boundary_values_phi);
  void output_results();
  void output_vectors();
  void output_rho();
  void setup();
  void initial_condition();
  void init_constraints();

  MPI_Comm mpi_communicator;
  parallel::distributed::Triangulation<dim>   triangulation;
  
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

  DoFHandler<dim>      dof_handler_P;
  FE_Q<dim>            fe_P;
  IndexSet             locally_owned_dofs_P;
  IndexSet             locally_relevant_dofs_P;
  
  ConditionalOStream                pcout;

  // SOLUTION VECTORS
  PETScWrappers::MPI::Vector locally_relevant_solution_phi;
  PETScWrappers::MPI::Vector locally_relevant_solution_u;
  PETScWrappers::MPI::Vector locally_relevant_solution_v;
  PETScWrappers::MPI::Vector locally_relevant_solution_p;
  PETScWrappers::MPI::Vector completely_distributed_solution_phi;
  PETScWrappers::MPI::Vector completely_distributed_solution_u;
  PETScWrappers::MPI::Vector completely_distributed_solution_v;
  PETScWrappers::MPI::Vector completely_distributed_solution_p;
  // BOUNDARY VECTORS
  std::vector<unsigned int> boundary_values_id_u;
  std::vector<unsigned int> boundary_values_id_v;
  std::vector<unsigned int> boundary_values_id_phi;
  std::vector<double> boundary_values_u;
  std::vector<double> boundary_values_v;
  std::vector<double> boundary_values_phi;

  ConstraintMatrix     constraints;

  double time;
  double time_step;
  double final_time;
  unsigned int timestep_number;
  double cfl;
  double umax;
  double min_h;

  double sharpness; 
  int sharpness_integer;

  unsigned int n_refinement;
  unsigned int output_number;
  double output_time;
  bool get_output;

  bool verbose;

  //FOR NAVIER STOKES
  double rho_fluid;
  double nu_fluid;
  double rho_air;
  double nu_air;
  double nu;
  double eps;

  //FOR TRANSPORT
  double cK; //compression coeff
  double cE; //entropy-visc coeff
  unsigned int TRANSPORT_TIME_INTEGRATION;
  std::string ALGORITHM;
  unsigned int PROBLEM;
};

template <int dim>
MultiPhase<dim>::MultiPhase (const unsigned int degree_LS, 
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
  dof_handler_P (triangulation),
  fe_P (degree_U-1), 
  pcout (std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator)== 0))
{}

template <int dim>
MultiPhase<dim>::~MultiPhase ()
{
  dof_handler_LS.clear ();
  dof_handler_U.clear ();
  dof_handler_P.clear ();
}

/////////////////////////////////////////
///////////////// SETUP /////////////////
/////////////////////////////////////////
template <int dim>
void MultiPhase<dim>::setup()
{ 
  // setup system LS
  dof_handler_LS.distribute_dofs (fe_LS);
  locally_owned_dofs_LS = dof_handler_LS.locally_owned_dofs ();
  DoFTools::extract_locally_relevant_dofs (dof_handler_LS,
					   locally_relevant_dofs_LS);
  // setup system U 
  dof_handler_U.distribute_dofs (fe_U);
  locally_owned_dofs_U = dof_handler_U.locally_owned_dofs ();
  DoFTools::extract_locally_relevant_dofs (dof_handler_U,
					   locally_relevant_dofs_U);
  // setup system P //
  dof_handler_P.distribute_dofs (fe_P);
  locally_owned_dofs_P = dof_handler_P.locally_owned_dofs ();
  DoFTools::extract_locally_relevant_dofs (dof_handler_P,
					   locally_relevant_dofs_P);
  // init vectors for phi
  locally_relevant_solution_phi.reinit(locally_owned_dofs_LS,locally_relevant_dofs_LS,mpi_communicator);
  locally_relevant_solution_phi = 0;
  completely_distributed_solution_phi.reinit (locally_owned_dofs_P,mpi_communicator);
  //init vectors for u
  locally_relevant_solution_u.reinit (locally_owned_dofs_U,locally_relevant_dofs_U,mpi_communicator);
  locally_relevant_solution_u = 0;
  completely_distributed_solution_u.reinit (locally_owned_dofs_U,mpi_communicator);
  //init vectors for v                                           
  locally_relevant_solution_v.reinit (locally_owned_dofs_U,locally_relevant_dofs_U,mpi_communicator);
  locally_relevant_solution_v = 0;
  completely_distributed_solution_v.reinit (locally_owned_dofs_U,mpi_communicator);
  //init vectors for p
  locally_relevant_solution_p.reinit (locally_owned_dofs_P,locally_relevant_dofs_P,mpi_communicator);
  locally_relevant_solution_p = 0;
  completely_distributed_solution_p.reinit (locally_owned_dofs_P,mpi_communicator);
  // INIT CONSTRAINTS
  init_constraints();
}

template <int dim>
void MultiPhase<dim>::initial_condition()
{
  time=0;
  // Initial conditions //
  // init condition for phi
  completely_distributed_solution_phi = 0;
  VectorTools::interpolate(dof_handler_LS,
  		   InitialPhi<dim>(PROBLEM, sharpness),
  		   completely_distributed_solution_phi);
  constraints.distribute (completely_distributed_solution_phi);
  locally_relevant_solution_phi = completely_distributed_solution_phi;
  // init condition for u=0
  completely_distributed_solution_u = 0;
  VectorTools::interpolate(dof_handler_U,
			   ZeroFunction<dim>(),
			   completely_distributed_solution_u);
  constraints.distribute (completely_distributed_solution_u);
  locally_relevant_solution_u = completely_distributed_solution_u;
  // init condition for v
  completely_distributed_solution_v = 0;
  VectorTools::interpolate(dof_handler_U,
			   ZeroFunction<dim>(),
			   completely_distributed_solution_v);
  constraints.distribute (completely_distributed_solution_v);
  locally_relevant_solution_v = completely_distributed_solution_v;
  // init condition for p
  completely_distributed_solution_p = 0;
  VectorTools::interpolate(dof_handler_P,
			   ZeroFunction<dim>(),
			   completely_distributed_solution_p);
  constraints.distribute (completely_distributed_solution_p);
  locally_relevant_solution_p = completely_distributed_solution_p;
}
  
template <int dim>
void MultiPhase<dim>::init_constraints()
{
  constraints.clear ();
  constraints.reinit (locally_relevant_dofs_LS);
  DoFTools::make_hanging_node_constraints (dof_handler_LS, constraints);
  constraints.close ();
}

template <int dim>
void MultiPhase<dim>::get_boundary_values_U()
{
  std::map<unsigned int, double> map_boundary_values_u;
  std::map<unsigned int, double> map_boundary_values_v;
  std::map<unsigned int, double> map_boundary_values_w;

  // NO-SLIP CONDITION 
  if (PROBLEM==BREAKING_DAM || PROBLEM==FALLING_DROP)
    {
      //LEFT
      VectorTools::interpolate_boundary_values (dof_handler_U,0,ZeroFunction<dim>(),map_boundary_values_u); 
      VectorTools::interpolate_boundary_values (dof_handler_U,0,ZeroFunction<dim>(),map_boundary_values_v); 
      // RIGHT
      VectorTools::interpolate_boundary_values (dof_handler_U,1,ZeroFunction<dim>(),map_boundary_values_u); 
      VectorTools::interpolate_boundary_values (dof_handler_U,1,ZeroFunction<dim>(),map_boundary_values_v); 
      // BOTTOM 
      VectorTools::interpolate_boundary_values (dof_handler_U,2,ZeroFunction<dim>(),map_boundary_values_u); 
      VectorTools::interpolate_boundary_values (dof_handler_U,2,ZeroFunction<dim>(),map_boundary_values_v); 
      // TOP
      VectorTools::interpolate_boundary_values (dof_handler_U,3,ZeroFunction<dim>(),map_boundary_values_u); 
      VectorTools::interpolate_boundary_values (dof_handler_U,3,ZeroFunction<dim>(),map_boundary_values_v); 
    } 
  else if (PROBLEM==SMALL_WAVE_PERTURBATION)
    { // no slip in bottom and top and slip in left and right
      //LEFT
      VectorTools::interpolate_boundary_values (dof_handler_U,0,ZeroFunction<dim>(),map_boundary_values_u); 
      // RIGHT
      VectorTools::interpolate_boundary_values (dof_handler_U,1,ZeroFunction<dim>(),map_boundary_values_u); 
      // BOTTOM 
      VectorTools::interpolate_boundary_values (dof_handler_U,2,ZeroFunction<dim>(),map_boundary_values_u); 
      VectorTools::interpolate_boundary_values (dof_handler_U,2,ZeroFunction<dim>(),map_boundary_values_v); 
      // TOP
      VectorTools::interpolate_boundary_values (dof_handler_U,3,ZeroFunction<dim>(),map_boundary_values_u); 
      VectorTools::interpolate_boundary_values (dof_handler_U,3,ZeroFunction<dim>(),map_boundary_values_v); 
    }
  else if (PROBLEM==FILLING_TANK)	   
    {
      //LEFT: entry in x, zero in y
      VectorTools::interpolate_boundary_values (dof_handler_U,0,BoundaryU<dim>(PROBLEM),map_boundary_values_u);
      VectorTools::interpolate_boundary_values (dof_handler_U,0,ZeroFunction<dim>(),map_boundary_values_v);
      //RIGHT: no-slip condition
      VectorTools::interpolate_boundary_values (dof_handler_U,1,ZeroFunction<dim>(),map_boundary_values_u);
      VectorTools::interpolate_boundary_values (dof_handler_U,1,ZeroFunction<dim>(),map_boundary_values_v);
      //BOTTOM: non-slip
      VectorTools::interpolate_boundary_values (dof_handler_U,2,ZeroFunction<dim>(),map_boundary_values_u);
      VectorTools::interpolate_boundary_values (dof_handler_U,2,ZeroFunction<dim>(),map_boundary_values_v);
      //TOP: exit in y, zero in x
      VectorTools::interpolate_boundary_values (dof_handler_U,3,ZeroFunction<dim>(),map_boundary_values_u);
      VectorTools::interpolate_boundary_values (dof_handler_U,3,BoundaryV<dim>(PROBLEM),map_boundary_values_v);
    }
  else 
    {
      pcout << "Error in type of PROBLEM at Boundary Conditions" << std::endl;
      abort();
    }
  boundary_values_id_u.resize(map_boundary_values_u.size());
  boundary_values_id_v.resize(map_boundary_values_v.size());
  boundary_values_u.resize(map_boundary_values_u.size());
  boundary_values_v.resize(map_boundary_values_v.size());
  std::map<unsigned int,double>::const_iterator boundary_value_u =map_boundary_values_u.begin();
  std::map<unsigned int,double>::const_iterator boundary_value_v =map_boundary_values_v.begin();
  
  for (int i=0; boundary_value_u !=map_boundary_values_u.end(); ++boundary_value_u, ++i)
    {
      boundary_values_id_u[i]=boundary_value_u->first;
      boundary_values_u[i]=boundary_value_u->second;
    }
  for (int i=0; boundary_value_v !=map_boundary_values_v.end(); ++boundary_value_v, ++i)
    {
      boundary_values_id_v[i]=boundary_value_v->first;
      boundary_values_v[i]=boundary_value_v->second;
    }
}

template <int dim>
void MultiPhase<dim>::set_boundary_inlet()
{
  const QGauss<dim-1>  face_quadrature_formula(1); // center of the face
  FEFaceValues<dim> fe_face_values (fe_U,face_quadrature_formula,
				    update_values | update_quadrature_points |
				    update_normal_vectors);
  const unsigned int n_face_q_points = face_quadrature_formula.size();
  std::vector<double>  u_value (n_face_q_points);
  std::vector<double>  v_value (n_face_q_points); 
  
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
	    u[0]=u_value[0];
	    u[1]=v_value[0];
	    if (fe_face_values.normal_vector(0)*u < -1e-14)
	      cell_U->face(face)->set_boundary_id(10); // SET ID 10 to inlet BOUNDARY (10 is an arbitrary number)
	  }    
}

template <int dim>
void MultiPhase<dim>::get_boundary_values_phi(std::vector<unsigned int> &boundary_values_id_phi,
					      std::vector<double> &boundary_values_phi)
{
  std::map<unsigned int, double> map_boundary_values_phi;
  unsigned int boundary_id=0;
  
  set_boundary_inlet();
  boundary_id=10; // inlet
  VectorTools::interpolate_boundary_values (dof_handler_LS,boundary_id,BoundaryPhi<dim>(1.0),map_boundary_values_phi);
  boundary_values_id_phi.resize(map_boundary_values_phi.size());
  boundary_values_phi.resize(map_boundary_values_phi.size());  
  std::map<unsigned int,double>::const_iterator boundary_value_phi = map_boundary_values_phi.begin();
  for (int i=0; boundary_value_phi !=map_boundary_values_phi.end(); ++boundary_value_phi, ++i)
    {
      boundary_values_id_phi[i]=boundary_value_phi->first;
      boundary_values_phi[i]=boundary_value_phi->second;
    }
}

template<int dim>
void MultiPhase<dim>::output_results()
{
  //output_vectors();
  output_rho();
  output_number++;
}

template <int dim>
void MultiPhase<dim>::output_vectors()
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler_LS);  
  data_out.add_data_vector (locally_relevant_solution_phi, "phi");
  data_out.build_patches ();
  
  const std::string filename = ("sol_vectors-" +
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
	filenames.push_back ("sol_vectors-" +
			     Utilities::int_to_string (output_number, 3) +
			     "." +
			     Utilities::int_to_string (i, 4) +
			     ".vtu");
      
      std::ofstream master_output ((filename + ".pvtu").c_str());
      data_out.write_pvtu_record (master_output, filenames);
    }
}

template <int dim>
void MultiPhase<dim>::output_rho()
{
  Postprocessor<dim> postprocessor(eps,rho_air,rho_fluid);  
  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler_LS);  
  data_out.add_data_vector (locally_relevant_solution_phi, postprocessor);
  
  data_out.build_patches ();
  
  const std::string filename = ("sol_rho-" +
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
	filenames.push_back ("sol_rho-" +
			     Utilities::int_to_string (output_number, 3) +
			     "." +
			     Utilities::int_to_string (i, 4) +
			     ".vtu");
      
      std::ofstream master_output ((filename + ".pvtu").c_str());
      data_out.write_pvtu_record (master_output, filenames);
    }
}

template <int dim>
void MultiPhase<dim>::run()
{
  ////////////////////////
  // GENERAL PARAMETERS //
  ////////////////////////
  umax=1;
  cfl=0.1;
  verbose = true;
  get_output = true;
  output_number = 0;
  n_refinement=8;
  output_time = 0.1;
  final_time = 10.0;
  //////////////////////////////////////////////
  // PARAMETERS FOR THE NAVIER STOKES PROBLEM //
  //////////////////////////////////////////////
  rho_fluid = 1000.;
  nu_fluid = 1.0;
  rho_air = 1.0;
  nu_air = 1.8e-2;
  PROBLEM=BREAKING_DAM;
  //PROBLEM=FILLING_TANK;
  //PROBLEM=SMALL_WAVE_PERTURBATION;
  //PROBLEM=FALLING_DROP;
  
  ForceTerms<dim> force_function(std::vector<double>{0.0,-1.0});
  //////////////////////////////////////
  // PARAMETERS FOR TRANSPORT PROBLEM //
  //////////////////////////////////////
  cK = 1.0;
  cE = 1.0;
  sharpness_integer=10; //this will be multipled by min_h
  //TRANSPORT_TIME_INTEGRATION=FORWARD_EULER;
  TRANSPORT_TIME_INTEGRATION=SSP33;
  //ALGORITHM = "MPP_u1";
  //ALGORITHM = "NMPP_uH";
  ALGORITHM = "MPP_uH";

  // ADJUST PARAMETERS ACCORDING TO PROBLEM 
  if (PROBLEM==FALLING_DROP)
    n_refinement=7;

  //////////////
  // GEOMETRY //
  //////////////
  if (PROBLEM==FILLING_TANK)
    GridGenerator::hyper_rectangle(triangulation,
				   Point<dim>(0.0,0.0), Point<dim>(0.4,0.4), true);
  else if (PROBLEM==BREAKING_DAM || PROBLEM==SMALL_WAVE_PERTURBATION)
    {
      std::vector< unsigned int > repetitions;
      repetitions.push_back(2);
      repetitions.push_back(1);
      GridGenerator::subdivided_hyper_rectangle 
	(triangulation, repetitions, Point<dim>(0.0,0.0), Point<dim>(1.0,0.5), true);
    }
  else if (PROBLEM==FALLING_DROP)
    {
      std::vector< unsigned int > repetitions;
      repetitions.push_back(1);
      repetitions.push_back(4);
      GridGenerator::subdivided_hyper_rectangle 
	(triangulation, repetitions, Point<dim>(0.0,0.0), Point<dim>(0.3,0.9), true);
    }
  triangulation.refine_global (n_refinement);
  // SETUP
  setup();

  // PARAMETERS FOR TIME STEPPING
  min_h = GridTools::minimal_cell_diameter(triangulation)/std::sqrt(2);
  time_step = cfl*min_h/umax;
  eps=1.*min_h; //For reconstruction of density in Navier Stokes
  sharpness=sharpness_integer*min_h; //adjust value of sharpness (for init cond of phi)
  
  // INITIAL CONDITIONS
  initial_condition();
  output_results();
  
  // NAVIER STOKES SOLVER
  NavierStokesSolver<dim> navier_stokes (degree_LS,degree_U,
					 time_step,eps,
					 rho_air,nu_air,
					 rho_fluid,nu_fluid,
					 force_function,
					 verbose,
					 triangulation,mpi_communicator);
  // BOUNDARY CONDITIONS FOR NAVIER STOKES
  get_boundary_values_U();
  navier_stokes.set_boundary_conditions(boundary_values_id_u, boundary_values_id_v,
					boundary_values_u, boundary_values_v);

  //set INITIAL CONDITION within NAVIER STOKES
  navier_stokes.initial_condition(locally_relevant_solution_phi,
				  locally_relevant_solution_u,
				  locally_relevant_solution_v,
				  locally_relevant_solution_p);
  // TRANSPORT SOLVER
  LevelSetSolver<dim> transport_solver (degree_LS,degree_U,
					time_step,cK,cE, 
					verbose, 
					ALGORITHM,
					TRANSPORT_TIME_INTEGRATION,
					triangulation, 
					mpi_communicator); 
  // BOUNDARY CONDITIONS FOR PHI
  get_boundary_values_phi(boundary_values_id_phi,boundary_values_phi);
  transport_solver.set_boundary_conditions(boundary_values_id_phi,boundary_values_phi);

  //set INITIAL CONDITION within TRANSPORT PROBLEM
  transport_solver.initial_condition(locally_relevant_solution_phi,
				     locally_relevant_solution_u,
				     locally_relevant_solution_v);
  int dofs_U = 2*dof_handler_U.n_dofs();
  int dofs_P = 2*dof_handler_P.n_dofs();
  int dofs_LS = dof_handler_LS.n_dofs();
  int dofs_TOTAL = dofs_U+dofs_P+dofs_LS;

  // NO BOUNDARY CONDITIONS for LEVEL SET
  pcout << "Cfl: " << cfl << "; umax: " << umax << "; min h: " << min_h 
	<< "; time step: " << time_step << std::endl;
  pcout << "   Number of active cells:       " 
	<< triangulation.n_global_active_cells() << std::endl
	<< "   Number of degrees of freedom: " << std::endl
	<< "      U: " << dofs_U << std::endl
	<< "      P: " << dofs_P << std::endl
	<< "      LS: " << dofs_LS << std::endl
	<< "      TOTAL: " << dofs_TOTAL
	<< std::endl;

  // TIME STEPPING
  for (timestep_number=1, time=time_step; time<=final_time;
       time+=time_step,++timestep_number)
    {
      pcout << "Time step " << timestep_number 
	    << " at t=" << time 
	    << std::endl;
      // GET NAVIER STOKES VELOCITY
      navier_stokes.set_phi(locally_relevant_solution_phi);
      navier_stokes.nth_time_step(); 
      navier_stokes.get_velocity(locally_relevant_solution_u,locally_relevant_solution_v);
      transport_solver.set_velocity(locally_relevant_solution_u,locally_relevant_solution_v);
      // GET LEVEL SET SOLUTION
      transport_solver.nth_time_step();
      transport_solver.get_unp1(locally_relevant_solution_phi);      
      if (get_output && time-(output_number)*output_time>0)
	output_results();
    }
  navier_stokes.get_velocity(locally_relevant_solution_u, locally_relevant_solution_v);
  transport_solver.get_unp1(locally_relevant_solution_phi);      
  if (get_output)
    output_results();
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
	unsigned int degree_LS = 1;
	unsigned int degree_U = 2;
        MultiPhase<2> multi_phase(degree_LS, degree_U);
        multi_phase.run();
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
