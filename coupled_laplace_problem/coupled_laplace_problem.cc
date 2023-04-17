// The included deal.II header files are the same as in the other example
// programs:
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
// In addition to the deal.II header files, we include the preCICE API in order
// to obtain access to preCICE specific functionality
#include <precice/SolverInterface.hpp>

#include <fstream>
#include <iostream>

using namespace dealii;

// Configuration parameters
//
// We set up a simple hard-coded struct containing all names we need for
// external coupling. The struct includes the name of the preCICE
// configuration file as well as the name of the simulation participant, the
// name of the coupling mesh and the name of the exchanged data. The last three
// names you also find in the preCICE configuration file. For real application
// cases, these names are better handled by a parameter file.
struct CouplingParamters
{
  const std::string config_file      = "precice-config.xml";
  const std::string participant_name = "laplace-solver";
  const std::string mesh_name        = "dealii-mesh";
  const std::string read_data_name   = "boundary-data";
};


// The Adapter class
//
// The Adapter class handles all functionalities to couple the deal.II solver
// code to other solvers with preCICE, i.e., data structures are set up and all
// relevant information is passed to preCICE.

template <int dim, typename ParameterClass>
class Adapter
{
public:
  Adapter(const ParameterClass &   parameters,
          const types::boundary_id dealii_boundary_interface_id);

  double
  initialize(const DoFHandler<dim> &                    dof_handler,
             std::map<types::global_dof_index, double> &boundary_data,
             const MappingQGeneric<dim> &               mapping);

  double
  advance(std::map<types::global_dof_index, double> &boundary_data,
          const double                               computed_timestep_length);

  // public precCICE solver interface
  precice::SolverInterface precice;

  // Boundary ID of the deal.II triangulation, associated with the coupling
  // interface. The variable is defined in the constructor of this class and
  // intentionally public so that it can be used during the grid generation and
  // system assembly. The only thing, one needs to make sure is that this ID is
  // unique for a particular triangulation.
  const unsigned int dealii_boundary_interface_id;

private:
  // preCICE related initializations
  // These variables are specified in and read from a parameter file, which is
  // in this simple tutorial program the CouplingParameter struct already
  // introduced in the beginning.
  const std::string mesh_name;
  const std::string read_data_name;

  // These IDs are filled by preCICE during the initialization. We set a default
  // value of -1 in order to detect potential errors more easily.
  int mesh_id;
  int read_data_id;
  int n_interface_nodes;

  // DoF IndexSet, containing relevant coupling DoF indices at the coupling
  // boundary
  IndexSet coupling_dofs;

  // Data containers which are passed to preCICE in an appropriate preCICE
  // specific format
  std::vector<int>    interface_nodes_ids;
  std::vector<double> read_data;

  // The MPI rank and total number of MPI ranks is required by preCICE when the
  // SolverInterface is created. Since this tutorial runs only in serial mode we
  // define the variables manually in this class instead of using the regular
  // MPI interface.
  static constexpr int this_mpi_process = 0;
  static constexpr int n_mpi_processes  = 1;

  // Function to transform the obtained data from preCICE into an appropriate
  // map for Dirichlet boundary conditions
  void
  format_precice_to_dealii(
    std::map<types::global_dof_index, double> &boundary_data) const;
};



// In the constructor of the Adapter class, we set up the preCICE
// SolverInterface. We need to tell preCICE our name as participant of the
// simulation and the name of the preCICE configuration file. Both have already
// been specified in the CouplingParameter class above. Thus, we pass the class
// directly to the constructor and read out all relevant information. As a
// second parameter, we need to specify the boundary ID of our triangulation,
// which is associated with the coupling interface.
template <int dim, typename ParameterClass>
Adapter<dim, ParameterClass>::Adapter(
  const ParameterClass &   parameters,
  const types::boundary_id deal_boundary_interface_id)
  : precice(parameters.participant_name,
            parameters.config_file,
            this_mpi_process,
            n_mpi_processes)
  , dealii_boundary_interface_id(deal_boundary_interface_id)
  , mesh_name(parameters.mesh_name)
  , read_data_name(parameters.read_data_name)
{}



// This function initializes preCICE (e.g. establishes communication channels
// and allocates memory) and passes all relevant data to preCICE. For surface
// coupling, relevant data is in particular the location of the data points at
// the associated interface(s). The `boundary_data` is an empty map, which is
// filled by preCICE, i.e., information of the other participant. Throughout
// the system assembly, the map can directly be used in order to apply the
// Dirichlet boundary conditions in the linear system. preCICE returns the
// maximum admissible time-step size during the initialization.
template <int dim, typename ParameterClass>
double
Adapter<dim, ParameterClass>::initialize(
  const DoFHandler<dim> &                    dof_handler,
  std::map<types::global_dof_index, double> &boundary_data,
  const MappingQGeneric<dim> &               mapping)
{
  Assert(dim > 1, ExcNotImplemented());
  AssertDimension(dim, precice.getDimensions());

  // In a first step, we get preCICE specific IDs from preCICE and store them in
  // the respective variables. Later, they are used for data transfer.
  mesh_id      = precice.getMeshID(mesh_name);
  read_data_id = precice.getDataID(read_data_name, mesh_id);


  // Afterwards, we extract the number of interface nodes and the coupling DoFs
  // at the coupling interface from our deal.II solver via
  // `extract_boundary_dofs()`
  std::set<types::boundary_id> couplingBoundary;
  couplingBoundary.insert(dealii_boundary_interface_id);

  // The `ComponentMask()` might be important in case we deal with vector valued
  // problems, because vector valued problems have a DoF for each component.
  coupling_dofs = DoFTools::extract_boundary_dofs(dof_handler,
                                                  ComponentMask(),
                                                  couplingBoundary);

  // The coupling DoFs are used to set up the `boundary_data` map. At the end,
  // we associate here each DoF with a respective boundary value.
  for (const auto i : coupling_dofs)
    boundary_data[i] = 0.0;

  // Since we deal with a scalar problem, the number of DoFs at the particular
  // interface corresponds to the number of interface nodes.
  n_interface_nodes = coupling_dofs.n_elements();

  std::cout << "\t Number of coupling nodes:     " << n_interface_nodes
            << std::endl;

  // Now, we need to tell preCICE the coordinates of the interface nodes. Hence,
  // we set up a std::vector to pass the node positions to preCICE. Each node is
  // specified only once.
  std::vector<double> interface_nodes_positions;
  interface_nodes_positions.reserve(dim * n_interface_nodes);

  // Set up the appropriate size of the data container needed for data
  // exchange. Here, we deal with a scalar problem, so that only a scalar value
  // is read/written per interface node.
  read_data.resize(n_interface_nodes);
  // The IDs are again filled by preCICE during the initializations.
  interface_nodes_ids.resize(n_interface_nodes);

  // The node location is obtained using `map_dofs_to_support_points()`.
  std::map<types::global_dof_index, Point<dim>> support_points;
  DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);

  // `support_points` contains now the coordinates of all DoFs. In the next
  // step, the relevant coordinates are extracted using the IndexSet with the
  // extracted coupling_dofs.
  for (const auto element : coupling_dofs)
    for (int i = 0; i < dim; ++i)
      interface_nodes_positions.push_back(support_points[element][i]);

  // Now we have all information to define the coupling mesh and pass the
  // information to preCICE.
  precice.setMeshVertices(mesh_id,
                          n_interface_nodes,
                          interface_nodes_positions.data(),
                          interface_nodes_ids.data());

  // Then, we initialize preCICE internally calling the API function
  // `initialize()`
  const double max_delta_t = precice.initialize();


  // read first coupling data from preCICE if available (i.e. deal.II is
  // the second participant in a serial coupling scheme)
  if (precice.isReadDataAvailable())
    {
      precice.readBlockScalarData(read_data_id,
                                  n_interface_nodes,
                                  interface_nodes_ids.data(),
                                  read_data.data());

      // After receiving the coupling data in `read_data`, we convert it to
      // the std::map `boundary_data` which is later needed in order to apply
      // Dirichlet boundary conditions
      format_precice_to_dealii(boundary_data);
    }

  return max_delta_t;
}


// The function `advance()` is called in the main time loop after the
// computation in each time step. Here,
// coupling data is passed to and obtained from preCICE.
template <int dim, typename ParameterClass>
double
Adapter<dim, ParameterClass>::advance(
  std::map<types::global_dof_index, double> &boundary_data,
  const double                               computed_timestep_length)
{
  // We specify the computed time-step length and pass it to preCICE. In
  // return, preCICE tells us the maximum admissible time-step size our
  // participant is allowed to compute in order to not exceed the next coupling
  // time step.
  const double max_delta_t = precice.advance(computed_timestep_length);

  // As a next step, we obtain data, i.e. the boundary condition, from another
  // participant. We have already all IDs and just need to convert our obtained
  // data to the deal.II compatible 'boundary map' , which is done in the
  // format_deal_to_precice function.
  precice.readBlockScalarData(read_data_id,
                              n_interface_nodes,
                              interface_nodes_ids.data(),
                              read_data.data());

  format_precice_to_dealii(boundary_data);

  return max_delta_t;
}



// This function takes the std::vector obtained by preCICE in `read_data` and
// inserts the values to the right position in the boundary map used throughout
// our deal.II solver for Dirichlet boundary conditions. The function is only
// used internally in the Adapter class and not called in the solver itself. The
// order, in which preCICE sorts the data in the `read_data` vector is exactly
// the same as the order of the initially passed vertices coordinates.
template <int dim, typename ParameterClass>
void
Adapter<dim, ParameterClass>::format_precice_to_dealii(
  std::map<types::global_dof_index, double> &boundary_data) const
{
  // We already stored the coupling DoF indices in the `boundary_data` map, so
  // that we can simply iterate over all keys in the map.
  auto dof_component = boundary_data.begin();
  for (int i = 0; i < n_interface_nodes; ++i)
    {
      AssertIndexRange(i, read_data.size());
      boundary_data[dof_component->first] = read_data[i];
      ++dof_component;
    }
}


// The solver class is essentially the same as in step-4. We only extend the
// stationary problem to a time-dependent problem and introduced the coupling.
// Comments are added at any point, where the workflow differs from step-4.
template <int dim>
class CoupledLaplaceProblem
{
public:
  CoupledLaplaceProblem();

  void
  run();

private:
  void
  make_grid();
  void
  setup_system();
  void
  assemble_system();
  void
  solve();
  void
  output_results() const;

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;
  MappingQ1<dim>     mapping;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> old_solution;
  Vector<double> system_rhs;

  // We allocate all structures required for the preCICE coupling: The map
  // is used to apply Dirichlet boundary conditions and filled in the Adapter
  // class with data from the other participant. The CouplingParameters hold the
  // preCICE configuration as described above. The interface boundary ID is the
  // ID associated to our coupling interface and needs to be specified, when we
  // set up the Adapter class object, because we pass it directly to the
  // Constructor of this class.
  std::map<types::global_dof_index, double> boundary_data;
  CouplingParamters                         parameters;
  const types::boundary_id                  interface_boundary_id;
  Adapter<dim, CouplingParamters>           adapter;

  // The time-step size delta_t is the acutual time-step size used for all
  // computations. The preCICE time-step size is obtained by preCICE in order to
  // ensure a synchronization at all coupling time steps. The solver time
  // step-size is the desired time-step size of our individual solver. In more
  // sophisticated computations, it might be determined adaptively. The
  // `time_step` counter is just used for the time-step number.
  double       delta_t;
  double       precice_delta_t;
  const double solver_delta_t = 0.1;
  unsigned int time_step      = 0;
};



template <int dim>
class RightHandSide : public Function<dim>
{
public:
  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;
};



template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;
};

template <int dim>
double
RightHandSide<dim>::value(const Point<dim> &p,
                          const unsigned int /*component*/) const
{
  double return_value = 0.0;
  for (unsigned int i = 0; i < dim; ++i)
    return_value += 4.0 * std::pow(p(i), 4.0);

  return return_value;
}


template <int dim>
double
BoundaryValues<dim>::value(const Point<dim> &p,
                           const unsigned int /*component*/) const
{
  return p.square();
}



template <int dim>
CoupledLaplaceProblem<dim>::CoupledLaplaceProblem()
  : fe(1)
  , dof_handler(triangulation)
  , interface_boundary_id(1)
  , adapter(parameters, interface_boundary_id)
{}


template <int dim>
void
CoupledLaplaceProblem<dim>::make_grid()
{
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(4);

  for (const auto &cell : triangulation.active_cell_iterators())
    for (const auto &face : cell->face_iterators())
      {
        // We choose the boundary in positive x direction for the
        // interface coupling.
        if (face->at_boundary() && (face->center()[0] == 1))
          face->set_boundary_id(interface_boundary_id);
      }

  std::cout << "   Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: " << triangulation.n_cells()
            << std::endl;
}


template <int dim>
void
CoupledLaplaceProblem<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  old_solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}



template <int dim>
void
CoupledLaplaceProblem<dim>::assemble_system()
{
  // Reset global structures
  system_rhs    = 0;
  system_matrix = 0;
  // Update old solution values
  old_solution = solution;

  QGauss<dim> quadrature_formula(fe.degree + 1);

  RightHandSide<dim> right_hand_side;

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  // The solution values from previous time steps are stored for each quadrature
  // point
  std::vector<double> local_values_old_solution(fe_values.n_quadrature_points);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;
      // Get the local values from the `fe_values' object
      fe_values.get_function_values(old_solution, local_values_old_solution);

      // The system matrix contains additionally a mass matrix due to the time
      // discretization. The RHS has contributions from the old solution values.
      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        for (const unsigned int i : fe_values.dof_indices())
          {
            for (const unsigned int j : fe_values.dof_indices())
              cell_matrix(i, j) +=
                ((fe_values.shape_value(i, q_index) *  // phi_i(x_q)
                  fe_values.shape_value(j, q_index)) + // phi_j(x_q)
                 (delta_t *                            // delta t
                  fe_values.shape_grad(i, q_index) *   // grad phi_i(x_q)
                  fe_values.shape_grad(j, q_index))) * // grad phi_j(x_q)
                fe_values.JxW(q_index);                // dx

            const auto  x_q         = fe_values.quadrature_point(q_index);
            const auto &local_value = local_values_old_solution[q_index];
            cell_rhs(i) += ((delta_t *                           // delta t
                             fe_values.shape_value(i, q_index) * // phi_i(x_q)
                             right_hand_side.value(x_q)) +       // f(x_q)
                            fe_values.shape_value(i, q_index) *
                              local_value) *       // phi_i(x_q)*val
                           fe_values.JxW(q_index); // dx
          }

      // Copy local to global
      cell->get_dof_indices(local_dof_indices);
      for (const unsigned int i : fe_values.dof_indices())
        {
          for (const unsigned int j : fe_values.dof_indices())
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));

          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }
  {
    // At first, we apply the Dirichlet boundary condition from step-4, as
    // usual.
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             BoundaryValues<dim>(),
                                             boundary_values);
    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       solution,
                                       system_rhs);
  }
  {
    // Afterwards, we apply the coupling boundary condition. The `boundary_data`
    // has already been filled by preCICE.
    MatrixTools::apply_boundary_values(boundary_data,
                                       system_matrix,
                                       solution,
                                       system_rhs);
  }
}



template <int dim>
void
CoupledLaplaceProblem<dim>::solve()
{
  SolverControl            solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence." << std::endl;
}



template <int dim>
void
CoupledLaplaceProblem<dim>::output_results() const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");

  data_out.build_patches(mapping);

  std::ofstream output("solution-" + std::to_string(time_step) + ".vtk");
  data_out.write_vtk(output);
}



template <int dim>
void
CoupledLaplaceProblem<dim>::run()
{
  std::cout << "Solving problem in " << dim << " space dimensions."
            << std::endl;

  make_grid();
  setup_system();

  // After we set up the system, we initialize preCICE using the functionalities
  // of the Adapter. preCICE returns the maximum admissible time-step size,
  // which needs to be compared to our desired solver time-step size.
  precice_delta_t = adapter.initialize(dof_handler, boundary_data, mapping);
  delta_t         = std::min(precice_delta_t, solver_delta_t);

  // preCICE steers the coupled simulation: `isCouplingOngoing` is
  // used to synchronize the end of the simulation with the coupling partner
  while (adapter.precice.isCouplingOngoing())
    {
      // The time step number is solely used to generate unique output files
      ++time_step;
      // In the time loop, we assemble the coupled system and solve it as
      // usual.
      assemble_system();
      solve();

      // After we solved the system, we advance the coupling to the next time
      // level. In a bi-directional coupled simulation, we would pass our
      // calculated data to and obtain new data from preCICE. Here, we simply
      // obtain new data from preCICE, so from the other participant. As before,
      // we obtain a maximum time-step size and compare it against the desired
      // solver time-step size.
      precice_delta_t = adapter.advance(boundary_data, delta_t);
      delta_t         = std::min(precice_delta_t, solver_delta_t);

      // Write an output file if the time step is completed. In case of an
      // implicit coupling, where individual time steps are computed more than
      // once, the function `isTimeWindowCompleted` prevents unnecessary result
      // writing. For this simple tutorial configuration (explicit coupling),
      // the function returns always `true`.
      if (adapter.precice.isTimeWindowComplete())
        output_results();
    }
}



int
main()
{
  CoupledLaplaceProblem<2> laplace_problem;
  laplace_problem.run();

  return 0;
}
