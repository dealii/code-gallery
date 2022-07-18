#ifndef INCLUDE_DG_UPWIND_H_
#define INCLUDE_DG_UPWIND_H_

// The first few files have already been covered in  tutorials and will
// thus not be further commented on:
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
// This header is needed for FEInterfaceValues to compute integrals on
// interfaces:
#include <deal.II/fe/fe_interface_values.h>
//Solver
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/sparse_direct.h>
// We are going to use gradients as refinement indicator.
#include <deal.II/numerics/derivative_approximation.h>
// Using using the mesh_loop from the MeshWorker framework
#include <deal.II/meshworker/mesh_loop.h>

#include <deal.II/base/convergence_table.h>

//To enable parameter handling
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/symbolic_function.h>

#include <deal.II/meshworker/copy_data.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/meshworker/scratch_data.h>

#include <iostream>
#include <fstream>

//This is a struct used only for throwing an exception when theta parameter is not okay.
struct theta_exc
{
	std::string message;
	theta_exc(std::string &&s) : message{std::move(s)} {};
	const char *what() const { return message.c_str(); }
};

using namespace dealii;
// @sect3{Class declaration}
// In the following we have the declaration of the functions used in the program. As we want to use
// parameter files, we need to derive our class from `ParameterAcceptor`.
template <int dim>
class AdvectionReaction : ParameterAcceptor
{
public:
	AdvectionReaction();
	void initialize_params(const std::string &filename);
	void run();

private:
	using Iterator = typename DoFHandler<dim>::active_cell_iterator;
	void parse_string(const std::string &parameters);
	void setup_system();
	void assemble_system();
	void solve();
	void refine_grid();
	void output_results(const unsigned int cycle) const;
	void compute_error();
	double compute_energy_norm();
	void compute_local_projection_and_estimate();

	Triangulation<dim> triangulation;
	const MappingQ1<dim> mapping;

	// Furthermore we want to use DG elements.
	std::unique_ptr<FE_DGQ<dim>> fe;
	DoFHandler<dim> dof_handler;

	SparsityPattern sparsity_pattern;
	SparseMatrix<double> system_matrix;

	Vector<double> solution;
	Vector<double> right_hand_side;
	Vector<double> energy_norm_square_per_cell;
	Vector<double> error_indicator_per_cell;

	// So far we declared the usual objects. Hereafter we declare `FunctionParser<dim>` objects
	FunctionParser<dim> exact_solution;
	FunctionParser<dim> boundary_conditions;
	FunctionParser<dim> rhs;
	FunctionParser<dim> advection_coeff;

	unsigned int fe_degree = 1;

	// and then we define default values that will be parsed from the following strings
	std::string exact_solution_expression = "tanh(100*(x+y-0.5))"; //internal layer solution
	std::string rhs_expression = "-200*tanh(100*x + 100*y - 50.0)^2 + tanh(100*x + 100*y - 50.0) + 200";
	std::string advection_coefficient_expression = "1.0";
	std::string boundary_conditions_expression = "tanh(100*x + 100*y - 50.0)";
	std::string refinement = "residual";
	std::string output_filename = "DG_estimator";
	std::map<std::string, double> constants;
	ParsedConvergenceTable error_table;

	bool use_direct_solver = true;
	unsigned int n_refinement_cycles = 14;
	unsigned int n_global_refinements = 3;
	double theta = 0.5; //default is 0.5 so that I have classical upwind flux
};

#endif /* INCLUDE_DG_UPWIND_H_ */
