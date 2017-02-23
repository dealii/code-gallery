/*
Viscoelastoplastic relaxation of Ceres
Author: Roger R. Fu
Adapted from Fu et al. 2014 Icarus 240, 133-145 starting Oct. 19, 2014
 */

/*
Summary of output files:

One per run:

- initial_mesh.eps					:  Visualization of initially imported mesh
- physical_times.txt				:  Columns are (1) step number corresponding to other files, (2) physical times at the time when each calculation is run in sec, (3) number of the final plasticity iteration in each timestep.  Written in do_elastic_steps() for elastic steps and do_flow_step() for viscous steps

One per time step:

- timeXX_elastic_displacements.txt	:  Vtk-readable file with columns (1) x, (2) y, (3) u_x, (4) u_y, (5) P.  Written in output_results() function, which is run immediately after solve().
- timeXX_baseviscosities.txt		:  Columns (1) cell x, (2) cell y, (3) base viscosity in Pa s.  Written in solution_stresses().
- timeXX_surface.txt				:  Surface (defined as where P=0 boundary condition is applied) vertices at the beginning of timestep, except for the final timestep.  Written in write_vertices() function, which is called immediately after setup_dofs() except for the final iteration, when it is called after move_mesh()

One per plasticity step:

- timeXX_flowYY.txt					:  Same as timeXX_elastic_displacements.txt above
- timeXX_principalstressesYY.txt	:  Columns with sigma1 and sigma3 at each cell.  Same order as timeXX_baseviscosities.txt.  Written in solution_stresses().
- timeXX_stresstensorYY.txt			:  Columns with components 11, 22, 33, and 13 of stress tensor at each cell.  Written in solution_stresses().
- timeXX_failurelocationsYY.txt		:  Gives x,y coordinates of all cells where failure occurred.  Written in solution_stresses().
- timeXX_viscositiesregYY.txt		:  Gives smoothed and regularized (i.e., floor and ceiling-filtered) effective viscosities.  Written at end of solution_stresses().

*/

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/numerics/fe_field_function.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>
#include <math.h>
#include <armadillo>

#include "../support_code/ellipsoid_grav.h"
#include "../support_code/ellipsoid_fit.h"
#include "../support_code/config_in.h"

// As in all programs, the namespace dealii
// is included:
namespace Step22 {

using namespace dealii;
using namespace arma;

template<int dim>
struct InnerPreconditioner;

template<>
struct InnerPreconditioner<2> {
	typedef SparseDirectUMFPACK type;
};

template<>
struct InnerPreconditioner<3> {
	typedef SparseILU<double> type;
};

// Auxiliary functions

template<int dim>
class AuxFunctions {
public:
	Tensor<2, 2> get_rotation_matrix(const std::vector<Tensor<1, 2> > &grad_u);
};

template<int dim>
Tensor<2, 2> AuxFunctions<dim>::get_rotation_matrix(
		const std::vector<Tensor<1, 2> > &grad_u) {
	const double curl = (grad_u[1][0] - grad_u[0][1]);

	const double angle = std::atan(curl);

	const double t[2][2] = { { cos(angle), sin(angle) }, { -sin(angle), cos(
			angle) } };
	return Tensor<2, 2>(t);
}

// Class for remembering material state/properties at each quadrature point

template<int dim>
struct PointHistory {
	SymmetricTensor<2, dim> old_stress;
	double old_phiphi_stress;
	double first_eta;
	double new_eta;
	double G;
};

// Primary class of this problem

template<int dim>
class StokesProblem {
public:
	StokesProblem(const unsigned int degree);
	void run();

private:
	void setup_dofs();
	void assemble_system();
	void solve();
	void output_results() const;
	void refine_mesh();
	void solution_stesses();
	void smooth_eta_field(std::vector<bool> failing_cells);

	void setup_initial_mesh();
	void do_elastic_steps();
	void do_flow_step();
	void update_time_interval();
	void initialize_eta_and_G();
	void move_mesh();
	void do_ellipse_fits();
	void append_physical_times(int max_plastic);
	void write_vertices(unsigned char);
	void write_mesh();
	void setup_quadrature_point_history();
	void update_quadrature_point_history();

	const unsigned int degree;

	Triangulation<dim> triangulation;
	const MappingQ1<dim> mapping;
	FESystem<dim> fe;
	DoFHandler<dim> dof_handler;
	unsigned int n_u = 0, n_p = 0;
	unsigned int plastic_iteration = 0;
	unsigned int last_max_plasticity = 0;

	QGauss<dim> quadrature_formula;
	std::vector< std::vector <Vector<double> > > quad_viscosities; // Indices for this object are [cell][q][q coords, eta]
	std::vector<double> cell_viscosities; // This vector is only used for output, not FE computations
	std::vector<PointHistory<dim> > quadrature_point_history;

	ConstraintMatrix constraints;

	BlockSparsityPattern sparsity_pattern;
	BlockSparseMatrix<double> system_matrix;

	BlockVector<double> solution;
	BlockVector<double> system_rhs;

	std_cxx1x::shared_ptr<typename InnerPreconditioner<dim>::type> A_preconditioner;

	ellipsoid_fit<dim>   ellipsoid;
};

// Class for boundary conditions and rhs

template<int dim>
class BoundaryValuesP: public Function<dim> {
public:
	BoundaryValuesP() :
			Function<dim>(dim + 1) {
	}

	virtual double value(const Point<dim> &p,
			const unsigned int component = 0) const;

	virtual void vector_value(const Point<dim> &p, Vector<double> &value) const;
};

template<int dim>
double BoundaryValuesP<dim>::value(const Point<dim> &p,
		const unsigned int component) const {
	Assert(component < this->n_components,
			ExcIndexRange (component, 0, this->n_components));

	Assert(p[0] >= -10, ExcLowerRange (p[0], 0)); //value of -10 is to permit some small numerical error moving nodes left of x=0; a << value is in fact sufficient

	return 0;
}

template<int dim>
void BoundaryValuesP<dim>::vector_value(const Point<dim> &p,
		Vector<double> &values) const {
	for (unsigned int c = 0; c < this->n_components; ++c)
		values(c) = BoundaryValuesP<dim>::value(p, c);
}

template<int dim>
class RightHandSide: public Function<dim> {
public:
	RightHandSide () : Function<dim>(dim+1) {}

	virtual double value(const Point<dim> &p, const unsigned int component,
			A_Grav_namespace::AnalyticGravity<dim> *aGrav) const;

	virtual void vector_value(const Point<dim> &p, Vector<double> &value,
			A_Grav_namespace::AnalyticGravity<dim> *aGrav) const;

	virtual void vector_value_list(const std::vector<Point<dim> > &points,
			std::vector<Vector<double> > &values,
			A_Grav_namespace::AnalyticGravity<dim> *aGrav) const;

};

template<int dim>
double RightHandSide<dim>::value(const Point<dim> &p,
		const unsigned int component,
		A_Grav_namespace::AnalyticGravity<dim> *aGrav) const {

	std::vector<double> temp_vector(2);
	aGrav->get_gravity(p, temp_vector);

	if (component == 0) {
		return temp_vector[0] + system_parameters::omegasquared * p[0];	// * 1.2805;
	} else {
		if (component == 1)
			return temp_vector[1];
		else
			return 0;
	}
}

template<int dim>
void RightHandSide<dim>::vector_value(const Point<dim> &p,
		Vector<double> &values,
		A_Grav_namespace::AnalyticGravity<dim> *aGrav) const {
	for (unsigned int c = 0; c < this->n_components; ++c)
		values(c) = RightHandSide<dim>::value(p, c, aGrav);
}

template<int dim>
void RightHandSide<dim>::vector_value_list(
		const std::vector<Point<dim> > &points,
		std::vector<Vector<double> > &values,
		A_Grav_namespace::AnalyticGravity<dim> *aGrav) const {
	// check whether component is in
	// the valid range is up to the
	// derived class
	Assert(values.size() == points.size(),
			ExcDimensionMismatch(values.size(), points.size()));

	for (unsigned int i = 0; i < points.size(); ++i)
		this->vector_value(points[i], values[i], aGrav);
}

// Class for linear solvers and preconditioners

template<class Matrix, class Preconditioner>
class InverseMatrix: public Subscriptor {
public:
	InverseMatrix(const Matrix &m, const Preconditioner &preconditioner);

	void vmult(Vector<double> &dst, const Vector<double> &src) const;

private:
	const SmartPointer<const Matrix> matrix;
	const SmartPointer<const Preconditioner> preconditioner;
};

template<class Matrix, class Preconditioner>
InverseMatrix<Matrix, Preconditioner>::InverseMatrix(const Matrix &m,
		const Preconditioner &preconditioner) :
		matrix(&m), preconditioner(&preconditioner) {
}

template<class Matrix, class Preconditioner>
void InverseMatrix<Matrix, Preconditioner>::vmult(Vector<double> &dst,
		const Vector<double> &src) const {
	SolverControl solver_control(1000 * src.size(), 1e-9 * src.l2_norm());

	SolverCG<> cg(solver_control);

	dst = 0;

	cg.solve(*matrix, dst, src, *preconditioner);
}

// Class for the SchurComplement

template<class Preconditioner>
class SchurComplement: public Subscriptor {
public:
	SchurComplement(const BlockSparseMatrix<double> &system_matrix,
			const InverseMatrix<SparseMatrix<double>, Preconditioner> &A_inverse);

	void vmult(Vector<double> &dst, const Vector<double> &src) const;

private:
	const SmartPointer<const BlockSparseMatrix<double> > system_matrix;
	const SmartPointer<const InverseMatrix<SparseMatrix<double>, Preconditioner> > A_inverse;

	mutable Vector<double> tmp1, tmp2;
};

template<class Preconditioner>
SchurComplement<Preconditioner>::SchurComplement(
		const BlockSparseMatrix<double> &system_matrix,
		const InverseMatrix<SparseMatrix<double>, Preconditioner> &A_inverse) :
		system_matrix(&system_matrix), A_inverse(&A_inverse), tmp1(
				system_matrix.block(0, 0).m()), tmp2(
				system_matrix.block(0, 0).m()) {
}

template<class Preconditioner>
void SchurComplement<Preconditioner>::vmult(Vector<double> &dst,
		const Vector<double> &src) const {
	system_matrix->block(0, 1).vmult(tmp1, src);
	A_inverse->vmult(tmp2, tmp1);
	system_matrix->block(1, 0).vmult(dst, tmp2);
}

// StokesProblem::StokesProblem

template<int dim>
StokesProblem<dim>::StokesProblem(const unsigned int degree) :
		degree(degree),
		mapping(),
		triangulation(Triangulation<dim>::maximum_smoothing),
		fe(FE_Q<dim>(degree + 1), dim, FE_Q<dim>(degree), 1),
		dof_handler(triangulation),
		quadrature_formula(degree + 2),
		ellipsoid(&triangulation)
				{}

// Set up dofs

template<int dim>
void StokesProblem<dim>::setup_dofs() {
	A_preconditioner.reset();
	system_matrix.clear();

	dof_handler.distribute_dofs(fe);
	DoFRenumbering::Cuthill_McKee(dof_handler);

	std::vector<unsigned int> block_component(dim + 1, 0);
	block_component[dim] = 1;
	DoFRenumbering::component_wise(dof_handler, block_component);

//========================================Apply Boundary Conditions=====================================
	{
		constraints.clear();
		std::vector<bool> component_maskP(dim + 1, false);
		component_maskP[dim] = true;
		DoFTools::make_hanging_node_constraints(dof_handler, constraints);
		VectorTools::interpolate_boundary_values(dof_handler, 1,
				BoundaryValuesP<dim>(), constraints, component_maskP);
	}
	{
		std::set<unsigned char> no_normal_flux_boundaries;
		no_normal_flux_boundaries.insert(99);
		VectorTools::compute_no_normal_flux_constraints(dof_handler, 0,
				no_normal_flux_boundaries, constraints);
	}

	constraints.close();

	std::vector<unsigned int> dofs_per_block(2);
	DoFTools::count_dofs_per_block(dof_handler, dofs_per_block,
			block_component);
	n_u = dofs_per_block[0];
	n_p = dofs_per_block[1];

	std::cout << "   Number of active cells: " << triangulation.n_active_cells()
			<< std::endl << "   Number of degrees of freedom: "
			<< dof_handler.n_dofs() << " (" << n_u << '+' << n_p << ')'
			<< std::endl;

	{
		BlockCompressedSimpleSparsityPattern csp(2, 2);

		csp.block(0, 0).reinit(n_u, n_u);
		csp.block(1, 0).reinit(n_p, n_u);
		csp.block(0, 1).reinit(n_u, n_p);
		csp.block(1, 1).reinit(n_p, n_p);

		csp.collect_sizes();

		DoFTools::make_sparsity_pattern(dof_handler, csp, constraints, false);
		sparsity_pattern.copy_from(csp);
	}

	system_matrix.reinit(sparsity_pattern);

	solution.reinit(2);
	solution.block(0).reinit(n_u);
	solution.block(1).reinit(n_p);
	solution.collect_sizes();

	system_rhs.reinit(2);
	system_rhs.block(0).reinit(n_u);
	system_rhs.block(1).reinit(n_p);
	system_rhs.collect_sizes();

}

// Viscosity and Shear modulus functions

template<int dim>
class Rheology {
public:
	double get_eta(double &r, double &z);
	double get_G(unsigned int mat_id);

private:
	std::vector<double> get_manual_eta_profile();

};

template<int dim>
std::vector<double> Rheology<dim>::get_manual_eta_profile()
{
	vector<double> etas;

	for(unsigned int i=0; i < system_parameters::sizeof_depths_eta; i++)
	{
		etas.push_back(system_parameters::depths_eta[i]);
		etas.push_back(system_parameters::eta_kinks[i]);
	}
	return etas;
}

template<int dim>
double Rheology<dim>::get_eta(double &r, double &z)
{
	// compute local depth
	double ecc = system_parameters::q_axes[0] / system_parameters::p_axes[0];
	double Rminusr = system_parameters::q_axes[0] - system_parameters::p_axes[0];
	double approx_a = std::sqrt(r * r + z * z * ecc * ecc);
	double approx_b = approx_a / ecc;
	double group1 = r * r + z * z - Rminusr * Rminusr;

		double a0 = approx_a;
		double error = 10000;
		// While loop finds the a axis of the "isodepth" ellipse for which the input point is on the surface.
		// An "isodepth" ellipse is defined as one whose axes a,b are related to the global axes A, B by: A-h = B-h

		if ((r > system_parameters::q_axes[0] - system_parameters::depths_eta.back()) ||
		    (z > system_parameters::p_axes[0] - system_parameters::depths_eta.back()))
		{

		    double eps = 10.0;
		    while (error >= eps)
		    {
			    double a02 = a0 * a0;
			    double a03 = a0 * a02;
			    double a04 = a0 * a03;
			    double fofa = a04 - (2 * Rminusr * a03) - (group1 * a02)
				            + (2 * r * r * Rminusr * a0) - (r * r * Rminusr * Rminusr);
			    double fprimeofa = 4 * a03 - (6 * Rminusr * a02) - (2 * group1 * a0)
					        + (2 * r * r * Rminusr);
			    double deltaa = -fofa / fprimeofa;
			    a0 += deltaa;
			    error = std::abs(deltaa);
//			cout << "error = " << error << endl;
		    }
		}
		else
		{
			a0 = 0.0;
		}

		double local_depth = system_parameters::q_axes[0] - a0;
		if (local_depth < 0)
			local_depth = 0;

		if (local_depth > system_parameters::depths_eta.back())
		{
			if (system_parameters::eta_kinks.back() < system_parameters::eta_floor)
			    return system_parameters::eta_floor;
			else if (system_parameters::eta_kinks.back() > system_parameters::eta_ceiling)
			    return system_parameters::eta_ceiling;
			else
				return system_parameters::eta_kinks.back();
		}

		std::vector<double> viscosity_function = get_manual_eta_profile();

		unsigned int n_visc_kinks = viscosity_function.size() / 2;

		//find the correct interval to do the interpolation in
		int n_minus_one = -1;
		for (unsigned int n = 1; n <= n_visc_kinks; n++) {
			unsigned int ndeep = 2 * n - 2;
			unsigned int nshallow = 2 * n;
			if (local_depth >= viscosity_function[ndeep] && local_depth <= viscosity_function[nshallow])
				n_minus_one = ndeep;
		}

		//find the viscosity interpolation
		if (n_minus_one == -1)
			return system_parameters::eta_ceiling;
		else {
			double visc_exponent =
					(viscosity_function[n_minus_one]
							- local_depth)
							/ (viscosity_function[n_minus_one]
									- viscosity_function[n_minus_one + 2]);
			double visc_base = viscosity_function[n_minus_one + 3]
					/ viscosity_function[n_minus_one + 1];
			// This is the true viscosity given the thermal profile
			double true_eta = viscosity_function[n_minus_one + 1] * std::pow(visc_base, visc_exponent);

			// Implement latitude-dependence viscosity
			if(system_parameters::lat_dependence)
			{
				double lat = 180 / PI * std::atan(z / r);
				if(lat > 80)
					lat = 80;
				double T_eq = 155;
				double T_surf = T_eq * std::sqrt( std::sqrt( std::cos( PI / 180 * lat ) ) );
				double taper_depth = 40000;
				double surface_taper = (taper_depth - local_depth) / taper_depth;
				if(surface_taper < 0)
					surface_taper = 0;
				double log_eta_contrast = surface_taper * system_parameters::eta_Ea * 52.5365 * (T_eq - T_surf) / T_eq / T_surf;
				true_eta *= std::pow(10, log_eta_contrast);
			}

			if(true_eta > system_parameters::eta_ceiling)
				return system_parameters::eta_ceiling;
			else
				if(true_eta < system_parameters::eta_floor)
					return system_parameters::eta_floor;
				else
					return true_eta;
		}
}


template<int dim>
double Rheology<dim>::get_G(unsigned int mat_id)
{
		return system_parameters::G[mat_id];
}


// Initialize the eta and G parts of the quadrature_point_history object

template<int dim>
void StokesProblem<dim>::initialize_eta_and_G() {
	FEValues<dim> fe_values(fe, quadrature_formula, update_quadrature_points);

	const unsigned int n_q_points = quadrature_formula.size();
	Rheology<dim> rheology;

	for (typename DoFHandler<dim>::active_cell_iterator cell =
			dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
		PointHistory<dim> *local_quadrature_points_history =
				reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
		Assert(
				local_quadrature_points_history >= &quadrature_point_history.front(),
				ExcInternalError());
		Assert(
				local_quadrature_points_history < &quadrature_point_history.back(),
				ExcInternalError());
		fe_values.reinit(cell);

		for (unsigned int q = 0; q < n_q_points; ++q) {

			double r_value = fe_values.quadrature_point(q)[0];
			double z_value = fe_values.quadrature_point(q)[1];

			//defines local viscosity
			double local_viscosity = 0;
			unsigned int m_id = cell->material_id();

			local_viscosity = rheology.get_eta(r_value, z_value);

			local_quadrature_points_history[q].first_eta = local_viscosity;
			local_quadrature_points_history[q].new_eta = local_viscosity;

			//defines local shear modulus
			double local_G = 0;

			unsigned int mat_id = cell->material_id();

			local_G = rheology.get_G(mat_id);
			local_quadrature_points_history[q].G = local_G;

			//initializes the phi-phi stress
			local_quadrature_points_history[q].old_phiphi_stress = 0;
		}
	}
}

//====================== ASSEMBLE THE SYSTEM ======================

template<int dim>
void StokesProblem<dim>::assemble_system() {
	system_matrix = 0;
	system_rhs = 0;

	FEValues<dim> fe_values(fe, quadrature_formula,
			update_values | update_quadrature_points | update_JxW_values
					| update_gradients);

	const unsigned int dofs_per_cell = fe.dofs_per_cell;

	const unsigned int n_q_points = quadrature_formula.size();

	FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
	Vector<double> local_rhs(dofs_per_cell);

	std::vector<unsigned int> local_dof_indices(dofs_per_cell);

	// runs the gravity script function
	const RightHandSide<dim> right_hand_side;

	A_Grav_namespace::AnalyticGravity<dim> * aGrav =
			new A_Grav_namespace::AnalyticGravity<dim>;
	std::vector<double> grav_parameters;
	grav_parameters.push_back(system_parameters::q_axes[system_parameters::present_timestep * 2 + 0]);
	grav_parameters.push_back(system_parameters::p_axes[system_parameters::present_timestep * 2 + 0]);
	grav_parameters.push_back(system_parameters::q_axes[system_parameters::present_timestep * 2 + 1]);
	grav_parameters.push_back(system_parameters::p_axes[system_parameters::present_timestep * 2 + 1]);
	grav_parameters.push_back(system_parameters::rho[0]);
	grav_parameters.push_back(system_parameters::rho[1]);

	std::cout << "Body parameters are: " ;
	for(int i=0; i<6; i++)
		std::cout << grav_parameters[i] << " ";
	std::cout << endl;

	aGrav->setup_vars(grav_parameters);

	std::vector<Vector<double> > rhs_values(n_q_points,
			Vector<double>(dim + 1));

	const FEValuesExtractors::Vector velocities(0);
	const FEValuesExtractors::Scalar pressure(dim);

	std::vector<SymmetricTensor<2, dim> > phi_grads_u(dofs_per_cell);
	std::vector<double> div_phi_u(dofs_per_cell);
	std::vector<Tensor<1, dim> > phi_u(dofs_per_cell);
	std::vector<double> phi_p(dofs_per_cell);

	typename DoFHandler<dim>::active_cell_iterator cell =
			dof_handler.begin_active(), first_cell = dof_handler.begin_active(),
			endc = dof_handler.end();

	for (; cell != endc; ++cell) {
		PointHistory<dim> *local_quadrature_points_history =
				reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
		Assert(
				local_quadrature_points_history >= &quadrature_point_history.front(),
				ExcInternalError());
		Assert(
				local_quadrature_points_history < &quadrature_point_history.back(),
				ExcInternalError());
		
		double cell_area = cell->measure();
		if(cell_area<0)
			append_physical_times(-1);
		AssertThrow(cell_area > 0
			,
		  ExcInternalError());


		unsigned int m_id = cell->material_id();

		//initializes the rhs vector to the correct g values
		fe_values.reinit(cell);
		right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
				rhs_values, aGrav);

		std::vector<Vector<double> > new_viscosities(quadrature_formula.size(), Vector<double>(dim + 1));

		// Finds vertices where the radius is zero DIM
		bool is_singular = false;
		unsigned int singular_vertex_id = 0;
		for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
			if (cell->face(f)->center()[0] == 0) {
				is_singular = true;
				singular_vertex_id = f;
			}
		}

		if (is_singular == false || system_parameters::cylindrical == false) {
			local_matrix = 0;
			local_rhs = 0;

			// ===== outputs the local gravity
			std::vector<Point<dim> > quad_points_list(n_q_points);
			quad_points_list = fe_values.get_quadrature_points();

			if (plastic_iteration
					== (system_parameters::max_plastic_iterations - 1)) {
				if (cell != first_cell) {
					std::ofstream fout("gravity_field.txt", std::ios::app);
					fout << quad_points_list[0] << " " << rhs_values[0];
					fout.close();
				} else {
					std::ofstream fout("gravity_field.txt");
					fout << quad_points_list[0] << " " << rhs_values[0];
					fout.close();
				}
			}

			for (unsigned int q = 0; q < n_q_points; ++q) {
				 SymmetricTensor<2, dim> &old_stress =
						local_quadrature_points_history[q].old_stress;
				double &local_old_phiphi_stress =
						local_quadrature_points_history[q].old_phiphi_stress;
				double r_value = fe_values.quadrature_point(q)[0];
				double z_value = fe_values.quadrature_point(q)[1];
				
				// if(system_parameters::present_timestep == system_parameters::initial_elastic_iterations)
				// {
				// 	old_stress *= 0;
				// 	local_old_phiphi_stress = 0;
				// }

				// get local density based on mat id
				double local_density = system_parameters::rho[m_id];

				//defines local viscosities
				double local_viscosity = 0;
				if (plastic_iteration == 0)
					local_viscosity = local_quadrature_points_history[q].first_eta;
				else
					local_viscosity = local_quadrature_points_history[q].new_eta;

				// Define the local viscoelastic constants
				double local_eta_ve = 2
						/ ((1 / local_viscosity)
								+ (1 / local_quadrature_points_history[q].G
										/ system_parameters::current_time_interval));
				double local_chi_ve = 1
						/ (1
								+ (local_quadrature_points_history[q].G
										* system_parameters::current_time_interval
										/ local_viscosity));

				for (unsigned int k = 0; k < dofs_per_cell; ++k) {
					phi_grads_u[k] = fe_values[velocities].symmetric_gradient(k,
							q);
					div_phi_u[k] = (fe_values[velocities].divergence(k, q));
					phi_u[k] = (fe_values[velocities].value(k, q));
					if (system_parameters::cylindrical == true) {
						div_phi_u[k] *= (r_value);
						div_phi_u[k] += (phi_u[k][0]);
					}
					phi_p[k] = fe_values[pressure].value(k, q);
				}

				for (unsigned int i = 0; i < dofs_per_cell; ++i) {
					for (unsigned int j = 0; j <= i; ++j) {
						if (system_parameters::cylindrical == true) {
							local_matrix(i, j) += (phi_grads_u[i]
									* phi_grads_u[j] * 2 * local_eta_ve
									* r_value
									+ 2 * phi_u[i][0] * phi_u[j][0]
											* local_eta_ve / r_value
									- div_phi_u[i] * phi_p[j]
											* system_parameters::pressure_scale
									- phi_p[i] * div_phi_u[j]
											* system_parameters::pressure_scale
									+ phi_p[i] * phi_p[j] * r_value
											* system_parameters::pressure_scale)
									* fe_values.JxW(q);
						} else {
							local_matrix(i, j) += (phi_grads_u[i]
									* phi_grads_u[j] * 2 * local_eta_ve
									- div_phi_u[i] * phi_p[j]
											* system_parameters::pressure_scale
									- phi_p[i] * div_phi_u[j]
											* system_parameters::pressure_scale
									+ phi_p[i] * phi_p[j]) * fe_values.JxW(q);
						}
					}
					if (system_parameters::cylindrical == true) {
						const unsigned int component_i =
								fe.system_to_component_index(i).first;
						local_rhs(i) += (fe_values.shape_value(i, q)
								* rhs_values[q](component_i) * r_value
								* local_density
								- local_chi_ve * phi_grads_u[i] * old_stress
										* r_value
								- local_chi_ve * phi_u[i][0]
										* local_old_phiphi_stress)
								* fe_values.JxW(q);
					} else {
						const unsigned int component_i =
								fe.system_to_component_index(i).first;
						local_rhs(i) += fe_values.shape_value(i, q)
								* rhs_values[q](component_i) * fe_values.JxW(q)
								* local_density;
					}
				}
			}
		} // end of non-singular
		else {
			local_matrix = 0;
			local_rhs = 0;

			// ===== outputs the local gravity
			std::vector<Point<dim> > quad_points_list(n_q_points);
			quad_points_list = fe_values.get_quadrature_points();

			for (unsigned int q = 0; q < n_q_points; ++q) {
				const SymmetricTensor<2, dim> &old_stress =
						local_quadrature_points_history[q].old_stress;
				double &local_old_phiphi_stress =
						local_quadrature_points_history[q].old_phiphi_stress;
				double r_value = fe_values.quadrature_point(q)[0];
				double z_value = fe_values.quadrature_point(q)[1];

                double local_density = system_parameters::rho[m_id];

				//defines local viscosities
				double local_viscosity = 0;
				if (plastic_iteration == 0)
				{
					local_viscosity = local_quadrature_points_history[q].first_eta;
				}
				else
					local_viscosity = local_quadrature_points_history[q].new_eta;

				// Define the local viscoelastic constants
				double local_eta_ve = 2
						/ ((1 / local_viscosity)
								+ (1 / local_quadrature_points_history[q].G
										/ system_parameters::current_time_interval));
				double local_chi_ve = 1
						/ (1
								+ (local_quadrature_points_history[q].G
										* system_parameters::current_time_interval
										/ local_viscosity));

				for (unsigned int k = 0; k < dofs_per_cell; ++k) {
					phi_grads_u[k] = fe_values[velocities].symmetric_gradient(k,
							q);
					div_phi_u[k] = (fe_values[velocities].divergence(k, q));
					phi_u[k] = (fe_values[velocities].value(k, q));
					if (system_parameters::cylindrical == true) {
						div_phi_u[k] *= (r_value);
						div_phi_u[k] += (phi_u[k][0]);
					}
					phi_p[k] = fe_values[pressure].value(k, q);
				}

				for (unsigned int i = 0; i < dofs_per_cell; ++i) {
					for (unsigned int j = 0; j <= i; ++j) {
						if (system_parameters::cylindrical == true) {
							local_matrix(i, j) += (phi_grads_u[i]
									* phi_grads_u[j] * 2 * local_eta_ve
									* r_value
									+ 2 * phi_u[i][0] * phi_u[j][0]
											* local_eta_ve / r_value
									- div_phi_u[i] * phi_p[j]
											* system_parameters::pressure_scale
									- phi_p[i] * div_phi_u[j]
											* system_parameters::pressure_scale
									+ phi_p[i] * phi_p[j] * r_value
											* system_parameters::pressure_scale)
									* fe_values.JxW(q);
						} else {
							local_matrix(i, j) += (phi_grads_u[i]
									* phi_grads_u[j] * 2 * local_eta_ve
									- div_phi_u[i] * phi_p[j]
											* system_parameters::pressure_scale
									- phi_p[i] * div_phi_u[j]
											* system_parameters::pressure_scale
									+ phi_p[i] * phi_p[j]) * fe_values.JxW(q);
						}
					}
					if (system_parameters::cylindrical == true) {
						const unsigned int component_i =
								fe.system_to_component_index(i).first;
						local_rhs(i) += (fe_values.shape_value(i, q)
								* rhs_values[q](component_i) * r_value
								* local_density
								- local_chi_ve * phi_grads_u[i] * old_stress
										* r_value
								- local_chi_ve * phi_u[i][0]
										* local_old_phiphi_stress)
								* fe_values.JxW(q);
					} else {
						const unsigned int component_i =
								fe.system_to_component_index(i).first;
						local_rhs(i) += fe_values.shape_value(i, q)
								* rhs_values[q](component_i) * fe_values.JxW(q)
								* local_density;
					}
				}
			}
		} // end of singular

		for (unsigned int i = 0; i < dofs_per_cell; ++i)
			for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
				local_matrix(i, j) = local_matrix(j, i);

		cell->get_dof_indices(local_dof_indices);
		constraints.distribute_local_to_global(local_matrix, local_rhs,
				local_dof_indices, system_matrix, system_rhs);
	}

	std::cout << "   Computing preconditioner..." << std::endl << std::flush;

	A_preconditioner = std_cxx1x::shared_ptr<
			typename InnerPreconditioner<dim>::type>(
			new typename InnerPreconditioner<dim>::type());
	A_preconditioner->initialize(system_matrix.block(0, 0),
			typename InnerPreconditioner<dim>::type::AdditionalData());

	delete aGrav;
}

//====================== SOLVER ======================

template<int dim>
void StokesProblem<dim>::solve() {
	const InverseMatrix<SparseMatrix<double>,
			typename InnerPreconditioner<dim>::type> A_inverse(
			system_matrix.block(0, 0), *A_preconditioner);
	Vector<double> tmp(solution.block(0).size());

	{
		Vector<double> schur_rhs(solution.block(1).size());
		A_inverse.vmult(tmp, system_rhs.block(0));
		system_matrix.block(1, 0).vmult(schur_rhs, tmp);
		schur_rhs -= system_rhs.block(1);

		SchurComplement<typename InnerPreconditioner<dim>::type> schur_complement(
				system_matrix, A_inverse);

		int n_iterations = system_parameters::iteration_coefficient
				* solution.block(1).size();
		double tolerance_goal = system_parameters::tolerance_coefficient
				* schur_rhs.l2_norm();

		SolverControl solver_control(n_iterations, tolerance_goal);
		SolverCG<> cg(solver_control);

		std::cout << "\nMax iterations and tolerance are:  " << n_iterations
				<< " and " << tolerance_goal << std::endl;

		SparseILU<double> preconditioner;
		preconditioner.initialize(system_matrix.block(1, 1),
				SparseILU<double>::AdditionalData());

		InverseMatrix<SparseMatrix<double>, SparseILU<double> > m_inverse(
				system_matrix.block(1, 1), preconditioner);

		cg.solve(schur_complement, solution.block(1), schur_rhs, m_inverse);

		constraints.distribute(solution);


		std::cout << "  " << solver_control.last_step()
				<< " outer CG Schur complement iterations for pressure"
				<< std::endl;
	}

	{
		system_matrix.block(0, 1).vmult(tmp, solution.block(1));
		tmp *= -1;
		tmp += system_rhs.block(0);

		A_inverse.vmult(solution.block(0), tmp);
		constraints.distribute(solution);
		solution.block(1) *= (system_parameters::pressure_scale);
	}
}

//====================== OUTPUT RESULTS ======================
template<int dim>
void StokesProblem<dim>::output_results() const {
	std::vector < std::string > solution_names(dim, "velocity");
	solution_names.push_back("pressure");

	std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(
			dim, DataComponentInterpretation::component_is_part_of_vector);
	data_component_interpretation.push_back(
			DataComponentInterpretation::component_is_scalar);

	DataOut<dim> data_out;
	data_out.attach_dof_handler(dof_handler);
	data_out.add_data_vector(solution, solution_names,
			DataOut<dim>::type_dof_data, data_component_interpretation);
	data_out.build_patches();

	std::ostringstream filename;
	if (system_parameters::present_timestep < system_parameters::initial_elastic_iterations)
	{
		filename << system_parameters::output_folder << "/time"
						<< Utilities::int_to_string(system_parameters::present_timestep, 2)
						<< "_elastic_displacements" << ".txt";
	}
	else
	{
		filename << system_parameters::output_folder << "/time"
				<< Utilities::int_to_string(system_parameters::present_timestep, 2)
				<< "_flow" << Utilities::int_to_string(plastic_iteration, 2) << ".txt";
	}

	std::ofstream output(filename.str().c_str());
	data_out.write_gnuplot(output);
}

//====================== FIND AND WRITE TO FILE THE STRESS TENSOR; IMPLEMENT PLASTICITY ======================

template<int dim>
void StokesProblem<dim>::solution_stesses() {
	//note most of this section only works with dim=2

	//name the output text files
	std::ostringstream stress_output;
	stress_output << system_parameters::output_folder << "/time"
			<< Utilities::int_to_string(system_parameters::present_timestep, 2)
			<< "_principalstresses" << Utilities::int_to_string(plastic_iteration, 2)
			<< ".txt";
	std::ofstream fout_snew(stress_output.str().c_str());
	fout_snew.close();

	std::ostringstream stresstensor_output;
	stresstensor_output << system_parameters::output_folder << "/time"
			<< Utilities::int_to_string(system_parameters::present_timestep, 2)
			<< "_stresstensor" << Utilities::int_to_string(plastic_iteration, 2)
			<< ".txt";
	std::ofstream fout_sfull(stresstensor_output.str().c_str());
	fout_sfull.close();

	std::ostringstream failed_cells_output;
	failed_cells_output << system_parameters::output_folder << "/time"
			<< Utilities::int_to_string(system_parameters::present_timestep, 2)
			<< "_failurelocations" << Utilities::int_to_string(plastic_iteration, 2)
			<< ".txt";
	std::ofstream fout_failed_cells(failed_cells_output.str().c_str());
	fout_failed_cells.close();

	std::ostringstream plastic_eta_output;
	plastic_eta_output << system_parameters::output_folder << "/time"
			<< Utilities::int_to_string(system_parameters::present_timestep, 2)
			<< "_viscositiesreg" << Utilities::int_to_string(plastic_iteration, 2)
			<< ".txt";
	std::ofstream fout_vrnew(plastic_eta_output.str().c_str());
	fout_vrnew.close();

	std::ostringstream initial_eta_output;
	if (plastic_iteration == 0)
	{
		initial_eta_output << system_parameters::output_folder << "/time"
			<< Utilities::int_to_string(system_parameters::present_timestep, 2)
			<< "_baseviscosities.txt";
		std::ofstream fout_baseeta(initial_eta_output.str().c_str());
		fout_baseeta.close();
	}

	std::cout << "Running stress calculations for plasticity iteration "
			<< plastic_iteration << "...\n";

	//This makes the set of points at which the stress tensor is calculated
	std::vector<Point<dim> > points_list(0);
	std::vector<unsigned int> material_list(0);
	typename DoFHandler<dim>::active_cell_iterator cell =
			dof_handler.begin_active(), endc = dof_handler.end();
	//This loop gets the gradients of the velocity field and saves it in the tensor_gradient_? objects DIM
	for (; cell != endc; ++cell) {
		points_list.push_back(cell->center());
		material_list.push_back(cell->material_id());
	}
	// Make the FEValues object to evaluate values and derivatives at quadrature points
	FEValues<dim> fe_values(fe, quadrature_formula,
			update_values | update_gradients | update_quadrature_points | update_JxW_values);

	// Make the object that will hold the velocities and velocity gradients at the quadrature points
	std::vector < std::vector<Tensor<1, dim> >> velocity_grads(quadrature_formula.size(),
			std::vector < Tensor<1, dim> > (dim + 1));
	std::vector<Vector<double> > velocities(quadrature_formula.size(),
			Vector<double>(dim + 1));
	// Make the object to find rheology
	Rheology<dim> rheology;

	// Write the solution flow velocity and derivative for each cell
	std::vector<Vector<double> > vector_values(0);
	std::vector < std::vector<Tensor<1, dim> > > gradient_values(0);
	std::vector<bool> failing_cells;
	// Write the stresses from the previous step into vectors
	std::vector<SymmetricTensor<2, dim>> old_stress;
	std::vector<double> old_phiphi_stress;
	std::vector<double> cell_Gs;
	for (typename DoFHandler<dim>::active_cell_iterator cell =
		dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
	{
		// Makes pointer to data in quadrature_point_history
		PointHistory<dim> *local_quadrature_points_history =
				reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());

		fe_values.reinit(cell);
		fe_values.get_function_gradients(solution, velocity_grads);
		fe_values.get_function_values(solution, velocities);
		Vector<double> current_cell_velocity(dim+1);
		std::vector<Tensor<1, dim>> current_cell_grads(dim+1);
		SymmetricTensor<2, dim> current_cell_old_stress;
		current_cell_old_stress = 0;
		double current_cell_old_phiphi_stress = 0;
		double cell_area = 0;

		// Averages across each cell to find mean velocities, gradients, and old stresses
		for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
		{
			cell_area += fe_values.JxW(q);
			velocities[q] *= fe_values.JxW(q);
			current_cell_velocity += velocities[q];
			for (unsigned int i = 0; i < (dim+1); i++)
			{
				velocity_grads[q][i] *= fe_values.JxW(q);
				current_cell_grads[i] += velocity_grads[q][i];
			}
			current_cell_old_stress += local_quadrature_points_history[q].old_stress * fe_values.JxW(q);
			current_cell_old_phiphi_stress += local_quadrature_points_history[q].old_phiphi_stress * fe_values.JxW(q);
		}
		current_cell_velocity /= cell_area;
		for (unsigned int i = 0; i < (dim+1); i++)
			current_cell_grads[i] /= cell_area;
		current_cell_old_stress /= cell_area;
		current_cell_old_phiphi_stress /= cell_area;

		vector_values.push_back(current_cell_velocity);
		gradient_values.push_back(current_cell_grads);
		old_stress.push_back(current_cell_old_stress);
		old_phiphi_stress.push_back(current_cell_old_phiphi_stress);

		// Get cell shear modulus: assumes it's constant for the cell
		unsigned int mat_id = cell->material_id();
		double local_G = rheology.get_G(mat_id);
		cell_Gs.push_back(local_G);
	}

	//tracks where failure occurred
	std::vector<double> reduction_factor;
	unsigned int total_fails = 0;
	if (plastic_iteration == 0)
		cell_viscosities.resize(0);
	//loop across all the cells to find and adjust eta of failing cells
	for (unsigned int i = 0; i < triangulation.n_active_cells(); i++)
	{
		double current_cell_viscosity = 0;

		// Fill viscosities vector, analytically if plastic_iteration == 0 and from previous viscosities for later iteration
		if (plastic_iteration == 0)
		{
			double local_viscosity;
		    local_viscosity = rheology.get_eta(points_list[i][0], points_list[i][1]);
			current_cell_viscosity = local_viscosity;
			cell_viscosities.push_back(current_cell_viscosity);
		}
		else
		{
			current_cell_viscosity = cell_viscosities[i];
		}


		double cell_eta_ve = 2
				/ ((1 / current_cell_viscosity)
						+ (1 / cell_Gs[i]
								/ system_parameters::current_time_interval));
		double cell_chi_ve = 1
				/ (1
						+ (cell_Gs[i]
								* system_parameters::current_time_interval
								/ current_cell_viscosity));

		//find local pressure
		double cell_p = vector_values[i].operator()(2);
		//find stresses tensor
		//makes non-diagonalized local matrix A
		double sigma13 = 0.5
				* (gradient_values[i][0][1] + gradient_values[i][1][0]);
		mat A;
		A << gradient_values[i][0][0] << 0 << sigma13 << endr
		  << 0 << vector_values[i].operator()(0) / points_list[i].operator()(0)<< 0 << endr
		  << sigma13 << 0 << gradient_values[i][1][1] << endr;
		mat olddevstress;
		olddevstress << old_stress[i][0][0] << 0 << old_stress[i][0][1] << endr
					 << 0 << old_phiphi_stress[i] << 0 << endr
					 << old_stress[i][0][1] << 0 << old_stress[i][1][1] << endr;
		vec P;
		P << cell_p << cell_p << cell_p;
		mat Pmat = diagmat(P);
		mat B;
		B = (cell_eta_ve * A + cell_chi_ve * olddevstress) - Pmat;

		//finds principal stresses
		vec eigval;
		mat eigvec;
		eig_sym(eigval, eigvec, B);
		double sigma1 = -min(eigval);
		double sigma3 = -max(eigval);

		// Writes text files for principal stresses, full stress tensor, base viscosities
		std::ofstream fout_snew(stress_output.str().c_str(), std::ios::app);
		fout_snew << " " << sigma1 << " " << sigma3 << "\n";
		fout_snew.close();

		std::ofstream fout_sfull(stresstensor_output.str().c_str(), std::ios::app);
		fout_sfull << A(0,0) << " " << A(1,1) << " " << A(2,2) << " " << A(0,2) << "\n";
		fout_sfull.close();

		if (plastic_iteration == 0)
		{
			std::ofstream fout_baseeta(initial_eta_output.str().c_str(), std::ios::app);
			fout_baseeta << points_list[i]<< " " << current_cell_viscosity << "\n";
			fout_baseeta.close();
		}

		// Finds adjusted effective viscosity
		double cell_effective_viscosity = 0;
		if (system_parameters::plasticity_on)
		{
			if(system_parameters::failure_criterion == 0) //Apply Byerlee's rule
			{
				if (sigma1 >= 5 * sigma3) // this guarantees that viscosities only go down, never up
					{
					failing_cells.push_back(true);
					double temp_reductionfactor = 1;
					if (sigma3 < 0)
						temp_reductionfactor = 100;
					else
						temp_reductionfactor = 1.9 * sigma1 / 5 / sigma3;

					reduction_factor.push_back(temp_reductionfactor);
					total_fails++;

					// Text file of all failure locations
					std::ofstream fout_failed_cells(failed_cells_output.str().c_str(), std::ios::app);
					fout_failed_cells << points_list[i] << "\n";
					fout_failed_cells.close();
					}
				else
				{
					reduction_factor.push_back(1);
					failing_cells.push_back(false);
				}
			}
			else
			{
				if(system_parameters::failure_criterion == 1) //Apply Schultz criterion for frozen sand, RMR=45
				{
					double temp_reductionfactor = 1;
					if(sigma3 < -114037)
					{
						//std::cout << " ext ";
						failing_cells.push_back(true);
						temp_reductionfactor = 10;
						reduction_factor.push_back(temp_reductionfactor);
						total_fails++;

						// Text file of all failure locations
						std::ofstream fout_failed_cells(failed_cells_output.str().c_str(), std::ios::app);
						fout_failed_cells << points_list[i] << "\n";
						fout_failed_cells.close();
					}
					else
					{
						double sigma_c = 160e6; //Unconfined compressive strength
						double yield_sigma1 = sigma3 + std::sqrt( (3.086 * sigma_c * sigma3) + (0.002 * sigma3 * sigma3) );
						if (sigma1 >= yield_sigma1)
						{
							//std::cout << " comp ";
							failing_cells.push_back(true);
							temp_reductionfactor = 1.0 * sigma1 / 5 / sigma3;

							reduction_factor.push_back(temp_reductionfactor);
							total_fails++;

							// Text file of all failure locations
							std::ofstream fout_failed_cells(failed_cells_output.str().c_str(), std::ios::app);
							fout_failed_cells << points_list[i] << "\n";
							fout_failed_cells.close();
						}
						else
						{
							reduction_factor.push_back(1);
							failing_cells.push_back(false);
						}
					}
				}
				else
				{
					std::cout << "Specified failure criterion not found\n";
				}
			}
		}
		else
			reduction_factor.push_back(1);
	}

	// If there are enough failed cells, update eta at all quadrature points and perform smoothing
	std::cout << "   Number of failing cells: " << total_fails << "\n";
	double last_max_plasticity_double = last_max_plasticity;
	double total_fails_double = total_fails;
	double decrease_in_plasticity = ((last_max_plasticity_double - total_fails_double) / last_max_plasticity_double);
	if(plastic_iteration == 0)
		decrease_in_plasticity = 1;
	last_max_plasticity = total_fails;
	if (total_fails <= 100 || decrease_in_plasticity <= 0.2)
	{
		system_parameters::continue_plastic_iterations = false;
		for(unsigned int j=0; j < triangulation.n_active_cells(); j++)
		{
			// Writes to file the undisturbed cell viscosities
			std::ofstream fout_vrnew(plastic_eta_output.str().c_str(), std::ios::app);
			fout_vrnew << " " << cell_viscosities[j] << "\n";
			fout_vrnew.close();
		}
	}
	else{
		quad_viscosities.resize(triangulation.n_active_cells());
		// Decrease the eta at quadrature points in failing cells
		unsigned int cell_no = 0;
		for (typename DoFHandler<dim>::active_cell_iterator cell =
			dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
		{
			fe_values.reinit(cell);
			// Make local_quadrature_points_history pointer to the cell data
			PointHistory<dim> *local_quadrature_points_history =
					reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
			Assert(
					local_quadrature_points_history >= &quadrature_point_history.front(),
					ExcInternalError());
			Assert(
					local_quadrature_points_history < &quadrature_point_history.back(),
					ExcInternalError());

			quad_viscosities[cell_no].resize(quadrature_formula.size());

			for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
			{
				if (plastic_iteration == 0)
					local_quadrature_points_history[q].new_eta = local_quadrature_points_history[q].first_eta;
				local_quadrature_points_history[q].new_eta /= reduction_factor[cell_no];
				// Prevents viscosities from dropping below the floor necessary for numerical stability
				if (local_quadrature_points_history[q].new_eta < system_parameters::eta_floor)
					local_quadrature_points_history[q].new_eta = system_parameters::eta_floor;

				quad_viscosities[cell_no][q].reinit(dim+1);
				for(unsigned int ii=0; ii<dim; ii++)
					quad_viscosities[cell_no][q](ii) = fe_values.quadrature_point(q)[ii];
				quad_viscosities[cell_no][q](dim) = local_quadrature_points_history[q].new_eta;
			}
			cell_no++;
		}
		smooth_eta_field(failing_cells);

		// Writes to file the smoothed eta field (which is defined at each quadrature point) for each cell
		cell_no = 0;
//		cell_viscosities.resize(triangulation.n_active_cells(), 0);
		for (typename DoFHandler<dim>::active_cell_iterator cell =
			dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
		{
			if(failing_cells[cell_no])
			{
				fe_values.reinit(cell);
				// Averages across each cell to find mean eta
				double cell_area = 0;
				for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
				{
					cell_area += fe_values.JxW(q);
					cell_viscosities[cell_no] += quad_viscosities[cell_no][q][dim] * fe_values.JxW(q);
				}
				cell_viscosities[cell_no] /= cell_area;

				// Writes to file
				std::ofstream fout_vrnew(plastic_eta_output.str().c_str(), std::ios::app);
				fout_vrnew << " " << cell_viscosities[cell_no] << "\n";
				fout_vrnew.close();
			}
			else
			{
				std::ofstream fout_vrnew(plastic_eta_output.str().c_str(), std::ios::app);
				fout_vrnew << " " << cell_viscosities[cell_no] << "\n";
				fout_vrnew.close();
			}
			cell_no++;
		}
	}
}

//====================== SMOOTHES THE VISCOSITY FIELD AT ALL QUADRATURE POINTS ======================

template<int dim>
void StokesProblem<dim>::smooth_eta_field(std::vector<bool> failing_cells)
{
	std::cout << "   Smoothing viscosity field...\n";
	unsigned int cell_no = 0;
	for (typename DoFHandler<dim>::active_cell_iterator cell =
			dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
	{
		if(failing_cells[cell_no])
		{
			FEValues<dim> fe_values(fe, quadrature_formula, update_quadrature_points);
			PointHistory<dim> *local_quadrature_points_history =
					reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());

			// Currently this algorithm does not permit refinement.  To permit refinement, daughter cells of neighbors must be identified
			// Find pointers and indices of all cells within certain radius
			bool find_more_cells = true;
			std::vector<bool> cell_touched(triangulation.n_active_cells(), false);
			std::vector< TriaIterator< CellAccessor<dim> > > neighbor_cells;
			std::vector<int> neighbor_indices;
			int start_cell = 0; // Which cell in the neighbor_cells vector to start from
			int new_cells_found = 0;
			neighbor_cells.push_back(cell);
			neighbor_indices.push_back(cell_no);
			cell_touched[cell_no] = true;
			while(find_more_cells)
			{
				new_cells_found = 0;
				for(int i = start_cell; i<neighbor_cells.size(); i++)
				{
					for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
					{
						if (!neighbor_cells[i]->face(f)->at_boundary())
						{
							int test_cell_no = neighbor_cells[i]->neighbor_index(f);
							if(!cell_touched[test_cell_no])
								if(cell->center().distance(neighbor_cells[i]->neighbor(f)->center()) < 2 * system_parameters::smoothing_radius)
								{
									// What to do if another nearby cell is found that hasn't been found before
									neighbor_cells.push_back(neighbor_cells[i]->neighbor(f));
									neighbor_indices.push_back(test_cell_no);
									cell_touched[test_cell_no] = true;
									start_cell++;
									new_cells_found++;
							}
						}
					}
				}
				if (new_cells_found == 0){
					find_more_cells = false;
				}
				else
					start_cell = neighbor_cells.size() - new_cells_found;
			}

			fe_values.reinit(cell);
			// Collect the viscosities at nearby quadrature points
			for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
			{
				std::vector<double> nearby_etas_q;
				for (unsigned int i = 0; i<neighbor_indices.size(); i++)
					for (unsigned int j=0; j<quadrature_formula.size(); j++)
					{
						Point<dim> test_q;
						for(unsigned int d=0; d<dim; d++)
							test_q(d) = quad_viscosities[neighbor_indices[i]][j][d];
						double qq_distance = fe_values.quadrature_point(q).distance(test_q);
						if(qq_distance < system_parameters::smoothing_radius)
							nearby_etas_q.push_back(quad_viscosities[neighbor_indices[i]][j][dim]);
					}
				// Write smoothed viscosities to quadrature_points_history; simple boxcar function is the smoothing kernel
				double mean_eta = 0;
				for(unsigned int l = 0; l<nearby_etas_q.size(); l++)
				{
					mean_eta += nearby_etas_q[l];
				}
				mean_eta /= nearby_etas_q.size();
				local_quadrature_points_history[q].new_eta = mean_eta;
//				std::cout << local_quadrature_points_history[q].new_eta << " ";
			}
		}
		cell_no++;
	}
}

//====================== SAVE STRESS TENSOR AT QUADRATURE POINTS ======================

template<int dim>
void StokesProblem<dim>::update_quadrature_point_history() {
	std::cout << "   Updating stress field...";

	FEValues<dim> fe_values(fe, quadrature_formula,
			update_values | update_gradients | update_quadrature_points);

	// Make the object that will hold the velocity gradients
	std::vector < std::vector<Tensor<1, dim> >> velocity_grads(quadrature_formula.size(),
			std::vector < Tensor<1, dim> > (dim + 1));
	std::vector<Vector<double> > velocities(quadrature_formula.size(),
			Vector<double>(dim + 1));

	for (typename DoFHandler<dim>::active_cell_iterator cell =
			dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
		PointHistory<dim> *local_quadrature_points_history =
				reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
		Assert(
				local_quadrature_points_history >= &quadrature_point_history.front(),
				ExcInternalError());
		Assert(
				local_quadrature_points_history < &quadrature_point_history.back(),
				ExcInternalError());

		fe_values.reinit(cell);
		fe_values.get_function_gradients(solution, velocity_grads);
		fe_values.get_function_values(solution, velocities);

		for (unsigned int q = 0; q < quadrature_formula.size(); ++q) {
			// Define the local viscoelastic constants
			double local_eta_ve = 2
					/ ((1 / local_quadrature_points_history[q].new_eta)
							+ (1 / local_quadrature_points_history[q].G
									/ system_parameters::current_time_interval));
			double local_chi_ve =
					1
							/ (1
									+ (local_quadrature_points_history[q].G
											* system_parameters::current_time_interval
											/ local_quadrature_points_history[q].new_eta));

			// Compute new stress at each quadrature point
			SymmetricTensor<2, dim> new_stress;
			for (unsigned int i = 0; i < dim; ++i)
				new_stress[i][i] =
						local_eta_ve * velocity_grads[q][i][i]
								+ local_chi_ve
										* local_quadrature_points_history[q].old_stress[i][i];

			for (unsigned int i = 0; i < dim; ++i)
				for (unsigned int j = i + 1; j < dim; ++j)
					new_stress[i][j] =
							local_eta_ve
									* (velocity_grads[q][i][j]
											+ velocity_grads[q][j][i]) / 2
									+ local_chi_ve
											* local_quadrature_points_history[q].old_stress[i][j];

			// Rotate new stress
			AuxFunctions<dim> rotation_object;
			const Tensor<2, dim> rotation = rotation_object.get_rotation_matrix(
					velocity_grads[q]);
			const SymmetricTensor<2, dim> rotated_new_stress = symmetrize(
					transpose(rotation)
							* static_cast<Tensor<2, dim> >(new_stress)
							* rotation);
			local_quadrature_points_history[q].old_stress = rotated_new_stress;

			// For axisymmetric case, make the phi-phi element of stress tensor
			local_quadrature_points_history[q].old_phiphi_stress =
					(2 * local_eta_ve * velocities[q](0)
							/ fe_values.quadrature_point(q)[0]
							+ local_chi_ve
									* local_quadrature_points_history[q].old_phiphi_stress);
		}
	}
}

//====================== REDEFINE THE TIME INTERVAL FOR THE VISCOUS STEPS ======================
template<int dim>
void StokesProblem<dim>::update_time_interval()
{
	double move_goal_per_step = system_parameters::initial_disp_target;
	if(system_parameters::present_timestep > system_parameters::initial_elastic_iterations)
	{
		move_goal_per_step = system_parameters::initial_disp_target -
			((system_parameters::initial_disp_target - system_parameters::final_disp_target) /
			system_parameters::total_viscous_steps *
			(system_parameters::present_timestep - system_parameters::initial_elastic_iterations));
	}

	double zero_tolerance = 1e-3;
	double max_velocity = 0;
	for (typename DoFHandler<dim>::active_cell_iterator cell =
				dof_handler.begin_active(); cell != dof_handler.end(); ++cell)// loop over all cells
	{
		if(cell->at_boundary())
		{
			int zero_faces = 0;
			for(unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; f++)
				for(unsigned int i=0; i<dim; i++)
					if (fabs(cell->face(f)->center()[i]) < zero_tolerance)
						zero_faces++;
			if (zero_faces==0)
			{
				for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
				{
					Point<dim> vertex_velocity;
					Point<dim> vertex_position;
					for (unsigned int d = 0; d < dim; ++d)
					{
						vertex_velocity[d] = solution(cell->vertex_dof_index(v, d));
						vertex_position[d] = cell->vertex(v)[d];
					}
					//velocity to be evaluated is the radial component of a surface vertex
					double local_velocity = 0;
					for (unsigned int d = 0; d < dim; ++d)
					{
						local_velocity += vertex_velocity[d] * vertex_position [d];
					}
					local_velocity /= std::sqrt( vertex_position.square() );
					if(local_velocity < 0)
						local_velocity *= -1;
					if(local_velocity > max_velocity)
					{
						max_velocity = local_velocity;
										}
				}
			}
		}
	}
	// NOTE: It is possible for this time interval to be very different from that used in the viscoelasticity calculation.
	system_parameters::current_time_interval = move_goal_per_step / max_velocity;
	double step_time_yr = system_parameters::current_time_interval / SECSINYEAR;
	std::cout << "Timestep interval changed to: "
			<< step_time_yr
			<< " years\n";
}

//====================== MOVE MESH ======================

template<int dim>
void StokesProblem<dim>::move_mesh() {

	std::cout << "\n" << "   Moving mesh...\n";
	std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
	for (typename DoFHandler<dim>::active_cell_iterator cell =
			dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
		for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
			if (vertex_touched[cell->vertex_index(v)] == false) {
				vertex_touched[cell->vertex_index(v)] = true;

				Point<dim> vertex_displacement;
				for (unsigned int d = 0; d < dim; ++d)
					vertex_displacement[d] = solution(
							cell->vertex_dof_index(v, d));
				cell->vertex(v) += vertex_displacement
						* system_parameters::current_time_interval;
			}
}

//====================== WRITE MESH TO FILE ======================

template<int dim>
void StokesProblem<dim>::write_mesh()
{
	// output mesh in ucd
	std::ostringstream initial_mesh_file;
	initial_mesh_file << system_parameters::output_folder << "/time" <<
    Utilities::int_to_string(system_parameters::present_timestep, 2) <<
	"_mesh.inp";
	std::ofstream out_ucd (initial_mesh_file.str().c_str());
	GridOut grid_out;
	grid_out.write_ucd (triangulation, out_ucd);
}

//====================== FIT ELLIPSE TO SURFACE AND WRITE RADII TO FILE ======================

template<int dim>
void StokesProblem<dim>::do_ellipse_fits()
{
	std::ostringstream ellipses_filename;
	ellipses_filename << system_parameters::output_folder << "/ellipse_fits.txt";
	// Find ellipsoidal axes for all layers
	std::vector<double> ellipse_axes(0);
	// compute fit to boundary 0, 1, 2 ...
	std::cout << endl;
	for(unsigned int i = 0; i<system_parameters::sizeof_material_id;i++)
	{
		ellipsoid.compute_fit(ellipse_axes, system_parameters::material_id[i]);
		system_parameters::q_axes.push_back(ellipse_axes[0]);
		system_parameters::p_axes.push_back(ellipse_axes[1]);

		std::cout << "a_"<< system_parameters::material_id[i] <<" = " << ellipse_axes[0]
				<< " " << " c_"<< system_parameters::material_id[i] <<" = " << ellipse_axes[1] << std::endl;
		ellipse_axes.clear();

		std::ofstream fout_ellipses(ellipses_filename.str().c_str(), std::ios::app);
		fout_ellipses << system_parameters::present_timestep << " a_"<< system_parameters::material_id[i] <<" = " << ellipse_axes[0]
						<< " " << " c_"<< system_parameters::material_id[i] <<" = " << ellipse_axes[1] << endl;
		fout_ellipses.close();
	}
}

//====================== APPEND LINE TO PHYSICAL_TIMES.TXT FILE WITH STEP NUMBER, PHYSICAL TIME, AND # PLASTIC ITERATIONS ======================

template<int dim>
void StokesProblem<dim>::append_physical_times(int max_plastic)
{
	std::ostringstream times_filename;
	times_filename << system_parameters::output_folder << "/physical_times.txt";
	std::ofstream fout_times(times_filename.str().c_str(), std::ios::app);
	fout_times << system_parameters::present_timestep << " "
								<< system_parameters::present_time/SECSINYEAR << " " 
									<< max_plastic << "\n";
								// << system_parameters::q_axes[0] << " " << system_parameters::p_axes[0] << " "
								// << system_parameters::q_axes[1] << " " << system_parameters::p_axes[1] << "\n";
	fout_times.close();
}

//====================== WRITE VERTICES TO FILE ======================

template<int dim>
void StokesProblem<dim>::write_vertices(unsigned char boundary_that_we_need) {
	std::ostringstream vertices_output;
	vertices_output << system_parameters::output_folder << "/time" <<
		   Utilities::int_to_string(system_parameters::present_timestep, 2) << "_" <<
		   Utilities::int_to_string(boundary_that_we_need, 2) <<
		   "_surface.txt";
	std::ofstream fout_final_vertices(vertices_output.str().c_str());
	fout_final_vertices.close();

	std::vector<bool> vertex_touched(triangulation.n_vertices(), false);

	if (boundary_that_we_need == 0)
	{
	// Figure out if the vertex is on the boundary of the domain
	for (typename Triangulation<dim>::active_cell_iterator cell =
			triangulation.begin_active(); cell != triangulation.end(); ++cell)
		for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
		{
			unsigned char boundary_ids = cell->face(f)->boundary_indicator();
			if(boundary_ids == boundary_that_we_need)
				{
				for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
					if (vertex_touched[cell->face(f)->vertex_index(v)] == false)
						{
							vertex_touched[cell->face(f)->vertex_index(v)] = true;
							std::ofstream fout_final_vertices(vertices_output.str().c_str(), std::ios::app);
							fout_final_vertices << cell->face(f)->vertex(v) << "\n";
							fout_final_vertices.close();
						}
				}
		}
	}
	else
	{		
		// Figure out if the vertex is on an internal boundary
		for (typename Triangulation<dim>::active_cell_iterator cell =
				triangulation.begin_active(); cell != triangulation.end(); ++cell)
			for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
			{
				if (cell->neighbor(f) != triangulation.end()) {
					if (cell->material_id() != cell->neighbor(f)->material_id()) //finds face is at internal boundary
							{
						int high_mat_id = std::max(cell->material_id(),
								cell->neighbor(f)->material_id());
						if (high_mat_id == boundary_that_we_need) //finds faces at the correct internal boundary
								{
							for (unsigned int v = 0;
									v < GeometryInfo<dim>::vertices_per_face;
									++v)
								if (vertex_touched[cell->face(f)->vertex_index(
										v)] == false) {
									vertex_touched[cell->face(f)->vertex_index(
											v)] = true;
									std::ofstream fout_final_vertices(vertices_output.str().c_str(), std::ios::app);
									fout_final_vertices << cell->face(f)->vertex(v) << "\n";
									fout_final_vertices.close();
								}
						}
					}
				}
			}
		}
}

//====================== SETUP INITIAL MESH ======================

template<int dim>
void StokesProblem<dim>::setup_initial_mesh() {
	GridIn<dim> grid_in;
	grid_in.attach_triangulation(triangulation);
	std::ifstream mesh_stream(system_parameters::mesh_filename,
			std::ifstream::in);
	grid_in.read_ucd(mesh_stream);

	// output initial mesh in eps
	std::ostringstream initial_mesh_file;
	initial_mesh_file << system_parameters::output_folder << "/initial_mesh.eps";
	std::ofstream out_eps (initial_mesh_file.str().c_str());
	GridOut grid_out;
	grid_out.write_eps (triangulation, out_eps);
	out_eps.close();

// set boundary ids
// boundary indicator 0 is outer free surface; 1, 2, 3 ... is boundary between layers, 99 is flat boundaries
    typename Triangulation<dim>::active_cell_iterator
    		      cell=triangulation.begin_active(), endc=triangulation.end();

    unsigned int how_many; // how many components away from cardinal planes

	std::ostringstream boundaries_file;
	boundaries_file << system_parameters::output_folder << "/boundaries.txt";
	std::ofstream fout_boundaries(boundaries_file.str().c_str());
	fout_boundaries.close();

	double zero_tolerance = 1e-3;
	for (; cell != endc; ++cell) // loop over all cells
	{
		for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) // loop over all vertices
		{
			if (cell->face(f)->at_boundary())
			{
				// print boundary
			    std::ofstream fout_boundaries(boundaries_file.str().c_str(), std::ios::app);
			    fout_boundaries << cell->face(f)->center()[0] << " " << cell->face(f)->center()[1]<< "\n";
			    fout_boundaries.close();

				how_many = 0;
				for(unsigned int i=0; i<dim; i++)
						if (fabs(cell->face(f)->center()[i]) > zero_tolerance)
					    how_many++;
				if (how_many==dim)
					cell->face(f)->set_all_boundary_indicators(0); // if face center coordinates > zero_tol, set bnry indicators to 0
				else
				    cell->face(f)->set_all_boundary_indicators(99);
			}
		}
	}

	std::ostringstream ellipses_filename;
	ellipses_filename << system_parameters::output_folder << "/ellipse_fits.txt";
	std::ofstream fout_ellipses(ellipses_filename.str().c_str());
	fout_ellipses.close();

	// Find ellipsoidal axes for all layers
	std::vector<double> ellipse_axes(0);
	// compute fit to boundary 0, 1, 2 ...
	std::cout << endl;
	for(unsigned int i = 0; i<system_parameters::sizeof_material_id;i++)
	{
		ellipsoid.compute_fit(ellipse_axes, system_parameters::material_id[i]);
		system_parameters::q_axes.push_back(ellipse_axes[0]);
		system_parameters::p_axes.push_back(ellipse_axes[1]);

		std::cout << "a_"<< system_parameters::material_id[i] <<" = " << ellipse_axes[0]
				<< " " << " c_"<< system_parameters::material_id[i] <<" = " << ellipse_axes[1] << std::endl;
		ellipse_axes.clear();

		std::ofstream fout_ellipses(ellipses_filename.str().c_str(), std::ios::app);
		fout_ellipses << system_parameters::present_timestep << " a_"<< system_parameters::material_id[i] <<" = " << ellipse_axes[0]
						<< " " << " c_"<< system_parameters::material_id[i] <<" = " << ellipse_axes[1] << endl;
		fout_ellipses.close();
	}

	triangulation.refine_global(system_parameters::global_refinement);


//refines crustal region
	if (system_parameters::crustal_refinement != 0) {
		double a = system_parameters::q_axes[0] - system_parameters::crust_refine_region;
		double b = system_parameters::p_axes[0] - system_parameters::crust_refine_region;


		for (unsigned int step = 0;
				step < system_parameters::crustal_refinement; ++step) {
			typename dealii::Triangulation<dim>::active_cell_iterator cell =
					triangulation.begin_active(), endc = triangulation.end();
			for (; cell != endc; ++cell)
				for (unsigned int v = 0;
						v < GeometryInfo<dim>::vertices_per_cell; ++v) {
					Point<dim> current_vertex = cell->vertex(v);

					const double x_coord = current_vertex.operator()(0);
					const double y_coord = current_vertex.operator()(1);
					double expected_z = -1;

					if ((x_coord - a) < -1e-10)
						expected_z = b
								* std::sqrt(1 - (x_coord * x_coord / a / a));

					if (y_coord >= expected_z) {
						cell->set_refine_flag();
						break;
					}
				}
			triangulation.execute_coarsening_and_refinement();
		}
	}


	// output initial mesh in eps
	std::ostringstream refined_mesh_file;
	refined_mesh_file << system_parameters::output_folder << "/refined_mesh.eps";
	std::ofstream out_eps_refined (refined_mesh_file.str().c_str());
	GridOut grid_out_refined;
	grid_out_refined.write_eps (triangulation, out_eps_refined);
	out_eps_refined.close();
	write_vertices(0);
	write_vertices(1);
	write_mesh();
}

//====================== REFINE MESH ======================

template<int dim>
void StokesProblem<dim>::refine_mesh() {
	Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

	std::vector<bool> component_mask(dim + 1, false);
	component_mask[dim] = true;
	KellyErrorEstimator<dim>::estimate(dof_handler, QGauss<dim - 1>(degree + 1),
			typename FunctionMap<dim>::type(), solution,
			estimated_error_per_cell, component_mask);

	GridRefinement::refine_and_coarsen_fixed_number(triangulation,
			estimated_error_per_cell, 0.3, 0.0);
	triangulation.execute_coarsening_and_refinement();
}

//====================== SET UP THE DATA STRUCTURES TO REMEMBER STRESS FIELD ======================
template<int dim>
void StokesProblem<dim>::setup_quadrature_point_history() {
	unsigned int our_cells = 0;
	for (typename Triangulation<dim>::active_cell_iterator cell =
			triangulation.begin_active(); cell != triangulation.end(); ++cell)
		++our_cells;

	triangulation.clear_user_data();

	quadrature_point_history.resize(our_cells * quadrature_formula.size());

	unsigned int history_index = 0;
	for (typename Triangulation<dim>::active_cell_iterator cell =
			triangulation.begin_active(); cell != triangulation.end(); ++cell) {
		cell->set_user_pointer(&quadrature_point_history[history_index]);
		history_index += quadrature_formula.size();
	}

	Assert(history_index == quadrature_point_history.size(), ExcInternalError());
}

//====================== DOES ELASTIC STEPS ======================
template<int dim>
void StokesProblem<dim>::do_elastic_steps()
{
	unsigned int elastic_iteration = 0;

	while (elastic_iteration < system_parameters::initial_elastic_iterations)
	{

		std::cout << "\n\nElastic iteration " << elastic_iteration
							<< "\n";
		setup_dofs();

		if (system_parameters::present_timestep == 0)
			initialize_eta_and_G();
		
		if(elastic_iteration == 0)
			system_parameters::current_time_interval =
				system_parameters::viscous_time; //This is the time interval needed in assembling the problem

		std::cout << "   Assembling..." << std::endl << std::flush;
		assemble_system();

		std::cout << "   Solving..." << std::flush;
		solve();

		output_results();
		update_quadrature_point_history();

		append_physical_times(0);
		elastic_iteration++;
		system_parameters::present_timestep++;
		do_ellipse_fits();
		write_vertices(0);
	    write_vertices(1);
		write_mesh();
		update_time_interval();
	}
}

//====================== DO A SINGLE VISCOELASTOPLASTIC TIMESTEP ======================
template<int dim>
void StokesProblem<dim>::do_flow_step() {
	plastic_iteration = 0;
	while (plastic_iteration < system_parameters::max_plastic_iterations) {
		if (system_parameters::continue_plastic_iterations == true) {
			std::cout << "Plasticity iteration " << plastic_iteration << "\n";
			setup_dofs();

			std::cout << "   Assembling..." << std::endl << std::flush;
			assemble_system();

			std::cout << "   Solving..." << std::flush;
			solve();

			output_results();
			solution_stesses();

			if (system_parameters::continue_plastic_iterations == false) 
				break;
			
			plastic_iteration++;
		}
	}
}

//====================== RUN ======================

template<int dim>
void StokesProblem<dim>::run()
{
	// Sets up mesh and data structure for viscosity and stress at quadrature points
	setup_initial_mesh();
	setup_quadrature_point_history();
	
	// Makes the physical_times.txt file
	std::ostringstream times_filename;
	times_filename << system_parameters::output_folder << "/physical_times.txt";
	std::ofstream fout_times(times_filename.str().c_str());
	fout_times.close();
	
	// Computes elastic timesteps
	do_elastic_steps();
	// Computes viscous timesteps
	unsigned int VEPstep = 0;
	while (system_parameters::present_timestep
			< (system_parameters::initial_elastic_iterations
					+ system_parameters::total_viscous_steps)) {
						
		if (system_parameters::continue_plastic_iterations == false)
			system_parameters::continue_plastic_iterations = true;
		std::cout << "\n\nViscoelastoplastic iteration " << VEPstep << "\n\n";
		// Computes plasticity
		do_flow_step();
		update_quadrature_point_history();
		move_mesh();
		append_physical_times(plastic_iteration);
		system_parameters::present_timestep++;
		system_parameters::present_time = system_parameters::present_time + system_parameters::current_time_interval;
		do_ellipse_fits();
		write_vertices(0);
		write_vertices(1);
		write_mesh();
		VEPstep++;
	}
		append_physical_times(-1);
 }

}

//====================== MAIN ======================

int main(int argc, char* argv[]) {

	// output program name
	std::cout << "Running: " << argv[0] << std::endl;

	char* cfg_filename = new char[120];

	if (argc == 1) // if no input parameters (as if launched from eclipse)
	{
		std::strcpy(cfg_filename,"config/ConfigurationV2.cfg");
	}
	else
		std::strcpy(cfg_filename,argv[1]);

	try {
		using namespace dealii;
		using namespace Step22;
		config_in cfg(cfg_filename);

		std::clock_t t1;
		std::clock_t t2;
		t1 = std::clock();

		deallog.depth_console(0);

		StokesProblem<2> flow_problem(1);
        flow_problem.run();

		std::cout << std::endl << "\a";

		t2 = std::clock();
		float diff (((float)t2 - (float)t1) / (float)CLOCKS_PER_SEC);
		std::cout  << "\n Program run in: " << diff << " seconds" << endl;
	} catch (std::exception &exc) {
		std::cerr << std::endl << std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		std::cerr << "Exception on processing: " << std::endl << exc.what()
				<< std::endl << "Aborting!" << std::endl
				<< "----------------------------------------------------"
				<< std::endl;

		return 1;
	} catch (...) {
		std::cerr << std::endl << std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		std::cerr << "Unknown exception!" << std::endl << "Aborting!"
				<< std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		return 1;
	}

	return 0;
}
