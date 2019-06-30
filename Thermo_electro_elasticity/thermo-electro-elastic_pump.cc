/* =========================
 * THERMO-ELECTRO-ELASTICITY
 * =========================
 * Problem description:
 *   Axial deformation (elongation) of a cylinder, with an electric
 *   field induced in the axial direction and a temperature gradient
 *   prescribed between the inner and outer radial surfaces.
 *
 * Initial implementation
 *   Author: Markus Mehnert (2015)
 *           Friedrich-Alexander University Erlangen-Nuremberg
 *   Description:
 *           Staggered one-way coupling of quasi-static
 *           linear-elasticity and thermal (conductivity) problems
 *
 * Extensions
 *   Author: Jean-Paul Pelteret (2015)
 *            Friedrich-Alexander University Erlangen-Nuremberg
 *   Description:
 *       [X] Nonlinear, finite deformation quasi-static  elasticity
 *       [X] Nonlinear quasi-static thermal (conductivity) problem
 *       [X] Nonlinear iterative solution scheme (Newton-Raphson)
 *           that encompasses staggered thermal / coupled EM
 *           solution update
 *       [X] Parallelisation via Trilinos (and possibly PETSc)
 *       [X] Parallel output of solution, residual
 *       [X] Choice of direct and indirect solvers
 *       [X] Adaptive grid refinement using Kelly error estimator
 *       [X] Parameter collection
 *       [X] Generic continuum point framework for integrating
 *           constitutive models
 *       [X] Nonlinear constitutive models
 *          [X] St. Venant Kirchoff
 *              + Materially linear thermal conductivity
 *          [X] Fully decoupled NeoHookean
 *              + Materially isotropic dielectric material
 *              + Spatially isotropic thermal conductivity
 *          [X] One-way coupled thermo-electro-mechanical model
 *              based on Markus' paper (Mehnert2015a)
 *              + Spatially isotropic thermal conductivity
 *
 *  References:
 *  Wriggers, P. Nonlinear finite element methods. 2008
 *  Holzapfel, G. A. Nonlinear solid mechanics. 2007
 *  Vu, K., On coupled BEM-FEM simulation of nonlinear electro-elastostatics
 */

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_selector.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/block_sparsity_pattern.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/manifold.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/fe/fe_nothing.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

// #include <deal.II/differentiation/ad.h>
#include <deal.II/physics/elasticity/kinematics.h>
// #include <deal.II/differentiation/ad/sacado_product_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>
#include <deal.II/differentiation/ad/sacado_math.h>
#include <Sacado.hpp>


#include <deal.II/lac/generic_linear_algebra.h>
#define USE_TRILINOS_LA
namespace LA
{
#ifdef USE_TRILINOS_LA
using namespace dealii::LinearAlgebraTrilinos;
#else
using namespace dealii::LinearAlgebraPETSc;
#endif
}

#include <mpi.h>
#include <fstream>
#include <iostream>

namespace Coupled_TEE
{
using namespace dealii;


struct Parameters
{
	// Formulation
	static const bool use_3_Field = true;

	// Geometry file
	static const std::string mesh_file;

	// Boundary ids
	static constexpr unsigned int boundary_id_bottom = 0;
	static constexpr unsigned int boundary_id_top = 1;
	static constexpr unsigned int boundary_id_inner_radius = 2;
	static constexpr unsigned int boundary_id_outer_radius = 3;
	static constexpr unsigned int boundary_id_cut_bottom = 4;
	static constexpr unsigned int boundary_id_cut_left = 5;
	static constexpr unsigned int boundary_id_frame = 6;
	static constexpr unsigned int boundary_id_outlet_inner_radius = 7;
	static constexpr unsigned int boundary_id_outlet_outer_radius = 8;
	static constexpr unsigned int boundary_id_cut_outlet = 9;

	// Boundary conditions
	static constexpr double Temperature_Difference = 0.0;

	//    Potential difference in MV
	static constexpr double potential_difference = 0.04; // 0.1

	// Time
	static constexpr double dt = 0.1;
	static constexpr unsigned int n_timesteps = 10;

	//J-Ps additions

	static constexpr double time_end = 50.0e-3;
	static constexpr double time_delta = time_end/(static_cast<double>(n_timesteps));

	// Refinement
	static constexpr unsigned int n_global_refinements = 0;
	static constexpr bool perform_AMR = false;
	static constexpr unsigned int n_ts_per_refinement = 10;
	static constexpr unsigned int max_grid_level = 5;
	static constexpr double frac_refine = 0.3;
	static constexpr double frac_coarsen = 0.03;

	// Finite element
	static constexpr unsigned int poly_order = 1;

	// Nonlinear solver
	static constexpr unsigned int max_newton_iterations = 20;
	static constexpr double max_res_T_norm = 1e-6;
	static constexpr double max_res_uV_norm = 1e-9;
	static constexpr double max_res_abs = 1e-6;

	// Linear solver: Thermal
	static const std::string solver_type_T;
	static constexpr double tol_rel_T = 1e-6;

	// Linear solver: Electro-mechanical
	static const std::string solver_type_EM;
	static constexpr double tol_rel_EM = 1e-6;
};
const std::string Parameters::mesh_file = "Pump_um_coarse.inp";
const std::string Parameters::solver_type_T = "Direct";
const std::string Parameters::solver_type_EM = "Direct";

namespace Material
{
struct Coefficients
{
	static constexpr double length_scale = 1.0;

	// Parameters in N, mm, V

	// Elastic parameters
	static constexpr double g_0 = 0.1e-6;// in N/um^2
	static constexpr double N = 7.84e5;

	// Electro parameters
	static constexpr double epsilon_0 = 8.854187817; // F/m = C/(V*m)= (A*s)/(V*m) = N/(uV*uV)
	static constexpr double c_1 = epsilon_0;
	static constexpr double c_2 = 2000*epsilon_0;


	// Independent of length and voltage units

	static constexpr double nu = 0.499; // Poisson ratio
	static constexpr double mu = g_0; // Small strain shear modulus

	static constexpr double lambda = 2.0*mu*nu/(1.0-2.0*nu); // Lame constant
	static constexpr double kappa = 2.0*mu*(1.0+nu)/(3.0*(1.0-2.0*nu)); // mu = 3*10^5 Pa

	// Thermal parameters
	static constexpr double c_0 = 460e6; // specific heat capacity in J/(kg*K)
	static constexpr double alpha = 20e-6; //thermal expansion coefficient in 1/K
	static constexpr double theta_0 = 293; // in K
	static constexpr double k = 0.50; // Heat conductivity in N/(s*K)



};
template<int dim>
struct Values_ad
{
	typedef Sacado::Fad::DFad<double> ad_type;

	Values_ad (const Tensor<2,dim,ad_type> & F,
			const Tensor<1,dim,ad_type> & E,
			const Tensor<1,dim> & Grad_T,
			const double theta,
			const ad_type J_tilde,
			const ad_type p,
			const double alpha,
			const double c_2)
	: F (F),
	  E (E),
	  Grad_T(Grad_T),
	  theta(theta),
	  J_tilde(J_tilde),
	  p(p),
	  alpha(alpha),
	  c_2(c_2),

	  C (symmetrize(transpose(F)*F)),
	  C_inv (symmetrize(invert(static_cast<Tensor<2,dim,ad_type> >(C)))),
	  J (determinant(F)),
	  C_bar(ad_type(std::pow(J ,-2.0/dim))*C),
	  C_bar_inv (symmetrize(invert(static_cast<Tensor<2,dim,ad_type> >(C_bar)))),

	  J_theta (ad_type(std::exp(dim*alpha*(theta-Material::Coefficients::theta_0)))),
	  F_theta (J_theta*unit_symmetric_tensor<dim,ad_type>()),
	  F_theta_inv (symmetrize(invert(static_cast<Tensor<2,dim,ad_type> >(F_theta)))),

	  F_EM (F_theta_inv*F),
	  J_EM (J/J_theta),
	  C_EM (symmetrize(transpose(F_EM)*F_EM)),
	  C_EM_inv (symmetrize(invert(static_cast<Tensor<2,dim,ad_type> >(C_EM)))),

	  C_EM_bar(ad_type(std::pow(J_EM ,-2.0/dim))*C_EM),
	  C_EM_bar_inv(symmetrize(invert(static_cast<Tensor<2,dim,ad_type> >(C_EM_bar)))),


	  I1 (first_invariant(C)), // tr(C)
	  I1_EM (first_invariant(C_EM)), // tr(C)
	  I1_bar (first_invariant(C_bar)),
	  I1_EM_bar (first_invariant(C_EM_bar)),
	  I4 (E*E), // [ExE].I
	  I5 (E*(C*E)), // [ExE].C
	  I5_bar (E*(C_bar*E)), // [ExE].C
	  I5_EM_bar (E*(C_EM_bar*E)) // [ExE].C

	{}
	//
	//	// Directly related to solution field
	const Tensor<2,dim,ad_type> F; // Deformation gradient
	const Tensor<1,dim,ad_type> E;
	const Tensor<1,dim> Grad_T;
	const double theta;
	const ad_type J_tilde;
	const ad_type p;
	const double alpha;
	const double c_2;

	// Commonly used elastic quantities
	const SymmetricTensor<2,dim,ad_type> C; // Right Cauchy-Green deformation tensor
	const SymmetricTensor<2,dim,ad_type> C_inv;
	const ad_type J;

	const SymmetricTensor<2,dim,ad_type> C_bar;
	const SymmetricTensor<2,dim,ad_type> C_bar_inv;

	const ad_type J_theta;
	const SymmetricTensor<2,dim,ad_type> F_theta;
	const SymmetricTensor<2,dim,ad_type> F_theta_inv;

	const Tensor<2,dim,ad_type> F_EM;
	const ad_type J_EM;
	const SymmetricTensor<2,dim,ad_type> C_EM;
	const SymmetricTensor<2,dim,ad_type> C_EM_inv;

	const SymmetricTensor<2,dim,ad_type> C_EM_bar;
	const SymmetricTensor<2,dim,ad_type> C_EM_bar_inv;



	//
	//	// Invariants
	const ad_type I1;
	const ad_type I1_EM;
	const ad_type I1_bar;
	const ad_type I1_EM_bar;
	const ad_type I4;
	const ad_type I5;
	const ad_type I5_bar;
	const ad_type I5_EM_bar;

	// === First derivatives ===
	// --- Mechanical ---
	SymmetricTensor<2,dim,ad_type>
	dI1_dC () const
	{
		return unit_symmetric_tensor<dim,ad_type>();
	}

	SymmetricTensor<2,dim,ad_type>
	dI1_bar_dC_bar () const
	{
		return unit_symmetric_tensor<dim,ad_type>();
	}


	SymmetricTensor<2,dim,ad_type>
	dI1_EM_bar_dC_EM_bar () const
	{
		return unit_symmetric_tensor<dim,ad_type>();
	}

	SymmetricTensor<2,dim,ad_type>
	dI4_dC () const
	{
		return SymmetricTensor<2,dim,ad_type>();
	}

	SymmetricTensor<2,dim,ad_type>
	dI4_dC_bar () const
	{
		return SymmetricTensor<2,dim,ad_type>();
	}

	SymmetricTensor<2,dim,ad_type>
	dI5_dC () const
	{
		const SymmetricTensor<2,dim,ad_type> ExE = symmetrize(outer_product(E,E));
		return ExE;
	}

	SymmetricTensor<2,dim,ad_type>
	dI5_bar_dC_bar () const
	{
		const SymmetricTensor<2,dim,ad_type> ExE = symmetrize(outer_product(E,E));
		return ExE;
	}

	SymmetricTensor<2,dim,ad_type>
	dI5_EM_bar_dC_EM_bar () const
	{
		const SymmetricTensor<2,dim,ad_type> ExE = symmetrize(outer_product(E,E));
		return ExE;
	}

	SymmetricTensor<2,dim,ad_type>
	dJ_dC () const
	{
		return ad_type((0.5*J))*C_inv;
	}

	SymmetricTensor<2,dim,ad_type>
	dJ_EM_dC_EM () const
	{
		return ad_type((0.5*J_EM))*C_EM_inv;
	}

	SymmetricTensor<4,dim,ad_type>
	dC_inv_dC () const
	{
		SymmetricTensor<4,dim,ad_type> dC_inv_dC;

		for (unsigned int A=0; A<dim; ++A)
			for (unsigned int B=A; B<dim; ++B)
				for (unsigned int C=0; C<dim; ++C)
					for (unsigned int D=C; D<dim; ++D)
						dC_inv_dC[A][B][C][D] = -0.5*(C_inv[A][C]*C_inv[B][D]+ C_inv[A][D]*C_inv[B][C]);

		return dC_inv_dC;
	}


	//	 --- Electric ---
	Tensor<1,dim,ad_type>
	dI4_dE () const
	{
		return 2.0*E;
	}

	Tensor<1,dim,ad_type>
	dI5_dE () const
	{
		return 2.0*(C*E);
	}

	Tensor<1,dim,ad_type>
	dI5_bar_dE () const
	{
		return 2.0*(C_bar*E);
	}

	Tensor<1,dim,ad_type>
	dI5_EM_bar_dE () const
	{
		return 2.0*(C_EM_bar*E);
	}

};

template<int dim>
struct CM_Base_ad
{
	typedef Sacado::Fad::DFad<double> ad_type;

	CM_Base_ad (const Tensor<2,dim,ad_type> & F,
			const Tensor<1,dim,ad_type> & E,
			const Tensor<1,dim> & Grad_T,
			const double theta,
			const ad_type J_tilde,
			const ad_type p,
			const double alpha,
			const double c_2)
	: values_ad (F,E,Grad_T,theta,J_tilde,p,alpha,c_2)
	{}

	virtual ~CM_Base_ad () {}

	// --- Kinematic Quantities ---

	const Values_ad<dim> values_ad;

	// --- Kinetic Quantities ---

	// Second Piola-Kirchhoff stress tensor
	inline SymmetricTensor<2,dim,ad_type>
	get_S_3Field () const
	{
		const double theta_ratio = values_ad.theta/293.0;
		const ad_type &J_theta = this->values_ad.J_theta;
		return ad_type(std::pow(J_theta ,-2.0/dim))*(get_S_iso_3Field()+2*theta_ratio*get_dPsi_p_dC_EM());
	}

	inline SymmetricTensor<2,dim,ad_type>
	get_S_1Field () const
	{
		return (get_S_iso_1Field()+get_S_vol());
	}


	inline SymmetricTensor<2,dim,ad_type>
	get_S_iso_3Field () const
	{
		return 2.0*get_dPsi_iso_dC_EM();
	}

	inline SymmetricTensor<2,dim,ad_type>
	get_S_iso_1Field () const
	{
		return 2.0*get_dPsi_iso_dC();
	}

	inline SymmetricTensor<2,dim,ad_type>
	get_S_vol () const
	{
		return 2.0*get_dPsi_vol_dC();
	}

	// Referential electric displacement vector
	inline Tensor<1,dim,ad_type>
	get_D () const
	{
		return -get_dPsi_dE();
	}

	inline ad_type
	get_dPsi_dp () const
	{
		const ad_type &J_EM = this->values_ad.J_EM;
		const ad_type &J_tilde = this->values_ad.J_tilde;

		return  J_EM - J_tilde;
	}

	inline ad_type
	get_dPsi_dJ_tilde () const
	{
		const ad_type &p = this->values_ad.p;
		const ad_type &J_tilde = this->values_ad.J_tilde;

		double kappa=Material::Coefficients::kappa;


		return 0.5* kappa * (J_tilde-1.0/J_tilde) - p;
	}


protected:
	// --- Pure mechanical volumetric response ---

	// Derivative of the volumetric free energy with respect to
	// $\widetilde{J}$ return $\frac{\partial
	// \Psi_{\text{vol}}(\widetilde{J})}{\partial \widetilde{J}}$
	ad_type
	get_dW_vol_elastic_dJ (const double kappa,
			const double g_0) const
	{
		const ad_type &J = values_ad.J;
		return (0.5*(kappa)*(J-1/J));

	}

	// Derivative of the volumetric free energy with respect to
	// $\widetilde{J}$ return $\frac{\partial
	// \Psi_{\text{vol}}(\widetilde{J})}{\partial \widetilde{J}}$
	ad_type
	get_dW_J_dJ (const double kappa,
			const double g_0) const
	{
		const ad_type &J = values_ad.J;
		return (0.5*(kappa)*(J-1/J));

	}

	SymmetricTensor<2,dim,ad_type>
	get_dW_J_dC (const double kappa,
			const double g_0) const
			{
		// See Wriggers p46 eqs. 3.123, 3.124; Holzapfel p230
		return get_dW_J_dJ(kappa, g_0)*this->values_ad.dJ_dC();
			}

	ad_type
	get_dW_vol_elastic_dJ_EM (const double kappa,
			const double g_0) const
	{
		const ad_type &J_EM = values_ad.J_EM;
		return (0.5*(kappa)*(J_EM-1/J_EM));

	}

	SymmetricTensor<2,dim,ad_type>
	get_dW_vol_elastic_dC (const double kappa,
			const double g_0) const
			{
		// See Wriggers p46 eqs. 3.123, 3.124; Holzapfel p230
		return get_dW_vol_elastic_dJ(kappa, g_0)*this->values_ad.dJ_dC();
			}
	SymmetricTensor<2,dim,ad_type>
	get_dW_vol_elastic_dC_EM (const double kappa,
			const double g_0) const
			{
		// See Wriggers p46 eqs. 3.123, 3.124; Holzapfel p230
		return get_dW_vol_elastic_dJ_EM(kappa, g_0)*this->values_ad.dJ_EM_dC_EM();
			}

	SymmetricTensor<2,dim,ad_type>
	get_dPsi_p_dC () const
	{
		const ad_type &p = values_ad.p;
		return p*this->values_ad.dJ_dC();

	}

	SymmetricTensor<2,dim,ad_type>
	get_dPsi_p_dC_EM () const
	{

		const ad_type &p = values_ad.p;
		return p*this->values_ad.dJ_EM_dC_EM();

	}


	// --- Coupled mechanical response ---

	virtual SymmetricTensor<2,dim,ad_type>
	get_dPsi_vol_dC () const = 0;

	virtual SymmetricTensor<2,dim,ad_type>
	get_dPsi_vol_dC_EM () const = 0;

	virtual SymmetricTensor<2,dim,ad_type>
	get_dPsi_iso_dC () const = 0;

	virtual SymmetricTensor<2,dim,ad_type>
	get_dPsi_iso_dC_EM () const = 0;

	//	virtual SymmetricTensor<2,dim,ad_type>
	//	get_dPsi_iso_dC2 () const = 0;

	// --- Coupled electric response ---

	virtual Tensor<1,dim,ad_type>
	get_dPsi_dE () const = 0;

};

//
//template<int dim>
//struct CM_Incompressible_Uncoupled_8Chain_ad : public CM_Base_ad<dim>
//{
//	typedef Sacado::Fad::DFad<double> ad_type;
//	CM_Incompressible_Uncoupled_8Chain_ad  (const Tensor<2,dim,ad_type> & F,
//			const Tensor<1,dim,ad_type> & E,
//			const Tensor<1,dim> & Grad_T,
//			const double theta,
//			const ad_type J_tilde,
//			const ad_type p,
//			const double alpha,
//			const double c_2)
//	: CM_Base_ad<dim> (F,E,Grad_T,theta,J_tilde,p,alpha,c_2)
//	  {}
//
//
//	virtual ~CM_Incompressible_Uncoupled_8Chain_ad () {}
//
//protected:
//
//	virtual SymmetricTensor<2,dim,ad_type>
//	get_dPsi_iso_dC () const
//	{
//
//		double mu=Material::Coefficients::g_0;
//		double N=Material::Coefficients::N;
//		double c_2=this->values_ad.c_2;
//
//		return theta_ratio()*(get_dW_FE_iso_elastic_dC(mu,N,c_2));
//	}
//
//	virtual SymmetricTensor<2,dim,ad_type>
//	get_dPsi_iso_dC_EM () const
//	{
//
//
//		double mu=Material::Coefficients::g_0;
//		double N=Material::Coefficients::N;
//		double c_2=this->values_ad.c_2;
//
//		return theta_ratio()*(get_dW_FE_iso_elastic_dC_EM(mu,N,c_2));
//	}
//
//
//	virtual SymmetricTensor<2,dim,ad_type>
//	get_dPsi_vol_dC () const
//	{
//		return theta_ratio()*this->get_dW_J_dC(Material::Coefficients::kappa, Material::Coefficients::mu)
//				- theta_difference()*get_dM_J_dC(Coefficients::kappa,Coefficients::alpha); // Thermal dilatory response M = M(J);
//	}
//
//	virtual SymmetricTensor<2,dim,ad_type>
//	get_dPsi_vol_dC_EM () const
//	{
//		return unit_symmetric_tensor<dim,ad_type>();
//	}
//
//	inline SymmetricTensor<2,dim,ad_type>
//	get_dW_FE_iso_elastic_dC (const double g_0,
//			const double N,
//			const double c_2) const
//			{
//
////		const ad_type &lambda=get_lambda();
////
////		return	ad_type((0.5*g_0/dim)*((dim*N-lambda*lambda)/(N-lambda*lambda)))*
////				(unit_symmetric_tensor<dim,ad_type>());
//		const SymmetricTensor<2,dim,ad_type> &C_inv = this->values_ad.C_inv;
//		 return ad_type(g_0)*(unit_symmetric_tensor<dim,ad_type>()-C_inv);
//
//
//			}
//
//	inline SymmetricTensor<2,dim,ad_type>
//	get_dW_FE_iso_elastic_dC_EM (const double g_0,
//			const double N,
//			const double c_2) const
//			{
//
//		const SymmetricTensor<4,dim,ad_type> P= get_Dev_P(this->values_ad.F_EM);//get_dC_bar_dC();
//		const SymmetricTensor<2,dim,ad_type> dW_FE_dC_EM_bar=get_dW_FE_iso_elastic_dC_EM_bar(g_0,N,c_2);
//
//		return (dW_FE_dC_EM_bar*P);
//
//
//			}
//
//	inline SymmetricTensor<2,dim,ad_type>
//	get_dW_FE_iso_elastic_dC_bar (const double g_0,
//			const double N,
//			const double c_2) const
//			{
////		const ad_type &lambda_bar=get_lambda_bar();
////
////		return	ad_type((0.5*g_0/dim)*((dim*N-lambda_bar*lambda_bar)/(N-lambda_bar*lambda_bar)))*
////				(unit_symmetric_tensor<dim,ad_type>());
//
//		 return g_0 /2.0*this->values_ad.dI1_bar_dC_bar();
//
//			}
//
//	inline SymmetricTensor<2,dim,ad_type>
//	get_dW_FE_iso_elastic_dC_EM_bar (const double g_0,
//			const double N,
//			const double c_2) const
//			{
////		const ad_type &lambda_bar=get_lambda_bar();
////
////		return	ad_type((0.5*g_0/dim)*((dim*N-lambda_bar*lambda_bar)/(N-lambda_bar*lambda_bar)))*
////				(unit_symmetric_tensor<dim,ad_type>());
//
//		 return g_0 /2.0*this->values_ad.dI1_EM_bar_dC_EM_bar();
//
//			}
//
//	inline ad_type
//	get_dM_J_dJ (const double kappa, const double alpha) const
//	{
//		const ad_type &J = this->values_ad.J;
//		return dim*kappa*alpha/J;
//	}
//
//	SymmetricTensor<2,dim,ad_type>
//	get_dM_J_dC (const double kappa,const double alpha) const
//	{
//		// See Wriggers p46 eqs. 3.123, 3.124; Holzapfel p230
//		return get_dM_J_dJ(kappa, alpha)*this->values_ad.dJ_dC();
//	}
//
//
//	inline SymmetricTensor<4, dim, ad_type>
//	get_Dev_P (const Tensor<2, dim, ad_type> &F) const
//	{
//		const ad_type det_F = determinant(F);
//		Assert(det_F > ad_type(0.0),
//				ExcMessage("Deformation gradient has a negative determinant."));
//		const Tensor<2,dim,ad_type> C_ns = transpose(F)*F;
//		const SymmetricTensor<2,dim,ad_type> C = symmetrize(C_ns);
//		const SymmetricTensor<2,dim,ad_type> C_inv = symmetrize(invert(C_ns));
//
//		// See Wriggers p46 equ 3.125 (but transpose indices)
//		SymmetricTensor<4,dim,ad_type> Dev_P = outer_product(C,C_inv);  // Dev_P = C_x_C_inv
//		Dev_P /= -dim;                                                  // Dev_P = -[1/dim]C_x_C_inv
//		Dev_P += Physics::Elasticity::StandardTensors< dim >::S;        // Dev_P = S - [1/dim]C_x_C_inv
//		Dev_P *= ad_type(std::pow(det_F, -2.0/dim));                    // Dev_P = J^{-2/dim} [S - [1/dim]C_x_C_inv]
//
//		return Dev_P;
//	}
//
//
//	// --- Electric contributions ---
//	virtual Tensor<1,dim,ad_type>
//	get_dPsi_dE () const
//	{
//		double c_1=Material::Coefficients::c_1;
//		double c_2=Material::Coefficients::c_2;
//
//
//		return get_dW_iso_elastic_dE(c_1,c_2) ;
//	}
//
//	inline Tensor<1,dim,ad_type>
//	get_dW_iso_elastic_dE (const double c_1,
//			const double c_2) const
//			{
//		return c_1*this->values_ad.dI4_dE();
//			}
//
//	ad_type
//	get_lambda() const
//	{
//		const ad_type &I1=this->values_ad.I1;
//		return std::sqrt(I1/dim);
//	}
//
//	ad_type
//	get_lambda_bar() const
//	{
//		const ad_type &I1_bar=this->values_ad.I1_bar;
//		return std::sqrt(I1_bar/dim);
//	}
//
//	ad_type
//	get_lambda_EM_bar() const
//	{
//		const ad_type &I1_EM_bar=this->values_ad.I1_EM_bar;
//		return std::sqrt(I1_EM_bar/dim);
//	}
//
//	double
//	theta_ratio () const
//	{
//		return this->values_ad.theta/Material::Coefficients::theta_0;
//	}
//
//	double
//	theta_difference () const
//	{
//		return this->values_ad.theta - Material::Coefficients::theta_0;
//	}
//
//};


template<int dim>
struct CM_Coupled_NeoHooke_ad : public CM_Base_ad<dim>
{
	typedef Sacado::Fad::DFad<double> ad_type;
	CM_Coupled_NeoHooke_ad  (const Tensor<2,dim,ad_type> & F,
			const Tensor<1,dim,ad_type> & E,
			const Tensor<1,dim> & Grad_T,
			const double theta,
			const ad_type J_tilde,
			const ad_type p,
			const double alpha,
			const double c_2)
	: CM_Base_ad<dim> (F,E,Grad_T,theta,J_tilde,p,alpha,c_2)
	  {}


	virtual ~CM_Coupled_NeoHooke_ad () {}

protected:

	virtual SymmetricTensor<2,dim,ad_type>
	get_dPsi_iso_dC () const
	{

		double mu=Material::Coefficients::g_0;
		double N=Material::Coefficients::N;
		double c_2=this->values_ad.c_2;

		return theta_ratio()*(get_dW_FE_iso_elastic_dC(mu,N,c_2));
	}

	virtual SymmetricTensor<2,dim,ad_type>
	get_dPsi_iso_dC_EM () const
	{


		double mu=Material::Coefficients::g_0;
		double N=Material::Coefficients::N;
		double c_2=this->values_ad.c_2;


		return theta_ratio()*(get_dW_FE_iso_elastic_dC_EM(mu,N,c_2));
	}


	virtual SymmetricTensor<2,dim,ad_type>
	get_dPsi_vol_dC () const
	{
		return theta_ratio()*this->get_dW_J_dC(Material::Coefficients::kappa, Material::Coefficients::mu)
				- theta_difference()*get_dM_J_dC(Coefficients::kappa,Coefficients::alpha); // Thermal dilatory response M = M(J);
	}

	virtual SymmetricTensor<2,dim,ad_type>
	get_dPsi_vol_dC_EM () const
	{
		return unit_symmetric_tensor<dim,ad_type>();
	}

	inline SymmetricTensor<2,dim,ad_type>
	get_dW_FE_iso_elastic_dC (const double g_0,
			const double N,
			const double c_2) const
			{

		//		const ad_type &lambda=get_lambda();
		//
		//		return	ad_type((0.5*g_0/dim)*((dim*N-lambda*lambda)/(N-lambda*lambda)))*
		//				(unit_symmetric_tensor<dim,ad_type>())+ c_2*this->values_ad.dI5_dC();
		//
		const SymmetricTensor<2,dim,ad_type> &C_inv = this->values_ad.C_inv;
		return (g_0/2)*(unit_symmetric_tensor<dim,ad_type>())+ c_2*this->values_ad.dI5_dC();

			}

	inline SymmetricTensor<2,dim,ad_type>
	get_dW_FE_iso_elastic_dC_EM (const double g_0,
			const double N,
			const double c_2) const
			{

		const SymmetricTensor<4,dim,ad_type> P= get_Dev_P(this->values_ad.F_EM);//get_dC_bar_dC();
		const SymmetricTensor<2,dim,ad_type> dW_FE_dC_EM_bar=get_dW_FE_iso_elastic_dC_EM_bar(g_0,N,c_2);

		return (dW_FE_dC_EM_bar*P);


			}

	inline SymmetricTensor<2,dim,ad_type>
	get_dW_FE_iso_elastic_dC_bar (const double g_0,
			const double N,
			const double c_2) const
			{
		//		const ad_type &lambda_bar=get_lambda_bar();
		//
		//		return	ad_type((0.5*g_0/dim)*((dim*N-lambda_bar*lambda_bar)/(N-lambda_bar*lambda_bar)))*
		//				(unit_symmetric_tensor<dim,ad_type>())+ c_2*this->values_ad.dI5_bar_dC_bar();

		return (g_0 /2.0)*this->values_ad.dI1_bar_dC_bar()+ c_2*this->values_ad.dI5_bar_dC_bar();

			}

	inline SymmetricTensor<2,dim,ad_type>
	get_dW_FE_iso_elastic_dC_EM_bar (const double g_0,
			const double N,
			const double c_2) const
			{
		//		const ad_type &lambda_EM_bar=get_lambda_EM_bar();
		//
		//		return	ad_type((0.5*g_0/dim)*((dim*N-lambda_EM_bar*lambda_EM_bar)/(N-lambda_EM_bar*lambda_EM_bar)))*
		//				(unit_symmetric_tensor<dim,ad_type>()) + c_2*this->values_ad.dI5_EM_bar_dC_EM_bar();

		return g_0 /2.0*this->values_ad.dI1_EM_bar_dC_EM_bar()+ c_2*this->values_ad.dI5_EM_bar_dC_EM_bar();

			}

	inline ad_type
	get_dM_J_dJ (const double kappa, const double alpha) const
	{
		const ad_type &J = this->values_ad.J;
		return dim*kappa*alpha/J;
	}

	SymmetricTensor<2,dim,ad_type>
	get_dM_J_dC (const double kappa,const double alpha) const
	{
		// See Wriggers p46 eqs. 3.123, 3.124; Holzapfel p230
		return get_dM_J_dJ(kappa, alpha)*this->values_ad.dJ_dC();
	}


	inline SymmetricTensor<4, dim, ad_type>
	get_Dev_P (const Tensor<2, dim, ad_type> &F) const
	{
		const ad_type det_F = determinant(F);
		Assert(det_F > ad_type(0.0),
				ExcMessage("Deformation gradient has a negative determinant."));
		const Tensor<2,dim,ad_type> C_ns = transpose(F)*F;
		const SymmetricTensor<2,dim,ad_type> C = symmetrize(C_ns);
		const SymmetricTensor<2,dim,ad_type> C_inv = symmetrize(invert(C_ns));

		// See Wriggers p46 equ 3.125 (but transpose indices)
		SymmetricTensor<4,dim,ad_type> Dev_P = outer_product(C,C_inv);  // Dev_P = C_x_C_inv
		Dev_P /= -dim;                                                  // Dev_P = -[1/dim]C_x_C_inv
		Dev_P += Physics::Elasticity::StandardTensors< dim >::S;        // Dev_P = S - [1/dim]C_x_C_inv
		Dev_P *= ad_type(std::pow(det_F, -2.0/dim));                    // Dev_P = J^{-2/dim} [S - [1/dim]C_x_C_inv]

		return Dev_P;
	}


	// --- Electric contributions ---
	virtual Tensor<1,dim,ad_type>
	get_dPsi_dE () const
	{
		double c_1=Material::Coefficients::c_1;
		double c_2=this->values_ad.c_2;

		return get_dW_iso_elastic_dE(c_1,c_2) ;
	}

	inline Tensor<1,dim,ad_type>
	get_dW_iso_elastic_dE (const double c_1,
			const double c_2) const
			{
		return c_1*this->values_ad.dI4_dE()+c_2*this->values_ad.dI5_dE();
			}

	ad_type
	get_lambda() const
	{
		const ad_type &I1=this->values_ad.I1;
		return std::sqrt(I1/dim);
	}

	ad_type
	get_lambda_bar() const
	{
		const ad_type &I1_bar=this->values_ad.I1_bar;
		return std::sqrt(I1_bar/dim);
	}

	ad_type
	get_lambda_EM_bar() const
	{
		const ad_type &I1_EM_bar=this->values_ad.I1_EM_bar;
		return std::sqrt(I1_EM_bar/dim);
	}

	double
	theta_ratio () const
	{
		return this->values_ad.theta/Material::Coefficients::theta_0;
	}

	double
	theta_difference () const
	{
		return this->values_ad.theta - Material::Coefficients::theta_0;
	}

};


}

template<int dim>
class CoupledProblem
{
	typedef Material::CM_Coupled_NeoHooke_ad<dim> Continuum_Point_Coupled_NeoHooke_ad;
	//	typedef Material::CM_Incompressible_Uncoupled_8Chain_ad<dim> Continuum_Point_8_Chain_uncoupled_ad;

public:
	CoupledProblem ();
	~CoupledProblem ();
	void
	run ();

private:
	void
	make_grid ();
	void
	set_active_fe_indices();
	void
	setup_system ();
	void
	make_constraints (const unsigned int newton_iteration, const unsigned int timestep);
	void
	assemble_system_thermo ();
	void
	solve_thermo (LA::MPI::BlockVector & solution_update);
	void
	assemble_system_mech ();
	void
	solve_mech (LA::MPI::BlockVector & solution_update);
	void
	solve_nonlinear_timestep (const int ts);
	void
	output_results (const unsigned int timestep) const;
	double
	get_norm(const SymmetricTensor<2,dim> X);

	const unsigned int n_blocks;
	const unsigned int first_u_component; // Displacement
	const unsigned int V_component; // Voltage / Potential difference
	const unsigned int J_component; // Temperature
	const unsigned int p_component; // Temperature
	const unsigned int T_component; // Temperature
	const unsigned int n_components;

	enum
	{
		uV_block = 0,
		T_block  = 1
	};

	enum
	{
		u_dof = 0,
		V_dof = 1,
		J_dof = 2,
		p_dof = 3,
		T_dof = 4
	};

	enum
	{
		coupled_material_id=1,
		uncoupled_material_id=2
	};



	static bool cell_is_in_3Field_domain(const typename hp::DoFHandler<dim>::cell_iterator &cell);
	static bool cell_is_in_1Field_domain(const typename hp::DoFHandler<dim>::cell_iterator &cell);

	const FEValuesExtractors::Vector displacement;
	const FEValuesExtractors::Scalar x_displacement;
	const FEValuesExtractors::Scalar y_displacement;
	const FEValuesExtractors::Scalar z_displacement;
	const FEValuesExtractors::Scalar voltage;
	const FEValuesExtractors::Scalar dilatation;
	const FEValuesExtractors::Scalar pressure;
	const FEValuesExtractors::Scalar temperature;


	MPI_Comm           mpi_communicator;
	const unsigned int n_mpi_processes;
	const unsigned int this_mpi_process;
	mutable ConditionalOStream pcout;
	mutable TimerOutput computing_timer;


	Triangulation<dim>    triangulation;
	hp::FECollection<dim> fe_collection;
	hp::DoFHandler<dim>   dof_handler;

	std::vector<IndexSet> all_locally_owned_dofs;
	IndexSet locally_owned_dofs;
	IndexSet locally_relevant_dofs;
	std::vector<IndexSet> locally_owned_partitioning;
	std::vector<IndexSet> locally_relevant_partitioning;

	const unsigned int poly_order;

	hp::QCollection<dim> q_collection;


	FESystem<dim> fe_cell_3Field;
	FESystem<dim> fe_cell_1Field;


	QGauss<dim> qf_cell_3Field;
	QGauss<dim> qf_cell_1Field;

	ConstraintMatrix hanging_node_constraints;
	ConstraintMatrix dirichlet_constraints;
	ConstraintMatrix periodicity_constraints;
	ConstraintMatrix all_constraints;

	LA::MPI::BlockSparseMatrix system_matrix;
	LA::MPI::BlockVector       system_rhs;
	//	LA::MPI::BlockVector       solution;
	LA::MPI::BlockVector locally_relevant_solution;
	LA::MPI::BlockVector locally_relevant_solution_update;
	LA::MPI::BlockVector completely_distributed_solution_update;


};

template<int dim>
CoupledProblem<dim>::CoupledProblem ()
:
n_blocks (2),
first_u_component (0), // Displacement
V_component (first_u_component + dim), // Voltage / Potential difference
J_component (V_component+1),
p_component (J_component+1),
T_component (p_component+1), // Temperature
n_components (T_component+1),

displacement(first_u_component),
x_displacement(first_u_component),
y_displacement(first_u_component+1),
z_displacement(dim==3 ? first_u_component+2 : first_u_component+1),
voltage(V_component),
dilatation(J_component),
pressure(p_component),
temperature(T_component),

mpi_communicator (MPI_COMM_WORLD),
n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
pcout(std::cout, this_mpi_process == 0),
computing_timer(mpi_communicator,
		pcout,
		TimerOutput::summary,
		TimerOutput::wall_times),
		triangulation(Triangulation<dim>::maximum_smoothing),

		dof_handler(triangulation),

		poly_order (Parameters::poly_order),
		fe_cell_3Field(FE_Q<dim> (poly_order), dim,
				FE_Q<dim> (poly_order), 1, // Voltage
				FE_DGPMonomial<dim>(poly_order - 1), 1,  // Dilatation
				FE_DGPMonomial<dim>(poly_order - 1), 1,  // Pressure
				FE_Q<dim> (poly_order), 1), // Temperature
				fe_cell_1Field(FE_Q<dim> (poly_order), dim,
						FE_Q<dim> (poly_order), 1, // Voltage
						FE_Nothing<dim>(), 1,  // Dilatation
						FE_Nothing<dim>(), 1, // Pressure
						FE_Q<dim> (poly_order), 1), // Temperature

						qf_cell_3Field(poly_order+1),
						qf_cell_1Field(poly_order+1)
						{
	fe_collection.push_back(fe_cell_3Field);
	fe_collection.push_back(fe_cell_1Field);
	q_collection.push_back(qf_cell_3Field);
	q_collection.push_back(qf_cell_1Field);
						}



template<int dim>
CoupledProblem<dim>::~CoupledProblem ()
{
	dof_handler.clear();
}


template<int dim>
void
CoupledProblem<dim>::make_grid () //Generate thick walled cylinder
{
	TimerOutput::Scope timer_scope (computing_timer, "Make grid");

	GridIn<dim> grid_in;
	grid_in.attach_triangulation (triangulation);
	std::ifstream input_file(Parameters::mesh_file.c_str());

	grid_in.read_abaqus (input_file);

	//	static CylindricalManifold<dim> manifold_cylinder_X (0,1); // Manifold id 1
	//	static CylindricalManifold<dim> manifold_cylinder_Z (2,1); // Manifold id 2

	// Set boundary and manifold ID's for this tricky geometry.
	// Note: X-aligned cylinder manifold > Z-aligned cylinder manifold > Straight/Planar manifold

	// Set straight/planar boundary and manifold IDs
	typename Triangulation<dim>::active_cell_iterator
	cell = triangulation.begin_active(),
	endc = triangulation.end();
	for (; cell != endc; ++cell)
	{
		for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
		{
			if (cell->face(f)->at_boundary())
			{
				const Point<dim> face_center = cell->face(f)->center();
				if (face_center[2] == 120)
				{ // Faces at cylinder bottom
					cell->face(f)->set_boundary_id(Parameters::boundary_id_bottom);
				}
				else if (face_center[2] == 80)
				{ // Faces at cylinder top
					cell->face(f)->set_boundary_id(Parameters::boundary_id_top);
				}
				else if (face_center[2] == 0)
				{ // Faces at cylinder top
					cell->face(f)->set_boundary_id(Parameters::boundary_id_frame);
				}
				else if (face_center[0] == 0)
				{ // Faces at cylinder top
					cell->face(f)->set_boundary_id(Parameters::boundary_id_cut_left);
				}
				else if (face_center[1] == 0)
				{ // Faces at cylinder top
					cell->face(f)->set_boundary_id(Parameters::boundary_id_cut_bottom);
				}
				else if (face_center[0] >= 539)
				{ // Faces at cylinder top
					cell->face(f)->set_boundary_id(Parameters::boundary_id_cut_outlet);
				}
				else
				{
					// Catch all, mainly for faces at external surface of the cylinder
					// Some of the erroneously set boundary ID's will be corrected later.
					cell->face(f)->set_boundary_id(Parameters::boundary_id_outer_radius);
				}
			}
		}
	}

	// Set Z-aligned cylinder boundary and manifold IDs
	cell = triangulation.begin_active();
	for (; cell != endc; ++cell)
	{
		for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
		{
			if (cell->face(f)->at_boundary())
			{
				const Point<dim> face_center = cell->face(f)->center();
				if (face_center[0] > 0 && face_center[1] > 0 && face_center[2] > 0)
					for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
					{
						const Point<dim> &pt = cell->face(f)->vertex(v);
						if (abs(sqrt(pt[0]*pt[0] + pt[1]*pt[1]) - 460) < 1e-3 &&
								face_center[2] < 80)
						{
							// Faces at internal surface of the cylinder
							cell->face(f)->set_boundary_id(Parameters::boundary_id_inner_radius);
							cell->face(f)->set_all_manifold_ids(2);
							break;
						}
						else if (abs(sqrt(pt[0]*pt[0] + pt[1]*pt[1]) - 490) < 1e-6)
						{
							// Faces at external surface of the cylinder
							cell->face(f)->set_boundary_id(Parameters::boundary_id_outer_radius);
							cell->face(f)->set_all_manifold_ids(2);
							break;
						}
					}

			}
		}
	}

	// Set X-aligned cylinder boundary and manifold IDs
	cell = triangulation.begin_active();
	for (; cell != endc; ++cell)
	{
		for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
		{
			if (cell->face(f)->at_boundary())
			{
				const Point<dim> face_center = cell->face(f)->center();
				if (face_center[0] > 0 && face_center[1] > 0 && face_center[2] > 0) // Face not on cartesian planes
					if (face_center[0] > 460 && face_center[0] < 540 &&
							sqrt(face_center[1]*face_center[1] + face_center[2]*face_center[2]) <= 70)
					{
						for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
						{
							const Point<dim> &pt = cell->face(f)->vertex(v);
							if (abs(sqrt(pt[1]*pt[1] + pt[2]*pt[2]) - 50) < 1e-6)
							{
								// Faces at internal surface of the cylinder
								cell->face(f)->set_boundary_id(Parameters::boundary_id_outlet_inner_radius);
								cell->face(f)->set_all_manifold_ids(1);
								break;
							}
							else if (abs(sqrt(pt[1]*pt[1] + pt[2]*pt[2]) - 70) < 1e-6 && face_center[0] > 490)
							{
								// Faces at external surface of the cylinder
								cell->face(f)->set_boundary_id(Parameters::boundary_id_outlet_outer_radius);
								cell->face(f)->set_all_manifold_ids(1);
								break;
							}
						}
					}
			}
		}
	}

	//	triangulation.set_manifold (1, manifold_cylinder_X);
	//	triangulation.set_manifold (2, manifold_cylinder_Z);

	triangulation.refine_global (Parameters::n_global_refinements);
}

template <int dim>
void
CoupledProblem<dim>::set_active_fe_indices()
{
	for (typename hp::DoFHandler<dim>::active_cell_iterator cell =
			dof_handler.begin_active();
			cell != dof_handler.end();
			++cell)
	{
		if (Parameters::use_3_Field)
			cell->set_active_fe_index(0);
		else if (Parameters::use_3_Field==false)
			cell->set_active_fe_index(1);
		else
			Assert(false, ExcNotImplemented());
	}
}

template<int dim>
void
CoupledProblem<dim>::setup_system ()
{
	TimerOutput::Scope timer_scope (computing_timer, "System setup");
	pcout << "Setting up the thermo-electro-mechanical system..." << std::endl;
	set_active_fe_indices();
	dof_handler.distribute_dofs(fe_collection);

	std::vector<types::global_dof_index>  block_component(n_components, uV_block); // Displacement
	block_component[V_component] = uV_block; // Voltage
	block_component[J_component] = uV_block; // dilatation
	block_component[p_component] = uV_block; // pressure
	block_component[T_component] = T_block; // Temperature

	DoFRenumbering::Cuthill_McKee(dof_handler);
	DoFRenumbering::component_wise(dof_handler, block_component);

	std::vector<types::global_dof_index> dofs_per_block(n_blocks);
	DoFTools::count_dofs_per_block(dof_handler, dofs_per_block, block_component);
	const types::global_dof_index &n_u_V = dofs_per_block[0];
	const types::global_dof_index &n_th = dofs_per_block[1];

	all_locally_owned_dofs = DoFTools::locally_owned_dofs_per_subdomain (dof_handler);
	std::vector<IndexSet> all_locally_relevant_dofs	= DoFTools::locally_relevant_dofs_per_subdomain (dof_handler);


	pcout
	<< "Number of active cells: "
	<< triangulation.n_active_cells()
	<< std::endl
	<< "Total number of cells: "
	<< triangulation.n_cells()
	<< std::endl
	<< "Number of degrees of freedom: "
	<< dof_handler.n_dofs()
	<< " (" << n_u_V << '+' << n_th << ')'
	<< std::endl;

	locally_owned_dofs.clear();
	locally_owned_partitioning.clear();
	Assert(all_locally_owned_dofs.size() > this_mpi_process, ExcInternalError());
	locally_owned_dofs = all_locally_owned_dofs[this_mpi_process];

	locally_relevant_dofs.clear();
	locally_relevant_partitioning.clear();
	Assert(all_locally_relevant_dofs.size() > this_mpi_process, ExcInternalError());
	locally_relevant_dofs = all_locally_relevant_dofs[this_mpi_process];

	locally_owned_partitioning.reserve(n_blocks);
	locally_relevant_partitioning.reserve(n_blocks);
	for (unsigned int b=0; b<n_blocks; ++b)
	{
		const types::global_dof_index idx_begin
		= std::accumulate(dofs_per_block.begin(),
				std::next(dofs_per_block.begin(),b), 0);
		const types::global_dof_index idx_end
		= std::accumulate(dofs_per_block.begin(),
				std::next(dofs_per_block.begin(),b+1), 0);
		locally_owned_partitioning.push_back(locally_owned_dofs.get_view(idx_begin, idx_end));
		locally_relevant_partitioning.push_back(locally_relevant_dofs.get_view(idx_begin, idx_end));
	}


	hanging_node_constraints.clear();
	hanging_node_constraints.reinit (locally_relevant_dofs);
	DoFTools::make_hanging_node_constraints(dof_handler,
			hanging_node_constraints);
	hanging_node_constraints.close();

	Table<2, DoFTools::Coupling> coupling(n_components, n_components);
	for (unsigned int ii = 0; ii < n_components; ++ii)
		for (unsigned int jj = 0; jj < n_components; ++jj)
			if (((ii < p_component) && (jj == J_component))
					|| ((ii == J_component) && (jj < p_component))
					|| ((ii == p_component) && (jj == p_component)))
				coupling[ii][jj] = DoFTools::none;
			else if (((ii < T_component) && (jj == T_component))
					|| ((ii == T_component) && (jj < T_component)))
				coupling[ii][jj] = DoFTools::none;
			else
				coupling[ii][jj] = DoFTools::always;

	TrilinosWrappers::BlockSparsityPattern sp (locally_owned_partitioning,
			locally_owned_partitioning,
			locally_relevant_partitioning,
			mpi_communicator);
	DoFTools::make_sparsity_pattern (dof_handler, sp,
			all_constraints, false,
			this_mpi_process);

	sp.compress();
	system_matrix.reinit (sp);


	system_rhs.reinit (locally_owned_partitioning,
			locally_relevant_partitioning,
			mpi_communicator,
			true);
	//	solution.reinit (locally_owned_partitioning,
	//			locally_relevant_partitioning,
	//			mpi_communicator,
	//			true);
	locally_relevant_solution.reinit (locally_relevant_partitioning,
			mpi_communicator);
	locally_relevant_solution_update.reinit (locally_relevant_partitioning,
			mpi_communicator);
	completely_distributed_solution_update.reinit(locally_owned_partitioning,
			mpi_communicator);


}


template<int dim>
void
CoupledProblem<dim>::make_constraints (const unsigned int newton_iteration, const unsigned int timestep)
{
	TimerOutput::Scope timer_scope (computing_timer, "Make constraints");

	if (newton_iteration >= 2)
	{
		pcout << std::string(14, ' ') << std::flush;
		return;
	}
	if (newton_iteration == 0)
	{
		dirichlet_constraints.clear();
		dirichlet_constraints.reinit (locally_relevant_dofs);

		pcout << "  CST T" << std::flush;

		const double temperature_difference_per_ts = Parameters::Temperature_Difference/static_cast<double>(Parameters::n_timesteps);
		if (timestep==1)
		{
			// Prescribed temperature at inner radius
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_inner_radius,
					ConstantFunction<dim>(293+Parameters::Temperature_Difference,n_components),
					dirichlet_constraints,
					fe_collection.component_mask(temperature));

			// Prescribed temperature at inner radius
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_outlet_inner_radius,
					ConstantFunction<dim>(293+Parameters::Temperature_Difference,n_components),
					dirichlet_constraints,
					fe_collection.component_mask(temperature));

			// Prescribed temperature at top
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_top,
					ConstantFunction<dim>(293+Parameters::Temperature_Difference,n_components),
					dirichlet_constraints,
					fe_collection.component_mask(temperature));

			// Prescribed temperature at bottom
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_bottom,
					ConstantFunction<dim>(293,n_components),
					dirichlet_constraints,
					fe_collection.component_mask(temperature));

			// Prescribed temperature at outer radius
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_outer_radius,
					ConstantFunction<dim>(293,n_components),
					dirichlet_constraints,
					fe_collection.component_mask(temperature));

			// Prescribed temperature at outer radius
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_outlet_outer_radius,
					ConstantFunction<dim>(293,n_components),
					dirichlet_constraints,
					fe_collection.component_mask(temperature));
		}
		else
		{
			// Prescribed temperature at top
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_top,
					ZeroFunction<dim>(n_components),
					dirichlet_constraints,
					fe_collection.component_mask(temperature));

			// Prescribed temperature at bottom
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_bottom,
					ZeroFunction<dim>(n_components),
					dirichlet_constraints,
					fe_collection.component_mask(temperature));

			// Prescribed temperature at inner radius
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_inner_radius,
					ZeroFunction<dim>(n_components),
					dirichlet_constraints,
					fe_collection.component_mask(temperature));

			// Prescribed temperature at outer radius
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_outer_radius,
					ZeroFunction<dim>(n_components),
					dirichlet_constraints,
					fe_collection.component_mask(temperature));

			// Prescribed temperature at outer radius
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_outlet_outer_radius,
					ZeroFunction<dim>(n_components),
					dirichlet_constraints,
					fe_collection.component_mask(temperature));

			// Prescribed temperature at inner radius
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_outlet_inner_radius,
					ZeroFunction<dim>(n_components),
					dirichlet_constraints,
					fe_collection.component_mask(temperature));
		}


		pcout << "  CST M" << std::flush;
		{
			const double potential_difference_per_ts = Parameters::potential_difference/(static_cast<double>(Parameters::n_timesteps));

			// Y-Cut Surface
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_cut_bottom,
					ZeroFunction<dim>(n_components),
					dirichlet_constraints,
					fe_collection.component_mask(y_displacement));

			// X-Cut Surface
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_cut_left,
					ZeroFunction<dim>(n_components),
					dirichlet_constraints,
					fe_collection.component_mask(x_displacement));

			// Frame Cut Surface
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_frame,
					ZeroFunction<dim>(n_components),
					dirichlet_constraints,
					fe_collection.component_mask(z_displacement));

			// Frame Cut Surface
			/*	VectorTools::interpolate_boundary_values(dof_handler,
						Parameters::boundary_id_outer_radius,
						ZeroFunction<dim>(n_components),
						dirichlet_constraints,
						fe_cell.component_mask(x_displacement)|fe_cell.component_mask(y_displacement));
			 */

			// Fixed outlet
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_cut_outlet,
					ZeroFunction<dim>(n_components),
					dirichlet_constraints,
					fe_collection.component_mask(x_displacement) |
					fe_collection.component_mask(y_displacement) |
					fe_collection.component_mask(z_displacement));

			// Prescribed voltage at lower surface
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_top,
					//ZeroFunction<dim>(n_components),
					ConstantFunction<dim>(+potential_difference_per_ts/2,n_components),
					dirichlet_constraints,
					fe_collection.component_mask(voltage));

			// Prescribed voltage at upper surface
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_bottom,
					//ZeroFunction<dim>(n_components),
					ConstantFunction<dim>(-potential_difference_per_ts/2,n_components),
					dirichlet_constraints,
					fe_collection.component_mask(voltage));
		}

		dirichlet_constraints.close();
	}
	else
	{
		pcout << "   CST ZERO   " << std::flush;
		// Remove inhomogenaities
		for (types::global_dof_index d=0; d<dof_handler.n_dofs(); ++d)
			if (dirichlet_constraints.can_store_line(d) == true)
				if (dirichlet_constraints.is_constrained(d) == true)
					if (dirichlet_constraints.is_inhomogeneously_constrained(d) == true)
						dirichlet_constraints.set_inhomogeneity(d,0.0);
	}

	// Combine constraint matrices
	all_constraints.clear();
	all_constraints.reinit (locally_relevant_dofs);
	all_constraints.merge(hanging_node_constraints);
	all_constraints.merge(dirichlet_constraints, ConstraintMatrix::left_object_wins);
	all_constraints.close();
}

template<int dim>
void
CoupledProblem<dim>::assemble_system_thermo()
{
	typedef Sacado::Fad::DFad<double> ad_type;

	TimerOutput::Scope timer_scope (computing_timer, "Assembly: Thermal");
	pcout << "  ASM T" << std::flush;

	hp::FEValues<dim> hp_fe_values(fe_collection,
			q_collection,
			update_values | update_quadrature_points |
			update_JxW_values | update_gradients);

	FullMatrix<double> cell_matrix;
	Vector<double> cell_rhs;

	std::vector<types::global_dof_index> local_dof_indices;

	typename hp::DoFHandler<dim>::active_cell_iterator cell = dof_handler
			.begin_active(),
			endc = dof_handler.end();
	for (; cell != endc; ++cell)
	{
		if (cell->is_locally_owned() == false) continue;


		hp_fe_values.reinit(cell);
		const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
		const unsigned int n_q_points = fe_values.n_quadrature_points;

		cell_matrix.reinit(cell->get_fe().dofs_per_cell,
				cell->get_fe().dofs_per_cell);
		cell_rhs.reinit(cell->get_fe().dofs_per_cell);
		cell_matrix = 0;
		cell_rhs = 0;

		local_dof_indices.resize(cell->get_fe().dofs_per_cell);
		cell->get_dof_indices(local_dof_indices);

		// Values at integration points
		std::vector< Tensor<2,dim> > Grad_u(n_q_points); // Material gradient of displacement
		std::vector< Tensor<1,dim> > Grad_V(n_q_points); // Material gradient of voltage
		std::vector< Tensor<1,dim> > Grad_T(n_q_points); // Material gradient of temperature
		std::vector<double> theta(n_q_points); // Temperature


		for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
		{
			unsigned int mat_id;
			mat_id = cell->material_id();
			const double &JxW = fe_values.JxW(q_point);

			fe_values[displacement].get_function_gradients(locally_relevant_solution, Grad_u);
			fe_values[voltage].get_function_gradients(locally_relevant_solution, Grad_V);
			fe_values[temperature].get_function_gradients(locally_relevant_solution, Grad_T);
			fe_values[temperature].get_function_values(locally_relevant_solution, theta);

			//Deformation gradient at quadrature point
			const Tensor<2,dim> F_q_point = (static_cast< Tensor<2,dim> >(unit_symmetric_tensor<dim>()) + Grad_u[q_point]);

			const Tensor<2,dim> F_inv = invert(F_q_point);
			Tensor<2,dim> K;



			K=Material::Coefficients::k*symmetrize(F_inv*transpose(F_inv));


			const Tensor<1,dim> Q = K*(-Grad_T[q_point]);

			for (unsigned int i = 0; i < cell->get_fe().dofs_per_cell; ++i)
			{
				const unsigned int i_group     = fe_cell_3Field.system_to_base_index(i).first.first;

				const Tensor<1,dim> &Grad_Nx_i_T = fe_values[temperature].gradient(i, q_point);


				for (unsigned int j = 0; j < cell->get_fe().dofs_per_cell; ++j)
				{

					const unsigned int j_group     = fe_cell_3Field.system_to_base_index(j).first.first;

					const Tensor<1,dim> &Grad_Nx_j_T = fe_values[temperature].gradient(j, q_point);


					if ((i_group == T_dof) && (j_group == T_dof))
					{
						// T-T terms
						cell_matrix(i, j) -= (Grad_Nx_i_T*K*Grad_Nx_j_T) * JxW;
					}
				}

				// RHS = -Residual
				if (i_group == T_dof)
				{
					// T terms
					cell_rhs(i) -= (Grad_Nx_i_T*Q) * JxW;

				}
			}
		}


		all_constraints.distribute_local_to_global(cell_matrix, cell_rhs,
				local_dof_indices,
				system_matrix, system_rhs);
	}
	//
	system_matrix.compress (VectorOperation::add);
	system_rhs.compress (VectorOperation::add);
}

template<int dim>
void
CoupledProblem<dim>::assemble_system_mech ()

{
	typedef Sacado::Fad::DFad<double> ad_type;

	TimerOutput::Scope timer_scope (computing_timer, "Assembly: Mechanical");
	pcout << "  ASM M" << std::flush;

	hp::FEValues<dim> hp_fe_values(fe_collection,
			q_collection,
			update_values | update_quadrature_points |
			update_JxW_values | update_gradients);


	FullMatrix<double> cell_matrix;
	Vector<double> cell_rhs;

	std::vector<types::global_dof_index> local_dof_indices;

	typename hp::DoFHandler<dim>::active_cell_iterator cell = dof_handler
			.begin_active(),
			endc = dof_handler.end();
	for (; cell != endc; ++cell)
	{
		if (cell->is_locally_owned() == false) continue;

		hp_fe_values.reinit(cell);
		const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
		const unsigned int n_q_points = fe_values.n_quadrature_points;
		cell_matrix.reinit(cell->get_fe().dofs_per_cell,
				cell->get_fe().dofs_per_cell);
		cell_rhs.reinit(cell->get_fe().dofs_per_cell);

		cell_matrix = 0;
		cell_rhs = 0;

		local_dof_indices.resize(cell->get_fe().dofs_per_cell);
		cell->get_dof_indices(local_dof_indices);

		{
			const unsigned int n_independent_variables = local_dof_indices.size();
			std::vector<double> local_dof_values(n_independent_variables);
			cell->get_dof_values(locally_relevant_solution, local_dof_values.begin(), local_dof_values.end());

			// We now retreive a set of degree-of-freedom values that
			// have the operations that are performed with them tracked.
			std::vector<ad_type> local_dof_values_ad (n_independent_variables);
			for (unsigned int i=0; i<n_independent_variables; ++i)
				local_dof_values_ad[i] = ad_type(n_independent_variables, i, local_dof_values[i]);

			// Compute all values, gradients etc. based on sensitive
			// AD degree-of-freedom values.
			std::vector< Tensor<2,dim,ad_type> > Grad_u_ad (n_q_points, Tensor<2,dim,ad_type>());
			fe_values[displacement].get_function_gradients_from_local_dof_values(local_dof_values_ad, Grad_u_ad);
			std::vector< Tensor<1,dim,ad_type> > Grad_V_ad (n_q_points, Tensor<1,dim,ad_type>());
			fe_values[voltage].get_function_gradients_from_local_dof_values(local_dof_values_ad, Grad_V_ad);
			std::vector< ad_type> J_tilde (n_q_points, 1.0);
			fe_values[dilatation].get_function_values_from_local_dof_values(local_dof_values_ad, J_tilde);
			std::vector< ad_type> p (n_q_points, 1.0);
			fe_values[pressure].get_function_values_from_local_dof_values(local_dof_values_ad, p);
			std::vector<double> theta(n_q_points); // Temperature
			fe_values[temperature].get_function_values(locally_relevant_solution, theta);
			std::vector< Tensor<1,dim> > Grad_T(n_q_points);
			fe_values[temperature].get_function_gradients(locally_relevant_solution, Grad_T);

			std::vector<ad_type> cell_residual_ad(cell->get_fe().dofs_per_cell, ad_type(0.0));


			for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
			{

				unsigned int mat_id;
				mat_id = cell->material_id();

				const Tensor<2,dim,ad_type> F_ad = Physics::Elasticity::Kinematics::F(Grad_u_ad[q_point]);

				SymmetricTensor<2,dim,ad_type> S_ad;
				Tensor<1,dim,ad_type> D_ad;
				ad_type dPsi_dJ_tilde;
				ad_type dPsi_dp;

				const double JxW = fe_values.JxW(q_point);

				const double alpha = Material::Coefficients::alpha;
				const double c_2 = Material::Coefficients::c_2;

				if (Parameters::use_3_Field)
				{
					if(mat_id==coupled_material_id)
					{

						const Continuum_Point_Coupled_NeoHooke_ad cp_ad (F_ad, -Grad_V_ad[q_point],Grad_T[q_point], theta[q_point],J_tilde[q_point],p[q_point],alpha,c_2);
						S_ad=cp_ad.get_S_3Field();
						D_ad=cp_ad.get_D();
						dPsi_dJ_tilde= cp_ad.get_dPsi_dJ_tilde();
						dPsi_dp= cp_ad.get_dPsi_dp();
					}
					else
					{
						const Continuum_Point_Coupled_NeoHooke_ad cp_ad (F_ad, -Grad_V_ad[q_point],Grad_T[q_point], theta[q_point],J_tilde[q_point],p[q_point],alpha,0.0);
						S_ad=cp_ad.get_S_3Field();
						D_ad=cp_ad.get_D();
						dPsi_dJ_tilde= cp_ad.get_dPsi_dJ_tilde();
						dPsi_dp= cp_ad.get_dPsi_dp();
					}

					for(unsigned int i = 0; i < cell->get_fe().dofs_per_cell; ++i)
					{
						const unsigned int i_group     = fe_cell_3Field.system_to_base_index(i).first.first;

						if (i_group == u_dof)
						{
							const SymmetricTensor<2,dim,ad_type> dE_ad_I = symmetrize(transpose(F_ad)*fe_values[displacement].gradient(i, q_point));
							cell_residual_ad[i] += (dE_ad_I*S_ad) * JxW; // residual

						}
						else if (i_group == V_dof)
						{
							const Tensor<1,dim> &Grad_Nx_i_V      = fe_values[voltage].gradient(i, q_point);
							cell_residual_ad[i] -= (Grad_Nx_i_V*D_ad) * JxW;
						}
						else if (i_group == J_dof)
						{
							const double &NJ_i_value = fe_values[dilatation].value(i,q_point);
							cell_residual_ad[i] -= NJ_i_value * dPsi_dJ_tilde  * JxW;
						}
						else if (i_group == p_dof)
						{
							const double &Np_i_value = fe_values[pressure].value(i,q_point);
							cell_residual_ad[i] -= Np_i_value * dPsi_dp  * JxW;
						}
					}
				}
				else
				{
					if(mat_id==coupled_material_id)
					{
						const Continuum_Point_Coupled_NeoHooke_ad cp_ad (F_ad, -Grad_V_ad[q_point],Grad_T[q_point], theta[q_point],J_tilde[q_point],p[q_point],alpha,c_2);
						S_ad=cp_ad.get_S_1Field();
						D_ad=cp_ad.get_D();
					}
					else
					{
						const Continuum_Point_Coupled_NeoHooke_ad cp_ad (F_ad, -Grad_V_ad[q_point],Grad_T[q_point], theta[q_point],J_tilde[q_point],p[q_point],alpha,0.0);
						S_ad=cp_ad.get_S_1Field();
						D_ad=cp_ad.get_D();
					}

					for(unsigned int i = 0; i < cell->get_fe().dofs_per_cell; ++i)
					{
						const unsigned int i_group     = fe_cell_1Field.system_to_base_index(i).first.first;

						if (i_group == u_dof)
						{
							const SymmetricTensor<2,dim,ad_type> dE_ad_I = symmetrize(transpose(F_ad)*fe_values[displacement].gradient(i, q_point));
							cell_residual_ad[i] += (dE_ad_I*S_ad) * JxW; // residual

						}
						else if (i_group == V_dof)
						{
							const Tensor<1,dim> &Grad_Nx_i_V      = fe_values[voltage].gradient(i, q_point);
							cell_residual_ad[i] -= (Grad_Nx_i_V*D_ad) * JxW;
						}
					}
				}




			}

			if (Parameters::use_3_Field)
			{
				for (unsigned int I=0; I<n_independent_variables; ++I)
				{
					const unsigned int i_group     = fe_cell_3Field.system_to_base_index(I).first.first;

					const ad_type &res_I = cell_residual_ad[I];
					cell_rhs(I) = -res_I.val();
					unsigned int mat_id;
					mat_id = cell->material_id();

					for (unsigned int J=0; J<n_independent_variables; ++J)
					{
						const double lin_IJ=res_I.dx(J);
						const unsigned int j_group  = fe_cell_3Field.system_to_base_index(J).first.first;
						cell_matrix(I,J) += lin_IJ; // Tangent Matrix

					}

				}
			}
			else
			{
				for (unsigned int I=0; I<n_independent_variables; ++I)
				{
					const unsigned int i_group     = fe_cell_1Field.system_to_base_index(I).first.first;

					const ad_type &res_I = cell_residual_ad[I];
					cell_rhs(I) = -res_I.val();
					unsigned int mat_id;
					mat_id = cell->material_id();


					for (unsigned int J=0; J<n_independent_variables; ++J)
					{
						const double lin_IJ=res_I.dx(J);
						const unsigned int j_group  = fe_cell_3Field.system_to_base_index(J).first.first;

						if (i_group == u_dof||i_group == V_dof)
						{
							cell_matrix(I,J) += lin_IJ; // Tangent Matrix
						}
					}

				}
			}

		}


		all_constraints.distribute_local_to_global(cell_matrix, cell_rhs,
				local_dof_indices,
				system_matrix, system_rhs);
		//		throw;

	}

	//	throw;
	//
	system_matrix.compress (VectorOperation::add);
	system_rhs.compress (VectorOperation::add);
}

template<int dim>
void
CoupledProblem<dim>::solve_thermo (LA::MPI::BlockVector &locally_relevant_solution_update)
{
	TimerOutput::Scope timer_scope (computing_timer, "Solve: Thermal");
	pcout << "  SLV T" << std::flush;

	//      const std::string solver_type = "Iterative";
	const std::string solver_type = "Direct";

	LA::MPI::BlockVector
	completely_distributed_solution_update (locally_owned_partitioning,
			mpi_communicator);

	{ // Direct solver
#ifdef USE_TRILINOS_LA
		SolverControl solver_control(1, 1e-12);
		TrilinosWrappers::SolverDirect solver (solver_control);


		solver.solve(system_matrix.block(T_block, T_block),
				completely_distributed_solution_update.block(T_block),
				system_rhs.block(T_block));
#else
		AssertThrow(false, ExcNotImplemented());
#endif
	}

	all_constraints.distribute(completely_distributed_solution_update);
	locally_relevant_solution_update.block(T_block) = completely_distributed_solution_update.block(T_block);
}


template<int dim>
void
CoupledProblem<dim>::solve_mech (LA::MPI::BlockVector &locally_relevant_solution_update)
{
	TimerOutput::Scope timer_scope (computing_timer, "Solve: Mechanical");
	pcout << "  SLV M" << std::flush;

	//	LA::MPI::BlockVector
	//	completely_distributed_solution_update (locally_owned_partitioning,
	//			mpi_communicator);

	{ // Direct solver
#ifdef USE_TRILINOS_LA
		SolverControl solver_control(1, 1e-12);
		TrilinosWrappers::SolverDirect solver (solver_control);

		solver.solve(system_matrix.block(uV_block, uV_block),
				completely_distributed_solution_update.block(uV_block),
				system_rhs.block(uV_block));
#else
		AssertThrow(false, ExcNotImplemented());
#endif
	}

	all_constraints.distribute(completely_distributed_solution_update);
	locally_relevant_solution_update.block(uV_block) = completely_distributed_solution_update.block(uV_block);

}

template<int dim>
void
CoupledProblem<dim>::output_results (const unsigned int timestep) const
{
	typedef Sacado::Fad::DFad<double> ad_type;
	TimerOutput::Scope timer_scope (computing_timer, "Post-processing");

	unsigned int scalar_components;
	unsigned int vector_components;
	unsigned int tensor_components;

	switch(dim)
	{
	case 2:
	{
		scalar_components = 1;
		vector_components = 2;
		tensor_components = 4;
	}
	break;

	case 3:
	{
		scalar_components = 1;
		vector_components = 3;
		tensor_components = 9;
	}
	break;
	default:
	{
		AssertThrow(false, ExcMessage("Magnetoelasticity::Output_Results -> vector_components -> dimension missmatch!"));
	}
	break;
	}

	std::vector<Vector<double> > deformation_gradient (tensor_components, Vector<double> (triangulation.n_active_cells()));
	std::vector<Vector<double> > Piola_Kirchhoff (tensor_components, Vector<double> (triangulation.n_active_cells()));

	std::vector<Vector<double> > electric_field (vector_components, Vector<double> (triangulation.n_active_cells()));
	std::vector<Vector<double> > electric_displacement (vector_components, Vector<double> (triangulation.n_active_cells()));

	//	FEValues<dim> fe_values(fe_cell,
	//			qf_cell,
	//			update_values |
	//			update_gradients |
	//			update_quadrature_points |
	//			update_JxW_values);

	hp::FEValues<dim> hp_fe_values(fe_collection,
			q_collection,
			update_values | update_quadrature_points |
			update_JxW_values | update_gradients);

	std::vector<types::global_dof_index> local_dof_indices;

	typename hp::DoFHandler<dim>::active_cell_iterator cell = dof_handler
			.begin_active(),
			endc = dof_handler.end();
	for (unsigned int i = 0; cell != endc; ++cell, ++i)
	{
		if (cell->is_locally_owned() == false) continue;

		hp_fe_values.reinit(cell);
		const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
		const unsigned int n_q_points = fe_values.n_quadrature_points;
		local_dof_indices.resize(cell->get_fe().dofs_per_cell);
		cell->get_dof_indices(local_dof_indices);

		{
			const unsigned int n_independent_variables = local_dof_indices.size();
			std::vector<double> local_dof_values(n_independent_variables);
			cell->get_dof_values(locally_relevant_solution, local_dof_values.begin(), local_dof_values.end());

			// We now retreive a set of degree-of-freedom values that
			// have the operations that are performed with them tracked.
			std::vector<ad_type> local_dof_values_ad (n_independent_variables);
			for (unsigned int i=0; i<n_independent_variables; ++i)
				local_dof_values_ad[i] = ad_type(n_independent_variables, i, local_dof_values[i]);

			// Compute all values, gradients etc. based on sensitive
			// AD degree-of-freedom values.
			std::vector< Tensor<2,dim,ad_type> > Grad_u_ad (n_q_points, Tensor<2,dim,ad_type>());
			fe_values[displacement].get_function_gradients_from_local_dof_values(local_dof_values_ad, Grad_u_ad);
			std::vector< Tensor<1,dim,ad_type> > Grad_V_ad (n_q_points, Tensor<1,dim,ad_type>());
			fe_values[voltage].get_function_gradients_from_local_dof_values(local_dof_values_ad, Grad_V_ad);
			std::vector< ad_type> J_tilde (n_q_points, 1.0);
			fe_values[dilatation].get_function_values_from_local_dof_values(local_dof_values_ad, J_tilde);
			std::vector< ad_type> p (n_q_points, 1.0);
			fe_values[pressure].get_function_values_from_local_dof_values(local_dof_values_ad, p);
			std::vector<double> theta(n_q_points); // Temperature
			fe_values[temperature].get_function_values(locally_relevant_solution, theta);
			std::vector< Tensor<1,dim> > Grad_T(n_q_points);
			fe_values[temperature].get_function_gradients(locally_relevant_solution, Grad_T);



			for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
			{

				unsigned int mat_id;
				mat_id = cell->material_id();

				const Tensor<2,dim,ad_type> F_ad = Physics::Elasticity::Kinematics::F(Grad_u_ad[q_point]);
				SymmetricTensor<2,dim,ad_type> S_ad;
				Tensor<1,dim,ad_type> D_ad;
				ad_type dPsi_dJ_tilde;
				ad_type dPsi_dp;


				const double alpha = Material::Coefficients::alpha;
				const double c_2 = Material::Coefficients::c_2;

				if(mat_id==coupled_material_id)
				{

					const Continuum_Point_Coupled_NeoHooke_ad cp_ad (F_ad, -Grad_V_ad[q_point],Grad_T[q_point], theta[q_point],J_tilde[q_point],p[q_point],alpha,c_2);
					S_ad=cp_ad.get_S_3Field();
					D_ad=cp_ad.get_D();
					dPsi_dJ_tilde= cp_ad.get_dPsi_dJ_tilde();
					dPsi_dp= cp_ad.get_dPsi_dp();
				}
				else
				{
					const Continuum_Point_Coupled_NeoHooke_ad cp_ad (F_ad, -Grad_V_ad[q_point],Grad_T[q_point], theta[q_point],J_tilde[q_point],p[q_point],alpha,0.0);
					S_ad=cp_ad.get_S_3Field();
					D_ad=cp_ad.get_D();
					dPsi_dJ_tilde= cp_ad.get_dPsi_dJ_tilde();
					dPsi_dp= cp_ad.get_dPsi_dp();
				}


				deformation_gradient[0][i] += (F_ad[0][0].val())/n_q_points; //F_xx
				deformation_gradient[1][i] += (F_ad[0][1].val())/n_q_points; //F_xy
				deformation_gradient[2][i] += (F_ad[1][0].val())/n_q_points; //F_yx
				deformation_gradient[3][i] += (F_ad[1][1].val())/n_q_points; //F_yy

				Piola_Kirchhoff[0][i] += (S_ad[0][0].val())/n_q_points; //S_xx
				Piola_Kirchhoff[1][i] += (S_ad[0][1].val())/n_q_points; //S_xy
				Piola_Kirchhoff[2][i] += (S_ad[1][0].val())/n_q_points; //S_yx
				Piola_Kirchhoff[3][i] += (S_ad[1][1].val())/n_q_points; //S_yy

				electric_field[0][i] += (-Grad_V_ad[q_point][0].val())/n_q_points; //E_x
				electric_field[1][i] += (-Grad_V_ad[q_point][1].val())/n_q_points; //E_y

				electric_displacement[0][i] += (D_ad[0].val())/n_q_points; //D_x
				electric_displacement[1][i] += (D_ad[1].val())/n_q_points; //D_x


			}
		}

	}

	// Write out main data file
	struct Filename
	{
		static std::string get_filename_vtu (unsigned int process,
				unsigned int cycle,
				const unsigned int n_digits = 4)
		{
			std::ostringstream filename_vtu;
			filename_vtu
			<< "solution-"
			<< (std::to_string(dim) + "d")
			<< "."
			<< Utilities::int_to_string (process, n_digits)
			<< "."
			<< Utilities::int_to_string(cycle, n_digits)
			<< ".vtu";
			return filename_vtu.str();
		}

		static std::string get_filename_pvtu (unsigned int timestep,
				const unsigned int n_digits = 4)
		{
			std::ostringstream filename_vtu;
			filename_vtu
			<< "solution-"
			<< (std::to_string(dim) + "d")
			<< "."
			<< Utilities::int_to_string(timestep, n_digits)
			<< ".pvtu";
			return filename_vtu.str();
		}

		static std::string get_filename_pvd (void)
		{
			std::ostringstream filename_vtu;
			filename_vtu
			<< "solution-"
			<< (std::to_string(dim) + "d")
			<< ".pvd";
			return filename_vtu.str();
		}
	};

	//	DataOut<dim> data_out;
	DataOut<dim, hp::DoFHandler<dim>> data_out;
	data_out.attach_dof_handler (dof_handler);

	std::vector<std::string> solution_names (n_components, "displacement");
	solution_names[V_component] = "voltage";
	solution_names[J_component] = "dilatation";
	solution_names[p_component] = "pressure";
	solution_names[T_component] = "temperature";

	std::vector<std::string> residual_names (solution_names);

	std::vector<std::string> reaction_forces_names (solution_names);

	for (unsigned int i=0; i < solution_names.size(); ++i)
	{
		solution_names[i].insert(0, "soln_");
		residual_names[i].insert(0, "res_");
	}

	std::vector<DataComponentInterpretation::DataComponentInterpretation>
	data_component_interpretation(dim,
			DataComponentInterpretation::component_is_part_of_vector);
	data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
	data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
	data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
	data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

	data_out.add_data_vector(locally_relevant_solution, solution_names,
			DataOut<dim, hp::DoFHandler<dim>>::type_dof_data,
			data_component_interpretation);

	data_out.add_data_vector (deformation_gradient[0], "F_xx");
	data_out.add_data_vector (deformation_gradient[1], "F_xy");
	data_out.add_data_vector (deformation_gradient[2], "F_yx");
	data_out.add_data_vector (deformation_gradient[3], "F_yy");

	data_out.add_data_vector (Piola_Kirchhoff[0], "S_xx");
	data_out.add_data_vector (Piola_Kirchhoff[1], "S_xy");
	data_out.add_data_vector (Piola_Kirchhoff[2], "S_yx");
	data_out.add_data_vector (Piola_Kirchhoff[3], "S_yy");

	data_out.add_data_vector (electric_field[0], "E_x");
	data_out.add_data_vector (electric_field[1], "E_y");

	data_out.add_data_vector (electric_displacement[0], "D_x");
	data_out.add_data_vector (electric_displacement[1], "D_y");


	data_out.build_patches (poly_order);

	const std::string filename_vtu = Filename::get_filename_vtu(this_mpi_process,
			timestep);
	std::ofstream output(filename_vtu.c_str());
	data_out.write_vtu(output);

	// Collection of files written in parallel
	// This next set of steps should only be performed
	// by master process
	if (this_mpi_process == 0)
	{
		// List of all files written out at this timestep by all processors
		std::vector<std::string> parallel_filenames_vtu;
		for (unsigned int p=0; p < n_mpi_processes; ++p)
		{
			parallel_filenames_vtu.push_back(Filename::get_filename_vtu(p,
					timestep));
		}

		const std::string filename_pvtu (Filename::get_filename_pvtu(timestep));
		std::ofstream pvtu_master(filename_pvtu.c_str());
		data_out.write_pvtu_record(pvtu_master,
				parallel_filenames_vtu);

		// Time dependent data master file
		static std::vector<std::pair<double,std::string> > time_and_name_history;
		time_and_name_history.push_back (std::make_pair (timestep,
				filename_pvtu));
		const std::string filename_pvd (Filename::get_filename_pvd());
		std::ofstream pvd_output (filename_pvd.c_str());
		DataOutBase::write_pvd_record (pvd_output, time_and_name_history);
	}
}



struct L2_norms
{
	L2_norms (const unsigned int block,
			const std::vector<IndexSet> &locally_owned_partitioning,
			const std::vector<IndexSet> &locally_relevant_partitioning,
			const MPI_Comm &mpi_communicator)
	: block (block),
	  locally_owned_partitioning (locally_owned_partitioning),
	  locally_relevant_partitioning (locally_relevant_partitioning),
	  mpi_communicator (mpi_communicator)
	{}

	const unsigned int block;
	const std::vector<IndexSet> &locally_owned_partitioning;
	const std::vector<IndexSet> &locally_relevant_partitioning;
	const MPI_Comm &mpi_communicator;

	double value = 1.0;
	double value_norm = 1.0;

	void
	set (const LA::MPI::BlockVector & vector,
			const ConstraintMatrix & all_constraints)
	{
		LA::MPI::BlockVector vector_zeroed;
		vector_zeroed.reinit (locally_owned_partitioning,
				locally_relevant_partitioning,
				mpi_communicator,
				true);
		vector_zeroed = vector;
		all_constraints.set_zero(vector_zeroed);

		value = vector_zeroed.block(block).l2_norm();

		// Reset if unsensible values
		if (value == 0.0) value = 1.0;
		value_norm = value;
	}

	void
	normalise (const L2_norms & norm_0)
	{
		value_norm/=norm_0.value;
	}
};

template<int dim>
double
CoupledProblem<dim>::get_norm(const SymmetricTensor<2,dim> X)
{

	double norm_squr(0);

	for (unsigned int A=0; A<dim; ++A)
		for (unsigned int B=0; B<dim; ++B)
		{
			norm_squr+=X[A][B]*X[A][B];
		}
	return std::sqrt(norm_squr);

}

template<int dim>
void
CoupledProblem<dim>::solve_nonlinear_timestep (const int ts)
{
	L2_norms ex_T  (T_block,
			locally_owned_partitioning,
			locally_relevant_partitioning,
			mpi_communicator);
	L2_norms ex_uV (uV_block,
			locally_owned_partitioning,
			locally_relevant_partitioning,
			mpi_communicator);

	//	locally_relevant_solution_t1 = locally_relevant_solution;


	L2_norms res_T_0(ex_T), update_T_0(ex_T);
	L2_norms res_T(ex_T), update_T(ex_T);
	L2_norms res_uV_0(ex_uV), update_uV_0(ex_uV);
	L2_norms res_uV(ex_uV), update_uV(ex_uV);

	pcout
	<< std::string(52,' ')
	<< "|"
	<< "  RES_T  " << std::string(2,' ')
	<< "  NUP_T  " << std::string(2,' ')
	<< "  RES_UV " << std::string(2,' ')
	<< "  NUP_UV "
	<< std::endl;

	for (unsigned int n=0; n < Parameters::max_newton_iterations; ++n)
	{
		pcout << "IT " << n << std::flush;

		LA::MPI::BlockVector locally_relevant_solution_update;
		locally_relevant_solution_update.reinit (locally_relevant_partitioning,
				mpi_communicator);

		make_constraints(n, ts);

		// === THERMAL PROBLEM ===

		system_matrix = 0;
		system_rhs = 0;
		locally_relevant_solution_update = 0;

		assemble_system_thermo();
		solve_thermo(locally_relevant_solution_update);
		locally_relevant_solution.block(T_block) += locally_relevant_solution_update.block(T_block);
		//      locally_relevant_solution.compress (VectorOperation::add);

		// Compute temperature residual
		{
			res_T.set(system_rhs, all_constraints);
			update_T.set(locally_relevant_solution_update,
					all_constraints);

			if (n == 0 || n == 1)
			{
				res_T_0.set(system_rhs, all_constraints);
				update_T_0.set(locally_relevant_solution_update,
						all_constraints);
			}

			res_T.normalise(res_T_0);
			update_T.normalise(update_T_0);
		}

		// === ELECTRO-MECHANICAL PROBLEM ===

		system_matrix = 0;
		system_rhs = 0;
		locally_relevant_solution_update = 0;

		assemble_system_mech();
		solve_mech(locally_relevant_solution_update);
		locally_relevant_solution.block(uV_block) += locally_relevant_solution_update.block(uV_block);
		//      locally_relevant_solution.compress (VectorOperation::add);

		// To analyse the residual, we must reassemble both
		// systems since they depend on one another
		//      assemble_system_thermo();
		//      assemble_system_mech();

		// Compute electro-mechanical residual
		{
			res_uV.set(system_rhs, all_constraints);
			update_uV.set(locally_relevant_solution_update,
					all_constraints);

			if (n == 0 || n == 1)
			{
				res_uV_0.set(system_rhs, all_constraints);
				update_uV_0.set(locally_relevant_solution_update,
						all_constraints);
			}

			res_uV.normalise(res_uV_0);
			update_uV.normalise(update_uV_0);
		}

		pcout
		<< std::fixed
		<< std::setprecision(3)
		<< std::setw(7)
		<< std::scientific
		<< "|"
		<< "  " << res_T.value_norm
		<< "  " << update_T.value_norm
		<< "  " << res_uV.value_norm
		<< "  " << update_uV.value_norm
		<< std::endl;

		bool converged_abs=false;
		bool converged_rel=false;

		{
			if((res_T.value < Parameters::max_res_abs) &&
					(res_uV.value < Parameters::max_res_abs))
			{
				converged_abs = true;
			}

			if((res_T.value_norm < Parameters::max_res_T_norm) &&
					(res_uV.value_norm < Parameters::max_res_uV_norm))
			{
				converged_rel = true;
			}
		}

		if (converged_abs || converged_rel)
		{
			pcout
			<< "Converged."
			<< std::endl;
			break;
		}

		if (n == (Parameters::max_newton_iterations-1))
		{
			pcout
			<< "No convergence... :-/"
			<< std::endl;
		}
	}

	pcout
	<< "Absolute values of residuals and Newton update:"
	<< std::endl
	<< "res_T:  " << res_T.value
	<< "\t update_T:  " << update_T.value
	<< std::endl
	<< "res_uV: " << res_uV.value
	<< "\t update_uV: " << update_uV.value
	<< std::endl;

}


template<int dim>
void
CoupledProblem<dim>::run ()
{
	make_grid();
	setup_system();

	{
		ConstraintMatrix constraints;
		constraints.close();

		const ComponentSelectFunction<dim> J_mask (J_component, n_components);

		hp::QCollection<dim> q_collection;

		{
			q_collection.push_back(qf_cell_3Field);
		}

		VectorTools::project (dof_handler,
				constraints,
				q_collection,
				J_mask,
				locally_relevant_solution);
	}

	output_results(0);

	double time = Parameters::dt;
	for (unsigned int ts = 1;
			ts<=Parameters::n_timesteps;
			++ts, time += Parameters::dt)
	{
		pcout
		<< std::endl
		<< std::string(100,'=')
		<< std::endl
		<< "Timestep: " << ts
		<< "\t Time: " << time
		<< std::endl
		<< std::string(100,'=')
		<< std::endl;
		solve_nonlinear_timestep(ts);
		output_results(ts);

	}


}
}


int
main (int argc, char *argv[])
{
	try
	{
		using namespace dealii;
		Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
		deallog.depth_console (0);

		Coupled_TEE::CoupledProblem<3> coupled_thermo_electro_elastic_problem_3d;
		coupled_thermo_electro_elastic_problem_3d.run();
	}
	catch (std::exception &exc)
	{
		std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl;
		std::cerr << "Exception on processing: "
				<< std::endl
				<< exc.what()
				<< std::endl
				<< "Aborting!"
				<< std::endl
				<< "----------------------------------------------------"
				<< std::endl;

		return 1;
	}
	catch (...)
	{
		std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl;
		std::cerr << "Unknown exception!"
				<< std::endl
				<< "Aborting!"
				<< std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		return 1;
	}

	return 0;
}
