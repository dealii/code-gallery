#pragma once
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/differentiation/ad.h>

#include "Deal_II_MMA.h"

namespace Elasticity_MF
{
	/** Lower bound for the SIMP interpolation. */
	constexpr double epsimin = 1e-2;

	/** SIMP penalization exponent. */
	constexpr double penal = 3.;

	constexpr double Material_Lambda = 1.0;
	constexpr double Material_Mu     = 1.0;

	using namespace dealii;
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <int dim, typename number>
    inline SymmetricTensor<4, dim, number> 
		get_stress_strain_tensor(const number &lambda,
                                 const number &mu)
    {
      SymmetricTensor<4, dim, number> tmp;
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          for (unsigned int k = 0; k < dim; ++k)
            for (unsigned int l = 0; l < dim; ++l)
              tmp[i][j][k][l] = (((i == k) && (j == l) ? mu : number(0.0)) +
                                 ((i == l) && (j == k) ? mu : number(0.0)) +
                                 ((i == j) && (k == l) ? lambda : number(0.0)));
      return tmp;
    }
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <int dim, int fe_degree, typename number>
	class Elasticity_Matrix_Free
		: public MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>
	{
	public:
		using VectorType = LinearAlgebra::distributed::Vector<number>;

		Elasticity_Matrix_Free();

		void clear() override;

		void evaluate_coefficient(const LinearAlgebra::distributed::Vector<number> &Simp_Rho,
			                      number Material_Lambda,
						          number Material_Mu);

		virtual void compute_diagonal() override;

		// \lambda(cell)
		Table<1, VectorizedArray<number>> lambda;
		// \mu(cell)
		Table<1, VectorizedArray<number>> mu;

	private:
		virtual void apply_add(VectorType &dst,
							const VectorType &src) const override;

		void local_apply(const MatrixFree<dim, number> &data,
						VectorType &dst,
						const VectorType &src,
						const std::pair<unsigned int, unsigned int> &cell_range) const;

		void local_compute_diagonal(
		FEEvaluation<dim, fe_degree, fe_degree + 1, dim, number> &fe_values) const;
	};

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <int dim, int fe_degree, typename number>
	Elasticity_Matrix_Free<dim, fe_degree, number>::Elasticity_Matrix_Free()
		: MatrixFreeOperators::Base<dim, VectorType>()
	{}

	template <int dim, int fe_degree, typename number>
	void Elasticity_Matrix_Free<dim, fe_degree, number>::clear()
	{
		lambda.reinit(0);
		mu.reinit(0);
		MatrixFreeOperators::Base<dim, VectorType>::clear();
	}

	template <int dim, int fe_degree, typename number>
	void Elasticity_Matrix_Free<dim, fe_degree, number>::
	evaluate_coefficient(const LinearAlgebra::distributed::Vector<number> &Simp_Rho,
		                 number Material_Lambda,
						 number Material_Mu)
	{
		const unsigned int n_cells = this->data->n_cell_batches();
		FEEvaluation<dim, fe_degree, fe_degree + 1, dim, number> fe_values(*this->data);

		lambda.reinit(n_cells);
		mu.reinit(n_cells);
		for (unsigned int cell = 0; cell < n_cells; ++cell)
		{
			fe_values.reinit(cell);
			fe_values.read_dof_values_plain(Simp_Rho);
			auto lambda_cell = make_vectorized_array<number>(Material_Lambda);
			auto mu_cell = make_vectorized_array<number>(Material_Mu);
			//
			lambda(cell) = (epsimin +
					 pow<number>(fe_values.get_dof_value(0)[0], penal) * (1. - epsimin)) *
					 lambda_cell;
			mu(cell) = (epsimin +
					 pow<number>(fe_values.get_dof_value(0)[0], penal) * (1. - epsimin)) *
					 mu_cell;
		}
	}

	template <int dim, int fe_degree, typename number>
	void Elasticity_Matrix_Free<dim, fe_degree, number>::apply_add(VectorType &dst,
																const VectorType &src) const
	{
		this->data->cell_loop(&Elasticity_Matrix_Free::local_apply, this, dst, src);
	}

	template <int dim, int fe_degree, typename number>
	void Elasticity_Matrix_Free<dim, fe_degree, number>::local_apply(
			const MatrixFree<dim, number> &data,
			VectorType &dst,
			const VectorType &src,
			const std::pair<unsigned int, unsigned int> &cell_range) const
	{
		FEEvaluation<dim, fe_degree, fe_degree + 1, dim, number> fe_values(data);

		for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
		{
			fe_values.reinit(cell);
			const auto C_ijkl = get_stress_strain_tensor<dim>(lambda(cell), mu(cell));

			fe_values.read_dof_values(src);
			fe_values.evaluate(EvaluationFlags::gradients);
			for (const unsigned int q : fe_values.quadrature_point_indices())
			{
				fe_values.submit_symmetric_gradient(C_ijkl * 
						fe_values.get_symmetric_gradient(q), q);
			}
			fe_values.integrate(EvaluationFlags::gradients);
			fe_values.distribute_local_to_global(dst);
		}
	}

	template <int dim, int fe_degree, typename number>
	void Elasticity_Matrix_Free<dim, fe_degree, number>::compute_diagonal()
	{
		this->inverse_diagonal_entries.reset(new DiagonalMatrix<VectorType>());

		VectorType &inverse_diagonal = this->inverse_diagonal_entries->get_vector();
		this->data->initialize_dof_vector(inverse_diagonal);

		MatrixFreeTools::compute_diagonal(*this->data,
										inverse_diagonal,
										&Elasticity_Matrix_Free::local_compute_diagonal,
										this);
	
		this->set_constrained_entries_to_one(inverse_diagonal);
	
		for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i)
		{
			Assert(inverse_diagonal.local_element(i) > 0.,
				ExcMessage("No diagonal entry in a positive definite operator "
							"should be zero"));
			inverse_diagonal.local_element(i) = 1. / inverse_diagonal.local_element(i);
		}
	}

	template <int dim, int fe_degree, typename number>
	void Elasticity_Matrix_Free<dim, fe_degree, number>::local_compute_diagonal(
		FEEvaluation<dim, fe_degree, fe_degree + 1, dim, number> &fe_values) const
	{
		const unsigned int cell = fe_values.get_current_cell_index();
		const auto C_ijkl = get_stress_strain_tensor<dim>(lambda(cell), mu(cell));
		fe_values.evaluate(EvaluationFlags::gradients);
		for (const unsigned int q : fe_values.quadrature_point_indices())
		{
			fe_values.submit_symmetric_gradient(C_ijkl * 
					fe_values.get_symmetric_gradient(q), q);
		}
		fe_values.integrate(EvaluationFlags::gradients);
	}

	// -------------------------------------------------------------------------
	// Helmholtz PDE filter
	// -------------------------------------------------------------------------

	/**
	 * @brief Matrix-free Helmholtz operator used as a PDE-based density filter.
	 *
	 * The operator corresponds to
	 * @f$ \bigl(1 - R_{min}^{2} \Delta\bigr) @f$,
	 * discretized with a matrix-free continuous FE space. It maps the
	 * unfiltered density to a smoothed density field. The filter radius
	 * enters as the squared, scaled radius @p Rmin_Pow2.
	 *
	 * @note The same Helmholtz matrix is shared by the density filter and
	 *       the topology-optimization sensitivity filter. Both applications
	 *       solve
	 *       @f$ \bigl(1 - R_{min}^2 \Delta\bigr)\, x = b @f$
	 *       with the same operator and differ only in the right-hand side
	 *       @f$ b @f$:
	 *         - density filter:      @f$ b = \rho @f$ (the design density),
	 *         - sensitivity filter:  @f$ b = \partial J / \partial \tilde{\rho} @f$
	 *           (the objective derivative w.r.t. the filtered density).
	 *       The matrix (and its multigrid hierarchy) therefore only needs
	 *       to be (re)assembled when the **mesh changes**; within one
	 *       optimization loop the same operator is reused and only the
	 *       RHS vector is re-assembled.
	 *
	 * @tparam dim        Spatial dimension.
	 * @tparam fe_degree  Polynomial degree (FE_Q(1) is sufficient here).
	 * @tparam number     Scalar type.
	 */
	template <int dim, int fe_degree, typename number>
	class Helmholtz_Matrix_Free
			: public MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>
	{
	public:
		using value_type = number;

		Helmholtz_Matrix_Free();

		void clear() override;

		/** @brief Compute the inverse diagonal (Jacobi preconditioner). */
		virtual void compute_diagonal() override;

		/** Squared (scaled) filter radius multiplying the Laplacian part. */
		number Rmin_Pow2;

	private:
		virtual void apply_add(
				LinearAlgebra::distributed::Vector<number> &dst,
				const LinearAlgebra::distributed::Vector<number> &src) const override;

		void local_apply(
				const MatrixFree<dim, number> &data,
				LinearAlgebra::distributed::Vector<number> &dst,
				const LinearAlgebra::distributed::Vector<number> &src,
				const std::pair<unsigned int, unsigned int> &cell_range) const;

		void local_compute_diagonal(
				const MatrixFree<dim, number> &data,
				LinearAlgebra::distributed::Vector<number> &dst,
				const unsigned int &dummy,
				const std::pair<unsigned int, unsigned int> &cell_range) const;
	};

	// -------------------------------------------------------------------------
	// Implementation of Helmholtz_Matrix_Free
	// -------------------------------------------------------------------------

	template <int dim, int fe_degree, typename number>
	Helmholtz_Matrix_Free<dim, fe_degree, number>::Helmholtz_Matrix_Free()
			: MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>()
	{}

	template <int dim, int fe_degree, typename number>
	void Helmholtz_Matrix_Free<dim, fe_degree, number>::clear()
	{
		MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>::clear();
	}

	template <int dim, int fe_degree, typename number>
	void Helmholtz_Matrix_Free<dim, fe_degree, number>::compute_diagonal()
	{
		this->inverse_diagonal_entries.reset(
				new DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>());
		LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
				this->inverse_diagonal_entries->get_vector();
		this->data->initialize_dof_vector(inverse_diagonal);

		unsigned int dummy = 0;
		this->data->cell_loop(&Helmholtz_Matrix_Free::local_compute_diagonal,
							  this,
							  inverse_diagonal,
							  dummy);

		this->set_constrained_entries_to_one(inverse_diagonal);
		for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); i++)
		{
			inverse_diagonal.local_element(i) =
					1. / inverse_diagonal.local_element(i);
		}
	}

	template <int dim, int fe_degree, typename number>
	void Helmholtz_Matrix_Free<dim, fe_degree, number>::apply_add(
			LinearAlgebra::distributed::Vector<number> &dst,
			const LinearAlgebra::distributed::Vector<number> &src) const
	{
		this->data->cell_loop(&Helmholtz_Matrix_Free::local_apply, this, dst, src);
	}

	template <int dim, int fe_degree, typename number>
	void Helmholtz_Matrix_Free<dim, fe_degree, number>::local_apply(
			const MatrixFree<dim, number> &data,
			LinearAlgebra::distributed::Vector<number> &dst,
			const LinearAlgebra::distributed::Vector<number> &src,
			const std::pair<unsigned int, unsigned int> &cell_range) const
	{
		FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> fe_values(data);

		for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
		{
			fe_values.reinit(cell);
			fe_values.read_dof_values(src);
			fe_values.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

			// Helmholtz operator: value + Rmin^2 * gradient contribution.
			for (const unsigned int q : fe_values.quadrature_point_indices())
			{
				fe_values.submit_value(fe_values.get_value(q), q);
				fe_values.submit_gradient(Rmin_Pow2 * fe_values.get_gradient(q), q);
			}

			fe_values.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
			fe_values.distribute_local_to_global(dst);
		}
	}

	template <int dim, int fe_degree, typename number>
	void Helmholtz_Matrix_Free<dim, fe_degree, number>::local_compute_diagonal(
			const MatrixFree<dim, number> &data,
			LinearAlgebra::distributed::Vector<number> &dst,
			const unsigned int &,
			const std::pair<unsigned int, unsigned int> &cell_range) const
	{
		FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> fe_values(data);
		AlignedVector<VectorizedArray<number>> diagonal(fe_values.dofs_per_cell);

		for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
		{
			fe_values.reinit(cell);

			for (unsigned int i = 0; i < fe_values.dofs_per_cell; ++i)
			{
				for (unsigned int j = 0; j < fe_values.dofs_per_cell; ++j)
				{
					fe_values.submit_dof_value(VectorizedArray<number>(), j);
				}
				fe_values.submit_dof_value(make_vectorized_array<number>(1.), i);

				fe_values.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
				for (const unsigned int q : fe_values.quadrature_point_indices())
				{
					fe_values.submit_value(fe_values.get_value(q), q);
					fe_values.submit_gradient(Rmin_Pow2 * fe_values.get_gradient(q), q);
				}
				fe_values.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
				diagonal[i] = fe_values.get_dof_value(i);
			}

			for (unsigned int i = 0; i < fe_values.dofs_per_cell; ++i)
			{
				fe_values.submit_dof_value(diagonal[i], i);
			}
			fe_values.distribute_local_to_global(dst);
		}
	}

	// -------------------------------------------------------------------------
	// Main driver
	// -------------------------------------------------------------------------
	template<unsigned int dim, unsigned degree>
	class Elasticity_Solver
	{
	public:
		Elasticity_Solver();

		/** Run the full topology optimization loop. */
		void run();

	private:
		// --- Mesh ---
		/** Generate the initial coarse mesh on the distributed triangulation. */
		void make_grid();

		// --- Elasticity system ---
		/** Set up DoFHandlers, constraints and the fine-level matrix-free Elasticity operator. */
		void setup_elasticity_system();
		/** Build the geometric multigrid hierarchy for the elasticity operator. */
		void setup_multigrid_elasticity_system();

		// --- Helmholtz PDE filter ---
		/** Set up the Helmholtz filter system on FE_Q(1). */
		void setup_filter_system();
		/** Build the geometric multigrid hierarchy for the filter operator. */
		void setup_multigrid_filter_system();

		// --- SIMP ---
		/** Allocate the discrete/continuous density vectors and cell volumes. */
		void setup_simp_system();

		// --- PDE filter ---
		/** Assemble the right-hand side of the Helmholtz filter from the given input field. */
		void assemble_filter_rhs(const LinearAlgebra::distributed::Vector<double> &input);
		/** Solve the filtered density problem with a Chebyshev-preconditioned CG. */
		void solve_filter_chebyshev();
		/** Average the filtered density onto the DGQ(0) representation. */
		void generate_average_vector(LinearAlgebra::distributed::Vector<double> &output);

		// --- elasticity solver and sensitivity ---
		/** Interpolate the discrete density onto the continuous temperature FE space. */
		void interpolate_simp_vector();
		/** Transfer the continuous density to all multigrid levels. */
		void transfer_simp_vector();
		/** Assemble the right-hand side of the thermal conduction problem. */
		void assemble_elasticity_rhs();
		/** Solve the primal problem with a Chebyshev-preconditioned CG. */
		void solve_chebyshev();
		/** Evaluate the discrete objective derivative (sensitivity). */
		void generate_object_discrete_derivative();

		// --- Utilities ---
		/** Compute the per-cell volume (used in the objective/constraints). */
		void get_cell_volume();
		/** Write VTU output for visualization. */
		void output_results();
		/** Print timing information (wraps TimerOutput). */
		void time();

		// --- Data members ---

		MPI_Comm mpi_communicator;
		parallel::distributed::Triangulation<dim> triangulation;

		// Mappings used for the primal and filter spaces.
		MappingFE<dim> mapping;
		MappingFE<dim> mapping_filter;

		// Finite elements:
		//  - fe         : continuous elasticity space,
		//  - fe_simp    : DGQ(0) space for the design density,
		//  - fe_filter  : continuous space for the filtered density.
		const FESystem<dim>    fe;
		const FE_DGQ<dim>  fe_simp;
		const FE_Q<dim>    fe_filter;

		DoFHandler<dim> dof_handler;
		DoFHandler<dim> dof_handler_simp;
		DoFHandler<dim> dof_handler_filter;

		AffineConstraints<double> constraints;
		AffineConstraints<double> constraints_filter;

		using VectorType      = LinearAlgebra::distributed::Vector<double>;
		using LeveLVectorType = LinearAlgebra::distributed::Vector<float>;

		using ElasticityMatrixType       = Elasticity_Matrix_Free<dim, degree, double>;
		using LeveLElasticityMatrixType  = Elasticity_Matrix_Free<dim, degree, float>;

		// Elasticity (thermal conduction) system.
		ElasticityMatrixType elasticity_matrix;
		VectorType           elasticity_rhs;
		VectorType           elasticity_displacement;
		MGConstrainedDoFs                       mg_constrained_dofs;
		MGLevelObject<LeveLElasticityMatrixType>   mg_matrices;

		// PDE filter system. FE_Q(1) is sufficient here; higher orders are
		// not needed for the filter.
		using FilterMatrixType      = Helmholtz_Matrix_Free<dim, 1, double>;
		using LeveLFilterMatrixType = Helmholtz_Matrix_Free<dim, 1, float>;
		FilterMatrixType filter_matrix;
		VectorType       filter_rhs;
		VectorType       filter_solution;
		MGConstrainedDoFs                       mg_constrained_dofs_filter;
		MGLevelObject<LeveLFilterMatrixType>    mg_matrices_filter;

		// Density vectors used by the multigrid hierarchy.
		VectorType                       Simp_Rho_Continuous;
		MGLevelObject<LeveLVectorType>   Simp_Rho_Continuous_Level;

		// Density vectors used by the optimization (before/after filtering).
		VectorType Simp_Rho_Discrete;
		VectorType Simp_Rho_Discrete_Filter;
		VectorType Simp_Rho_Discrete_Max;
		VectorType Simp_Rho_Discrete_Min;

		// Input data for the MMA optimizer.
		VectorType              Object_Discrete_Derivative;
		std::vector<VectorType> Constraint_Discrete_Derivative;
		VectorType              Cell_Volume_Discrete;

		// MMA optimizer.
		Deal_II_MMA_Solver<VectorType> MMA_Optimizer;

		ConditionalOStream pcout;
		TimerOutput        computing_timer;

		/** Filter radius for the PDE-based density filter. */
		double Rmin;
	};
};

template class Elasticity_MF::Elasticity_Solver<2u,1u>;
template class Elasticity_MF::Elasticity_Solver<3u,1u>;