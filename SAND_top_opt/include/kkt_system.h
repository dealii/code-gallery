//
// Created by justin on 2/17/21.
//

#ifndef SAND_KKT_SYSTEM_H
#define SAND_KKT_SYSTEM_H
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/hp/fe_collection.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>


#include "../include/schur_preconditioner.h"
#include "../include/density_filter.h"
#include "matrix_free_elasticity.h"

#include <deal.II/base/conditional_ostream.h>

#include <iostream>
#include <fstream>
#include <algorithm>
namespace SAND {



    namespace LA
    {
        using namespace dealii::LinearAlgebraTrilinos;
    }
    using namespace dealii;

    template<int dim>
    class KktSystem {

    using SystemMFMatrixType = MF_Elasticity_Operator<dim, 1, double>;
    using LevelMFMatrixType = MF_Elasticity_Operator<dim, 1, double>;
    public:
        MPI_Comm  mpi_communicator;
        std::vector<IndexSet> owned_partitioning;
        std::vector<IndexSet> relevant_partitioning;

        KktSystem();

        void
        create_triangulation();

        void
        setup_boundary_values();

        void
        setup_filter_matrix();

        void
        setup_block_system();

        void
        assemble_block_system(const LA::MPI::BlockVector &state, const double barrier_size);

        LA::MPI::BlockVector
        solve(const LA::MPI::BlockVector &state);

        LA::MPI::BlockVector
        get_initial_state();

        double
        calculate_objective_value(const LA::MPI::BlockVector &state) const;

        double
        calculate_barrier_distance(const LA::MPI::BlockVector &state) const;

        double
        calculate_feasibility(const LA::MPI::BlockVector &state, const double barrier_size) const;

        double
        calculate_convergence(const LA::MPI::BlockVector &state) const;

        void
        output(const LA::MPI::BlockVector &state, const unsigned int j) const;

        void
        calculate_initial_rhs_error();

        double
        calculate_rhs_norm(const LA::MPI::BlockVector &state, const double barrier_size) const;

        void
        output_stl(const LA::MPI::BlockVector &state);

    private:

        LA::MPI::BlockVector
        calculate_rhs(const LA::MPI::BlockVector &test_solution, const double barrier_size) const;

        BlockDynamicSparsityPattern dsp;
        BlockSparsityPattern sparsity_pattern;
        mutable LA::MPI::BlockSparseMatrix system_matrix;
        mutable LA::MPI::BlockVector locally_relevant_solution;
        mutable LA::MPI::BlockVector distributed_solution;
        LA::MPI::BlockVector system_rhs;
        parallel::distributed::Triangulation<dim> triangulation;
        DoFHandler<dim> dof_handler;
        DoFHandler<dim> dof_handler_displacement;
        DoFHandler<dim> dof_handler_density;

        std::map<types::global_dof_index,types::global_dof_index> displacement_to_system_dof_index_map;
        MGLevelObject<std::map<types::global_dof_index,types::global_dof_index>> level_displacement_to_system_dof_index_map;

        AffineConstraints<double> constraints;
        AffineConstraints<double> displacement_constraints;
        FESystem<dim> fe_nine;
        FESystem<dim> fe_ten;
        hp::FECollection<dim> fe_collection;
        FESystem<dim> fe_displacement;
        FE_DGQ<dim> fe_density;
        const double density_ratio;
        const double density_penalty_exponent;

        mutable DensityFilter<dim> density_filter;

        std::map<types::global_dof_index, double> boundary_values;
        MGLevelObject<std::map<types::global_dof_index, double>> level_boundary_values;
        ConditionalOStream pcout;

        double initial_rhs_error;

        MGConstrainedDoFs mg_constrained_dofs;
        SystemMFMatrixType elasticity_matrix_mf;
        MGLevelObject<LevelMFMatrixType> mg_matrices;

        OperatorCellData<dim, GMGNumberType> active_cell_data;
        MGLevelObject<OperatorCellData<dim, GMGNumberType>> level_cell_data;
        dealii::LinearAlgebra::distributed::Vector<double> active_density_vector;
        dealii::LinearAlgebra::distributed::Vector<double> relevant_density_vector;
        MGLevelObject<dealii::LinearAlgebra::distributed::Vector<double>> level_density_vector;

        MGTransferMatrixFree<dim,GMGNumberType> transfer;
        MGTransferMatrixFree<dim, double> mg_transfer;

        using SmootherType = PreconditionChebyshev<LevelMFMatrixType, LinearAlgebra::distributed::Vector<double>>;
        mg::SmootherRelaxation<SmootherType,LinearAlgebra::distributed::Vector<double>> mg_smoother;
        MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
        MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<double>> mg_coarse;
        MGLevelObject<std::set<types::boundary_id>> level_dirichlet_boundary_dofs;
        MGLevelObject<AffineConstraints<double>> mg_level_constraints;
        MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMFMatrixType>> mg_interface_matrices;

        std::set<types::boundary_id> dirichlet_boundary;

        LinearAlgebra::distributed::Vector<double> distributed_displacement_sol;
        LinearAlgebra::distributed::Vector<double> distributed_displacement_rhs;

    };
}

#endif //SAND_KKT_SYSTEM_H
