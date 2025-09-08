//
// Created by justin on 5/13/21.
//

#ifndef SAND_DENSITY_FILTER_H
#define SAND_DENSITY_FILTER_H


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/cell_id.h>

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

#include <iostream>
#include <fstream>
#include <algorithm>

/* This object is designed to calculate and form the matrix corresponding to a convolution of the unfiltered density.
 * Once formed, we have F*\sigma = \rho*/
namespace SAND {
    using namespace dealii;
    namespace LA
    {
        using namespace dealii::LinearAlgebraTrilinos;
    }
    template<int dim>
    class DensityFilter {
    public:

        MPI_Comm  mpi_communicator;
        std::vector<IndexSet> owned_partitioning;
        std::vector<IndexSet> relevant_partitioning;

        DensityFilter();
        DynamicSparsityPattern filter_dsp;
        LA::MPI::SparseMatrix filter_matrix;
        LA::MPI::SparseMatrix filter_matrix_transpose;
        SparsityPattern filter_sparsity_pattern;
        void initialize(DoFHandler<dim> &dof_handler);
        std::set<types::global_dof_index> find_relevant_neighbors(types::global_dof_index cell_index) const;

    private:
        std::vector<double> cell_m;
        std::vector<double> x_coord;
        std::vector<double> y_coord;
        std::vector<double> z_coord;
        std::vector<double> cell_m_part;
        std::vector<double> x_coord_part;
        std::vector<double> y_coord_part;
        std::vector<double> z_coord_part;

        ConditionalOStream pcout;

    };
}
#endif //SAND_DENSITY_FILTER_H
