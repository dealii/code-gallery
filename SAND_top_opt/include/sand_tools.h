//
// Created by justin on 7/3/21.
//

#ifndef SAND_MY_TOOLS_H
#define SAND_MY_TOOLS_H
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/matrix_out.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

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
#include <deal.II/base/config.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/table.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>

#include <deal.II/differentiation/ad/ad_number_traits.h>

#include <deal.II/lac/exceptions.h>
#include <deal.II/lac/identity_matrix.h>



namespace SAND{
    namespace LA
    {
        using namespace dealii::LinearAlgebraTrilinos;
    }

    using namespace dealii;


    void build_matrix_element_by_element (const LinearOperator<LA::MPI::Vector,LA::MPI::Vector,dealii::TrilinosWrappers::internal::LinearOperatorImplementation::TrilinosPayload> &op_X,
                                          FullMatrix<double>   &X_matrix,
                                          LA::MPI::Vector &exemplar_vector)
    {
        Threads::TaskGroup<void> tasks;
        for (unsigned int j=0; j<X_matrix.n(); ++j)
            tasks += Threads::new_task ([&op_X, &X_matrix, &exemplar_vector, j]()
                                        {
                                            LA::MPI::Vector e_j (exemplar_vector);
                                            LA::MPI::Vector r_j (exemplar_vector);

                                            e_j = 0;
                                            e_j(j) = 1;
                                            r_j = op_X * e_j;

                                            for (unsigned int i=0; i<X_matrix.m(); ++i)
                                                X_matrix(i,j) = r_j(i);
                                        });

        tasks.join_all();
    }

    void print_matrix (std::string name, FullMatrix<double>   &X_matrix)
    {
        const unsigned int n = X_matrix.n();
        const unsigned int m = X_matrix.m();
        std::ofstream Xmat(name);
        for (unsigned int i = 0; i < m; i++)
        {
            Xmat << X_matrix(i, 0);
            for (unsigned int j = 1; j < n; j++)
            {
                Xmat << "," << X_matrix(i, j);
            }
            Xmat << "\n";
        }
        Xmat.close();
    }

    void print_matrix (std::string &name, SparseMatrix<double>   &X_matrix)
    {
        const unsigned int n = X_matrix.n();
        const unsigned int m = X_matrix.m();
        std::ofstream Xmat(name);
        for (unsigned int i = 0; i < m; i++)
        {
            Xmat << X_matrix.el(i, 0);
            for (unsigned int j = 1; j < n; j++)
            {
                Xmat << "," << X_matrix.el(i, j);
            }
            Xmat << "\n";
        }
        Xmat.close();
    }

    void print_matrix (std::string &name, LA::MPI::SparseMatrix &X_matrix)
    {
        const unsigned int n = X_matrix.n();
        const unsigned int m = X_matrix.m();
        std::ofstream Xmat(name);
        for (unsigned int i = 0; i < m; i++)
        {
            Xmat << X_matrix.el(i, 0);
            for (unsigned int j = 1; j < n; j++)
            {
                Xmat << "," << X_matrix.el(i, j);
            }
            Xmat << "\n";
        }
        Xmat.close();
    }

}




#endif //SAND_MY_TOOLS_H
