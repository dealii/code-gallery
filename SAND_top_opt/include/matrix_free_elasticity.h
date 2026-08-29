#ifndef MATRIX_FREE_GMG_H
#define MATRIX_FREE_GMG_H

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/base/utilities.h>

#include "../include/parameters_and_components.h"
#include "../include/input_information.h"



namespace SAND
{
using GMGNumberType = double;
using namespace dealii;

template <int dim, typename number>
struct OperatorCellData
{
    Table<2, VectorizedArray<number>> density;
    std::size_t
    memory_consumption() const;
};

template <int dim, int fe_degree, typename number>
class MF_Elasticity_Operator
        : public MatrixFreeOperators::
        Base<dim, LinearAlgebra::distributed::Vector<number>>
{
public:

    MPI_Comm  mpi_communicator;

    using value_type = number;

    MF_Elasticity_Operator();

    void set_cell_data (const OperatorCellData<dim,number> &data);

    void compute_diagonal() override;

    void clear() override;

private:

    void apply_add (dealii::LinearAlgebra::distributed::Vector<number> &dst,
                    const dealii::LinearAlgebra::distributed::Vector<number> &src) const override;


    void local_apply (const dealii::MatrixFree<dim, number> &data,
                      dealii::LinearAlgebra::distributed::Vector<number> &dst,
                      const dealii::LinearAlgebra::distributed::Vector<number> &src,
                      const std::pair<unsigned int, unsigned int> &cell_range) const;

    void local_apply_face (const dealii::MatrixFree<dim, number> &data,
                           dealii::LinearAlgebra::distributed::Vector<number> &dst,
                           const dealii::LinearAlgebra::distributed::Vector<number> &src,
                           const std::pair<unsigned int, unsigned int> &face_range) const;


    void local_apply_boundary_face (const dealii::MatrixFree<dim, number> &data,
                                    dealii::LinearAlgebra::distributed::Vector<number> &dst,
                                    const dealii::LinearAlgebra::distributed::Vector<number> &src,
                                    const std::pair<unsigned int, unsigned int> &face_range) const;

    void local_compute_diagonal (const MatrixFree<dim,number>                     &data,
                              dealii::LinearAlgebra::distributed::Vector<number>  &dst,
                              const unsigned int &,
                              const std::pair<unsigned int,unsigned int>       &cell_range) const;


    const OperatorCellData<dim,number> *cell_data;

    ConditionalOStream pcout;

};


}




#endif // MATRIX_FREE_GMG_H
