//
// Created by justin on 2/17/21.
//
#include "../include/matrix_free_elasticity.h"
#include "../include/input_information.h"
#include "../include/parameters_and_components.h"
namespace SAND {
using namespace dealii;

///Constructor
template <int dim, int fe_degree, typename number>
MF_Elasticity_Operator<dim, fe_degree, number>::MF_Elasticity_Operator()
    : MatrixFreeOperators::Base<dim,LinearAlgebra::distributed::Vector<number>>(),
    mpi_communicator(MPI_COMM_WORLD),
    pcout(std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
{
}

///Clears the objects, removes density information
template <int dim, int fe_degree, typename number>
void MF_Elasticity_Operator<dim, fe_degree, number>::clear()
{
    this->cell_data = nullptr;
    MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>::clear();
}

///Computes the diagonal for a preconditioner on the coarsest level.
template <int dim, int fe_degree, typename number>
void
MF_Elasticity_Operator<dim,fe_degree,number>::compute_diagonal ()
{
    this->inverse_diagonal_entries.
    reset(new DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<number>>());
    this->diagonal_entries.
    reset(new DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<number>>());

    dealii::LinearAlgebra::distributed::Vector<number> &inverse_diagonal = this->inverse_diagonal_entries->get_vector();

    dealii::LinearAlgebra::distributed::Vector<number> &diagonal = this->diagonal_entries->get_vector();

    this->data->initialize_dof_vector(inverse_diagonal);
    this->data->initialize_dof_vector(diagonal);
    unsigned int dummy = 0;


    this->data->cell_loop (&MF_Elasticity_Operator::local_compute_diagonal, this,
                           diagonal, dummy);
    this->data->cell_loop (&MF_Elasticity_Operator::local_compute_diagonal, this,
                           inverse_diagonal, dummy);

    this->set_constrained_entries_to_one(diagonal);
    this->set_constrained_entries_to_one(inverse_diagonal);

    // diagonal.compress(VectorOperation::add);
    
    for (auto &local_element : inverse_diagonal)
      {
//        Assert(local_element > 0.,
//               ExcMessage("No diagonal entry in a positive definite operator "
//                          "should be zero or negative."));
        local_element = 1./local_element;
      }
      // inverse_diagonal.compress(VectorOperation::insert);
    //   diagona.print(lstd::cout);
      pcout << "diag size: " << diagonal.size() << " with l2 norm " << diagonal.l2_norm() << std::endl;
}

///Computes the diagonal value locally for a cell
template <int dim, int fe_degree, typename number>
void
MF_Elasticity_Operator<dim,fe_degree,number>
::local_compute_diagonal (const MatrixFree<dim,number>                     &data,
                          dealii::LinearAlgebra::distributed::Vector<number>  &dst,
                          const unsigned int &,
                          const std::pair<unsigned int,unsigned int>       &cell_range) const
{
  // pcout << "local_compute_diagonal" << std::endl;
  FEEvaluation<dim,fe_degree,fe_degree+1,dim,number> displacement (data, 0);

  AlignedVector<VectorizedArray<number>> diagonal(displacement.dofs_per_cell);
  
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {

      VectorizedArray<number> cell_density = cell_data->density(cell, 0);
      double penalized_density = std::pow(cell_density[0],Input::density_penalty_exponent);

      displacement.reinit(cell);

      for (unsigned int i=0; i<displacement.dofs_per_cell; ++i)
        {
          for (unsigned int j=0; j<displacement.dofs_per_cell; ++j)
            displacement.begin_dof_values()[j] = VectorizedArray<number>();

          displacement.begin_dof_values()[i] = make_vectorized_array<number> (1.);

          displacement.evaluate (EvaluationFlags::gradients);

          for (unsigned int q=0; q<displacement.n_q_points; ++q)
            {
              SymmetricTensor< 2, dim, VectorizedArray<double> > symgrad_term = penalized_density* 2.0 * Input::material_mu *displacement.get_symmetric_gradient(q);
              VectorizedArray<number> div_term = trace(displacement.get_symmetric_gradient(q));

              for (unsigned int d = 0; d < dim; ++d)
              {
                  symgrad_term[d][d] += penalized_density * Input::material_lambda * div_term;
              }


              displacement.submit_symmetric_gradient( symgrad_term , q);
            }

          displacement.integrate (EvaluationFlags::gradients);

          diagonal[i] = displacement.begin_dof_values()[i];
        }
      
      for (unsigned int i=0; i<displacement.dofs_per_cell; ++i)
        displacement.begin_dof_values()[i] = diagonal[i];
      displacement.distribute_local_to_global (dst);     

    }
}

///Applies the elasticity operator locally. Matches what happens in KKT System
template <int dim, int fe_degree, typename number>
void
MF_Elasticity_Operator<dim, fe_degree, number>::local_apply(
        const MatrixFree<dim, number> &                   data,
        LinearAlgebra::distributed::Vector<number> &      dst,
        const LinearAlgebra::distributed::Vector<number> &src,
        const std::pair<unsigned int, unsigned int> &     cell_range) const
{
  // pcout << "local_apply" << std::endl;
    FEEvaluation<dim, 1, 2, dim, double> displacement(data,0);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
        VectorizedArray<number> cell_density = cell_data->density(cell, 0);
        double penalized_density = std::pow(cell_density[0],Input::density_penalty_exponent);

        displacement.reinit(cell);
//        displacement.read_dof_values(src);

        displacement.gather_evaluate(src, EvaluationFlags::gradients);

        for (unsigned int q = 0; q < displacement.n_q_points; ++q)
        {
            SymmetricTensor< 2, dim, VectorizedArray<double> > symgrad_term = penalized_density* 2.0 * Input::material_mu *displacement.get_symmetric_gradient(q);
            VectorizedArray<number> div_term = penalized_density * Input::material_lambda * trace(displacement.get_symmetric_gradient(q));

            for (unsigned int d = 0; d < dim; ++d)
            {
                symgrad_term[d][d] +=  div_term;
            }

            

            displacement.submit_symmetric_gradient(symgrad_term, q);
        }
        displacement.integrate_scatter(EvaluationFlags::gradients, dst);
    }

}

///Nothing is applied on a face on the LHS, so left blank.
template <int dim, int fe_degree, typename number>
void
MF_Elasticity_Operator<dim, fe_degree, number>
::local_apply_face(const dealii::MatrixFree<dim, number> &,
                   dealii::LinearAlgebra::distributed::Vector<number> &,
                   const dealii::LinearAlgebra::distributed::Vector<number> &,
                   const std::pair<unsigned int, unsigned int> &) const
{
}

///Nothing is applied on a face on the LHS, so left blank.
template <int dim, int fe_degree, typename number>
void
MF_Elasticity_Operator<dim, fe_degree, number>
::local_apply_boundary_face(const dealii::MatrixFree<dim, number> &,
                            dealii::LinearAlgebra::distributed::Vector<number> &,
                            const dealii::LinearAlgebra::distributed::Vector<number> &,
                            const std::pair<unsigned int, unsigned int> &) const
{
}

///Loops over all cells to apply the elasticity operatorto the entire LHS vector
template <int dim, int fe_degree, typename number>
void
MF_Elasticity_Operator<dim,fe_degree,number>
::apply_add (dealii::LinearAlgebra::distributed::Vector<number> &dst,
             const dealii::LinearAlgebra::distributed::Vector<number> &src) const
{
    MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::Vector<number>>::
            data->cell_loop(&MF_Elasticity_Operator::local_apply, this, dst, src);
}

///Sets cell data (density) to be input given.
template <int dim, int fe_degree, typename number>
void
MF_Elasticity_Operator<dim,fe_degree,number>::set_cell_data (const OperatorCellData<dim,number> &data)
{
    this->cell_data = &data;
}



}

template class SAND::MF_Elasticity_Operator<2,1,double>;
template class SAND::MF_Elasticity_Operator<3,1,double>;

