#ifndef dealii__cdr_system_matrix_h
#define dealii__cdr_system_matrix_h
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/constraint_matrix.h>

#include <deal.II-cdr/parameters.h>

#include <functional>

namespace CDR
{
  using namespace dealii;

  template<int dim, typename MatrixType>
  void create_system_matrix
  (const DoFHandler<dim>                                 &dof_handler,
   const QGauss<dim>                                     &quad,
   const std::function<Tensor<1, dim>(const Point<dim>)> convection_function,
   const CDR::Parameters                                 &parameters,
   const double                                           time_step,
   MatrixType                                            &system_matrix);

  template<int dim, typename MatrixType>
  void create_system_matrix
  (const DoFHandler<dim>                                 &dof_handler,
   const QGauss<dim>                                     &quad,
   const std::function<Tensor<1, dim>(const Point<dim>)> convection_function,
   const CDR::Parameters                                 &parameters,
   const double                                          time_step,
   const ConstraintMatrix                                &constraints,
   MatrixType                                            &system_matrix);
}
#endif
