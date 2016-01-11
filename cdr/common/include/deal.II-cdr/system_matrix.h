#ifndef dealii__cdr_system_matrix_h
#define dealii__cdr_system_matrix_h
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/constraint_matrix.h>

#include <deal.II-cdr/parameters.h>

#include <functional>

// One of the goals I had in writing this entry was to split up functions into
// different compilation units instead of using one large file. This is the
// header file for a pair of functions (only one of which I ultimately use)
// which build the system matrix.
namespace CDR
{
  using namespace dealii;

  template<int dim, typename MatrixType>
  void create_system_matrix
  (const DoFHandler<dim>                                 &dof_handler,
   const QGauss<dim>                                     &quad,
   const std::function<Tensor<1, dim>(const Point<dim>)> &convection_function,
   const CDR::Parameters                                 &parameters,
   const double                                           time_step,
   MatrixType                                            &system_matrix);

  template<int dim, typename MatrixType>
  void create_system_matrix
  (const DoFHandler<dim>                                 &dof_handler,
   const QGauss<dim>                                     &quad,
   const std::function<Tensor<1, dim>(const Point<dim>)> &convection_function,
   const CDR::Parameters                                 &parameters,
   const double                                           time_step,
   const ConstraintMatrix                                &constraints,
   MatrixType                                            &system_matrix);
}
#endif
