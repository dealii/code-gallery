#ifndef dealii__cdr_system_rhs_h
#define dealii__cdr_system_rhs_h
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II-cdr/parameters.h>

#include <functional>

// Similarly to <code>create_system_matrix</code>, I wrote a separate function
// to compute the right hand side.
namespace CDR
{
  using namespace dealii;

  template<int dim, typename VectorType>
  void create_system_rhs
  (const DoFHandler<dim>                                 &dof_handler,
   const QGauss<dim>                                     &quad,
   const std::function<Tensor<1, dim>(const Point<dim>)> &convection_function,
   const std::function<double(double, const Point<dim>)> &forcing_function,
   const CDR::Parameters                                 &parameters,
   const VectorType                                      &previous_solution,
   const AffineConstraints<double>                       &constraints,
   const double                                           current_time,
   VectorType                                            &system_rhs);
}
#endif
