#ifndef dealii__cdr_assemble_system_h
#define dealii__cdr_assemble_system_h
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function_parser.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/constraint_matrix.h>

#include <array>
#include <functional>

#include <deal.II-cdr/parameters.h>

namespace CDR
{
  using namespace dealii;

  template<int dim, typename MatrixType, typename VectorType>
  void assemble_system
  (const DoFHandler<dim>                                 &dof_handler,
   const QGauss<dim>                                     &quad,
   const std::function<Tensor<1, dim>(const Point<dim>)> convection_function,
   const std::function<double(double, const Point<dim>)> forcing_function,
   const CDR::Parameters                                 &parameters,
   const VectorType                                      &current_solution,
   const ConstraintMatrix                                &constraints,
   const double                                          current_time,
   MatrixType                                            &system_matrix,
   VectorType                                            &system_rhs);
}
#endif
