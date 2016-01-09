#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II-cdr/system_rhs.templates.h>

namespace CDR
{
  using namespace dealii;

  template
  void create_system_rhs<2, Vector<double>>
  (const DoFHandler<2>                                 &dof_handler,
   const QGauss<2>                                     &quad,
   const std::function<Tensor<1, 2>(const Point<2>)>    convection_function,
   const std::function<double(double, const Point<2>)>  forcing_function,
   const CDR::Parameters                               &parameters,
   const Vector<double>                                &previous_solution,
   const ConstraintMatrix                              &constraints,
   const double                                         current_time,
   Vector<double>                                      &system_rhs);

  template
  void create_system_rhs<3, Vector<double>>
  (const DoFHandler<3>                                 &dof_handler,
   const QGauss<3>                                     &quad,
   const std::function<Tensor<1, 3>(const Point<3>)>    convection_function,
   const std::function<double(double, const Point<3>)>  forcing_function,
   const CDR::Parameters                               &parameters,
   const Vector<double>                                &previous_solution,
   const ConstraintMatrix                              &constraints,
   const double                                         current_time,
   Vector<double>                                      &system_rhs);

  template
  void create_system_rhs<2, TrilinosWrappers::MPI::Vector>
  (const DoFHandler<2>                                 &dof_handler,
   const QGauss<2>                                     &quad,
   const std::function<Tensor<1, 2>(const Point<2>)>    convection_function,
   const std::function<double(double, const Point<2>)>  forcing_function,
   const CDR::Parameters                               &parameters,
   const TrilinosWrappers::MPI::Vector                 &previous_solution,
   const ConstraintMatrix                              &constraints,
   const double                                         current_time,
   TrilinosWrappers::MPI::Vector                       &system_rhs);

  template
  void create_system_rhs<3, TrilinosWrappers::MPI::Vector>
  (const DoFHandler<3>                                 &dof_handler,
   const QGauss<3>                                     &quad,
   const std::function<Tensor<1, 3>(const Point<3>)>    convection_function,
   const std::function<double(double, const Point<3>)>  forcing_function,
   const CDR::Parameters                               &parameters,
   const TrilinosWrappers::MPI::Vector                 &previous_solution,
   const ConstraintMatrix                              &constraints,
   const double                                         current_time,
   TrilinosWrappers::MPI::Vector                       &system_rhs);
}
