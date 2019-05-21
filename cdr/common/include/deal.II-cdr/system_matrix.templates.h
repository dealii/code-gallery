#ifndef dealii__cdr_system_matrix_templates_h
#define dealii__cdr_system_matrix_templates_h
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II-cdr/parameters.h>
#include <deal.II-cdr/system_matrix.h>

#include <functional>
#include <vector>

namespace CDR
{
  using namespace dealii;

  // This is the actual implementation of the <code>create_system_matrix</code>
  // function described in the header file. It is similar to the system matrix
  // assembly routine in step-40.
  template<int dim, typename UpdateFunction>
  void internal_create_system_matrix
  (const DoFHandler<dim>                                 &dof_handler,
   const QGauss<dim>                                     &quad,
   const std::function<Tensor<1, dim>(const Point<dim>)> &convection_function,
   const CDR::Parameters                                 &parameters,
   const double                                           time_step,
   UpdateFunction                                         update_system_matrix)
  {
    auto &fe = dof_handler.get_fe();
    const auto dofs_per_cell = fe.dofs_per_cell;
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    FEValues<dim> fe_values(fe, quad, update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    std::vector<types::global_dof_index> local_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            cell_matrix = 0.0;
            cell->get_dof_indices(local_indices);
            for (unsigned int q = 0; q < quad.size(); ++q)
              {
                const auto current_convection =
                  convection_function(fe_values.quadrature_point(q));

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        const auto convection_contribution = current_convection
                                                             *fe_values.shape_grad(j, q);
                        cell_matrix(i, j) += fe_values.JxW(q)*
                                             // Here are the time step, mass, and reaction parts:
                                             ((1.0 + time_step/2.0*parameters.reaction_coefficient)
                                              *fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                              + time_step/2.0*
                                              // and the convection part:
                                              (fe_values.shape_value(i, q)*convection_contribution
                                               // and, finally, the diffusion part:
                                               + parameters.diffusion_coefficient
                                               *(fe_values.shape_grad(i, q)*fe_values.shape_grad(j, q)))
                                             );
                      }
                  }
              }
            update_system_matrix(local_indices, cell_matrix);
          }
      }
  }

  template<int dim, typename MatrixType>
  void create_system_matrix
  (const DoFHandler<dim>                                 &dof_handler,
   const QGauss<dim>                                     &quad,
   const std::function<Tensor<1, dim>(const Point<dim>)> &convection_function,
   const CDR::Parameters                                 &parameters,
   const double                                           time_step,
   const AffineConstraints<double>                       &constraints,
   MatrixType                                            &system_matrix)
  {
    internal_create_system_matrix<dim>
    (dof_handler, quad, convection_function, parameters, time_step,
     [&constraints, &system_matrix](const std::vector<types::global_dof_index> &local_indices,
                                    const FullMatrix<double> &cell_matrix)
    {
      constraints.distribute_local_to_global
      (cell_matrix, local_indices, system_matrix);
    });
  }

  template<int dim, typename MatrixType>
  void create_system_matrix
  (const DoFHandler<dim>                                 &dof_handler,
   const QGauss<dim>                                     &quad,
   const std::function<Tensor<1, dim>(const Point<dim>)> &convection_function,
   const CDR::Parameters                                 &parameters,
   const double                                           time_step,
   MatrixType                                            &system_matrix)
  {
    internal_create_system_matrix<dim>
    (dof_handler, quad, convection_function, parameters, time_step,
     [&system_matrix](const std::vector<types::global_dof_index> &local_indices,
                      const FullMatrix<double> &cell_matrix)
    {
      system_matrix.add(local_indices, cell_matrix);
    });
  }
}
#endif
