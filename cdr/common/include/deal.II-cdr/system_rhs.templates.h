#ifndef dealii__cdr_system_rhs_templates_h
#define dealii__cdr_system_rhs_templates_h
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II-cdr/parameters.h>
#include <deal.II-cdr/system_rhs.h>

#include <functional>
#include <vector>

namespace CDR
{
  using namespace dealii;

  template<int dim, typename VectorType>
  void create_system_rhs
  (const DoFHandler<dim>                                 &dof_handler,
   const QGauss<dim>                                     &quad,
   const std::function<Tensor<1, dim>(const Point<dim>)> convection_function,
   const std::function<double(double, const Point<dim>)> forcing_function,
   const CDR::Parameters                                 &parameters,
   const VectorType                                      &previous_solution,
   const ConstraintMatrix                                &constraints,
   const double                                          current_time,
   VectorType                                            &system_rhs)
  {
    auto &fe = dof_handler.get_fe();
    const auto dofs_per_cell = fe.dofs_per_cell;
    const double time_step = (parameters.stop_time - parameters.start_time)
      /parameters.n_time_steps;
    FEValues<dim> fe_values(fe, quad, update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    Vector<double> cell_rhs(dofs_per_cell);
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

    Vector<double> current_fe_coefficients(dofs_per_cell);
    std::vector<types::global_dof_index> local_indices(dofs_per_cell);

    const double previous_time {current_time - time_step};

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            cell_rhs = 0.0;
            cell->get_dof_indices(local_indices);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                current_fe_coefficients[i] = previous_solution[local_indices[i]];
              }

            for (unsigned int q = 0; q < quad.size(); ++q)
              {
                const auto current_convection =
                  convection_function(fe_values.quadrature_point(q));

                const double current_forcing = forcing_function
                  (current_time, fe_values.quadrature_point(q));
                const double previous_forcing = forcing_function
                  (previous_time, fe_values.quadrature_point(q));
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        const auto convection_contribution = current_convection
                          *fe_values.shape_grad(j, q);

                        cell_rhs(i) += fe_values.JxW(q)*
                          // mass and reaction part
                          (((1.0 - time_step/2.0*parameters.reaction_coefficient)
                            *fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                            - time_step/2.0*
                            // convection part
                            (fe_values.shape_value(i, q)*convection_contribution
                             // Laplacian part
                             + parameters.diffusion_coefficient
                             *(fe_values.shape_grad(i, q)*fe_values.shape_grad(j, q))))
                           *current_fe_coefficients[j]
                           // forcing parts
                           + time_step/2.0*
                           (current_forcing + previous_forcing)
                           *fe_values.shape_value(i, q));
                      }
                  }
              }
            constraints.distribute_local_to_global(cell_rhs, local_indices, system_rhs);
          }
      }
  }
}
#endif
