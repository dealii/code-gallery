/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2026 by Jonas Plank
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 */

#include "../include/random_darcy.h"
#include <deal.II/lac/sparse_direct.h>

template <int dim>
Discretization::RandomDarcy<dim>::RandomDarcy()
: fe(1)
, dof_handler(coarse_tria)
{}

template <int dim>
void Discretization::RandomDarcy<dim>::set_tria(bool fine)
{
    if(fine)
    {
        dof_handler.reinit(fine_tria);
    }
    else 
    {
        dof_handler.reinit(coarse_tria);
    }
}

template <int dim>
void Discretization::RandomDarcy<dim>::generate_mesh(double domain_length)
{
    GridGenerator::hyper_cube(coarse_tria, 0, domain_length, true);
    GridGenerator::hyper_cube(fine_tria, 0, domain_length, true);

    coarse_tria.refine_global(3);
    fine_tria.refine_global(3);      
}

template <int dim>
void Discretization::RandomDarcy<dim>::setup_system()
{
    dof_handler.distribute_dofs(fe);
 
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
    
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    // 1. Left boundary (indicator 0): u = 1.0
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ConstantFunction<dim>(1.0),
                                             constraints);

    // 2. Right boundary (indicator 1): u = 0.0
    VectorTools::interpolate_boundary_values(dof_handler,
                                             1,
                                             Functions::ConstantFunction<dim>(0.0),
                                             constraints);
    
    constraints.close();
    
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ false);
    
    sparsity_pattern.copy_from(dsp);
    
    system_matrix.reinit(sparsity_pattern);
}

template <int dim>
void Discretization::RandomDarcy<dim>::assemble_system(RandomField::RandomPermeability<dim>& permeability)
{
    const QGauss<dim> quadrature_formula(fe.degree + 1);
 
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
 
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
 
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
 
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
 
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
 
      cell_matrix = 0;
      cell_rhs    = 0;
 
      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          const double current_coefficient =
            permeability.value(fe_values.quadrature_point(q_index));
          for (const unsigned int i : fe_values.dof_indices())
            {
              for (const unsigned int j : fe_values.dof_indices())
                cell_matrix(i, j) +=
                  (current_coefficient *              // a(x_q)
                   fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                   fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                   fe_values.JxW(q_index));           // dx
 
              cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                              1.0 *                               // f(x)
                              fe_values.JxW(q_index));            // dx
            }
        }
 
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
    
}

template <int dim>
void Discretization::RandomDarcy<dim>::solve()
{
    SparseDirectUMFPACK A_direct;
 
    solution = system_rhs;
    A_direct.solve(system_matrix, solution);

    constraints.distribute(solution);
}

template <int dim>
void Discretization::RandomDarcy<dim>::refine_grid(bool firstRun)
{
    fine_tria.refine_global(1);
    if(!firstRun)
    {
        coarse_tria.refine_global(1);
    }
}

template <int dim>
void Discretization::RandomDarcy<dim>::output_results(bool firstRun, unsigned int level)
{
    DataOut<dim> data_out;
  
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
  
    data_out.build_patches();
  
    std::ofstream output(firstRun ? "solutionCoarse" + std::to_string(level) + ".vtk" : "solutionFine" + std::to_string(level) + ".vtk");
    data_out.write_vtk(output);
}

template <int dim>
double Discretization::RandomDarcy<dim>::compute_Keff(RandomField::RandomPermeability<dim>& permeability)
{
  const QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);
  FEFaceValues<dim> fe_face_values(fe,
                                       face_quadrature_formula,
                                       update_gradients | update_normal_vectors |
                                   update_JxW_values | update_quadrature_points);

  std::vector<Tensor<1, dim>> solution_gradients(face_quadrature_formula.size());
  double Keff = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    { for (const auto face_no : cell->face_indices())
      {
        if (cell->face(face_no)->at_boundary() &&
              (cell->face(face_no)->boundary_id() == 1))
            {
              fe_face_values.reinit(cell, face_no);
              fe_face_values.get_function_gradients(solution, solution_gradients);
              for (const unsigned int q_index : fe_face_values.quadrature_point_indices())
                {
                  const double current_coefficient = permeability.value(fe_face_values.quadrature_point(q_index));
                  Keff -= current_coefficient*solution_gradients[q_index][0]*fe_face_values.JxW(q_index); 
                }
            }

      }
    }

    return Keff;
}

template class Discretization::RandomDarcy<1>;
template class Discretization::RandomDarcy<2>;

