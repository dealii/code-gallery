/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2026 by Jonas Plank
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 */
#pragma once
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/base/function.h>
#include <fstream>

#include "random_permeability.h"

namespace Discretization
{
    using namespace dealii;

    /**
     * This class is essentially a modification of Step-5. As described in the README,
     * we treat the finite element solver as a black box. This class demonstrates 
     * how easily this can be achieved; the only required modifications are 
     * input/output adjustments and the interface to the random field.
     */
    template <int dim>
    class RandomDarcy
    {
    public:
        RandomDarcy();                                                                   

        // Allows switching between coarse and fine triangulations 
        // with virtually no code duplication.
        void set_tria(bool fine);                                                       

        void generate_mesh(double domain_length);

        // Implementation follows the logic of deal.II tutorial Step-5.                                         
        void setup_system();                                                            

        void assemble_system(RandomField::RandomPermeability<dim>& random_constant); 

        /** 
         * Using an iterative solver for MLMC is generally not ideal.
         * The matrices can become highly ill-conditioned due to the high 
         * variations in the random permeability field.
         */
        void solve();

        // If we are on the first level, we only want to refine the fine triangulation.                                                                           
        void refine_grid(bool firstRun);  

        // On the first run, we label the output as "coarse" since both meshes 
        // are identical. After the first level, the meshes diverge, and we 
        // output the finer mesh.
        void output_results(bool firstRun, unsigned int level);                             

        // This is our Quantity of Interest (QoI).
        // It is defined as: $K_{eff} = - \int_{\Gamma_{right}} k \frac{\partial p}{\partial x_1} dx_2$
        double compute_Keff(RandomField::RandomPermeability<dim>& permeability);

        private:
        // define two triangulations
        Triangulation<dim> coarse_tria;
        Triangulation<dim> fine_tria;

        const FE_Q<dim> fe;
        DoFHandler<dim> dof_handler;

        AffineConstraints<double> constraints;

        SparseMatrix<double> system_matrix;
        SparsityPattern      sparsity_pattern;
        
        Vector<double> solution;
        Vector<double> system_rhs;
    };
}
