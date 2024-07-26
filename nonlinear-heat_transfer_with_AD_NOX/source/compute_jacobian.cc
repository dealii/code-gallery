/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
 * Copyright (C) 2024 by Narasimhan Swaminathan
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 */

#include "allheaders.h"
#include "nonlinear_heat.h"
/**
 * This function sets up the residual in a format to allow automatic differentiation and also calculates the Jacobian.
 * We need to calculate the residual again because, we want to use the TRILINOS wrappers based NOX to solve the nonlinear set.
 * This is not a serious problem, because NOX would call this function only when it needs to and not every iteration. Most of the
 * documentation is exactly as in the copmute_residual() function, except a few which are documented below.
 */
void nonlinear_heat::compute_jacobian(const Vector<double> &evaluation_point)
{
    const QGauss<2> quadrature_formula(
            fe.degree + 1);

    const QGauss<1> face_quadrature_formula(fe.degree+1);

    FEValues<2> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                          update_quadrature_points |
                          update_JxW_values);

    FEFaceValues<2> fe_face_values(fe,face_quadrature_formula,update_values|update_quadrature_points|
                                                              update_normal_vectors | update_JxW_values);


    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_q_f_points = face_quadrature_formula.size();
    Vector<double> cell_rhs(dofs_per_cell);
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell); /*!<Defining a local (Stiffness/Jacobian) matrix*/
    using ADHelper = Differentiation::AD::ResidualLinearization<Differentiation::AD::NumberTypes::sacado_dfad,double>;
    using ADNumberType = typename ADHelper::ad_type;
    const FEValuesExtractors::Scalar t(0);
    /**
     * #system_matrix will hold the numerical value of the Jacobian (evaluated at #evaluation_points).
     */
    system_matrix = 0.0;
    std::vector<types::global_dof_index> local_dof_indices(
            dofs_per_cell);
    /*==================================================================*/
    std::vector<double> consol(n_q_points);
    std::vector<ADNumberType> old_solution(
            n_q_points);

    std::vector<Tensor<1, 2>> consol_grad(
            n_q_points);
    std::vector<Tensor<1, 2,ADNumberType>> old_solution_grad(
            n_q_points);

    for (const auto &cell: dof_handler.active_cell_iterators())
    {
        cell_rhs = 0;
        cell_matrix = 0;
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        const unsigned int n_independent_variables = local_dof_indices.size();
        const unsigned int n_dependent_variables = dofs_per_cell;
        ADHelper ad_helper(n_independent_variables, n_dependent_variables);
        ad_helper.register_dof_values(evaluation_point,local_dof_indices);
        const std::vector<ADNumberType> &dof_values_ad = ad_helper.get_sensitive_dof_values();
        fe_values[t].get_function_values_from_local_dof_values(dof_values_ad,
                                         old_solution);
        fe_values[t].get_function_gradients_from_local_dof_values(dof_values_ad,
                                            old_solution_grad);

        fe_values[t].get_function_values(converged_solution, consol);
        fe_values[t].get_function_gradients(converged_solution, consol_grad);
        std::vector<ADNumberType> residual_ad(n_dependent_variables,
                                              ADNumberType(0.0));
        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {

                ADNumberType MijTjcurr = Cp*rho*fe_values[t].value(i,q_index)*old_solution[q_index];
                ADNumberType MijTjprev = Cp*rho*fe_values[t].value(i,q_index)*consol[q_index];
                ADNumberType k_curr = a + b*old_solution[q_index] + c*std::pow(old_solution[q_index],2);
                ADNumberType k_prev = a + b*consol[q_index] + c*std::pow(consol[q_index],2);
                ADNumberType Licurr =  alpha * delta_t *  (fe_values[t].gradient(i,q_index)*k_curr*old_solution_grad[q_index]);
                ADNumberType Liprev =  (1-alpha) * delta_t *  (fe_values[t].gradient(i,q_index)*k_prev*consol_grad[q_index]);
                residual_ad[i] +=  (MijTjcurr+Licurr-MijTjprev+Liprev)*fe_values.JxW(q_index);
            }
        }


        for (unsigned int face_number = 0;face_number<GeometryInfo<2>::faces_per_cell; ++face_number)
        {

            if (cell->face(face_number)->boundary_id() == 3)
            {
                fe_face_values.reinit(cell, face_number);
                for (unsigned int q_point=0;q_point<n_q_f_points;++q_point)
                {
                    for (unsigned int i =0;i<dofs_per_cell;++i)
                        residual_ad[i]+= -delta_t*(-10)*fe_face_values[t].value(i,q_point)*fe_face_values.JxW(q_point);

                }
            }
        }

        ad_helper.register_residual_vector(residual_ad);
        ad_helper.compute_residual(cell_rhs);
        /**
         * In this step, we calculate the <b> local<b> jacobian in numerical form.
         */
        ad_helper.compute_linearization(cell_matrix);
        cell->get_dof_indices(local_dof_indices);
        /**
         * The following loop assembles it to the #system_matrix.
         */

        for (unsigned int i =0;i < dofs_per_cell; ++i){
            for(unsigned int j = 0;j < dofs_per_cell;++j){
                system_matrix.add(local_dof_indices[i],local_dof_indices[j],cell_matrix(i,j));
            }
        }
    }
    /**
     * The following lines applies the boundary conditions to the problem. That is, wherever the Dirichlet boundary
     * values are defined, those rows and columns of the jacobian are removed.
     */
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             1,
                                             Functions::ZeroFunction<2>(1),
                                             boundary_values);

    Vector<double> dummy_solution(dof_handler.n_dofs());
    Vector<double> dummy_rhs(dof_handler.n_dofs());
    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       dummy_solution,
                                       dummy_rhs);

    {
        std::cout << "  Factorizing Jacobian matrix" << std::endl;
        matrix_factorization = std::make_unique<SparseDirectUMFPACK>();
        matrix_factorization->factorize(system_matrix);
    }
}
