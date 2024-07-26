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
 * This function sets up the residual in a format to allow automatic differentiation to calculate the Jacobian.
 * #evaluation_point is the point (solution) where we need to evaluate this residual.
 * @param evaluation_point Point where the residual is to be evaluated.
 * @param residual The residual vector to be used in the non-linear solution process.
 */
void nonlinear_heat::compute_residual(const Vector<double> & evaluation_point, Vector<double> & residual)
{
    /**
     * The following lines should be clear. We need the FEFaceValues<2> definition, because, we want to apply Newmann boundary conditions
     * to the right end.
     */
    const QGauss<2> quadrature_formula(
            fe.degree + 1);/**< Define a quadrature to perform the integration over the 2D finite element */
    const QGauss<1> face_quadrature_formula(fe.degree+1); //Define quadrature for integration over faces */
    FEValues<2> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                          update_quadrature_points |
                          update_JxW_values); /*!< Define what aspects of inside the finite element you need for this problem*/

    FEFaceValues<2> fe_face_values(fe,face_quadrature_formula,update_values|update_quadrature_points|
                                                              update_normal_vectors | update_JxW_values);
    const unsigned int dofs_per_cell = fe.dofs_per_cell; /*!< Number of degree of freedom per cell*/
    const unsigned int n_q_points = quadrature_formula.size();/*!< Number of quadrature points over the domain of the finite element*/
    const unsigned int n_q_f_points = face_quadrature_formula.size();/*!< Number of quadrature point over the boundary of a finite element*/
    /**
     * #cell_rhs holds the local rhs. That is the residual evaluated at the
     * #evaluation_point.
     */
    Vector<double> cell_rhs(dofs_per_cell); /*!<Defining a local (residual) vector*/

    /**
     * Here we define the type of <code>Number<code> we want to use to define our variables
     * so that they are suitable for automatic differentiation.
     */
    using ADHelper = Differentiation::AD::ResidualLinearization<Differentiation::AD::NumberTypes::sacado_dfad,double>;
    using ADNumberType = typename ADHelper::ad_type;
    /**
     * The FEValuesExtractors is used as usual to get the degree of freedom we are interested in.
     * For this single variable problem, this may not be needed. We still use it to show
     * how it needs to be declared, when multiple variables exist and when we want to define component mask
     * to define boundary conditions for specific variables.
     */
    const FEValuesExtractors::Scalar t(
            0);
    std::vector<types::global_dof_index> local_dof_indices(
            dofs_per_cell); /*!< Local to Global Degree of freedom indices */
    /**
     * #consol now holds the #converged_solution at the Gauss points of a current cell. These are of the double type i.e., regular numbers.
     */

    std::vector<double> consol(n_q_points); /* Converged solution at the Gauss points from the previous time step*/
    std::vector<ADNumberType> old_solution(
            n_q_points); /* Current solution at the Gauss points at this iteration for the current time*/
    /**
     * #concol_grad now holds the #converged_solution at the Gauss points of the current cell. These are regular numbers.
     */
    std::vector<Tensor<1, 2>> consol_grad(
            n_q_points); /* Converged gradients of the solutions at the Gauss points from the previous time step */
    std::vector<Tensor<1, 2,ADNumberType>> old_solution_grad(
            n_q_points);

    ComponentMask t_mask = fe.component_mask(t);
    /**
     * Actual, numerical residual.
     */
    residual = 0.0;
    for (const auto &cell: dof_handler.active_cell_iterators())
    {
        cell_rhs = 0;
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        const unsigned int n_independent_variables = local_dof_indices.size();
        const unsigned int n_dependent_variables = dofs_per_cell;
        ADHelper ad_helper(n_independent_variables, n_dependent_variables);
        /**
         * It is in the following line, we say that the #evalaution_point needs to be substituted in
         * place of the #old_solution (or #old_solution_grad) to get the #residual (in numerical form).
         */
        ad_helper.register_dof_values(evaluation_point,local_dof_indices);

        const std::vector<ADNumberType> &dof_values_ad = ad_helper.get_sensitive_dof_values();
        fe_values[t].get_function_values_from_local_dof_values(dof_values_ad,
                                         old_solution);
        fe_values[t].get_function_gradients_from_local_dof_values(dof_values_ad,
                                            old_solution_grad);
        /**
         * In the following steps, #consol and #consol_grad are used to grab value of the #converged_solution
         * and its gradient at the gauss points, respectively.
         */
        fe_values[t].get_function_values(converged_solution, consol);
        fe_values[t].get_function_gradients(converged_solution, consol_grad);
        /**
         * residual_ad is defined and initialized in its symbolic form.
         */
        std::vector<ADNumberType> residual_ad(n_dependent_variables,
                                              ADNumberType(0.0));
        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                /**
                 * We here define the entire residual using all intermediate variables, which are also of the type ADNumberType.
                 */
                ADNumberType MijTjcurr = Cp*rho*fe_values[t].value(i,q_index)*old_solution[q_index];
                ADNumberType MijTjprev = Cp*rho*fe_values[t].value(i,q_index)*consol[q_index];
                ADNumberType k_curr = a + b*old_solution[q_index] + c*std::pow(old_solution[q_index],2);
                ADNumberType k_prev = a + b*consol[q_index] + c*std::pow(consol[q_index],2);
                ADNumberType Licurr =  alpha * delta_t *  (fe_values[t].gradient(i,q_index)*k_curr*old_solution_grad[q_index]);
                ADNumberType Liprev =  (1-alpha) * delta_t *  (fe_values[t].gradient(i,q_index)*k_prev*consol_grad[q_index]);
                residual_ad[i] +=  (MijTjcurr+Licurr-MijTjprev+Liprev)*fe_values.JxW(q_index);
            }
        }
            /**
             * The following lines, apply the Newmann boundary conditions to the right hand side.
             */

        for (unsigned int face_number = 0;face_number<GeometryInfo<2>::faces_per_cell; ++face_number)
        {

            if (cell->face(face_number)->boundary_id() == 3)
            {
                fe_face_values.reinit(cell, face_number);
                for (unsigned int q_point=0;q_point<n_q_f_points;++q_point)
                {
                    for (unsigned int i =0;i<dofs_per_cell;++i)
                        /**
                         * The Newumann boundary condition (-10) is applied to the right edge, with boundary id 3.
                         */
                        residual_ad[i]+= -delta_t*(-10)*fe_face_values[t].value(i,q_point)*fe_face_values.JxW(q_point);

                }
            }
        }
        /**
         * Here, we tell the ad_helper that the residual (in its symbolic form) is given by #residual_ad.
         */
        ad_helper.register_residual_vector(residual_ad);
        /**
         * Here, the residual is calculated at the values given by #evaluation_point.
         * Note that the residual does not have the -ve sign as in the regular way of solving.
         * This is because, the NOX solvers, only needs the actual reisdual.
         */
        ad_helper.compute_residual(cell_rhs);
        cell->get_dof_indices(local_dof_indices);

        for (unsigned int i =0;i < dofs_per_cell; ++i)
            residual(local_dof_indices[i])+= cell_rhs(i);

    }

    for(const types::global_dof_index i: DoFTools::extract_boundary_dofs(dof_handler,t_mask,{1}))
        residual(i) = 0;

    std::cout << " The Norm is :: = " << residual.l2_norm() << std::endl;
}
