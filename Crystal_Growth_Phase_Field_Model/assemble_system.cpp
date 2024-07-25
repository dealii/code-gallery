/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
 * Copyright (C) 2024 by Umair Hussain
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 */

#include "PhaseFieldSolver.h"
#include <cmath>

void PhaseFieldSolver::assemble_system() {
    //Separating each variable as a scalar to easily call the respective shape functions
    FEValuesExtractors::Scalar phase_parameter(0);
    FEValuesExtractors::Scalar temperature(1);

    QGauss<2> quadrature_formula(fe.degree + 1);
    FEValues<2> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

    Vector<double>     cell_rhs(dofs_per_cell);

    //To copy values and gradients of solution from previous iteration
    //Old Newton iteration
    std::vector<Tensor<1, 2>> old_newton_solution_gradients_p(n_q_points);
    std::vector<double> old_newton_solution_values_p(n_q_points);
    std::vector<Tensor<1, 2>> old_newton_solution_gradients_t(n_q_points);
    std::vector<double> old_newton_solution_values_t(n_q_points);
    //Old time step iteration
    std::vector<Tensor<1, 2>> old_time_solution_gradients_p(n_q_points);
    std::vector<double> old_time_solution_values_p(n_q_points);
    std::vector<Tensor<1, 2>> old_time_solution_gradients_t(n_q_points);
    std::vector<double> old_time_solution_values_t(n_q_points);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    jacobian_matrix.operator=(0.0);
    system_rhs.operator=(0.0);

    for (const auto &cell : dof_handler.active_cell_iterators()){
        if (cell->subdomain_id() == this_mpi_process) {
            cell_matrix = 0;
            cell_rhs = 0;

            fe_values.reinit(cell);

            //Copying old solution values
            fe_values[phase_parameter].get_function_values(conv_solution_np,old_newton_solution_values_p);
            fe_values[phase_parameter].get_function_gradients(conv_solution_np,old_newton_solution_gradients_p);
            fe_values[temperature].get_function_values(conv_solution_np,old_newton_solution_values_t);
            fe_values[temperature].get_function_gradients(conv_solution_np,old_newton_solution_gradients_t);
            fe_values[phase_parameter].get_function_values(old_solution_np,old_time_solution_values_p);
            fe_values[phase_parameter].get_function_gradients(old_solution_np,old_time_solution_gradients_p);
            fe_values[temperature].get_function_values(old_solution_np,old_time_solution_values_t);
            fe_values[temperature].get_function_gradients(old_solution_np,old_time_solution_gradients_t);

            for (unsigned int q = 0; q < n_q_points; ++q){
                double khi = get_random_number();
                //Old solution values
                double p_on = old_newton_solution_values_p[q]; //old newton solution
                auto grad_p_on = old_newton_solution_gradients_p[q];
                double p_ot = old_time_solution_values_p[q]; //old time step solution
                auto grad_p_ot = old_time_solution_gradients_p[q];
                double t_on = old_newton_solution_values_t[q];
                auto grad_t_on = old_newton_solution_gradients_t[q];
                double t_ot = old_time_solution_values_t[q];
                auto grad_t_ot = old_time_solution_gradients_t[q];
                for (unsigned int i = 0; i < dofs_per_cell; ++i){
                    //Shape Functions
                    double psi_i = fe_values[phase_parameter].value(i,q);
                    auto grad_psi_i = fe_values[phase_parameter].gradient(i,q);
                    double phi_i = fe_values[temperature].value(i,q);
                    auto grad_phi_i = fe_values[temperature].gradient(i,q);
                    for (unsigned int j = 0; j < dofs_per_cell; ++j){
                        //Shape Functions
                        double psi_j = fe_values[phase_parameter].value(j,q);
                        auto grad_psi_j = fe_values[phase_parameter].gradient(j,q);
                        double phi_j = fe_values[temperature].value(j,q);
                        auto grad_phi_j = fe_values[temperature].gradient(j,q);

                        double mp = psi_i*(tau*psi_j);
                        double kp = grad_psi_i*(std::pow(epsilon,2)*grad_psi_j);
                        double m  = (alpha/M_PI)*std::atan(gamma*(t_eq - t_on));
                        double t1 = (1-p_on)*(p_on-0.5+m);
                        double t2 = -(p_on)*(p_on-0.5+m);
                        double t3 = (p_on)*(1-p_on);
                        double nl_p = psi_i*((t1+t2+t3)*psi_j);
                        //Adding random noise at the interface
                        nl_p -= a*khi*psi_i*((1.0 - 2*(p_on))*psi_j);
                        double f1_p= mp + time_step*theta*kp - time_step*theta*nl_p; // doh f1 by doh p (first Jacobian terms)

                        double t4 = (p_on)*(1-p_on)*(-(alpha*gamma/(M_PI*(1+std::pow((gamma*(t_eq-t_on)),2)))));
                        double nl_t = psi_i*(t4*phi_j);
                        double f1_t = -time_step*theta*nl_t; // doh f1 by doh t (second Jacobian terms)

                        double mpt = phi_i*(latent_heat*psi_j);
                        double f2_p = -mpt; // doh f2 by doh p (third Jacobian terms)

                        double mt = phi_i*(phi_j);
                        double kt = grad_phi_i*(grad_phi_j);
                        double f2_t = mt + time_step*theta*kt; // doh f2 by doh t (fourth Jacobian terms)

                        //Assembling Jacobian matrix
                        cell_matrix(i,j) += (f1_p + f1_t + f2_p + f2_t)*fe_values.JxW(q);

                    }
                    //Finding f1 and f2 at previous iteration for rhs vector
                    double mp_n = psi_i*(tau*p_on);
                    double kp_n = grad_psi_i*(std::pow(epsilon,2)*grad_p_on);
                    double m_n = (alpha/M_PI)*std::atan(gamma*(t_eq-t_on));
                    double nl_n = psi_i*((p_on)*(1-p_on)*(p_on-0.5+m_n));
                    double mp_t = psi_i*(tau*p_ot);
                    double kp_t = grad_psi_i*(tau*grad_p_ot);
                    double m_t = (alpha/M_PI)*std::atan(gamma*(t_eq-t_ot));
                    double nl_t = psi_i*(p_ot)*(1-p_ot)*(p_ot-0.5+m_t);
                    //Adding random noise at the interface
                    nl_n -= psi_i*(a*khi*(p_on)*(1-p_on));
                    nl_t -= psi_i*(a*khi*(p_ot)*(1-p_ot));

                    double f1n = mp_n + time_step*theta*kp_n - time_step*theta*nl_n - mp_t + time_step*(1-theta)*kp_t - time_step*(1-theta)*nl_t; //f1 at last newton iteration

                    double mt_n = phi_i*(t_on);
                    double kt_n = grad_phi_i*(grad_t_on);
                    double mpt_n = phi_i*(latent_heat*p_on);
                    double mt_t = phi_i*(t_ot);
                    double kt_t = grad_phi_i*(grad_t_ot);
                    double mpt_t = phi_i*(latent_heat*p_ot);

                    double f2n = mt_n + time_step*theta*kt_n - mpt_n - mt_t + time_step*(1-theta)*kt_t + mpt_t; //f2 at last newton iteration

                    //Assembling RHS vector
                    cell_rhs(i) -= (f1n + f2n)*fe_values.JxW(q);
                }
            }

            cell->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    jacobian_matrix.add(local_dof_indices[i],
                                        local_dof_indices[j],
                                        cell_matrix(i, j));
                system_rhs(local_dof_indices[i]) += cell_rhs(i);
            }
        }
    }

    jacobian_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    //Applying zero BC
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             1,
                                             Functions::ZeroFunction<2>(2),
                                             boundary_values);
    MatrixTools::apply_boundary_values(boundary_values,
                                       jacobian_matrix,
                                       solution_update,
                                       system_rhs, false);

    jacobian_matrix.compress(VectorOperation::insert);

    system_rhs.compress(VectorOperation::insert);
}
