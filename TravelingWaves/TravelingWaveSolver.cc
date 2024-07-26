/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
 * Copyright (C) 2024 by Shamil Magomedov
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 */

#include "TravelingWaveSolver.h"

namespace TravelingWave
{
  using namespace dealii;

  // Constructor of the class that takes parameters of the problem and an initial guess for Newton's iterations.
  TravelingWaveSolver::TravelingWaveSolver(const Parameters &parameters, const SolutionStruct &initial_guess_input) 
    : params(parameters)
    , problem(params.problem)
    , number_of_quadrature_points((params.fe.quadrature_points_number > 0) ? params.fe.quadrature_points_number : (params.fe.poly_degree + 1))
    , triangulation_uploaded(false)
    , fe(FE_Q<1>(params.fe.poly_degree), 1, 
          FE_Q<1>(params.fe.poly_degree), 1,
          FE_Q<1>(params.fe.poly_degree), 1)   // 3 fe basis sets, corresponding to du, dT, dlambda
    , dof_handler(triangulation)
    , current_wave_speed(0.)
    , initial_guess(initial_guess_input)
    , computing_timer(std::cout, TimerOutput::never, TimerOutput::wall_times)
  {
    // Table with values of some parameters to be written to the standard output before calculations.
    TableHandler table;
    table.add_value("Parameter name", "number of quadrature points");
    table.add_value("value", number_of_quadrature_points);

    table.add_value("Parameter name", "delta");
    table.add_value("value", params.problem.delta);

    table.add_value("Parameter name", "epsilon");
    table.add_value("value", params.problem.epsilon);
    
    table.add_value("Parameter name", "params.problem.wave_speed_init");
    table.add_value("value", params.problem.wave_speed_init);

    table.add_value("Parameter name", "initial_guess.wave_speed");
    table.add_value("value", initial_guess.wave_speed);
    
    table.add_value("Parameter name", "T_left");
    table.add_value("value", params.problem.T_left);
    
    table.set_precision("value", 2);
    table.set_scientific("value", true);
    
    std::cout << "\n";
    table.write_text(std::cout, TableHandler::TextOutputFormat::org_mode_table);
    std::cout << "\n";
  }

  // A function that takes a triangulation and assigns it to the member variable <code>triangulation </code>.
  void TravelingWaveSolver::set_triangulation(const Triangulation<1> &itriangulation)
  {
    triangulation.clear();
    triangulation.copy_triangulation(itriangulation);
    triangulation_uploaded = true;
  }

  // Here we find the indices of the degrees of freedom, associated with the boundary vertices, and the degree of freedom, associated with the vertex with coordinate $\xi = 0$, and corresponding to temperature.
  void TravelingWaveSolver::find_boundary_and_centering_dof_numbers()
  {
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      for (const auto &v_ind : cell->vertex_indices())
      {
        if (isapprox(cell->vertex(v_ind)[0], params.mesh.interval_left))
        {
          boundary_and_centering_dof_numbers["u_left"]        = cell->vertex_dof_index(v_ind, 0);
          boundary_and_centering_dof_numbers["T_left"]        = cell->vertex_dof_index(v_ind, 1);
          boundary_and_centering_dof_numbers["lambda_left"]   = cell->vertex_dof_index(v_ind, 2);
        }
        else if (isapprox(cell->vertex(v_ind)[0], params.mesh.interval_right))
        {
          boundary_and_centering_dof_numbers["u_right"]       = cell->vertex_dof_index(v_ind, 0);
          boundary_and_centering_dof_numbers["T_right"]       = cell->vertex_dof_index(v_ind, 1);
          boundary_and_centering_dof_numbers["lambda_right"]  = cell->vertex_dof_index(v_ind, 2);
        }
        else if (isapprox(cell->vertex(v_ind)[0], 0.))
        {
          boundary_and_centering_dof_numbers["T_zero"]        = cell->vertex_dof_index(v_ind, 1);
        }
      }
    }
  }

  // Set solution values, corresponding to Dirichlet boundary conditions and the centering condition $T(0) = T_{\mathrm{ign}}$.
  void TravelingWaveSolver::set_boundary_and_centering_values()
  {
    current_solution[boundary_and_centering_dof_numbers["u_right"]] = problem.u_right;

    current_solution[boundary_and_centering_dof_numbers["T_left"]] = problem.T_left;
    if (problem.T_r_bc_type == 1)	// 1 for "Dirichlet"
    {
      current_solution[boundary_and_centering_dof_numbers["T_right"]] = problem.T_right;
    } // else is 0 for "Neumann"
    current_solution[boundary_and_centering_dof_numbers["T_zero"]] = problem.T_ign;

    current_solution[boundary_and_centering_dof_numbers["lambda_right"]] = 0.;
  }


  void TravelingWaveSolver::setup_system(const bool initial_step)
  {
    TimerOutput::Scope t(computing_timer, "set up");

    dof_handler.distribute_dofs(fe);

    std::cout << "Number of dofs : " << dof_handler.n_dofs() << std::endl;

    extended_solution_dim = dof_handler.n_dofs() + 1;

    find_boundary_and_centering_dof_numbers();

    // Boundary condition constraints for $du$, $dT$ and $d\lambda$.
    zero_boundary_constraints.clear();

    // Dirichlet homogeneous boundary condition for $du$ at the right boundary.
    zero_boundary_constraints.add_line(boundary_and_centering_dof_numbers["u_right"]);

    // Dirichlet homogeneous boundary condition for $dT$ at the left boundary.
    zero_boundary_constraints.add_line(boundary_and_centering_dof_numbers["T_left"]);
    // For the temperature at the left boundary there are two possibilities:
    if (problem.T_r_bc_type == 1)	// 1 for "Dirichlet"
    {
      std::cout << "Dirichlet condition for the temperature at the right boundary." << std::endl;
      zero_boundary_constraints.add_line(boundary_and_centering_dof_numbers["T_right"]);
    } // else is 0 for "Neumann"
    else
    {
      std::cout << "Neumann condition for the temperature at the right boundary." << std::endl;
    }

    // Dirichlet homogeneous boundary condition for $d\lambda$ at the right boundary. (At the left boundary we consider the homogeneous Neumann boundary condition for $d\lambda$.)
    zero_boundary_constraints.add_line(boundary_and_centering_dof_numbers["lambda_right"]);
    
    zero_boundary_constraints.close();

    // We create extended dynamic sparsity pattern with an additional row and an additional column.
    DynamicSparsityPattern dsp(extended_solution_dim);
    {
      std::vector<types::global_dof_index> dofs_on_this_cell;
      dofs_on_this_cell.reserve(dof_handler.get_fe_collection().max_dofs_per_cell());

      for (const auto &cell : dof_handler.active_cell_iterators())
      {
        const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
        dofs_on_this_cell.resize(dofs_per_cell);
        cell->get_dof_indices(dofs_on_this_cell);

        zero_boundary_constraints.add_entries_local_to_global(dofs_on_this_cell,
                              dsp,
                              /*keep_constrained_dofs*/ true);
      }

      // Adding elements to the last column.
      for (unsigned int i = 0; i < extended_solution_dim; ++i)
      {
        dsp.add(i, extended_solution_dim - 1);
      }
      // Adding one element to the last row, corresponding to the T(0).
      dsp.add(extended_solution_dim - 1, boundary_and_centering_dof_numbers["T_zero"]);
    }
    
    // Initialization
    sparsity_pattern_extended.copy_from(dsp);
    jacobian_matrix_extended.reinit(sparsity_pattern_extended); 
    jacobian_matrix_extended_factorization.reset();

    current_solution_extended.reinit(extended_solution_dim);

    if (initial_step)
    {
      current_solution.reinit(dof_handler.n_dofs());
    }

  }


  void TravelingWaveSolver::set_initial_guess()
  {
    current_wave_speed = initial_guess.wave_speed;

    // The initial condition is a discrete set of coordinates $\xi$ and values of functions $u$, $T$ and $\lambda$. From the three sets we create three continuous functions using interpolation, which then form one continuous vector function of <code> SolutionVectorFunction </code> type.
    Interpolant u_interpolant(initial_guess.x, initial_guess.u);
    Interpolant T_interpolant(initial_guess.x, initial_guess.T);
    Interpolant lambda_interpolant(initial_guess.x, initial_guess.lambda);

    SolutionVectorFunction init_guess_func(u_interpolant, T_interpolant, lambda_interpolant);

    VectorTools::interpolate(dof_handler, init_guess_func, current_solution); 

    set_boundary_and_centering_values();

    for (unsigned int i = 0; i < extended_solution_dim - 1; ++i)
    {
      current_solution_extended(i) = current_solution(i);
    }
    current_solution_extended(extended_solution_dim - 1) = current_wave_speed;
  }

  // Heaviside function.
  double TravelingWaveSolver::Heaviside_func(double x) const
  {
    if (x > 0) 
    {
      return 1.;
    }
    else
    {
      return 0.;
    }
  }


  void TravelingWaveSolver::compute_and_factorize_jacobian(const Vector<double> &evaluation_point_extended)
  {
    {
      TimerOutput::Scope t(computing_timer, "assembling the Jacobian");

      Vector<double> evaluation_point(dof_handler.n_dofs());
      for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
      {
        evaluation_point(i) = evaluation_point_extended(i);
      }

      const double wave_speed = evaluation_point_extended(extended_solution_dim - 1);

      std::cout << "Computing Jacobian matrix ... " << std::endl;
  
      const QGauss<1> quadrature_formula(number_of_quadrature_points);
  
      jacobian_matrix_extended = 0;

      FEValues<1> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients | 
                            update_quadrature_points | update_JxW_values);

      const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
      const unsigned int n_q_points    = quadrature_formula.size();

      FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
      Vector<double> row_last_element_vector(dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
      
      const FEValuesExtractors::Scalar velocity(0);
      const FEValuesExtractors::Scalar temperature(1);
      const FEValuesExtractors::Scalar lambda(2);

      std::vector<double> current_velocity_values(n_q_points);
      std::vector<double> current_temperature_values(n_q_points);
      std::vector<double> current_lambda_values(n_q_points);

      std::vector<Tensor<1, 1>> current_velocity_gradients(n_q_points);
      std::vector<Tensor<1, 1>> current_temperature_gradients(n_q_points);
      std::vector<Tensor<1, 1>> current_lambda_gradients(n_q_points);

      std::vector<double> phi_u(dofs_per_cell);
      std::vector<Tensor<1, 1>> grad_phi_u(dofs_per_cell);
      std::vector<double> phi_T(dofs_per_cell);
      std::vector<Tensor<1, 1>> grad_phi_T(dofs_per_cell);
      std::vector<double> phi_lambda(dofs_per_cell);
      std::vector<Tensor<1, 1>> grad_phi_lambda(dofs_per_cell);

      for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;
        row_last_element_vector = 0;

        fe_values.reinit(cell);

        fe_values[velocity].get_function_values(evaluation_point, current_velocity_values);
        fe_values[temperature].get_function_values(evaluation_point, current_temperature_values);
        fe_values[lambda].get_function_values(evaluation_point, current_lambda_values);

        fe_values[velocity].get_function_gradients(evaluation_point, current_velocity_gradients);
        fe_values[temperature].get_function_gradients(evaluation_point, current_temperature_gradients);
        fe_values[lambda].get_function_gradients(evaluation_point, current_lambda_gradients);

        auto kappa_1 = [=](double T, double lambda){
          return problem.k * (1 - lambda) * std::exp(-problem.theta / T) * (
            problem.theta / (T * T) * Heaviside_func(T - problem.T_ign) /* + Delta_function(T - problem.T_ign) */
          );
        };

        auto kappa_2 = [=](double T, double lambda){
          return -problem.k * std::exp(-problem.theta / T) * Heaviside_func(T - problem.T_ign);
        };

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
          for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            phi_u[k]            = fe_values[velocity].value(k, q);
            grad_phi_u[k]       = fe_values[velocity].gradient(k, q);
            phi_T[k]            = fe_values[temperature].value(k, q);
            grad_phi_T[k]       = fe_values[temperature].gradient(k, q);
            phi_lambda[k]       = fe_values[lambda].value(k, q);
            grad_phi_lambda[k]  = fe_values[lambda].gradient(k, q);
          }

          const double del_Pr_eps = (problem.Pr * 4 * problem.delta / (3 * problem.epsilon));
          const double del_Le = (problem.delta / problem.Le);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              cell_matrix(i, j) += (

                del_Pr_eps * (-grad_phi_u[i] * grad_phi_u[j])
                + phi_u[i] * (
                  - (1 - wave_speed + problem.epsilon * current_velocity_values[q]) * grad_phi_u[j][0]
                  - problem.epsilon * current_velocity_gradients[q][0] * phi_u[j]
                  - problem.epsilon / 2. * grad_phi_T[j][0]
                )
                
                + problem.delta * (-grad_phi_T[i] * grad_phi_T[j])
                + phi_T[i] * (
                  - wave_speed * grad_phi_u[j][0]
                  + wave_speed * grad_phi_T[j][0]
                  + problem.q * kappa_1(current_temperature_values[q], current_lambda_values[q]) * phi_T[j]
                  + problem.q * kappa_2(current_temperature_values[q], current_lambda_values[q]) * phi_lambda[j]
                )

                + del_Le * (-grad_phi_lambda[i] * grad_phi_lambda[j])
                + phi_lambda[i] * (
                  kappa_1(current_temperature_values[q], current_lambda_values[q]) * phi_T[j]
                  + wave_speed * grad_phi_lambda[j][0]
                  + kappa_2(current_temperature_values[q], current_lambda_values[q]) * phi_lambda[j]
                )

              ) * fe_values.JxW(q);

            }

            row_last_element_vector(i) += (
              (phi_u[i] * current_velocity_gradients[q][0])
              + (phi_T[i] * current_temperature_gradients[q][0])
              - (phi_T[i] * current_velocity_gradients[q][0])
              + (phi_lambda[i] * current_lambda_gradients[q][0])
            ) * fe_values.JxW(q);
          }

        }
  
        cell->get_dof_indices(local_dof_indices);
        
        for (const unsigned int i : fe_values.dof_indices())
        {
          for (const unsigned int j : fe_values.dof_indices())
          {
            jacobian_matrix_extended.add(local_dof_indices[i],
                                          local_dof_indices[j],
                                          cell_matrix(i, j));
          }
          
          // Adding elements to the last column.
          jacobian_matrix_extended.add(local_dof_indices[i],
                                        extended_solution_dim - 1,
                                        row_last_element_vector(i));
        }

      }

      // Global dof indices of dofs for $dT$ and $d\lambda$, associated with vertex $\xi = 0$.
      types::global_dof_index T_zero_point_dof_ind(0), lambda_zero_point_dof_ind(0);	
      
      // Approximating the derivative of $T$ at $\xi = 0$ as done in step-14.
      double T_point_derivative(0.);
      double T_at_zero_point(0.);
      double lambda_at_zero_point(0.);
      {
        double derivative_evaluation_point = 0.; 	// Point at which T = T_ign.

        const QTrapezoid<1> quadrature_formula;
        FEValues<1>	fe_values(fe,
                              quadrature_formula,
                              update_values | update_gradients | update_quadrature_points);

        const FEValuesExtractors::Scalar temperature(1);
        const FEValuesExtractors::Scalar lambda(2);
        
        const unsigned int n_q_points = quadrature_formula.size();
        std::vector<double> current_temperature_values(n_q_points);
        std::vector<Tensor<1, 1>> current_temperature_gradients(n_q_points);
        std::vector<double> current_lambda_values(n_q_points);
        
        unsigned int derivative_evaluation_point_hits = 0;

        for (const auto &cell : dof_handler.active_cell_iterators())
        {
          for (const auto &vertex : cell->vertex_indices())
          {
            if (isapprox(cell->vertex(vertex)[0], derivative_evaluation_point))
            {
              T_zero_point_dof_ind = cell->vertex_dof_index(vertex, 1);
              lambda_zero_point_dof_ind = cell->vertex_dof_index(vertex, 2);

              fe_values.reinit(cell);
              fe_values[temperature].get_function_values(current_solution, current_temperature_values);
              fe_values[temperature].get_function_gradients(current_solution, current_temperature_gradients);
              fe_values[lambda].get_function_values(current_solution, current_lambda_values);

              unsigned int q_point = 0;
              for (; q_point < n_q_points; ++q_point)
              {
                if (isapprox(fe_values.quadrature_point(q_point)[0], derivative_evaluation_point))
                {
                  break;
                }
              }

              T_at_zero_point = current_temperature_values[q_point];
              lambda_at_zero_point = current_lambda_values[q_point];

              T_point_derivative += current_temperature_gradients[q_point][0];
              ++derivative_evaluation_point_hits;

              break;
            }
          }
        }
        T_point_derivative /= static_cast<double>(derivative_evaluation_point_hits);
      }

      // Here we add to the matrix the terms that appear after integrating the terms with the Dirac delta function (which we skipped inside the loop).
      double term_with_delta_func(0.);
      term_with_delta_func = problem.k * std::exp(-problem.theta / T_at_zero_point) * (1 - lambda_at_zero_point) / std::abs(T_point_derivative);
      jacobian_matrix_extended.add(T_zero_point_dof_ind, T_zero_point_dof_ind, problem.q * term_with_delta_func);
      jacobian_matrix_extended.add(lambda_zero_point_dof_ind, T_zero_point_dof_ind, term_with_delta_func);

      // Add 1 to the position <code> T_zero_point_dof_ind </code> of the last row of the matrix.
      jacobian_matrix_extended.add(extended_solution_dim - 1, T_zero_point_dof_ind, 1.);
      
      zero_boundary_constraints.condense(jacobian_matrix_extended);
    }

    {
      TimerOutput::Scope t(computing_timer, "factorizing the Jacobian");

      std::cout << "Factorizing Jacobian matrix" << std::endl;

      jacobian_matrix_extended_factorization = std::make_unique<SparseDirectUMFPACK>();
      jacobian_matrix_extended_factorization->factorize(jacobian_matrix_extended);
    }

  }


  double TravelingWaveSolver::compute_residual(const Vector<double> &evaluation_point_extended, Vector<double> &residual)
  {
    TimerOutput::Scope t(computing_timer, "assembling the residual");
  
    std::cout << "Computing residual vector ... " << std::endl; 
    
    residual = 0;

    Vector<double> evaluation_point(dof_handler.n_dofs());
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
    {
      evaluation_point(i) = evaluation_point_extended(i);
    }

    const double wave_speed = evaluation_point_extended(extended_solution_dim - 1);

    const QGauss<1> quadrature_formula(number_of_quadrature_points);
    FEValues<1> fe_values(fe,
                quadrature_formula,
                update_values | update_gradients | 
                update_quadrature_points | update_JxW_values);
 
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();
   
    Vector<double> cell_residual(dofs_per_cell); 
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Scalar velocity(0);
    const FEValuesExtractors::Scalar temperature(1);
    const FEValuesExtractors::Scalar lambda(2);

    std::vector<double> current_velocity_values(n_q_points);
    std::vector<Tensor<1, 1>> current_velocity_gradients(n_q_points);
    std::vector<double> current_temperature_values(n_q_points);
    std::vector<Tensor<1, 1>> current_temperature_gradients(n_q_points);
    std::vector<double> current_lambda_values(n_q_points);
    std::vector<Tensor<1, 1>> current_lambda_gradients(n_q_points);

    std::vector<double> phi_u(dofs_per_cell);
    std::vector<Tensor<1, 1>> grad_phi_u(dofs_per_cell);
    std::vector<double> phi_T(dofs_per_cell);
    std::vector<Tensor<1, 1>> grad_phi_T(dofs_per_cell);
    std::vector<double> phi_lambda(dofs_per_cell);
    std::vector<Tensor<1, 1>> grad_phi_lambda(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_residual = 0;

      fe_values.reinit(cell);

      fe_values[velocity].get_function_values(evaluation_point, current_velocity_values);
      fe_values[velocity].get_function_gradients(evaluation_point, current_velocity_gradients);
      fe_values[temperature].get_function_values(evaluation_point, current_temperature_values);
      fe_values[temperature].get_function_gradients(evaluation_point, current_temperature_gradients);
      fe_values[lambda].get_function_values(evaluation_point, current_lambda_values);
      fe_values[lambda].get_function_gradients(evaluation_point, current_lambda_gradients);

      auto omega = [=](double T, double lambda){
        return problem.k * (1 - lambda) * std::exp(-problem.theta / T) * Heaviside_func(T - problem.T_ign);
      };

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          phi_u[k]            = fe_values[velocity].value(k, q);
          grad_phi_u[k]       = fe_values[velocity].gradient(k, q);
          phi_T[k]            = fe_values[temperature].value(k, q);
          grad_phi_T[k]       = fe_values[temperature].gradient(k, q);
          phi_lambda[k]       = fe_values[lambda].value(k, q);
          grad_phi_lambda[k]  = fe_values[lambda].gradient(k, q);
        }

        double del_Pr_eps = (problem.Pr * 4 * problem.delta / (3 * problem.epsilon));
        double del_Le = (problem.delta / problem.Le);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          cell_residual(i) += (

            del_Pr_eps * (-grad_phi_u[i] * current_velocity_gradients[q])
            + phi_u[i] * (
              - current_velocity_gradients[q][0] * (1 - wave_speed + problem.epsilon * current_velocity_values[q]) 
              - problem.epsilon / 2. * current_temperature_gradients[q][0]
            )

            + problem.delta * (-grad_phi_T[i] * current_temperature_gradients[q])
            + phi_T[i] * (
              wave_speed * (current_temperature_gradients[q][0] - current_velocity_gradients[q][0])
              + problem.q * omega(current_temperature_values[q], current_lambda_values[q])
            )

            + del_Le * (-grad_phi_lambda[i] * current_lambda_gradients[q])
            + phi_lambda[i] * (
              wave_speed * current_lambda_gradients[q][0] + omega(current_temperature_values[q], current_lambda_values[q])
            )

          ) * fe_values.JxW(q);
        }

      }											

      cell->get_dof_indices(local_dof_indices);

      for (const unsigned int i : fe_values.dof_indices())
      {					
        residual(local_dof_indices[i]) += cell_residual(i);
      }
    }

    residual(extended_solution_dim - 1) = 0.;

    zero_boundary_constraints.condense(residual);

    double residual_norm = residual.l2_norm();
    
    std::cout << std::defaultfloat;
    std::cout << "norm of residual = " << residual_norm << std::endl;

    return residual_norm;
  }

  // Split the solution vector into two parts: one part is the solution $u$, $T$ and $\lambda$, and another part is the wave speed.
  void TravelingWaveSolver::split_extended_solution_vector()
  {
    for (unsigned int i = 0; i < extended_solution_dim - 1; ++i)
    {
      current_solution(i) = current_solution_extended(i);
    }

    current_wave_speed = current_solution_extended(extended_solution_dim - 1);
  }


  void TravelingWaveSolver::solve(const Vector<double> &rhs, Vector<double> &solution_extended, const double /*tolerance*/)
  {
    TimerOutput::Scope t(computing_timer, "linear system solve");
 
    std::cout << "Solving linear system ... " << std::endl;

    jacobian_matrix_extended_factorization->vmult(solution_extended, rhs);
 
    zero_boundary_constraints.distribute(solution_extended);

  }


  // Function for adaptive mesh refinement based on <code> KellyErrorEstimator </code>.
  void TravelingWaveSolver::refine_mesh()
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    const FEValuesExtractors::Scalar lambda(2);

    KellyErrorEstimator<1>::estimate(
      dof_handler,
      QGauss<0>( 0 /* number_of_quadrature_points */),
      {},
      current_solution,
      estimated_error_per_cell,
      fe.component_mask(lambda)
    );

    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                            estimated_error_per_cell,
                            0.1,
                            0.05);

    triangulation.prepare_coarsening_and_refinement();

    SolutionTransfer<1> solution_transfer(dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(current_solution);

    triangulation.execute_coarsening_and_refinement();

    setup_system(/*initial_step=*/ false);

    Vector<double> tmp(dof_handler.n_dofs());
    solution_transfer.interpolate(current_solution, tmp);
    current_solution = std::move(tmp);

    set_boundary_and_centering_values();

    for (unsigned int i = 0; i < extended_solution_dim - 1; ++i)
    {
      current_solution_extended(i) = current_solution(i);
    }
    current_solution_extended(extended_solution_dim - 1) = current_wave_speed;

  }


  double TravelingWaveSolver::run_newton_iterations(const double target_tolerance)
  {
    
    double residual_norm = 0.;
    {
      typename SUNDIALS::KINSOL< Vector<double> >::AdditionalData additional_data;
      additional_data.function_tolerance = target_tolerance;
  
      SUNDIALS::KINSOL<Vector<double>> nonlinear_solver(additional_data);
  
      nonlinear_solver.reinit_vector = [&](Vector<double> &x) {
        x.reinit(extended_solution_dim);
      };

      nonlinear_solver.residual = [&](const Vector<double> &evaluation_point, Vector<double> &residual) {
        residual_norm = compute_residual(evaluation_point, residual);

        return 0;
      };
  
      nonlinear_solver.setup_jacobian = [&](const Vector<double> &evaluation_point, const Vector<double> & /*current_f*/) {
        compute_and_factorize_jacobian(evaluation_point);

        return 0;
      };

      nonlinear_solver.solve_with_jacobian = [&](const Vector<double> &rhs, Vector<double> &solution, const double tolerance) {
        this->solve(rhs, solution, tolerance);

        return 0;
      };

      nonlinear_solver.solve(current_solution_extended);
    }

    return residual_norm;

  }

  // Output the solution ($u$, $T$ and $\lambda$) and the wave speed into two separate files with double precision. The files can be read by gnuplot.
  void TravelingWaveSolver::output_with_double_precision(const Vector<double> &solution, const double wave_speed, const std::string filename)
  {
    TimerOutput::Scope t(computing_timer, "graphical output txt");

    const std::string file_for_solution = filename + ".txt";
    std::ofstream output(file_for_solution);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      for (const auto &v_ind : cell->vertex_indices())
      {
        double u = solution(cell->vertex_dof_index(v_ind, 0));
        double T = solution(cell->vertex_dof_index(v_ind, 1));
        double lambda = solution(cell->vertex_dof_index(v_ind, 2));

        output << std::scientific << std::setprecision(16);
        output << cell->vertex(v_ind)[0];

        output << std::scientific << std::setprecision(16);
        output << std::scientific << " " << u << " " << T << " " << lambda << "\n";
      }
      output  << "\n";
    }

    output.close();

    std::ofstream file_for_wave_speed_output("wave_speed-" + file_for_solution);
    file_for_wave_speed_output << std::scientific << std::setprecision(16);
    file_for_wave_speed_output << wave_speed << std::endl;
    file_for_wave_speed_output.close();
  }

  // Copy the solution into the <code> SolutionStruct </code> object, that stores the solution in an ordered manner. 
  void TravelingWaveSolver::get_solution(SolutionStruct &solution) const
  {
    // To obtain an ordered solution array, we first create a set consisting of the elements <code> {x, u, T, lambda} </code> in which the sorting is done by coordinate, and then copy the contents of the set into the arrays of the <code> SolutionStruct </code> object.
    auto comp = [](const std::vector<double> &a, const std::vector<double> &b) {
      return a[0] < b[0];
    };
    std::set<std::vector<double>, decltype(comp)> solution_set(comp);
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      for (const auto &v_ind : cell->vertex_indices())
      {
        double x = cell->vertex(v_ind)[0];
        double u = current_solution(cell->vertex_dof_index(v_ind, 0));
        double T = current_solution(cell->vertex_dof_index(v_ind, 1));
        double lambda = current_solution(cell->vertex_dof_index(v_ind, 2));
        solution_set.insert({x, u, T, lambda});
      }
    }

    solution.x.clear();
    solution.u.clear();
    solution.T.clear();
    solution.lambda.clear();

    solution.x.reserve(solution_set.size());
    solution.u.reserve(solution_set.size());
    solution.T.reserve(solution_set.size());
    solution.lambda.reserve(solution_set.size());

    for (auto it = solution_set.begin(); it != solution_set.end(); ++it)
    {
      solution.x.push_back((*it)[0]);
      solution.u.push_back((*it)[1]);
      solution.T.push_back((*it)[2]);
      solution.lambda.push_back((*it)[3]);
    }

    solution.wave_speed = current_wave_speed;

  }


  void TravelingWaveSolver::get_triangulation(Triangulation<1> &otriangulation) const
  {
    otriangulation.clear();
    otriangulation.copy_triangulation(triangulation);
  }


  void TravelingWaveSolver::run(const std::string filename, const bool save_solution_to_file)
  {
    const int mesh_refinement_type    = params.mesh.adaptive;
    const unsigned int n_refinements  = params.mesh.refinements_number;
    const double tol                  = params.solver.tol;

    if (triangulation_uploaded == false)  // If the triangulation is not loaded from outside, we will create one.
    {
      // We create two triangulations: one to the left and one to the right of zero coordinate. After that we merge them to obtain one triangulation, which contains zero point.
      Triangulation<1> triangulation_left;
      GridGenerator::subdivided_hyper_cube(
        triangulation_left,
        static_cast<unsigned int>(std::abs( 0. - params.mesh.interval_left )),
        params.mesh.interval_left, 0.
      );

      Triangulation<1> triangulation_right;
      GridGenerator::subdivided_hyper_cube(
        triangulation_right,
        static_cast<unsigned int>(std::abs( params.mesh.interval_right - 0. )),
        0., params.mesh.interval_right
      );

      GridGenerator::merge_triangulations(triangulation_left, triangulation_right, triangulation);

    }

    if (triangulation_uploaded == false)
    {
      if (mesh_refinement_type == 1)        // For ADAPTIVE mesh refinement.
      {
        triangulation.refine_global(1);     // refine initial mesh globally, before adaptive refinement cycles.
      }
      else if (mesh_refinement_type == 0)   // For GLOBAL mesh refinement.
      {
        triangulation.refine_global(n_refinements);
      }
    }

    setup_system(/*initial step*/ true);
    set_initial_guess();

    if (save_solution_to_file)
    {
      output_with_double_precision(current_solution, current_wave_speed, "solution_initial_data");
    }

    if (mesh_refinement_type == 1)    // Compute with ADAPTIVE mesh refinement.
    {
      double residual_norm = 0.;
      {
        Vector<double> tmp_residual(extended_solution_dim);
        residual_norm = compute_residual(current_solution_extended, tmp_residual);
      }

      unsigned int refinement_cycle = 0;
      while ((residual_norm > tol) && (refinement_cycle < n_refinements))
      {
        computing_timer.reset();
        std::cout << "Mesh refinement step " << refinement_cycle << std::endl;

        if (refinement_cycle != 0) { refine_mesh(); }
        
        const double target_tolerance = 0.1 * std::pow(0.1, refinement_cycle);    // Decrease tolerance for Newton solver at each refinement step.
        std::cout << "  Target_tolerance: " << target_tolerance << std::endl;
  
        residual_norm = run_newton_iterations(target_tolerance);
        split_extended_solution_vector();

        {
          std::cout << std::scientific << std::setprecision(16);
          std::cout << "current_wave_speed = " << current_wave_speed << std::endl;
          std::cout << std::defaultfloat;
        }

        computing_timer.print_summary();

        ++refinement_cycle;
      }
      if (save_solution_to_file)
      {
        output_with_double_precision(current_solution, current_wave_speed, filename);
      }

    }
    else if (mesh_refinement_type == 0)   // Compute with GLOBAL mesh refinement.
    {
      run_newton_iterations(tol);
      split_extended_solution_vector();
      
      if (save_solution_to_file)
      {
        output_with_double_precision(current_solution, current_wave_speed, filename);
      }
      
      {
        std::cout << std::scientific << std::setprecision(16);
        std::cout << "current_wave_speed = " << current_wave_speed << std::endl;
        std::cout << std::defaultfloat;
      }

      computing_timer.print_summary();

    }

  }


} // namespace TravelingWave
