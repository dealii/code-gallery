#include "TravelingWaveSolver.h"
#include "calculate_profile.h"

namespace TravelingWave
{
  using namespace dealii;

  // Computation of the limit case (ideal) solution, corresponding to $\delta = 0$, by solving the ODE. The output is the part of the solution to the left of zero. Here u_0, T_0, lambda_0 are the values of the medium state to the right of zero.
  void compute_limit_sol_left_part(const Parameters &parameters, 
                                    const double wave_speed, 
                                    const double u_0, 
                                    const double T_0, 
                                    const double lambda_0, 
                                    SolutionStruct &LimitSol, 
                                    const double root_sign)
  {
    LimitSolution limit_sol(parameters, lambda_0, u_0, T_0, root_sign);
    limit_sol.set_wave_speed(wave_speed);
    
    {
      // We take more integration points to better resolve the transition layer.
      std::vector<double> t_span(static_cast<unsigned int>(std::abs( 0. - parameters.mesh.interval_left )));
      double finer_mesh_starting_value = -9.1;
      linspace(parameters.mesh.interval_left, finer_mesh_starting_value, t_span); 
      std::vector<double> fine_grid(10000);
      linspace(finer_mesh_starting_value + 1e-4, 0., fine_grid); 
      t_span.insert(t_span.end(), fine_grid.begin(), fine_grid.end());

      // Reverse the order of the elements (because we need to perform back in time integration).
      std::reverse(t_span.begin(), t_span.end());

      state_type lambda_val(1);
      lambda_val[0] = lambda_0; 		// initial value
      IntegrateSystemAtTimePoints(limit_sol.lambda_vec, limit_sol.t_vec, t_span,
        limit_sol, 
        lambda_val,
        -1e-2, Integrator_Type::dopri5);
    }

    limit_sol.calculate_u_T_omega();

    // Reverse the order of elements
    std::reverse(limit_sol.t_vec.begin(), limit_sol.t_vec.end());
    std::reverse(limit_sol.lambda_vec.begin(), limit_sol.lambda_vec.end());
    std::reverse(limit_sol.u_vec.begin(), limit_sol.u_vec.end());
    std::reverse(limit_sol.T_vec.begin(), limit_sol.T_vec.end());
    std::reverse(limit_sol.omega_vec.begin(), limit_sol.omega_vec.end());

    SaveSolutionIntoFile(limit_sol.lambda_vec, limit_sol.t_vec, "solution_lambda_limit.txt");
    SaveSolutionIntoFile(limit_sol.u_vec, limit_sol.t_vec, "solution_u_limit.txt");
    SaveSolutionIntoFile(limit_sol.T_vec, limit_sol.t_vec, "solution_T_limit.txt");
    SaveSolutionIntoFile(limit_sol.omega_vec, limit_sol.t_vec, "solution_omega_limit.txt");

    LimitSol.reinit(limit_sol.t_vec.size());
    LimitSol.wave_speed = wave_speed;
    for (unsigned int i=0; i < limit_sol.t_vec.size(); ++i)
    {
      LimitSol.x[i] = limit_sol.t_vec[i];
      LimitSol.u[i] = limit_sol.u_vec[i][0];
      LimitSol.T[i] = limit_sol.T_vec[i][0];
      LimitSol.lambda[i] = limit_sol.lambda_vec[i][0];
    }
  }


  // Construction of an initial guess for detonation wave solution. The ODE is solved for the ideal system with $\delta = 0$.
  void compute_initial_guess_detonation(const Parameters &params, SolutionStruct &initial_guess, const double root_sign)
  {
    const Problem &problem = params.problem;
    double current_wave_speed(problem.wave_speed_init);

    {	// Here we compute the exact value of the wave speed $c$ for the detonation case. We can do this because we have the Dirichlet boundary conditions $T_l$, $T_r$ and $u_r$. Exact values of $u_l$ and $c$ are obtained using the integral relations.
      double DeltaT = problem.T_left - problem.T_right;
      double qDT = problem.q - DeltaT;
      current_wave_speed = 1. + problem.epsilon * (problem.u_right - (qDT * qDT + DeltaT) / (2 * qDT));
    }

    double u_0 = problem.u_right;
    double T_0 = problem.T_right;
    double lambda_0 = 0.;

    compute_limit_sol_left_part(params, current_wave_speed, u_0, T_0, lambda_0, initial_guess, root_sign);

    initial_guess.wave_speed = current_wave_speed;

    for (int i = initial_guess.x.size() - 1; i > - 1; --i)
    {
      if (isapprox(initial_guess.x[i], 0.))
      {
        initial_guess.u[i] = problem.u_right;
        initial_guess.T[i] = problem.T_ign;
        initial_guess.lambda[i] = 0.;
        break;
      }
    }

    // Adding the points to the right part of the interval (w.r.t. $\xi = 0$).
    unsigned int number_of_additional_points = 5;
    for (unsigned int i = 0; i < number_of_additional_points; ++i)
    {
      initial_guess.x.push_back(params.mesh.interval_right / (std::pow(2., number_of_additional_points - 1 - i)));
      initial_guess.u.push_back(problem.u_right);
      initial_guess.T.push_back(problem.T_right);
      initial_guess.lambda.push_back(0.);
    }

  }


  // Construction of a piecewise constant initial guess for deflagration wave solution.
  void compute_initial_guess_deflagration(const Parameters &params, SolutionStruct &initial_guess)
  {
    const Problem &problem = params.problem;
    double current_wave_speed(problem.wave_speed_init);

    double del_Pr_eps = (problem.Pr * 4 * problem.delta / (3 * problem.epsilon));
    double del_Le = (problem.delta / problem.Le);

    auto u_init_guess_func = [&](double x) {
      if (x < 0.)
      {
        return problem.u_left;
      }
      else
      {
        return problem.u_right;
      }
    };

    auto T_init_guess_func = [&](double x) {
      if (x < 0.)
      {
        return problem.T_left;
      }
      else if (isapprox(x, 0.))
      {
        return problem.T_ign;
      }
      else
      {
        return problem.T_right;
      }
    };

    auto lambda_init_guess_func = [=](double x) {
      if (x < 0.)
      {
        return -std::exp(x * std::abs(1 - current_wave_speed) / del_Pr_eps) + 1;
      }
      else 
      {
        return 0.;
      }
    };

    unsigned int multiplier_for_number_of_points = 7;
    unsigned int number_of_points = multiplier_for_number_of_points * static_cast<unsigned int>(std::trunc(std::abs( params.mesh.interval_right - params.mesh.interval_left )));
    std::vector<double> x_span(number_of_points);
    linspace(params.mesh.interval_left, params.mesh.interval_right, x_span);

    std::vector<double> u_init_arr(number_of_points);
    std::vector<double> T_init_arr(number_of_points);
    std::vector<double> lambda_init_arr(number_of_points);

    for (unsigned int i = 0; i < number_of_points; ++i)
    {
      u_init_arr[i] = u_init_guess_func(x_span[i]);
      T_init_arr[i] = T_init_guess_func(x_span[i]);
      lambda_init_arr[i] = lambda_init_guess_func(x_span[i]);
    }

    initial_guess.x = x_span;
    initial_guess.u = u_init_arr;
    initial_guess.T = T_init_arr;
    initial_guess.lambda = lambda_init_arr;
    initial_guess.wave_speed = current_wave_speed;

  }


  // Compute the traveling-wave profile. The continuation method can be switched on by setting the argument <code> continuation_for_delta </code> as <code> true </code>.
  void calculate_profile(Parameters& parameters,
                                    const bool continuation_for_delta /* Compute with the continuation. */, 
                                    const double delta_start /* The starting value of delta for the continuation method. */, 
                                    const unsigned int number_of_continuation_points)
  {
    SolutionStruct sol;

    if (parameters.problem.wave_type == 1) 				// detonation wave
    {
      compute_initial_guess_detonation(parameters, sol);
    }
    else if (parameters.problem.wave_type == 0)		// deflagration wave
    {
      compute_initial_guess_deflagration(parameters, sol);
    }
    
    if (continuation_for_delta == false)
    {
      TravelingWaveSolver wave(parameters, sol);
      std::string filename = "solution_delta-" + Utilities::to_string(parameters.problem.delta) + "_eps-" 
                                                        + Utilities::to_string(parameters.problem.epsilon);
      wave.run(filename);
      wave.get_solution(sol);
    }
    else	// Run with continuation_for_delta.
    {
      double delta_target = parameters.problem.delta;
      parameters.problem.delta = delta_start;

      std::vector<double> delta_span(number_of_continuation_points);

      // Generate a sequence of delta values being uniformly distributed in log10 scale.
      {
        double delta_start_log10 = std::log10(delta_start);
        double delta_target_log10 = std::log10(delta_target);

        std::vector<double> delta_log_span(delta_span.size());
        linspace(delta_start_log10, delta_target_log10, delta_log_span);

        for (unsigned int i = 0; i < delta_span.size(); ++i)
        {
          delta_span[i] = std::pow(10, delta_log_span[i]);
        }
      }

      Triangulation<1> refined_triangulation;
      bool first_iter_flag = true;

      for (double delta : delta_span)
      {
        parameters.problem.delta = delta;
        std::string filename = "solution_delta-" + Utilities::to_string(parameters.problem.delta) + "_eps-" 
                                                    + Utilities::to_string(parameters.problem.epsilon);

        TravelingWaveSolver wave(parameters, sol);

        if (first_iter_flag)
        {
          first_iter_flag = false;
        }
        else
        {
          wave.set_triangulation(refined_triangulation);	
        }

        wave.run(filename);
        wave.get_solution(sol);
        wave.get_triangulation(refined_triangulation);
      }

    }

    // Error estimation.
    {
      unsigned int sol_length = sol.x.size();
      double u_r = sol.u[sol_length-1];     // Dirichlet boundary condition
      double T_r = sol.T[sol_length-1];     // Dirichlet condition only for detonation case
      double u_l = sol.u[0];
      double T_l = sol.T[0];                // Dirichlet boundary condition
      double wave_speed = sol.wave_speed;

      std::cout << "Error estimates:" << std::endl;
      double DeltaT = T_l - T_r;
      double qDT = parameters.problem.q - DeltaT;

      double wave_speed_formula = 1. + parameters.problem.epsilon * (u_r - (qDT * qDT + DeltaT) / (2 * qDT));
      std::cout << std::setw(18) << std::left << "For wave speed" << " :  " << std::setw(5) << wave_speed - wave_speed_formula << std::endl;

      double u_l_formula = DeltaT + u_r - parameters.problem.q;
      std::cout << std::setw(18) << std::left << "For u_l" << " :  " << std::setw(5) << u_l - u_l_formula << std::endl;
    }

  }

} // namespace TravelingWave