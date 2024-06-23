#include "nonlinear_heat.h"
/**
 * Solves the linear system, arising during every nonlinear iteration.
 * @param rhs Right hand side (Residual)
 * @param solution Solution is captured here.
 */
void nonlinear_heat::solve(const Vector<double> & rhs, Vector<double> & solution, const double /*tolerance*/)
{
    std::cout << "  Solving linear system" << std::endl;
    matrix_factorization->vmult(solution, rhs);
}

void nonlinear_heat::run()
{
    GridIn<2> gridin;
    gridin.attach_triangulation(triangulation);
    std::ifstream f("mesh/mesh.msh");
    gridin.read_msh(f);
    triangulation.refine_global(1);

    double time = 0;
    unsigned int timestep_number = 0;
    unsigned int prn =0;


    while (time <=tot_time)
    {
        if(time ==0)
        {
            setup_system(timestep_number);

            VectorTools::interpolate(dof_handler,
                                     Initialcondition(),
                                     present_solution);

            VectorTools::interpolate(dof_handler,
                                     Initialcondition(),
                                     converged_solution);
        }
        else
        {
            /**
             * For times > 0 , the initial guess should be the #converged_solution from the previous time step.
             */
            present_solution = converged_solution;
        }
        std::cout<<">>>>> Time now  is: "<<time <<std::endl;
        std::cout<<">>>>> Time step is:"<<timestep_number<<std::endl;
        set_boundary_conditions(time);
        {

            const double target_tolerance = 1e-3;
            /**
             * Setting up of the Nox solver with some additional data, such as the number of nonlinear iterations,
             * tolerance to check for convergence.
             */
            typename TrilinosWrappers::NOXSolver<Vector<double>>::AdditionalData additional_data;
            additional_data.abs_tol = target_tolerance;
            additional_data.max_iter = 100;
            TrilinosWrappers::NOXSolver<Vector<double>> nonlinear_solver(
                    additional_data);

            /**
             * Defines how the NOX solver should calculate the residual and where it needs to evaluate it.
             */
            nonlinear_solver.residual =
                    [&](const Vector<double> &evaluation_point,
                        Vector<double> &residual) {
                        compute_residual(evaluation_point, residual);
                    };

            /**
             * Sets up the jacobian, which will be called whenever solve_with_jacobian() is invoked.
             */

            nonlinear_solver.setup_jacobian =
                    [&](const Vector<double> &current_u) {
                        compute_jacobian(current_u);
                    };
            /**
             * Solve the nonlinear problem with the jacobian.
             */

            nonlinear_solver.solve_with_jacobian = [&](const Vector<double> &rhs,
                                                       Vector<double> &dst,
                                                       const double tolerance) {
                solve(rhs, dst, tolerance);
            };
            /**
             * #present_solution is used as an initial guess. Then the non-linear solver is called. The solver now repeatedly
             * solves the set of equations until convergence and stores the final (converged) solution in #present_solution.
             */
            nonlinear_solver.solve(present_solution);
        }
        /**
         * So, the #converged_solution is assigned the converged value of #present_solution.
         */
        converged_solution = present_solution;

        if(timestep_number % 1 == 0) {

            output_results(prn);
            prn = prn +1;
        }
        timestep_number++;
        time=time+delta_t;

    }
}
