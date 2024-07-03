#include "PhaseFieldSolver.h"
#include <time.h>

void PhaseFieldSolver::run() {
    make_grid_and_dofs();
    pcout << "Processors used: " << n_mpi_processes << std::endl;
    pcout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
          << " (by partition:";
    for (unsigned int p = 0; p < n_mpi_processes; ++p)
        pcout << (p == 0 ? ' ' : '+')
              << (DoFTools::count_dofs_with_subdomain_association(dof_handler,
                                                                  p));
    pcout << ")" << std::endl;
    //Initialise the solution
    InitialValues initial_value;
    VectorTools::interpolate(dof_handler,
                             initial_value,
                             old_solution);
    VectorTools::interpolate(dof_handler,
                             initial_value,
                             conv_solution);
    //Applying Boundary Conditions at t=0
    applying_bc();
    //Plotting initial solution
    output_results(0);

    //Time steps begin here:
    unsigned int timestep_number = 1;
    for (; time <= final_time; time += time_step, ++timestep_number) {

        pcout << "Time step " << timestep_number << " at t=" << time+time_step
                  << std::endl;

        conv_solution.operator=(old_solution); // initialising the newton solution

        //Newton-Raphson iterations begin here:
        for (unsigned int it = 1; it <= 100; ++it) {
            pcout << "Newton iteration number:" << it << std::endl;

            if (it == 100) {
                pcout << "Convergence Failure!!!!!!!!!!!!!!!" << std::endl;
                std::exit(0);
            }
            //Saving parallel vectors as non-parallel ones
            conv_solution_np = conv_solution;
            old_solution_np = old_solution;
            //Initialise the delta solution as zero
            VectorTools::interpolate(dof_handler,
                                     ZeroFunction<2>(2),
                                     solution_update);
            solution_update.compress(VectorOperation::insert);
            //Assemble Jacobian and Residual
            assemble_system();
            //Solving to get delta solution
            solve();
            //Checking for convergence
            double residual_norm = system_rhs.l2_norm(); //the norm of residual should converge to zero as the solution converges
            //pcout << "Nothing wrong till here!!!!!!" << std::endl;
            pcout << "the residual is:" << residual_norm << std::endl;
            if (residual_norm <= (1e-4)) {
                pcout << "Solution Converged!" << std::endl;
                break; //Break to next time step if the N-R iterations converge
            }
        }
        //Transfer the converged solution to the old_solution vector to plot output
        old_solution.operator=(conv_solution);
        old_solution.compress(VectorOperation::insert);
        //output the solution at only specific number of time steps
        if (timestep_number%10 == 0)
            output_results(timestep_number);
    }
}
