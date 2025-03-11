#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_refinement.h>
#include <iostream>
#include "../include/markov_filter.h"
#include "../include/kkt_system.h"
#include "../include/input_information.h"
#include "../include/watchdog.h"
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>

///Above are fairly normal files to include.  I also use the sparse direct package, which requiresBLAS/LAPACK
/// to  perform  a  direct  solve  while  I  work  on  a  fast  iterative  solver  for  this problem.

namespace SAND {
    namespace LA
    {
        using namespace dealii::LinearAlgebraTrilinos;
    }


    using namespace dealii;

    ///Constructor
    template<int dim>
    NonlinearWatchdog<dim>::NonlinearWatchdog()
            :
              mpi_communicator(MPI_COMM_WORLD),
              pcout(std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
              overall_timer(pcout, TimerOutput::never, TimerOutput::wall_times)
    {
    }

    ///A binary search figures out the maximum step that meets the dual feasibility - that s>0 and z>0. The fraction to boundary increases as the barrier size decreases.

    template<int dim>
    std::pair<double,double>
    NonlinearWatchdog<dim>::calculate_max_step_size(const LA::MPI::BlockVector &state, const LA::MPI::BlockVector &step) const {

        double step_size_s_low = 0;
        double step_size_z_low = 0;
        double step_size_s_high = 1;
        double step_size_z_high = 1;
        double step_size_s, step_size_z;
        LA::MPI::BlockVector state_test_s = state;
        state_test_s = 0;
        LA::MPI::BlockVector state_test_z = state;
        state_test_z = 0;
        for (unsigned int k = 0; k < 50; k++)
        {

            step_size_s = (step_size_s_low + step_size_s_high) / 2;
            step_size_z = (step_size_z_low + step_size_z_high) / 2;
            const LA::MPI::BlockVector state_test_s =
                    (Input::fraction_to_boundary * state) + (step_size_s * step);

            const LA::MPI::BlockVector state_test_z =
                    (Input::fraction_to_boundary * state) + (step_size_z * step);

            const bool accept_s = (state_test_s.block(SolutionBlocks::density_lower_slack).is_non_negative())
                                  && (state_test_s.block(SolutionBlocks::density_upper_slack).is_non_negative());
            const bool accept_z = (state_test_z.block(SolutionBlocks::density_lower_slack_multiplier).is_non_negative())
                                  && (state_test_z.block(SolutionBlocks::density_upper_slack_multiplier).is_non_negative());

            if (accept_s) {
                step_size_s_low = step_size_s;
            } else {
                step_size_s_high = step_size_s;
            }
            if (accept_z) {
                step_size_z_low = step_size_z;
            } else {
                step_size_z_high = step_size_z;
            }
        }
        pcout << "s step : " << step_size_s_low << " z size : " << step_size_z_low << std::endl;
        return {step_size_s_low, step_size_z_low};
    }

    ///Creates a rhs vector that we can use to look at the magnitude of the KKT conditions.  This is then used for testing the convergence before shrinking barrier size, as well as in the calculation of the l1 merit.

    template<int dim>
    const LA::MPI::BlockVector
    NonlinearWatchdog<dim>::find_max_step(const LA::MPI::BlockVector &state)
    {
        TimerOutput::Scope t(overall_timer, "find step");
        {
            TimerOutput::Scope t(overall_timer, "assemble");
            kkt_system.assemble_block_system(state, barrier_size);
        }

        pcout << "pre" << std::endl;
        LA::MPI::BlockVector step;
        {
            TimerOutput::Scope t(overall_timer, "solve");
            step = kkt_system.solve(state);
        }

        pcout << "post" << std::endl;
        const auto max_step_sizes= calculate_max_step_size(state,step);
        const double step_size_s = max_step_sizes.first;
        const double step_size_z = max_step_sizes.second;
        LA::MPI::BlockVector max_step(10);

        max_step.block(SolutionBlocks::density) = step_size_s * step.block(SolutionBlocks::density);
        max_step.block(SolutionBlocks::displacement) = step_size_s * step.block(SolutionBlocks::displacement);
        max_step.block(SolutionBlocks::unfiltered_density) = step_size_s * step.block(SolutionBlocks::unfiltered_density);
        max_step.block(SolutionBlocks::density_lower_slack) = step_size_s * step.block(SolutionBlocks::density_lower_slack);
        max_step.block(SolutionBlocks::density_upper_slack) = step_size_s * step.block(SolutionBlocks::density_upper_slack);
        max_step.block(SolutionBlocks::unfiltered_density_multiplier) = step_size_z * step.block(SolutionBlocks::unfiltered_density_multiplier);
        max_step.block(SolutionBlocks::density_lower_slack_multiplier) = step_size_z * step.block(SolutionBlocks::density_lower_slack_multiplier);
        max_step.block(SolutionBlocks::density_upper_slack_multiplier) = step_size_z * step.block(SolutionBlocks::density_upper_slack_multiplier);
        max_step.block(SolutionBlocks::displacement_multiplier) = step_size_z * step.block(SolutionBlocks::displacement_multiplier);
        max_step.block(SolutionBlocks::total_volume_multiplier) = step_size_z * step.block(SolutionBlocks::total_volume_multiplier);

        pcout << "here" << std::endl;
        return max_step;
    }

    ///This is a simple back-stepping algorithm for a line search - keeps shrinking step size until it finds a step where the markov filter requirement is met.

    template<int dim>
    LA::MPI::BlockVector
    NonlinearWatchdog<dim>::take_scaled_step(const LA::MPI::BlockVector &state,const LA::MPI::BlockVector &max_step) const
    {
        double step_size = 1;
            for(unsigned int k = 0; k<10; k++)
            {
                if(markov_filter.check_filter(kkt_system.calculate_objective_value(state), kkt_system.calculate_barrier_distance(state), kkt_system.calculate_feasibility(state,barrier_size)))
                {
                    break;
                }
                else
                {
                    step_size = step_size/2;
                }
            }
        return state + (step_size * max_step);

    }



    ///Checks to see if the KKT conditions are sufficiently met to lower barrier size.
    template<int dim>
    bool
    NonlinearWatchdog<dim>::check_convergence(const LA::MPI::BlockVector &state) const
    {
              if (kkt_system.calculate_convergence(state) < Input::required_norm)
              {
                  return true;
              }
              else
              {
                  return false;
              }
    }

    ///This updates the barrier value using the selected barrier scheme - more work could be done to optimize
    /// the performance of the mixed method
    template<int dim>
    void
    NonlinearWatchdog<dim>::update_barrier(LA::MPI::BlockVector &current_state)
    {
        ///The LOQO scheme uses information about the similarity of the slack/slack multiplier product as a
        /// heuristic for decreasing barrier value
        if (Input::barrier_reduction == BarrierOptions::loqo)
        {
            double loqo_min = 1000;
            double loqo_average;
            double lower_prod;
            double full_lower_prod;
            double upper_prod;
            double full_upper_prod;
            unsigned int vect_size = current_state.block(SolutionBlocks::density_lower_slack).size();
            for(unsigned int k = 0; k < vect_size; k++)
            {
                lower_prod = 1;
                if (current_state.block(SolutionBlocks::density_lower_slack).in_local_range(k))
                    lower_prod=lower_prod * current_state.block(SolutionBlocks::density_lower_slack)[k];
                if (current_state.block(SolutionBlocks::density_lower_slack_multiplier).in_local_range(k))
                    lower_prod=lower_prod * current_state.block(SolutionBlocks::density_lower_slack_multiplier)[k];

                upper_prod=1;
                if (current_state.block(SolutionBlocks::density_upper_slack).in_local_range(k))
                    upper_prod=upper_prod * current_state.block(SolutionBlocks::density_upper_slack)[k];
                if (current_state.block(SolutionBlocks::density_upper_slack_multiplier).in_local_range(k))
                    upper_prod=upper_prod * current_state.block(SolutionBlocks::density_upper_slack_multiplier)[k];

                MPI_Allreduce(&lower_prod, &full_lower_prod, 1, MPI_DOUBLE, MPI_PROD, MPI_COMM_WORLD);
                MPI_Allreduce(&upper_prod, &full_upper_prod, 1, MPI_DOUBLE, MPI_PROD, MPI_COMM_WORLD);
                if (full_lower_prod < loqo_min)
                {
                    loqo_min = full_lower_prod;
                }
                if (full_upper_prod < loqo_min)
                {
                    loqo_min = full_upper_prod;
                }
            }
            loqo_average = (current_state.block(SolutionBlocks::density_lower_slack)*current_state.block(SolutionBlocks::density_lower_slack_multiplier)
                            + current_state.block(SolutionBlocks::density_upper_slack)*current_state.block(SolutionBlocks::density_upper_slack_multiplier)
                           )/(2*vect_size);
            double loqo_complimentarity_deviation = loqo_min/loqo_average;
            pcout << "loqo cd: " << loqo_complimentarity_deviation << std::endl;
            double loqo_multiplier;
            if((.05 * (1-loqo_complimentarity_deviation)/loqo_complimentarity_deviation)<2)
            {
                loqo_multiplier = .1*std::pow((.05 * (1-loqo_complimentarity_deviation)/loqo_complimentarity_deviation),3);
            }
            else
            {
                loqo_multiplier = .8;
            }
            pcout << "loqo mult: " << loqo_multiplier << std::endl;
            if (loqo_multiplier< 0)
            {
                barrier_size = std::abs(loqo_multiplier) * loqo_average;
            }
            else
            {
                barrier_size = loqo_multiplier * loqo_average;
            }
            if (barrier_size < Input::min_barrier_size)
            {
                barrier_size=Input::min_barrier_size;
            }
        }

        ///The monotome scheme fully solves the problem with one barrier size before decreasing the
        /// barrier and starting again
        if (Input::barrier_reduction == BarrierOptions::monotone)
        {
            if (kkt_system.calculate_rhs_norm(current_state,barrier_size) < barrier_size * 1e-3)
            {
                barrier_size = barrier_size * .7;
            }
            if (barrier_size < Input::min_barrier_size)
            {
                barrier_size=Input::min_barrier_size;
            }
        }

        ///The mixed method uses LOQO unless it gets stuck, at which point it switches to monotone, allowing for an adaptive method
        /// that still globally converges the barrier value to 0.
        if (Input::barrier_reduction == BarrierOptions::mixed)
        {
            if (mixed_barrier_monotone_mode)
            {
                if (kkt_system.calculate_rhs_norm(current_state,barrier_size) < barrier_size)
                {
                    barrier_size = barrier_size * .8;
                    mixed_barrier_monotone_mode=false;
                    pcout << "monotone mode turned off" << std::endl;
                }
            }
            else
            {
                double loqo_min = 1000;
                double loqo_average;
                unsigned int vect_size = current_state.block(SolutionBlocks::density_lower_slack).size();
                double lower_prod, full_lower_prod, upper_prod, full_upper_prod;
                for(unsigned int k = 0; k < vect_size; k++)
                {
                    lower_prod = 1;
                    if (current_state.block(SolutionBlocks::density_lower_slack).in_local_range(k))
                        lower_prod=lower_prod * current_state.block(SolutionBlocks::density_lower_slack)[k];
                    if (current_state.block(SolutionBlocks::density_lower_slack_multiplier).in_local_range(k))
                        lower_prod=lower_prod * current_state.block(SolutionBlocks::density_lower_slack_multiplier)[k];

                    upper_prod=1;
                    if (current_state.block(SolutionBlocks::density_upper_slack).in_local_range(k))
                        upper_prod=upper_prod * current_state.block(SolutionBlocks::density_upper_slack)[k];
                    if (current_state.block(SolutionBlocks::density_upper_slack_multiplier).in_local_range(k))
                        upper_prod=upper_prod * current_state.block(SolutionBlocks::density_upper_slack_multiplier)[k];

                    MPI_Allreduce(&lower_prod, &full_lower_prod, 1, MPI_DOUBLE, MPI_PROD, MPI_COMM_WORLD);
                    MPI_Allreduce(&upper_prod, &full_upper_prod, 1, MPI_DOUBLE, MPI_PROD, MPI_COMM_WORLD);

                    if (full_lower_prod < loqo_min)
                    {
                        loqo_min = full_lower_prod;
                    }
                    if (full_upper_prod < loqo_min)
                    {
                        loqo_min = full_upper_prod;
                    }
                }
                loqo_average = (current_state.block(SolutionBlocks::density_lower_slack)*current_state.block(SolutionBlocks::density_lower_slack_multiplier)
                                + current_state.block(SolutionBlocks::density_upper_slack)*current_state.block(SolutionBlocks::density_upper_slack_multiplier)
                               )/(2*vect_size);
                double loqo_complimentarity_deviation = loqo_min/loqo_average;
                double loqo_multiplier;
                if((.05 * (1-loqo_complimentarity_deviation)/loqo_complimentarity_deviation)<2)
                {
                    loqo_multiplier = .1*std::pow((.05 * (1-loqo_complimentarity_deviation)/loqo_complimentarity_deviation),3);
                }
                else
                {
                    loqo_multiplier = 1/.8;
                    mixed_barrier_monotone_mode = true;
                    pcout << "monotone mode turned on" << std::endl;
                }
                if (loqo_multiplier<.01)
                {
                    barrier_size = .01 * loqo_average;
                }
                else
                {
                    barrier_size = loqo_multiplier * loqo_average;
                }
                if (barrier_size < Input::min_barrier_size)
                {
                    barrier_size=Input::min_barrier_size;
                }
            }
        }

    }

    template<int dim>
    void
    NonlinearWatchdog<dim>::perform_initial_setup()
    {
        barrier_size = Input::initial_barrier_size;
        kkt_system.create_triangulation();
        kkt_system.setup_boundary_values();
        pcout << "setup kkt system" << std::endl;
        kkt_system.setup_block_system();
        pcout << "setup kkt system" << std::endl;

        if (Input::barrier_reduction==BarrierOptions::mixed)
        {
            mixed_barrier_monotone_mode = false;
        }
    }


    template<int dim>
    void
    NonlinearWatchdog<dim>::nonlinear_step(LA::MPI::BlockVector &current_state, LA::MPI::BlockVector &current_step, const unsigned int max_uphill_steps, unsigned int &iteration_number)
    {

        bool converged = false;
        //while not converged
        while(!converged && iteration_number < Input::max_steps)
        {
            bool found_step = false;
            //save current state as watchdog state

            const LA::MPI::BlockVector watchdog_state = current_state;
            LA::MPI::BlockVector watchdog_step;
            //for 1-8 steps - this is the number of steps away we will let it go uphill before demanding downhill
            for(unsigned int k = 0; k<max_uphill_steps; k++)
            {

                //compute step from current state  - function from kktSystem
                current_step = find_max_step(current_state);

                // save the first of these as the watchdog step
                if(k==0)
                {
                    watchdog_step = current_step;
                    if (iteration_number == 0)
                    {
                        kkt_system.calculate_initial_rhs_error();
                    }
                }

                //apply full step to current state
                current_state=current_state+current_step;


                //if new state passes filter
                if(markov_filter.check_filter(kkt_system.calculate_objective_value(current_state), kkt_system.calculate_barrier_distance(current_state), kkt_system.calculate_feasibility(current_state,barrier_size)))
                {
                    //Accept current state
                    //iterate number of steps by number of steps taken in this process
                    iteration_number = iteration_number + k + 1;
                    found_step = true;
                    pcout << "found workable step after " << k+1 << " iterations"<<std::endl;
                    //break for loop
                    markov_filter.add_point(kkt_system.calculate_objective_value(current_state), kkt_system.calculate_barrier_distance(current_state), kkt_system.calculate_feasibility(current_state,barrier_size));
                    break;
                    //end if
                }
                //end for
            }
            //if found step = false
            if (!found_step)
            {
                //Compute step from current state
                current_step = find_max_step(current_state);
                //find step length so that merit of stretch state - sized step from current length - is less than merit of (current state + descent requirement * linear derivative of merit of current state in direction of current step)
                //update stretch state with found step length
                const LA::MPI::BlockVector stretch_state = take_scaled_step(current_state, current_step);
                //if current merit is less than watchdog merit, or if stretch merit is less than earlier goal merit
                if(markov_filter.check_filter(kkt_system.calculate_objective_value(current_state), kkt_system.calculate_barrier_distance(current_state), kkt_system.calculate_feasibility(current_state,barrier_size)))
                {
                    current_state = stretch_state;
                    iteration_number = iteration_number + max_uphill_steps + 1;
                    markov_filter.add_point(kkt_system.calculate_objective_value(current_state), kkt_system.calculate_barrier_distance(current_state), kkt_system.calculate_feasibility(current_state,barrier_size));
                }
                else
                {
                    //if merit of stretch state is bigger than watchdog merit
                    if (markov_filter.check_filter(kkt_system.calculate_objective_value(current_state), kkt_system.calculate_barrier_distance(current_state), kkt_system.calculate_feasibility(current_state,barrier_size)))
                    {
                        //find step length from watchdog state that meets descent requirement
                        current_state = take_scaled_step(watchdog_state, watchdog_step);
                        //update iteration count
                        iteration_number = iteration_number +  max_uphill_steps + 1;
                        markov_filter.add_point(kkt_system.calculate_objective_value(current_state), kkt_system.calculate_barrier_distance(current_state), kkt_system.calculate_feasibility(current_state,barrier_size));

                    }
                    else
                    {
                        //calculate direction from stretch state
                        const LA::MPI::BlockVector stretch_step = find_max_step(stretch_state);
                        //find step length from stretch state that meets descent requirement
                        current_state = take_scaled_step(stretch_state, stretch_step);
                        //update iteration count
                        iteration_number = iteration_number + max_uphill_steps + 2;
                        markov_filter.add_point(kkt_system.calculate_objective_value(current_state), kkt_system.calculate_barrier_distance(current_state), kkt_system.calculate_feasibility(current_state,barrier_size));
                    }
                }
            }
            //output current state
            kkt_system.output(current_state,iteration_number);

            converged = check_convergence(current_state);
            update_barrier(current_state);
            markov_filter.update_barrier_value(barrier_size);
            pcout << "barrier size is now " << barrier_size << " on iteration number " << iteration_number << std::endl;


            overall_timer.leave_subsection();
            overall_timer.print_summary();
            overall_timer.enter_subsection("Total Time");

        }//end while
    }

    ///Contains watchdog algorithm
    template<int dim>
    void
    NonlinearWatchdog<dim>::run() {
        overall_timer.enter_subsection("Total Time");

        perform_initial_setup();

        const unsigned int max_uphill_steps = 8;
        unsigned int iteration_number = 0;

        //while barrier value above minimal value and total iterations under some value
        LA::MPI::BlockVector current_state = kkt_system.get_initial_state();
        LA::MPI::BlockVector current_step;

        markov_filter.setup(kkt_system.calculate_objective_value(current_state), kkt_system.calculate_barrier_distance(current_state), kkt_system.calculate_feasibility(current_state,barrier_size), barrier_size);

        // std::cout << "finished setup - beginning watchdog steps" << std::endl;

        while((barrier_size > Input::min_barrier_size || !check_convergence(current_state)) && iteration_number < Input::max_steps)
        {
            nonlinear_step(current_state, current_step, max_uphill_steps, iteration_number);
        }
//        kkt_system.output_stl(current_state);
    }

} // namespace SAND


template class SAND::NonlinearWatchdog<2>;
template class SAND::NonlinearWatchdog<3>;
