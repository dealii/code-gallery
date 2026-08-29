#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_refinement.h>
#include <iostream>
#include "../include/markov_filter.h"
#include "../include/kkt_system.h"
#include "../include/input_information.h"
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

    /// Below is the main class for solving this problem. It handles the nonlinear solver portion of the problem,
    /// taking information from the KKTSystem class for step directions, and calculating step lengths. This class
    /// not only takes those steps, but handles the barrier parameter for the log barrier used.
    template<int dim>
    class NonlinearWatchdog {
    public:
        NonlinearWatchdog();

        void
        run();

    private:
        MPI_Comm  mpi_communicator;
        std::pair<double,double>
        calculate_max_step_size(const LA::MPI::BlockVector &state, const LA::MPI::BlockVector &step) const;

        const LA::MPI::BlockVector
        find_max_step(const LA::MPI::BlockVector &state);

        LA::MPI::BlockVector
        take_scaled_step(const LA::MPI::BlockVector &state,const LA::MPI::BlockVector &max_step) const;

        bool
        check_convergence(const LA::MPI::BlockVector &state) const;

        void
        update_barrier(LA::MPI::BlockVector &current_state);

        void
        perform_initial_setup();

        void
        nonlinear_step(LA::MPI::BlockVector &current_state, LA::MPI::BlockVector &current_step, const unsigned int max_uphill_steps, unsigned int &iteration_number);

        KktSystem<dim> kkt_system;
        MarkovFilter markov_filter;
        double barrier_size;
        bool mixed_barrier_monotone_mode;
        ConditionalOStream pcout;
        TimerOutput overall_timer;
    };

} // namespace SAND
