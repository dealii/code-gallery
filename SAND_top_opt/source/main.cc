#include "../include/watchdog.h"
#include "../include/input_information.h"
#include <cstdlib>

///Above are fairly normal files to include.  I also use the sparse direct package, which requiresBLAS/LAPACK
/// to  perform  a  direct  solve  while  I  work  on  a  fast  iterative  solver  for  this problem.


int
main(int argc, char *argv[]) {
    try
    {
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
        SAND::NonlinearWatchdog<SAND::Input::dim> elastic_problem;
        elastic_problem.run();
    }
    catch (std::exception &exc) {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------" << std::endl;
        std::cerr << "Exception on processing: " << std::endl << exc.what()
                  << std::endl << "Aborting!" << std::endl
                  << "----------------------------------------------------" << std::endl;

        return 1;
    }
    catch (...) {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------" << std::endl;
        std::cerr << "Unknown exception!" << std::endl << "Aborting!" << std::endl
                  << "----------------------------------------------------" << std::endl;
        return 1;
    }

    return 0;
}
