/**
 * Contributed by Narasimhan Swaminathan
 * 20 Jun 2024
 */

#include "allheaders.h"
#include "nonlinear_heat.h"
int main()
{
  try
    {
      nonlinear_heat nlheat; /*!< Instantiates the nonlinear heat object */
      nlheat.run(); /*!< Runs the 'run' function of the object */
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
