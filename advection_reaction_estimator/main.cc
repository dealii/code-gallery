#include "include/DG_advection_reaction.h"

int main(int argc, char **argv)
{

  try
  {
    std::string par_name = "";
    if (argc > 1)
    {
      par_name = argv[1];
    }
    deallog.depth_console(2); //solver infos
    AdvectionReaction<2> dgmethod;
    if (par_name != "")
    {
      dgmethod.initialize_params(par_name);
    }

    dgmethod.run();
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
  catch (const theta_exc &theta_range)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << theta_range.what() << std::endl
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
