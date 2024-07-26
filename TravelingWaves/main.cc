/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
 * Copyright (C) 2024 by Shamil Magomedov
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 */

#include "calculate_profile.h"

int main(int argc, char *argv[])
{

  try
  {
    using namespace TravelingWave;
  
    Parameters parameters;

    std::string prm_filename = "ParametersList.prm";
    if (argc > 1)
    {
      // Check if file argv[1] exists.
      if (file_exists(argv[1])) 
      {
        prm_filename = argv[1];
      }
      else
      {
        std::string errorMessage = "File \"" + std::string(argv[1]) + "\" is not found.";
        throw std::runtime_error(errorMessage);
      }
    }
    else
    {
      // Check if the file "ParametersList.prm" exists in the current or in the parent directory.
      if (!(file_exists(prm_filename) || file_exists("../" + prm_filename)))
      {
        std::string errorMessage = "File \"" + prm_filename + "\" is not found.";
        throw std::runtime_error(errorMessage);
      }
      else
      {
        if (!file_exists(prm_filename))
        {
          prm_filename = "../" + prm_filename;
        }
      }
    }

    std::cout << "Reading parameters... " << std::flush;
    ParameterAcceptor::initialize(prm_filename);
    std::cout << "done" << std::endl;
    
    calculate_profile(parameters, /* With continuation_for_delta */ false, 0.1, 3);
    
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
