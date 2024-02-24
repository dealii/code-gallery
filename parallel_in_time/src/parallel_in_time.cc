/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2018 by Joshua Christopher
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 *
 * Author: Joshua Christopher, Colorado State University, 2018
 */

#include "BraidFuncs.hh"
#include "HeatEquation.hh"
#include "Utilities.hh"

#include <fstream>
#include <iostream>

int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;

      /* Initialize MPI */
      MPI_Comm      comm; //, comm_x, comm_t;
      int rank;
      MPI_Init(&argc, &argv);
      comm   = MPI_COMM_WORLD;
      MPI_Comm_rank(comm, &rank);
      procID = rank;

      // Set up X-Braid
      /* Initialize Braid */
      braid_Core core;
      double tstart = 0.0;
      double tstop = 0.002;
      int    ntime = 10;
      my_App *app = new(my_App);

      braid_Init(MPI_COMM_WORLD, comm, tstart, tstop, ntime, app,
                 my_Step, my_Init, my_Clone, my_Free, my_Sum, my_SpatialNorm,
                 my_Access, my_BufSize, my_BufPack, my_BufUnpack, &core);

      /* Define XBraid parameters
       * See -help message forf descriptions */
      int       max_levels    = 3;
      // int       nrelax        = 1;
      // int       skip          = 0;
      double    tol           = 1.e-7;
      // int       cfactor       = 2;
      int       max_iter      = 5;
      // int       min_coarse    = 10;
      // int       fmg           = 0;
      // int       scoarsen      = 0;
      // int       res           = 0;
      // int       wrapper_tests = 0;
      int       print_level   = 1;
      int       access_level  = 1;
      int       use_sequential= 0;

      braid_SetPrintLevel( core, print_level);
      braid_SetAccessLevel( core, access_level);
      braid_SetMaxLevels(core, max_levels);
      //       braid_SetMinCoarse( core, min_coarse );
      //       braid_SetSkip(core, skip);
      //       braid_SetNRelax(core, -1, nrelax);
      braid_SetAbsTol(core, tol);
      //       braid_SetCFactor(core, -1, cfactor);
      braid_SetMaxIter(core, max_iter);
      braid_SetSeqSoln(core, use_sequential);

      app->eq.define();
      app->final_step = ntime;

      braid_Drive(core);

      // Free the memory now that we are done
      braid_Destroy(core);

      delete app;

      // Clean up MPI
      // MPI_Comm_free(&comm);
      MPI_Finalize();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl << exc.what()
                << std::endl << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl << "Aborting!"
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}

