#include <iostream>

#include "braid.h"
#include "braid_test.h"

#include "BraidFuncs.hh"

int main(int argc, char** argv)
{
  MPI_Comm comm;
  int rank;
  MPI_Init(&argc, &argv);
  comm   = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);

  my_App *app = new(my_App);
  app->eq.define();

  double time = 0.2;

  braid_Int init_access_result = braid_TestInitAccess(app,
                                                      comm,
                                                      stdout,
                                                      time,
                                                      my_Init,
                                                      my_Access,
                                                      my_Free);

  braid_Int clone_result = braid_TestClone(app,
                                           comm,
                                           stdout,
                                           time,
                                           my_Init,
                                           my_Access,
                                           my_Free,
                                           my_Clone);

  braid_Int sum_result = braid_TestSum(app,
                                       comm,
                                       stdout,
                                       time,
                                       my_Init,
                                       my_Access,
                                       my_Free,
                                       my_Clone,
                                       my_Sum);

  braid_Int norm_result = braid_TestSpatialNorm(app,
                                                comm,
                                                stdout,
                                                time,
                                                my_Init,
                                                my_Free,
                                                my_Clone,
                                                my_Sum,
                                                my_SpatialNorm);

  braid_Int buf_result = braid_TestBuf(app,
                                       comm,
                                       stdout,
                                       time,
                                       my_Init,
                                       my_Free,
                                       my_Sum,
                                       my_SpatialNorm,
                                       my_BufSize,
                                       my_BufPack,
                                       my_BufUnpack);
  //     /* Create spatial communicator for wrapper-tests */
  //     braid_SplitCommworld(&comm, 1, &comm_x, &comm_t);
// 
//     braid_TestAll(app, comm_x, stdout, 0.0, (tstop-tstart)/ntime,
//                   2*(tstop-tstart)/ntime, my_Init, my_Free, my_Clone,
//                   my_Sum, my_SpatialNorm, my_BufSize, my_BufPack,
//                   my_BufUnpack, my_Coarsen, my_Interp, my_Residual, my_Step);

  /* Finalize MPI */
  MPI_Finalize();
}
