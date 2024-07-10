/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2018 by Joshua Christopher
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 */

#ifndef _BRAIDFUNCS_H_
#define _BRAIDFUNCS_H_

/**
 *   \file BraidFuncs.cc
 *   \brief Contains the implementation of the mandatory X-Braid functions
 *
 *  X-Braid mandates several functions in order to drive the solution.
 *  This file contains the implementation of said mandatory functions.
 *  See the X-Braid documentation for more information.
 *  There are several functions that are optional in X-Braid that may
 *  or may not be implemented in here.
 *
 */


/*-------- Third Party --------*/
#include <deal.II/numerics/vector_tools.h>

#include <braid.h>
#include <braid_test.h>

/*-------- Project --------*/
#include "HeatEquation.hh"

// This struct contains all data that changes with time. For now
// this is just the solution data. When doing AMR this should
// probably include the triangulization, the sparsity pattern,
// constraints, etc.
/**
 * \brief Struct that contains the deal.ii vector.
 */
typedef struct _braid_Vector_struct
{
  dealii::Vector<double> data;
} my_Vector;

// This struct contains all the data that is unchanging with time.
/**
 * \brief Struct that contains the HeatEquation and final
 * time step number.
 */
typedef struct _braid_App_struct
{
  HeatEquation<2> eq;
  int final_step;
} my_App;


/**
 * @brief my_Step - Takes a step in time, advancing the u vector
 *
 * @param app - The braid app struct
 * @param ustop - The solution data at the end of this time step
 * @param fstop - RHS data (such as forcing function?)
 * @param u - The solution data at the beginning of this time step
 * @param status - Status structure that contains various info of this time
 *
 * @return Success (0) or failure (1)
 **/
int my_Step(braid_App        app,
            braid_Vector     ustop,
            braid_Vector     fstop,
            braid_Vector     u,
            braid_StepStatus status);


/**
 * @brief my_Init - Initializes a solution data at the given time
 * For now, initializes the solution to zero no matter what time we are at
 *
 * @param app - The braid app struct containing user data
 * @param t - Time at which the solution is initialized
 * @param u_ptr - The solution data that needs to be filled
 *
 * @return Success (0) or failure (1)
 **/
int
my_Init(braid_App     app,
        double        t,
        braid_Vector *u_ptr);


/**
 * @brief my_Clone - Clones a vector into a new vector
 *
 * @param app - The braid app struct containing user data
 * @param u - The existing vector containing data
 * @param v_ptr - The empty vector that needs to be filled
 *
 * @return Success (0) or failure (1)
 **/
int
my_Clone(braid_App     app,
         braid_Vector  u,
         braid_Vector *v_ptr);


/**
 * @brief my_Free - Deletes a vector
 *
 * @param app - The braid app struct containing user data
 * @param u - The vector that needs to be deleted
 *
 * @return Success (0) or failure (1)
 **/
int
my_Free(braid_App    app,
        braid_Vector u);


/**
 * @brief my_Sum - Sums two vectors in an AXPY operation
 * The operation is y = alpha*x + beta*y
 *
 * @param app - The braid app struct containing user data
 * @param alpha - The coefficient in front of x
 * @param x - A vector that is multiplied by alpha then added to y
 * @param beta - The coefficient of y
 * @param y - A vector that is multiplied by beta then summed with x
 *
 * @return Success (0) or failure (1)
 **/
int
my_Sum(braid_App app,
       double alpha,
       braid_Vector x,
       double beta,
       braid_Vector y);

/**
 *  \brief Returns the spatial norm of the provided vector
 *
 *  Calculates and returns the spatial norm of the provided vector.
 *  Interestingly enough, X-Braid does not specify a particular norm.
 *  to keep things simple, we implement the Euclidean norm.
 *
 *  \param app - The braid app struct containing user data
 *  \param u - The vector we need to take the norm of
 *  \param norm_ptr - Pointer to the norm that was calculated, need to modify this
 *  \return Success (0) or failure (1)
 */
int
my_SpatialNorm(braid_App     app,
               braid_Vector  u,
               double       *norm_ptr);

/**
 *  \brief Allows the user to output details
 *
 *  The Access function is called at various points to allow the user to output
 *  information to the screen or to files.
 *  The astatus parameter provides various information about the simulation,
 *  see the XBraid documentation for details on what information you can get.
 *  Example information is what the current timestep number and current time is.
 *  If the access level (in parallel_in_time.cc) is set to 0, this function is
 *  never called.
 *  If the access level is set to 1, the function is called after the last
 *  XBraid cycle.
 *  If the access level is set to 2, it is called every XBraid cycle.
 *
 *  \param app - The braid app struct containing user data
 *  \param u - The vector containing the data at the status provided
 *  \param astatus - The Braid status structure
 *  \return Success (0) or failure (1)
 */
int
my_Access(braid_App          app,
          braid_Vector       u,
          braid_AccessStatus astatus);

/**
 *  \brief Calculates the size of a buffer for MPI data transfer
 *
 *  Calculates the size of the buffer that is needed to transfer
 *  a solution vector to another processor.
 *  The bstatus parameter provides various information on the
 *  simulation, see the XBraid documentation for all possible
 *  fields.
 *
 *  \param app - The braid app struct containing user data
 *  \param size_ptr A pointer to the calculated size
 *  \param bstatus The XBraid status structure
 *  \return Success (0) or failure (1)
 */
int
my_BufSize(braid_App           app,
           int                 *size_ptr,
           braid_BufferStatus  bstatus);

/**
 *  \brief Linearizes a vector to be sent to another processor
 *
 *  Linearizes (packs) a data buffer with the contents of
 *  some solution state u.
 *
 *  \param app - The braid app struct containing user data
 *  \param u The vector that must be packed into buffer
 *  \param buffer The buffer that must be filled with u
 *  \param bstatus The XBraid status structure
 *  \return Success (0) or failure (1)
 */
int
my_BufPack(braid_App           app,
           braid_Vector        u,
           void               *buffer,
           braid_BufferStatus  bstatus);

/**
 *  \brief Unpacks a vector that was sent from another processor
 *
 *  Unpacks a linear data buffer into the vector pointed to by
 *  u_ptr.
 *
 *  \param app - The braid app struct containing user data
 *  \param buffer The buffer that must be unpacked
 *  \param u_ptr The pointer to the vector that is filled
 *  \param bstatus The XBraid status structure
 *  \return Success (0) or failure (1)
 */
int
my_BufUnpack(braid_App           app,
             void               *buffer,
             braid_Vector       *u_ptr,
             braid_BufferStatus  bstatus);

#endif // _BRAIDFUNCS_H_
