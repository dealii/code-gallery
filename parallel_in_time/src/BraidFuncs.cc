#include "BraidFuncs.hh"

// This advances the solution forward by one time step.
// First some data is collected from the status struct,
// namely the start and stop time and the current timestep
// number. The timestep size $\Delta t$ is calculated,
// and the step function from the HeatEquation is used to
// advance the solution.
int my_Step(braid_App        app,
            braid_Vector     ustop,
            braid_Vector     fstop,
            braid_Vector     u,
            braid_StepStatus status)
{
  UNUSED(ustop);
  UNUSED(fstop);
  double tstart;             /* current time */
  double tstop;              /* evolve to this time*/
  int level;
  double deltaT;

  int index;
  braid_StepStatusGetLevel(status, &level);
  braid_StepStatusGetTstartTstop(status, &tstart, &tstop);
  braid_StepStatusGetTIndex(status, &index);

  deltaT = tstop - tstart;

  dealii::Vector<double>& solution = u->data;

  HeatEquation<2>& heateq = app->eq;

  heateq.step(solution, deltaT, tstart, index);

  return 0;
}


// In this function we initialize a vector at an arbitrary time.
// At this point we don't know anything about what the solution
// looks like, and we can really initialize to anything, so in
// this case use reinit to initialize the memory and set the
// values to zero.
int
my_Init(braid_App     app,
        double        t,
        braid_Vector *u_ptr)
{
  my_Vector *u = new(my_Vector);
  int size = app->eq.size();
  u->data.reinit(size);

  app->eq.initialize(t, u->data);

  *u_ptr = u;

  return 0;
}

// Here we need to copy the vector u into the vector v. We do this
// by allocating a new vector, then reinitializing the deal.ii
// vector to the correct size. The deal.ii reinitialization sets
// every value to zero, so next we need to iterate over the vector
// u and copy the values to the new vector v.
int
my_Clone(braid_App     app,
         braid_Vector  u,
         braid_Vector *v_ptr)
{
  UNUSED(app);
  my_Vector *v = new(my_Vector);
  int size = u->data.size();
  v->data.reinit(size);
  for(size_t i=0, end=v->data.size(); i != end; ++i)
    {
      v->data[i] = u->data[i];
    }
  *v_ptr = v;

  return 0;
}

// Here we need to free the memory used by vector u. This is
// pretty simple since the deal.ii vector is stored inside the
// XBraid vector, so we just delete the XBraid vector u and it
// puts the deal.ii vector out of scope and releases its memory.
int
my_Free(braid_App    app,
        braid_Vector u)
{
  UNUSED(app);
  delete u;

  return 0;
}

// This is to perform an axpy type operation. That is to say we
// do $y = \alpha x + \beta y$. Fortunately deal.ii already has
// this operation built in to its vector class, so we get the
// reference to the vector y and call the sadd method.
int my_Sum(braid_App app,
           double alpha,
           braid_Vector x,
           double beta,
           braid_Vector y)
{
  UNUSED(app);
  Vector<double>& vec = y->data;
  vec.sadd(beta, alpha, x->data);

  return 0;
}

// This calculates the spatial norm using the l2 norm. According
// to XBraid, this could be just about any spatial norm but we'll
// keep it simple and used deal.ii vector's built in l2_norm method.
int
my_SpatialNorm(braid_App     app,
               braid_Vector  u,
               double       *norm_ptr)
{
  UNUSED(app);
  double dot = 0.0;
  dot = u->data.l2_norm();
  *norm_ptr = dot;

  return 0;
}

// This function is called at various points depending on the access
// level specified when configuring the XBraid struct. This function
// is used to print out data during the run time, such as plots of the
// data. The status struct contains a ton of information about the
// simulation run. Here we get the current time and timestep number.
// The output_results function is called to plot the solution data.
// If the method of manufactured solutions is being used, then the
// error of this time step is computed and processed.
int
my_Access(braid_App          app,
          braid_Vector       u,
          braid_AccessStatus astatus)
{
  double     t;
  int index;

  braid_AccessStatusGetT(astatus, &t);
  braid_AccessStatusGetTIndex(astatus, &index);

  app->eq.output_results(index, t, u->data);

#if DO_MFG
  if(index == app->final_step)
    {
      app->eq.process_solution(t, index, u->data);
    }
#endif

  return 0;
}

// This calculates the size of buffer needed to pack the solution
// data into a linear buffer for transfer to another processor via
// MPI. We query the size of the data from the HeatEquation class
// and return the buffer size.
int
my_BufSize(braid_App           app,
           int                 *size_ptr,
           braid_BufferStatus  bstatus)
{
  UNUSED(bstatus);
  int size = app->eq.size();
  *size_ptr = (size+1)*sizeof(double);

  return 0;
}

// This function packs a linear buffer with data so that the buffer
// may be sent to another processor via MPI. The buffer is cast to
// a type we can work with. The first element of the buffer is the
// size of the buffer. Then we iterate over soltuion vector u and
// fill the buffer with our solution data. Finally we tell XBraid
// how much data we wrote.
int
my_BufPack(braid_App           app,
           braid_Vector        u,
           void               *buffer,
           braid_BufferStatus  bstatus)
{

  UNUSED(app);
  double *dbuffer = (double*)buffer;
  int size = u->data.size();
  dbuffer[0] = size;
  for(int i=0; i != size; ++i)
    {
      dbuffer[i+1] = (u->data)[i];
    }
  braid_BufferStatusSetSize(bstatus, (size+1)*sizeof(double));

  return 0;
}

// This function unpacks a buffer that was recieved from a different
// processor via MPI. The size of the buffer is read from the first
// element, then we iterate over the size of the buffer and fill
// the values of solution vector u with the data in the buffer.
int
my_BufUnpack(braid_App           app,
             void               *buffer,
             braid_Vector       *u_ptr,
             braid_BufferStatus  bstatus)
{
  UNUSED(app);
  UNUSED(bstatus);

  my_Vector *u = NULL;
  double *dbuffer = (double*)buffer;
  int size = static_cast<int>(dbuffer[0]);
  u = new(my_Vector);
  u->data.reinit(size);

  for(int i = 0; i != size; ++i)
    {
      (u->data)[i] = dbuffer[i+1];
    }
  *u_ptr = u;

  return 0;
}
