#include "Utilities.hh"

#include <string>
#include <fstream>

#include <mpi.h>

int procID = 0;

// The shared variables

static std::string       s_pout_filename ;
static std::string       s_pout_basename ;
static std::ofstream     s_pout ;

static bool              s_pout_init = false ;
static bool              s_pout_open = false ;

#ifdef USE_MPI
// in parallel, compute the filename give the basename
//[NOTE: dont call this before MPI is initialized.]
static void setFileName()
{
  static const size_t ProcnumSize = 1 + 10 + 1 ;  //'.' + 10digits + '\0'
  char procnum[ProcnumSize] ;
  snprintf( procnum ,ProcnumSize ,".%d" ,procID);
            s_pout_filename = s_pout_basename + procnum ;
}

// in parallel, close the file if nec., open it and check for success
static void openFile()
{
  if ( s_pout_open )
  {
    s_pout.close();
  }
  s_pout.open( s_pout_filename.c_str() );
  // if open() fails, we have problems, but it's better
  // to try again later than to make believe it succeeded
  s_pout_open = (bool)s_pout ;
}

#else
// in serial, filename is always cout
static void setFileName()
{
  s_pout_filename = "cout" ;
}

// in serial, this does absolutely nothing
static void openFile()
{
}
#endif

std::ostream& pout()
{
#ifdef USE_MPI
  // the common case is _open == true, which just returns s_pout
  if ( ! s_pout_open )
    {
      // the uncommon cae: the file isn't opened, MPI may not be
      // initialized, and the basename may not have been set
      int flag_i, flag_f;
      MPI_Initialized(&flag_i);
      MPI_Finalized(&flag_f);
      // app hasn't set a basename yet, so set the default
      if ( ! s_pout_init )
        {
          s_pout_basename = "pout" ;
          s_pout_init = true ;
        }
      // if MPI not initialized, we cant open the file so return cout
      if ( ! flag_i || flag_f)
        {
          return std::cout; // MPI hasn't been started yet, or has ended....
        }
      // MPI is initialized, so file must not be, so open it
      setFileName() ;
      openFile() ;
      // finally, in case the open failed, return cout
      if ( ! s_pout_open )
        {
          return std::cout ;
        }
    }
  return s_pout ;
#else
  return std::cout;
#endif
}
