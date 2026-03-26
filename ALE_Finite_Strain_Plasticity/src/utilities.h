#ifndef _UTILITIES_H
#define _UTILITIES_H

#include <deal.II/base/utilities.h>
#include <mpi.h>

namespace PlasticityLab {

  static MPI_Comm mpi_communicator(MPI_COMM_WORLD);

  struct NotImplementedException : std::exception {
    const char *what() const _GLIBCXX_USE_NOEXCEPT {
      return "Not Implemented.!\n";
    }
  };

} /*namespace PlasticityLab*/


#endif /*_UTILITIES_H*/
