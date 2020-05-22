#ifndef _UTILITIES_H_
#define _UTILITIES_H_

#include <iostream>

// This preprocessor macro is used on function arguments
// that are not used in the function. It is used to
// suppress compiler warnings.
#define UNUSED(x) (void)(x)

// Contains the current MPI processor ID.
extern int procID;

// Function to return the ostream to write out to. In MPI
// mode it returns a stream to a file named pout.<#> where
// <#> is the procID. This allows the user to write output
// from each processor to a separate file. In serial mode
// (no MPI), it returns the standard output.
std::ostream& pout();
#endif
