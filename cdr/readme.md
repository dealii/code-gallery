# Gallery Entry: deal.II-cdr


## Overview
I started this project with the intent of better understanding adaptive mesh
refinement, parallel computing, and `CMake`. In particular, I started by writing
a uniform mesh, single process solver and then ultimately expanded it into
`solver/cdr.cc`. This example program might be useful to look at if you want to
see:
* A more complex `CMake` setup, which builds a shared object library and an
  executable
* A simple parallel time stepping problem
* Use of `C++11` lambda functions

The other solvers are available [here](http://www.github.com/drwells/dealii-cdr).


## Requirements
* A `C++-11` compliant compiler
* The development version of `deal.II`


## Compiling and running
Like the example programs, run
```
cmake -DDEAL_II_DIR=/path/to/deal.II .
make
```
in this directory. The solver may be run as
```
make run
```
or, for parallelism across `16` processes,
```
mpirun -np 16 ./cdr
```


## Why use convection-diffusion-reaction?
This equation exhibits very fine boundary layers (usually, from the literature,
these layers have width equal to the square root of the diffusion coefficient on
internal layers and the diffusion coefficient on boundary layers). A good way to
solve it is to use adaptive mesh refinement to refine the mesh only at interior
boundary layers. At the same time, this problem is linear (and does not have a
pressure term) so it is much simpler to solve than the Navier-Stokes equations
with comparable diffusion (Reynolds number).

I use relatively large diffusion values so I can get away without any additional
scheme like SUPG.


## Recommended Literature
There are many good books and papers on numerical methods for this equation. A
good starting point is "Robust Numerical Methods for Singularly Perturbed
Problems" by Roos, Stynes, and Tobiska.
