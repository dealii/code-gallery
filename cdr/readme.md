## Overview
I started this project with the intent of better understanding adaptive mesh
refinement, parallel computing, and <code>CMake</code>. In particular, I started
by writing a uniform mesh, single process solver and then ultimately expanded it
into <code>solver/cdr.cc</code>. This example program might be useful to look at
if you want to see:
<ul>
  <li>A more complex `CMake` setup, which builds a shared object library and an
  executable</li>
  <li>A simple parallel time stepping problem</li>
  <li>Use of `C++11` lambda functions</li>
</ul>

The other solvers are available [here](http://www.github.com/drwells/dealii-cdr).

Unlike the other tutorial programs, I have split this solver into a number of
files in nested directories. In particular, I used the following strategy
(more-or-less copied [ASPECT](http://aspect.dealii.org)):
<ul>
  <li>The <code>common</code> directory, which hosts files common to the four
  solvers I wrote along the way. Most of the source files in
  <code>common/source/</code> are just <i>template specializations</i>; they
  compile the template code for specific dimensions and linear algebra (matrix
  or vector) types. The <code>common/include/deal.II-cdr/</code> directory
  contains both templates and plain header files.</li>
  <li>The <code>solver/</code> directory contains the actual solver class and
  strongly resembles a tutorial program. The file <code>solver/cdr.cc</code>
  just sets up data structures and then calls routines in
  <code>libdeal.II-cdr-common</code> to populate them and produce output.</li>
</ul>




## Requirements
<ul>
  <li>A <code>C++-11</code> compliant compiler</li>
  <li>Version 8.4 of <code>deal.II</code></li>
</ul>


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
