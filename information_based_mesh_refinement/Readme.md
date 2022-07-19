Readme file for "Information density-based mesh refinement"
===========================================================

@note This program implements the ideas and algorithms described in
  the paper "Estimating and using information in inverse problems" by
  Wolfgang Bangerth, Chris R. Johnson, Dennis K. Njeru, and Bart van
  Bloemen Waanders, 2022. See there for more information.

Motivation
----------

Inverse problems are problems where we would like to infer properties
of a system from measurements of the system's state or response to
external stimuli. The specific example this program addresses is that
we want to identify the source term (i.e., right hand side function)
in an advection-diffusion equation from point measurements of the
solution of the equation. A typical application is that we would like
to find out the locations and strengths of pollution sources based on
measuring the concentration of the polluting substance at a number of
points.

It is clear that in order to solve such problems, one needs to "know"
something about the system's state (here: the pollution concentration)
through measurements. Intuitively, it is also clear that we know
"more" about the pollution sources by (i) measuring at more points,
and (ii) by measuring *downstream* from the sources than we would if
we had measured *upstream*. Intuitive concepts such as this motivate
wondering whether we can define an "information density" function
whose value at a point $\mathbf x$ describes how much we know about potential
sources located at $\mathbf x$.

The paper which this code accompanies explores the concept of
information in inverse problems. It defines an "information density"
by solving auxiliary problems for each measurement, and then outlines
possible applications for these information densities in three
vignettes: spatially variable regularization; mesh refinement; and
optimal experimental design. It then considers one of these in detail
through numerical experiments, namely mesh refinement. This program
implements the algorithms shown there and produces the numerical
results.


To run the code
---------------

After running `cmake` and compiling via `make` (or, if you have used
the `-G ...` option of `cmake`, compiling the program via your
favorite integrated development environment), you can run the
executable by either just saying `make run` or using `./mesh_refinement`
on the command line. The default is to compile in "debug mode"; you
can switch to "release mode" by saying `make release` and then
compiling everything again.

The program contains a switch that decides which mesh refinement
algorithm to use. By default, it refines the mesh based on the
information criterion discussed in the paper; it runs a sequence
of 7 mesh refinement cycles. In debug mode, running the program as
is takes about 50 CPU minutes on a reasonably modern laptop. (The
program takes about five and a half minutes in release mode.) It
parallelizes certain operations, so the actual run time may be shorter
depending on how many cores are available.

For each cycle, it outputs the solution as a VTU file, along with the
$A$, $B$, $C$, and $M$ matrices discussed in the paper. These matrices
can then be used to compute the eigenvalues of the $H$ matrix defined
by $H = B^T A^{-T} C A^{-1} B + \beta M$ where $\beta$ is the
regularization parameters.

Some of the pictures shown in the paper are also reproduced as part of
this code gallery program. See the paper for captions and more information.

