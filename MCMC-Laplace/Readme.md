Readme file for MCMC-Laplace
============================

@note The intent and implementation of this program is extensively
  described in D. Aristoff and W. Bangerth: "A benchmark for the Bayesian
  inversion of coefficients in partial differential equations",
  submitted, 2021. A preprint can be found
  [here](https://arxiv.org/abs/2102.07263). See there for more information.

Motivation for project
----------------------

Inverse problems are problems in which one (typically) wants to infer
something about the internal properties of body by measuring how it
reacts to an external stimulus. An example would be that you want to
determine the stiffness parameters of a membrane by applying an
external force to it and measuring how it deforms. A more complicated
inverse problem is determining the three-dimensional make-up of the
Earth by measuring the time it takes for seismic waves to travel
from the source of an Earthquake to far-away detectors. Most
biomedical imaging techniques are also inverse problems.

The traditional approach to inverse problems is to ask the question
which hypothesized make-up of the body would result in predicted
reactions that are "closest" to the measured one. This formulation of
the problem is what is now generally called the "deterministic inverse
problem", and it is an optimization problem: Among all possible
make-ups of the body, find the one which *minimizes* the difference between
predicted measurements and actual measurements.

Since the late 1990s, a second paradigm for the formulation has come
into play: "Bayesian inverse problems". It rests on the observation
that our measurements are not exact but rather that certain values are
just more or less likely to show up on the dial of the instrument we
measure with. For example, if a device measures the
deformation of a membrane as 2.85 cm, and if we know that the measuring
device has a Gaussian-distributed uncertainty with standard deviation
0.05 cm, then the Bayesian inverse problem asks for finding a probability
distribution among all of the make-ups of the body so that the
predicted measurements have the observed distribution of a Gaussian
with mean 2.85 cm and standard deviation 0.05 cm.

To make things more concrete, let us denote the parameters that
describe the internal make-up of the membrane as the vector $\mathbf
a$, and the measured deflections at a set of measurement points as
$\mathbf z$. Assume that we have measured a set of values $\hat
{\mathbf z}$,
and that we know that each of these measurements is normal distributed
with standard deviation $\sigma$, i.e., that the "real" values are
$\mathbf z \sim N(\hat {\mathbf z}, \sigma I)$ -- i.e., normally
distributed with mean $\hat {\mathbf z}$ and covariance matrix $\sigma
I$.

Let us further assume that for each set of parameters $\mathbf a$, we
can predict measurements $\mathbf z=\mathbf F(\mathbf a)$ with some
function $\mathbf F(\cdot)$ that in general will involve solving a
partial differential equation with known external applied force and
given trial coefficients $\mathbf a$. What we are interested in is
what the probability distribution $\pi(\mathbf a)$ is so that the
corresponding $\pi(\mathbf z)=\pi(\mathbf F(\mathbf a))=N(\hat{\mathbf
z},\sigma I)$. This problem can, in general, not be solved exactly
because we only know $\mathbf F$, the parameters-to-measurements map
that can be evaluated by solving the PDE and then evaluating the
solution at individual points, but not the inverse of $\mathbf
F$. But, it is possible to *sample* from the distribution $\pi(\mathbf
a)$ using Monte Carlo Markov Chain (MCMC) methods.

This is what this program does, in essence. The formulation of the
problem is marginally more complicated than outlined above, also
taking into account a prior distribution that describes some
assumptions we may have on the parameter. But in essence, this is what
we are doing:

- There is a Metropolis-Hastings that implements a Markov Chain to
  sample from $\pi(\mathbf a)$. The Markov chain that results from
  this is then a (very long) sequence of samples $\mathbf a_1, \mathbf
  a_2, \ldots$ that are written to a (potentially very large) output
  file. If you don't know what a Metropolis-Hastings sampler is, or
  how a sequence of samples approximates a probability distribution,
  then you will probably want to take a look at the Wikipedia pages
  for
  [Markov Chain Monte
  Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo)
  methods and for the
  [Metropolis-Hastings
  algorithm](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm).

- As part of the sampling process, we need to solve the PDE that
  describes the physical system here.

- The remainder of the program is devoted to the description of the
  distribution $\pi(\mathbf z)$ we provide as input, as well as a
  number of other pieces of information that enter into the definition
  of what exactly the Metropolis-Hastings sampler does here.

@note This program computes samples the brute force way. We want to
  compute billions of samples using a simple algorithm because we want
  a benchmark that smarter algorithms can be tested again. The point
  isn't that the Metropolis-Hastings algorithm as implemented here
  (using, in particular, an isotropic proposal distribution) is the
  sharpest tool in the shed -- it isn't -- but that it is reliable and
  robust. We want to see what it converges to, and how fast, so that
  we can test better sampling methods against this baseline.


More detailed properties of the code in MCMC-Laplace
----------------------------------------------------

To be concise, the problem we are considering is the following: We are
assuming that the membrane we are deforming through an external force
is a square with edge length 1 (i.e., the domain is
$\Omega=(0,1)^2$) and that it is made up of $8\times 8$ smaller
squares each of which has a constant stiffness $a_k,
k=0,\ldots,63$. In other words, we would like to find the vector
$\mathbf a=(a_0,\ldots,a_{63})^T$ for which the predicted deformation
matches our measurements $\hat{\mathbf z}$ in the sense discussed
above.

The model of deformation we consider is the Poisson equation with a
non-constant coefficient:
@f{align*}{
  -\nabla \cdot (a(\mathbf x) \nabla u(\mathbf x) &= f(\mathbf x)
  \qquad\qquad &&\text{in}\ \Omega,
  \\
  u(\mathbf x) &= 0
  \qquad\qquad &&\text{on}\ \partial\Omega.
@f}
Here, the spatially variable coefficient $a(\mathbf x)$ corresponds to
the 64 values in $\mathbf a$ by mapping the elements of $\mathbf a$ to
regions of the mesh. We choose $f=10$, which results in a solution
that is approximately equal to one at its maximum. The following
picture shows this solution $u$:
![Solution u(x)](./doc/exact-solution.png)
The coefficient values that correspond to this solution (the "exact"
coefficient from which the measurements $\hat{\mathbf z}$ were
generated) looks as follows:
![Exact coefficient a(x)](./doc/exact-coefficient.png)

For every given coefficient $\mathbf a$, the corresponding measurement
values $z_i, i=0,\ldots,168$ are then obtained by evaluating the
solution $u$ on a $13\times 13$ grid of equidistance points $\mathbf
x_i$.

You will find these concepts mapped into the code as part of the
`PoissonSolver` class. Of particular interest may be the fact that the
computation of $\mathbf z$ by evaluating $u$ at individual points is a
linear operation, and consequently can be represented using a matrix
applied to the solution vector. (In the code, this corresponds to the
`PoissonSolver::measurement_matrix` member variable.) Furthermore, we
make the assumption that the mesh used in solving the PDE is at least
as fine as the $8\times 8$ mesh used to represent the coefficient
$\mathbf a$ we would like to infer; then, the coefficient is constant
on each cell, and we can get the value of the coefficient on a given
cell by looking up the corresponding value of the element of the
vector $\mathbf a$. We store the index of this vector element in the
`user_index` property that deal.II provides for each cell, and set
this connection up in `PoissonSolver::setup_system()`.

The only other part worth discussing about this program is that it is
set up for *speed*. This program implementing a benchmark, we are
interested generating as many samples as possible -- the paper
mentioned at the top of this page shows data obtained from more than
$10^{10}$ samples. To compute this many samples, solving the PDE
cannot take too long or we would never finish the paper. The question
then is how, given a set of coefficients $\mathbf a$, we can assemble
and solve the linear systems for the Poisson equation as quickly as
possible. In the current program, this is done using the observation
that the local contribution to the global matrix is simply a matrix
that is the same for every cell (because we are using a mesh in which
every cell looks the same) times the coefficient for the current
cell. This is because we know that the coefficient is constant on
every cell, as discussed above. As a consequence, we compute the local
matrix (with a unit coefficient) only once, in
`PoissonProblem::setup_system()`, using the very first cell. We do the
same with the local right hand side vector, which is again the same
for every cell because the right hand side function is constant.

During assembly of the linear system, we then only need to recall
these local matrix and right hand side contributions, multiply the
local matrix by the coefficient of the current cell, and then copy
everything into the global matrix as usual.

When solving the linear system, it turns out that the problems we
consider are small enough that a direct solver (specifically, the
`SparseDirectUMFPACK` class) is the fastest method.


To run the code
---------------

After running `cmake` and compiling via `make` (or, if you have used
the `-G ...` option of `cmake`, compiling the program via your
favorite integrated development environment), you can run the
executable by either just saying `make run` or using `./mcmc-laplace`
on the command line. The default is to compile in "debug mode"; you
can switch to "release mode" by saying `make release` and then
compiling everything again.

The program as is will run in around 40 seconds on a current machine
at the time of writing this program when compiled in release
mode. This is in the test mode that is the default setting selected in
the `main()` function, and it produces 10,000 samples. This is enough
to get an idea of what the program does. For real simulations, such as
those discussed in the paper referenced at the top, one of course
wants to have many many more samples; if you select `testing = false`
at the top of `main()`, the program will create
250*60*60*24*30=648,000,000 samples, which will take around a month to
run in release mode. That may be more than you've bargained for, but
you can always terminate the program, or just select a smaller number
of samples at the bottom of `main()`.

When not in testing mode, the program initializes all random number
generators that are part of the Metropolis-Hastins algorithms with a
seed that is created using the
[`std::random_device()`](https://en.cppreference.com/w/cpp/numeric/random/random_device)
function, a function that uses the operating system to create a seed
that may take into account the current time, the amount of data
written to disk over the past hour, the amount of internet traffic
that has gone through the machine in the last hour, and similar pieces
of pretty much random information. As a consequence, the seed is then
pretty much guaranteed to be different from program invokation to
program invokation, and consequently we will get different random
number sequences every time. The output file is tagged with a string
representation of this random seed, so that it is safe to run the same
program multiple times at the same time in the same directory, with
each running program writing a different sequence of samples into
separate files.

The end result of the program is a file that contains the
samples. Each line has 66 entries:
- The first entry is the logarithm of the (non-normalized) posterior
  probability of the sample; because the posterior is only known up to
  a normalization constant, the absolute value is not relevant, but
  the relative values of different samples are informative.
- The second entry is the number of samples accepted up to this
  point. By counting how many lines one is into a given file (i.e.,
  counting the total number of samples up to this point), this number
  is useful to compute the acceptance rate of the Metropolis-Hastings
  algorithm.
- The remaining 64 numbers are the entries of the current sample
  vector.
