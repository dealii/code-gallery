# Parallel Flow Routing

This code implements a distributed-memory parallel solver for water flow routing on
large-scale topographic domains. The key problem it solves is: **Given a digital
elevation model (DEM) and knowledge of how much water originates where (e.g., through
rainfall), how much water flows at every point of the landscape?**

## Problem Description

Flow routing is a fundamental problem in terrain analysis and hydrology with many
applications:
- Predicting where water-driven erosion or landslides might occur
- Anticipating flooding risks
- Understanding river drainage patterns
- Computing drainage areas (the upstream area that contributes water to a point)

The traditional approach uses a sequential "high-to-low" algorithm that processes
grid points in decreasing order of elevation. This algorithm is elegant and efficient
for single-core machines, but it is inherently sequential. On the other hand,
modern digital elevation
models can have billions of data points that require thousands of parallel processors
to handle efficiently.

## The Parallel Algorithm

This code reformulates flow routing as solving a system of linear equations. The key
insight is that water conservation at each point can be expressed mathematically:
the water available at a point equals the rainfall plus any water received from
upstream neighbors. By reformulating this as $A \mathbf{w} = \mathbf{r}$ (where
$\mathbf{w}$ is the water flow rate, $A$ encodes the flow topology, and $\mathbf{r}$
is the rainfall), we can use parallel iterative solvers.

The algorithm uses four different optimized formulations:

1. **Matrix-based**: Explicitly stores the sparse matrix $A$ and uses a block-diagonal
   preconditioner based on per-process high-to-low operations.

2. **Matrix-free**: Avoids storing the matrix by computing matrix-vector products
   on-the-fly from the local flow routing information.

3. **I+X scheme**: Combines the matrix and preconditioner into a single operator
   $I+X=D^{-1}A$ where $D$ contains the diagonal blocks. This reduces operations
   compared to applying $A$ and the preconditioner separately.

4. **(I-X)(I+X) scheme**: Further preconditions the $I+X$ operator using $I-X$,
   inspired by the approximation $(I+X)^{-1} \approx I-X$. This reduces iterations
   by approximately half at the cost of doubling work per iteration.

## Key Features of the Code

The main features of this code are the following:

- **Distributed memory parallelism**: It uses MPI to partition the domain among many
  processes. Each process stores only its subset of the DEM plus one layer of ghost
  nodes from neighbors, as is usual in deal.II.

- **Scalable to large problems**: Can handle digital elevation models with over a
  billion points by distributing memory and computation across supercomputers.

- **Exact sequential behavior**: When run with a single process, all methods are
  mathematically equivalent to the classical high-to-low algorithm and consequently
  very fast.

- **Excellent Speedups**: Achieves speedups of 50-100+ on 100+ processes for
  realistically-sized problems. The paper shows results with up to many thousands of
  cores, with problem sizes of 1.88 billion grid points.

- **Depression Filling**: Preprocesses the DEM using the priority-flood algorithm
  to fill local depressions, ensuring all water can flow to the boundary.

The implementation is an extension of the implicit drainage area (IDA) algorithm of Richardson,
Hill, and Perron, but with substantial improvements through better preconditioner
selection and specialized algorithms.

## Building and Running

```
cmake -DDEAL_II_DIR=/path/to/dealii .
make
make run
```

The program reads a digital elevation model (provided for Colorado), optionally
refines the mesh to different resolutions, and solves the flow routing problem.

## Example Output

The program computes the water flow rate for the Colorado topography assuming a
spatially constant rainfall rate of 375 liters/(m² year) (approximately Colorado's
eastern plains rainfall). It generates VTU output files showing where water accumulates.

The program validates correctness by checking that the total outflow rate from the
domain equals the total rainfall rate to better than $10^{-11}$ relative error. Sample
output using six MPI processes with the given input file looks like this:
```
Making grid... 
   Number of cells: 458752
   Area of the domain: 2.69008e+11m^2
Setting up system... 
   Number of degrees of freedom: 920322 (elevation: 460161, waterflow: 460161)
   Number of waterflow degrees of freedom per process: 76693 (average) x 6 (number of processes)
Interpolating elevation... 
   Reading elevation data from cache file colorado-topography-1800m.cache7
   Read 897 x 513 elevation points from cache file
Sorting DoFs high to low... 
   Elevations range between 1003.08m and 4304.08m.
Computing local routing... 
Assembling linear system... 
Assembling matrix-free operators... 
Solving for global water routing... 
   Solved matrix-based in 14 iterations.
   Solved matrix-free in 14 iterations.
   Solved I+X-based in 14 iterations.
   Solved (I-X)(I+X)-based in 7 iterations.
   Relative error between matrix-based and matrix-free: 6.88966e-16
   Relative error between matrix-based and I+X solution: 6.49799e-16
   Relative error between matrix-based and (I-X)(I+X) solution: 6.1953e-16
Conservation check (water)
   Input:          1.00873e+11
   Output:         1.00873e+11
   Relative error: 2.27204e-13
Writing output... 

+-------------------------------------------------------+------------+------------+
| Total wallclock time elapsed since start              |      1.07s |            |
|                                                       |            |            |
| Section                                   | no. calls |  wall time | % of total |
+-------------------------------------------+-----------+------------+------------+
| Compute local routing                     |         1 |    0.0975s |       9.1% |
| Initial conditions: interpolate elevation |         1 |    0.0931s |       8.7% |
| Initial conditions: sort DoFs high to low |         1 |    0.0824s |       7.7% |
| Make grid                                 |         1 |     0.178s |        17% |
| Output                                    |         1 |     0.314s |        29% |
| Setup system                              |         1 |    0.0466s |       4.4% |
| Solver 1: Assemble system                 |         1 |     0.105s |       9.8% |
| Solver 1: Solve for water matrix-based    |         1 |    0.0203s |       1.9% |
| Solver 2: Assemble matrix-free operators  |         1 |   0.00233s |      0.22% |
| Solver 2: Solve for water matrix-free     |         1 |    0.0528s |       4.9% |
| Solver 3: Solve for water I+X             |         1 |    0.0299s |       2.8% |
| Solver 4: Solve for water (I-X)(I+X)      |         1 |    0.0308s |       2.9% |
| Water conservation check                  |         1 |    0.0155s |       1.5% |
+-------------------------------------------+-----------+------------+------------+

Times per section: 0.0974998 0.0931045 0.0823681 0.178445 0.313694 0.0465547 0.104885 0.0202612 0.00232828 0.0528359 0.0298852 0.0308111 0.0155219
```
For the much larger problems in the paper, output would look like this:
```
Making grid... 
   Number of cells: 1879048192
   Area of the domain: 2.68989e+11m^2
Setting up system... 
   Number of degrees of freedom: 3758276610 (elevation: 1879138305, waterflow: 1879138305)
   Number of waterflow degrees of freedom per process: 305849 (average) x 6144 (number of processes)
Interpolating elevation... 
   Reading elevation data from cache file colorado-topography-1800m.cache13
   Read 57345 x 32769 elevation points from cache file
Sorting DoFs high to low... 
   Elevations range between 1001.57m and 4359.87m.
Computing local routing... 
Assembling linear system... 
Assembling matrix-free operators... 
Solving for global water routing... 
   Solved matrix-based in 245 iterations.
   Solved matrix-free in 245 iterations.
   Solved I+X-based in 243 iterations.
   Solved (I-X)(I+X)-based in 122 iterations.
   Relative error between matrix-based and matrix-free: 6.82606e-15
   Relative error between matrix-based and I+X solution: 1.77314e-09
   Relative error between matrix-based and (I-X)(I+X) solution: 9.04634e-10
Conservation check (water)
   Input:          1.00873e+11
   Output:         1.00873e+11
   Relative error: 1.72445e-14


+-------------------------------------------------------+------------+------------+
| Total wallclock time elapsed since start              |       275s |            |
|                                                       |            |            |
| Section                                   | no. calls |  wall time | % of total |
+-------------------------------------------+-----------+------------+------------+
| Compute local routing                     |         1 |     0.883s |      0.32% |
| Initial conditions: interpolate elevation |         1 |       236s |        86% |
| Initial conditions: sort DoFs high to low |         1 |     0.752s |      0.27% |
| Make grid                                 |         1 |      2.35s |      0.85% |
| Setup system                              |         1 |     0.623s |      0.23% |
| Solver 1: Assemble system                 |         1 |      1.23s |      0.45% |
| Solver 1: Solve for water matrix-based    |         1 |      5.25s |       1.9% |
| Solver 2: Assemble matrix-free operators  |         1 |    0.0824s |         0% |
| Solver 2: Solve for water matrix-free     |         1 |        13s |       4.7% |
| Solver 3: Solve for water I+X             |         1 |      7.75s |       2.8% |
| Solver 4: Solve for water (I-X)(I+X)      |         1 |      6.45s |       2.3% |
| Water conservation check                  |         1 |     0.145s |         0% |
+-------------------------------------------+-----------+------------+------------+

Times per section: 0.883375 236.296 0.752251 2.34596 0.622755 1.23387 5.24744 0.0823774 12.9801 7.75294 6.44665 0.144779 
```

## Mathematical Background

The reformulation of flow routing as a linear system is based on the water
conservation principle. In the $D_8$ routing scheme (where each node routes water
to one of its 8 neighbors), the linear system matrix is sparse with at most two
entries per column: a +1 on the diagonal and a -1 entry indicating which nodes
sends water to the node of the current node.

The key innovation is recognizing that when nodes are sorted high-to-low (within
each parallel partition), the diagonal blocks of the matrix are triangular. This
makes them easy to invert exactly, providing an excellent preconditioner:
$B = (D^{pp\downarrow})^{-1}$ where $D^{pp\downarrow}$ contains the diagonal blocks.

Each preconditioning operation applies the high-to-low algorithm independently
within each process's partition, which can be done in parallel.


## References

This code implements the methods described in:
- W. Bangerth (2026), "Massively parallel flow routing and drainage area determination,"
  submitted.

It builds upon this paper:
- Richardson, A., Hill, C. N., & Perron, J. T. (2014). IDA: An implicit, parallelizable
  method for calculating drainage area. Water Resources Research, 50 (5), pp. 4110-4130.
  Retrieved from https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1002/2013WR014326.
  DOI: 10.1002/2013WR014326.

The Colorado topography data comes from OpenTopography and is based on the
Copernicus Global Digital Elevation Models (90 meters resolution), downsampled
to 1800 meters for efficiency. This elevation model is stored in the file
`colorado-topography-1800m.txt.gz` in the current directory and is derived from
data obtained from the European Space Agency's Copernicus program, see 
https://doi.org/10.5069/G9028PQB.
