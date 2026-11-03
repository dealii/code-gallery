# PDE-Based Density and Sensitivity Filtering

This note explains why the implementation uses a Helmholtz-type partial-differential-equation (PDE) filter rather than an explicitly assembled convolution filter. The discussion complements the [main README](../../Readme.md).

## Explicit convolution filters

A conventional density filter can be written in discrete form as

@f[
\widetilde{\boldsymbol\rho}=\mathbf H\boldsymbol\rho,
@f]

where the entries of the filter matrix are determined from distances between cell centers, for example

@f[
H_{ij}=w(\|x_i-x_j\|),
@f]

with a weight function $w$ truncated at the filter radius $R_{\min}$. In a typical implementation, the neighborhood of each cell must be identified and the resulting sparse matrix must be constructed and stored.

This approach is straightforward on structured meshes, where regular neighbor patterns can be exploited. On unstructured meshes, the neighborhood search is more involved and may require spatial-search data structures and additional communication between MPI subdomains. The storage and application costs also grow with the number of neighbors included in the filter.

For a quasi-uniform mesh with characteristic cell size $h$, the number of cells in a radius-$R_{\min}$ neighborhood scales approximately as

@f[
O\!\left((R_{\min}/h)^d\right),
@f]

where $d$ is the spatial dimension. Thus, a direct neighborhood filter can require approximately

@f[
O\!\left(N(R_{\min}/h)^d\right)
@f]

nonzero entries for $N$ design cells, up to constants and boundary effects. The precise cost depends on the mesh, neighborhood-search strategy, filter kernel, and parallel partitioning.

## Helmholtz-type PDE filter

The implementation instead obtains the filtered density by solving

@f[
\left(1-R_{\min}^2\Delta\right)\widetilde\rho=\rho
\quad\text{in }\Omega,
@f]

with homogeneous Neumann boundary conditions:

@f[
\nabla\widetilde\rho\cdot n=0
\quad\text{on }\partial\Omega.
@f]

The weak form is: find $\widetilde\rho$ such that, for all test functions $v$,

@f[
\int_\Omega \widetilde\rho v\,\mathrm dx
+R_{\min}^2\int_\Omega\nabla\widetilde\rho\cdot\nabla v\,\mathrm dx
=\int_\Omega\rho v\,\mathrm dx.
@f]

The resulting operator is symmetric positive definite under the usual finite-element assumptions. Its application involves local mass- and stiffness-type contributions, so it does not require explicit cell-neighborhood searches or storage of a nonlocal convolution matrix.

### Why use the PDE filter?

- **Local finite-element operator.** The filter is represented by a mass term and a Laplace term. Its cell-wise action can be implemented using the same matrix-free finite-element machinery as other elliptic operators.
- **No explicit neighborhood graph.** The filter does not require constructing a list of all cells within the filter radius or storing a sparse convolution matrix.
- **Natural parallel decomposition.** The operator is assembled or applied from local element contributions, with communication handled through the usual distributed-vector and multigrid operations.
- **A reusable symmetric operator.** The same Helmholtz operator can be reused for density filtering and for sensitivity smoothing. For the exact discrete adjoint of the density-filter map, the transpose operator and the mass-matrix mapping between the design and filtered-density spaces must be included consistently. Since the Helmholtz operator is symmetric, its transpose is the same operator.
- **Natural zero-flux boundary treatment.** The homogeneous Neumann condition imposes zero normal filter flux at the domain boundary; no explicit exterior-cell neighborhoods or boundary-specific convolution weights are needed.

The PDE formulation itself is not tied to a particular mesh geometry. However, the availability and efficiency of a specific `MatrixFree`/`FEEvaluation` implementation depend on the reference cell, finite-element family, and deal.II interfaces being used. The tensor-product matrix-free kernels in this project should not be interpreted as automatically supporting every tetrahedral or polyhedral discretization without additional implementation work.

## Computational cost and solver behavior

A matrix-free filter application costs work proportional to the number of degrees of freedom for a fixed polynomial degree. With an effective geometric multigrid preconditioner, the work per iteration can also be close to linear in the problem size. This makes the PDE filter attractive for large problems and avoids the explicit nonlocal matrix storage required by a convolution filter.

The number of Krylov iterations is not universally independent of mesh size or of $R_{\min}/h$. It depends on the discretization, filter-radius-to-mesh-size ratio, multigrid hierarchy, transfer operators, smoother, coarse-grid solver, and parallel configuration. The near-constant iteration counts observed in a particular benchmark are empirical results for that setup, not a general guarantee. In this project, the filter is solved with FGMRES preconditioned by geometric multigrid and Chebyshev smoothing.

## Filter radius used in this project

The current implementation chooses

@f[
R_{\min}=\frac{h}{2\sqrt{3}},
\qquad
h=\left(\max_K|K|\right)^{1/d}.
@f]

This is the mesh-dependent choice used in the example. For strongly nonuniform or anisotropic meshes, a single value based on the maximum cell measure may not represent a uniform physical filter length. In such cases, the filter-radius definition should be reconsidered and tested against the intended minimum feature size.

## Sensitivity filtering

The code reuses the Helmholtz operator and supplies the objective sensitivity as a new right-hand side to obtain a smoothed sensitivity field. This avoids constructing and storing a separate convolution matrix or its transpose.

When the density filter is part of the mathematical design map, distinguish between (a) smoothing a sensitivity as an optimization heuristic and (b) computing the exact derivative of the objective with respect to the unfiltered design variables. The latter requires applying the transpose of the discrete density-filter map, including the appropriate mass-matrix factors if the design and filtered fields use different finite-element spaces. Since the implemented Helmholtz operator is symmetric, the same linear operator can be used, but the right-hand-side representation still has to match the discrete map.
