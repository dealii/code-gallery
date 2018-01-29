

# Introduction
This program presents the implementation of an arbitrary order multipoint flux mixed finite element method for the Darcy equation of flow in porous medium and illustrates the use case of the new enhanced Raviart-Thomas finite element for the purposes of local elimination of velocity degrees of freedom.

# Higher order Multipoint Flux Mixed Finite Element methods
Mixed finite element (MFE) methods are commonly used for modeling of fluid flow and transport, as they provide accurate and locally mass conservative velocities and robustness with respect to heterogeneous, anisotropic, and discontinuous coefficients. A main disadvantage of the MFE methods in their standard form is that they result in coupled velocity-pressure algebraic systems of saddle-point type, which restricts the use of efficient iterative solvers (see step-20 for example). One way to address this issue, a special MFE method, the multipoint flux mixed finite  element (MFMFE) method was developed, which reduces to cell-centered finite differences on quadrilateral, hexahedral and simplicial grids, and exhibits robust performance for discontinuous full tensor coefficients. The method was motivated by the multipoint flux approximation (MPFA) method, which was developed as a finite volume method. The method utilizes the trapezoidal quadrature rule for the velocity mass matrix, which reduces it to a block-diagonal form with blocks associated with mesh vertices. The velocities can then be easily eliminated, resulting in a cell-centered pressure system. The aforementioned MFMFE methods are limited to the lowest order approximation. In a recent work we developed a family of arbitrary order symmetric MFMFE methods on quadrilateral and hexahedral grids, that uses the enhanced Raviart-Thomas finite element space and the tensor-product Gauss-Lobatto quadrature rule to achieve a block-diagonal velocity mass-matrix with blocks corresponding to the nodes associated with the veloicty DOFs.

# Formulation of the method
The method is defined as follows: find $(\mathbf{u}_h,p_h) \in \mathbf{V}^k_h\times
W^{k-1}_h$, where $k\ge 1$, 
@f{align}
\left(\mathbf{K}^{-1}\mathbf{u}_h, \mathbf{v} \right)_Q -\left(p_h,\nabla\cdot\mathbf{v}\right) &= -\left\langle\mathcal{R}^{k-1}_h g, \mathbf{v}\right\rangle_{\Gamma_D}, &&\quad \mathbf{v}\in\mathbf{V}^k_h, \nonumber\\
\left(\nabla\cdot\mathbf{u}_h, w\right) &= \left(f,w\right), &&\quad w\in W_h^{k-1}. \nonumber
@f}
Here, $(\cdot,\cdot)_Q$ indicates that the term is to be assembled with the use of Gauss-Lobatto quadrature rule with $k+1$ points. Note that this leads to non-exact integration, however the optimal order of convergence is maintained. Another important point is related to the Dirichlet boundary data $g$, that has to be projected to $Q^{k-1}(\Gamma_D)$, a space of piecewise polynomials of order at most $k-1$. This requirement is needed to obtain the optimal order of convergence both in theory and practice. While this might look like an extra complication for the implementation, one can use the fact that the function evaluated in $k$ Gaussian points is $\mathcal{O}(h^{k+1})$ close to its $L^2$-projection onto the space $Q^k$, hence for the assembling of the RHS we will be using Gaussian quadrature rule of degree $k$. For this method, enhanced Raviart-Thomas space <code>FE_RT_Bubbles</code> of order $k$ is used for the velocity space $\mathbf{V}^k_h$ and the space of discontinuous piecewise polynomials <code>FE_DGQ</code> of order $k-1$ is used for the pressure space $W_h^{k-1}$.

## Reduction to a pressure system and its stencil
Since the DOFs of $\mathbf{V}_h^k(K)$ are chosen as the `dim` vector components at the tensor-product Gauss-Lobatto quadrature points, in the velocity mass matrix obtained from the bilinear form $(\mathbf{K}^{-1} \mathbf{u}_h,\mathbf{v})_Q$, the `dim` DOFs associated with a quadrature point in an element $K$ are completely decoupled from other DOFs in $K$. Due to the continuity of normal components across faces, there are couplings with DOFs from neighboring elements. We distinguish three types of velocity couplings. 

 - The first involves localization of degrees of freedom around each vertex in the grid. Only this type occurs in the lowest order case $k=1$. The number of DOFs that are coupled around a vertex equals the number of faces $n_v$ that share the vertex.  
 - The second type of coupling is around nodes located on faces, but not at vertices. In 2d, these are edge DOFs. In 3d, there are two cases to consider for this type of coupling. One case is for nodes located on faces, but not on edges. The second case in 3d is for nodes located on edges, but not at vertices.
 - The third type of coupling involves nodes interior to the elements, in which case only the `dim` DOFs associated with the node are coupled. 

Due to the localization of DOF interactions described above, the velocity mass matrix obtained from the bilinear form $(\mathbf{K}^{-1} \mathbf{u}_h,\mathbf{v})$, is block-diagonal with blocks associated with the Gauss-Lobatto quadrature points. In particular, in 2d, there are $n_v \times n_v$ blocks at vertices ($n_v$ is the number of neighboring edges), $3 \times 3$ blocks at edge points, and $2 \times 2$ blocks at interior points. In 3d, there are $n_v \times n_v$ blocks at vertices ($n_v$ is the number of neighboring faces), $2n_e \times 2n_e$ blocks
at edge points ($n_e$ is the number of neighboring elements), $5 \times 5$ blocks at face points, and $3 \times 3$ blocks at interior points.

## Elimination procedure
The local elimination procedure is done as follows (it is very similar to the Schur complement approach, except everything is done locally). Having a system of equations corresponding to a particular node $i$
@f{align}
\begin{pmatrix}
	A_i & B_i \\ 
	-B_i^T  & 0
\end{pmatrix} 
\begin{pmatrix}
u \\ p
\end{pmatrix}=
\begin{pmatrix}
f_i \\ g_i
\end{pmatrix},\nonumber
@f}
we first write the velocity in terms of pressure from the first equation in the system, i.e.
@f{align} 
u = A_i^{-1}f - A_i^{-1}B_i p.\nonumber
@f}
Here, $A_i$ are small local matrices (full matrices), that are cheap to invert. We also store their inverses as they are further used in velocity solution recovery. With this, the second equation in the system above yields
@f{align}
B_i^TA_i^{-1}B_i p = g_i - B_i^TA_i^{-1}f,\nonumber
@f} 
where $B_i^TA_i^{-1}B_i$ is a local node's contribution to the global pressure system.
By following the above steps, one gets the global cell-centered SPD pressure matrix with a compact stencil. After solving for the pressure variable, we use the expression for local velocities above in order to recover the global velocity solution.

## Convergence properties
While the proposed schemes can be defined and are well posed on general quadrilateral or hexahedra, for the convergence analysis we need to impose a restriction on the element geometry. This is due to the reduced approximation properties of the MFE spaces on arbitrarily shaped quadrilaterals or hexahedra that our new family of elements inherits as well. However, introducing the notion of $h^2$-parallelograms in 2d and regular $h^2$-parallelepipeds in 3d, one can show that there is no reduction in accuracy. 

A (generalized) quadrilateral with vertices $\mathbf{r}_i$, $i=1,\dots,4$,
is called an $h^2$-parallelogram if
@f{align}
|\mathbf{r}_{34} - \mathbf{r}_{21}|_{\mathbb{R}^{dim}} \le Ch^2,\nonumber
@f} 
and a hexahedral element is called an $h^2$-parallelepiped if all of its faces are $h^2$-parallelograms. Furthermore, an $h^2$-parallelepiped with vertices $\mathbf{r}_i,\, i=1,\dots,8$, is called regular if
@f{align}
|(\mathbf{r}_{21} - \mathbf{r}_{34}) - (\mathbf{r}_{65} - \mathbf{r}_{78})|_{\mathbb{R}^{dim}} \le Ch^3.\nonumber
@f}
With the above restriction on the geometry of an element, the $k$-th order MFMFE method converges with order $\mathcal{O}(h^{k})$ for all variables in their natural norms, i.e. $H_{div}$ for the velocity and $L^2$ for pressure. The method also exhibits superconvergence of order $\mathcal{O}(h^{k+1})$ for pressure variable computed in $k$ Gaussian points.

# Numerical results
We test the method in 2d on a unit square domain. We start with initial grid with $h = \frac14$, and then distort it randomly using <code>GridTools::distort_random()</code> function. The pressure analytical solution is chosen to be
@f{align}
p = x^3y^4 + x^2 + \sin(xy)\cos(xy), \nonumber
@f}
and the full permeability tensor coefficient is given by
@f{align}
\mathbf{K} = 
\begin{pmatrix}
	(x+1)^2 + y^2 & \sin{(xy)} \\ 
	\sin{(xy)}	  & (x+1)^2
\end{pmatrix}.\nonumber
@f}
The problem is then solved on a sequence of uniformly refined grids, with the errors and convergence rates for the case $k=2$ shown in the following table.
| Cycle | Cells | # DOFs |  $\|\mathbf{u} - \mathbf{u}_h\|_{L^2}$  | Rate |  $\|\nabla\cdot(\mathbf{u} - \mathbf{u}_h)\|_{L^2}$  | Rate | $\|p - p_h\|_{L^2}$ | Rate | $\|\mathcal{Q}_h^{1}p - p_h\|_{L^2}$ | Rate |
|-------|-------|--------|------------------------:|----:|-------------------------------------:|----:|----------------------:|----:|-----------------------:|----:|
| 0     | 16    | 280    | 1.24E-01                | -   | 8.77E-01                             | -   | 9.04E-03              | -   | 7.95E-04               | -   |
| 1     | 64    | 1072   | 3.16E-02                | 2.0 | 2.21E-01                             | 2.0 | 2.24E-03              | 2.0 | 1.07E-04               | 2.9 |
| 2     | 256   | 4192   | 7.87E-03                | 2.0 | 5.55E-02                             | 2.0 | 5.59E-04              | 2.0 | 1.43E-05               | 2.9 |
| 3     | 1024  | 16576  | 1.96E-03                | 2.0 | 1.39E-02                             | 2.0 | 1.40E-04              | 2.0 | 1.87E-06               | 2.9 |
| 4     | 4096  | 65920  | 4.89E-04                | 2.0 | 3.47E-03                             | 2.0 | 3.49E-05              | 2.0 | 2.38E-07               | 3.0 |
| 5     | 16384 | 262912 | 1.22E-04                | 2.0 | 8.68E-04                             | 2.0 | 8.73E-06              | 2.0 | 3.01E-08               | 3.0 |

We are also interested in performance of the method, hence the following table summarizes the wall time cost of the different parts of the program for the finest grid (i.e., $k=2$,  $h=\frac{1}{128}$):
| Section                    | wall time | % of total |
|:---------------------------|----------:|-----------:|
| Compute errors             |    0.734s |        13% |
| Make sparsity pattern      |    0.422s |       7.5% |
| Nodal assembly             |    0.965s |        17% |
| Output results             |    0.204s |       3.6% |
| Pressure CG solve          |     1.89s |        33% |
| Pressure matrix assembly   |    0.864s |        15% |
| Velocity solution recovery |   0.0853s |       1.5% |
| Total time                 |     5.64s |       100% |
So one can see that the method solves the problem with 262k unknowns in about 4.5 seconds, with the rest of the time spent for the post-processing. These results were obtained with 8-core Ryzen 1700 CPU and 9.0.0-pre version of deal.II in release configuration.
# References
- I. Ambartsumyan, J. Lee, E. Khattatov, and I. Yotov, <i><a href="https://arxiv.org/abs/1710.06742">Higher order multipoint flux mixed finite 
element methods on quadrilaterals and hexahedra</a></i>, to appear in Math. Comput.
- R. Ingram, M. F. Wheeler, and I. Yotov, <i><a href="http://www.math.pitt.edu/~yotov/research/publications/mfmfe.pdf"> A multipoint 
flux mixed finite element method</a></i>, SIAM J. Numer. Anal., 48:4 (2010) 1281-1312.
- M. F. Wheeler and I. Yotov, <i><a href="http://www.math.pitt.edu/~yotov/research/publications/mfmfe3D.pdf"> A multipoint 
flux mixed finite element method on hexahedra</a></i>, SIAM J. Numer. Anal. 44:5 (2006) 2082-2106.