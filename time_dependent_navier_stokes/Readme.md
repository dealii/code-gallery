Time-dependent Navier-Stokes
------------------------------------------

### General description of the problem ###

We solve the time-dependent Navier-Stokes equations with implicit-explicit (IMEX) scheme.
Here is the equations we want to solve:
@f{eqnarray*}
{\bold{u}}_{,t} - \nu {\nabla}^2\bold{u} + (\bold{u}\cdot\nabla)\bold{u} + \nabla p = \bold{f}
@f}

The idea is as follows: we use backward Euler time for time discretization. The diffusion term
is treated implicitly and the convection term is treated explicitly. Let $(u, p)$ denote the
velocity and pressure, respectively and $(v, q)$ denote the corresponding test functions, we
end up with the following linear system:
@f{eqnarray*}
m(u^{n+1}, v) + \Delta{t}\cdot a((u^{n+1}, p^{n+1}), (v, q))=m(u^n, v)-\Delta{t}c(u^n;u^n, v)
@f}

where $a((u, p), (v, q))$ is the bilinear form of the diffusion term:
@f{eqnarray*}
a((u, p), (v, q)) = \int_\Omega \nu\nabla{u}\nabla{v}-p\nabla\cdot v-q\nabla\cdot ud\Omega
@f}

$m(u, v)$ is the mass matrix:
@f{eqnarray*}
m(u, v) = \int_{\Omega} u \cdot v d\Omega
@f}

and $c(u;u, v)$ is the convection term:
@f{eqnarray*}
c(u;u, v) = \int_{\Omega} (u \cdot \nabla u) \cdot v d\Omega
@f}

Substracting $m(u^n, v) + \Delta{t}a((u^n, p^n), (v, q))$ from both sides of the equation,
we have the incremental form:
@f{eqnarray*}
m(\Delta{u}, v) + \Delta{t}\cdot a((\Delta{u}, \Delta{p}), (v, q)) = \Delta{t}(-a(u^n, p^n), (q, v)) - \Delta{t}c(u^n;u^n, v)
@f}


The system we want to solve can be written in matrix form:

@f{eqnarray*}
    \left(
      \begin{array}{cc}
        A & B^{T} \\
        B & 0 \\
      \end{array}
    \right)
    \left(
      \begin{array}{c}
        U \\
        P \\
      \end{array}
    \right)
    =
    \left(
      \begin{array}{c}
        F \\
        0 \\
      \end{array}
    \right)
@f}

#### Grad-Div stablization ####

Similar to step-57, we add $\gamma B^T M_p^{-1} B$ to the upper left block of the system,
thus the system becomes:

@f{eqnarray*}
    \left(
      \begin{array}{cc}
        \tilde{A} & B^{T} \\
        B & 0 \\
      \end{array}
    \right)
    \left(
      \begin{array}{c}
        U \\
        P \\
      \end{array}
    \right)
    =
    \left(
      \begin{array}{c}
        F \\
        0 \\
      \end{array}
    \right)
@f}
where $\tilde{A} = A + \gamma B^T M_p^{-1} B$.

Detailed explaination of the Grad-Div stablization can be found in [1].

#### Block preconditioner ####

The block preconditioner is pretty much the same as in step-22, except for two additional terms,
namely the inertial term (mass matrix) and the Grad-Div term.

The block preconditioner can be written as:
@f{eqnarray*}
    P^{-1}
    =
    \left(
      \begin{array}{cc}
        {\tilde{A}}^{-1} & 0 \\
        {\tilde{S}}^{-1}B{\tilde{A}}^{-1} & -{\tilde{S}}^{-1} \\
      \end{array}
    \right)
@f}
where ${\tilde{S}}$ is the Schur complement of ${\tilde{A}}$, which can be decomposed 
into the Schur complements of the mass matrix, diffusion matrix, and the Grad-Div term:
@f{eqnarray*}
    {\tilde{S}}^{-1}
    \approx 
    {S_{mass}}^{-1} + {S_{diff}}^{-1} + {S_{Grad-Div}}^{-1}
    \approx 
    [B^T (diag M)^{-1} B]^{-1} + \Delta{t}(\nu + \gamma)M_p^{-1}
@f}

For more information about preconditioning incompressible Navier-Stokes equations, please refer
to [1] and [2].

#### Test case ####
We test the code with a classical benchmark case, flow past a cylinder, with Reynold's
number 100. The geometry setup of the case can be found on
[this webpage](http://www.featflow.de/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html).

### Acknowledgements ###
Thank Wolfgang Bangerth, Timo Heister and Martin Kronbichler for their helpful discussions
on my numerical formulation and implementation.

------------------------------------------
### References ###
[1] Timo Heister. A massively parallel finite element framework with application to incompressible flows. Doctoral dissertation, University of Gottingen, 2011.

[2] M. Kronbichler, A. Diagne and H. Holmgren. A fast massively parallel two-phase flow solver for microfluidic chip simulation, International Journal of High Performance Computing Applications, 2016.