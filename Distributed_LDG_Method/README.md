# Distributed Local Discontinuous Galerkin Methods

## Introduction
This code is designed to numerically solve the 
<a href="https://en.wikipedia.org/wiki/Poisson's_equation">Poisson equation</a>

@f{align}
- \nabla \cdot  \left(\ \nabla u \ \right)&= f(\textbf{x}) && \mbox{in} \
\Omega,\nonumber \\
-\nabla u \cdot \textbf{n}  &= g_{N}(\textbf{x}) && \mbox{on} \ \partial 
\Omega_{N} \nonumber\\
u &= g_{D}(\textbf{x}) && \mbox{on}  \ \partial \Omega_{D}. \nonumber
@f}

in 2D and 3D using the local discontinuous Galerkin (LDG) method</a> from 
scratch. The tutorial codes step-12 and step-39 use the MeshWorker interface
to build
<a href="https://en.wikipedia.org/wiki/Discontinuous_Galerkin_method">
discontinuous Galerkin (DG) methods</a>. While this is very convenient, 
I could not use this framework for solving my research problem and I 
needed to write the LDG method from scratch. I thought it 
would be helpful for others to have access to 
this example that goes through writing a discontinuous Galerkin method from 
scratch and also shows how to do it in a distributed setting using the 
<a href="https://www.trilinos.org">Trilinos</a> library. This example may also
be of interest to users that wish to use the LDG method, as the method is 
distinctly different from the 
<a href="http://www3.nd.edu/~zxu2/acms60790S14/unified-analy-dg-elliptic-eq.pdf">
Interior Penalty Discontinuous Galerkin (IPDG)</a>
methods and was not covered in other tutorials on DG methods.  The LDG method
is very useful when one is working with a differential equation and desires 
both approximations to the scalar unknown function as well as its flux. 
The application of a mixed method offers a mechanism whereby one 
can obtain both the scalar unknown function as well as its flux, however, 
the LDG method has fewer degrees of freedom compared to the 
<a href="https://link.springer.com/chapter/10.1007/BFb0064470">mixed method with 
the Raviart-Thomas element</a>. It also approximates the scalar unknown function 
and its flux using discontinuous polynomial basis functions and are much more 
suitable when one wishes to use local refinement.

## Compiling and Running
To generate a makefile for this code using CMake, type the following command 
into the terminal from the main directory:

	cmake . -DDEAL_II_DIR=/path/to/deal.II

To compile the code in debug mode use:

	make

To compile the code in release mode use:

	make release	

Either of these commands will create the executable, <code>main</code>, 
however the release mode will make a faster executable.

To run the code on <code>N</code> processors type the following command into
the terminal from the main directory,

	mpirun -np N ./main

The output of the code will be in <code>.vtu</code> and <code>.pvtu</code> 
format and be written to disk in parallel.  The results can be viewed using 
<a href="http://www.paraview.org/">ParaView</a>. 


## Local Discontinuous Galerkin Method

In this section we discuss the LDG method and first introduce some notation. 
Let $\mathcal{T}_{h} = \mathcal{T}_{h}(\Omega) \, = \, \left\{ \, \Omega_{e} 
\, \right\}_{e=1}^{N}$ be the general triangulation of a domain $\Omega \; 
\subset \; \mathbb{R}^{d}, \; d \, = \, 1, 2, 3$, into $N$ non-overlapping 
elements $\Omega_{e}$ of diameter $h_{e}$.  The maximum size of the diameters
of all elements is $h = \max( \, h_{e}\, )$.  We define $\mathcal{E}_{h}$ 
to be the set of all element faces and $\mathcal{E}_{h}^{i} $ to be the set of
all interior faces of elements which do not intersect the total boundary 
$(\partial \Omega)$. We define $\mathcal{E}_{D}$ and $\mathcal{E}_{N}$ to be
the sets of all element faces and on the Dirichlet and Neumann boundaries 
respectively. Let $\partial \Omega_{e} \in \mathcal{E}_{h}^{i}$ be a interior
boundary face element, we define the unit normal vector to be,
@f{align}
\textbf{n} \; = \; \text{unit normal vector to } \partial \Omega_{e}  
\text{ pointing from } \Omega_{e}^{-} \, \rightarrow  \, \Omega_{e}^{+}.
@f}


We take the following definition on limits of functions on element faces,
@f{align}
w^{-} (\textbf{x} ) \, \vert_{\partial \Omega_{e} } \; = \; 
\lim_{s \rightarrow 0^{-}} \, w(\textbf{x}  +  s  \textbf{n}),  && 
w^{+} (\textbf{x} ) \, \vert_{\partial \Omega_{e} } \; = \; 
\lim_{s \rightarrow 0^{+}} \, w(\textbf{x}  + s  \textbf{n}).
@f} 

We define the average and jump of a function across an element face as,
@f{align}
\{f\} \; = \; \frac{1}{2}(f^-+f^+) , 
\qquad \mbox{and} \qquad 
\left[ f \right] 
\; = \; f^+ \textbf{n}^+ + f^- \textbf{n}^-
\; = \; (f^+  - f^-) \textbf{n}^+,
@f}

and,
@f{align}
\{\textbf{f} \} \; = \; \frac{1}{2}(\textbf{f}^- + \textbf{f}^+), 
\qquad \mbox{and}\qquad  
\left[ \textbf{f} \right] 
\; = \;
\textbf{f}^+ \cdot \textbf{n}^+ + \textbf{f}^- \cdot \textbf{n}^- 
\; = \;
(\textbf{f}^+ - \textbf{f}^-) \cdot \textbf{n}^+ , 
@f}

where $f$ is a scalar function and $\textbf{f}$ is vector-valued function. 
We note that for faces that are on the boundary of the domain we have,
@f{align}
\left[ f \right] \; = \; f \,  \textbf{n} 
\qquad \mbox{and}\qquad  
\left[ \textbf{f} \right] \; = \; \textbf{f} \cdot \textbf{n}.
@f}


We denote the volume integrals and surface integrals using the $L^{2}$
inner products by $( \, \cdot \, , \, \cdot \, )_{\Omega}$ and $\langle  \, 
\cdot \, , \, \cdot \,  \rangle_{\partial \Omega}$ respectively. 

As with the mixed finite element method with the Raviart-Thomas element, 
the LDG discretization requires the 
Poisson equations be written as a first-order system.  We do this by 
introducing an auxiliary variable which we call the current flux variable 
$\textbf{q}$:
@f{align}
\nabla \cdot \textbf{q}
\; &= \; 
f(\textbf{x}) && \text{in} \ \Omega, \label{eq:Primary} \\
\textbf{q}
\; &= \;
 -\nabla u && \text{in} \ \Omega,  \label{eq:Auxiliary} \\
\textbf{q}  \cdot \textbf{n} 
\; &= \; g_{N}(\textbf{x}) && \text{on} \ \partial \Omega_{N},\\
u &= g_{D}(\textbf{x}) && \mbox{on}\ \partial \Omega_{D}.
@f}

In our numerical methods we will use approximations to scalar valued functions
that reside in the finite-dimensional broken Sobolev spaces,
@f{align}
W_{h,k}
\, &= \, 
\left\{ w \in L^{2}(\Omega) \, : \; w  \vert_{\Omega_{e}} \in 
\mathcal{Q}_{k,k}(\Omega_{e}), \quad \forall \, \Omega_{e}  \in \mathcal{T}_{h} 
\right\}, 
@f}

where $\mathcal{Q}_{k,k}(\Omega_{e})$ denotes the tensor product of 
discontinuous polynomials of order $k$ on the element $\Omega_{e}$. We use 
approximations of vector valued functions that are in,
@f{align}
\textbf{W}_{h,k} 
\, &= \, 
\left\{  \textbf{w}  \in \left(L^{2}(\Omega)\right)^{d} \, :
 \; \textbf{w}  \vert_{\Omega_{e}} \in \left( \mathcal{Q}_{k,k}(\Omega_{e}) 
 \right)^{d}, \quad \forall \, \Omega_{e} \in  \mathcal{T}_{h} \right\}
@f}

We seek approximations for densities $u_{h} \in W_{h,k}$ and gradients 
$\textbf{q}_{h}\in \textbf{W}_{h,k}$. Multiplying (6) by $w \in W_{h,k}$ and 
(7) by $\textbf{w} \in \textbf{W}_{h,k}$ and integrating the divergence terms 
by parts over an element $\Omega_{e} \in \mathcal{T}_{h}$ we obtain,
@f{align}
-
\left( \nabla w  \, , \, \textbf{q}_{h}  \right)_{\Omega_{e}}
+
\langle w \, , \,  \textbf{q}_{h}  \rangle_{\partial \Omega_{e}} 
\ &= \
\left( w , \, f \right)_{\Omega_{e}} ,  \\
\left( \textbf{w} \, , \, \textbf{q}_{h} \right)_{\Omega_{e}}
-
\left(  \nabla \cdot \textbf{w} \, , \,  u_{h} \right)_{\Omega_{e}}
+
\langle   \textbf{w}  \, ,   \, u_{h}
\rangle_{\partial \Omega_{e}} 
\ &= \
0 
@f}

Summing over all the elements leads to the weak formulation:

Find $u_{h} \in W_{h,k}$ and $\textbf{q}_{h} \in  \textbf{W}_{h,k} $ such that,

@f{align}
-
\sum_{e} \left( \nabla w,  \,  \textbf{q}_{h}  \right)_{\Omega_{e}}
 +
\langle  \left[ \,  w \, \right] \, , \,  \widehat{\textbf{q}_{h} } 
\rangle_{\mathcal{E}_{h}^{i} }
 +
\langle  \left[ \,  w \, \right] \, , \,  \widehat{\textbf{q}_{h} } 
\rangle_{\mathcal{E}_{D} \cup \mathcal{E}_{N}} \ &= \  
\sum_{e} \left( w , \, f  \right)_{\Omega_{e}}   \\
\sum_{e} \left( \textbf{w} \, , \, \textbf{q}_{h} \right)_{\Omega_{e}}
-
\sum_{e} \left(  \nabla \cdot \textbf{w} , \,  u_{h} \right)_{\Omega_{e}}
+  
\langle \, \left[ \,  \textbf{w} \, \right] \, ,   \, \widehat{u_{h}}
\rangle_{\mathcal{E}_{h}^{i}}
+ 
\langle  \left[ \,  \textbf{w} \, \right] \, , \,  \widehat{u_{h}}  
\rangle_{\mathcal{E}_{D} \cup \mathcal{E}_{N}} 
\ &= \
0 
@f}

for all $(w,\textbf{w}) \in W_{h,k} \times \textbf{W}_{h,k}$.


The terms $\widehat{\textbf{q}_{h}}$ and $\widehat{u_{h}}$ are the numerical 
fluxes. The numerical fluxes are introduced to ensure consistency, stability, 
and enforce the boundary conditions weakly, for more info see the book:
<a href="http://www.springer.com/us/book/9780387720654">
Nodal Discontinuous Galerkin Methods</a>. The flux $\widehat{u_{h}}$
is,

@f{align} 
\widehat{u_{h}} \; = \; \left\{
\begin{array}{cl}
\left\{ u_{h} \right\} \ + \ \boldsymbol \beta \cdot [ u_{h} ] \, &
\ \text{in} \  \mathcal{E}_{h}^{i} \\
u_{h} &  \ \text{in} \ \mathcal{E}_{N}\\
g_{D}(\textbf{x})  & \ \text{in} \ \mathcal{E}_{D} \\
\end{array}
\right.
@f}



The flux $\widehat{\textbf{q}_{h}}$ is,
@f{align} 
\widehat{\textbf{q}_{h}}  \; = \; \left\{
\begin{array}{cl}
\left\{ \textbf{q}_{h} \right\} \ - \  \left[ \textbf{q}_{h} \right] \, 
\boldsymbol \beta \ + \ \sigma \, \left[ \, u_{h} \, 
\right] & \ \text{in} \ \mathcal{E}_{h}^{i} \\
g_{N}(\textbf{x}) \, \textbf{n} \,  & \ \text{in} \ \mathcal{E}_{N}\\
\textbf{q}_{h} \ + \ \sigma \, \left(u_{h} - g_{D}(\textbf{x}) \right) \, 
\textbf{n} & \ \text{in} \ \mathcal{E}_{D} \\
\end{array}
\right.
@f}


The term $\boldsymbol \beta$ is a constant unit vector which does not lie 
parallel to any element face in $ \mathcal{E}_{h}^{i}$.  For 
$\boldsymbol \beta =  0$,  $\widehat{\textbf{q}_{h}}$ and $\widehat{u_{h}}$ 
are called the central or Brezzi et. al. fluxes. For 
$\boldsymbol \beta \neq  0$, $\widehat{\textbf{q}_{h}}$ and $\widehat{u_{h}}$ 
are called the LDG/alternating fluxes, see 
<a href="http://www3.nd.edu/~zxu2/acms60790S14/unified-analy-dg-elliptic-eq.pdf">
here</a> and <a href="http://www.springer.com/us/book/9780387720654">here</a>.

The term $\sigma$ is the penalty 
parameter that is defined as,
@f{align}
\sigma \; = \; \left\{ 
\begin{array}{cc}
\tilde{\sigma} \, \min \left( h^{-1}_{e_{1}}, h^{-1}_{e_{2}} \right) & 
\textbf{x} \in \langle \Omega_{e_{1}}, \Omega_{e_{2}} \rangle \\
\tilde{\sigma}  \, h^{-1}_{e} & \textbf{x} \in \partial \Omega_{e} \cap 
\in \mathcal{E}_{D}
\end{array}
\right. 
\label{eq:Penalty}
@f}


with $\tilde{\sigma}$ being a positive constant.  There are other choices of
penalty values $\sigma$, but the one above produces in approximations to solutions
that are the most accurate, see this 
<a href="http://epubs.siam.org/doi/abs/10.1137/S0036142900371003">
reference</a> for more info.


We can now substitute (16) and (17) into (14) and (15) to obtain the solution 
pair $(u_{h}, \textbf{q}_{h})$ to the LDG approximation to the Poisson 
equation given by:

 
Find $u_{h} \in W_{h,k}$ and $\textbf{q}_{h} \in  \textbf{W}_{h,k}$ such that,

@f{align}
a(\textbf{w}, \textbf{q}_{h})  \ +  \ b^{T}(\textbf{w}, u_{h}) \ &= \
G(\textbf{w}) \nonumber \\
b(w, \textbf{q}_{h}) \ + \   c(w, u_{h})  \ &= \ F(w)
\label{eq:LDG_bilinear}
@f}

for all $(w, \textbf{w}) \in W_{h,k} \times \textbf{W}_{h,k}$.  This leads to 
the linear system,

@f{align}
\left[ 
\begin{matrix}
A  & -B^{T}  \\
B & C 
\end{matrix}
\right]
\left[
\begin{matrix}
\textbf{Q}\\
\textbf{U}
\end{matrix}
\right]
\ = \
\left[
\begin{matrix}
\textbf{G}\\
\textbf{F}
\end{matrix}
\right]
@f}

Where $\textbf{U}$ and $\textbf{Q}$ are the degrees of freedom vectors for 
$u_{h}$ and $\textbf{q}_{h}$ respectively. The terms $\textbf{G}$ and 
$\textbf{F}$ are the corresponding vectors to $G(\textbf{w})$ and $F(w)$ 
respectively. The matrix in for the LDG system is non-singular for any 
$\sigma > 0$.


The bilinear forms in (19) and right hand functions are defined as,

@f{align}
b(w, \textbf{q}_{h}) \, &= \,
-
\sum_{e} \left(\nabla w, \textbf{q}_{h} \right)_{\Omega_{e}}
+
\langle \left[ w \right], 
\left\{\textbf{q}_{h} \right\} - \left[ \textbf{q}_{h} \right] \boldsymbol 
\beta \rangle_{\mathcal{E}_{h}^{i}} 
 + 
\langle w, \textbf{n} \cdot \textbf{q}_{h} \rangle_{\mathcal{E}_{D}}\\
a(\textbf{w},\textbf{q}_{h}) \, &= \,  
\sum_{e} \left(\textbf{w}, \textbf{q}_{h} \right)_{\Omega_{e}} \\
-b^{T}(w, \textbf{q}_{h}) \, &= \,
-
\sum_{e} \left(\nabla \cdot \textbf{w}, u_{h} \right)_{\Omega_{e}} 
+ 
\langle  \left[ \textbf{w} \right], \left\{u_{h} \right\} + \boldsymbol 
\beta \cdot \left[ u_{h} \right] \rangle_{\mathcal{E}_h^{i} }
+
\langle w, u_{h} \rangle_{\mathcal{E}_{N} } \\
c(w,u_{h}) \, &= \,
\langle \left[ w \right],  \sigma \left[ u_{h} \right] 
\rangle_{\mathcal{E}_{h}^{i}}
+
\langle  w, \sigma u_{h} \rangle_{\mathcal{E}_{D}}  \\
G(\textbf{w}) \ & = \ - \langle \textbf{w}, g_{D} 
\rangle_{\mathcal{E}_{D}}\\
F(w) \ & = \  \sum_{e} (w,f)_{\Omega_{e}} - \langle w, g_{N} 
\rangle_{\mathcal{E}_{N}}  + \langle w, \sigma g_{D} \rangle_{\mathcal{E}_{D}} 
@f}

As discussed in step-20, we won't be assembling the bilinear terms explicitly, 
instead we will assemble all the solid integrals and fluxes at once.  We note 
that in order to actually build the flux terms in our local flux matrices we 
will substitute in the definitions in the bilinear terms above.

## Useful References

These are some useful references on the LDG and DG methods:

- <a href="http://epubs.siam.org/doi/abs/10.1137/s0036142997316712">
The Local Discontinuous Galerkin Method for Time-Dependent 
Convection-Diffusion Systems</a>

- <a href="http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.26.7688">
Some Extensions Of The Local Discontinuous Galerkin Method For 
Convection-Diffusion Equations In Multidimensions</a>

- <a href="http://epubs.siam.org/doi/abs/10.1137/S1064827502410657">
Preconditioning Methods for Local Discontinuous Galerkin Discretizations</a>

- <a href="http://epubs.siam.org/doi/abs/10.1137/S0036142900371003">
An A Priori Error Analysis of the Local Discontinuous Galerkin
Method for Elliptic Problems</a>

- <a href="http://www3.nd.edu/~zxu2/acms60790S14/unified-analy-dg-elliptic-eq.pdf">
Unified Analysis Of Discontinuous Galerkin Methods For Elliptic Problems</a>

- <a href="http://www.springer.com/us/book/9780387720654">
Nodal Discontinuous Galerkin Methods</a>

- <a href="http://epubs.siam.org/doi/book/10.1137/1.9780898717440">
Discontinuous Galerkin Methods for Solving Elliptic and Parabolic 
Equations: Theory and Implementation</a>


# The Commented Code
