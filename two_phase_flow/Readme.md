Two Phase Flow 
-----------------------------------

### General description of the problem ###

We consider the problem of two-phase incompressible flow. 
We start with an initial state of two phases (fluids) that 
define density and viscosity fields. 
Using these fields we solve the incompressible 
Navier-Stokes equations to obtain a velocity field. 

We use the initial state to define a representation of the 
interface via a Level Set function $\phi\in[-1, 1]$. 
The zero level set $\{\phi=0\}$ defines the interface of 
the phases. Positive values of the level set function 
represent water while negative values represent air. 

Using the velocity field from the Navier-Stokes equations 
we transport the level set function. To do this we assume 
the velocity is divergence free and write the transport 
equation in conservation form. 

Using the advected level set function we reconstruct 
density and viscosity fields. We repeat the process until 
the final desired time. 

The Navier-Stokes equations are solved using a projection 
scheme based on [1]. To solve the level set we use continuous 
Galerkin Finite Elements with high-order stabilization based on the entropy 
residual of the solution [2] and artificial compression inspired by [3] and [4]. 

-----------------------------------
### General description of the code ###
##### Driver code: MultiPhase #####
The driver code of the simulation is the run function within MultiPhase.cc. 
The general idea is to define here everything that has to do with the problem, 
set all the (physical and numerical) parameters and perform the time loop. 
The run function does the following: 
* Set some physical parameters like final time, density and viscosity 
coefficients, etc. and numerical parameters like cfl, numerical constants, 
algorithms to be used, etc.
* Creates the geometry for the specified problem. Currently we have the following problems:
    * Breaking Dam problem in 2D. 
    * Filling a tank in 2D. 
    * Small wave perturbation in 2D. 
    * Falling drop in 2D. 
* Creates an object of the class **NavierStokesSolver** and an object of the class **LevelSetSolver**.  
* Set the initial condition for each of the solvers. 
* Performs the time loop. Within the time loop we do the following: 
    * Pass the current level set function to the Navier Stokes Solver. 
    * Ask the Navier Stokes Solver to perform one time step. 
    * Get the velocity field from the Navier Stokes Solver. 
    * Pass the velocity field to the Level Set Solver. 
    * Ask the Level Set Solver to perform one time step. 
    * Get the level set function from the Level Set Solver. 
    * Repeat until the final time.
* Output the solution at the requested times. 

##### Navier Stokes Solver #####
The NavierStokesSolver class is responsible for solving the Navier Stokes equation for 
just one time step. It requires density and viscosity information. This information can be 
passed by either a function or by passing a vector containing the DOFs of the level set function. For this reason the class contains the following two constructors:
* First constructor. Here we have to pass density and viscosity constants for the two phases. In addition, we have to pass a vector of DOFs defining the level set function. This constructor is meant to be used during the two-phase flow simulations. 
* Second constructor. Here we have to pass functions to define the viscosity and density fields. This is meant to test the convergence properties of the method (and to validate the implementation). 

##### Level Set Solver #####
The LevelSetSolver.cc code is responsible for solving the Level Set for just one time step. It requires information about the velocity field and provides the transported level set function. The velocity field can be interpolated (outside of this class) from a given function to test the method (and to validate the implementation). Alternatively, the velocity can be provided from the solution of the Navier-Stokes equations (for the two phase flow simulations). 

##### Testing the Navier Stokes Solver #####
The TestNavierStokes.cc code is used to test the convergence (in time) of the Navier-Stokes solver. To run it uncomment the line **SET(TARGET "TestNavierStokes")** within CMakeLists.txt (and make sure to comment **SET(TARGET "TestLevelSet")** and **SET(TARGET "MultiPhase")**. Then cmake and compile. The convergence can be done in 2 or 3 dimensions. Different exact solutions (and force terms) are used in each case. The dimension can 
be set in the line **TestNavierStokes<2> test_navier_stokes(degree_LS, degree_U)** within the main function. 

##### Testing the Level Set Solver #####
The TestLevelSet.cc code is used to test the level set solver. To run it uncomment the corresponding line within CMakeLists.txt. Then cmake and compile. There are currently just two problems implemented: diagonal advection and circular rotation. If the velocity is independent of time set the flag **VARIABLE_VELOCITY** to zero to avoid interpolating the velocity field at every time step. 

##### Utility files #####
The files utilities.cc, utilities_test_LS.cc and utilities_test_NS.cc contain functions required in MultiPhase.cc, TestLevelSet.cc and TestNavierStokes.cc respectively. 
    The script clean.sh ereases all files created by cmake, compile and run any example. 

-----------------------------------
### References ###
[1] J.-L. Guermond and A. Salgado. A splitting method for incompressible flows with
variable density based on a pressure Poisson equation. Journal of Computational Physics, 228(8):2834–2846, 2009.

[2] J.-L. Guermond, R. Pasquetti, and B. Popov. Entropy viscosity method for nonlinear conservation laws. Journal of Computational Physics, 230(11):4248–
4267, 2011.

[3] A. Harten. The artificial compression method for computation of shocks and contact discontinuities. I. Single conservation laws. Communications on Pure
and Applied Mathematics, 30(5):611–638, 1977.

[4] A. Harten. The artificial compression method for computation of shocks and contact discontinuities. III. Self-adjusting hybrid schemes. Mathematics of
Computation, 32:363–389, 1978.