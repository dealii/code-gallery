Readme file for CeresFE
=======================

Motivation for project
----------------------

This code was made to simulate the evolution of global-scale topography on planetary bodies.  Specifically, it is designed to compute the rates of topography relaxation on the dwarf planet Ceres.  The NASA Dawn mission, in orbit around Ceres since March, 2015, has produced a high resolution shape model of its surface.  As on other planets including the Earth, topography on Ceres is subject to decay over time due to processes such as viscous flow and brittle failure.  Because the efficiency of these processes is dependent on the material properties of the body at depth, simulating the decay of topography and comparing it to the observed shape model permits insights into Ceres' internal structure. 

Some previous applications of this basic idea- using topography to constrain internal structure- may be found in the following references:  

 1. Takeuchi, H. and Hasegawa, Y. (1965) Viscosity distribution within the Earth. Geophys. J. R. astr. Soc. 9, 503-508.
 2. Anderson, D. L. and O'Connell, R. (1967) Viscosity of the Earth. Geophys. J. R. astr. Soc. 14, 287-295.
 3. Solomon, S. C., Comer, R. P., Head, J. W. (1982) The Evolution of impact basins: Viscous relaxation of topographic relief.
 4. Zuber, M. T. et al. (2000) Internal structure and early thermal evolution of Mars from Mars Global Surveyor topography and gravity. Science 287, 1788-1793.
 5. Fu, R. R. et al. (2014) Efficient early global relaxation of asteroid Vesta. Icarus 240, 133-145.

The code included here is a development of a simpler code for the asteroid Vesta, published as reference 5 above.  Because both versions of the code were written specifically to model long wavelength topography on these small bodies, the code is rather specific.  We hope certain components of it may be useful to the reader even if the problem of topographic relaxation on asteroid belt bodies is not on everyone's radar. 



Quick facts about the code
--------------------------

* Viscoelastoplastic
* Asymmetric
* Lagrangian
* Uses analytical self-gravity
* One sentence purpose: Simulates evolution of topography due to self-gravity on axisymmetry planetary body.

More detailed properties of the code in CeresFE
-----------------------------------------------

### Viscoelastoplasticity

* The code is viscoelastoplastic: it solves the Stokes equations modified to include elasticity and iteratively uses the stress solution to account for displacement due to brittle failure
* The implementation of viscoelasticity follows mainly section 2.2.3 of Keller, T. et al. (2013) Numerical modelling of magma dynamics coupled to tectonic deformation of lithosphere and crust. Geophys. J. Int. 195, 1406-1442.
* At the end of each FE calculation, the principle stresses (in 3D) are computed in all cells.  Each cell is evaluated according to either Byerlee's Rule or a damaged rock brittle failure criterion to determine if failure occurred.  See Byerlee, J. (1978) Friction of rocks, Pageoph. 116, 615-626. and Schultz, R. A. (1993) Brittle strength of basalitc rock masses with applications to Venus. J. Geophys. Res. 98, 10,883-10,895. 
* If a cell failed, its viscosity is lowered by a computed amount to simulate plastic yielding.  The viscosity fields is smoothed and the FE model run again.  This is repeated until the number of failed cells falls below a prescribed number.  The final viscosity field (i.e., the effective viscosity) is then used to compute velocities and advance the mesh.  

### Domain and boundary conditions

* The domain of the model is 2D, but the Stokes equations are cast in axisymmetric form.
* The domain consists of approximately a quarter ellipse, with two straight edges corresponding to the rotation axis and equator of the body.  No normal flux boundary conditions are applied to these edges.
* The remaining curved edge that corresponds to the surface of the body is assigned a zero pressure boundary condition
* With respect to self-gravity, an ellipse is fitted to the outer surface and any internal density surfaces at each time step and a gravity field is computed analytically following Pohanka, V. (2011) Gravitational field of the homogeneous rotational ellipsoidal body: a simple derivation and applications. Contrib. Geophys. Geodesy 41, 117-157. 


Description of files in repo
----------------------------

* src/ceres.cc                     Main code
* support_code/config_in.h         Reads config file and initializes system parameters
* support_code/ellipsoid_fit.h     Finds best-fit ellipse for surface and internal density boundaries.  Also uses deal.II
* support_code/ellipsoid_grav.h    Analytically computes self gravity of layered ellipsoids structure
* support_code/local_math.h        Defines some constants for convenience
* meshes/sample_CeresFE_mesh.inp   Sample input mesh
* config/sample_CeresFE_config.cfg ample configurations file with simulation parameters

Other dependencies
------------------

Two more code packages are necessary to run CeresFE:

1. config++: https://sourceforge.net/projects/config/
2. Armadillo: http://arma.sourceforge.net

To run the code
---------------

After running cmake and compiling, run the executable with one argument, 
which is the config file:

$ceres ${SOURCE_DIR}/config/sample_CeresFE_config.cfg

