Readme file for nonlinear-poro-viscoelasticity
==============================================

Overview
--------

We implemented a nonlinear poro-viscoelastic formulation with the aim of characterising brain tissue response to cyclic loading. Our model captures both experimentally observed fluid flow and conditioning aspects of brain tissue behavior in addition to its well-established nonlinear, preconditioning, hysteretic, and tension-compression asymmetric characteristics.

The tissue is modelled as a biphasic material consisting in an immiscible aggregate of a nonlinear viscoelastic solid skeleton saturated with pore fluid. The governing equations are linearised using automatic differentiation and solved monolithically for the unknown solid displacements and fluid pore pressure values. 

A detailed description of the formulation, it's verification, and the results obtained can be found in:
* E. Comellas, S. Budday, J.-P. Pelteret, G. A. Holzapfel and P. Steinamnn (2020), Modeling the porous and viscous responses of human brain tissue behavior, Computer Methods in Applied Mechanics and Engineering (accepted for publication).

In this paper we show that nonlinear poroelasticity alone can reproduce consolidation experiments, yet it is insufficient to capture stress conditioning due to cyclic loading. We also discuss how the poroelastic response exhibits preconditioning and hysteresis in the fluid flow space, with porous and viscous effects being highly interrelated.


Quick facts on the code
-----------------------

* Biphasic material following the Theory of Porous Media
* Nonlinear finite viscoelasticity built on Ogden hyperelasticity
* Darcy-like fluid flow
* Spatial discretisation with continuous Q2P1 Lagrangian finite elements
* Temporal discretisation with a stable implicit one-step backward differentiation method
* Newton-Raphson scheme to solve the nonlinear system of governing equations
* Forward mode automatic differentiation with the number of derivative components chosen at run-time (Sacado library within Trilinos package) to linearise the equations
* Trilinos direct solver for the (non-symmetric) linear system of equations using a monolithic scheme
* Parallelization through Threading Building Blocks and across nodes via MPI (Trilinos package)
* Based on step-44 and code gallery contributions 'Quasi-Static Finite-Strain Compressible Elasticity' and 'Quasi-Static Finite-Strain Quasi-incompressible Visco-elasticity'
* Only works in 3D


Running the code
---------------- 
### Requirements
* Version 9.1 or greater of deal.II
* C++11, MPI, Metis and Trilinos with Sacado must be enabled

### Compiling and running
Similar to the example programs, run
```
cmake -DDEAL_II_DIR=/path/to/deal.II .
```
in this directory to configure the problem.  
You can switch between debug and release mode by calling either
```
make debug
```
or
```
make release
```
The problem may then be run in serial mode with
```
make run
```
and in parallel (in this case, on `6` processors) with
```
mpirun -np 6 ./nonlinear-poro-viscoelasticity
```

Alternatively, to keep it a bit more tidy, create a folder, e.g. run and copy the input file in there. Then type:
```
mpirun -np 6 -wdir run ../nonlinear-poro-viscoelasticity
```
All the input files used to produce the results shown in the paper are provided in in the input-files folder. Simply replace the parameters.prm file in the main directory. For the verification examples, run the python script 'run-multi-calc.py' instead:
```
python run-multi-calc.py
```
The 'run-multi-calc.py' and 'runPoro.sh' files provided must both be in the main directory. This will automatically generate the required input files and run them in sequence.


Reference for this work
-----------------------

If you use this program as a basis for your own work, please consider citing the paper referenced in the introduction. The initial version of this work was contributed to the deal.II project by E. Comellas and J.-P. Pelteret.



Recommended literature
----------------------

* W. Ehlers and G. Eipper (1999), Finite Elastic Deformations in Liquid-Saturated and Empty Porous Solids, Transport in Porous Media 34(1/3):179-191. DOI: [10.1023/A:1006565509095](https://doi.org/10.1023/A:1006565509095);
* S. Reese and  S. Govindjee (2001), A theory of finite viscoelasticity and numerical aspects, International Journal of Solids and Structures 35(26-27):3455-3482. DOI: [10.1016/S0020-7683(97)00217-5](https://doi.org/10.1016/S0020-7683(97)00217-5);
* G. Franceschini, D. Bigoni, P. Regitnig and G. A. Holzapfel (2006), Brain tissue deforms similarly to  filled elastomers and follows consolidation theory, Journal of the Mechanics and Physics of Solids 54(12):2592-2620. DOI: [10.1016/j.jmps.2006.05.004](https://doi.org/10.1016/j.jmps.2006.05.004);
* S. Budday, G. Sommer, J. Haybaeck, P. Steinmann, G. A. Holzapfel and E. Kuhl (2017), Rheological characterization of human brain tissue, Acta Biomaterialia 60:315-329. DOI: [10.1016/j.actbio.2017.06.024](https://doi.org/10.1016/j.actbio.2017.06.024);
* G.A. Holzapfel (2001), Nonlinear Solid Mechanics. A Continuum Approach for Engineering, John Wiley & Sons. ISBN: [978-0-471-82319-3](http://eu.wiley.com/WileyCDA/WileyTitle/productCd-0471823198.html);


