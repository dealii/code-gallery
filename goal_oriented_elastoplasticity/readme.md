### 3d goal-oriented mesh adaptivity in elastoplasticity problems

The code deals with solving an elastoplasticity problem with linear isotropic hardening. At each load/displacement step, the error based on a prescribed quantity of interest (Goal-oriented error estimation) is computed by using the dual-weighted residual method.

Based on a prescribed error bound and estimated elementwise errors,
the mesh is then refined/coarsened. Afterwards, the solution is projected to the new mesh and the analysis process is repeated.

The applied methodology and the solved numerical examples can be found in the following paper:

> Ghorashi SSh, Rabczuk T.:
> "Goal-Oriented Error Estimation and Mesh
>  Adaptivity in 3d Elastoplasticity Problems".
> International Journal of Fracture. Accepted. 2016.


### Running the code

The code contains several examples consisting of the three examples presented in the aforementioned paper. To run each of them you can switch to them, e.g.
```
git checkout Thick_tube_internal_pressure

git checkout  Perforated_strip_displacement_3d

git checkout  Cantiliver_II_beam_3d
```

Then by compiling it using the following commands
```
cmake -DDEAL_II_DIR=/path/to/deal.II .
make
```
you can run it by typing
```
./elastoplastic Thick_tube_internal_pressure.prm
```
or
```
./elastoplastic Perforated_strip_displacement_3d.prm
```
or
```
./elastoplastic Cantiliver_II_beam_3d.prm
```
 
The three named input files are already in the current directory, and
have the following options:
```
set polynomial degree                [defines the polynomial order of
                                      the (primal) problem]

set number of initial refinements    [defines how many times the mesh
                                      is globally refined to construct
                                      the  initial mesh]

set refinement strategy              [can be chosen as global for
                                      global refinement or percentage
                                      in order to mesh adaptation with
                                      the assumption of refinement of
                                      30% of the elements with higher
                                      errors and coarsening of 3% of
                                      element with the least errors
                                      (these percentages can be
                                      changed inside the code)]

set error estimation strategy        [can be set kelly_error or
                                      residual_error or
                                      weighted_residual_error to apply
                                      different error estimation
                                      methods (see the paper for more
                                      information)]

set maximum relative error           [set a criterion value for
                                      perfoming the mesh adaptivity] 

set output directory                 [determine a directory to save
                                      the output results] 

set transfer solution                [by assuming true, the solution
                                      variables at each step are
                                      transferred to the next
                                      load/displacement step] 

set base mesh                        [determines the problem:
                                      Timoshenko beam /
                                      Thick_tube_internal_pressure /
                                      Perforated_strip_tension /
                                      Cantiliver_beam_3d]

set elasticity modulus

set Poissons ratio

set yield stress

set isotropic hardening parameter

set show stresses                    [by setting true or false,
                                      determines if the stress results
                                      be saved or not (to save the
                                      analysis time, we can set it as
                                      false when we do not need to
                                      illustrate them)]
```
