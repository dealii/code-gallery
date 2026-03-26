# PlasticityLab: ALE finite-strain thermoplasticity (axisymmetric)
## Overview

This code-gallery entry provides a deal.II-based implementation of a large-deformation thermomechanical solver for finite-strain thermoplasticity, developed to support numerical simulation of severe-deformation metal forming processes and related problems involving strong thermomechanical coupling and viscoplastic flow.

The solver implements a finite-strain associative coupled thermoplasticity model.

## Key idea: ALE via incremental reference motion

The solver incorporates an Arbitrary Lagrangian–Eulerian (ALE) formulation for coupled finite-strain thermoplasticity in which the motion of the reference configuration is represented incrementally through a reference velocity field. This avoids the need to explicitly track the deformation from the initial material configuration, either as a deformation field or through storing and updating the full deformation gradient history. 

The method targets regimes where large accumulated strains can cause excessive mesh distortion in purely Lagrangian finite element simulations. The ALE formulation reduces sensitivity to mesh distortion and enables stable simulation without requiring prohibitively small time steps.

## Physics and models
### Finite-strain thermoplasticity

The constitutive model is a finite-strain, associative, coupled thermoplasticity formulation based on a multiplicative decomposition of the deformation gradient and a J2 (von Mises) flow theory. The implementation supports viscoplastic behavior via rate-dependent flow stress models (including Johnson-Cook). 

### Thermomechanical coupling

The thermal problem is coupled to the mechanical response through plastic dissipation and heat conduction. 

### Discretization and solution strategy (high level)

Mixed finite element formulation to mitigate volumetric locking (Jacobian/pressure treated as additional unknowns).

Newton-Raphson nonlinear solve with consistent tangent moduli (targeting second-order convergence behavior).

Mechanical-thermal operator splitting per time step (mechanical, then thermal, then mechanical sub-step). 

### Axisymmetric reduction and benchmark problems

Although the formulation is derived for general 3D settings, the code uses an axisymmetric approximation for benchmark problems and representative manufacturing-process simulations.

The entry validates and illustrates the approach using benchmark problems including:

thermally triggered necking of a circular bar (thermoplasticity benchmark), 

Taylor anvil impact of a circular bar (dynamic high-rate deformation benchmark),

## To run
```
# in a build directory:
$ cmake -DDEAL_II_DIR=<path-to-deal-ii> <path-to-entry>
$ make release
$ make -j 8 &&  mpirun -n 18 ./PlasticityLab
```

## Notes on configuration

Geometry / triangulation: configured in PlasticityLabProgDrivers.cpp in run().

Material model selection and parameters: configured in main.cpp. 

Time step settings: configured in PlasticityLabProg.h. 

## References
```
@article{HamedMcBrideReddy2023_ALE_Thermoplasticity_FrictionWelding,
  author  = {Hamed, M. M. O. and McBride, A. T. and Reddy, B. D.},
  title   = {An {ALE} approach for large-deformation thermoplasticity with application to friction welding},
  journal = {Computational Mechanics},
  volume  = {72},
  pages   = {803--826},
  year    = {2023},
  doi     = {10.1007/s00466-023-02303-0}
}
```