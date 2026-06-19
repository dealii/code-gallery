## Overview
Thermo-electro-elastic pump: implementation of a problem with thermo-electro-mechanical coupling.

The electro-mechanical and thermal problems are solved in a staggered manner.
Two implementations of the electro-mechanical problem are implemented:
- the standard two-field formulation using the electric scalar potential and a
  displacement field (valid for compressible materials), and
- a four-field formulation, extending the above with an additional dilatation
  and pressure field (valid for both compressible and near-incompressible 
  materials).

Concepts related to this implementation are presented in:

@Article{Mehnert2017a,
  author    = {M. Mehnert and J.-P. Pelteret and P. Steinmann},
  title     = {Numerical modelling of nonlinear thermo-electro-elasticity},
  journal   = {Mathematics and Mechanics of Solids},
  year      = {2017},
  volume    = {22},
  number    = {11},
  pages     = {2196--2213},
  month     = nov,
  doi       = {10.1177/1081286517729867},
  publisher = {{SAGE} Publications},
}

@Article{Pelteret2016a,
  author    = {Pelteret, J.-P. V. and Davydov, D. and McBride, A. and Vu, D. K. and Steinmann, P.},
  title     = {Computational electro-elasticity and magneto-elasticity for quasi-incompressible media immersed in free space},
  journal   = {International Journal for Numerical Methods in Engineering},
  year      = {2016},
  volume    = {108},
  number    = {11},
  pages     = {1307--1342},
  month     = dec,
  doi       = {10.1002/nme.5254},
}

### Related papers

@Article{Hamkar2012a,
  author  = {Hamkar, A-W and Hartmann, Stefan},
  title   = {Theoretical and numerical aspects in weak-compressible finite strain thermo-elasticity},
  journal = {Journal of Theoretical and Applied Mechanics},
  year    = {2012},
  volume  = {50},
  pages   = {3--22},
  file    = {:Articles/Hamkar2012a.pdf:PDF},
}

@article{mehnert2018numerical,
  title={Numerical modeling of thermo-electro-viscoelasticity with field-dependent material parameters},
  author={Mehnert, Markus and Hossain, Mokarram and Steinmann, Paul},
  journal={International Journal of Non-Linear Mechanics},
  volume={106},
  pages={13--24},
  year={2018},
  publisher={Elsevier}
}

@article{mehnert2019experimental,
  title={Experimental and numerical investigations of the electro-viscoelastic behavior of VHB 4905TM},
  author={Mehnert, Markus and Hossain, Mokarram and Steinmann, Paul},
  journal={European Journal of Mechanics-A/Solids},
  pages={103797},
  year={2019},
  publisher={Elsevier}
}
