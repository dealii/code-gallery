#include "Parameters.h"

namespace TravelingWave
{
  using namespace dealii;
  
  Problem::Problem() 
    : ParameterAcceptor("Problem")
  {
    add_parameter("delta", delta = 0.01); 
    add_parameter("epsilon", epsilon = 0.1);            
    add_parameter("Prandtl number", Pr = 0.75);        
    add_parameter("Lewis number", Le = 1.0);      
    add_parameter("Constant of reaction rate", k = 1.0);
    add_parameter("Activation energy", theta = 1.65);
    add_parameter("Heat release", q = 1.7);
    add_parameter("Ignition Temperature", T_ign = 1.0);
    add_parameter("Type of the wave (deflagration / detonation)", wave_type = 1);   // 0 for "deflagration"; 1 for "detonation".

    add_parameter("Type of boundary condition for the temperature at the right boundary", T_r_bc_type = 1);   // 0 for "Neumann" (deflagration); 1 for "Dirichlet" (detonation).

    add_parameter("T_left", T_left = 5.3);    // Dirichlet boundary condition.
    add_parameter("T_right", T_right = 0.9);  // For detonation waves the value serves as a Dirichlet boundary condition. For deflagration waves it serves for construction of the piecewise constant initial guess.
    add_parameter("u_left", u_left = -0.2);   // For detonation waves the value is ignored. For deflagration waves it serves for construction of the piecewise constant initial guess.
    add_parameter("u_right", u_right = 0.);   // Dirichlet boundary condition.

    add_parameter("Initial guess for the wave speed", wave_speed_init = 1.2);   // For detonation waves the value is ignored. For deflagration waves it serves as an initial guess for the wave speed.
  }

  FiniteElements::FiniteElements()
    : ParameterAcceptor("Finite elements")
  {
    add_parameter("Polynomial degree", poly_degree = 1);
    add_parameter("Number of quadrature points", quadrature_points_number = 3);
  }

  Mesh::Mesh()
    : ParameterAcceptor("Mesh")
  {
    add_parameter("Interval left boundary", interval_left = -50.0);  
    add_parameter("Interval right boundary", interval_right = 20.0);
    add_parameter<unsigned int>("Refinements number", refinements_number = 10);
    add_parameter("Adaptive mesh refinement", adaptive = 1);    // 1 for adaptive; 0 for global.
  }

  Solver::Solver()
    : ParameterAcceptor("Solver")
  {
    add_parameter("Tolerance", tol = 1e-10);
  }

} // namespace TravelingWave
