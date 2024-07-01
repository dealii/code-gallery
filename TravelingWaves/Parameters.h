#ifndef PARAMETERS
#define PARAMETERS

#include <deal.II/base/parameter_acceptor.h>

namespace TravelingWave
{
  using namespace dealii;

  struct Problem : ParameterAcceptor
  {
    Problem();

    double delta, epsilon;
    double Pr, Le;
    double k, theta, q;
    double T_ign;
    int wave_type;
    int T_r_bc_type;
    double T_left, T_right;
    double u_left, u_right;
    
    double wave_speed_init;
  };

  struct FiniteElements : ParameterAcceptor
  {
	  FiniteElements();

    unsigned int poly_degree;
    unsigned int quadrature_points_number;
  };

  struct Mesh : ParameterAcceptor
  {
    Mesh();
    
    double interval_left;
    double interval_right;
    unsigned int refinements_number;
    int adaptive;
  };

  struct Solver : ParameterAcceptor
  {
    Solver();

    double       tol;
  };

  struct Parameters
  {
    Problem           problem;
    FiniteElements 		fe;
    Mesh			 	      mesh;
    Solver   		      solver;
  };

} // namespace TravelingWave

#endif
