#ifndef LIMIT_SOLUTION
#define LIMIT_SOLUTION

#include "Parameters.h"
#include <iostream>
#include <vector>

namespace TravelingWave
{
  typedef std::vector< double > state_type;

  class LimitSolution
  {
  public:
    LimitSolution(const Parameters &parameters, const double ilambda_0, const double iu_0, const double iT_0, const double root_sign = 1.);

    void operator() (const state_type &x , state_type &dxdt , const double /* t */);
    void calculate_u_T_omega();
    void set_wave_speed(double iwave_speed);

    std::vector<double> t_vec;
    std::vector<state_type> omega_vec;
    std::vector<state_type> lambda_vec;
    std::vector<state_type> u_vec;
    std::vector<state_type> T_vec;

  private:
    double omega_func(const double lambda, const double T) const;
    double u_func(const double lambda) const;
    double T_func(const double lambda) const;

    void calculate_constants_A_B();

    const Parameters &params;
    const Problem    &problem;
    double wave_speed;

    const double lambda_0, u_0, T_0;  // Initial values.
    double A, B;                      // Integration constants.

    const double root_sign;           // Plus or minus one.
  };


} // namespace TravelingWave

#endif