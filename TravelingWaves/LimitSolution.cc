#include "LimitSolution.h"

namespace TravelingWave
{

  LimitSolution::LimitSolution(const Parameters &parameters, const double ilambda_0, const double iu_0, const double iT_0, const double iroot_sign)
    : params(parameters)
    , problem(params.problem)
    , wave_speed(problem.wave_speed_init)
    , lambda_0(ilambda_0)
    , u_0(iu_0)
    , T_0(iT_0)
    , root_sign(iroot_sign)
  {
    calculate_constants_A_B();
  }

  double LimitSolution::omega_func(const double lambda, const double T) const
  {
    return problem.k * (1. - lambda) * std::exp(-problem.theta / T);
  }

  void LimitSolution::operator() (const state_type &x , state_type &dxdt , const double /* t */)
  {
    dxdt[0] = -1. / wave_speed * omega_func(x[0], T_func(x[0]));
  }

  double LimitSolution::u_func(const double lambda) const
  {
    double coef = 2 * (wave_speed - 1) / problem.epsilon - 1;
    return (coef + root_sign * std::sqrt(coef * coef - 4 * (problem.q * lambda + B - 2 * A / problem.epsilon))) / 2;
  }

  double LimitSolution::T_func(const double lambda) const
  {
    return u_func(lambda) + problem.q * lambda + B;
  }

  void LimitSolution::calculate_constants_A_B()
  {
    B = T_0 - u_0;
    A = u_0 * (1 - wave_speed) + problem.epsilon * (u_0 * u_0  + T_0) / 2;
  }

  void LimitSolution::set_wave_speed(double iwave_speed)
  {
    wave_speed = iwave_speed;
    calculate_constants_A_B();
  }

  void LimitSolution::calculate_u_T_omega()
  {
    if (!t_vec.empty() && !lambda_vec.empty())
    {
      u_vec.resize(lambda_vec.size());
      T_vec.resize(lambda_vec.size());
      omega_vec.resize(lambda_vec.size());
      for (unsigned int i = 0; i < lambda_vec.size(); ++i)
      {
        u_vec[i].resize(1);
        T_vec[i].resize(1);
        omega_vec[i].resize(1);

        u_vec[i][0] = u_func(lambda_vec[i][0]);
        T_vec[i][0] = T_func(lambda_vec[i][0]);
        omega_vec[i][0] = omega_func(lambda_vec[i][0], T_vec[i][0]);
      }
    }
    else
    {
      std::cout << "t_vec or lambda_vec vector is empty!" << std::endl;
    }
  }

} // namespace TravelingWave