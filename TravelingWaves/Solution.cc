#include "Solution.h"

namespace TravelingWave
{

  using namespace dealii;
  
  SolutionStruct::SolutionStruct() {}
  SolutionStruct::SolutionStruct(const std::vector<double> &ix, const std::vector<double> &iu, 
                      const std::vector<double> &iT, const std::vector<double> &ilambda, double iwave_speed)
    : x(ix)
    , u(iu)
    , T(iT)
    , lambda(ilambda)
    , wave_speed(iwave_speed)
  {}
  SolutionStruct::SolutionStruct(const std::vector<double> &ix, const std::vector<double> &iu, 
                      const std::vector<double> &iT, const std::vector<double> &ilambda)
    : SolutionStruct(ix, iu, iT, ilambda, 0.)
  {}

  void SolutionStruct::reinit(const unsigned int number_of_elements)
  {
    wave_speed = 0.;
    x.clear();
    u.clear();
    T.clear();
    lambda.clear();

    x.resize(number_of_elements);
    u.resize(number_of_elements);
    T.resize(number_of_elements);
    lambda.resize(number_of_elements);
  }

  void SolutionStruct::save_to_file(std::string filename = "sol") const
  {
    const std::string file_for_solution = filename + ".txt";
    std::ofstream output(file_for_solution);

    output << std::scientific << std::setprecision(16);
    for (unsigned int i = 0; i < x.size(); ++i)
    {
      output << std::fixed << x[i]; 
      output << std::scientific << " " << u[i] << " " << T[i] << " " << lambda[i] << "\n";
    }
    output.close();

    std::ofstream file_for_wave_speed_output("wave_speed-" + file_for_solution);
    file_for_wave_speed_output << std::scientific << std::setprecision(16);
    file_for_wave_speed_output << wave_speed << std::endl;
    file_for_wave_speed_output.close();
  }


  Interpolant::Interpolant(const std::vector<double> &ix_points, const std::vector<double> &iy_points)
    : interpolant(ix_points, iy_points)
  {}

  double Interpolant::value(const Point<1> &p, const unsigned int component) const
  {
    double x = p[0];
    double res = interpolant.value(x);

    return res;
  }


  template <typename InterpolantType>
  SolutionVectorFunction<InterpolantType>::SolutionVectorFunction(InterpolantType iu_interpolant, InterpolantType iT_interpolant, InterpolantType ilambda_interpolant) 
    : Function<1>(3)
    , u_interpolant(iu_interpolant)
    , T_interpolant(iT_interpolant)
    , lambda_interpolant(ilambda_interpolant)
  {}

  template <typename InterpolantType>
  double SolutionVectorFunction<InterpolantType>::value(const Point<1> &p, const unsigned int component) const
  {
    double res = 0.;
    if (component == 0) 			{	res = u_interpolant.value(p); }
    else if (component == 1) 	{ res = T_interpolant.value(p);	}
    else if (component == 2)	{	res = lambda_interpolant.value(p);	}

    return res;
  }

  template class SolutionVectorFunction<Interpolant>;

} // namespace TravelingWave