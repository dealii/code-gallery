#ifndef INTEGRATE_SYSTEM
#define INTEGRATE_SYSTEM

#include <boost/numeric/odeint.hpp>

#include <iostream>
#include <fstream>
#include <string>

template <typename state_T, typename time_T>
void SaveSolutionIntoFile(const std::vector<state_T>& x_vec, const std::vector<time_T>& t_vec, std::string filename="output_ode_sol.txt") 
{
  if (!x_vec.empty() && !t_vec.empty()) 
  {
    std::ofstream output(filename);
    output << std::setprecision(16);

    size_t dim = x_vec[0].size();
    for (size_t i = 0; i < t_vec.size(); ++i) 
    {
      output << std::fixed << t_vec[i];
      for (size_t j = 0; j < dim; ++j) 
      {
        output << std::scientific << " " << x_vec[i][j];
      }
      output << "\n";
    }
    output.close();
  } 
  else 
  {
    std::cout << "Solution is not saved into file.\n";
  }
}

// type of RK integrator 
enum class Integrator_Type 
{
  dopri5,
  cash_karp54,
  fehlberg78
};

// Observer
template <typename state_type>
class Push_back_state_time
{
public:
  std::vector<state_type>& m_states;
  std::vector<double>& m_times;

  Push_back_state_time(std::vector<state_type>& states, std::vector<double>& times)
    : m_states(states), m_times(times) 
  {}

  void operator() (const state_type& x, double t) 
  {
    m_states.push_back(x);
    m_times.push_back(t);
  }
};


// Integrate system at specified points. 
template <typename ODE_obj_T, typename state_type, typename Iterable_type>
void IntegrateSystemAtTimePoints(
  std::vector<state_type>& x_vec, std::vector<double>& t_vec, const Iterable_type& iterable_time_span,
  const ODE_obj_T& ode_system_obj, 
  state_type& x, const double dt,
  Integrator_Type integrator_type=Integrator_Type::dopri5, bool save_to_file_flag=false,
  const double abs_er_tol=1.0e-15, const double rel_er_tol=1.0e-15
  )
{
  using namespace boost::numeric::odeint;

  if (integrator_type == Integrator_Type::dopri5) 
  {
    typedef runge_kutta_dopri5< state_type > error_stepper_type;
    integrate_times( make_controlled< error_stepper_type >(abs_er_tol, rel_er_tol),
              ode_system_obj, x, iterable_time_span.begin(), iterable_time_span.end(), dt, Push_back_state_time< state_type >(x_vec, t_vec) );
  } 
  else if (integrator_type == Integrator_Type::cash_karp54) 
  {
    typedef runge_kutta_cash_karp54< state_type > error_stepper_type;
    integrate_times( make_controlled< error_stepper_type >(abs_er_tol, rel_er_tol),
              ode_system_obj, x, iterable_time_span.begin(), iterable_time_span.end(), dt, Push_back_state_time< state_type >(x_vec, t_vec) );
  }
  else 
  { // Integrator_Type::fehlberg78
    typedef runge_kutta_fehlberg78< state_type > error_stepper_type;
    integrate_times( make_controlled< error_stepper_type >(abs_er_tol, rel_er_tol),
              ode_system_obj, x, iterable_time_span.begin(), iterable_time_span.end(), dt, Push_back_state_time< state_type >(x_vec, t_vec) );
  }

  if (save_to_file_flag) 
  {
    SaveSolutionIntoFile(x_vec, t_vec);
  } 

}

#endif