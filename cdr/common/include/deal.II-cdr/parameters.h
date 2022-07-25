#ifndef dealii__cdr_parameters_h
#define dealii__cdr_parameters_h

#include <deal.II/base/parameter_handler.h>

#include <string>

// I prefer to use the ParameterHandler class in a slightly different way than
// usual: The class Parameters creates, uses, and then destroys a
// ParameterHandler inside the <code>read_parameter_file</code> method instead
// of keeping it around. This is nice because now all of the run time
// parameters are contained in a simple class and it can be copied or passed
// around very easily.
namespace CDR
{
  using namespace dealii;

  class Parameters
  {
  public:
    double inner_radius;
    double outer_radius;

    double diffusion_coefficient;
    double reaction_coefficient;
    bool   time_dependent_forcing;

    unsigned int initial_refinement_level;
    unsigned int max_refinement_level;
    unsigned int fe_order;

    double       start_time;
    double       stop_time;
    unsigned int n_time_steps;

    unsigned int save_interval;
    unsigned int patch_level;

    void
    read_parameter_file(const std::string &file_name);

  private:
    void
    configure_parameter_handler(ParameterHandler &parameter_handler);
  };
} // namespace CDR
#endif
