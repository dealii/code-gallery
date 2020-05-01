#include <deal.II-cdr/parameters.h>

#include <fstream>
#include <string>

namespace CDR
{
  void
  Parameters::configure_parameter_handler(ParameterHandler &parameter_handler)
  {
    parameter_handler.enter_subsection("Geometry");
    {
      parameter_handler.declare_entry("inner_radius",
                                      "1.0",
                                      Patterns::Double(0.0),
                                      "Inner radius.");
      parameter_handler.declare_entry("outer_radius",
                                      "2.0",
                                      Patterns::Double(0.0),
                                      "Outer radius.");
    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("Physical Parameters");
    {
      parameter_handler.declare_entry("diffusion_coefficient",
                                      "1.0",
                                      Patterns::Double(0.0),
                                      "Diffusion coefficient.");
      parameter_handler.declare_entry("reaction_coefficient",
                                      "1.0",
                                      Patterns::Double(0.0),
                                      "Reaction coefficient.");
      parameter_handler.declare_entry("time_dependent_forcing",
                                      "true",
                                      Patterns::Bool(),
                                      "Whether or not "
                                      "the forcing function depends on time.");
    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("Finite Element");
    {
      parameter_handler.declare_entry("initial_refinement_level",
                                      "1",
                                      Patterns::Integer(1),
                                      "Initial number of levels in the mesh.");
      parameter_handler.declare_entry("max_refinement_level",
                                      "1",
                                      Patterns::Integer(1),
                                      "Maximum number of levels in the mesh.");
      parameter_handler.declare_entry("fe_order",
                                      "1",
                                      Patterns::Integer(1),
                                      "Finite element order.");
    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("Time Step");
    {
      parameter_handler.declare_entry("start_time",
                                      "0.0",
                                      Patterns::Double(0.0),
                                      "Start time.");
      parameter_handler.declare_entry("stop_time",
                                      "1.0",
                                      Patterns::Double(1.0),
                                      "Stop time.");
      parameter_handler.declare_entry("n_time_steps",
                                      "1",
                                      Patterns::Integer(1),
                                      "Number of time steps.");
    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("Output");
    {
      parameter_handler.declare_entry("save_interval",
                                      "10",
                                      Patterns::Integer(1),
                                      "Save interval.");
      parameter_handler.declare_entry("patch_level",
                                      "2",
                                      Patterns::Integer(0),
                                      "Patch level.");
    }
    parameter_handler.leave_subsection();
  }

  void
  Parameters::read_parameter_file(const std::string &file_name)
  {
    ParameterHandler parameter_handler;
    {
      std::ifstream file(file_name);
      configure_parameter_handler(parameter_handler);
      parameter_handler.parse_input(file);
    }

    parameter_handler.enter_subsection("Geometry");
    {
      inner_radius = parameter_handler.get_double("inner_radius");
      outer_radius = parameter_handler.get_double("outer_radius");
    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("Physical Parameters");
    {
      diffusion_coefficient =
        parameter_handler.get_double("diffusion_coefficient");
      reaction_coefficient =
        parameter_handler.get_double("reaction_coefficient");
      time_dependent_forcing =
        parameter_handler.get_bool("time_dependent_forcing");
    }
    parameter_handler.leave_subsection();


    parameter_handler.enter_subsection("Finite Element");
    {
      initial_refinement_level =
        parameter_handler.get_integer("initial_refinement_level");
      max_refinement_level =
        parameter_handler.get_integer("max_refinement_level");
      fe_order = parameter_handler.get_integer("fe_order");
    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("Time Step");
    {
      start_time   = parameter_handler.get_double("start_time");
      stop_time    = parameter_handler.get_double("stop_time");
      n_time_steps = parameter_handler.get_integer("n_time_steps");
    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("Output");
    {
      save_interval = parameter_handler.get_integer("save_interval");
      patch_level   = parameter_handler.get_integer("patch_level");
    }
    parameter_handler.leave_subsection();
  }
} // namespace CDR
