/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2022 by Giuseppe Orlando
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 */

// We start by including all the necessary deal.II header files
//
#include <deal.II/base/parameter_handler.h>

// @sect3{Run time parameters}
//
// Since our method has several parameters that can be fine-tuned we put them
// into an external file, so that they can be determined at run-time.
//
namespace RunTimeParameters {
  using namespace dealii;

  class Data_Storage {
  public:
    Data_Storage();

    void read_data(const std::string& filename);

    double initial_time;
    double final_time;

    double Reynolds;
    double dt;

    unsigned int n_refines;            /*--- Number of refinements ---*/
    unsigned int max_loc_refinements; /*--- Number of maximum local refinements allowed ---*/
    unsigned int min_loc_refinements; /*--- Number of minimum local refinements allowed
                                            once reached that level ---*/

    /*--- Parameters related to the linear solver ---*/
    unsigned int max_iterations;
    double       eps;

    bool         verbose;
    unsigned int output_interval;

    std::string dir; /*--- Auxiliary string variable for output storage ---*/

    unsigned int refinement_iterations; /*--- Auxiliary variable about how many steps perform remeshing ---*/

  protected:
    ParameterHandler prm;
  };

  // In the constructor of this class we declare all the parameters in suitable (but arbitrary) subsections.
  //
  Data_Storage::Data_Storage(): initial_time(0.0),
                                final_time(1.0),
                                Reynolds(1.0),
                                dt(5e-4),
                                n_refines(0),
                                max_loc_refinements(0),
                                min_loc_refinements(0),
                                max_iterations(1000),
                                eps(1e-12),
                                verbose(true),
                                output_interval(15),
                                refinement_iterations(0) {
    prm.enter_subsection("Physical data");
    {
      prm.declare_entry("initial_time",
                        "0.0",
                        Patterns::Double(0.0),
                        " The initial time of the simulation. ");
      prm.declare_entry("final_time",
                        "1.0",
                        Patterns::Double(0.0),
                        " The final time of the simulation. ");
      prm.declare_entry("Reynolds",
                        "1.0",
                        Patterns::Double(0.0),
                        " The Reynolds number. ");
    }
    prm.leave_subsection();

    prm.enter_subsection("Time step data");
    {
      prm.declare_entry("dt",
                        "5e-4",
                        Patterns::Double(0.0),
                        " The time step size. ");
    }
    prm.leave_subsection();

    prm.enter_subsection("Space discretization");
    {
      prm.declare_entry("n_of_refines",
                        "100",
                        Patterns::Integer(0, 1500),
                        " The number of cells we want on each direction of the mesh. ");
      prm.declare_entry("max_loc_refinements",
                        "4",
                         Patterns::Integer(0, 10),
                         " The number of maximum local refinements. ");
      prm.declare_entry("min_loc_refinements",
                        "2",
                         Patterns::Integer(0, 10),
                         " The number of minimum local refinements. ");
    }
    prm.leave_subsection();

    prm.enter_subsection("Data solve");
    {
      prm.declare_entry("max_iterations",
                        "1000",
                        Patterns::Integer(1, 30000),
                        " The maximal number of iterations linear solvers must make. ");
      prm.declare_entry("eps",
                        "1e-12",
                        Patterns::Double(0.0),
                        " The stopping criterion. ");
    }
    prm.leave_subsection();

    prm.declare_entry("refinement_iterations",
                      "0",
                      Patterns::Integer(0),
                      " This number indicates how often we need to "
                      "refine the mesh");

    prm.declare_entry("saving directory", "SimTest");

    prm.declare_entry("verbose",
                      "true",
                      Patterns::Bool(),
                      " This indicates whether the output of the solution "
                      "process should be verbose. ");

    prm.declare_entry("output_interval",
                      "1",
                      Patterns::Integer(1),
                      " This indicates between how many time steps we print "
                      "the solution. ");
  }

  // We need now a routine to read all declared parameters in the constructor
  //
  void Data_Storage::read_data(const std::string& filename) {
    std::ifstream file(filename);
    AssertThrow(file, ExcFileNotOpen(filename));

    prm.parse_input(file);

    prm.enter_subsection("Physical data");
    {
      initial_time = prm.get_double("initial_time");
      final_time   = prm.get_double("final_time");
      Reynolds     = prm.get_double("Reynolds");
    }
    prm.leave_subsection();

    prm.enter_subsection("Time step data");
    {
      dt = prm.get_double("dt");
    }
    prm.leave_subsection();

    prm.enter_subsection("Space discretization");
    {
      n_refines           = prm.get_integer("n_of_refines");
      max_loc_refinements = prm.get_integer("max_loc_refinements");
      min_loc_refinements = prm.get_integer("min_loc_refinements");
    }
    prm.leave_subsection();

    prm.enter_subsection("Data solve");
    {
      max_iterations = prm.get_integer("max_iterations");
      eps            = prm.get_double("eps");
    }
    prm.leave_subsection();

    dir = prm.get("saving directory");

    refinement_iterations = prm.get_integer("refinement_iterations");

    verbose = prm.get_bool("verbose");

    output_interval = prm.get_integer("output_interval");
  }

} // namespace RunTimeParameters
