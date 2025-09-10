/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2006 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Tao Jin
 *         University of Ottawa, Ottawa, Ontario, Canada
 *         April. 2024
 *
 * How to cite:
 *         Jin T, Li Z, Chen K. A novel phase-field monolithic scheme for brittle crack
 *         propagation based on the limited-memory BFGS method with adaptive mesh refinement.
 *         Int J Numer Methods Eng. 2024;e7572. doi: 10.1002/nme.7572
 */

/* A monolithic scheme based on the L-BFGS method to solve the phase-field crack problem
 * 1. The phase-field formulation itself is based on "A phase field model for rate-independent
 *    crack propagation - Robust algorithmic implementation based on operator splits"
 *    by Christian Miehe , Martina Hofacker, Fabian Welschinger
 * 2. This code implements a monolithic approach. The phase-field irreversibility
 *    is enforced through the history field Phi_0^+ and the viscosity parameter.
 * 3. Using TBB for stiffness assembly and Gauss point calculation.
 * 4. Using adaptive mesh refinement.
 */

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/base/timer.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>


#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/solver_selector.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/error_estimator.h>

#include <deal.II/physics/elasticity/standard_tensors.h>

#include <deal.II/base/quadrature_point_data.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/base/work_stream.h>

#include <deal.II/numerics/solution_transfer.h>

#include <fstream>
#include <iostream>
#include <deal.II/base/logstream.h>

#include "SpectrumDecomposition.h"
#include "Utilities.h"

namespace PhaseField
{
  using namespace dealii;

  // body force
  template <int dim>
  void right_hand_side(const std::vector<Point<dim>> &points,
		       std::vector<Tensor<1, dim>> &  values,
		       const double fx,
		       const double fy,
		       const double fz)
  {
    Assert(values.size() == points.size(),
           ExcDimensionMismatch(values.size(), points.size()));
    Assert(dim >= 2, ExcNotImplemented());

    for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
      {
	if (dim == 2)
	  {
	    values[point_n][0] = fx;
	    values[point_n][1] = fy;
	  }
	else
	  {
	    values[point_n][0] = fx;
	    values[point_n][1] = fy;
	    values[point_n][2] = fz;
	  }
      }
  }

  double degradation_function(const double d)
  {
    return (1.0 - d) * (1.0 - d);
  }

  double degradation_function_derivative(const double d)
  {
    return 2.0 * (d - 1.0);
  }

  double degradation_function_2nd_order_derivative(const double d)
  {
    (void) d;
    return 2.0;
  }

  namespace Parameters
  {
    struct Scenario
    {
      unsigned int m_scenario;
      std::string m_logfile_name;
      bool m_output_iteration_history;
      std::string m_type_nonlinear_solver;
      std::string m_type_line_search;
      std::string m_type_linear_solver;
      std::string m_refinement_strategy;
      unsigned int m_LBFGS_m;
      unsigned int m_global_refine_times;
      unsigned int m_local_prerefine_times;
      unsigned int m_max_adaptive_refine_times;
      int m_max_allowed_refinement_level;
      double m_phasefield_refine_threshold;
      double m_allowed_max_h_l_ratio;
      unsigned int m_total_material_regions;
      std::string m_material_file_name;
      int m_reaction_force_face_id;

      static void declare_parameters(ParameterHandler &prm);
      void parse_parameters(ParameterHandler &prm);
    };

    void Scenario::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Scenario");
      {
        prm.declare_entry("Scenario number",
                          "1",
                          Patterns::Integer(0),
                          "Geometry, loading and boundary conditions scenario");

        prm.declare_entry("Log file name",
			  "Output.log",
                          Patterns::FileName(Patterns::FileName::input),
			  "Name of the file for log");

        prm.declare_entry("Output iteration history",
			  "yes",
                          Patterns::Selection("yes|no"),
			  "Shall we write iteration history to the log file?");

        prm.declare_entry("Nonlinear solver type",
                          "Newton",
                          Patterns::Selection("Newton|BFGS|LBFGS"),
                          "Type of solver used to solve the nonlinear system");

        prm.declare_entry("Line search type",
                          "GradientBased",
                          Patterns::Selection("GradientBased|StrongWolfe"),
                          "Type of line search method, the gradient-based method "
                          "should be preferred since it is generally faster");

        prm.declare_entry("Linear solver type",
                          "Direct",
                          Patterns::Selection("Direct|CG"),
                          "Type of solver used to solve the linear system B0");

        prm.declare_entry("Mesh refinement strategy",
                          "adaptive-refine",
                          Patterns::Selection("pre-refine|adaptive-refine"),
                          "Mesh refinement strategy: pre-refine or adaptive-refine");

        prm.declare_entry("LBFGS m",
                          "40",
                          Patterns::Integer(0),
                          "Number of vectors used for LBFGS");

        prm.declare_entry("Global refinement times",
                          "0",
                          Patterns::Integer(0),
                          "Global refinement times (across the entire domain)");

        prm.declare_entry("Local prerefinement times",
                          "0",
                          Patterns::Integer(0),
                          "Local pre-refinement times (assume crack path is known a priori), "
                          "only refine along the crack path.");

        prm.declare_entry("Max adaptive refinement times",
                          "100",
                          Patterns::Integer(0),
                          "Maximum number of adaptive refinement times allowed in each step");

        prm.declare_entry("Max allowed refinement level",
                          "100",
                          Patterns::Integer(0),
                          "Maximum allowed cell refinement level");

        prm.declare_entry("Phasefield refine threshold",
			  "0.8",
			  Patterns::Double(),
			  "Phasefield-based refinement threshold value");

        prm.declare_entry("Allowed max hl ratio",
			  "0.25",
			  Patterns::Double(),
			  "Allowed maximum ratio between mesh size h and length scale l");

        prm.declare_entry("Material regions",
                          "1",
                          Patterns::Integer(0),
                          "Number of material regions");

        prm.declare_entry("Material data file",
                          "1",
                          Patterns::FileName(Patterns::FileName::input),
                          "Material data file");

        prm.declare_entry("Reaction force face ID",
                          "1",
                          Patterns::Integer(),
                          "Face id where reaction forces should be calculated "
                          "(negative integer means not to calculate reaction force)");
      }
      prm.leave_subsection();
    }

    void Scenario::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Scenario");
      {
        m_scenario = prm.get_integer("Scenario number");
        m_logfile_name = prm.get("Log file name");
        m_output_iteration_history = prm.get_bool("Output iteration history");
        m_type_nonlinear_solver = prm.get("Nonlinear solver type");
        m_type_line_search = prm.get("Line search type");
        m_type_linear_solver = prm.get("Linear solver type");
        m_refinement_strategy = prm.get("Mesh refinement strategy");
        m_LBFGS_m = prm.get_integer("LBFGS m");
        m_global_refine_times = prm.get_integer("Global refinement times");
        m_local_prerefine_times = prm.get_integer("Local prerefinement times");
        m_max_adaptive_refine_times = prm.get_integer("Max adaptive refinement times");
        m_max_allowed_refinement_level = prm.get_integer("Max allowed refinement level");
        m_phasefield_refine_threshold = prm.get_double("Phasefield refine threshold");
        m_allowed_max_h_l_ratio = prm.get_double("Allowed max hl ratio");
        m_total_material_regions = prm.get_integer("Material regions");
        m_material_file_name = prm.get("Material data file");
        m_reaction_force_face_id = prm.get_integer("Reaction force face ID");
      }
      prm.leave_subsection();
    }

    struct FESystem
    {
      unsigned int m_poly_degree;
      unsigned int m_quad_order;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };


    void FESystem::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        prm.declare_entry("Polynomial degree",
                          "1",
                          Patterns::Integer(0),
                          "Phase field polynomial order");

        prm.declare_entry("Quadrature order",
                          "2",
                          Patterns::Integer(0),
                          "Gauss quadrature order");
      }
      prm.leave_subsection();
    }

    void FESystem::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        m_poly_degree = prm.get_integer("Polynomial degree");
        m_quad_order  = prm.get_integer("Quadrature order");
      }
      prm.leave_subsection();
    }

    // body force (N/m^3)
    struct BodyForce
    {
      double m_x_component;
      double m_y_component;
      double m_z_component;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };

    void BodyForce::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Body force");
      {
        prm.declare_entry("Body force x component",
			  "0.0",
			  Patterns::Double(),
			  "Body force x-component (N/m^3)");

        prm.declare_entry("Body force y component",
			  "0.0",
			  Patterns::Double(),
			  "Body force y-component (N/m^3)");

        prm.declare_entry("Body force z component",
			  "0.0",
			  Patterns::Double(),
			  "Body force z-component (N/m^3)");
      }
      prm.leave_subsection();
    }

    void BodyForce::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Body force");
      {
        m_x_component = prm.get_double("Body force x component");
        m_y_component = prm.get_double("Body force y component");
        m_z_component = prm.get_double("Body force z component");
      }
      prm.leave_subsection();
    }

    struct NonlinearSolver
    {
      unsigned int m_max_iterations_NR;
      unsigned int m_max_iterations_BFGS;
      bool m_relative_residual;

      double       m_tol_u_residual;
      double       m_tol_d_residual;
      double       m_tol_u_incr;
      double       m_tol_d_incr;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };

    void NonlinearSolver::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        prm.declare_entry("Max iterations Newton-Raphson",
                          "10",
                          Patterns::Integer(0),
                          "Number of Newton-Raphson iterations allowed");

        prm.declare_entry("Max iterations BFGS",
                          "20",
                          Patterns::Integer(0),
                          "Number of BFGS iterations allowed");

        prm.declare_entry("Relative residual",
			  "yes",
                          Patterns::Selection("yes|no"),
			  "Shall we use relative residual for convergence?");

        prm.declare_entry("Tolerance displacement residual",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Displacement residual tolerance");

        prm.declare_entry("Tolerance phasefield residual",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Phasefield residual tolerance");

        prm.declare_entry("Tolerance displacement increment",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Displacement increment tolerance");

        prm.declare_entry("Tolerance phasefield increment",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Phasefield increment tolerance");
      }
      prm.leave_subsection();
    }

    void NonlinearSolver::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        m_max_iterations_NR = prm.get_integer("Max iterations Newton-Raphson");
        m_max_iterations_BFGS = prm.get_integer("Max iterations BFGS");
        m_relative_residual = prm.get_bool("Relative residual");

        m_tol_u_residual           = prm.get_double("Tolerance displacement residual");
        m_tol_d_residual           = prm.get_double("Tolerance phasefield residual");
        m_tol_u_incr               = prm.get_double("Tolerance displacement increment");
        m_tol_d_incr               = prm.get_double("Tolerance phasefield increment");
      }
      prm.leave_subsection();
    }

    struct TimeInfo
    {
      double m_end_time;
      std::string m_time_file_name;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };

    void TimeInfo::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        prm.declare_entry("End time", "1", Patterns::Double(), "End time");

        prm.declare_entry("Time data file",
                          "1",
                          Patterns::FileName(Patterns::FileName::input),
                          "Time data file");
      }
      prm.leave_subsection();
    }

    void TimeInfo::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        m_end_time = prm.get_double("End time");
        m_time_file_name = prm.get("Time data file");
      }
      prm.leave_subsection();
    }

    struct AllParameters : public Scenario,
	                   public FESystem,
	                   public BodyForce,
			   public NonlinearSolver,
			   public TimeInfo
    {
      AllParameters(const std::string &input_file);

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };

    AllParameters::AllParameters(const std::string &input_file)
    {
      ParameterHandler prm;
      declare_parameters(prm);
      prm.parse_input(input_file);
      parse_parameters(prm);
    }

    void AllParameters::declare_parameters(ParameterHandler &prm)
    {
      Scenario::declare_parameters(prm);
      FESystem::declare_parameters(prm);
      BodyForce::declare_parameters(prm);
      NonlinearSolver::declare_parameters(prm);
      TimeInfo::declare_parameters(prm);
    }

    void AllParameters::parse_parameters(ParameterHandler &prm)
    {
      Scenario::parse_parameters(prm);
      FESystem::parse_parameters(prm);
      BodyForce::parse_parameters(prm);
      NonlinearSolver::parse_parameters(prm);
      TimeInfo::parse_parameters(prm);
    }
  } // namespace Parameters

  class Time
  {
  public:
    Time(const double time_end)
      : m_timestep(0)
      , m_time_current(0.0)
      , m_time_end(time_end)
      , m_delta_t(0.0)
      , m_magnitude(1.0)
    {}

    virtual ~Time() = default;

    double current() const
    {
      return m_time_current;
    }
    double end() const
    {
      return m_time_end;
    }
    double get_delta_t() const
    {
      return m_delta_t;
    }
    double get_magnitude() const
    {
      return m_magnitude;
    }
    unsigned int get_timestep() const
    {
      return m_timestep;
    }
    void increment(std::vector<std::array<double, 4>> time_table)
    {
      double t_1, t_delta, t_magnitude;
      for (auto & time_group : time_table)
        {
	  t_1 = time_group[1];
	  t_delta = time_group[2];
	  t_magnitude = time_group[3];

	  if (m_time_current < t_1 - 1.0e-6*t_delta)
	    {
	      m_delta_t = t_delta;
	      m_magnitude = t_magnitude;
	      break;
	    }
        }

      m_time_current += m_delta_t;
      ++m_timestep;
    }

  private:
    unsigned int m_timestep;
    double       m_time_current;
    const double m_time_end;
    double m_delta_t;
    double m_magnitude;
  };

  template <int dim>
  class LinearIsotropicElasticityAdditiveSplit
  {
  public:
    LinearIsotropicElasticityAdditiveSplit(const double lame_lambda,
			                   const double lame_mu,
				           const double residual_k,
					   const double length_scale,
					   const double viscosity,
					   const double gc)
      : m_lame_lambda(lame_lambda)
      , m_lame_mu(lame_mu)
      , m_residual_k(residual_k)
      , m_length_scale(length_scale)
      , m_eta(viscosity)
      , m_gc(gc)
      , m_phase_field_value(0.0)
      , m_grad_phasefield(Tensor<1, dim>())
      , m_strain(SymmetricTensor<2, dim>())
      , m_stress(SymmetricTensor<2, dim>())
      , m_stress_positive(SymmetricTensor<2, dim>())
      , m_mechanical_C(SymmetricTensor<4, dim>())
      , m_strain_energy_positive(0.0)
      , m_strain_energy_negative(0.0)
      , m_strain_energy_total(0.0)
      , m_crack_energy_dissipation(0.0)
    {
      Assert(  ( lame_lambda / (2*(lame_lambda + lame_mu)) <= 0.5)
	     & ( lame_lambda / (2*(lame_lambda + lame_mu)) >=-1.0),
	     ExcInternalError() );
    }

    const SymmetricTensor<4, dim> & get_mechanical_C() const
    {
      return m_mechanical_C;
    }

    const SymmetricTensor<2, dim> & get_cauchy_stress() const
    {
      return m_stress;
    }

    const SymmetricTensor<2, dim> & get_cauchy_stress_positive() const
    {
      return m_stress_positive;
    }

    double get_positive_strain_energy() const
    {
      return m_strain_energy_positive;
    }

    double get_negative_strain_energy() const
    {
      return m_strain_energy_negative;
    }

    double get_total_strain_energy() const
    {
      return m_strain_energy_total;
    }

    double get_crack_energy_dissipation() const
    {
      return m_crack_energy_dissipation;
    }

    double get_phase_field_value() const
    {
      return m_phase_field_value;
    }

    const Tensor<1, dim> get_phase_field_gradient() const
    {
      return m_grad_phasefield;
    }

    void update_material_data(const SymmetricTensor<2, dim> & strain,
			      const double phase_field_value,
			      const Tensor<1, dim> & grad_phasefield,
			      const double phase_field_value_previous_step,
			      const double delta_time)
    {
      m_strain = strain;
      m_phase_field_value = phase_field_value;
      m_grad_phasefield = grad_phasefield;
      Vector<double>              eigenvalues(dim);
      std::vector<Tensor<1, dim>> eigenvectors(dim);
      usr_spectrum_decomposition::spectrum_decomposition<dim>(m_strain,
    							      eigenvalues,
    							      eigenvectors);

      SymmetricTensor<2, dim> strain_positive, strain_negative;
      strain_positive = usr_spectrum_decomposition::positive_tensor(eigenvalues, eigenvectors);
      strain_negative = usr_spectrum_decomposition::negative_tensor(eigenvalues, eigenvectors);

      SymmetricTensor<4, dim> projector_positive, projector_negative;
      usr_spectrum_decomposition::positive_negative_projectors(eigenvalues,
    							       eigenvectors,
							       projector_positive,
							       projector_negative);

      SymmetricTensor<2, dim> stress_positive, stress_negative;
      const double degradation = degradation_function(m_phase_field_value) + m_residual_k;
      const double I_1 = trace(m_strain);
      stress_positive = m_lame_lambda * usr_spectrum_decomposition::positive_ramp_function(I_1)
                                      * Physics::Elasticity::StandardTensors<dim>::I
                      + 2 * m_lame_mu * strain_positive;
      stress_negative = m_lame_lambda * usr_spectrum_decomposition::negative_ramp_function(I_1)
                                      * Physics::Elasticity::StandardTensors<dim>::I
      		      + 2 * m_lame_mu * strain_negative;

      m_stress = degradation * stress_positive + stress_negative;
      m_stress_positive = stress_positive;

      SymmetricTensor<4, dim> C_positive, C_negative;
      C_positive = m_lame_lambda * usr_spectrum_decomposition::heaviside_function(I_1)
                                 * Physics::Elasticity::StandardTensors<dim>::IxI
		 + 2 * m_lame_mu * projector_positive;
      C_negative = m_lame_lambda * usr_spectrum_decomposition::heaviside_function(-I_1)
                                 * Physics::Elasticity::StandardTensors<dim>::IxI
      		 + 2 * m_lame_mu * projector_negative;
      m_mechanical_C = degradation * C_positive + C_negative;

      m_strain_energy_positive = 0.5 * m_lame_lambda * usr_spectrum_decomposition::positive_ramp_function(I_1)
                                                     * usr_spectrum_decomposition::positive_ramp_function(I_1)
                               + m_lame_mu * strain_positive * strain_positive;

      m_strain_energy_negative = 0.5 * m_lame_lambda * usr_spectrum_decomposition::negative_ramp_function(I_1)
                                                     * usr_spectrum_decomposition::negative_ramp_function(I_1)
                               + m_lame_mu * strain_negative * strain_negative;

      m_strain_energy_total = degradation * m_strain_energy_positive + m_strain_energy_negative;

      m_crack_energy_dissipation = m_gc * (  0.5 / m_length_scale * m_phase_field_value * m_phase_field_value
	                                   + 0.5 * m_length_scale * m_grad_phasefield * m_grad_phasefield)
	                                   // the term due to viscosity regularization
	                                   + (m_phase_field_value - phase_field_value_previous_step)
					   * (m_phase_field_value - phase_field_value_previous_step)
				           * 0.5 * m_eta / delta_time;
      //(void)delta_time;
      //(void)phase_field_value_previous_step;
    }

  private:
    const double m_lame_lambda;
    const double m_lame_mu;
    const double m_residual_k;
    const double m_length_scale;
    const double m_eta;
    const double m_gc;
    double m_phase_field_value;
    Tensor<1, dim> m_grad_phasefield;
    SymmetricTensor<2, dim> m_strain;
    SymmetricTensor<2, dim> m_stress;
    SymmetricTensor<2, dim> m_stress_positive;
    SymmetricTensor<4, dim> m_mechanical_C;
    double m_strain_energy_positive;
    double m_strain_energy_negative;
    double m_strain_energy_total;
    double m_crack_energy_dissipation;
  };


  template <int dim>
  class PointHistory
  {
  public:
    PointHistory()
      : m_length_scale(0.0)
      , m_gc(0.0)
      , m_viscosity(0.0)
      , m_history_max_positive_strain_energy(0.0)
    {}

    virtual ~PointHistory() = default;

    void setup_lqp(const double lame_lambda,
		   const double lame_mu,
		   const double length_scale,
		   const double gc,
		   const double viscosity,
		   const double residual_k)
    {
      m_material =
              std::make_shared<LinearIsotropicElasticityAdditiveSplit<dim>>(lame_lambda,
        	                                                            lame_mu,
								            residual_k,
									    length_scale,
									    viscosity,
									    gc);
      m_history_max_positive_strain_energy = 0.0;
      m_length_scale = length_scale;
      m_gc = gc;
      m_viscosity = viscosity;

      update_field_values(SymmetricTensor<2, dim>(), 0.0, Tensor<1, dim>(), 0.0, 1.0);
    }

    void update_field_values(const SymmetricTensor<2, dim> & strain,
		             const double phase_field_value,
			     const Tensor<1, dim> & grad_phasefield,
			     const double phase_field_value_previous_step,
			     const double delta_time)
    {
      m_material->update_material_data(strain, phase_field_value, grad_phasefield,
				       phase_field_value_previous_step, delta_time);
    }

    void update_history_variable()
    {
      double current_positive_strain_energy = m_material->get_positive_strain_energy();
      m_history_max_positive_strain_energy = std::fmax(m_history_max_positive_strain_energy,
					               current_positive_strain_energy);
    }

    double get_current_positive_strain_energy() const
    {
      return m_material->get_positive_strain_energy();
    }

    const SymmetricTensor<4, dim> & get_mechanical_C() const
    {
      return m_material->get_mechanical_C();
    }

    const SymmetricTensor<2, dim> & get_cauchy_stress() const
    {
      return m_material->get_cauchy_stress();
    }

    const SymmetricTensor<2, dim> & get_cauchy_stress_positive() const
    {
      return m_material->get_cauchy_stress_positive();
    }

    double get_total_strain_energy() const
    {
      return m_material->get_total_strain_energy();
    }

    double get_crack_energy_dissipation() const
    {
      return m_material->get_crack_energy_dissipation();
    }

    double get_phase_field_value() const
    {
      return m_material->get_phase_field_value();
    }

    const Tensor<1, dim> get_phase_field_gradient() const
    {
      return m_material->get_phase_field_gradient();
    }

    double get_history_max_positive_strain_energy() const
    {
      return m_history_max_positive_strain_energy;
    }

    double get_length_scale() const
    {
      return m_length_scale;
    }

    double get_critical_energy_release_rate() const
    {
      return m_gc;
    }

    double get_viscosity() const
    {
      return m_viscosity;
    }
  private:
    std::shared_ptr<LinearIsotropicElasticityAdditiveSplit<dim>> m_material;
    double m_length_scale;
    double m_gc;
    double m_viscosity;
    double m_history_max_positive_strain_energy;
  };

  template <int dim>
  class PhaseFieldMonolithicSolve
  {
  public:
    PhaseFieldMonolithicSolve(const std::string &input_file);

    virtual ~PhaseFieldMonolithicSolve() = default;
    void run();

  private:
    struct PerTaskData_ASM;
    struct ScratchData_ASM;

    struct PerTaskData_ASM_RHS_BFGS;
    struct ScratchData_ASM_RHS_BFGS;

    struct PerTaskData_UQPH;
    struct ScratchData_UQPH;

    Parameters::AllParameters m_parameters;
    Triangulation<dim> m_triangulation;

    CellDataStorage<typename Triangulation<dim>::cell_iterator,
                    PointHistory<dim>>
      m_quadrature_point_history;

    Time                m_time;
    std::ofstream m_logfile;
    mutable TimerOutput m_timer;

    DoFHandler<dim>                  m_dof_handler;
    FESystem<dim>                    m_fe;
    const unsigned int               m_dofs_per_cell;
    const FEValuesExtractors::Vector m_u_fe;
    const FEValuesExtractors::Scalar m_d_fe;

    static const unsigned int m_n_blocks          = 2;
    static const unsigned int m_n_components      = dim + 1;
    static const unsigned int m_first_u_component = 0;
    static const unsigned int m_d_component       = dim;

    enum
    {
      m_u_dof = 0,
      m_d_dof = 1
    };

    std::vector<types::global_dof_index> m_dofs_per_block;

    const QGauss<dim>     m_qf_cell;
    const QGauss<dim - 1> m_qf_face;
    const unsigned int    m_n_q_points;

    double m_vol_reference;

    AffineConstraints<double> m_constraints;
    BlockSparsityPattern      m_sparsity_pattern;
    BlockSparseMatrix<double> m_tangent_matrix;
    BlockVector<double>       m_system_rhs;
    BlockVector<double>       m_solution;
    SparseDirectUMFPACK       m_A_direct;


    std::map<unsigned int, std::vector<double>> m_material_data;

    std::vector<std::pair<double, std::vector<double>>> m_history_reaction_force;
    std::vector<std::pair<double, std::array<double, 3>>> m_history_energy;


    struct Errors
    {
      Errors()
        : m_norm(1.0)
        , m_u(1.0)
        , m_d(1.0)
      {}

      void reset()
      {
        m_norm = 1.0;
        m_u    = 1.0;
        m_d    = 1.0;
      }

      void normalize(const Errors &rhs)
      {
        if (rhs.m_norm != 0.0)
          m_norm /= rhs.m_norm;
        if (rhs.m_u != 0.0)
          m_u /= rhs.m_u;
        if (rhs.m_d != 0.0)
          m_d /= rhs.m_d;
      }

      double m_norm, m_u, m_d;
    };

    Errors m_error_residual, m_error_residual_0, m_error_residual_norm, m_error_update,
      m_error_update_0, m_error_update_norm;

    void get_error_residual(Errors &error_residual);
    void get_error_update(const BlockVector<double> &newton_update,
                          Errors & error_update);

    void make_grid();
    void make_grid_case_1();
    void make_grid_case_2();
    void make_grid_case_3();
    void make_grid_case_4();
    void make_grid_case_5();
    void make_grid_case_6();
    void make_grid_case_7();
    void make_grid_case_8();
    void make_grid_case_9();
    void make_grid_case_11();

    void setup_system();

    void determine_component_extractors();

    void make_constraints(const unsigned int it_nr);

    void assemble_system_newton(const BlockVector<double> & solution_old);

    void assemble_system_B0(const BlockVector<double> & solution_old);

    void assemble_system_newton_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_ASM &                                     scratch,
      PerTaskData_ASM &                                     data) const;

    void assemble_system_B0_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_ASM &                                     scratch,
      PerTaskData_ASM &                                     data) const;

    void assemble_system_rhs_BFGS_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_ASM_RHS_BFGS &                           scratch,
      PerTaskData_ASM_RHS_BFGS &                           data) const;

    void assemble_system_rhs_BFGS(const BlockVector<double> & solution_old,
				  BlockVector<double> & system_rhs);

    void assemble_system_rhs_BFGS_parallel(const BlockVector<double> & solution_old,
    				           BlockVector<double> & system_rhs);

    bool solve_nonlinear_timestep_newton(BlockVector<double> &solution_delta);

    void solve_nonlinear_timestep_BFGS(BlockVector<double> &solution_delta);

    void solve_nonlinear_timestep_LBFGS(BlockVector<double> &solution_delta,
					BlockVector<double> & LBFGS_update_refine);

    double line_search_stepsize_strong_wolfe(const double phi_0,
				             const double phi_0_prime,
				             const BlockVector<double> & BFGS_p_vector,
				             const BlockVector<double> & solution_delta);

    double line_search_stepsize_gradient_based(const BlockVector<double> & BFGS_p_vector,
					       const BlockVector<double> & solution_delta);

    double line_search_zoom_strong_wolfe(double phi_low, double phi_low_prime, double alpha_low,
					 double phi_high, double phi_high_prime, double alpha_high,
					 double phi_0, double phi_0_prime, const BlockVector<double> & BFGS_p_vector,
					 double c1, double c2, unsigned int max_iter,
					 const BlockVector<double> & solution_delta);

    double line_search_interpolation_cubic(const double alpha_0, const double phi_0, const double phi_0_prime,
					   const double alpha_1, const double phi_1, const double phi_1_prime);

    std::pair<double, double> calculate_phi_and_phi_prime(const double alpha,
							  const BlockVector<double> & BFGS_p_vector,
							  const BlockVector<double> & solution_delta);

    std::vector<double> solve_linear_system(BlockVector<double> &newton_update);

    void LBFGS_B0(BlockVector<double> & LBFGS_r_vector,
		  BlockVector<double> & LBFGS_q_vector);

    void update_history_field_step();

    void output_results() const;

    void setup_qph();

    void update_qph_incremental(const BlockVector<double> &solution_delta,
				const BlockVector<double> &solution_old,
				const bool is_print);

    void update_qph_incremental_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_UQPH &                                    scratch,
      PerTaskData_UQPH &                                    data);

    void copy_local_to_global_UQPH(const PerTaskData_UQPH & /*data*/)
    {}

    BlockVector<double>
    get_total_solution(const BlockVector<double> &solution_delta) const;

    // Should not make this function const
    void read_material_data(const std::string &data_file,
			    const unsigned int total_material_regions);

    void read_time_data(const std::string &data_file,
    		        std::vector<std::array<double, 4>> & time_table);

    void print_conv_header_newton();

    void print_conv_header_BFGS();

    void print_conv_header_LBFGS();

    void print_parameter_information();

    void calculate_reaction_force(unsigned int face_ID);

    void write_history_data();

    double calculate_energy_functional() const;

    std::pair<double, double> calculate_total_strain_energy_and_crack_energy_dissipation() const;

    bool local_refine_and_solution_transfer(BlockVector<double> & solution_delta,
					    BlockVector<double> & LBFGS_update_refine);
  }; // class PhaseFieldSplitSolve


  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::get_error_residual(Errors &error_residual)
  {
    BlockVector<double> error_res(m_dofs_per_block);

    for (unsigned int i = 0; i < m_dof_handler.n_dofs(); ++i)
      if (!m_constraints.is_constrained(i))
        error_res(i) = m_system_rhs(i);

    error_residual.m_norm = error_res.l2_norm();
    error_residual.m_u    = error_res.block(m_u_dof).l2_norm();
    error_residual.m_d    = error_res.block(m_d_dof).l2_norm();
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::get_error_update(const BlockVector<double> &newton_update,
                                                        Errors & error_update)
  {
    BlockVector<double> error_ud(m_dofs_per_block);
    for (unsigned int i = 0; i < m_dof_handler.n_dofs(); ++i)
      if (!m_constraints.is_constrained(i))
        error_ud(i) = newton_update(i);

    error_update.m_norm = error_ud.l2_norm();
    error_update.m_u    = error_ud.block(m_u_dof).l2_norm();
    error_update.m_d    = error_ud.block(m_d_dof).l2_norm();
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::read_material_data(const std::string &data_file,
				                     const unsigned int total_material_regions)
  {
    std::ifstream myfile (data_file);

    double lame_lambda, lame_mu, length_scale, gc, viscosity, residual_k;
    int material_region;
    double poisson_ratio;
    if (myfile.is_open())
      {
        m_logfile << "Reading material data file ..." << std::endl;

        while ( myfile >> material_region
                       >> lame_lambda
		       >> lame_mu
		       >> length_scale
		       >> gc
		       >> viscosity
		       >> residual_k)
          {
            m_material_data[material_region] = {lame_lambda,
        	                                lame_mu,
						length_scale,
						gc,
						viscosity,
                                                residual_k};
            poisson_ratio = lame_lambda / (2*(lame_lambda + lame_mu));
            Assert( (poisson_ratio <= 0.5)&(poisson_ratio >=-1.0) , ExcInternalError());

            m_logfile << "\tRegion " << material_region << " : " << std::endl;
            m_logfile << "\t\tLame lambda = " << lame_lambda << std::endl;
            m_logfile << "\t\tLame mu = "  << lame_mu << std::endl;
            m_logfile << "\t\tPoisson ratio = "  << poisson_ratio << std::endl;
            m_logfile << "\t\tPhase field length scale (l) = " << length_scale << std::endl;
            m_logfile << "\t\tCritical energy release rate (gc) = "  << gc << std::endl;
            m_logfile << "\t\tViscosity for regularization (eta) = "  << viscosity << std::endl;
            m_logfile << "\t\tResidual_k (k) = "  << residual_k << std::endl;
          }

        if (m_material_data.size() != total_material_regions)
          {
            m_logfile << "Material data file has " << m_material_data.size() << " rows. However, "
        	      << "the mesh has " << total_material_regions << " material regions."
		      << std::endl;
            Assert(m_material_data.size() == total_material_regions,
                       ExcDimensionMismatch(m_material_data.size(), total_material_regions));
          }
        myfile.close();
      }
    else
      {
	m_logfile << "Material data file : " << data_file << " not exist!" << std::endl;
	Assert(false, ExcMessage("Failed to read material data file"));
      }
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::read_time_data(const std::string &data_file,
				                      std::vector<std::array<double, 4>> & time_table)
  {
    std::ifstream myfile (data_file);

    double t_0, t_1, delta_t, t_magnitude;

    if (myfile.is_open())
      {
	m_logfile << "Reading time data file ..." << std::endl;

	while ( myfile >> t_0
		       >> t_1
		       >> delta_t
		       >> t_magnitude)
	  {
	    Assert( t_0 < t_1,
		    ExcMessage("For each time pair, "
			       "the start time should be smaller than the end time"));
	    time_table.push_back({{t_0, t_1, delta_t, t_magnitude}});
	  }

	Assert(std::fabs(t_1 - m_parameters.m_end_time) < 1.0e-9,
	       ExcMessage("End time in time table is inconsistent with input data in parameters.prm"));

	Assert(time_table.size() > 0,
	       ExcMessage("Time data file is empty."));
	myfile.close();
      }
    else
      {
        m_logfile << "Time data file : " << data_file << " not exist!" << std::endl;
        Assert(false, ExcMessage("Failed to read time data file"));
      }

    for (auto & time_group : time_table)
      {
	m_logfile << "\t\t"
	          << time_group[0] << ",\t"
	          << time_group[1] << ",\t"
		  << time_group[2] << ",\t"
		  << time_group[3] << std::endl;
      }
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::setup_qph()
  {
    m_logfile << "\t\tSetting up quadrature point data ("
	      << m_n_q_points
	      << " points per cell)" << std::endl;

    m_quadrature_point_history.clear();
    for (auto const & cell : m_triangulation.active_cell_iterators())
      {
	m_quadrature_point_history.initialize(cell, m_n_q_points);
      }

    unsigned int material_id;
    double lame_lambda = 0.0;
    double lame_mu = 0.0;
    double length_scale = 0.0;
    double gc = 0.0;
    double viscosity = 0.0;
    double residual_k = 0.0;

    for (const auto &cell : m_triangulation.active_cell_iterators())
      {
        material_id = cell->material_id();
        if (m_material_data.find(material_id) != m_material_data.end())
          {
            lame_lambda                = m_material_data[material_id][0];
            lame_mu                    = m_material_data[material_id][1];
            length_scale               = m_material_data[material_id][2];
            gc                         = m_material_data[material_id][3];
            viscosity                  = m_material_data[material_id][4];
            residual_k                 = m_material_data[material_id][5];
	  }
        else
          {
            m_logfile << "Could not find material data for material id: " << material_id << std::endl;
            AssertThrow(false, ExcMessage("Could not find material data for material id."));
          }

        const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
          m_quadrature_point_history.get_data(cell);
        Assert(lqph.size() == m_n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < m_n_q_points; ++q_point)
          lqph[q_point]->setup_lqp(lame_lambda, lame_mu, length_scale,
				   gc, viscosity, residual_k);
      }
  }

  template <int dim>
  BlockVector<double> PhaseFieldMonolithicSolve<dim>::get_total_solution(
    const BlockVector<double> &solution_delta) const
  {
    BlockVector<double> solution_total(m_solution);
    solution_total += solution_delta;
    return solution_total;
  }

  template <int dim>
  void
  PhaseFieldMonolithicSolve<dim>::update_qph_incremental(const BlockVector<double> &solution_delta,
							 const BlockVector<double> &solution_old,
							 const bool is_print)
  {
    m_timer.enter_subsection("Update QPH data");
    if (is_print && m_parameters.m_output_iteration_history)
      m_logfile << " UQPH " << std::flush;

    const BlockVector<double> solution_total(get_total_solution(solution_delta));

    const UpdateFlags uf_UQPH(update_values | update_gradients);
    PerTaskData_UQPH  per_task_data_UQPH;
    ScratchData_UQPH  scratch_data_UQPH(m_fe,
					m_qf_cell,
					uf_UQPH,
					solution_total,
					solution_old,
					m_time.get_delta_t());

    auto worker = [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
	                 ScratchData_UQPH & scratch,
	                 PerTaskData_UQPH & data)
      {
        this->update_qph_incremental_one_cell(cell, scratch, data);
      };

    auto copier = [this](const PerTaskData_UQPH &data)
      {
        this->copy_local_to_global_UQPH(data);
      };

    WorkStream::run(
	m_dof_handler.begin_active(),
	m_dof_handler.end(),
	worker,
	copier,
	scratch_data_UQPH,
	per_task_data_UQPH);

    m_timer.leave_subsection();
  }

  template <int dim>
  struct PhaseFieldMonolithicSolve<dim>::PerTaskData_UQPH
  {
    void reset()
    {}
  };

  template <int dim>
  struct PhaseFieldMonolithicSolve<dim>::ScratchData_UQPH
  {
    const BlockVector<double> & m_solution_UQPH;

    std::vector<SymmetricTensor<2, dim>> m_solution_symm_grads_u_cell;
    std::vector<double>         m_solution_values_phasefield_cell;
    std::vector<Tensor<1, dim>> m_solution_grad_phasefield_cell;

    FEValues<dim> m_fe_values;

    const BlockVector<double>&       m_solution_previous_step;
    std::vector<double>              m_phasefield_previous_step_cell;

    const double                     m_delta_time;

    ScratchData_UQPH(const FiniteElement<dim> & fe_cell,
                     const QGauss<dim> &        qf_cell,
                     const UpdateFlags          uf_cell,
                     const BlockVector<double> &solution_total,
		     const BlockVector<double> &solution_old,
		     const double delta_time)
      : m_solution_UQPH(solution_total)
      , m_solution_symm_grads_u_cell(qf_cell.size())
      , m_solution_values_phasefield_cell(qf_cell.size())
      , m_solution_grad_phasefield_cell(qf_cell.size())
      , m_fe_values(fe_cell, qf_cell, uf_cell)
      , m_solution_previous_step(solution_old)
      , m_phasefield_previous_step_cell(qf_cell.size())
      , m_delta_time(delta_time)
    {}

    ScratchData_UQPH(const ScratchData_UQPH &rhs)
      : m_solution_UQPH(rhs.m_solution_UQPH)
      , m_solution_symm_grads_u_cell(rhs.m_solution_symm_grads_u_cell)
      , m_solution_values_phasefield_cell(rhs.m_solution_values_phasefield_cell)
      , m_solution_grad_phasefield_cell(rhs.m_solution_grad_phasefield_cell)
      , m_fe_values(rhs.m_fe_values.get_fe(),
                    rhs.m_fe_values.get_quadrature(),
                    rhs.m_fe_values.get_update_flags())
      , m_solution_previous_step(rhs.m_solution_previous_step)
      , m_phasefield_previous_step_cell(rhs.m_phasefield_previous_step_cell)
      , m_delta_time(rhs.m_delta_time)
    {}

    void reset()
    {
      const unsigned int n_q_points = m_solution_symm_grads_u_cell.size();
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          m_solution_symm_grads_u_cell[q]  = 0.0;
          m_solution_values_phasefield_cell[q] = 0.0;
          m_solution_grad_phasefield_cell[q] = 0.0;
          m_phasefield_previous_step_cell[q] = 0.0;
        }
    }
  };

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::update_qph_incremental_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData_UQPH & scratch,
    PerTaskData_UQPH & /*data*/)
  {
    scratch.reset();

    scratch.m_fe_values.reinit(cell);

    const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
      m_quadrature_point_history.get_data(cell);
    Assert(lqph.size() == m_n_q_points, ExcInternalError());

    const FEValuesExtractors::Vector displacement(0);

    scratch.m_fe_values[m_u_fe].get_function_symmetric_gradients(
      scratch.m_solution_UQPH, scratch.m_solution_symm_grads_u_cell);
    scratch.m_fe_values[m_d_fe].get_function_values(
      scratch.m_solution_UQPH, scratch.m_solution_values_phasefield_cell);
    scratch.m_fe_values[m_d_fe].get_function_gradients(
      scratch.m_solution_UQPH, scratch.m_solution_grad_phasefield_cell);

    scratch.m_fe_values[m_d_fe].get_function_values(
      scratch.m_solution_previous_step, scratch.m_phasefield_previous_step_cell);

    for (const unsigned int q_point :
         scratch.m_fe_values.quadrature_point_indices())
      lqph[q_point]->update_field_values(scratch.m_solution_symm_grads_u_cell[q_point],
                                         scratch.m_solution_values_phasefield_cell[q_point],
					 scratch.m_solution_grad_phasefield_cell[q_point],
					 scratch.m_phasefield_previous_step_cell[q_point],
					 scratch.m_delta_time);
  }

  template <int dim>
  struct PhaseFieldMonolithicSolve<dim>::PerTaskData_ASM
  {
    FullMatrix<double>                   m_cell_matrix;
    Vector<double>                       m_cell_rhs;
    std::vector<types::global_dof_index> m_local_dof_indices;

    PerTaskData_ASM(const unsigned int dofs_per_cell)
      : m_cell_matrix(dofs_per_cell, dofs_per_cell)
      , m_cell_rhs(dofs_per_cell)
      , m_local_dof_indices(dofs_per_cell)
    {}

    void reset()
    {
      m_cell_matrix = 0.0;
      m_cell_rhs    = 0.0;
    }
  };

  template <int dim>
  struct PhaseFieldMonolithicSolve<dim>::PerTaskData_ASM_RHS_BFGS
  {
    Vector<double>                       m_cell_rhs;
    std::vector<types::global_dof_index> m_local_dof_indices;

    PerTaskData_ASM_RHS_BFGS(const unsigned int dofs_per_cell)
      : m_cell_rhs(dofs_per_cell)
      , m_local_dof_indices(dofs_per_cell)
    {}

    void reset()
    {
      m_cell_rhs    = 0.0;
    }
  };

  template <int dim>
  struct PhaseFieldMonolithicSolve<dim>::ScratchData_ASM
  {
    FEValues<dim>     m_fe_values;
    FEFaceValues<dim> m_fe_face_values;

    std::vector<std::vector<double>>                  m_Nx_phasefield;      // shape function values for phase-field
    std::vector<std::vector<Tensor<1, dim>>>          m_grad_Nx_phasefield; // gradient of shape function values for phase field

    std::vector<std::vector<Tensor<1, dim>>>          m_Nx_disp;       // shape function values for displacement
    std::vector<std::vector<Tensor<2, dim>>>          m_grad_Nx_disp;  // gradient of shape function values for displacement
    std::vector<std::vector<SymmetricTensor<2, dim>>> m_symm_grad_Nx_disp;  // symmetric gradient of shape function values for displacement

    const BlockVector<double>&       m_solution_previous_step;
    std::vector<double>              m_phasefield_previous_step_cell;

    ScratchData_ASM(const FiniteElement<dim> & fe_cell,
                    const QGauss<dim> &        qf_cell,
                    const UpdateFlags          uf_cell,
		    const QGauss<dim - 1> &    qf_face,
		    const UpdateFlags          uf_face,
		    const BlockVector<double>& solution_old)
      : m_fe_values(fe_cell, qf_cell, uf_cell)
      , m_fe_face_values(fe_cell, qf_face, uf_face)
      , m_Nx_phasefield(qf_cell.size(),
	                std::vector<double>(fe_cell.n_dofs_per_cell()))
      , m_grad_Nx_phasefield(qf_cell.size(),
		             std::vector<Tensor<1, dim>>(fe_cell.n_dofs_per_cell()))
      , m_Nx_disp(qf_cell.size(),
		  std::vector<Tensor<1, dim>>(fe_cell.n_dofs_per_cell()))
      , m_grad_Nx_disp(qf_cell.size(),
                       std::vector<Tensor<2, dim>>(fe_cell.n_dofs_per_cell()))
      , m_symm_grad_Nx_disp(qf_cell.size(),
                            std::vector<SymmetricTensor<2, dim>>(fe_cell.n_dofs_per_cell()))
      , m_solution_previous_step(solution_old)
      , m_phasefield_previous_step_cell(qf_cell.size())
    {}

    ScratchData_ASM(const ScratchData_ASM &rhs)
      : m_fe_values(rhs.m_fe_values.get_fe(),
                    rhs.m_fe_values.get_quadrature(),
                    rhs.m_fe_values.get_update_flags())
      , m_fe_face_values(rhs.m_fe_face_values.get_fe(),
	                 rhs.m_fe_face_values.get_quadrature(),
	                 rhs.m_fe_face_values.get_update_flags())
      , m_Nx_phasefield(rhs.m_Nx_phasefield)
      , m_grad_Nx_phasefield(rhs.m_grad_Nx_phasefield)
      , m_Nx_disp(rhs.m_Nx_disp)
      , m_grad_Nx_disp(rhs.m_grad_Nx_disp)
      , m_symm_grad_Nx_disp(rhs.m_symm_grad_Nx_disp)
      , m_solution_previous_step(rhs.m_solution_previous_step)
      , m_phasefield_previous_step_cell(rhs.m_phasefield_previous_step_cell)
    {}

    void reset()
    {
      const unsigned int n_q_points      = m_Nx_phasefield.size();
      const unsigned int n_dofs_per_cell = m_Nx_phasefield[0].size();
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          Assert(m_Nx_phasefield[q_point].size() == n_dofs_per_cell,
		 ExcInternalError());

          Assert(m_grad_Nx_phasefield[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          Assert(m_Nx_disp[q_point].size() == n_dofs_per_cell,
		 ExcInternalError());

          Assert(m_grad_Nx_disp[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          Assert(m_symm_grad_Nx_disp[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          m_phasefield_previous_step_cell[q_point] = 0.0;
          for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
            {
              m_Nx_phasefield[q_point][k]           = 0.0;
              m_grad_Nx_phasefield[q_point][k]      = 0.0;
              m_Nx_disp[q_point][k]                 = 0.0;
              m_grad_Nx_disp[q_point][k]            = 0.0;
              m_symm_grad_Nx_disp[q_point][k]       = 0.0;
            }
        }
    }
  };


  template <int dim>
  struct PhaseFieldMonolithicSolve<dim>::ScratchData_ASM_RHS_BFGS
  {
    FEValues<dim>     m_fe_values;
    FEFaceValues<dim> m_fe_face_values;

    std::vector<std::vector<double>>                  m_Nx_phasefield;      // shape function values for phase-field
    std::vector<std::vector<Tensor<1, dim>>>          m_grad_Nx_phasefield; // gradient of shape function values for phase field

    std::vector<std::vector<Tensor<1, dim>>>          m_Nx_disp;       // shape function values for displacement
    std::vector<std::vector<Tensor<2, dim>>>          m_grad_Nx_disp;  // gradient of shape function values for displacement
    std::vector<std::vector<SymmetricTensor<2, dim>>> m_symm_grad_Nx_disp;  // symmetric gradient of shape function values for displacement

    const BlockVector<double>&       m_solution_previous_step;
    std::vector<double>              m_phasefield_previous_step_cell;

    ScratchData_ASM_RHS_BFGS(const FiniteElement<dim> & fe_cell,
                             const QGauss<dim> &        qf_cell,
                             const UpdateFlags          uf_cell,
		             const QGauss<dim - 1> &    qf_face,
		             const UpdateFlags          uf_face,
		             const BlockVector<double>& solution_old)
      : m_fe_values(fe_cell, qf_cell, uf_cell)
      , m_fe_face_values(fe_cell, qf_face, uf_face)
      , m_Nx_phasefield(qf_cell.size(),
	                std::vector<double>(fe_cell.n_dofs_per_cell()))
      , m_grad_Nx_phasefield(qf_cell.size(),
		             std::vector<Tensor<1, dim>>(fe_cell.n_dofs_per_cell()))
      , m_Nx_disp(qf_cell.size(),
		  std::vector<Tensor<1, dim>>(fe_cell.n_dofs_per_cell()))
      , m_grad_Nx_disp(qf_cell.size(),
                       std::vector<Tensor<2, dim>>(fe_cell.n_dofs_per_cell()))
      , m_symm_grad_Nx_disp(qf_cell.size(),
                            std::vector<SymmetricTensor<2, dim>>(fe_cell.n_dofs_per_cell()))
      , m_solution_previous_step(solution_old)
      , m_phasefield_previous_step_cell(qf_cell.size())
    {}

    ScratchData_ASM_RHS_BFGS(const ScratchData_ASM_RHS_BFGS &rhs)
      : m_fe_values(rhs.m_fe_values.get_fe(),
                    rhs.m_fe_values.get_quadrature(),
                    rhs.m_fe_values.get_update_flags())
      , m_fe_face_values(rhs.m_fe_face_values.get_fe(),
	                 rhs.m_fe_face_values.get_quadrature(),
	                 rhs.m_fe_face_values.get_update_flags())
      , m_Nx_phasefield(rhs.m_Nx_phasefield)
      , m_grad_Nx_phasefield(rhs.m_grad_Nx_phasefield)
      , m_Nx_disp(rhs.m_Nx_disp)
      , m_grad_Nx_disp(rhs.m_grad_Nx_disp)
      , m_symm_grad_Nx_disp(rhs.m_symm_grad_Nx_disp)
      , m_solution_previous_step(rhs.m_solution_previous_step)
      , m_phasefield_previous_step_cell(rhs.m_phasefield_previous_step_cell)
    {}

    void reset()
    {
      const unsigned int n_q_points      = m_Nx_phasefield.size();
      const unsigned int n_dofs_per_cell = m_Nx_phasefield[0].size();
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          Assert(m_Nx_phasefield[q_point].size() == n_dofs_per_cell,
		 ExcInternalError());

          Assert(m_grad_Nx_phasefield[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          Assert(m_Nx_disp[q_point].size() == n_dofs_per_cell,
		 ExcInternalError());

          Assert(m_grad_Nx_disp[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          Assert(m_symm_grad_Nx_disp[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          m_phasefield_previous_step_cell[q_point] = 0.0;
          for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
            {
              m_Nx_phasefield[q_point][k]           = 0.0;
              m_grad_Nx_phasefield[q_point][k]      = 0.0;
              m_Nx_disp[q_point][k]                 = 0.0;
              m_grad_Nx_disp[q_point][k]            = 0.0;
              m_symm_grad_Nx_disp[q_point][k]       = 0.0;
            }
        }
    }
  };

  // constructor has no return type
  template <int dim>
  PhaseFieldMonolithicSolve<dim>::PhaseFieldMonolithicSolve(const std::string &input_file)
    : m_parameters(input_file)
    , m_triangulation(Triangulation<dim>::maximum_smoothing)
    , m_time(m_parameters.m_end_time)
    , m_logfile(m_parameters.m_logfile_name)
    , m_timer(m_logfile, TimerOutput::summary, TimerOutput::wall_times)
    , m_dof_handler(m_triangulation)
    , m_fe(FE_Q<dim>(m_parameters.m_poly_degree),
	   dim, // displacement
	   FE_Q<dim>(m_parameters.m_poly_degree),
	   1)   // phasefield
    , m_dofs_per_cell(m_fe.n_dofs_per_cell())
    , m_u_fe(m_first_u_component)
    , m_d_fe(m_d_component)
    , m_dofs_per_block(m_n_blocks)
    , m_qf_cell(m_parameters.m_quad_order)
    , m_qf_face(m_parameters.m_quad_order)
    , m_n_q_points(m_qf_cell.size())
    , m_vol_reference(0.0)
  {}

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::make_grid()
  {
    if (m_parameters.m_scenario == 1)
      make_grid_case_1();
    else if (m_parameters.m_scenario == 2)
      make_grid_case_2();
    else if (m_parameters.m_scenario == 3)
      make_grid_case_3();
    else if (m_parameters.m_scenario == 4)
      make_grid_case_4();
    else if (m_parameters.m_scenario == 5)
      make_grid_case_5();
    else if (m_parameters.m_scenario == 6)
      make_grid_case_6();
    else if (m_parameters.m_scenario == 7)
      make_grid_case_7();
    else if (m_parameters.m_scenario == 8)
      make_grid_case_8();
    else if (m_parameters.m_scenario == 9)
      make_grid_case_9();
    else if (m_parameters.m_scenario == 11)
      make_grid_case_11();
    else
      Assert(false, ExcMessage("The scenario has not been implemented!"));

    m_logfile << "\t\tTriangulation:"
              << "\n\t\t\tNumber of active cells: "
              << m_triangulation.n_active_cells()
              << "\n\t\t\tNumber of used vertices: "
              << m_triangulation.n_used_vertices()
	      << std::endl;

    std::ofstream out("original_mesh.vtu");
    GridOut       grid_out;
    grid_out.write_vtu(m_triangulation, out);

    m_vol_reference = GridTools::volume(m_triangulation);
    m_logfile << "\t\tGrid:\n\t\t\tReference volume: " << m_vol_reference << std::endl;
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::make_grid_case_1()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\tSquare tension (unstructured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim==2, ExcMessage("The dimension has to be 2D!"));

    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f("square_tension_unstructured.msh");
    gridin.read_msh(f);

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[1] + 0.5 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 0.5 ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else
	        face->set_boundary_id(2);
	    }
	}

    m_triangulation.refine_global(m_parameters.m_global_refine_times);

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	unsigned int material_id;
	double length_scale;
	for (unsigned int i = 0; i < m_parameters.m_local_prerefine_times; i++)
	  {
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (   std::fabs(cell->center()[1]) < 0.01
		    && cell->center()[0] > 0.495)
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      cell->set_refine_flag();
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (   std::fabs(cell->center()[1] - 0.0) < 0.05
		    && std::fabs(cell->center()[0] - 0.5) < 0.05)
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }


  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::make_grid_case_2()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tSquare shear (unstructured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim==2, ExcMessage("The dimension has to be 2D!"));

    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f("square_shear_unstructured.msh");
    gridin.read_msh(f);

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[1] + 0.5 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 0.5 ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else if (   (std::fabs(face->center()[0] - 0.0 ) < 1.0e-9)
		       || (std::fabs(face->center()[0] - 1.0 ) < 1.0e-9))
	        face->set_boundary_id(2);
	      else
	        face->set_boundary_id(3);
	    }
	}

    m_triangulation.refine_global(m_parameters.m_global_refine_times);

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	unsigned int material_id;
	double length_scale;
	for (unsigned int i = 0; i < m_parameters.m_local_prerefine_times; i++)
	  {
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    (cell->center()[0] > 0.45)
		     && (cell->center()[1] < 0.05) )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      cell->set_refine_flag();
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    std::fabs(cell->center()[0] - 0.5) < 0.025
		     && cell->center()[1] < 0.0 && cell->center()[1] > -0.025)
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::make_grid_case_3()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\tSquare tension (structured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim==2, ExcMessage("The dimension has to be 2D!"));

    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f("square_tension_structured.msh");
    gridin.read_msh(f);

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[1] - 0.0 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 1.0 ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else
	        face->set_boundary_id(2);
	    }
	}

    m_triangulation.refine_global(m_parameters.m_global_refine_times);

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	unsigned int material_id;
	double length_scale;
	for (unsigned int i = 0; i < m_parameters.m_local_prerefine_times; i++)
	  {
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    (std::fabs(cell->center()[1] - 0.5) < 0.025)
		     && (cell->center()[0] > 0.475) )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      cell->set_refine_flag();
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    std::fabs(cell->center()[0] - 0.5) < 0.025
		     && std::fabs(cell->center()[1] - 0.5) < 0.025 )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::make_grid_case_4()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tSquare shear (structured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim==2, ExcMessage("The dimension has to be 2D!"));

    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f("square_shear_structured.msh");
    gridin.read_msh(f);

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[1] - 0.0 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 1.0 ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else if (   (std::fabs(face->center()[0] - 0.0 ) < 1.0e-9)
		       || (std::fabs(face->center()[0] - 1.0 ) < 1.0e-9))
	        face->set_boundary_id(2);
	      else
	        face->set_boundary_id(3);
	    }
	}

    m_triangulation.refine_global(m_parameters.m_global_refine_times);

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	unsigned int material_id;
	double length_scale;
	for (unsigned int i = 0; i < m_parameters.m_local_prerefine_times; i++)
	  {
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    (cell->center()[0] > 0.475)
		     && (cell->center()[1] < 0.525) )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      cell->set_refine_flag();
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    std::fabs(cell->center()[0] - 0.5) < 0.025
		     && cell->center()[1] < 0.5 && cell->center()[1] > 0.475 )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::make_grid_case_5()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tThree-point bending (structured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim==2, ExcMessage("The dimension has to be 2D!"));

    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f("three_point_bending_structured.msh");
    gridin.read_msh(f);

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[1] - 0.0 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 2.0 ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else
	        face->set_boundary_id(2);
	    }
	}

    m_triangulation.refine_global(m_parameters.m_global_refine_times);

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	for (const auto &cell : m_triangulation.active_cell_iterators())
	  {
	    if (    std::fabs(cell->center()[0] - 4.0) < 0.075
		 && cell->center()[1] < 1.6)
	      {
		cell->set_refine_flag();
	      }
	  }
	m_triangulation.execute_coarsening_and_refinement();

	unsigned int material_id;
	double length_scale;
	for (unsigned int i = 0; i < m_parameters.m_local_prerefine_times; i++)
	  {
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    std::fabs(cell->center()[0] - 4.0) < 0.05
		     && cell->center()[1] < 1.6)
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      cell->set_refine_flag();
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    std::fabs(cell->center()[0] - 4.0) < 0.075
		     && std::fabs(cell->center()[1] - 0.4) < 0.075 )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::make_grid_case_6()
  {
    AssertThrow(dim==3, ExcMessage("The dimension has to be 3D!"));

    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tSphere inclusion (3D structured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    Triangulation<dim> tria_inner;
    GridGenerator::hyper_ball(tria_inner, Point<dim>(), 0.5);

    Triangulation<dim> tria_outer;
    GridGenerator::hyper_shell(
      tria_outer, Point<dim>(), 0.5, std::sqrt(dim), 2 * dim);

    Triangulation<dim> tmp_triangulation;

    GridGenerator::merge_triangulations(tria_inner, tria_outer, tmp_triangulation);

    tmp_triangulation.reset_all_manifolds();
    tmp_triangulation.set_all_manifold_ids(0);

    for (const auto &cell : tmp_triangulation.cell_iterators())
      {
        for (const auto &face : cell->face_iterators())
          {
            bool face_at_sphere_boundary = true;
            for (const auto v : face->vertex_indices())
              {
                if (std::abs(face->vertex(v).norm_square() - 0.25) > 1e-12)
                  face_at_sphere_boundary = false;
              }
            if (face_at_sphere_boundary)
              face->set_all_manifold_ids(1);
          }
        if (cell->center().norm_square() < 0.25)
          cell->set_material_id(1);
        else
          cell->set_material_id(0);
      }

    tmp_triangulation.set_manifold(1, SphericalManifold<dim>());

    TransfiniteInterpolationManifold<dim> transfinite_manifold;
    transfinite_manifold.initialize(tmp_triangulation);
    tmp_triangulation.set_manifold(0, transfinite_manifold);

    tmp_triangulation.refine_global(m_parameters.m_global_refine_times);

    std::set<typename Triangulation< dim >::active_cell_iterator >
      cells_to_remove;

    for (const auto &cell : tmp_triangulation.active_cell_iterators())
      {
	if (   cell->center()[0] < 0.0
	    || cell->center()[1] < 0.0
	    || cell->center()[2] < 0.0)
	  {
	    cells_to_remove.insert(cell);
	  }
      }

    GridGenerator::create_triangulation_with_removed_cells(tmp_triangulation,
							   cells_to_remove,
							   m_triangulation);

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[0] - 0.0 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 0.0 ) < 1.0e-9)
		face->set_boundary_id(1);
	      else if (std::fabs(face->center()[2] - 0.0 ) < 1.0e-9)
		face->set_boundary_id(2);
	      else if (std::fabs(face->center()[2] - 1.0 ) < 1.0e-9)
		face->set_boundary_id(3);
	      else
		face->set_boundary_id(4);
	    }
	}

    if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    cell->center()[2] > 0.525
		     && cell->center()[2] < 0.575
		     && cell->center()[0] < 0.05
		     && cell->center()[1] < 0.05 )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::cbrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
			cell->set_refine_flag();
			initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
		    ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::make_grid_case_7()
  {
    AssertThrow(dim==3, ExcMessage("The dimension has to be 3D!"));

    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tSphere inclusion (3D structured version 2)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    Triangulation<dim> tria_inner;
    GridGenerator::hyper_ball(tria_inner, Point<dim>(), 0.49);

    Triangulation<dim> tria_outer;
    GridGenerator::hyper_shell(
      tria_outer, Point<dim>(), 0.49, std::sqrt(dim)*0.5, 2 * dim);

    Triangulation<dim> cube1;
    GridGenerator::hyper_rectangle(cube1, Point<dim>(0, 0, 0.5), Point<dim>(1, 1, 1.5));
    Triangulation<dim> cube2;
    GridGenerator::hyper_rectangle(cube2, Point<dim>(0, 0.5, -0.5), Point<dim>(1, 1.5, 0.5));
    Triangulation<dim> cube3;
    GridGenerator::hyper_rectangle(cube3, Point<dim>(0.5, -0.5, -0.5), Point<dim>(1.5, 0.5, 0.5));

    Triangulation<dim> tmp_triangulation;
    GridGenerator::merge_triangulations({&tria_inner, &tria_outer,
                                         &cube1, &cube2, &cube3}, tmp_triangulation);

    tmp_triangulation.reset_all_manifolds();
    tmp_triangulation.set_all_manifold_ids(0);

    for (const auto &cell : tmp_triangulation.cell_iterators())
      {
        for (const auto &face : cell->face_iterators())
          {
            bool face_at_sphere_boundary = true;
            for (const auto v : face->vertex_indices())
              {
                if (std::abs(face->vertex(v).norm_square() - 0.49 * 0.49) > 1e-12)
                  face_at_sphere_boundary = false;
              }
            if (face_at_sphere_boundary)
              face->set_all_manifold_ids(1);
          }
        if (cell->center().norm_square() < 0.1)
          cell->set_material_id(1);
        else
          cell->set_material_id(0);
      }

    tmp_triangulation.set_manifold(1, SphericalManifold<dim>());

    TransfiniteInterpolationManifold<dim> transfinite_manifold;
    transfinite_manifold.initialize(tmp_triangulation);
    tmp_triangulation.set_manifold(0, transfinite_manifold);

    tmp_triangulation.refine_global(m_parameters.m_global_refine_times);

    std::set<typename Triangulation< dim >::active_cell_iterator >
      cells_to_remove;

    for (const auto &cell : tmp_triangulation.active_cell_iterators())
      {
	if (   cell->center()[0] < 0.0
	    || cell->center()[1] < 0.0
	    || cell->center()[2] < 0.0
	    || cell->center()[0] > 1.0
	    || cell->center()[1] > 1.0
	    || cell->center()[2] > 1.0)
	  {
	    cells_to_remove.insert(cell);
	  }
      }

    GridGenerator::create_triangulation_with_removed_cells(tmp_triangulation,
							   cells_to_remove,
							   m_triangulation);

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[0] - 0.0 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 0.0 ) < 1.0e-9)
		face->set_boundary_id(1);
	      else if (std::fabs(face->center()[2] - 0.0 ) < 1.0e-9)
		face->set_boundary_id(2);
	      else if (std::fabs(face->center()[2] - 1.0 ) < 1.0e-9)
		face->set_boundary_id(3);
	      else
		face->set_boundary_id(4);
	    }
	}

    if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    cell->center()[2] > 0.505
		     && cell->center()[2] < 0.575
		     && cell->center()[0] < 0.05
		     && cell->center()[1] < 0.05 )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::cbrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
			cell->set_refine_flag();
			initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
		    ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }


  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::make_grid_case_8()
  {
    AssertThrow(dim==3, ExcMessage("The dimension has to be 3D!"));

    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tSphere inclusion (3D structured version 2 with barriers)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    Triangulation<dim> tria_inner;
    GridGenerator::hyper_ball(tria_inner, Point<dim>(), 0.49);

    Triangulation<dim> tria_outer;
    GridGenerator::hyper_shell(
      tria_outer, Point<dim>(), 0.49, std::sqrt(dim)*0.5, 2 * dim);

    Triangulation<dim> cube1;
    GridGenerator::hyper_rectangle(cube1, Point<dim>(0, 0, 0.5), Point<dim>(1, 1, 1.5));
    Triangulation<dim> cube2;
    GridGenerator::hyper_rectangle(cube2, Point<dim>(0, 0.5, -0.5), Point<dim>(1, 1.5, 0.5));
    Triangulation<dim> cube3;
    GridGenerator::hyper_rectangle(cube3, Point<dim>(0.5, -0.5, -0.5), Point<dim>(1.5, 0.5, 0.5));

    Triangulation<dim> tmp_triangulation;
    GridGenerator::merge_triangulations({&tria_inner, &tria_outer,
                                         &cube1, &cube2, &cube3}, tmp_triangulation);

    tmp_triangulation.reset_all_manifolds();
    tmp_triangulation.set_all_manifold_ids(0);

    for (const auto &cell : tmp_triangulation.cell_iterators())
      {
        for (const auto &face : cell->face_iterators())
          {
            bool face_at_sphere_boundary = true;
            for (const auto v : face->vertex_indices())
              {
                if (std::abs(face->vertex(v).norm_square() - 0.49 * 0.49) > 1e-12)
                  face_at_sphere_boundary = false;
              }
            if (face_at_sphere_boundary)
              face->set_all_manifold_ids(1);
          }
        if (cell->center().norm_square() < 0.1)
          cell->set_material_id(1);
        else
          cell->set_material_id(0);
      }

    tmp_triangulation.set_manifold(1, SphericalManifold<dim>());

    TransfiniteInterpolationManifold<dim> transfinite_manifold;
    transfinite_manifold.initialize(tmp_triangulation);
    tmp_triangulation.set_manifold(0, transfinite_manifold);

    tmp_triangulation.refine_global(m_parameters.m_global_refine_times);

    // some extra barriers
    for (const auto &cell : tmp_triangulation.cell_iterators())
      {
        if (    std::fabs(cell->center()[1] - 0.75) < 0.05
             && std::fabs(cell->center()[2] - 0.5625) < 0.05
             && std::fabs(cell->center()[0] - 0.0) < 0.2)
          cell->set_material_id(1);

        if (    std::fabs(cell->center()[1] - 0.0) < 0.2
             && std::fabs(cell->center()[2] - 0.5) < 0.1
             && std::fabs(cell->center()[0] - 0.75) < 0.05)
          cell->set_material_id(1);
      }

    std::set<typename Triangulation< dim >::active_cell_iterator >
      cells_to_remove;

    for (const auto &cell : tmp_triangulation.active_cell_iterators())
      {
	if (   cell->center()[0] < 0.0
	    || cell->center()[1] < 0.0
	    || cell->center()[2] < 0.0
	    || cell->center()[0] > 1.0
	    || cell->center()[1] > 1.0
	    || cell->center()[2] > 1.0)
	  {
	    cells_to_remove.insert(cell);
	  }
      }

    GridGenerator::create_triangulation_with_removed_cells(tmp_triangulation,
							   cells_to_remove,
							   m_triangulation);

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[0] - 0.0 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 0.0 ) < 1.0e-9)
		face->set_boundary_id(1);
	      else if (std::fabs(face->center()[2] - 0.0 ) < 1.0e-9)
		face->set_boundary_id(2);
	      else if (std::fabs(face->center()[2] - 1.0 ) < 1.0e-9)
		face->set_boundary_id(3);
	      else
		face->set_boundary_id(4);
	    }
	}

    if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    cell->center()[2] > 0.505
		     && cell->center()[2] < 0.575
		     && cell->center()[0] < 0.05
		     && cell->center()[1] < 0.05 )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::cbrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
			cell->set_refine_flag();
			initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
		    ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }


  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::make_grid_case_9()
  {
    AssertThrow(dim==2, ExcMessage("The dimension has to be 2D!"));

    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tL-shape bending (2D structured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f("L-Shape.msh");
    gridin.read_msh(f);

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[1] - 0.0 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else
	        face->set_boundary_id(1);
	    }
	}

    m_triangulation.refine_global(m_parameters.m_global_refine_times);

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	unsigned int material_id;
	double length_scale;
	for (unsigned int i = 0; i < m_parameters.m_local_prerefine_times; i++)
	  {
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    (cell->center()[1] > 242.0)
		     && (cell->center()[1] < 312.5)
		     && (cell->center()[0] < 258.0) )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      cell->set_refine_flag();
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (             (cell->center()[0] - 250) < 0.0
		     &&          (cell->center()[0] - 240) > 0.0
		     && std::fabs(cell->center()[1] - 250) < 10.0 )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::make_grid_case_11()
  {
    AssertThrow(dim==3, ExcMessage("The dimension has to be 3D!"));

    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tBrokenshire torsion (3D structured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    Triangulation<2> triangulation_2d;

    double const length = 200.0;
    double const width = 50.0;
    double const height = 50.0;
    double const delta_L = 25.0;
    double const tan_theta = delta_L / (0.5*width);

    std::vector<unsigned int> repetitions(2, 1);
    repetitions[0] = 20;
    repetitions[1] = 5;

    Point<2> point1(0.0, 0.0);
    Point<2> point2(length, width);

    GridGenerator::subdivided_hyper_rectangle(triangulation_2d,
					      repetitions,
					      point1,
					      point2 );

    typename Triangulation<2>::vertex_iterator vertex_ptr;
    vertex_ptr = triangulation_2d.begin_active_vertex();
    while (vertex_ptr != triangulation_2d.end_vertex())
      {
	Point<2> & vertex_point = vertex_ptr->vertex();

	const double delta_x = (vertex_point(1) - 0.5*width) * tan_theta;

	if (std::fabs(vertex_point(0) - 0.5*length) < 1.0e-6)
	  {
	    vertex_point(0) += delta_x;
	  }
	else if (std::fabs(vertex_point(0) + length/repetitions[0] - 0.5*length) < 1.0e-6)
	  {
	    vertex_point(0) += (delta_x + length/repetitions[0]*0.5);
	  }
	else if (std::fabs(vertex_point(0) - length/repetitions[0] - 0.5*length) < 1.0e-6)
	  {
	    vertex_point(0) += (delta_x - length/repetitions[0]*0.5);
	  }
	else if (vertex_point(0) < 0.5*length - length/repetitions[0] - 1.0e-6)
	  {
	    vertex_point(0) += (delta_x + length/repetitions[0]*0.5) * vertex_point(0)/(0.5*length - length/repetitions[0]);
	  }
	else if (vertex_point(0) > 0.5*length + length/repetitions[0] + 1.0e-6)
	  {
	    vertex_point(0) += (delta_x - length/repetitions[0]*0.5) * (length - vertex_point(0))/(0.5*length - length/repetitions[0]);
	  }

	++vertex_ptr;
      }

    Triangulation<dim> tmp_triangulation;
    const unsigned int n_layer = repetitions[1] + 1;
    GridGenerator::extrude_triangulation(triangulation_2d, n_layer, height, tmp_triangulation);

    tmp_triangulation.refine_global(m_parameters.m_global_refine_times);

    std::set<typename Triangulation< dim >::active_cell_iterator >
      cells_to_remove;

    for (const auto &cell : tmp_triangulation.active_cell_iterators())
      {
	if (    (std::fabs(cell->center()[0] - (cell->center()[1] - 0.5*width)*tan_theta - 0.5*length) < 2.5)
	     && cell->center()[2] > 0.5* height  )
	  {
	    cells_to_remove.insert(cell);
	  }
      }

    GridGenerator::create_triangulation_with_removed_cells(tmp_triangulation,
							   cells_to_remove,
							   m_triangulation);

    if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (  (std::fabs(cell->center()[0] - (cell->center()[1] - 0.5*width)*tan_theta - 0.5*length) < 5.0)
		    && cell->center()[2] <= 0.5*height
		    && cell->center()[2] > 0.5*height - 5.0 )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::cbrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
			cell->set_refine_flag();
			initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
		    ExcMessage("Selected mesh refinement strategy not implemented!"));
      }


    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[0] - length) < 1.0e-6 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[0] - 0.0) < 1.0e-6 )
		face->set_boundary_id(1);
	      else
		face->set_boundary_id(2);
	    }
	}
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::setup_system()
  {
    m_timer.enter_subsection("Setup system");

    std::vector<unsigned int> block_component(m_n_components,
                                              m_u_dof); // displacement
    block_component[m_d_component] = m_d_dof;           // phasefield

    m_dof_handler.distribute_dofs(m_fe);
    DoFRenumbering::Cuthill_McKee(m_dof_handler);
    DoFRenumbering::component_wise(m_dof_handler, block_component);

    m_constraints.clear();
    DoFTools::make_hanging_node_constraints(m_dof_handler, m_constraints);
    m_constraints.close();

    m_dofs_per_block =
      DoFTools::count_dofs_per_fe_block(m_dof_handler, block_component);

    m_logfile << "\t\tTriangulation:"
              << "\n\t\t\t Number of active cells: "
              << m_triangulation.n_active_cells()
              << "\n\t\t\t Number of used vertices: "
              << m_triangulation.n_used_vertices()
              << "\n\t\t\t Number of active edges: "
              << m_triangulation.n_active_lines()
              << "\n\t\t\t Number of active faces: "
              << m_triangulation.n_active_faces()
              << "\n\t\t\t Number of degrees of freedom (total): "
	      << m_dof_handler.n_dofs()
	      << "\n\t\t\t Number of degrees of freedom (disp): "
	      << m_dofs_per_block[m_u_dof]
	      << "\n\t\t\t Number of degrees of freedom (phasefield): "
	      << m_dofs_per_block[m_d_dof]
              << std::endl;

    m_tangent_matrix.clear();
    {
      BlockDynamicSparsityPattern dsp(m_dofs_per_block, m_dofs_per_block);

      Table<2, DoFTools::Coupling> coupling(m_n_components, m_n_components);
      for (unsigned int ii = 0; ii < m_n_components; ++ii)
        for (unsigned int jj = 0; jj < m_n_components; ++jj)
          coupling[ii][jj] = DoFTools::always;

      DoFTools::make_sparsity_pattern(
        m_dof_handler, coupling, dsp, m_constraints, false);
      m_sparsity_pattern.copy_from(dsp);
    }

    m_tangent_matrix.reinit(m_sparsity_pattern);

    m_system_rhs.reinit(m_dofs_per_block);
    m_solution.reinit(m_dofs_per_block);

    setup_qph();

    m_timer.leave_subsection();
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::make_constraints(const unsigned int it_nr)
  {
    const bool apply_dirichlet_bc = (it_nr == 0);

    if (it_nr > 1)
      {
	if (m_parameters.m_output_iteration_history)
          m_logfile << " --- " << std::flush;
        return;
      }

    if (m_parameters.m_output_iteration_history)
      m_logfile << " CST " << std::flush;

    if (apply_dirichlet_bc)
      {
	m_constraints.clear();
	DoFTools::make_hanging_node_constraints(m_dof_handler,
						m_constraints);

	const FEValuesExtractors::Scalar x_displacement(0);
	const FEValuesExtractors::Scalar y_displacement(1);
	const FEValuesExtractors::Scalar z_displacement(2);

	const FEValuesExtractors::Vector displacements(0);

	if (   m_parameters.m_scenario == 1
	    || m_parameters.m_scenario == 3)
	  {
	    // Dirichlet B,C. bottom surface
	    const int boundary_id_bottom_surface = 0;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_bottom_surface,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(y_displacement));

	    typename Triangulation<dim>::active_vertex_iterator vertex_itr;
	    vertex_itr = m_triangulation.begin_active_vertex();
	    std::vector<types::global_dof_index> node_xy(m_fe.dofs_per_vertex);

	    for (; vertex_itr != m_triangulation.end_vertex(); ++vertex_itr)
	      {
		if (   (std::fabs(vertex_itr->vertex()[0] - 0.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[1] - 0.0) < 1.0e-9) )
		  {
		    node_xy = usr_utilities::get_vertex_dofs(vertex_itr, m_dof_handler);
		  }
	      }
	    m_constraints.add_line(node_xy[0]);
	    m_constraints.set_inhomogeneity(node_xy[0], 0.0);

	    m_constraints.add_line(node_xy[1]);
	    m_constraints.set_inhomogeneity(node_xy[1], 0.0);

	    const int boundary_id_top_surface = 1;
	    /*
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_top_surface,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(x_displacement));
	    */
            const double time_inc = m_time.get_delta_t();
            double disp_magnitude = m_time.get_magnitude();
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_top_surface,
						     Functions::ConstantFunction<dim>(
						       disp_magnitude*time_inc, m_n_components),
						     m_constraints,
						     m_fe.component_mask(y_displacement));
	  }
	else if (   m_parameters.m_scenario == 2
	         || m_parameters.m_scenario == 4)
	  {
	    // Dirichlet B,C. bottom surface
	    const int boundary_id_bottom_surface = 0;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_bottom_surface,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(displacements));

	    const int boundary_id_top_surface = 1;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_top_surface,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(y_displacement));

	    const double time_inc = m_time.get_delta_t();
	    double disp_magnitude = m_time.get_magnitude();
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_top_surface,
						     Functions::ConstantFunction<dim>(
						       disp_magnitude*time_inc, m_n_components),
						     m_constraints,
						     m_fe.component_mask(x_displacement));

	    const int boundary_id_side_surfaces = 2;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_side_surfaces,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(y_displacement));
	  }
	else if (m_parameters.m_scenario == 5)
	  {
	    typename Triangulation<dim>::active_vertex_iterator vertex_itr;
	    vertex_itr = m_triangulation.begin_active_vertex();
	    std::vector<types::global_dof_index> node_bottomleft(m_fe.dofs_per_vertex);
	    std::vector<types::global_dof_index> node_bottomright(m_fe.dofs_per_vertex);
	    std::vector<types::global_dof_index> node_topcenter(m_fe.dofs_per_vertex);

	    for (; vertex_itr != m_triangulation.end_vertex(); ++vertex_itr)
	      {
		if (   (std::fabs(vertex_itr->vertex()[0] - 0.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[1] - 0.0) < 1.0e-9) )
		  {
		    node_bottomleft = usr_utilities::get_vertex_dofs(vertex_itr, m_dof_handler);
		  }
		if (   (std::fabs(vertex_itr->vertex()[0] - 8.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[1] - 0.0) < 1.0e-9) )
		  {
		    node_bottomright = usr_utilities::get_vertex_dofs(vertex_itr, m_dof_handler);
		  }
		if (   (std::fabs(vertex_itr->vertex()[0] - 4.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[1] - 2.0) < 1.0e-9) )
		  {
		    node_topcenter = usr_utilities::get_vertex_dofs(vertex_itr, m_dof_handler);
		  }
	      }
	    // bottom-left node fixed in both x- and y-directions
	    m_constraints.add_line(node_bottomleft[0]);
	    m_constraints.set_inhomogeneity(node_bottomleft[0], 0.0);

	    m_constraints.add_line(node_bottomleft[1]);
	    m_constraints.set_inhomogeneity(node_bottomleft[1], 0.0);

	    // bottom-right node only fixed in y-direction
	    m_constraints.add_line(node_bottomright[1]);
	    m_constraints.set_inhomogeneity(node_bottomright[1], 0.0);

	    // top-center node applied with y-displacement
	    const double time_inc = m_time.get_delta_t();
	    double disp_magnitude = m_time.get_magnitude();

	    m_constraints.add_line(node_topcenter[1]);
	    m_constraints.set_inhomogeneity(node_topcenter[1], disp_magnitude*time_inc);
	  }
	else if (   m_parameters.m_scenario == 6
	         || m_parameters.m_scenario == 7
		 || m_parameters.m_scenario == 8)
	  {
	    const int x0_surface = 0;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     x0_surface,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(x_displacement));
	    const int y0_surface = 1;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     y0_surface,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(y_displacement));
	    const int z0_surface = 2;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     z0_surface,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(z_displacement));

	    const int z1_surface = 3;
	    const double time_inc = m_time.get_delta_t();
	    double disp_magnitude = m_time.get_magnitude();
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     z1_surface,
						     Functions::ConstantFunction<dim>(
						       disp_magnitude*time_inc, m_n_components),
						     m_constraints,
						     m_fe.component_mask(z_displacement));
	  }
	else if (m_parameters.m_scenario == 9)
	  {
	    // Dirichlet B,C. bottom surface
	    const int boundary_id_bottom_surface = 0;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_bottom_surface,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(displacements));

	    typename Triangulation<dim>::active_vertex_iterator vertex_itr;
	    vertex_itr = m_triangulation.begin_active_vertex();
	    std::vector<types::global_dof_index> node_disp_control(m_fe.dofs_per_vertex);

	    for (; vertex_itr != m_triangulation.end_vertex(); ++vertex_itr)
	      {
		if (   (std::fabs(vertex_itr->vertex()[0] - 470.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[1] - 250.0) < 1.0e-9) )
		  {
		    node_disp_control = usr_utilities::get_vertex_dofs(vertex_itr, m_dof_handler);
	            // node applied with y-displacement
		    const double time_inc = m_time.get_delta_t();
		    double disp_magnitude = m_time.get_magnitude();

		    m_constraints.add_line(node_disp_control[1]);
		    m_constraints.set_inhomogeneity(node_disp_control[1], disp_magnitude*time_inc);
		  }
	      }
	  }
	else if (m_parameters.m_scenario == 11)
	  {
	    // Dirichlet B,C. right surface
	    const int boundary_id_right_surface = 0;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_right_surface,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(displacements));

	    // Dirichlet B,C. left surface
	    const int boundary_id_left_surface = 1;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_left_surface,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(x_displacement));

	    typename Triangulation<dim>::active_vertex_iterator vertex_itr;
	    vertex_itr = m_triangulation.begin_active_vertex();
	    std::vector<types::global_dof_index> node_rotate(m_fe.dofs_per_vertex);
	    double node_dist = 0.0;
	    double disp_mag = 0.0;
	    double angle_theta = 0.0;
	    double disp_y = 0;
	    double disp_z = 0;

	    for (; vertex_itr != m_triangulation.end_vertex(); ++vertex_itr)
	      {
		if (std::fabs(vertex_itr->vertex()[0] - 0.0) < 1.0e-9)
		  {
		    node_rotate = usr_utilities::get_vertex_dofs(vertex_itr, m_dof_handler);
		    node_dist = std::sqrt(  vertex_itr->vertex()[1] * vertex_itr->vertex()[1]
					  + vertex_itr->vertex()[2] * vertex_itr->vertex()[2]);

		    angle_theta = m_time.get_delta_t() * m_time.get_magnitude();
		    disp_mag = node_dist * std::tan(angle_theta);

		    if (node_dist > 0)
		      {
			disp_y = vertex_itr->vertex()[2]/node_dist * disp_mag;
			disp_z = -vertex_itr->vertex()[1]/node_dist * disp_mag;
		      }
		    else
		      {
			disp_y = 0.0;
			disp_z = 0.0;
		      }

		    m_constraints.add_line(node_rotate[1]);
		    m_constraints.set_inhomogeneity(node_rotate[1], disp_y);

		    m_constraints.add_line(node_rotate[2]);
		    m_constraints.set_inhomogeneity(node_rotate[2], disp_z);
		  }
	      }
	  }
	else
	  Assert(false, ExcMessage("The scenario has not been implemented!"));
      }
    else  // inhomogeneous constraints
      {
        if (m_constraints.has_inhomogeneities())
          {
            AffineConstraints<double> homogeneous_constraints(m_constraints);
            for (unsigned int dof = 0; dof != m_dof_handler.n_dofs(); ++dof)
              if (homogeneous_constraints.is_inhomogeneously_constrained(dof))
                homogeneous_constraints.set_inhomogeneity(dof, 0.0);
            m_constraints.clear();
            m_constraints.copy_from(homogeneous_constraints);
          }
      }
    m_constraints.close();
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::assemble_system_newton(const BlockVector<double> & solution_old)
  {
    m_timer.enter_subsection("Assemble system");

    if (m_parameters.m_output_iteration_history)
      m_logfile << " ASM_SYS " << std::flush;

    m_tangent_matrix = 0.0;
    m_system_rhs    = 0.0;

    const UpdateFlags uf_cell(update_values | update_gradients |
			      update_quadrature_points | update_JxW_values);
    const UpdateFlags uf_face(update_values | update_normal_vectors |
                              update_JxW_values);

    PerTaskData_ASM per_task_data(m_fe.n_dofs_per_cell());
    ScratchData_ASM scratch_data(m_fe, m_qf_cell, uf_cell, m_qf_face, uf_face, solution_old);

    auto worker =
      [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
	     ScratchData_ASM & scratch,
	     PerTaskData_ASM & data)
      {
        this->assemble_system_newton_one_cell(cell, scratch, data);
      };

    auto copier = [this](const PerTaskData_ASM &data)
      {
        this->m_constraints.distribute_local_to_global(data.m_cell_matrix,
                                                       data.m_cell_rhs,
                                                       data.m_local_dof_indices,
						       m_tangent_matrix,
						       m_system_rhs);
      };

    WorkStream::run(
      m_dof_handler.active_cell_iterators(),
      worker,
      copier,
      scratch_data,
      per_task_data);

    m_timer.leave_subsection();
  }


  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::assemble_system_B0(const BlockVector<double> & solution_old)
  {
    m_timer.enter_subsection("Assemble B0");

    m_tangent_matrix = 0.0;

    const UpdateFlags uf_cell(update_values | update_gradients |
			      update_quadrature_points | update_JxW_values);
    const UpdateFlags uf_face(update_values | update_normal_vectors |
                              update_JxW_values);

    PerTaskData_ASM per_task_data(m_fe.n_dofs_per_cell());
    ScratchData_ASM scratch_data(m_fe, m_qf_cell, uf_cell, m_qf_face, uf_face, solution_old);

    auto worker =
      [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
	     ScratchData_ASM & scratch,
	     PerTaskData_ASM & data)
      {
        this->assemble_system_B0_one_cell(cell, scratch, data);
      };

    auto copier = [this](const PerTaskData_ASM &data)
      {
        this->m_constraints.distribute_local_to_global(data.m_cell_matrix,
                                                       data.m_local_dof_indices,
						       m_tangent_matrix);
      };

    WorkStream::run(
      m_dof_handler.active_cell_iterators(),
      worker,
      copier,
      scratch_data,
      per_task_data);

    m_timer.leave_subsection();
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::assemble_system_rhs_BFGS_parallel(const BlockVector<double> & solution_old,
								         BlockVector<double> & system_rhs)
  {
    m_timer.enter_subsection("Assemble RHS");

    //m_logfile << " A_RHS " << std::flush;

    system_rhs = 0.0;

    const UpdateFlags uf_cell(update_values | update_gradients |
			      update_quadrature_points | update_JxW_values);
    const UpdateFlags uf_face(update_values | update_normal_vectors |
			      update_JxW_values);

    PerTaskData_ASM_RHS_BFGS per_task_data(m_fe.n_dofs_per_cell());
    ScratchData_ASM_RHS_BFGS scratch_data(m_fe, m_qf_cell, uf_cell, m_qf_face, uf_face, solution_old);

    auto worker =
      [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
	     ScratchData_ASM_RHS_BFGS & scratch,
	     PerTaskData_ASM_RHS_BFGS & data)
      {
        this->assemble_system_rhs_BFGS_one_cell(cell, scratch, data);
      };

    auto copier = [this, &system_rhs](const PerTaskData_ASM_RHS_BFGS &data)
      {
        this->m_constraints.distribute_local_to_global(data.m_cell_rhs,
                                                       data.m_local_dof_indices,
						       system_rhs);
      };

    WorkStream::run(
      m_dof_handler.active_cell_iterators(),
      worker,
      copier,
      scratch_data,
      per_task_data);

    m_timer.leave_subsection();
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::assemble_system_rhs_BFGS_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_ASM_RHS_BFGS & scratch,
      PerTaskData_ASM_RHS_BFGS & data) const
  {
    data.reset();
    scratch.reset();
    scratch.m_fe_values.reinit(cell);
    cell->get_dof_indices(data.m_local_dof_indices);

    scratch.m_fe_values[m_d_fe].get_function_values(
      scratch.m_solution_previous_step, scratch.m_phasefield_previous_step_cell);

    const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
      m_quadrature_point_history.get_data(cell);
    Assert(lqph.size() == m_n_q_points, ExcInternalError());

    const double time_ramp = (m_time.current() / m_time.end());
    std::vector<Tensor<1, dim>> rhs_values(m_n_q_points);

    right_hand_side(scratch.m_fe_values.get_quadrature_points(),
		    rhs_values,
		    m_parameters.m_x_component*1.0,
		    m_parameters.m_y_component*1.0,
		    m_parameters.m_z_component*1.0);

    const double delta_time = m_time.get_delta_t();

    for (const unsigned int q_point : scratch.m_fe_values.quadrature_point_indices())
      {
        for (const unsigned int k : scratch.m_fe_values.dof_indices())
          {
            const unsigned int k_group = m_fe.system_to_base_index(k).first.first;

            if (k_group == m_u_dof)
              {
                scratch.m_Nx_disp[q_point][k] =
                  scratch.m_fe_values[m_u_fe].value(k, q_point);
                scratch.m_grad_Nx_disp[q_point][k] =
                  scratch.m_fe_values[m_u_fe].gradient(k, q_point);
                scratch.m_symm_grad_Nx_disp[q_point][k] =
                  symmetrize(scratch.m_grad_Nx_disp[q_point][k]);
              }
            else if (k_group == m_d_dof)
              {
		scratch.m_Nx_phasefield[q_point][k] =
		  scratch.m_fe_values[m_d_fe].value(k, q_point);
		scratch.m_grad_Nx_phasefield[q_point][k] =
		  scratch.m_fe_values[m_d_fe].gradient(k, q_point);
              }
            else
              Assert(k_group <= m_d_dof, ExcInternalError());
          }
      }

    for (const unsigned int q_point : scratch.m_fe_values.quadrature_point_indices())
      {
	const double length_scale            = lqph[q_point]->get_length_scale();
	const double gc                      = lqph[q_point]->get_critical_energy_release_rate();
	const double eta                     = lqph[q_point]->get_viscosity();
	const double history_strain_energy   = lqph[q_point]->get_history_max_positive_strain_energy();
	const double current_positive_strain_energy = lqph[q_point]->get_current_positive_strain_energy();

	double history_value = history_strain_energy;
	if (current_positive_strain_energy > history_strain_energy)
	  history_value = current_positive_strain_energy;

	const double phasefield_value        = lqph[q_point]->get_phase_field_value();
	const Tensor<1, dim> phasefield_grad = lqph[q_point]->get_phase_field_gradient();

        const std::vector<double>         &      N_phasefield = scratch.m_Nx_phasefield[q_point];
        const std::vector<Tensor<1, dim>> & grad_N_phasefield = scratch.m_grad_Nx_phasefield[q_point];
        const double                old_phasefield = scratch.m_phasefield_previous_step_cell[q_point];

        const SymmetricTensor<2, dim> & cauchy_stress = lqph[q_point]->get_cauchy_stress();

        const std::vector<Tensor<1,dim>> & N_disp = scratch.m_Nx_disp[q_point];
        const std::vector<SymmetricTensor<2, dim>> & symm_grad_N_disp =
          scratch.m_symm_grad_Nx_disp[q_point];
        const double JxW = scratch.m_fe_values.JxW(q_point);

        SymmetricTensor<2, dim> symm_grad_Nx_i_x_C;

        for (const unsigned int i : scratch.m_fe_values.dof_indices())
          {
            const unsigned int i_group = m_fe.system_to_base_index(i).first.first;

            if (i_group == m_u_dof)
              {
                data.m_cell_rhs(i) += (symm_grad_N_disp[i] * cauchy_stress) * JxW;

		// contributions from the body force to right-hand side
		data.m_cell_rhs(i) -= N_disp[i] * rhs_values[q_point] * JxW;
              }
            else if (i_group == m_d_dof)
              {
    	        data.m_cell_rhs(i) += (    gc * length_scale * grad_N_phasefield[i] * phasefield_grad
    	                                +  (   gc / length_scale * phasefield_value
					     + eta / delta_time  * (phasefield_value - old_phasefield)
					     + degradation_function_derivative(phasefield_value) * history_value )
					  * N_phasefield[i]
				      ) * JxW;
              }
            else
              Assert(i_group <= m_d_dof, ExcInternalError());
          }  // i
      }  // q_point

    // if there is surface pressure, this surface pressure always applied to the
    // reference configuration
    const unsigned int face_pressure_id = 100;
    const double p0 = 0.0;

    for (const auto &face : cell->face_iterators())
      if (face->at_boundary() && face->boundary_id() == face_pressure_id)
        {
          scratch.m_fe_face_values.reinit(cell, face);

          for (const unsigned int f_q_point : scratch.m_fe_face_values.quadrature_point_indices())
            {
              const Tensor<1, dim> &N = scratch.m_fe_face_values.normal_vector(f_q_point);

              const double         pressure  = p0 * time_ramp;
              const Tensor<1, dim> traction  = pressure * N;

              for (const unsigned int i : scratch.m_fe_values.dof_indices())
                {
                  const unsigned int i_group = m_fe.system_to_base_index(i).first.first;

                  if (i_group == m_u_dof)
                    {
    		      const unsigned int component_i = m_fe.system_to_component_index(i).first;
    		      const double Ni = scratch.m_fe_face_values.shape_value(i, f_q_point);
    		      const double JxW = scratch.m_fe_face_values.JxW(f_q_point);
    		      data.m_cell_rhs(i) -= (Ni * traction[component_i]) * JxW;
                    }
                }
            }
        }
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::assemble_system_newton_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_ASM & scratch,
      PerTaskData_ASM & data) const
  {
    data.reset();
    scratch.reset();
    scratch.m_fe_values.reinit(cell);
    cell->get_dof_indices(data.m_local_dof_indices);

    scratch.m_fe_values[m_d_fe].get_function_values(
      scratch.m_solution_previous_step, scratch.m_phasefield_previous_step_cell);

    const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
      m_quadrature_point_history.get_data(cell);
    Assert(lqph.size() == m_n_q_points, ExcInternalError());

    const double time_ramp = (m_time.current() / m_time.end());
    std::vector<Tensor<1, dim>> rhs_values(m_n_q_points);

    right_hand_side(scratch.m_fe_values.get_quadrature_points(),
		    rhs_values,
		    m_parameters.m_x_component*1.0,
		    m_parameters.m_y_component*1.0,
		    m_parameters.m_z_component*1.0);

    const double delta_time = m_time.get_delta_t();

    for (const unsigned int q_point : scratch.m_fe_values.quadrature_point_indices())
      {
        for (const unsigned int k : scratch.m_fe_values.dof_indices())
          {
            const unsigned int k_group = m_fe.system_to_base_index(k).first.first;

            if (k_group == m_u_dof)
              {
                scratch.m_Nx_disp[q_point][k] =
                  scratch.m_fe_values[m_u_fe].value(k, q_point);
                scratch.m_grad_Nx_disp[q_point][k] =
                  scratch.m_fe_values[m_u_fe].gradient(k, q_point);
                scratch.m_symm_grad_Nx_disp[q_point][k] =
                  symmetrize(scratch.m_grad_Nx_disp[q_point][k]);
              }
            else if (k_group == m_d_dof)
              {
		scratch.m_Nx_phasefield[q_point][k] =
		  scratch.m_fe_values[m_d_fe].value(k, q_point);
		scratch.m_grad_Nx_phasefield[q_point][k] =
		  scratch.m_fe_values[m_d_fe].gradient(k, q_point);
              }
            else
              Assert(k_group <= m_d_dof, ExcInternalError());
          }
      }

    for (const unsigned int q_point : scratch.m_fe_values.quadrature_point_indices())
      {
	const double length_scale            = lqph[q_point]->get_length_scale();
	const double gc                      = lqph[q_point]->get_critical_energy_release_rate();
	const double eta                     = lqph[q_point]->get_viscosity();
	const double history_strain_energy   = lqph[q_point]->get_history_max_positive_strain_energy();
	const double current_positive_strain_energy = lqph[q_point]->get_current_positive_strain_energy();

	double history_value = history_strain_energy;
	if (current_positive_strain_energy > history_strain_energy)
	  history_value = current_positive_strain_energy;

	const double phasefield_value        = lqph[q_point]->get_phase_field_value();
	const Tensor<1, dim> phasefield_grad = lqph[q_point]->get_phase_field_gradient();

        const std::vector<double>         &      N_phasefield = scratch.m_Nx_phasefield[q_point];
        const std::vector<Tensor<1, dim>> & grad_N_phasefield = scratch.m_grad_Nx_phasefield[q_point];
        const double                old_phasefield = scratch.m_phasefield_previous_step_cell[q_point];

        const SymmetricTensor<2, dim> & cauchy_stress = lqph[q_point]->get_cauchy_stress();
        const SymmetricTensor<2, dim> & cauchy_stress_positive = lqph[q_point]->get_cauchy_stress_positive();
        const SymmetricTensor<4, dim> & mechanical_C  = lqph[q_point]->get_mechanical_C();

        const std::vector<Tensor<1,dim>> & N_disp = scratch.m_Nx_disp[q_point];
        const std::vector<SymmetricTensor<2, dim>> & symm_grad_N_disp =
          scratch.m_symm_grad_Nx_disp[q_point];
        const double JxW = scratch.m_fe_values.JxW(q_point);

        SymmetricTensor<2, dim> symm_grad_Nx_i_x_C;

        for (const unsigned int i : scratch.m_fe_values.dof_indices())
          {
            const unsigned int i_group = m_fe.system_to_base_index(i).first.first;

            if (i_group == m_u_dof)
              {
                data.m_cell_rhs(i) -= (symm_grad_N_disp[i] * cauchy_stress) * JxW;

		// contributions from the body force to right-hand side
		data.m_cell_rhs(i) += N_disp[i] * rhs_values[q_point] * JxW;
              }
            else if (i_group == m_d_dof)
              {
    	        data.m_cell_rhs(i) -= (    gc * length_scale * grad_N_phasefield[i] * phasefield_grad
    	                                +  (   gc / length_scale * phasefield_value
					     + eta / delta_time  * (phasefield_value - old_phasefield)
					     + degradation_function_derivative(phasefield_value) * history_value )
					  * N_phasefield[i]
				      ) * JxW;
              }
            else
              Assert(i_group <= m_d_dof, ExcInternalError());

            if (i_group == m_u_dof)
              {
                symm_grad_Nx_i_x_C = symm_grad_N_disp[i] * mechanical_C;
              }

            for (const unsigned int j : scratch.m_fe_values.dof_indices())
              {
                const unsigned int j_group = m_fe.system_to_base_index(j).first.first;

                if ((i_group == j_group) && (i_group == m_u_dof))
                  {
                    data.m_cell_matrix(i, j) += symm_grad_Nx_i_x_C * symm_grad_N_disp[j] * JxW;
                  }
                else if ((i_group == j_group) && (i_group == m_d_dof))
                  {
                    data.m_cell_matrix(i, j) += (  (  gc/length_scale + eta/delta_time +
                	                              degradation_function_2nd_order_derivative(phasefield_value)
						    * history_value  )
                	                          * N_phasefield[i] * N_phasefield[j]
					          + gc * length_scale * grad_N_phasefield[i] * grad_N_phasefield[j]
					        ) * JxW;
                  }
                else if ((i_group == m_u_dof) && (j_group == m_d_dof))
                  {
                    data.m_cell_matrix(i, j) +=  symm_grad_N_disp[i] * cauchy_stress_positive
                	                       * degradation_function_derivative(phasefield_value)
                	                       * N_phasefield[j] * JxW;
                  }
                else if ((i_group == m_d_dof) && (j_group == m_u_dof))
                  {
                    if (current_positive_strain_energy > history_strain_energy)
                      data.m_cell_matrix(i, j) +=  N_phasefield[i]
			         	         * degradation_function_derivative(phasefield_value)
					         * cauchy_stress_positive
					         * symm_grad_N_disp[j]
					         * JxW;
                    else
                      data.m_cell_matrix(i, j) += 0.0;
                  }
                else
                  Assert((i_group <= m_d_dof) && (j_group <= m_d_dof),
                         ExcInternalError());
              } // j
          }  // i
      }  // q_point

    // if there is surface pressure, this surface pressure always applied to the
    // reference configuration
    const unsigned int face_pressure_id = 100;
    const double p0 = 0.0;

    for (const auto &face : cell->face_iterators())
      if (face->at_boundary() && face->boundary_id() == face_pressure_id)
        {
          scratch.m_fe_face_values.reinit(cell, face);

          for (const unsigned int f_q_point : scratch.m_fe_face_values.quadrature_point_indices())
            {
              const Tensor<1, dim> &N = scratch.m_fe_face_values.normal_vector(f_q_point);

              const double         pressure  = p0 * time_ramp;
              const Tensor<1, dim> traction  = pressure * N;

              for (const unsigned int i : scratch.m_fe_values.dof_indices())
                {
                  const unsigned int i_group = m_fe.system_to_base_index(i).first.first;

                  if (i_group == m_u_dof)
                    {
    		      const unsigned int component_i = m_fe.system_to_component_index(i).first;
    		      const double Ni = scratch.m_fe_face_values.shape_value(i, f_q_point);
    		      const double JxW = scratch.m_fe_face_values.JxW(f_q_point);
    		      data.m_cell_rhs(i) += (Ni * traction[component_i]) * JxW;
                    }
                }
            }
        }
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::assemble_system_B0_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_ASM & scratch,
      PerTaskData_ASM & data) const
  {
    data.reset();
    scratch.reset();
    scratch.m_fe_values.reinit(cell);
    cell->get_dof_indices(data.m_local_dof_indices);

    scratch.m_fe_values[m_d_fe].get_function_values(
      scratch.m_solution_previous_step, scratch.m_phasefield_previous_step_cell);

    const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
      m_quadrature_point_history.get_data(cell);
    Assert(lqph.size() == m_n_q_points, ExcInternalError());

    const double delta_time = m_time.get_delta_t();

    for (const unsigned int q_point : scratch.m_fe_values.quadrature_point_indices())
      {
        for (const unsigned int k : scratch.m_fe_values.dof_indices())
          {
            const unsigned int k_group = m_fe.system_to_base_index(k).first.first;

            if (k_group == m_u_dof)
              {
                scratch.m_Nx_disp[q_point][k] =
                  scratch.m_fe_values[m_u_fe].value(k, q_point);
                scratch.m_grad_Nx_disp[q_point][k] =
                  scratch.m_fe_values[m_u_fe].gradient(k, q_point);
                scratch.m_symm_grad_Nx_disp[q_point][k] =
                  symmetrize(scratch.m_grad_Nx_disp[q_point][k]);
              }
            else if (k_group == m_d_dof)
              {
		scratch.m_Nx_phasefield[q_point][k] =
		  scratch.m_fe_values[m_d_fe].value(k, q_point);
		scratch.m_grad_Nx_phasefield[q_point][k] =
		  scratch.m_fe_values[m_d_fe].gradient(k, q_point);
              }
            else
              Assert(k_group <= m_d_dof, ExcInternalError());
          }
      }

    for (const unsigned int q_point : scratch.m_fe_values.quadrature_point_indices())
      {
	const double length_scale            = lqph[q_point]->get_length_scale();
	const double gc                      = lqph[q_point]->get_critical_energy_release_rate();
	const double eta                     = lqph[q_point]->get_viscosity();
	const double history_strain_energy   = lqph[q_point]->get_history_max_positive_strain_energy();
	const double current_positive_strain_energy = lqph[q_point]->get_current_positive_strain_energy();

	double history_value = history_strain_energy;
	if (current_positive_strain_energy > history_strain_energy)
	  history_value = current_positive_strain_energy;

	const double phasefield_value        = lqph[q_point]->get_phase_field_value();

        const std::vector<double>         &      N_phasefield = scratch.m_Nx_phasefield[q_point];
        const std::vector<Tensor<1, dim>> & grad_N_phasefield = scratch.m_grad_Nx_phasefield[q_point];

        //const SymmetricTensor<2, dim> & cauchy_stress_positive = lqph[q_point]->get_cauchy_stress_positive();
        const SymmetricTensor<4, dim> & mechanical_C  = lqph[q_point]->get_mechanical_C();

        const std::vector<SymmetricTensor<2, dim>> & symm_grad_N_disp =
          scratch.m_symm_grad_Nx_disp[q_point];
        const double JxW = scratch.m_fe_values.JxW(q_point);

        SymmetricTensor<2, dim> symm_grad_Nx_i_x_C;

        for (const unsigned int i : scratch.m_fe_values.dof_indices())
          {
            const unsigned int i_group = m_fe.system_to_base_index(i).first.first;

            if (i_group == m_u_dof)
              {
                symm_grad_Nx_i_x_C = symm_grad_N_disp[i] * mechanical_C;
              }

            for (const unsigned int j : scratch.m_fe_values.dof_indices())
              {
                const unsigned int j_group = m_fe.system_to_base_index(j).first.first;

                if ((i_group == j_group) && (i_group == m_u_dof))
                  {
                    data.m_cell_matrix(i, j) += symm_grad_Nx_i_x_C * symm_grad_N_disp[j] * JxW;
                  }
                else if ((i_group == j_group) && (i_group == m_d_dof))
                  {
                    data.m_cell_matrix(i, j) += (  (   gc/length_scale + eta/delta_time
                	                             + degradation_function_2nd_order_derivative(phasefield_value)
						     * history_value  )
                	                          * N_phasefield[i] * N_phasefield[j]
					          + gc * length_scale * grad_N_phasefield[i] * grad_N_phasefield[j]
					        ) * JxW;
                  }
                else
                  Assert((i_group <= m_d_dof) && (j_group <= m_d_dof),
                         ExcInternalError());
              } // j
          }  // i
      }  // q_point
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::assemble_system_rhs_BFGS(const BlockVector<double> & solution_old,
								BlockVector<double> & system_rhs)
  {
    m_timer.enter_subsection("Assemble RHS");

    //m_logfile << " A_RHS " << std::flush;

    system_rhs = 0.0;

    Vector<double> cell_rhs(m_dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(m_dofs_per_cell);

    const double time_ramp = (m_time.current() / m_time.end());
    const double delta_time = m_time.get_delta_t();

    std::vector<Tensor<1, dim>> rhs_values(m_n_q_points);
    const UpdateFlags uf_cell(update_values | update_gradients |
			      update_quadrature_points | update_JxW_values);
    const UpdateFlags uf_face(update_values | update_normal_vectors |
			      update_JxW_values);

    FEValues<dim> fe_values(m_fe, m_qf_cell, uf_cell);
    FEFaceValues<dim> fe_face_values(m_fe, m_qf_face, uf_face);

    // shape function values for displacement field
    std::vector<std::vector<Tensor<1, dim>>>
      Nx_disp(m_qf_cell.size(), std::vector<Tensor<1, dim>>(m_dofs_per_cell));
    std::vector<std::vector<Tensor<2, dim>>>
      grad_Nx_disp(m_qf_cell.size(), std::vector<Tensor<2, dim>>(m_dofs_per_cell));
    std::vector<std::vector<SymmetricTensor<2, dim>>>
      symm_grad_Nx_disp(m_qf_cell.size(), std::vector<SymmetricTensor<2, dim>>(m_dofs_per_cell));

    // shape function values for phase field
    std::vector<std::vector<double>>
      Nx_phasefield(m_qf_cell.size(), std::vector<double>(m_dofs_per_cell));
    std::vector<std::vector<Tensor<1, dim>>>
      grad_Nx_phasefield(m_qf_cell.size(), std::vector<Tensor<1, dim>>(m_dofs_per_cell));

    std::vector<double> phasefield_previous_step_cell(m_qf_cell.size());

    for (const auto &cell : m_dof_handler.active_cell_iterators())
      {
	const std::vector<std::shared_ptr< PointHistory<dim>>> lqph =
	  m_quadrature_point_history.get_data(cell);
	Assert(lqph.size() == m_n_q_points, ExcInternalError());

	cell_rhs = 0.0;
	fe_values.reinit(cell);
	right_hand_side(fe_values.get_quadrature_points(),
			rhs_values,
			m_parameters.m_x_component*time_ramp,
			m_parameters.m_y_component*time_ramp,
			m_parameters.m_z_component*time_ramp);

	fe_values[m_d_fe].get_function_values(
	    solution_old, phasefield_previous_step_cell);

	for (const unsigned int q_point : fe_values.quadrature_point_indices())
	  {
	    for (const unsigned int k : fe_values.dof_indices())
	      {
		const unsigned int k_group = m_fe.system_to_base_index(k).first.first;

		if (k_group == m_u_dof)
		  {
		    Nx_disp[q_point][k] = fe_values[m_u_fe].value(k, q_point);
		    grad_Nx_disp[q_point][k] = fe_values[m_u_fe].gradient(k, q_point);
		    symm_grad_Nx_disp[q_point][k] = symmetrize(grad_Nx_disp[q_point][k]);
		  }
		else if (k_group == m_d_dof)
		  {
		    Nx_phasefield[q_point][k] = fe_values[m_d_fe].value(k, q_point);
		    grad_Nx_phasefield[q_point][k] = fe_values[m_d_fe].gradient(k, q_point);
		  }
		else
		  Assert(k_group <= m_d_dof, ExcInternalError());
	      }
	  }

	for (const unsigned int q_point : fe_values.quadrature_point_indices())
	  {
	    const double length_scale            = lqph[q_point]->get_length_scale();
	    const double gc                      = lqph[q_point]->get_critical_energy_release_rate();
	    const double eta                     = lqph[q_point]->get_viscosity();
	    const double history_strain_energy   = lqph[q_point]->get_history_max_positive_strain_energy();
	    const double current_positive_strain_energy = lqph[q_point]->get_current_positive_strain_energy();

	    double history_value = history_strain_energy;
	    if (current_positive_strain_energy > history_strain_energy)
	      history_value = current_positive_strain_energy;

	    const double phasefield_value        = lqph[q_point]->get_phase_field_value();
	    const Tensor<1, dim> phasefield_grad = lqph[q_point]->get_phase_field_gradient();

	    const std::vector<double>         &      N_phasefield = Nx_phasefield[q_point];
	    const std::vector<Tensor<1, dim>> & grad_N_phasefield = grad_Nx_phasefield[q_point];
	    const double                old_phasefield = phasefield_previous_step_cell[q_point];

	    const SymmetricTensor<2, dim> & cauchy_stress = lqph[q_point]->get_cauchy_stress();

	    const std::vector<Tensor<1,dim>> & N = Nx_disp[q_point];
	    const std::vector<SymmetricTensor<2, dim>> & symm_grad_N = symm_grad_Nx_disp[q_point];
	    const double JxW = fe_values.JxW(q_point);

	    for (const unsigned int i : fe_values.dof_indices())
	      {
		const unsigned int i_group = m_fe.system_to_base_index(i).first.first;

		if (i_group == m_u_dof)
		  {
		    cell_rhs(i) += (symm_grad_N[i] * cauchy_stress) * JxW;
		    // contributions from the body force to right-hand side
		    cell_rhs(i) -= N[i] * rhs_values[q_point] * JxW;
		  }
		else if (i_group == m_d_dof)
		  {
		    cell_rhs(i) += (    gc * length_scale * grad_N_phasefield[i] * phasefield_grad
	    	                     +  (   gc / length_scale * phasefield_value
			                  + eta / delta_time  * (phasefield_value - old_phasefield)
				          + degradation_function_derivative(phasefield_value) * history_value )
				     * N_phasefield[i]
				   ) * JxW;
		  }
		else
		  Assert(i_group <= m_d_dof, ExcInternalError());
	      }
	  }

	// if there is surface pressure, this surface pressure always applied to the
	// reference configuration
	const unsigned int face_pressure_id = 100;
	const double p0 = 0.0;

	for (const auto &face : cell->face_iterators())
	  {
	    if (face->at_boundary() && face->boundary_id() == face_pressure_id)
	      {
		fe_face_values.reinit(cell, face);

		for (const unsigned int f_q_point : fe_face_values.quadrature_point_indices())
		  {
		    const Tensor<1, dim> &N = fe_face_values.normal_vector(f_q_point);

		    const double         pressure  = p0 * time_ramp;
		    const Tensor<1, dim> traction  = pressure * N;

		    for (const unsigned int i : fe_values.dof_indices())
		      {
			const unsigned int i_group = m_fe.system_to_base_index(i).first.first;

			if (i_group == m_u_dof)
			  {
			    const unsigned int component_i = m_fe.system_to_component_index(i).first;
			    const double Ni = fe_face_values.shape_value(i, f_q_point);
			    const double JxW = fe_face_values.JxW(f_q_point);
			    cell_rhs(i) -= (Ni * traction[component_i]) * JxW;
			  }
		      }
		  }
	      }
	  }

	cell->get_dof_indices(local_dof_indices);
	for (const unsigned int i : fe_values.dof_indices())
	  system_rhs(local_dof_indices[i]) += cell_rhs(i);
      } // for (const auto &cell : m_dof_handler.active_cell_iterators())

    m_timer.leave_subsection();
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::update_history_field_step()
  {
    m_logfile << "\t\tUpdate history variable" << std::endl;

    for (const auto &cell : m_triangulation.active_cell_iterators())
      {
        std::vector<std::shared_ptr< PointHistory<dim>>> lqph =
          m_quadrature_point_history.get_data(cell);
        Assert(lqph.size() == m_n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < m_n_q_points; ++q_point)
          {
            lqph[q_point]->update_history_variable();
          }
      }
  }

  template <int dim>
  double PhaseFieldMonolithicSolve<dim>::line_search_stepsize_gradient_based(const BlockVector<double> & BFGS_p_vector,
				                                             const BlockVector<double> & solution_delta)
  {
    BlockVector<double> g_old(m_system_rhs);

    // BFGS_p_vector is the search direction
    BlockVector<double> solution_delta_trial(solution_delta);
    // take a full step size 1.0
    solution_delta_trial.add(1.0, BFGS_p_vector);

    update_qph_incremental(solution_delta_trial, m_solution, false);

    BlockVector<double> g_new(m_dofs_per_block);
    assemble_system_rhs_BFGS_parallel(m_solution, g_new);

    BlockVector<double> y_old(m_dofs_per_block);

    y_old = g_new - g_old;

    double alpha = 1.0;

    double alpha_old = 0.0;

    double delta_alpha_old = alpha - alpha_old;

    double delta_alpha_new;

    unsigned int ls_max = 10;

    for (unsigned int i = 1; i <= ls_max; ++i)
      {
	delta_alpha_new = -delta_alpha_old
	                * (g_new * BFGS_p_vector)/(y_old * BFGS_p_vector);
	alpha += delta_alpha_new;

	if (std::fabs(delta_alpha_new) < 1.0e-5)
	  break;

        if (i == ls_max)
          {
            alpha = 1.0;
            break;
          }

        g_old = g_new;

        // BFGS_p_vector is the search direction
        solution_delta_trial = solution_delta;
        solution_delta_trial.add(alpha, BFGS_p_vector);
        update_qph_incremental(solution_delta_trial, m_solution, false);
        assemble_system_rhs_BFGS_parallel(m_solution, g_new);

        y_old = g_new - g_old;

        delta_alpha_old = delta_alpha_new;
      }

    if (alpha < 1.0e-3)
      alpha = 1.0;

    return alpha;
  }

  template <int dim>
  double PhaseFieldMonolithicSolve<dim>::line_search_stepsize_strong_wolfe(const double phi_0,
				                                           const double phi_0_prime,
				                                           const BlockVector<double> & BFGS_p_vector,
				                                           const BlockVector<double> & solution_delta)
  {
    //AssertThrow(phi_0_prime < 0,
    //            ExcMessage("The derivative of phi at alpha = 0 should be negative!"));

    // Some line search parameters
    const double c1 = 0.0001;
    const double c2 = 0.9;
    const double alpha_max = 100.0;
    const unsigned int max_iter = 20;
    double alpha = 1.0;

    double phi_old = phi_0;
    double phi_prime_old = phi_0_prime;
    double alpha_old = 0.0;

    double phi, phi_prime;

    std::pair<double, double> current_phi_phi_prime;

    unsigned int i = 0;
    for (; i < max_iter; ++i)
      {
	current_phi_phi_prime = calculate_phi_and_phi_prime(alpha, BFGS_p_vector, solution_delta);
	phi = current_phi_phi_prime.first;
	phi_prime = current_phi_phi_prime.second;

	if (   ( phi > (phi_0 + c1 * alpha * phi_0_prime) )
	    || ( i > 0 && phi > phi_old ) )
	  {
	    return line_search_zoom_strong_wolfe(phi_old, phi_prime_old, alpha_old,
						 phi,     phi_prime,     alpha,
						 phi_0,   phi_0_prime,   BFGS_p_vector,
						 c1,      c2,            max_iter, solution_delta);
	  }

	if (std::fabs(phi_prime) <= c2 * std::fabs(phi_0_prime))
	  {
	    return alpha;
	  }

	if (phi_prime >= 0)
	  {
	    return line_search_zoom_strong_wolfe(phi,     phi_prime,     alpha,
						 phi_old, phi_prime_old, alpha_old,
						 phi_0,   phi_0_prime,   BFGS_p_vector,
						 c1,      c2,            max_iter, solution_delta);
	  }

	phi_old = phi;
	phi_prime_old = phi_prime;
	alpha_old = alpha;

	alpha = std::min(2.0*alpha, alpha_max);

	//AssertThrow(alpha < alpha_max,
	//	    ExcMessage("alpha is bigger than alpha_max, line search failed!"));
      }

    //AssertThrow(i < max_iter,
    //            ExcMessage("max number attempts arrived, line search failed!"));
    // Instead of terminating the program, we can just take a full step.
    if (i == max_iter)
      alpha = 1.0;

    return alpha;
  }

  template <int dim>
  double PhaseFieldMonolithicSolve<dim>::
    line_search_zoom_strong_wolfe(double phi_low, double phi_low_prime, double alpha_low,
				  double phi_high, double phi_high_prime, double alpha_high,
				  double phi_0, double phi_0_prime, const BlockVector<double> & BFGS_p_vector,
				  double c1, double c2, unsigned int max_iter, const BlockVector<double> & solution_delta)
  {
    double alpha = 0;
    std::pair<double, double> current_phi_phi_prime;
    double phi, phi_prime;

    unsigned int i = 0;
    for (; i < max_iter; ++i)
      {
	// a simple bisection is faster than cubic interpolation
	alpha = 0.5 * (alpha_low + alpha_high);
	//alpha = line_search_interpolation_cubic(alpha_low, phi_low, phi_low_prime,
	//					alpha_high, phi_high, phi_high_prime);
	current_phi_phi_prime = calculate_phi_and_phi_prime(alpha, BFGS_p_vector, solution_delta);
	phi = current_phi_phi_prime.first;
	phi_prime = current_phi_phi_prime.second;

	if (   (phi > phi_0 + c1 * alpha * phi_0_prime)
	    || (phi > phi_low) )
	  {
	    alpha_high = alpha;
	    phi_high = phi;
	    phi_high_prime = phi_prime;
	  }
	else
	  {
	    if (std::fabs(phi_prime) <= c2 * std::fabs(phi_0_prime))
	      {
		//if (alpha < 1.0e-3)
		//  alpha = 1.0e-3;
		return alpha;
	      }

	    if (phi_prime * (alpha_high - alpha_low) >= 0.0)
	      {
		alpha_high = alpha_low;
		phi_high_prime = phi_low_prime;
		phi_high = phi_low;
	      }

	    alpha_low = alpha;
	    phi_low_prime = phi_prime;
	    phi_low = phi;
	  }
      }

    if (alpha < 1.0e-3)
      alpha = 1.0;

    // avoid unused variable warnings from compiler
    (void)phi_high;
    (void)phi_high_prime;
    return alpha;
  }

  template <int dim>
  double PhaseFieldMonolithicSolve<dim>::
    line_search_interpolation_cubic(const double alpha_0, const double phi_0, const double phi_0_prime,
  			            const double alpha_1, const double phi_1, const double phi_1_prime)
  {
    const double d1 = phi_0_prime + phi_1_prime - 3.0 * (phi_0 - phi_1) / (alpha_0 - alpha_1);

    const double temp = d1 * d1 - phi_0_prime * phi_1_prime;

    if (temp < 0.0)
      return 0.5 * (alpha_0 + alpha_1);

    int sign;
    if (alpha_1 > alpha_0)
      sign = 1;
    else
      sign = -1;

    const double d2 = sign * std::sqrt(temp);

    const double alpha = alpha_1 - (alpha_1 - alpha_0)
	               * (phi_1_prime + d2 - d1) / (phi_1_prime - phi_0_prime + 2*d2);

    if (    (alpha_1 > alpha_0)
	 && (alpha > alpha_1 || alpha < alpha_0))
      return 0.5 * (alpha_0 + alpha_1);

    if (    (alpha_0 > alpha_1)
	 && (alpha > alpha_0 || alpha < alpha_1))
      return 0.5 * (alpha_0 + alpha_1);

    return alpha;
  }

  template <int dim>
  std::pair<double, double> PhaseFieldMonolithicSolve<dim>::
    calculate_phi_and_phi_prime(const double alpha,
				const BlockVector<double> & BFGS_p_vector,
				const BlockVector<double> & solution_delta)
  {
    // the first component is phi(alpha), the second component is phi_prime(alpha),
    std::pair<double, double> phi_values;

    BlockVector<double> solution_delta_trial(solution_delta);
    solution_delta_trial.add(alpha, BFGS_p_vector);

    update_qph_incremental(solution_delta_trial, m_solution, false);

    BlockVector<double> system_rhs(m_dofs_per_block);
    assemble_system_rhs_BFGS_parallel(m_solution, system_rhs);
    //m_constraints.condense(system_rhs);

    phi_values.first = calculate_energy_functional();
    phi_values.second = system_rhs * BFGS_p_vector;
    return phi_values;
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::LBFGS_B0(BlockVector<double> & LBFGS_r_vector,
						BlockVector<double> & LBFGS_q_vector)
  {
    m_timer.enter_subsection("Solve B0");

    assemble_system_B0(m_solution);

    if (m_parameters.m_type_linear_solver == "Direct")
      {
	SparseDirectUMFPACK A_direct;
	A_direct.initialize(m_tangent_matrix);
	A_direct.vmult(LBFGS_r_vector,
		       LBFGS_q_vector);
      }
    else if (m_parameters.m_type_linear_solver == "CG")
      {
/*
	SolverControl            solver_control(1e6, 1e-9);
	SolverCG<BlockVector<double>> cg(solver_control);

	PreconditionJacobi<BlockSparseMatrix<double>> preconditioner;
	preconditioner.initialize(m_tangent_matrix, 1.0);

	cg.solve(m_tangent_matrix,
		 LBFGS_r_vector,
		 LBFGS_q_vector,
		 preconditioner);
*/
	SolverControl            solver_control_uu(1e6, 1e-9);
	SolverCG<Vector<double>> cg_uu(solver_control_uu);

	PreconditionJacobi<SparseMatrix<double>> preconditioner_uu;
	preconditioner_uu.initialize(m_tangent_matrix.block(m_u_dof, m_u_dof), 1.0);
	cg_uu.solve(m_tangent_matrix.block(m_u_dof, m_u_dof),
	            LBFGS_r_vector.block(m_u_dof),
	            LBFGS_q_vector.block(m_u_dof),
	            preconditioner_uu);

	SolverControl            solver_control_dd(1e6, 1e-15);
	SolverCG<Vector<double>> cg_dd(solver_control_dd);

	PreconditionJacobi<SparseMatrix<double>> preconditioner_dd;
	preconditioner_dd.initialize(m_tangent_matrix.block(m_d_dof, m_d_dof), 1.0);
	cg_dd.solve(m_tangent_matrix.block(m_d_dof, m_d_dof),
	            LBFGS_r_vector.block(m_d_dof),
	            LBFGS_q_vector.block(m_d_dof),
	            preconditioner_dd);
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected linear solver not implemented!"));
      }

    m_timer.leave_subsection();
  }

  template <int dim>
  std::vector<double>
    PhaseFieldMonolithicSolve<dim>::solve_linear_system(BlockVector<double> & newton_update)
  {
    m_timer.enter_subsection("Solve coupled linear system");

    if (m_parameters.m_output_iteration_history)
      m_logfile << " SLV " << std::flush;

    std::vector<double> linear_solver_parameters(3);
/*
    {
      SolverControl            solver_control(1e6, 1e-9);
      SolverCG<Vector<double>> cg(solver_control);
      cg.connect_condition_number_slot(
	  [&] (double condition_number)
	  {
	    linear_solver_parameters[0] = condition_number;
	    //m_logfile << "   Estimated condition number = "<< condition_number << std::endl;
	  },
	  false);

      PreconditionSSOR<SparseMatrix<double>> preconditioner;
      preconditioner.initialize(m_system_matrix_displacement, 1.2);

      cg.solve(m_system_matrix_displacement,
	       newton_update,
	       m_system_rhs_displacement,
	       preconditioner);

      //m_logfile << "   " << solver_control.last_step()
		  //<< " CG iterations needed to obtain convergence." << std::endl;
      linear_solver_parameters[1] = solver_control.last_step();
      linear_solver_parameters[2] = solver_control.last_value();
    }
*/
    SparseDirectUMFPACK A_direct;
    A_direct.initialize(m_tangent_matrix);
    A_direct.vmult(newton_update,
		   m_system_rhs);

    m_constraints.distribute(newton_update);

    m_timer.leave_subsection();
    return linear_solver_parameters;
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::print_conv_header_newton()
  {
    static const unsigned int l_width = 135;
    m_logfile << '\t' << '\t';
    for (unsigned int i = 0; i < l_width; ++i)
      m_logfile << '_';
    m_logfile << std::endl;

    m_logfile << "                  SOLVER STEP (Newton)      "
              << " |        Cond No.   Lin_Iter   Lin_Res    Res_Norm  "
              << " Res_u      Res_d      Inc_Norm  "
              << " Inc_u      Inc_d" << std::endl;

    m_logfile << '\t' << '\t';
    for (unsigned int i = 0; i < l_width; ++i)
      m_logfile << '_';
    m_logfile << std::endl;
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::print_conv_header_BFGS()
  {
    static const unsigned int l_width = 125;
    m_logfile << '\t' << '\t';
    for (unsigned int i = 0; i < l_width; ++i)
      m_logfile << '_';
    m_logfile << std::endl;

    m_logfile << "                  SOLVER STEP (BFGS)   "
              << " |    Line Search alpha    Energy    Res_Norm    "
              << " Res_u      Res_d    Inc_Norm    "
              << " Inc_u      Inc_d" << std::endl;

    m_logfile << '\t' << '\t';
    for (unsigned int i = 0; i < l_width; ++i)
      m_logfile << '_';
    m_logfile << std::endl;
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::print_conv_header_LBFGS()
  {
    static const unsigned int l_width = 120;
    m_logfile << '\t' << '\t';
    for (unsigned int i = 0; i < l_width; ++i)
      m_logfile << '_';
    m_logfile << std::endl;

    m_logfile << "                  SOLVER STEP (LBFGS)  "
              << " |  LS-alpha     Energy      Res_Norm    "
              << " Res_u      Res_d    Inc_Norm   "
              << " Inc_u      Inc_d" << std::endl;

    m_logfile << '\t' << '\t';
    for (unsigned int i = 0; i < l_width; ++i)
      m_logfile << '_';
    m_logfile << std::endl;
  }

  template <int dim>
  bool PhaseFieldMonolithicSolve<dim>::
    solve_nonlinear_timestep_newton(BlockVector<double> & solution_delta)
  {
    BlockVector<double> newton_update(m_dofs_per_block);

    m_error_residual.reset();
    m_error_residual_0.reset();
    m_error_residual_norm.reset();
    m_error_update.reset();
    m_error_update_0.reset();
    m_error_update_norm.reset();

    if (m_parameters.m_output_iteration_history)
      print_conv_header_newton();

    unsigned int newton_iteration = 0;
    for (; newton_iteration < m_parameters.m_max_iterations_NR; ++newton_iteration)
      {
        if (m_parameters.m_output_iteration_history)
          m_logfile << '\t' << '\t' << std::setw(2) << newton_iteration << ' '
                    << std::flush;

        make_constraints(newton_iteration);
        assemble_system_newton(m_solution);

        get_error_residual(m_error_residual);
        if (newton_iteration == 0)
          m_error_residual_0 = m_error_residual;

        m_error_residual_norm = m_error_residual;
        m_error_residual_norm.normalize(m_error_residual_0);

        if (newton_iteration > 0 && m_error_update_norm.m_u <= m_parameters.m_tol_u_incr
                                 && m_error_residual_norm.m_u <= m_parameters.m_tol_u_residual
				 && m_error_update_norm.m_d <= m_parameters.m_tol_d_incr
				 && m_error_residual_norm.m_d <= m_parameters.m_tol_d_residual)
          {
            if (m_parameters.m_output_iteration_history)
              {
		m_logfile << " CONVERGED!";
		m_logfile << " | " << std::fixed << std::setprecision(3) << std::setw(7)
			  << std::scientific
		      << "  " << "  ----   "
		      << "  " << "  ----   "
		      << "  " << "  ----   "
		      << "  " << m_error_residual_norm.m_norm
		      << "  " << m_error_residual_norm.m_u
		      << "  " << m_error_residual_norm.m_d
		      << "  " << m_error_update_norm.m_norm
		      << "  " << m_error_update_norm.m_u
		      << "  " << m_error_update_norm.m_d
		      << "  " << std::endl;

		m_logfile << '\t' << '\t';
		for (unsigned int i = 0; i < 135; ++i)
		  m_logfile << '_';
		m_logfile << std::endl;
              }

            m_logfile << "\t\tConvergence is reached after "
        	      << newton_iteration << " Newton iterations."<< std::endl;

            m_logfile << "\t\tResidual information of convergence:" << std::endl;

            m_logfile << "\t\t\tRelative residual of disp. equation: "
        	      << m_error_residual_norm.m_u << std::endl;

            m_logfile << "\t\t\tAbsolute residual of disp. equation: "
        	      << m_error_residual_norm.m_u * m_error_residual_0.m_u << std::endl;

            m_logfile << "\t\t\tRelative residual of phasefield equation: "
        	      << m_error_residual_norm.m_d << std::endl;

            m_logfile << "\t\t\tAbsolute residual of phasefield equation: "
        	      << m_error_residual_norm.m_d * m_error_residual_0.m_d << std::endl;

            m_logfile << "\t\t\tRelative increment of disp.: "
        	      << m_error_update_norm.m_u << std::endl;

            m_logfile << "\t\t\tAbsolute increment of disp.: "
        	      << m_error_update_norm.m_u * m_error_update_0.m_u << std::endl;

            m_logfile << "\t\t\tRelative increment of phasefield: "
        	      << m_error_update_norm.m_d << std::endl;

            m_logfile << "\t\t\tAbsolute increment of phasefield: "
        	      << m_error_update_norm.m_d * m_error_update_0.m_d << std::endl;

            //break;
            return true;
          }

        std::vector<double> linear_solver_parameters(3);

        linear_solver_parameters = solve_linear_system(newton_update);

        get_error_update(newton_update, m_error_update);
        if (newton_iteration == 0)
          m_error_update_0 = m_error_update;

        m_error_update_norm = m_error_update;
        m_error_update_norm.normalize(m_error_update_0);

        solution_delta += newton_update;
        update_qph_incremental(solution_delta, m_solution, true);

        if (m_parameters.m_output_iteration_history)
          {
	    m_logfile << " | " << std::fixed << std::setprecision(3) << std::setw(7)
		      << std::scientific
		      << "  " << linear_solver_parameters[0]
		      << "  " << linear_solver_parameters[1]
		      << "  " << linear_solver_parameters[2]
		      << "  " << m_error_residual_norm.m_norm
		      << "  " << m_error_residual_norm.m_u
		      << "  " << m_error_residual_norm.m_d
		      << "  " << m_error_update_norm.m_norm
		      << "  " << m_error_update_norm.m_u
		      << "  " << m_error_update_norm.m_d
		      << "  " << std::endl;
          }
      }

    //AssertThrow(newton_iteration < m_parameters.m_max_iterations_NR,
    //            ExcMessage("No convergence in Newton-Raphson nonlinear solver!"));
    return false;
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::
  solve_nonlinear_timestep_BFGS(BlockVector<double> & solution_delta)
  {
    AssertThrow(false,
                ExcMessage("BFGS requires too much memory. Please use L-BFGS!"));

    BlockVector<double> BFGS_update(m_dofs_per_block);

    m_error_residual.reset();
    m_error_residual_0.reset();
    m_error_residual_norm.reset();
    m_error_update.reset();
    m_error_update_0.reset();
    m_error_update_norm.reset();

    print_conv_header_BFGS();

    unsigned int BFGS_iteration = 0;

    // Initial guess B_0, which is a full matrix and takes a lot of memory
    FullMatrix<double> BFGS_matrix = IdentityMatrix(m_dof_handler.n_dofs());
    Vector<double> BFGS_r_vector(m_dof_handler.n_dofs());
    Vector<double> BFGS_p_vector(m_dof_handler.n_dofs());
    Vector<double> BFGS_y_vector(m_dof_handler.n_dofs());
    Vector<double> BFGS_temp_vector(m_dof_handler.n_dofs());

    double line_search_parameter, rho;

    // Most likely, we will not be able to create a second full matrix since
    // we will run out of memory on a laptop workstation
    FullMatrix<double> temp_matrix_1(m_dof_handler.n_dofs());
    FullMatrix<double> temp_matrix_2(m_dof_handler.n_dofs());

    for (; BFGS_iteration < m_parameters.m_max_iterations_BFGS; ++BFGS_iteration)
      {
        m_logfile << '\t' << '\t' << std::setw(2) << BFGS_iteration << ' '
                  << std::flush;

        make_constraints(BFGS_iteration);

        // At the first step, we simply distribute the inhomogeneous part of
        // the constraints
        if (BFGS_iteration == 0)
          {
            m_constraints.distribute(BFGS_update);
            solution_delta += BFGS_update;
            m_logfile << " --- " << std::flush;
            m_logfile << " --- " << std::flush;
            update_qph_incremental(solution_delta, m_solution, false);
            m_logfile << " ---  |" << std::flush;
            m_logfile << std::endl;
            continue;
          }
        else if (BFGS_iteration == 1)
          {
	    // Calculate the residual vector r. NOTICE that in the context of
	    // BFGS, this r is the gradient of the energy functional (objective function),
	    // NOT the negative gradient of the energy functional
	    assemble_system_rhs_BFGS(m_solution, m_system_rhs);

	    // We cannot simply zero out the dofs that are constrained, since we might
	    // have hanging node constraints. In this case, we need to modify the RHS
	    // as C^T * b, which C contains entries of 0.5 (x_3 = 0.5*x_1 + 0.5*x_2)
	    //for (unsigned int i = 0; i < m_dof_handler.n_dofs(); ++i)
	      //if (m_constraints.is_constrained(i))
		//m_system_rhs(i) = 0.0;

	    // if m_constraints has inhomogeneity, we cannot call m_constraints.condense(m_system_rhs),
	    // since the m_system_matrix needs to be provided to modify the RHS properly. However, this
	    // error will not be detected in the release mode and only will be detected on the debug mode
	    m_constraints.condense(m_system_rhs);
          }

	m_logfile << " --- " << std::flush;
	m_logfile << " --- " << std::flush;
	m_logfile << " --- " << std::flush;

        get_error_residual(m_error_residual);
        if (BFGS_iteration == 1)
          m_error_residual_0 = m_error_residual;

        m_error_residual_norm = m_error_residual;
        m_error_residual_norm.normalize(m_error_residual_0);

        if (BFGS_iteration > 1 && m_error_update_norm.m_u <= m_parameters.m_tol_u_incr
                               && m_error_residual_norm.m_u <= m_parameters.m_tol_u_residual
			       && m_error_update_norm.m_d <= m_parameters.m_tol_d_incr
			       && m_error_residual_norm.m_d <= m_parameters.m_tol_d_residual)
          {
            m_logfile << " CONVERGED!";
            m_logfile << "| " << std::fixed << std::setprecision(3) << std::setw(7)
                      << std::scientific
    		  << "  " << "  ----   "
    		  << "  " << "  ----   "
    		  << "  " << "  ----   "
    		  << "  " << m_error_residual_norm.m_norm
    		  << "  " << m_error_residual_norm.m_u
    		  << "  " << m_error_residual_norm.m_d
                  << "  " << m_error_update_norm.m_norm
                  << "  " << m_error_update_norm.m_u
                  << "  " << m_error_update_norm.m_d
		  << "  " << std::endl;

            m_logfile << '\t' << '\t';
            for (unsigned int i = 0; i < 135; ++i)
              m_logfile << '_';
            m_logfile << std::endl;

            break;
          }

        // BFGS algorithm
        BFGS_r_vector = m_system_rhs;
        BFGS_matrix.vmult(BFGS_p_vector, BFGS_r_vector);
        BFGS_p_vector *= -1.0;
        m_constraints.distribute(BFGS_p_vector);

        // We need a line search algorithm to decide line_search_parameter
        const double phi_0 = calculate_energy_functional();
        const double phi_0_prime = BFGS_r_vector * BFGS_p_vector;

        BlockVector<double> BFGS_p_vector_block(m_dofs_per_block);
        BFGS_p_vector_block = BFGS_p_vector;
        line_search_parameter = line_search_stepsize_strong_wolfe(phi_0,
						                  phi_0_prime,
								  BFGS_p_vector_block,
						                  solution_delta);

        BFGS_p_vector *= line_search_parameter;
        BFGS_update = BFGS_p_vector;

        get_error_update(BFGS_update, m_error_update);
        if (BFGS_iteration == 1)
          m_error_update_0 = m_error_update;

        m_error_update_norm = m_error_update;
        m_error_update_norm.normalize(m_error_update_0);

        solution_delta += BFGS_update;
        update_qph_incremental(solution_delta, m_solution, false);

        BFGS_y_vector = m_system_rhs;
        BFGS_y_vector *= -1.0;
        assemble_system_rhs_BFGS(m_solution, m_system_rhs);
        m_constraints.condense(m_system_rhs);
        BFGS_temp_vector = m_system_rhs;
        BFGS_y_vector += BFGS_temp_vector;

        // rho should be positive with the proper line search
        rho = BFGS_y_vector * BFGS_p_vector;
        rho = 1.0/rho;

        if (rho < 0)
          m_logfile << "Rho is negative!" << std::endl;

        // In the first step, we scale the identity matrix as
        // the BFGS matrix
        if (BFGS_iteration == 1)
          {
            double scale_parameter = (BFGS_y_vector * BFGS_p_vector) / (BFGS_y_vector.norm_sqr());
            BFGS_matrix *= scale_parameter;
          }

        temp_matrix_1.outer_product(BFGS_p_vector, BFGS_y_vector);
        temp_matrix_2 = IdentityMatrix(m_dof_handler.n_dofs());
        temp_matrix_2.add(-rho, temp_matrix_1);

        temp_matrix_2.mmult(temp_matrix_1, BFGS_matrix);
        temp_matrix_1.mTmult(BFGS_matrix, temp_matrix_2);

        temp_matrix_1.outer_product(BFGS_p_vector, BFGS_p_vector);

        BFGS_matrix.add(rho, temp_matrix_1);

        const double energy_functional = calculate_energy_functional();

        m_logfile << " | " << std::fixed << std::setprecision(3) << std::setw(7)
                  << std::scientific
		  << "  " << line_search_parameter
		  << "       " << energy_functional
		  << "  " << m_error_residual_norm.m_norm
		  << "  " << m_error_residual_norm.m_u
		  << "  " << m_error_residual_norm.m_d
                  << "  " << m_error_update_norm.m_norm
                  << "  " << m_error_update_norm.m_u
                  << "  " << m_error_update_norm.m_d
		  << "  " << std::endl;
      }

    AssertThrow(BFGS_iteration < m_parameters.m_max_iterations_BFGS,
                ExcMessage("No convergence in BFGS nonlinear solver!"));
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::
  solve_nonlinear_timestep_LBFGS(BlockVector<double> & solution_delta,
				 BlockVector<double> & LBFGS_update_refine)
  {
    BlockVector<double> LBFGS_update(m_dofs_per_block);

    LBFGS_update = 0.0;

    m_error_residual.reset();
    m_error_residual_0.reset();
    m_error_residual_norm.reset();
    m_error_update.reset();
    m_error_update_0.reset();
    m_error_update_norm.reset();

    if (m_parameters.m_output_iteration_history)
      print_conv_header_LBFGS();

    unsigned int LBFGS_iteration = 0;

    BlockVector<double> LBFGS_r_vector(m_dofs_per_block);
    BlockVector<double> LBFGS_y_vector(m_dofs_per_block);
    BlockVector<double> LBFGS_q_vector(m_dofs_per_block);
    BlockVector<double> LBFGS_s_vector(m_dofs_per_block);
    std::list<std::pair< std::pair<BlockVector<double>,
                                   BlockVector<double>>,
                         double>> LBFGS_vector_list;

    const unsigned int LBFGS_m = m_parameters.m_LBFGS_m;
    std::list<double> LBFGS_alpha_list;

    double line_search_parameter = 0.0;
    double LBFGS_beta = 0.0;
    double rho = 0.0;

    for (; LBFGS_iteration < m_parameters.m_max_iterations_BFGS; ++LBFGS_iteration)
      {
	if (m_parameters.m_output_iteration_history)
	  m_logfile << '\t' << '\t' << std::setw(2) << LBFGS_iteration << ' '
                    << std::flush;

        make_constraints(LBFGS_iteration);

        // At the first step, we simply distribute the inhomogeneous part of
        // the constraints
        if (LBFGS_iteration == 0)
          {
            // use the solution from the previous solve on the
            // refined mesh as initial guess
            LBFGS_update = LBFGS_update_refine;

            m_constraints.distribute(LBFGS_update);
            solution_delta += LBFGS_update;
            if (m_parameters.m_output_iteration_history)
              {
                m_logfile << " --- " << std::flush;
                m_logfile << " --- " << std::flush;
              }
            update_qph_incremental(solution_delta, m_solution, false);
            if (m_parameters.m_output_iteration_history)
              {
                m_logfile << " ---  |" << std::flush;
                m_logfile << std::endl;
              }
            continue;
          }
        else if (LBFGS_iteration == 1)
          {
	    // Calculate the residual vector r. NOTICE that in the context of
	    // BFGS, this r is the gradient of the energy functional (objective function),
	    // NOT the negative gradient of the energy functional
	    assemble_system_rhs_BFGS_parallel(m_solution, m_system_rhs);

	    // We cannot simply zero out the dofs that are constrained, since we might
	    // have hanging node constraints. In this case, we need to modify the RHS
	    // as C^T * b, which C contains entries of 0.5 (x_3 = 0.5*x_1 + 0.5*x_2)
	    //for (unsigned int i = 0; i < m_dof_handler.n_dofs(); ++i)
	      //if (m_constraints.is_constrained(i))
		//m_system_rhs(i) = 0.0;

	    // if m_constraints has inhomogeneity, we cannot call m_constraints.condense(m_system_rhs),
	    // since the m_system_matrix needs to be provided to modify the RHS properly. However, this
	    // error will not be detected in the release mode and only will be detected on the debug mode
	    // if we use assemble_system_rhs_BFGS_parallel, then condense() is not necessary
	    //m_constraints.condense(m_system_rhs);
          }
	if (m_parameters.m_output_iteration_history)
	  {
            m_logfile << " --- " << std::flush;
            m_logfile << " --- " << std::flush;
            m_logfile << " --- " << std::flush;
	  }

        get_error_residual(m_error_residual);
        if (LBFGS_iteration == 1)
          m_error_residual_0 = m_error_residual;

        m_error_residual_norm = m_error_residual;
        // For three-point bending problem and 3D problem, we use absolute residual
        // for convergence test
        if (m_parameters.m_relative_residual)
          m_error_residual_norm.normalize(m_error_residual_0);

        if (LBFGS_iteration > 1 && m_error_update_norm.m_u <= m_parameters.m_tol_u_incr
                                && m_error_residual_norm.m_u <= m_parameters.m_tol_u_residual
			        && m_error_update_norm.m_d <= m_parameters.m_tol_d_incr
			        && m_error_residual_norm.m_d <= m_parameters.m_tol_d_residual
				)
          {
            if (m_parameters.m_output_iteration_history)
              {
		m_logfile << " | ";
		m_logfile << " CONVERGED! " << std::fixed << std::setprecision(3) << std::setw(7)
			  << std::scientific
		      << "    ----    "
		      << "  " << m_error_residual_norm.m_norm
		      << "  " << m_error_residual_norm.m_u
		      << "  " << m_error_residual_norm.m_d
		      << "  " << m_error_update_norm.m_norm
		      << "  " << m_error_update_norm.m_u
		      << "  " << m_error_update_norm.m_d
		      << "  " << std::endl;

		m_logfile << '\t' << '\t';
		for (unsigned int i = 0; i < 120; ++i)
		  m_logfile << '_';
		m_logfile << std::endl;
              }

            m_logfile << "\t\tConvergence is reached after "
        	      << LBFGS_iteration << " L-BFGS iterations."<< std::endl;

            m_logfile << "\t\tResidual information of convergence:" << std::endl;

            if (m_parameters.m_relative_residual)
              {
		m_logfile << "\t\t\tRelative residual of disp. equation: "
			  << m_error_residual_norm.m_u << std::endl;

		m_logfile << "\t\t\tAbsolute residual of disp. equation: "
			  << m_error_residual_norm.m_u * m_error_residual_0.m_u << std::endl;

		m_logfile << "\t\t\tRelative residual of phasefield equation: "
			  << m_error_residual_norm.m_d << std::endl;

		m_logfile << "\t\t\tAbsolute residual of phasefield equation: "
			  << m_error_residual_norm.m_d * m_error_residual_0.m_d << std::endl;

		m_logfile << "\t\t\tRelative increment of disp.: "
			  << m_error_update_norm.m_u << std::endl;

		m_logfile << "\t\t\tAbsolute increment of disp.: "
			  << m_error_update_norm.m_u * m_error_update_0.m_u << std::endl;

		m_logfile << "\t\t\tRelative increment of phasefield: "
			  << m_error_update_norm.m_d << std::endl;

		m_logfile << "\t\t\tAbsolute increment of phasefield: "
			  << m_error_update_norm.m_d * m_error_update_0.m_d << std::endl;
              }
            else
              {
		m_logfile << "\t\t\tAbsolute residual of disp. equation: "
			  << m_error_residual_norm.m_u << std::endl;

		m_logfile << "\t\t\tAbsolute residual of phasefield equation: "
			  << m_error_residual_norm.m_d << std::endl;

		m_logfile << "\t\t\tAbsolute increment of disp.: "
			  << m_error_update_norm.m_u << std::endl;

		m_logfile << "\t\t\tAbsolute increment of phasefield: "
			  << m_error_update_norm.m_d << std::endl;
              }

            break;
          }

        // LBFGS algorithm
        LBFGS_q_vector = m_system_rhs;

        LBFGS_alpha_list.clear();
        for (auto itr = LBFGS_vector_list.begin(); itr != LBFGS_vector_list.end(); ++itr)
          {
            LBFGS_s_vector = (itr->first).first;
            LBFGS_y_vector = (itr->first).second;
            rho = itr->second;

            const double alpha = rho * (LBFGS_s_vector * LBFGS_q_vector);
            LBFGS_alpha_list.push_back(alpha);

            LBFGS_q_vector.add(-alpha, LBFGS_y_vector);
          }
/*
        double scale_gamma = 0.0;
        if (LBFGS_iteration == 1)
          {
            scale_gamma = 1.0;
          }
        else
          {
            LBFGS_s_vector = LBFGS_vector_list.front().first.first;
            LBFGS_y_vector = LBFGS_vector_list.front().first.second;
            scale_gamma = (LBFGS_s_vector * LBFGS_y_vector)/(LBFGS_y_vector * LBFGS_y_vector);
          }

        LBFGS_q_vector *= scale_gamma;
        LBFGS_r_vector = LBFGS_q_vector;
*/
        LBFGS_B0(LBFGS_r_vector,
		 LBFGS_q_vector);

        for (auto itr = LBFGS_vector_list.rbegin(); itr != LBFGS_vector_list.rend(); ++itr)
          {
            LBFGS_s_vector = (itr->first).first;
            LBFGS_y_vector = (itr->first).second;
            rho = itr->second;

            LBFGS_beta = rho * (LBFGS_y_vector * LBFGS_r_vector);

            const double alpha = LBFGS_alpha_list.back();
            LBFGS_alpha_list.pop_back();

            LBFGS_r_vector.add(alpha - LBFGS_beta, LBFGS_s_vector);
          }

        LBFGS_r_vector *= -1.0; // this is the p_vector (search direction)

        m_constraints.distribute(LBFGS_r_vector);

        // We need a line search algorithm to decide line_search_parameter
        if(m_parameters.m_type_line_search == "StrongWolfe")
          {
	    const double phi_0 = calculate_energy_functional();
	    const double phi_0_prime = m_system_rhs * LBFGS_r_vector;

	    line_search_parameter = line_search_stepsize_strong_wolfe(phi_0,
								      phi_0_prime,
								      LBFGS_r_vector,
								      solution_delta);
          }
        else if(m_parameters.m_type_line_search == "GradientBased")
          {
	    // LBFGS_r_vector is the search direction
	    line_search_parameter = line_search_stepsize_gradient_based(LBFGS_r_vector,
									solution_delta);
          }
        else
          {
            Assert(false, ExcMessage("An unknown line search method is called!"));
          }

        LBFGS_r_vector *= line_search_parameter;
        LBFGS_update = LBFGS_r_vector;

        get_error_update(LBFGS_update, m_error_update);
        if (LBFGS_iteration == 1)
          m_error_update_0 = m_error_update;

        m_error_update_norm = m_error_update;
        // For three-point bending problem and the sphere inclusion problem,
        // we use absolute residual for convergence test
        if (m_parameters.m_relative_residual)
          m_error_update_norm.normalize(m_error_update_0);

        solution_delta += LBFGS_update;
        update_qph_incremental(solution_delta, m_solution, false);

        LBFGS_y_vector = m_system_rhs;
        LBFGS_y_vector *= -1.0;
        assemble_system_rhs_BFGS_parallel(m_solution, m_system_rhs);
        // if we use assemble_system_rhs_BFGS_parallel, then condense() is not necessary
        //m_constraints.condense(m_system_rhs);
        LBFGS_y_vector += m_system_rhs;

        LBFGS_s_vector = LBFGS_update;

        const double g_norm = m_system_rhs.l2_norm();

        const double yxs = LBFGS_y_vector * LBFGS_s_vector;

        const double sxs = LBFGS_s_vector * LBFGS_s_vector;

        if (yxs/sxs >= 1.0e-6 * g_norm)
          {
	    if (LBFGS_iteration > LBFGS_m)
	      LBFGS_vector_list.pop_back();

	    rho = 1.0 / yxs;

	    LBFGS_vector_list.push_front(std::make_pair(std::make_pair(LBFGS_s_vector,
								       LBFGS_y_vector),
							rho));
          }

        if (m_parameters.m_output_iteration_history)
          {
	    const double energy_functional = calculate_energy_functional();

	    m_logfile << " | " << std::fixed << std::setprecision(3) << std::setw(1)
		      << std::scientific
		      << "" << line_search_parameter
		      << std::fixed << std::setprecision(6) << std::setw(1)
					<< std::scientific
		      << "  " << energy_functional
		      << std::fixed << std::setprecision(3) << std::setw(1)
					<< std::scientific
		      << "  " << m_error_residual_norm.m_norm
		      << "  " << m_error_residual_norm.m_u
		      << "  " << m_error_residual_norm.m_d
		      << "  " << m_error_update_norm.m_norm
		      << "  " << m_error_update_norm.m_u
		      << "  " << m_error_update_norm.m_d
		      << "  " << std::endl;
          }
      }

    AssertThrow(LBFGS_iteration < m_parameters.m_max_iterations_BFGS,
                ExcMessage("No convergence in L-BFGS nonlinear solver!"));
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::output_results() const
  {
    m_timer.enter_subsection("Output results");

    DataOut<dim> data_out;

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);

    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

    std::vector<std::string> solution_name(dim, "displacement");
    solution_name.emplace_back("phasefield");

    data_out.attach_dof_handler(m_dof_handler);
    data_out.add_data_vector(m_solution,
                             solution_name,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);


    Vector<double> cell_material_id(m_triangulation.n_active_cells());
    // output material ID for each cell
    for (const auto &cell : m_triangulation.active_cell_iterators())
      {
	cell_material_id(cell->active_cell_index()) = cell->material_id();
      }
    data_out.add_data_vector(cell_material_id, "materialID");

    // Stress L2 projection
    DoFHandler<dim> stresses_dof_handler_L2(m_triangulation);
    FE_Q<dim>     stresses_fe_L2(m_parameters.m_poly_degree); //FE_Q element is continuous
    stresses_dof_handler_L2.distribute_dofs(stresses_fe_L2);
    AffineConstraints<double> constraints;
    constraints.clear();
    DoFTools::make_hanging_node_constraints(stresses_dof_handler_L2, constraints);
    constraints.close();
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
	  data_component_interpretation_stress(1,
					       DataComponentInterpretation::component_is_scalar);

    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = i; j < dim; ++j)
	{
	  Vector<double> stress_field_L2;
	  stress_field_L2.reinit(stresses_dof_handler_L2.n_dofs());

	  MappingQ<dim> mapping(m_parameters.m_poly_degree + 1);
	  VectorTools::project(mapping,
			       stresses_dof_handler_L2,
			       constraints,
			       m_qf_cell,
			       [&] (const typename DoFHandler<dim>::active_cell_iterator & cell,
				    const unsigned int q) -> double
			       {
				 return m_quadrature_point_history.get_data(cell)[q]->get_cauchy_stress()[i][j];
			       },
			       stress_field_L2);

	  std::string stress_name = "Cauchy_stress_" + std::to_string(i+1) + std::to_string(j+1)
				  + "_L2";

	  data_out.add_data_vector(stresses_dof_handler_L2,
				   stress_field_L2,
				   stress_name,
				   data_component_interpretation_stress);
	}

    data_out.build_patches(m_parameters.m_poly_degree);

    std::ofstream output("Solution-" + std::to_string(dim) + "d-" +
			 Utilities::int_to_string(m_time.get_timestep(),4) + ".vtu");

    data_out.write_vtu(output);
    m_timer.leave_subsection();
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::calculate_reaction_force(unsigned int face_ID)
  {
    m_timer.enter_subsection("Calculate reaction force");

    BlockVector<double>       system_rhs;
    system_rhs.reinit(m_dofs_per_block);

    Vector<double> cell_rhs(m_dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(m_dofs_per_cell);

    const double time_ramp = (m_time.current() / m_time.end());
    std::vector<Tensor<1, dim>> rhs_values(m_n_q_points);
    const UpdateFlags uf_cell(update_values | update_gradients |
			      update_quadrature_points | update_JxW_values);
    const UpdateFlags uf_face(update_values | update_normal_vectors |
                              update_JxW_values);

    FEValues<dim> fe_values(m_fe, m_qf_cell, uf_cell);
    FEFaceValues<dim> fe_face_values(m_fe, m_qf_face, uf_face);

    // shape function values for displacement field
    std::vector<std::vector<Tensor<1, dim>>>
      Nx(m_qf_cell.size(), std::vector<Tensor<1, dim>>(m_dofs_per_cell));
    std::vector<std::vector<Tensor<2, dim>>>
      grad_Nx(m_qf_cell.size(), std::vector<Tensor<2, dim>>(m_dofs_per_cell));
    std::vector<std::vector<SymmetricTensor<2, dim>>>
      symm_grad_Nx(m_qf_cell.size(), std::vector<SymmetricTensor<2, dim>>(m_dofs_per_cell));

    for (const auto &cell : m_dof_handler.active_cell_iterators())
      {
	// if calculate_reaction_force() is defined as const, then
	// we also need to put a const in std::shared_ptr,
	// that is, std::shared_ptr<const PointHistory<dim>>
	const std::vector<std::shared_ptr< PointHistory<dim>>> lqph =
	  m_quadrature_point_history.get_data(cell);
	Assert(lqph.size() == m_n_q_points, ExcInternalError());
        cell_rhs = 0.0;
        fe_values.reinit(cell);
        right_hand_side(fe_values.get_quadrature_points(),
    		        rhs_values,
    		        m_parameters.m_x_component*time_ramp,
    		        m_parameters.m_y_component*time_ramp,
    		        m_parameters.m_z_component*time_ramp);

        for (const unsigned int q_point : fe_values.quadrature_point_indices())
          {
            for (const unsigned int k : fe_values.dof_indices())
              {
                const unsigned int k_group = m_fe.system_to_base_index(k).first.first;

                if (k_group == m_u_dof)
                  {
    		    Nx[q_point][k] = fe_values[m_u_fe].value(k, q_point);
    		    grad_Nx[q_point][k] = fe_values[m_u_fe].gradient(k, q_point);
    		    symm_grad_Nx[q_point][k] = symmetrize(grad_Nx[q_point][k]);
                  }
              }
          }

        for (const unsigned int q_point : fe_values.quadrature_point_indices())
          {
            const SymmetricTensor<2, dim> & cauchy_stress = lqph[q_point]->get_cauchy_stress();

            const std::vector<Tensor<1,dim>> & N = Nx[q_point];
            const std::vector<SymmetricTensor<2, dim>> & symm_grad_N = symm_grad_Nx[q_point];
            const double JxW = fe_values.JxW(q_point);

            for (const unsigned int i : fe_values.dof_indices())
              {
                const unsigned int i_group = m_fe.system_to_base_index(i).first.first;

                if (i_group == m_u_dof)
                  {
                    cell_rhs(i) -= (symm_grad_N[i] * cauchy_stress) * JxW;
    		    // contributions from the body force to right-hand side
    		    cell_rhs(i) += N[i] * rhs_values[q_point] * JxW;
                  }
              }
          }

        // if there is surface pressure, this surface pressure always applied to the
        // reference configuration
        const unsigned int face_pressure_id = 100;
        const double p0 = 0.0;

        for (const auto &face : cell->face_iterators())
          {
	    if (face->at_boundary() && face->boundary_id() == face_pressure_id)
	      {
		fe_face_values.reinit(cell, face);

		for (const unsigned int f_q_point : fe_face_values.quadrature_point_indices())
		  {
		    const Tensor<1, dim> &N = fe_face_values.normal_vector(f_q_point);

		    const double         pressure  = p0 * time_ramp;
		    const Tensor<1, dim> traction  = pressure * N;

		    for (const unsigned int i : fe_values.dof_indices())
		      {
			const unsigned int i_group = m_fe.system_to_base_index(i).first.first;

			if (i_group == m_u_dof)
			  {
			    const unsigned int component_i = m_fe.system_to_component_index(i).first;
			    const double Ni = fe_face_values.shape_value(i, f_q_point);
			    const double JxW = fe_face_values.JxW(f_q_point);
			    cell_rhs(i) += (Ni * traction[component_i]) * JxW;
			  }
		      }
		  }
	      }
          }

        cell->get_dof_indices(local_dof_indices);
        for (const unsigned int i : fe_values.dof_indices())
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
      } // for (const auto &cell : m_dof_handler.active_cell_iterators())

    // The difference between the above assembled system_rhs and m_system_rhs
    // is that m_system_rhs is condensed by the m_constraints, which zero out
    // the rhs values associated with the constrained DOFs and modify the rhs
    // values associated with the unconstrained DOFs.

    std::vector< types::global_dof_index > mapping;
    std::set<types::boundary_id> boundary_ids;
    boundary_ids.insert(face_ID);
    DoFTools::map_dof_to_boundary_indices(m_dof_handler,
					  boundary_ids,
					  mapping);

    std::vector<double> reaction_force(dim, 0.0);

    for (unsigned int i = 0; i < m_dofs_per_block[m_u_dof]; ++i)
      {
	if (mapping[i] != numbers::invalid_dof_index)
	  {
	    reaction_force[i % dim] += system_rhs.block(m_u_dof)(i);
	  }
      }

    for (unsigned int i = 0; i < dim; i++)
      m_logfile << "\t\tReaction force in direction " << i << " on boundary ID " << face_ID
                << " = "
		<< std::fixed << std::setprecision(3) << std::setw(1)
                << std::scientific
		<< reaction_force[i] << std::endl;

    std::pair<double, std::vector<double>> time_force;
    time_force.first = m_time.current();
    time_force.second = reaction_force;
    m_history_reaction_force.push_back(time_force);

    m_timer.leave_subsection();
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::write_history_data()
  {
    m_logfile << "\t\tWrite history data ... \n"<<std::endl;

    std::ofstream myfile_reaction_force ("Reaction_force.hist");
    if (myfile_reaction_force.is_open())
    {
      myfile_reaction_force << 0.0 << "\t";
      if (dim == 2)
	myfile_reaction_force << 0.0 << "\t"
	       << 0.0 << std::endl;
      if (dim == 3)
	myfile_reaction_force << 0.0 << "\t"
	       << 0.0 << "\t"
	       << 0.0 << std::endl;

      for (auto const time_force : m_history_reaction_force)
	{
	  myfile_reaction_force << time_force.first << "\t";
	  if (dim == 2)
	    myfile_reaction_force << time_force.second[0] << "\t"
	           << time_force.second[1] << std::endl;
	  if (dim == 3)
	    myfile_reaction_force << time_force.second[0] << "\t"
	           << time_force.second[1] << "\t"
		   << time_force.second[2] << std::endl;
	}
      myfile_reaction_force.close();
    }
    else
      m_logfile << "Unable to open file";

    std::ofstream myfile_energy ("Energy.hist");
    if (myfile_energy.is_open())
    {
      myfile_energy << std::fixed << std::setprecision(10) << std::scientific
                    << 0.0 << "\t"
                    << 0.0 << "\t"
	            << 0.0 << "\t"
	            << 0.0 << std::endl;

      for (auto const time_energy : m_history_energy)
	{
	  myfile_energy << std::fixed << std::setprecision(10) << std::scientific
	                << time_energy.first     << "\t"
                        << time_energy.second[0] << "\t"
	                << time_energy.second[1] << "\t"
		        << time_energy.second[2] << std::endl;
	}
      myfile_energy.close();
    }
    else
      m_logfile << "Unable to open file";
  }

  template <int dim>
  double PhaseFieldMonolithicSolve<dim>::calculate_energy_functional() const
  {
    double energy_functional = 0.0;

    FEValues<dim> fe_values(m_fe, m_qf_cell, update_JxW_values);

    for (const auto &cell : m_dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);

        const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
          m_quadrature_point_history.get_data(cell);
        Assert(lqph.size() == m_n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < m_n_q_points; ++q_point)
          {
            const double JxW = fe_values.JxW(q_point);
            energy_functional += lqph[q_point]->get_total_strain_energy() * JxW;
            energy_functional += lqph[q_point]->get_crack_energy_dissipation() * JxW;
          }
      }

    return energy_functional;
  }

  template <int dim>
  std::pair<double, double>
    PhaseFieldMonolithicSolve<dim>::calculate_total_strain_energy_and_crack_energy_dissipation() const
  {
    double total_strain_energy = 0.0;
    double crack_energy_dissipation = 0.0;

    FEValues<dim> fe_values(m_fe, m_qf_cell, update_JxW_values);

    for (const auto &cell : m_dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);

        const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
          m_quadrature_point_history.get_data(cell);
        Assert(lqph.size() == m_n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < m_n_q_points; ++q_point)
          {
            const double JxW = fe_values.JxW(q_point);
            total_strain_energy += lqph[q_point]->get_total_strain_energy() * JxW;
            crack_energy_dissipation += lqph[q_point]->get_crack_energy_dissipation() * JxW;
          }
      }

    return std::make_pair(total_strain_energy, crack_energy_dissipation);
  }


  template <int dim>
  bool PhaseFieldMonolithicSolve<dim>::local_refine_and_solution_transfer(BlockVector<double> & solution_delta,
									  BlockVector<double> & LBFGS_update_refine)
  {
    // This is the solution at (n+1) obtained from the old (coarse) mesh
    BlockVector<double> solution_next_step(m_dofs_per_block);
    solution_next_step = m_solution + solution_delta;
    bool mesh_is_same = true;
    bool cell_refine_flag = true;

    unsigned int material_id;
    double length_scale;
    double cell_length;
    while(cell_refine_flag)
      {
	cell_refine_flag = false;

	std::vector<types::global_dof_index> local_dof_indices(m_fe.dofs_per_cell);
	for (const auto &cell : m_dof_handler.active_cell_iterators())
	  {
	    cell->get_dof_indices(local_dof_indices);

	    for (unsigned int i = 0; i< m_fe.dofs_per_cell; ++i)
	      {
		const unsigned int comp_i = m_fe.system_to_component_index(i).first;
		if (comp_i == m_d_component) //phasefield component
		  {
		    if (  solution_next_step(local_dof_indices[i])
			> m_parameters.m_phasefield_refine_threshold )
		      {
			material_id = cell->material_id();
	                length_scale = m_material_data[material_id][2];
	                if (dim == 2)
	                  cell_length = std::sqrt(cell->measure());
	                else
	                  cell_length = std::cbrt(cell->measure());
			if (  cell_length
			    > length_scale * m_parameters.m_allowed_max_h_l_ratio )
			  {
			    if (cell->level() < m_parameters.m_max_allowed_refinement_level)
			      {
			        cell->set_refine_flag();
			        break;
			      }
			  }
		      }
		  }
	      }
	  }

	for (const auto &cell : m_dof_handler.active_cell_iterators())
	  {
	    if (cell->refine_flag_set())
	      {
		cell_refine_flag = true;
		break;
	      }
	  }

	// if any cell is refined, we need to project the solution
	// to the newly refined mesh
	if (cell_refine_flag)
	  {
	    mesh_is_same = false;

	    std::vector<BlockVector<double> > old_solutions(2);
	    old_solutions[0] = solution_next_step;
	    old_solutions[1] = m_solution;

	    m_triangulation.prepare_coarsening_and_refinement();
	    SolutionTransfer<dim, BlockVector<double>> solution_transfer(m_dof_handler);
	    solution_transfer.prepare_for_coarsening_and_refinement(old_solutions);
	    m_triangulation.execute_coarsening_and_refinement();

	    setup_system();

	    std::vector<BlockVector<double>> tmp_solutions(2);
	    tmp_solutions[0].reinit(m_dofs_per_block);
	    tmp_solutions[1].reinit(m_dofs_per_block);

	    solution_transfer.interpolate(tmp_solutions);
	    // If an older version of dealII is used, for example, 9.4.0, interpolate()
            // needs to use the following interface.
            //solution_transfer.interpolate(old_solutions, tmp_solutions);
	    solution_next_step = tmp_solutions[0];
	    m_solution = tmp_solutions[1];

	    // make sure the projected solutions still satisfy
	    // hanging node constraints
	    m_constraints.distribute(solution_next_step);
	    m_constraints.distribute(m_solution);
	  } // if (cell_refine_flag)
      } // while(cell_refine_flag)

    // calculate field variables for newly refined cells
    if (!mesh_is_same)
      {
	BlockVector<double> temp_solution_delta(m_dofs_per_block);
	BlockVector<double> temp_previous_solution(m_dofs_per_block);
	temp_solution_delta = 0.0;
	temp_previous_solution = 0.0;
	update_qph_incremental(temp_solution_delta, temp_previous_solution, false);
	update_history_field_step();

	// initial guess for the resolve on the refined mesh
	LBFGS_update_refine = solution_next_step - m_solution;
      }

    return mesh_is_same;
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::print_parameter_information()
  {
    m_logfile << "Scenario number = " << m_parameters.m_scenario << std::endl;
    m_logfile << "Log file = " << m_parameters.m_logfile_name << std::endl;
    m_logfile << "Write iteration history to log file? = " << std::boolalpha
	      << m_parameters.m_output_iteration_history << std::endl;
    m_logfile << "Nonlinear solver type = " << m_parameters.m_type_nonlinear_solver << std::endl;
    m_logfile << "Line search type = " << m_parameters.m_type_line_search << std::endl;
    m_logfile << "Linear solver type = " << m_parameters.m_type_linear_solver << std::endl;
    m_logfile << "Mesh refinement strategy = " << m_parameters.m_refinement_strategy << std::endl;
    m_logfile << "L-BFGS_m = " << m_parameters.m_LBFGS_m << std::endl;
    m_logfile << "Global refinement times = " << m_parameters.m_global_refine_times << std::endl;
    m_logfile << "Local prerefinement times = " <<m_parameters. m_local_prerefine_times << std::endl;
    m_logfile << "Maximum adaptive refinement times allowed in each step = "
	      << m_parameters.m_max_adaptive_refine_times << std::endl;
    m_logfile << "Maximum allowed cell refinement level = "
    	      << m_parameters.m_max_allowed_refinement_level << std::endl;
    m_logfile << "Phasefield-based refinement threshold value = "
	      << m_parameters.m_phasefield_refine_threshold << std::endl;
    m_logfile << "Allowed maximum h/l ratio = " << m_parameters.m_allowed_max_h_l_ratio << std::endl;
    m_logfile << "total number of material types = " << m_parameters.m_total_material_regions << std::endl;
    m_logfile << "material data file name = " << m_parameters.m_material_file_name << std::endl;
    if (m_parameters.m_reaction_force_face_id >= 0)
      m_logfile << "Calculate reaction forces on Face ID = " << m_parameters.m_reaction_force_face_id << std::endl;
    else
      m_logfile << "No need to calculate reaction forces." << std::endl;

    if (m_parameters.m_relative_residual)
      m_logfile << "Relative residual for convergence." << std::endl;
    else
      m_logfile << "Absolute residual for convergence." << std::endl;

    m_logfile << "Body force = (" << m_parameters.m_x_component << ", "
                                  << m_parameters.m_y_component << ", "
	                          << m_parameters.m_z_component << ") (N/m^3)"
				  << std::endl;

    m_logfile << "End time = " << m_parameters.m_end_time << std::endl;
    m_logfile << "Time data file name = " << m_parameters.m_time_file_name << std::endl;
  }

  template <int dim>
  void PhaseFieldMonolithicSolve<dim>::run()
  {
    print_parameter_information();

    read_material_data(m_parameters.m_material_file_name,
    		       m_parameters.m_total_material_regions);

    std::vector<std::array<double, 4>> time_table;

    read_time_data(m_parameters.m_time_file_name, time_table);

    make_grid();
    setup_system();
    output_results();

    m_time.increment(time_table);

    while(m_time.current() < m_time.end() + m_time.get_delta_t()*1.0e-6)
      {
	m_logfile << std::endl
		  << "Timestep " << m_time.get_timestep() << " @ " << m_time.current()
		  << 's' << std::endl;

        bool mesh_is_same = false;

        // initial guess for the resolve on the refined mesh
	BlockVector<double> LBFGS_update_refine(m_dofs_per_block);
	LBFGS_update_refine = 0.0;

        // local adaptive mesh refinement loop
	unsigned int adp_refine_iteration = 0;
        for (; adp_refine_iteration < m_parameters.m_max_adaptive_refine_times + 1; ++adp_refine_iteration)
          {
	    if (m_parameters.m_refinement_strategy == "adaptive-refine")
	      m_logfile << "\tAdaptive refinement-"<< adp_refine_iteration << ": " << std::endl;

	    BlockVector<double> solution_delta(m_dofs_per_block);
	    solution_delta = 0.0;

	    if (m_parameters.m_type_nonlinear_solver == "Newton")
	      {
		bool newton_success = false;
		newton_success = solve_nonlinear_timestep_newton(solution_delta);
		AssertThrow(newton_success,
		            ExcMessage("No convergence in Newton-Raphson nonlinear solver!"));
		/*
		// if Newton-Raphson failed, use LBFGS solver
		if (!newton_success)
		  {
		    solution_delta = 0.0;
		    solve_nonlinear_timestep_LBFGS(solution_delta, LBFGS_update_refine);
		  }
		*/
	      }
	    else if (m_parameters.m_type_nonlinear_solver == "BFGS")
	      solve_nonlinear_timestep_BFGS(solution_delta);
	    else if (m_parameters.m_type_nonlinear_solver == "LBFGS")
	      solve_nonlinear_timestep_LBFGS(solution_delta, LBFGS_update_refine);
	    else
	      AssertThrow(false, ExcMessage("Nonlinear solver type not implemented"));

	    if (m_parameters.m_refinement_strategy == "adaptive-refine")
	      {

		if (adp_refine_iteration == m_parameters.m_max_adaptive_refine_times)
		  {
		    m_solution += solution_delta;
		    break;
		  }

		mesh_is_same = local_refine_and_solution_transfer(solution_delta,
								  LBFGS_update_refine);

		if (mesh_is_same)
		  {
		    m_solution += solution_delta;
		    break;
		  }
	      }
	    else if (m_parameters.m_refinement_strategy == "pre-refine")
	      {
		m_solution += solution_delta;
	        break;
	      }
	    else
	      {
		AssertThrow(false,
		            ExcMessage("Selected mesh refinement strategy not implemented!"));
	      }
          } // for (; adp_refine_iteration < m_parameters.m_max_adaptive_refine_times; ++adp_refine_iteration)

        //AssertThrow(adp_refine_iteration < m_parameters.m_max_adaptive_refine_times,
        //            ExcMessage("Number of local adaptive mesh refinement exceeds allowed maximum times!"));

	update_history_field_step();
	// output vtk files every 10 steps if there are too
	// many time steps
	//if (m_time.get_timestep() % 10 == 0)
        output_results();

	double energy_functional_current = calculate_energy_functional();
	m_logfile << "\t\tEnergy functional (J) = " << std::fixed << std::setprecision(10) << std::scientific
	          << energy_functional_current << std::endl;

	std::pair<double, double> energy_pair = calculate_total_strain_energy_and_crack_energy_dissipation();
	m_logfile << "\t\tTotal strain energy (J) = " << std::fixed << std::setprecision(10) << std::scientific
		  << energy_pair.first << std::endl;
	m_logfile << "\t\tCrack energy dissipation (J) = " << std::fixed << std::setprecision(10) << std::scientific
		  << energy_pair.second << std::endl;

	std::pair<double, std::array<double, 3>> time_energy;
	time_energy.first = m_time.current();
	time_energy.second[0] = energy_pair.first;
	time_energy.second[1] = energy_pair.second;
	time_energy.second[2] = energy_pair.first + energy_pair.second;
	m_history_energy.push_back(time_energy);

	int face_ID = m_parameters.m_reaction_force_face_id;
	if (face_ID >= 0)
	  calculate_reaction_force(face_ID);

        write_history_data();

	m_time.increment(time_table);
      } // while(m_time.current() < m_time.end() + m_time.get_delta_t()*1.0e-6)
  }
} // namespace PhaseField


int main(int argc, char* argv[])
{

  using namespace dealii;

  if (argc != 2)
    AssertThrow(false,
    		ExcMessage("The number of arguments provided to the program has to be 2!"));

  const unsigned int dim = std::stoi(argv[1]);
  if (dim == 2 )
    {
      PhaseField::PhaseFieldMonolithicSolve<2> FEQ1Full("parameters.prm");
      FEQ1Full.run();
    }
  else if (dim == 3)
    {
      PhaseField::PhaseFieldMonolithicSolve<3> SphereInclusion3D("parameters.prm");
      SphereInclusion3D.run();
    }
  else
    {
      AssertThrow(false,
                  ExcMessage("Dimension has to be either 2 or 3"));
    }

  return 0;
}
