/* ---------------------------------------------------------------------
 * Copyright (C) 2017 by the deal.II authors and
 *                           Jean-Paul Pelteret and Tim Hamann
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 */

/*
 * Authors: Jean-Paul Pelteret, Tim Hamann,
 *          University of Erlangen-Nuremberg, 2017
 *
 * The support of this work by the European Research Council (ERC) through
 * the Advanced Grant 289049 MOCOPOLY is gratefully acknowledged by the
 * first author.
 */

// @sect3{Include files}

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/physics/transformations.h>

#include <fstream>
#include <iostream>
#include <vector>

namespace LMM
{
  using namespace dealii;

// @sect3{Run-time parameters}
//
// There are several parameters that can be set in the code so we set up a
// ParameterHandler object to read in the choices at run-time.
  namespace Parameters
  {
// @sect4{Finite Element system}

// Here we specify the polynomial order used to approximate the solution.
// The quadrature order should be adjusted accordingly.
    struct FESystem
    {
      unsigned int poly_degree;
      unsigned int quad_order;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void FESystem::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        prm.declare_entry("Polynomial degree", "1",
                          Patterns::Integer(0),
                          "Displacement system polynomial order");

        prm.declare_entry("Quadrature order", "2",
                          Patterns::Integer(0),
                          "Gauss quadrature order");
      }
      prm.leave_subsection();
    }

    void FESystem::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        poly_degree = prm.get_integer("Polynomial degree");
        quad_order = prm.get_integer("Quadrature order");
      }
      prm.leave_subsection();
    }

// @sect4{Problem}

// Choose which problem is going to be solved
    struct Problem
    {
      std::string problem;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void Problem::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Problem");
      {
        prm.declare_entry("Problem", "IsotonicContraction",
                          Patterns::Selection("IsotonicContraction|BicepsBrachii"),
                          "The problem that is to be solved");
      }
      prm.leave_subsection();
    }

    void Problem::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Problem");
      {
        problem = prm.get("Problem");
      }
      prm.leave_subsection();
    }

// @sect4{IsotonicContractionGeometry}

// Make adjustments to the geometry and discretisation of the
// isotonic contraction model from Martins2006.

    struct IsotonicContraction
    {
      const double half_length_x = 10e-3/2.0;
      const double half_length_y = 10e-3/2.0;
      const double half_length_z = 1e-3/2.0;
      const types::boundary_id bid_CC_dirichlet_symm_X = 1;
      const types::boundary_id bid_CC_dirichlet_symm_Z = 2;
      const types::boundary_id bid_CC_neumann = 10;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void IsotonicContraction::declare_parameters(ParameterHandler &/*prm*/)
    {

    }

    void IsotonicContraction::parse_parameters(ParameterHandler &/*prm*/)
    {

    }

// @sect4{BicepsBrachiiGeometry}

// Make adjustments to the geometry and discretisation of the
// biceps model.

    struct BicepsBrachii
    {
      double       axial_length;
      double       radius_insertion_origin;
      double       radius_midpoint;
      double       scale;
      unsigned int elements_along_axis;
      unsigned int n_refinements_radial;
      bool         include_gravity;
      double       axial_force;

      const types::boundary_id bid_BB_dirichlet_X = 1;
      const types::boundary_id bid_BB_neumann = 10;
      const types::boundary_id mid_BB_radial = 100;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void BicepsBrachii::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Biceps Brachii geometry");
      {
        prm.declare_entry("Axial length", "250",
                          Patterns::Double(0),
                          "Axial length of the muscle");

        prm.declare_entry("Radius insertion and origin", "5",
                          Patterns::Double(0),
                          "Insertion and origin radius");

        prm.declare_entry("Radius midpoint", "7.5",
                          Patterns::Double(0),
                          "Radius at the midpoint of the muscle");

        prm.declare_entry("Grid scale", "1e-3",
                          Patterns::Double(0.0),
                          "Global grid scaling factor");

        prm.declare_entry("Elements along axis", "32",
                          Patterns::Integer(2),
                          "Number of elements along the muscle axis");

        prm.declare_entry("Radial refinements", "4",
                          Patterns::Integer(0),
                          "Control the discretisation in the radial direction");

        prm.declare_entry("Gravity", "false",
                          Patterns::Bool(),
                          "Include the effects of gravity (in the y-direction; "
                          " perpendicular to the muscle axis)");

        prm.declare_entry("Axial force", "1",
                          Patterns::Double(),
                          "Applied distributed axial force (in Newtons)");
      }
      prm.leave_subsection();
    }

    void BicepsBrachii::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Biceps Brachii geometry");
      {
        axial_length = prm.get_double("Axial length");
        radius_insertion_origin = prm.get_double("Radius insertion and origin");
        radius_midpoint = prm.get_double("Radius midpoint");
        scale = prm.get_double("Grid scale");
        elements_along_axis = prm.get_integer("Elements along axis");
        n_refinements_radial = prm.get_integer("Radial refinements");
        include_gravity = prm.get_bool("Gravity");
        axial_force = prm.get_double("Axial force");
      }
      prm.leave_subsection();

      AssertThrow(radius_midpoint >= radius_insertion_origin,
                  ExcMessage("Unrealistic geometry"));
    }

// @sect4{Neurological signal}

    struct NeurologicalSignal
    {
      double neural_signal_start_time;
      double neural_signal_end_time;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void NeurologicalSignal::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Neurological signal");
      {
        prm.declare_entry("Start time", "1.0",
                          Patterns::Double(0),
                          "Time at which to start muscle activation");

        prm.declare_entry("End time", "2.0",
                          Patterns::Double(0),
                          "Time at which to remove muscle activation signal");
      }
      prm.leave_subsection();
    }

    void NeurologicalSignal::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Neurological signal");
      {
        neural_signal_start_time = prm.get_double("Start time");
        neural_signal_end_time = prm.get_double("End time");
      }
      prm.leave_subsection();

      Assert(neural_signal_start_time < neural_signal_end_time,
             ExcMessage("Invalid neural signal times."));
    }

// @sect4{Time}

// Set the timestep size $ \varDelta t $ and the simulation end-time.
    struct Time
    {
      double delta_t;
      double end_time;
      double end_ramp_time;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void Time::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        prm.declare_entry("End time", "3",
                          Patterns::Double(0),
                          "End time");

        prm.declare_entry("End ramp time", "1",
                          Patterns::Double(0),
                          "Force ramp end time");

        prm.declare_entry("Time step size", "0.1",
                          Patterns::Double(0),
                          "Time step size");
      }
      prm.leave_subsection();
    }

    void Time::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        end_time = prm.get_double("End time");
        end_ramp_time = prm.get_double("End ramp time");
        delta_t = prm.get_double("Time step size");
      }
      prm.leave_subsection();
    }

// @sect4{All parameters}

// Finally we consolidate all of the above structures into a single container
// that holds all of our run-time selections.
    struct AllParameters : public FESystem,
      public Problem,
      public IsotonicContraction,
      public BicepsBrachii,
      public NeurologicalSignal,
      public Time
    {
      AllParameters(const std::string &input_file);

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
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
      FESystem::declare_parameters(prm);
      Problem::declare_parameters(prm);
      IsotonicContraction::declare_parameters(prm);
      BicepsBrachii::declare_parameters(prm);
      NeurologicalSignal::declare_parameters(prm);
      Time::declare_parameters(prm);
    }

    void AllParameters::parse_parameters(ParameterHandler &prm)
    {
      FESystem::parse_parameters(prm);
      Problem::parse_parameters(prm);
      IsotonicContraction::parse_parameters(prm);
      BicepsBrachii::parse_parameters(prm);
      NeurologicalSignal::parse_parameters(prm);
      Time::parse_parameters(prm);

      // Override time setting for test defined
      // in the literature
      if (problem == "IsotonicContraction")
        {
          end_time = 3.0;
          end_ramp_time = 1.0;
          delta_t = 0.1;

          neural_signal_start_time = 1.0;
          neural_signal_end_time = 2.0;
        }
    }
  }

  // @sect3{Body force values}

  template <int dim>
  class BodyForce :  public Function<dim>
  {
  public:
    BodyForce (const double rho,
               const Tensor<1,dim> direction);
    virtual ~BodyForce () {}

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &values) const;

    virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                    std::vector<Vector<double> >   &value_list) const;

    const double rho;
    const double g;
    const Tensor<1,dim> M;
  };


  template <int dim>
  BodyForce<dim>::BodyForce (const double rho,
                             const Tensor<1,dim> direction)
    :
    Function<dim> (dim),
    rho (rho),
    g (9.81),
    M (direction)
  {
    Assert(M.norm() == 1.0, ExcMessage("Direction vector is not a unit vector"));
  }


  template <int dim>
  inline
  void BodyForce<dim>::vector_value (const Point<dim> &/*p*/,
                                     Vector<double>   &values) const
  {
    Assert (values.size() == dim,
            ExcDimensionMismatch (values.size(), dim));
    Assert (dim >= 2, ExcNotImplemented());
    for (unsigned int d=0; d<dim; ++d)
      {
        values(d) = rho*g*M[d];
      }
  }


  template <int dim>
  void BodyForce<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                          std::vector<Vector<double> >   &value_list) const
  {
    Assert (value_list.size() == points.size(),
            ExcDimensionMismatch (value_list.size(), points.size()));

    const unsigned int n_points = points.size();

    for (unsigned int p=0; p<n_points; ++p)
      BodyForce<dim>::vector_value (points[p],
                                    value_list[p]);
  }

  template <int dim>
  class Traction :  public Function<dim>
  {
  public:
    Traction (const double force,
              const double area);
    virtual ~Traction () {}

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &values) const;

    virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                    std::vector<Vector<double> >   &value_list) const;

    const double t;
  };


  template <int dim>
  Traction<dim>::Traction (const double force,
                           const double area)
    :
    Function<dim> (dim),
    t (force/area)
  {}


  template <int dim>
  inline
  void Traction<dim>::vector_value (const Point<dim> &/*p*/,
                                    Vector<double>   &values) const
  {
    Assert (values.size() == dim,
            ExcDimensionMismatch (values.size(), dim));
    Assert (dim == 3, ExcNotImplemented());

    // Assume uniform distributed load
    values(0) = t;
    values(1) = 0.0;
    values(2) = 0.0;
  }


  template <int dim>
  void Traction<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                         std::vector<Vector<double> >   &value_list) const
  {
    Assert (value_list.size() == points.size(),
            ExcDimensionMismatch (value_list.size(), points.size()));

    const unsigned int n_points = points.size();

    for (unsigned int p=0; p<n_points; ++p)
      Traction<dim>::vector_value (points[p],
                                   value_list[p]);
  }

  // @sect3{Utility functions}

  template <int dim>
  inline
  Tensor<2,dim> get_deformation_gradient (std::vector<Tensor<1,dim> > &grad)
  {
    Assert (grad.size() == dim, ExcInternalError());

    Tensor<2,dim> F (unit_symmetric_tensor<dim>());
    for (unsigned int i=0; i<dim; ++i)
      for (unsigned int j=0; j<dim; ++j)
        F[i][j] += grad[i][j];
    return F;
  }

  template <int dim>
  inline
  SymmetricTensor<2,dim> get_small_strain (std::vector<Tensor<1,dim> > &grad)
  {
    Assert (grad.size() == dim, ExcInternalError());

    SymmetricTensor<2,dim> strain;
    for (unsigned int i=0; i<dim; ++i)
      strain[i][i] = grad[i][i];

    for (unsigned int i=0; i<dim; ++i)
      for (unsigned int j=i+1; j<dim; ++j)
        strain[i][j] = (grad[i][j] + grad[j][i]) / 2;
    return strain;
  }

  // @sect3{Properties for muscle matrix}

  struct MuscleMatrix
  {
    static const double E; // Young's modulus
    static const double nu; // Poisson ratio

    static const double mu; // Shear modulus
    static const double lambda; // Lame parameter
  };

  const double MuscleMatrix::E = 26e3;
  const double MuscleMatrix::nu = 0.45;
  const double MuscleMatrix::mu = MuscleMatrix::E/(2.0*(1.0 + MuscleMatrix::nu));
  const double MuscleMatrix::lambda = 2.0*MuscleMatrix::mu *MuscleMatrix::nu/(1.0 - 2.0*MuscleMatrix::nu);

// @sect3{Local data for muscle fibres}

#define convert_gf_to_N 1.0/101.97
#define convert_gf_per_cm2_to_N_per_m2 convert_gf_to_N*1e2*1e2
#define T0 6280.0*convert_gf_per_cm2_to_N_per_m2

  // A struct that governs the functioning of a single muscle fibre
  template <int dim>
  struct MuscleFibre
  {
    MuscleFibre (void)
      : alpha (0.0),
        alpha_t1 (0.0),
        epsilon_f (0.0),
        epsilon_c (0.0),
        epsilon_c_t1 (0.0),
        epsilon_c_dot (0.0)
    {

    }

    MuscleFibre(const Tensor<1,dim> &direction)
      : M (direction),
        alpha (0.0),
        alpha_t1 (0.0),
        epsilon_f (0.0),
        epsilon_c (0.0),
        epsilon_c_t1 (0.0),
        epsilon_c_dot (0.0)
    {
      Assert(M.norm() == 1.0,
             ExcMessage("Fibre direction is not a unit vector"));
    }

    void update_alpha (const double u,
                       const double dt);

    void update_state(const SymmetricTensor<2,dim> &strain_tensor,
                      const double dt);

    const Tensor<1,dim> &get_M () const
    {
      return M;
    }
    double get_m_p () const;
    double get_m_s () const;
    double get_beta (const double dt) const;
    double get_gamma (const double dt) const;

    // Postprocessing
    const double &get_alpha() const
    {
      return alpha;
    }
    const double &get_epsilon_f() const
    {
      return epsilon_f;
    }
    const double &get_epsilon_c() const
    {
      return epsilon_c;
    }
    const double &get_epsilon_c_dot() const
    {
      return epsilon_c_dot;
    }

  private:
    Tensor<1,dim> M; // Direction

    double alpha;    // Activation level at current timestep
    double alpha_t1; // Activation level at previous timestep

    double epsilon_f;     // Fibre strain at current timestep
    double epsilon_c;     // Contractile strain at current timestep
    double epsilon_c_t1;  // Contractile strain at previous timestep
    double epsilon_c_dot; // Contractile velocity at previous timestep

    double get_f_c_L () const;
    double get_m_c_V () const;
    double get_c_c_V () const;
  };

  template <int dim>
  void MuscleFibre<dim>::update_alpha (const double u,
                                       const double dt)
  {
    static const double tau_r = 0.15; // s
    static const double tau_f = 0.15; // s
    static const double alpha_min = 0;

    if (u == 1.0)
      alpha = (alpha_t1*tau_r*tau_f + dt*tau_f) / (tau_r*tau_f + dt*tau_f);
    else if (u == 0)
      alpha = (alpha_t1*tau_r*tau_f + dt*alpha_min*tau_r) / (tau_r*tau_f + dt*tau_r);
    else
      {
        const double b = 1.0/tau_r - 1.0/tau_f;
        const double c = 1.0/tau_f;
        const double d = alpha_min/tau_f;
        const double f1 = 1.0/tau_r - alpha_min/tau_f;
        const double p = b*u + c;
        const double q = f1*u + d;

        alpha = (q*dt + alpha_t1)/(1.0 + p*dt);
      }
  }


  template <int dim>
  double MuscleFibre<dim>::get_m_p () const
  {
    static const double A = 8.568e-4*convert_gf_per_cm2_to_N_per_m2;
    static const double a = 12.43;
    if (epsilon_f >= 0.0)
      {
        // 100 times more compliant than Martins2006
        static const double m_p = 2.0*A*a/1e2;
        return m_p;
      }
    else
      return 0.0;
  }

  template <int dim>
  double MuscleFibre<dim>::get_m_s (void) const
  {
    const double epsilon_s = epsilon_f - epsilon_c; // Small strain assumption
    if (epsilon_s >= -1e-6) // Tolerant check
      return 10.0;
    else
      return 0.0;
  }

  template <int dim>
  double MuscleFibre<dim>::get_f_c_L (void) const
  {
    if (epsilon_c <= 0.5 && epsilon_c >= -0.5)
      return 1.0;
    else
      return 0.0;
  }

  template <int dim>
  double MuscleFibre<dim>::get_m_c_V (void) const
  {
    if (epsilon_c_dot < -5.0)
      return 0.0;
    else if (epsilon_c_dot <= 3.0)
      return 1.0/5.0;
    else
      return 0.0;
  }

  template <int dim>
  double MuscleFibre<dim>::get_c_c_V (void) const
  {
    if (epsilon_c_dot < -5.0)
      return 0.0;
    else if (epsilon_c_dot <= 3.0)
      return 1.0;
    else
      return 1.6;
  }

  template <int dim>
  double MuscleFibre<dim>::get_beta(const double dt) const
  {
    return get_f_c_L()*get_m_c_V()*alpha/dt + get_m_s();
  }

  template <int dim>
  double MuscleFibre<dim>::get_gamma(const double dt) const
  {
    return get_f_c_L()*alpha*(get_m_c_V()*epsilon_c_t1/dt - get_c_c_V());
  }

  template <int dim>
  void MuscleFibre<dim>::update_state(const SymmetricTensor<2,dim> &strain_tensor,
                                      const double dt)
  {
    // Values from previous state
    // These were the values that were used in the assembly,
    // so we must use them in the update step to be consistant.
    // Need to compute these before we overwrite epsilon_c_t1
    const double m_s = get_m_s();
    const double beta = get_beta(dt);
    const double gamma = get_gamma(dt);

    // Update current state
    alpha_t1 = alpha;
    epsilon_f = M*static_cast< Tensor<2,dim> >(strain_tensor)*M;
    epsilon_c_t1 = epsilon_c;
    epsilon_c = (m_s*epsilon_f + gamma)/beta;
    epsilon_c_dot = (epsilon_c - epsilon_c_t1)/dt;
  }


  // @sect3{The <code>LinearMuscleModelProblem</code> class template}

  template <int dim>
  class LinearMuscleModelProblem
  {
  public:
    LinearMuscleModelProblem (const std::string &input_file);
    ~LinearMuscleModelProblem ();
    void run ();

  private:
    void make_grid ();
    void setup_muscle_fibres ();
    double get_neural_signal (const double time);
    void update_fibre_activation (const double time);
    void update_fibre_state ();
    void setup_system ();
    void assemble_system (const double time);
    void apply_boundary_conditions ();
    void solve ();
    void output_results (const unsigned int timestep,
                         const double time) const;

    Parameters::AllParameters parameters;

    Triangulation<dim>   triangulation;
    DoFHandler<dim>      dof_handler;

    FESystem<dim>        fe;
    QGauss<dim>          qf_cell;
    QGauss<dim-1>        qf_face;

    ConstraintMatrix     hanging_node_constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double>       solution;
    Vector<double>       system_rhs;

    // Time
    const double t_end;
    const double dt;
    const double t_ramp_end; // Force ramp end time

    // Loading
    const BodyForce<dim> body_force;
    const Traction<dim>  traction;

    // Local data
    std::vector< std::vector<MuscleFibre<dim> > > fibre_data;

    // Constitutive functions for assembly
    SymmetricTensor<4,dim> get_stiffness_tensor (const unsigned int cell,
                                                 const unsigned int q_point_cell) const;
    SymmetricTensor<2,dim> get_rhs_tensor (const unsigned int cell,
                                           const unsigned int q_point_cell) const;
  };

  // @sect4{LinearMuscleModelProblem::LinearMuscleModelProblem}

  template <int dim>
  LinearMuscleModelProblem<dim>::LinearMuscleModelProblem (const std::string &input_file)
    :
    parameters(input_file),
    dof_handler (triangulation),
    fe (FE_Q<dim>(parameters.poly_degree), dim),
    qf_cell (parameters.quad_order),
    qf_face (parameters.quad_order),
    t_end (parameters.end_time),
    dt (parameters.delta_t),
    t_ramp_end(parameters.end_ramp_time),
    body_force ((parameters.problem == "BicepsBrachii"  &&parameters.include_gravity == true) ?
    BodyForce<dim>(0.375*1000.0, Tensor<1,dim>({0,-1,0}))  : // (reduced) Density and direction
  BodyForce<dim>(0.0, Tensor<1,dim>({0,0,1})) ),
            traction (parameters.problem == "BicepsBrachii" ?
                      Traction<dim>(parameters.axial_force, // Force, area
                                    M_PI*std::pow(parameters.radius_insertion_origin *parameters.scale,2.0) ) :
                      Traction<dim>(4.9*convert_gf_to_N, // Force; Conversion of gf to N,
                                    (2.0*parameters.half_length_y)*(2.0*parameters.half_length_z)) ) // Area
  {
    Assert(dim==3, ExcNotImplemented());
  }


  // @sect4{LinearMuscleModelProblem::~LinearMuscleModelProblem}

  template <int dim>
  LinearMuscleModelProblem<dim>::~LinearMuscleModelProblem ()
  {
    dof_handler.clear ();
  }


  // @sect4{LinearMuscleModelProblem::make_grid}

  template<int dim>
  struct BicepsGeometry
  {
    BicepsGeometry(const double axial_length,
                   const double radius_ins_orig,
                   const double radius_mid)
      :
      ax_lgth (axial_length),
      r_ins_orig (radius_ins_orig),
      r_mid (radius_mid)
    {}

    // The radial profile of the muscle
    // This provides the new coordinates for points @p pt
    // on a cylinder of radius r_ins_orig and length
    // ax_lgth to be moved to in order to create the
    // physiologically representative geometry of
    // the muscle
    Point<dim> profile (const Point<dim> &pt_0) const
    {
      Assert(pt_0[0] > -1e-6,
             ExcMessage("All points must have x-coordinate > 0"));

      const double r_scale = get_radial_scaling_factor(pt_0[0]);
      return pt_0 + Point<dim>(0.0, r_scale*pt_0[1], r_scale*pt_0[2]);
    }

    Point<dim> operator() (const Point<dim> &pt) const
    {
      return profile(pt);
    }

    // Provides the muscle direction at the point @p pt
    // in the real geometry (one that has undergone the
    // transformation given by the profile() function)
    // and subequent grid rescaling.
    // The directions are given by the gradient of the
    // transformation function (i.e. the fibres are
    // orientated by the curvature of the muscle).
    //
    // So, being lazy, we transform the current point back
    // to the original point on the completely unscaled
    // cylindrical grid. We then evaluate the transformation
    // at two points (axially displaced) very close to the
    // point of interest. The normalised vector joining the
    // transformed counterparts of the perturbed points is
    // the gradient of the transformation function and,
    // thus, defines the fibre direction.
    Tensor<1,dim> direction (const Point<dim> &pt_scaled,
                             const double     &grid_scale) const
    {
      const Point<dim> pt = (1.0/grid_scale)*pt_scaled;
      const Point<dim> pt_0 = inv_profile(pt);

      static const double eps = 1e-6;
      const Point<dim> pt_0_eps_p = pt_0 + Point<dim>(+eps,0,0);
      const Point<dim> pt_0_eps_m = pt_0 + Point<dim>(-eps,0,0);
      const Point<dim> pt_eps_p = profile(pt_0_eps_p);
      const Point<dim> pt_eps_m = profile(pt_0_eps_m);

      static const double tol = 1e-9;
      Assert(profile(pt_0).distance(pt) < tol, ExcInternalError());
      Assert(inv_profile(pt_eps_p).distance(pt_0_eps_p) < tol, ExcInternalError());
      Assert(inv_profile(pt_eps_m).distance(pt_0_eps_m) < tol, ExcInternalError());

      Tensor<1,dim> dir = pt_eps_p-pt_eps_m;
      dir /= dir.norm();
      return dir;
    }

  private:
    const double ax_lgth;
    const double r_ins_orig;
    const double r_mid;

    double get_radial_scaling_factor (const double &x) const
    {
      // Expect all grid points with X>=0, but we provide a
      // tolerant location for points "on" the Cartesian plane X=0
      const double lgth_frac = std::max(x/ax_lgth,0.0);
      const double amplitude = 0.25*(r_mid - r_ins_orig);
      const double phase_shift = M_PI;
      const double y_shift = 1.0;
      const double wave_func = y_shift + std::cos(phase_shift + 2.0*M_PI*lgth_frac);
      Assert(wave_func >= 0.0, ExcInternalError());
      return std::sqrt(amplitude*wave_func);
    }

    Point<dim> inv_profile (const Point<dim> &pt) const
    {
      Assert(pt[0] > -1e-6,
             ExcMessage("All points must have x-coordinate > 0"));

      const double r_scale = get_radial_scaling_factor(pt[0]);
      const double trans_inv_scale = 1.0/(1.0+r_scale);
      return Point<dim>(pt[0], trans_inv_scale*pt[1], trans_inv_scale*pt[2]);
    }
  };

  template <int dim>
  void LinearMuscleModelProblem<dim>::make_grid ()
  {
    Assert (dim == 3, ExcNotImplemented());

    if (parameters.problem == "IsotonicContraction")
      {
        const Point<dim> p1(-parameters.half_length_x,
                            -parameters.half_length_y,
                            -parameters.half_length_z);
        const Point<dim> p2( parameters.half_length_x,
                             parameters.half_length_y,
                             parameters.half_length_z);

        GridGenerator::hyper_rectangle (triangulation, p1, p2);

        typename Triangulation<dim>::active_cell_iterator cell =
          triangulation.begin_active(), endc = triangulation.end();
        for (; cell != endc; ++cell)
          {
            for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
              {
                if (cell->face(face)->at_boundary() == true)
                  {
                    if (cell->face(face)->center()[0] == -parameters.half_length_x) // -X oriented face
                      cell->face(face)->set_boundary_id(parameters.bid_CC_dirichlet_symm_X); // Dirichlet
                    else if (cell->face(face)->center()[0] == parameters.half_length_x) // +X oriented face
                      cell->face(face)->set_boundary_id(parameters.bid_CC_neumann); // Neumann
                    else if (std::abs(cell->face(face)->center()[2]) == parameters.half_length_z) // -Z/+Z oriented face
                      cell->face(face)->set_boundary_id(parameters.bid_CC_dirichlet_symm_Z); // Dirichlet
                  }
              }
          }

        triangulation.refine_global (1);
      }
    else if (parameters.problem == "BicepsBrachii")
      {
        SphericalManifold<2> manifold_cap;
        Triangulation<2> tria_cap;
        GridGenerator::hyper_ball(tria_cap,
                                  Point<2>(),
                                  parameters.radius_insertion_origin);
        for (typename Triangulation<2>::active_cell_iterator
             cell = tria_cap.begin_active();
             cell != tria_cap.end(); ++cell)
          {
            for (unsigned int face = 0; face < GeometryInfo<2>::faces_per_cell; ++face)
              {
                if (cell->face(face)->at_boundary() == true)
                  cell->face(face)->set_all_manifold_ids(0);
              }
          }
        tria_cap.set_manifold (0, manifold_cap);
        tria_cap.refine_global(parameters.n_refinements_radial);

        Triangulation<2> tria_cap_flat;
        GridGenerator::flatten_triangulation(tria_cap, tria_cap_flat);

        GridGenerator::extrude_triangulation(tria_cap_flat,
                                             parameters.elements_along_axis,
                                             parameters.axial_length,
                                             triangulation);

        struct GridRotate
        {
          Point<dim> operator() (const Point<dim> &in) const
          {
            static const Tensor<2,dim> rot_mat = Physics::Transformations::Rotations::rotation_matrix_3d(Point<dim>(0,1,0), M_PI/2.0);
            return Point<dim>(rot_mat*in);
          }
        };

        // Rotate grid so that the length is axially
        // coincident and aligned with the X-axis
        GridTools::transform (GridRotate(), triangulation);

        // Deform the grid into something that vaguely
        // resemble's a Biceps Brachii
        GridTools::transform (BicepsGeometry<dim>(parameters.axial_length,
                                                  parameters.radius_insertion_origin,
                                                  parameters.radius_midpoint), triangulation);

        // Set boundary IDs
        typename Triangulation<dim>::active_cell_iterator cell =
          triangulation.begin_active(), endc = triangulation.end();
        for (; cell != endc; ++cell)
          {
            for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
              {
                if (cell->face(face)->at_boundary() == true)
                  {
                    static const double tol =1e-6;
                    if (std::abs(cell->face(face)->center()[0]) < tol) // -X oriented face
                      cell->face(face)->set_boundary_id(parameters.bid_BB_dirichlet_X); // Dirichlet
                    else if (std::abs(cell->face(face)->center()[0] - parameters.axial_length) < tol) // +X oriented face
                      cell->face(face)->set_boundary_id(parameters.bid_BB_neumann); // Neumann
                  }
              }
          }

        // Finally resize the grid
        GridTools::scale (parameters.scale, triangulation);
      }
    else
      AssertThrow(false, ExcNotImplemented());
  }

  // @sect4{LinearMuscleModelProblem::setup_muscle_fibres}

  template <int dim>
  void LinearMuscleModelProblem<dim>::setup_muscle_fibres ()
  {
    fibre_data.clear();
    const unsigned int n_cells = triangulation.n_active_cells();
    fibre_data.resize(n_cells);
    const unsigned int n_q_points_cell = qf_cell.size();

    if (parameters.problem == "IsotonicContraction")
      {
        MuscleFibre<dim> fibre_template (Tensor<1,dim>({1,0,0}));

        for (unsigned int cell_no=0; cell_no<triangulation.n_active_cells(); ++cell_no)
          {
            fibre_data[cell_no].resize(n_q_points_cell);
            for (unsigned int q_point_cell=0; q_point_cell<n_q_points_cell; ++q_point_cell)
              {
                fibre_data[cell_no][q_point_cell] = fibre_template;
              }
          }
      }
    else if (parameters.problem == "BicepsBrachii")
      {
        FEValues<dim> fe_values (fe, qf_cell, update_quadrature_points);
        BicepsGeometry<dim> bicep_geom (parameters.axial_length,
                                        parameters.radius_insertion_origin,
                                        parameters.radius_midpoint);

        unsigned int cell_no = 0;
        for (typename Triangulation<dim>::active_cell_iterator
             cell = triangulation.begin_active();
             cell != triangulation.end();
             ++cell, ++cell_no)
          {
            Assert(cell_no<fibre_data.size(), ExcMessage("Trying to access fibre data not stored for this cell index"));
            fe_values.reinit(cell);

            fibre_data[cell_no].resize(n_q_points_cell);
            for (unsigned int q_point_cell=0; q_point_cell<n_q_points_cell; ++q_point_cell)
              {
                const Point<dim> pt = fe_values.get_quadrature_points()[q_point_cell];
                fibre_data[cell_no][q_point_cell] = MuscleFibre<dim>(bicep_geom.direction(pt,parameters.scale));
              }
          }
      }
    else
      AssertThrow(false, ExcNotImplemented());
  }

  // @sect4{LinearMuscleModelProblem::update_fibre_state}

  template <int dim>
  double LinearMuscleModelProblem<dim>::get_neural_signal (const double time)
  {
    // Note: 40 times less force generated than Martins2006
    // This is necessary due to the (compliant) linear tissue model
    return (time > parameters.neural_signal_start_time && time < parameters.neural_signal_end_time ?
            1.0/40.0 :
            0.0);
  }

  template <int dim>
  void LinearMuscleModelProblem<dim>::update_fibre_activation (const double time)
  {
    const double u = get_neural_signal(time);

    const unsigned int n_q_points_cell = qf_cell.size();
    for (unsigned int cell=0; cell<triangulation.n_active_cells(); ++cell)
      {
        for (unsigned int q_point_cell=0; q_point_cell<n_q_points_cell; ++q_point_cell)
          {
            MuscleFibre<dim> &fibre = fibre_data[cell][q_point_cell];
            fibre.update_alpha(u,dt);
          }
      }
  }

  template <int dim>
  void LinearMuscleModelProblem<dim>::update_fibre_state ()
  {
    const unsigned int n_q_points_cell = qf_cell.size();

    FEValues<dim> fe_values (fe, qf_cell, update_gradients);

    // Displacement gradient
    std::vector< std::vector< Tensor<1,dim> > > u_grads (n_q_points_cell,
                                                         std::vector<Tensor<1,dim> >(dim));

    unsigned int cell_no = 0;
    for (typename DoFHandler<dim>::active_cell_iterator
         cell = dof_handler.begin_active();
         cell!=dof_handler.end(); ++cell, ++cell_no)
      {
        Assert(cell_no<fibre_data.size(), ExcMessage("Trying to access fibre data not stored for this cell index"));
        fe_values.reinit(cell);
        fe_values.get_function_gradients (solution, u_grads);

        for (unsigned int q_point_cell=0; q_point_cell<n_q_points_cell; ++q_point_cell)
          {
            Assert(q_point_cell<fibre_data[cell_no].size(), ExcMessage("Trying to access fibre data not stored for this cell and qp index"));

            const SymmetricTensor<2,dim> strain_tensor = get_small_strain (u_grads[q_point_cell]);
            MuscleFibre<dim> &fibre = fibre_data[cell_no][q_point_cell];
            fibre.update_state(strain_tensor, dt);
          }
      }
  }

  // @sect4{LinearMuscleModelProblem::setup_system}

  template <int dim>
  void LinearMuscleModelProblem<dim>::setup_system ()
  {
    dof_handler.distribute_dofs (fe);
    hanging_node_constraints.clear ();
    DoFTools::make_hanging_node_constraints (dof_handler,
                                             hanging_node_constraints);
    hanging_node_constraints.close ();
    sparsity_pattern.reinit (dof_handler.n_dofs(),
                             dof_handler.n_dofs(),
                             dof_handler.max_couplings_between_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);

    hanging_node_constraints.condense (sparsity_pattern);

    sparsity_pattern.compress();

    system_matrix.reinit (sparsity_pattern);

    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());

    std::cout << "   Number of active cells:       "
              << triangulation.n_active_cells()
              << std::endl;

    std::cout << "   Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;
  }

  // @sect4{LinearMuscleModelProblem::assemble_system}

  template <int dim>
  SymmetricTensor<4,dim>
  LinearMuscleModelProblem<dim>::get_stiffness_tensor (const unsigned int cell,
                                                       const unsigned int q_point_cell) const
  {
    static const SymmetricTensor<2,dim> I = unit_symmetric_tensor<dim>();

    Assert(cell<fibre_data.size(), ExcMessage("Trying to access fibre data not stored for this cell index"));
    Assert(q_point_cell<fibre_data[cell].size(), ExcMessage("Trying to access fibre data not stored for this cell and qp index"));
    const MuscleFibre<dim> &fibre = fibre_data[cell][q_point_cell];

    // Matrix
    const double lambda = MuscleMatrix::lambda;
    const double mu = MuscleMatrix::mu;
    // Fibre
    const double m_p = fibre.get_m_p();
    const double m_s = fibre.get_m_s();
    const double beta = fibre.get_beta(dt);
    AssertThrow(beta != 0.0, ExcInternalError());
    const double Cf = T0*(m_p + m_s*(1.0 - m_s/beta));
    const Tensor<1,dim> &M = fibre.get_M();

    SymmetricTensor<4,dim> C;
    for (unsigned int i=0; i < dim; ++i)
      for (unsigned int j=i; j < dim; ++j)
        for (unsigned int k=0; k < dim; ++k)
          for (unsigned int l=k; l < dim; ++l)
            {
              // Matrix contribution
              C[i][j][k][l] = lambda * I[i][j]*I[k][l]
                              + mu * (I[i][k]*I[j][l] + I[i][l]*I[j][k]);

              // Fibre contribution (Passive + active branches)
              C[i][j][k][l] += Cf * M[i]*M[j]*M[k]*M[l];
            }

    return C;
  }

  template <int dim>
  SymmetricTensor<2,dim>
  LinearMuscleModelProblem<dim>::get_rhs_tensor (const unsigned int cell,
                                                 const unsigned int q_point_cell) const
  {
    Assert(cell<fibre_data.size(), ExcMessage("Trying to access fibre data not stored for this cell index"));
    Assert(q_point_cell<fibre_data[cell].size(), ExcMessage("Trying to access fibre data not stored for this cell and qp index"));
    const MuscleFibre<dim> &fibre = fibre_data[cell][q_point_cell];

    const double m_s = fibre.get_m_s();
    const double beta = fibre.get_beta(dt);
    const double gamma = fibre.get_gamma(dt);
    AssertThrow(beta != 0.0, ExcInternalError());
    const double Sf = T0*(m_s*gamma/beta);
    const Tensor<1,dim> &M = fibre.get_M();

    SymmetricTensor<2,dim> S;
    for (unsigned int i=0; i < dim; ++i)
      for (unsigned int j=i; j < dim; ++j)
        {
          // Fibre contribution (Active branch)
          S[i][j] = Sf * M[i]*M[j];
        }

    return S;
  }

  // @sect4{LinearMuscleModelProblem::assemble_system}

  template <int dim>
  void LinearMuscleModelProblem<dim>::assemble_system (const double time)
  {
    // Reset system
    system_matrix = 0;
    system_rhs = 0;

    FEValues<dim> fe_values (fe, qf_cell,
                             update_values | update_gradients |
                             update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_face_values (fe, qf_face,
                                      update_values |
                                      update_quadrature_points | update_JxW_values);

    const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int   n_q_points_cell = qf_cell.size();
    const unsigned int   n_q_points_face = qf_face.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    // Loading
    std::vector<Vector<double> > body_force_values (n_q_points_cell,
                                                    Vector<double>(dim));
    std::vector<Vector<double> > traction_values (n_q_points_face,
                                                  Vector<double>(dim));

    unsigned int cell_no = 0;
    for (typename DoFHandler<dim>::active_cell_iterator
         cell = dof_handler.begin_active();
         cell!=dof_handler.end(); ++cell, ++cell_no)
      {
        cell_matrix = 0;
        cell_rhs = 0;

        fe_values.reinit (cell);
        body_force.vector_value_list (fe_values.get_quadrature_points(),
                                      body_force_values);

        for (unsigned int q_point_cell=0; q_point_cell<n_q_points_cell; ++q_point_cell)
          {
            const SymmetricTensor<4,dim> C = get_stiffness_tensor (cell_no, q_point_cell);
            const SymmetricTensor<2,dim> R = get_rhs_tensor(cell_no, q_point_cell);

            for (unsigned int I=0; I<dofs_per_cell; ++I)
              {
                const unsigned int
                component_I = fe.system_to_component_index(I).first;

                for (unsigned int J=0; J<dofs_per_cell; ++J)
                  {
                    const unsigned int
                    component_J = fe.system_to_component_index(J).first;

                    for (unsigned int k=0; k < dim; ++k)
                      for (unsigned int l=0; l < dim; ++l)
                        cell_matrix(I,J)
                        += (fe_values.shape_grad(I,q_point_cell)[k] *
                            C[component_I][k][component_J][l] *
                            fe_values.shape_grad(J,q_point_cell)[l]) *
                           fe_values.JxW(q_point_cell);
                  }
              }

            for (unsigned int I=0; I<dofs_per_cell; ++I)
              {
                const unsigned int
                component_I = fe.system_to_component_index(I).first;

                cell_rhs(I)
                += fe_values.shape_value(I,q_point_cell) *
                   body_force_values[q_point_cell](component_I) *
                   fe_values.JxW(q_point_cell);

                for (unsigned int k=0; k < dim; ++k)
                  cell_rhs(I)
                  += (fe_values.shape_grad(I,q_point_cell)[k] *
                      R[component_I][k]) *
                     fe_values.JxW(q_point_cell);
              }
          }

        for (unsigned int face = 0; face <GeometryInfo<dim>::faces_per_cell; ++face)
          {
            if (cell->face(face)->at_boundary() == true &&
                ((parameters.problem == "IsotonicContraction" &&
                  cell->face(face)->boundary_id() == parameters.bid_CC_neumann) ||
                 (parameters.problem == "BicepsBrachii" &&
                  cell->face(face)->boundary_id() == parameters.bid_BB_neumann)) )
              {
                fe_face_values.reinit(cell, face);
                traction.vector_value_list (fe_face_values.get_quadrature_points(),
                                            traction_values);

                // Scale applied traction according to time
                const double ramp = (time <= t_ramp_end ? time/t_ramp_end : 1.0);
                Assert(ramp >= 0.0 && ramp <= 1.0, ExcMessage("Invalid force ramp"));
                for (unsigned int q_point_face = 0; q_point_face < n_q_points_face; ++q_point_face)
                  traction_values[q_point_face] *= ramp;

                for (unsigned int q_point_face = 0; q_point_face < n_q_points_face; ++q_point_face)
                  {
                    for (unsigned int I=0; I<dofs_per_cell; ++I)
                      {
                        const unsigned int
                        component_I = fe.system_to_component_index(I).first;

                        cell_rhs(I)
                        += fe_face_values.shape_value(I,q_point_face)*
                           traction_values[q_point_face][component_I]*
                           fe_face_values.JxW(q_point_face);
                      }
                  }
              }
          }

        cell->get_dof_indices (local_dof_indices);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              system_matrix.add (local_dof_indices[i],
                                 local_dof_indices[j],
                                 cell_matrix(i,j));

            system_rhs(local_dof_indices[i]) += cell_rhs(i);
          }
      }

    hanging_node_constraints.condense (system_matrix);
    hanging_node_constraints.condense (system_rhs);
  }

  template <int dim>
  void LinearMuscleModelProblem<dim>::apply_boundary_conditions ()
  {
    std::map<types::global_dof_index,double> boundary_values;

    if (parameters.problem == "IsotonicContraction")
      {
        // Symmetry condition on -X faces
        {
          ComponentMask component_mask_x (dim, false);
          component_mask_x.set(0, true);
          VectorTools::interpolate_boundary_values (dof_handler,
                                                    parameters.bid_CC_dirichlet_symm_X,
                                                    ZeroFunction<dim>(dim),
                                                    boundary_values,
                                                    component_mask_x);
        }
        // Symmetry condition on -Z/+Z faces
        {
          ComponentMask component_mask_z (dim, false);
          component_mask_z.set(2, true);
          VectorTools::interpolate_boundary_values (dof_handler,
                                                    parameters.bid_CC_dirichlet_symm_Z,
                                                    ZeroFunction<dim>(dim),
                                                    boundary_values,
                                                    component_mask_z);
        }
        // Fixed point on -X face
        {
          const Point<dim> fixed_point (-parameters.half_length_x,0.0,0.0);
          std::vector<types::global_dof_index> fixed_dof_indices;
          bool found_point_of_interest = false;

          for (typename DoFHandler<dim>::active_cell_iterator
               cell = dof_handler.begin_active(),
               endc = dof_handler.end(); cell != endc; ++cell)
            {
              for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
                {
                  // We know that the fixed point is on the -X Dirichlet boundary
                  if (cell->face(face)->at_boundary() == true &&
                      cell->face(face)->boundary_id() == parameters.bid_CC_dirichlet_symm_X)
                    {
                      for (unsigned int face_vertex_index = 0; face_vertex_index < GeometryInfo<dim>::vertices_per_face; ++face_vertex_index)
                        {
                          if (cell->face(face)->vertex(face_vertex_index).distance(fixed_point) < 1e-6)
                            {
                              found_point_of_interest = true;
                              for (unsigned int index_component = 0; index_component < dim; ++index_component)
                                fixed_dof_indices.push_back(cell->face(face)->vertex_dof_index(face_vertex_index,
                                                            index_component));
                            }

                          if (found_point_of_interest == true) break;
                        }
                    }
                  if (found_point_of_interest == true) break;
                }
              if (found_point_of_interest == true) break;
            }

          Assert(found_point_of_interest == true, ExcMessage("Didn't find point of interest"));
          AssertThrow(fixed_dof_indices.size() == dim, ExcMessage("Didn't find the correct number of DoFs to fix"));

          for (unsigned int i=0; i < fixed_dof_indices.size(); ++i)
            boundary_values[fixed_dof_indices[i]] = 0.0;
        }
      }
    else if (parameters.problem == "BicepsBrachii")
      {
        if (parameters.include_gravity == false)
          {
            // Symmetry condition on -X surface
            {
              ComponentMask component_mask_x (dim, false);
              component_mask_x.set(0, true);
              VectorTools::interpolate_boundary_values (dof_handler,
                                                        parameters.bid_BB_dirichlet_X,
                                                        ZeroFunction<dim>(dim),
                                                        boundary_values,
                                                        component_mask_x);
            }

            // Fixed central point on -X surface
            {
              const Point<dim> fixed_point (0.0,0.0,0.0);
              std::vector<types::global_dof_index> fixed_dof_indices;
              bool found_point_of_interest = false;

              for (typename DoFHandler<dim>::active_cell_iterator
                   cell = dof_handler.begin_active(),
                   endc = dof_handler.end(); cell != endc; ++cell)
                {
                  for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
                    {
                      // We know that the fixed point is on the -X Dirichlet boundary
                      if (cell->face(face)->at_boundary() == true &&
                          cell->face(face)->boundary_id() == parameters.bid_BB_dirichlet_X)
                        {
                          for (unsigned int face_vertex_index = 0; face_vertex_index < GeometryInfo<dim>::vertices_per_face; ++face_vertex_index)
                            {
                              if (cell->face(face)->vertex(face_vertex_index).distance(fixed_point) < 1e-6)
                                {
                                  found_point_of_interest = true;
                                  for (unsigned int index_component = 0; index_component < dim; ++index_component)
                                    fixed_dof_indices.push_back(cell->face(face)->vertex_dof_index(face_vertex_index,
                                                                index_component));
                                }

                              if (found_point_of_interest == true) break;
                            }
                        }
                      if (found_point_of_interest == true) break;
                    }
                  if (found_point_of_interest == true) break;
                }

              Assert(found_point_of_interest == true, ExcMessage("Didn't find point of interest"));
              AssertThrow(fixed_dof_indices.size() == dim, ExcMessage("Didn't find the correct number of DoFs to fix"));

              for (unsigned int i=0; i < fixed_dof_indices.size(); ++i)
                boundary_values[fixed_dof_indices[i]] = 0.0;
            }
          }
        else
          {
            // When we apply gravity, some additional constraints
            // are required to support the load of the muscle, as
            // the material response is more compliant than would
            // be the case in reality.

            // Symmetry condition on -X surface
            {
              ComponentMask component_mask_x (dim, true);
              VectorTools::interpolate_boundary_values (dof_handler,
                                                        parameters.bid_BB_dirichlet_X,
                                                        ZeroFunction<dim>(dim),
                                                        boundary_values,
                                                        component_mask_x);
            }
            // Symmetry condition on -X surface
            {
              ComponentMask component_mask_x (dim, false);
              component_mask_x.set(1, true);
              component_mask_x.set(2, true);
              VectorTools::interpolate_boundary_values (dof_handler,
                                                        parameters.bid_BB_neumann,
                                                        ZeroFunction<dim>(dim),
                                                        boundary_values,
                                                        component_mask_x);
            }
          }

        // Roller condition at central point on +X face
        {
          const Point<dim> roller_point (parameters.axial_length*parameters.scale,0.0,0.0);
          std::vector<types::global_dof_index> fixed_dof_indices;
          bool found_point_of_interest = false;

          for (typename DoFHandler<dim>::active_cell_iterator
               cell = dof_handler.begin_active(),
               endc = dof_handler.end(); cell != endc; ++cell)
            {
              for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
                {
                  // We know that the fixed point is on the +X Neumann boundary
                  if (cell->face(face)->at_boundary() == true &&
                      cell->face(face)->boundary_id() == parameters.bid_BB_neumann)
                    {
                      for (unsigned int face_vertex_index = 0; face_vertex_index < GeometryInfo<dim>::vertices_per_face; ++face_vertex_index)
                        {
                          if (cell->face(face)->vertex(face_vertex_index).distance(roller_point) < 1e-6)
                            {
                              found_point_of_interest = true;
                              for (unsigned int index_component = 1; index_component < dim; ++index_component)
                                fixed_dof_indices.push_back(cell->face(face)->vertex_dof_index(face_vertex_index,
                                                            index_component));
                            }

                          if (found_point_of_interest == true) break;
                        }
                    }
                  if (found_point_of_interest == true) break;
                }
              if (found_point_of_interest == true) break;
            }

          Assert(found_point_of_interest == true, ExcMessage("Didn't find point of interest"));
          AssertThrow(fixed_dof_indices.size() == dim-1, ExcMessage("Didn't find the correct number of DoFs to fix"));

          for (unsigned int i=0; i < fixed_dof_indices.size(); ++i)
            boundary_values[fixed_dof_indices[i]] = 0.0;
        }
      }
    else
      AssertThrow(false, ExcNotImplemented());

    MatrixTools::apply_boundary_values (boundary_values,
                                        system_matrix,
                                        solution,
                                        system_rhs);
  }


  // @sect4{LinearMuscleModelProblem::solve}

  template <int dim>
  void LinearMuscleModelProblem<dim>::solve ()
  {
    SolverControl solver_control (system_matrix.m(), 1e-12);
    SolverCG<>    cg (solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    cg.solve (system_matrix, solution, system_rhs,
              preconditioner);

    hanging_node_constraints.distribute (solution);
  }


  // @sect4{LinearMuscleModelProblem::output_results}


  template <int dim>
  void LinearMuscleModelProblem<dim>::output_results (const unsigned int timestep,
                                                      const double time) const
  {
    // Visual output: FEM results
    {
      std::string filename = "solution-";
      filename += Utilities::int_to_string(timestep,4);
      filename += ".vtk";
      std::ofstream output (filename.c_str());

      DataOut<dim> data_out;
      data_out.attach_dof_handler (dof_handler);

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(dim,
                                    DataComponentInterpretation::component_is_part_of_vector);
      std::vector<std::string> solution_name(dim, "displacement");

      data_out.add_data_vector (solution, solution_name,
                                DataOut<dim>::type_dof_data,
                                data_component_interpretation);
      data_out.build_patches ();
      data_out.write_vtk (output);
    }

    // Visual output: FEM data
    {
      std::string filename = "fibres-";
      filename += Utilities::int_to_string(timestep,4);
      filename += ".vtk";
      std::ofstream output (filename.c_str());

      output
          << "# vtk DataFile Version 3.0" << std::endl
          << "# " << std::endl
          << "ASCII"<< std::endl
          << "DATASET POLYDATA"<< std::endl << std::endl;

      // Extract fibre data from quadrature points
      const unsigned int n_cells    = triangulation.n_active_cells();
      const unsigned int n_q_points_cell = qf_cell.size();

      // Data that we'll be outputting
      std::vector<std::string> results_fibre_names;
      results_fibre_names.push_back("alpha");
      results_fibre_names.push_back("epsilon_f");
      results_fibre_names.push_back("epsilon_c");
      results_fibre_names.push_back("epsilon_c_dot");

      const unsigned int n_results = results_fibre_names.size();
      const unsigned int n_data_points = n_cells*n_q_points_cell;
      std::vector< Point<dim> > output_points(n_data_points);
      std::vector< Tensor<1,dim> > output_displacements(n_data_points);
      std::vector< Tensor<1,dim> > output_directions(n_data_points);
      std::vector< std::vector<double> > output_values(n_results, std::vector<double>(n_data_points));

      // Displacement
      std::vector< Vector<double> > u_values (n_q_points_cell,
                                              Vector<double>(dim));
      // Displacement gradient
      std::vector< std::vector< Tensor<1,dim> > > u_grads (n_q_points_cell,
                                                           std::vector<Tensor<1,dim> >(dim));

      FEValues<dim> fe_values (fe, qf_cell,
                               update_values | update_gradients | update_quadrature_points);
      unsigned int cell_no = 0;
      unsigned int fibre_no = 0;
      for (typename DoFHandler<dim>::active_cell_iterator
           cell = dof_handler.begin_active();
           cell != dof_handler.end();
           ++cell, ++cell_no)
        {
          fe_values.reinit (cell);
          fe_values.get_function_values (solution, u_values);
          fe_values.get_function_gradients (solution, u_grads);

          for (unsigned int q_point_cell=0; q_point_cell<n_q_points_cell; ++q_point_cell, ++fibre_no)
            {
              const MuscleFibre<dim> &fibre = fibre_data[cell_no][q_point_cell];
              output_points[fibre_no] = fe_values.get_quadrature_points()[q_point_cell]; // Position
              for (unsigned int d=0; d<dim; ++d)
                output_displacements[fibre_no][d] = u_values[q_point_cell][d]; // Displacement
              // Direction (spatial configuration)
              output_directions[fibre_no] = get_deformation_gradient(u_grads[q_point_cell])*fibre.get_M();
              output_directions[fibre_no] /= output_directions[fibre_no].norm();

              // Fibre values
              output_values[0][fibre_no] = fibre.get_alpha();
              output_values[1][fibre_no] = fibre.get_epsilon_f();
              output_values[2][fibre_no] = fibre.get_epsilon_c();
              output_values[3][fibre_no] = fibre.get_epsilon_c_dot();
            }
        }

      // FIBRE POSITION
      output
          << "POINTS "
          << n_data_points
          << " float" << std::endl;
      for (unsigned int i=0; i < n_data_points; ++i)
        {
          for (unsigned int j=0; j < dim; ++j)
            {
              output << (output_points)[i][j] << "\t";
            }
          output << std::endl;
        }

      // HEADER FOR POINT DATA
      output  << "\nPOINT_DATA "
              << n_data_points
              << std::endl << std::endl;

      // FIBRE DISPLACEMENTS
      output
          << "VECTORS displacement float"
          << std::endl;
      for (unsigned int i = 0; i < n_data_points; ++i)
        {
          for (unsigned int j=0; j < dim; ++j)
            {
              output << (output_displacements)[i][j] << "\t";
            }
          output << std::endl;
        }
      output << std::endl;

      // FIBRE DIRECTIONS
      output
          << "VECTORS direction float"
          << std::endl;
      for (unsigned int i = 0; i < n_data_points; ++i)
        {
          for (unsigned int j=0; j < dim; ++j)
            {
              output << (output_directions)[i][j] << "\t";
            }
          output << std::endl;
        }
      output << std::endl;

      // POINT DATA
      for (unsigned int v=0; v < n_results; ++v)
        {
          output
              << "SCALARS  "
              << results_fibre_names[v]
              << "  float 1" << std::endl
              << "LOOKUP_TABLE default "
              << std::endl;
          for (unsigned int i=0; i<n_data_points; ++i)
            {
              output << (output_values)[v][i] << " ";
            }
          output << std::endl;
        }
    }

    // Output X-displacement at measured point
    {
      const Point<dim> meas_pt (parameters.problem == "IsotonicContraction" ?
                                Point<dim>(parameters.half_length_x, 0.0, 0.0) :
                                Point<dim>(parameters.axial_length*parameters.scale, 0.0, 0.0) );


      const unsigned int index_of_interest = 0;
      bool found_point_of_interest = false;
      types::global_dof_index dof_of_interest = numbers::invalid_dof_index;

      for (typename DoFHandler<dim>::active_cell_iterator
           cell = dof_handler.begin_active(),
           endc = dof_handler.end(); cell != endc; ++cell)
        {
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
            {
              // We know that the measurement point is on the Neumann boundary
              if (cell->face(face)->at_boundary() == true &&
                  ((parameters.problem == "IsotonicContraction" &&
                    cell->face(face)->boundary_id() == parameters.bid_CC_neumann) ||
                   (parameters.problem == "BicepsBrachii" &&
                    cell->face(face)->boundary_id() == parameters.bid_BB_neumann)) )
                {
                  for (unsigned int face_vertex_index = 0; face_vertex_index < GeometryInfo<dim>::vertices_per_face; ++face_vertex_index)
                    {
                      if (cell->face(face)->vertex(face_vertex_index).distance(meas_pt) < 1e-6)
                        {
                          found_point_of_interest = true;
                          dof_of_interest = cell->face(face)->vertex_dof_index(face_vertex_index,
                                                                               index_of_interest);
                        }

                      if (found_point_of_interest == true) break;
                    }
                }
              if (found_point_of_interest == true) break;
            }
          if (found_point_of_interest == true) break;
        }

      Assert(found_point_of_interest == true, ExcMessage("Didn't find point of interest"));
      Assert(dof_of_interest != numbers::invalid_dof_index, ExcMessage("Didn't find DoF of interest"));
      Assert(dof_of_interest < dof_handler.n_dofs(), ExcMessage("DoF index out of range"));

      const std::string filename = "displacement_POI.csv";
      std::ofstream output;
      if (timestep == 0)
        {
          output.open(filename.c_str(), std::ofstream::out);
          output
              << "Time [s]" << "," << "X-displacement [mm]" << std::endl;
        }
      else
        output.open(filename.c_str(), std::ios_base::app);

      output
          << time
          << ","
          << solution[dof_of_interest]*1e3
          << std::endl;
    }
  }



  // @sect4{LinearMuscleModelProblem::run}

  template <int dim>
  void LinearMuscleModelProblem<dim>::run ()
  {
    make_grid();
    setup_system ();
    setup_muscle_fibres ();

//    const bool do_grid_refinement = false;
    double time = 0.0;
    for (unsigned int timestep=0; time<=t_end; ++timestep, time+=dt)
      {
        std::cout
            << "Timestep " << timestep
            << " @ time " << time
            << std::endl;

        // First we update the fibre activation level
        // based on the current time
        update_fibre_activation(time);

        // Next we assemble the system and enforce boundary
        // conditions.
        // Here we assume that the system and fibres have
        // a fixed state, and we will assemble based on how
        // epsilon_c will update given the current state of
        // the body.
        assemble_system (time);
        apply_boundary_conditions ();

        // Then we solve the linear system
        solve ();

        // Now we update the fibre state based on the new
        // displacement solution and the constitutive
        // parameters assumed to govern the stiffness of
        // the fibres at the previous state. i.e. We
        // follow through with assumed update conditions
        // used in the assembly phase.
        update_fibre_state();

        // Output some values to file
        output_results (timestep, time);
      }
  }
}

// @sect3{The <code>main</code> function}

int main ()
{
  try
    {
      dealii::deallog.depth_console (0);
      const unsigned int dim = 3;

      LMM::LinearMuscleModelProblem<dim> lmm_problem ("parameters.prm");
      lmm_problem.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
