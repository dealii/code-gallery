/* ---------------------------------------------------------------------
 * Copyright (C) 2018 by the deal.II authors and
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
 * Authors: Jean-Paul Pelteret
 *          University of Erlangen-Nuremberg, 2018
 */

// @sect3{Include files}

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
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
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

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

    // @sect4{Boundary conditions}

    // The boundary conditions to be applied to the problem.

        struct BoundaryConditions
        {
          double final_stretch;

          static void
          declare_parameters(ParameterHandler &prm);

          void
          parse_parameters(ParameterHandler &prm);
        };

        void BoundaryConditions::declare_parameters(ParameterHandler &prm)
        {
          prm.enter_subsection("Boundary conditions");
          {
            prm.declare_entry("Final stretch", "2.0",
                              Patterns::Double(0),
                              "The amount by which the specimen is to be stretched");
          }
          prm.leave_subsection();
        }

        void BoundaryConditions::parse_parameters(ParameterHandler &prm)
        {
          prm.enter_subsection("Boundary conditions");
          {
            final_stretch = prm.get_double("Final stretch");
          }
          prm.leave_subsection();
        }

// @sect4{Geometry}

// Make adjustments to the geometry of the notched specimen.

    struct Geometry
    {
      std::string  geometry_type;
      double       length;
      double       radius;
      double       notch_length;
      double       notch_radius;
      double       scale;
      unsigned int n_global_refinement_steps;
      unsigned int n_local_refinement_steps;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void Geometry::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        prm.declare_entry("Geometry type", "Notched cylinder",
                          Patterns::Selection("Notched cylinder|Notched tensile specimen"),
                          "The geometry to be modelled");

        prm.declare_entry("Length", "100",
                          Patterns::Double(0),
                          "Overall length of the specimen");

        prm.declare_entry("Radius", "10",
                          Patterns::Double(0),
                          "Overall radius of the specimen");

        prm.declare_entry("Notch length", "10",
                          Patterns::Double(0),
                          "Overall length of the notch in the specimen");

        prm.declare_entry("Notch radius", "1",
                          Patterns::Double(0),
                          "Overall radius of the notch in the specimen");

        prm.declare_entry("Grid scale", "1e-3",
                          Patterns::Double(0),
                          "Global grid scaling factor");

        prm.declare_entry("Global refinement steps", "1",
                          Patterns::Integer(0),
                          "Number of global refinement steps");

        prm.declare_entry("Local refinement steps", "1",
                          Patterns::Integer(0),
                          "Number of initial local refinement cycles");
      }
      prm.leave_subsection();
    }

    void Geometry::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        geometry_type = prm.get("Geometry type");
        length = prm.get_double("Length");
        radius = prm.get_double("Radius");
        notch_length = prm.get_double("Notch length");
        notch_radius = prm.get_double("Notch radius");
        scale = prm.get_double("Grid scale");
        n_global_refinement_steps = prm.get_integer("Global refinement steps");
        n_local_refinement_steps = prm.get_integer("Local refinement steps");
      }
      prm.leave_subsection();

      AssertThrow(length > notch_length,
                  ExcMessage("Cannot create geometry with given dimensions"));
      AssertThrow(radius > notch_radius,
                  ExcMessage("Cannot create geometry with given dimensions"));
    }

    // @sect4{Material properties}

    // Make adjustments to the material of the notched specimen.

        struct MaterialProperties
        {
          std::string mat_type;
          double mu_e; // Shear modulus
          double nu_e; // Poisson ratio
          double k_p; // Isotropic hardening constant
          double sigma_y_p; // Yield stress

          static void
          declare_parameters(ParameterHandler &prm);

          void
          parse_parameters(ParameterHandler &prm);
        };

        void MaterialProperties::declare_parameters(ParameterHandler &prm)
        {
          prm.enter_subsection("Material properties");
          {
            prm.declare_entry("Type", "Elastic",
                              Patterns::Selection("Elastic|Elastoplastic (isotropic hardening)"),
                              "Type of material that composes the specimen");

            prm.declare_entry("Shear modulus", "100",
                              Patterns::Double(0),
                              "Shear modulus of the specimen");

            prm.declare_entry("Poisson ratio", "0.3",
                              Patterns::Double(-1,0.5),
                              "Poisson ratio of the specimen");

            prm.declare_entry("Isotropic hardening constant", "10",
                              Patterns::Double(0),
                              "Isotropic hardening constant");

            prm.declare_entry("Yield stress", "20",
                              Patterns::Double(0),
                              "Yield stress");
          }
          prm.leave_subsection();
        }

        void MaterialProperties::parse_parameters(ParameterHandler &prm)
        {
          prm.enter_subsection("Material properties");
          {
            mat_type = prm.get("Type");
            mu_e = prm.get_double("Shear modulus");
            nu_e = prm.get_double("Poisson ratio");
            k_p = prm.get_double("Isotropic hardening constant");
            sigma_y_p = prm.get_double("Yield stress");
          }
          prm.leave_subsection();
        }



// @sect4{Time}

// Set the timestep size $ \varDelta t $ and the simulation end-time.
    struct Time
    {
      double delta_t;
      double end_time;

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
        delta_t = prm.get_double("Time step size");
      }
      prm.leave_subsection();
    }

// @sect4{All parameters}

// Finally we consolidate all of the above structures into a single container
// that holds all of our run-time selections.
    struct AllParameters : public FESystem,
      public BoundaryConditions,
      public Geometry,
      public MaterialProperties,
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
      BoundaryConditions::declare_parameters(prm);
      Geometry::declare_parameters(prm);
      MaterialProperties::declare_parameters(prm);
      Time::declare_parameters(prm);
    }

    void AllParameters::parse_parameters(ParameterHandler &prm)
    {
      FESystem::parse_parameters(prm);
      BoundaryConditions::parse_parameters(prm);
      Geometry::parse_parameters(prm);
      MaterialProperties::parse_parameters(prm);
      Time::parse_parameters(prm);
    }
  }

  // @sect3{Time}

  class Time
  {
  public:
    Time (const double time_end,
          const double delta_t)
      :
      timestep(0),
      time_current(0.0),
      time_end(time_end),
      delta_t(delta_t)
    {}
    virtual ~Time()
    {}
    double current() const
    {
      return time_current;
    }
    double end() const
    {
      return time_end;
    }
    double get_delta_t() const
    {
      return delta_t;
    }
    unsigned int get_timestep() const
    {
      return timestep;
    }
    void increment()
    {
      time_current += delta_t;
      ++timestep;
    }
  private:
    unsigned int timestep;
    double       time_current;
    const double time_end;
    const double delta_t;
};


  template <int dim>
  class Material_Base
  {
  public:
    Material_Base()
    {}

    virtual ~Material_Base()
    {}

    // Cauchy stress
    virtual SymmetricTensor<2,dim>
    get_sigma(const SymmetricTensor<2, dim> &epsilon) const = 0;

    // Elastic / Elastoplastic tangent
    virtual SymmetricTensor<4,dim>
    get_K(const SymmetricTensor<2, dim> &epsilon) const = 0;

    virtual void
    update_internal_equilibrium(const SymmetricTensor<2, dim> &epsilon) = 0;

    virtual void
    update_end_timestep() = 0;
  };


  /**
   * Linear elastic material
   */
  template <int dim>
  class Material_Linear_Elastic : public Material_Base<dim>
  {
  public:
    Material_Linear_Elastic(const double mu_e,
                            const double nu_e)
      :
      kappa_e((2.0 * mu_e * (1.0 + nu_e)) / (3.0 * (1.0 - 2.0 * nu_e))),
      mu_e(mu_e)
    {
      Assert(kappa_e > 0, ExcInternalError());
    }

    virtual ~Material_Linear_Elastic()
    {}

    // Cauchy stress
    SymmetricTensor<2,dim>
    get_sigma(const SymmetricTensor<2, dim> &epsilon) const override
    {
      // Assume isochoric/deviatoric split
      const SymmetricTensor<2, dim> epsilon_vol = (trace(epsilon)/dim)*Physics::Elasticity::StandardTensors<dim>::I;
      const SymmetricTensor<2, dim> epsilon_dev = epsilon - epsilon_vol;

      return dim*kappa_e*epsilon_vol
          + 2.0*mu_e*epsilon_dev;
    }

    // Elastic / Elastoplastic tangent
    SymmetricTensor<4,dim> get_K(const SymmetricTensor<2, dim> &epsilon) const override
    {
      (void)epsilon;

      // Elastic part of tangent, assuming an isochoric/deviatoric split
      const SymmetricTensor<4,dim> K_e
        = (kappa_e - 2.0/dim*mu_e)*Physics::Elasticity::StandardTensors<dim>::IxI
        + (2.0*mu_e)*Physics::Elasticity::StandardTensors<dim>::S;

      // Check that stress-strain relationship really is linear and
      // correctly implemented
      Assert((get_sigma(epsilon) - K_e*epsilon).norm() < 1e-9, ExcInternalError());

      return K_e;
    }

    void
    update_internal_equilibrium(const SymmetricTensor<2, dim> &epsilon)  override
    {
      (void)epsilon;
    }

    void
    update_end_timestep() override
    {}

  protected:
    const double kappa_e; // buld modulus
    const double mu_e; // shear modulus
  };


  /**
   * Linear elastoplastic material: von Mises plasticity with linear isotropic hardening
   */
  template <int dim>
  class Material_Linear_Elastoplastic_Isotropic_Hardening : public Material_Base<dim>
  {
  public:
    Material_Linear_Elastoplastic_Isotropic_Hardening(const double mu_e,
                                  const double nu_e,
                                  const double k_p,
                                  const double sigma_y_p,
                                  const Time  &time)
      :
      kappa_e((2.0 * mu_e * (1.0 + nu_e)) / (3.0 * (1.0 - 2.0 * nu_e))),
      mu_e(mu_e),
      k_p (k_p),
      sigma_y_p (sigma_y_p),
      time(time),
      phi_t(std::numeric_limits<double>::min()),
      n_t(),
      delta_lambda_t(0.0),
      sigma_dev_trial_norm(1.0),
      epsilon_p_t(),
      epsilon_p_t1(),
      alpha_p_t (0.0),
      alpha_p_t1 (0.0)
    {
      Assert(kappa_e > 0, ExcInternalError());
    }

    virtual ~Material_Linear_Elastoplastic_Isotropic_Hardening()
    {}

    // Cauchy stress
    SymmetricTensor<2,dim>
    get_sigma(const SymmetricTensor<2, dim> &epsilon) const override
    {
      return get_sigma(epsilon, epsilon_p_t);
    }

    // Isotropic hardening stress
    double
    get_R() const
    {
      return get_R(alpha_p_t);
    }

    // Elastic / Elastoplastic tangent
    SymmetricTensor<4,dim>
    get_K(const SymmetricTensor<2, dim> &epsilon) const override
    {
      (void)epsilon;

      // Elastic part of tangent, assuming an isochoric/deviatoric split
      const SymmetricTensor<4,dim> K_e
        = (kappa_e - 2.0/dim*mu_e)*Physics::Elasticity::StandardTensors<dim>::IxI
        + (2.0*mu_e)*Physics::Elasticity::StandardTensors<dim>::S;

      if (phi_t <= 0)
      {
        // Check that stress-strain relationship really is linear and
        // correctly implemented
        Assert((get_sigma(epsilon) - K_e*epsilon).norm() < 1e-9, ExcInternalError());

        return K_e;
      }
      else
      {
        // Compute correction for radial projection during plastic flow
        static const SymmetricTensor<4,dim> I_dev
          = Physics::Elasticity::StandardTensors<dim>::S
          - (1.0/dim)*Physics::Elasticity::StandardTensors<dim>::IxI;
        const SymmetricTensor<4,dim> n_t_x_n_t = outer_product(n_t,n_t);

        Assert(sigma_dev_trial_norm != 0.0, ExcInternalError());
        const SymmetricTensor<4,dim> K_p_correction
          = - (Utilities::fixed_power<2>(2.0*mu_e))/(2.0*mu_e + 2.0/dim*kappa_e)*n_t_x_n_t
          - (Utilities::fixed_power<2>(2.0*mu_e)*delta_lambda_t)/(sigma_dev_trial_norm)*(I_dev - n_t_x_n_t);

        return K_e + K_p_correction;
      }
    }

    void
    update_internal_equilibrium(const SymmetricTensor<2, dim> &epsilon) override
    {
      // Trial stress
      const SymmetricTensor<2, dim> sigma_trial = get_sigma(epsilon,epsilon_p_t1);
      // Deviatoric part of trial stress
      const SymmetricTensor<2, dim> sigma_dev_trial = deviator(sigma_trial);
      // Trial hardening stress
      const double R_trial = get_R(alpha_p_t1);

      // Update yield function value (von Mises plasticity for linear
      // isotropic hardening)
      {
        phi_t = sigma_dev_trial.norm() - std::sqrt(2.0/dim)*(sigma_y_p - R_trial);

        // Update internal variables based condition of
        // elasticity or plastic flow
        if (phi_t <= 0.0)
        {
          // Elastic update
          epsilon_p_t = epsilon_p_t1;
          alpha_p_t = alpha_p_t1;
        }
        else
        {
          // Plastic update:
          // Linear isotropic hardening -> use radial return algorithm

          // Internal variables for radial return algorithm
          sigma_dev_trial_norm = sigma_dev_trial.norm();
          n_t = sigma_dev_trial / sigma_dev_trial_norm;
          delta_lambda_t = phi_t / (2.0*mu_e + 2.0/dim*kappa_e);

          // Update internal plastic variables
          epsilon_p_t = epsilon_p_t1 + delta_lambda_t*n_t;
          alpha_p_t = alpha_p_t1 + delta_lambda_t*std::sqrt(2.0/dim);
        }
      }
    }

    void
    update_end_timestep() override
    {
      // For the linear problem, it is easier to do the update
      // between the previous and the current timestep in the
      // call to update_internal_equilibrium(). For a nonlinear
      // problem, it is more useful to employ this function to
      // accept the accumulation of incremental values in the
      // context of a Newton scheme.
    }

  protected:
    const double kappa_e; // buld modulus
    const double mu_e; // shear modulus
    const double k_p; // isotropic hardening modulus
    const double sigma_y_p; // yield stress

    const Time  &time;

    // Variables related to yield function at current timestep
    double phi_t; // yield function
    SymmetricTensor<2,dim> n_t; // radial return direction
    double delta_lambda_t;// increment in plastic multiplier
    double sigma_dev_trial_norm; // norm of deviatoric part of trial stress

    // Internal variables:
    // Plastic strain
    SymmetricTensor<2,dim> epsilon_p_t;  // current timestep
    SymmetricTensor<2,dim> epsilon_p_t1;  // previous timestep
    // Isotropic hardening variable
    double alpha_p_t;  // current timestep
    double alpha_p_t1; // previous timestep

    SymmetricTensor<2,dim>
    get_sigma(const SymmetricTensor<2, dim> &epsilon,
        const SymmetricTensor<2, dim> &epsilon_p) const
    {
      // Assume isochoric/deviatoric split
      const SymmetricTensor<2, dim> epsilon_vol = (trace(epsilon)/dim)*Physics::Elasticity::StandardTensors<dim>::I;
      const SymmetricTensor<2, dim> epsilon_dev = epsilon - epsilon_vol;

      const SymmetricTensor<2,dim> sigma
      = dim*kappa_e*epsilon_vol
      + 2.0*mu_e*(epsilon_dev - epsilon_p);
      return sigma;
    }

    double
    get_R(const double alpha_p) const
    {
      return -k_p*alpha_p;
    }
  };


  template <int dim>
  class PointHistory
  {
  public:
    PointHistory()
    {}
    virtual ~PointHistory()
    {}
    void
    setup_lqp (const types::material_id         mat_id,
               const Parameters::AllParameters &parameters,
               const Time                      &time)
    {
      if (mat_id == 0)
      {
        material.reset(new Material_Linear_Elastic<dim>(
                         parameters.mu_e, parameters.nu_e));
      }
      else if (mat_id == 1)
      {
        material.reset(new Material_Linear_Elastoplastic_Isotropic_Hardening<dim>(
                         parameters.mu_e, parameters.nu_e,
                         parameters.k_p, parameters.sigma_y_p,
                         time));
      }
      else
        AssertThrow(mat_id <= 1, ExcMessage("Unknown material id"));
    }

    SymmetricTensor<2, dim>
    get_sigma(const SymmetricTensor<2, dim> &epsilon) const
    {
      return material->get_sigma(epsilon);
    }
    SymmetricTensor<4, dim>
    get_K(const SymmetricTensor<2, dim> &epsilon) const
    {
      return material->get_K(epsilon);
    }
    void
    update_internal_equilibrium(const SymmetricTensor<2, dim> &epsilon)
    {
      material->update_internal_equilibrium(epsilon);
    }
    void
    update_end_timestep()
    {
      material->update_end_timestep();
    }
  private:
    std::shared_ptr< Material_Base<dim> > material;
};


  // @sect3{The <code>LinearElastoplasticProblem</code> class template}

  template <int dim>
  class LinearElastoplasticProblem
  {
  public:
    LinearElastoplasticProblem (const std::string &input_file);
    ~LinearElastoplasticProblem ();
    void run ();

  private:
    void make_grid ();
    void setup_system ();
    void setup_qph();
    void assemble_system ();
    void apply_boundary_conditions ();
    void solve ();
    void output_results () const;

    Parameters::AllParameters parameters;

    Triangulation<dim>   triangulation;
    DoFHandler<dim>      dof_handler;

    Time time;

    FESystem<dim>        fe;
    QGauss<dim>          qf_cell;
    QGauss<dim-1>        qf_face;

    AffineConstraints<double>     hanging_node_constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double>       solution;
    Vector<double>       system_rhs;

    // Local data
    CellDataStorage<typename Triangulation<dim>::cell_iterator,
    PointHistory<dim> > quadrature_point_history;

    const FEValuesExtractors::Vector u_fe;
    static const unsigned int n_components = dim;
    static const unsigned int first_u_component = 0;
  };

  // @sect4{LinearElastoplasticProblem::LinearElastoplasticProblem}

  template <int dim>
  LinearElastoplasticProblem<dim>::LinearElastoplasticProblem (const std::string &input_file)
    :
    parameters(input_file),
    dof_handler (triangulation),
    time (parameters.end_time, parameters.delta_t),
    fe (FE_Q<dim>(parameters.poly_degree), dim),
    qf_cell (parameters.quad_order),
    qf_face (parameters.quad_order),
    u_fe(first_u_component)
  {
    Assert(dim==3, ExcNotImplemented());
  }


  // @sect4{LinearElastoplasticProblem::~LinearElastoplasticProblem}

  template <int dim>
  LinearElastoplasticProblem<dim>::~LinearElastoplasticProblem ()
  {
    dof_handler.clear ();
  }


  // @sect4{LinearElastoplasticProblem::make_grid}

  template <int dim>
  void LinearElastoplasticProblem<dim>::make_grid ()
  {
    Assert (dim == 3, ExcNotImplemented());

    if (parameters.geometry_type == "Notched cylinder")
    {
      Triangulation<2> tria_cross_section;
      {
        const Point<2> centre;
        const double notch_radius_inner = parameters.notch_radius - (parameters.radius-parameters.notch_radius);
        Triangulation<2> tria_inner_1;
        GridGenerator::hyper_ball (tria_inner_1,centre,
                                   notch_radius_inner,false);

        Triangulation<2> tria_inner_2;
        GridGenerator::hyper_shell  (tria_inner_2,centre,
                                     notch_radius_inner,
                                     parameters.notch_radius,
                                     4);
        GridTools::rotate(M_PI/4,tria_inner_2);

        Triangulation<2> tria_outer;
        GridGenerator::hyper_shell  (tria_outer,centre,
                                     parameters.notch_radius,
                                     parameters.radius,
                                     4);
        GridTools::rotate(M_PI/4,tria_outer);

        Triangulation<2> tria_inner;
        GridGenerator::merge_triangulations(tria_inner_1,tria_inner_2,tria_inner);
        GridGenerator::merge_triangulations(tria_inner,tria_outer,tria_cross_section);
      }

      Triangulation<dim> tria_unnotched;
      {
        std::vector<double> slice_coordinates;
        slice_coordinates.push_back(0.0);
        slice_coordinates.push_back(parameters.notch_length);
        slice_coordinates.push_back(2.0*parameters.notch_length);
        slice_coordinates.push_back(parameters.length);
        GridGenerator::extrude_triangulation  (tria_cross_section,
                                               slice_coordinates,
                                               tria_unnotched);
        GridTools::rotate(M_PI/2,1,tria_unnotched);
      }

      // Remove all cells within the notch length and greater than
      // the notch radius
      std::set<typename Triangulation<dim>::active_cell_iterator> cells_to_remove;
      for (typename Triangulation<dim>::active_cell_iterator
           cell = tria_unnotched.begin_active();
           cell!=tria_unnotched.end(); ++cell)
        {
          if (cell->center()[0] < parameters.notch_length)
          {
            // Outer layer of cells correspond to the notch to be removed
            for (unsigned int face=0; face < GeometryInfo<dim>::faces_per_cell; ++face)
              if (cell->face(face)->at_boundary() == true &&
                  std::abs(cell->face(face)->center()[0] - parameters.notch_length/2.0) < 1e-12)
                cells_to_remove.insert(cell);

            // Additional verification
            Point<dim> rad = cell->center();
            rad[0] = 0.0;
            const double radius = rad.norm();
            if (radius > parameters.notch_radius)
              cells_to_remove.insert(cell);
          }
        }

      AssertThrow(cells_to_remove.empty() == false, ExcMessage("Found no cells to remove."));
      GridGenerator::create_triangulation_with_removed_cells(tria_unnotched,cells_to_remove,triangulation);

      // Set boundary and manifold IDs
      for (typename Triangulation<dim>::active_cell_iterator
           cell = triangulation.begin_active();
           cell!=triangulation.end(); ++cell)
        {
          for (unsigned int face=0; face < GeometryInfo<dim>::faces_per_cell; ++face)
          {
            if (cell->face(face)->at_boundary() == true)
            {
              if (std::abs(cell->face(face)->center()[0] - 0.0) < 1e-6)
                cell->face(face)->set_boundary_id(1);
              else if (std::abs(cell->face(face)->center()[0] - parameters.length) < 1e-6)
                cell->face(face)->set_boundary_id(2);
            }
          }
        }

      triangulation.refine_global(1);
      GridTools::scale (parameters.scale, triangulation);
    }
    else if (parameters.geometry_type == "Notched tensile specimen")
    {
      AssertThrow(false, ExcNotImplemented());
    }
    else
      AssertThrow(false, ExcMessage("Unknown geometry"));

    for (typename Triangulation<dim>::active_cell_iterator
         cell = triangulation.begin_active();
         cell!=triangulation.end(); ++cell)
      {
        if (parameters.mat_type == "Elastic")
          cell->set_material_id(0);
        else if (parameters.mat_type == "Elastoplastic (isotropic hardening)")
          cell->set_material_id(1);
        else
          AssertThrow(false, ExcMessage("Unknown material id"));
      }
  }

  // @sect4{LinearElastoplasticProblem::setup_system}

  template <int dim>
  void LinearElastoplasticProblem<dim>::setup_system ()
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

    // Setup quadrature point history
    setup_qph();
  }


  template <int dim>
  void LinearElastoplasticProblem<dim>::setup_qph()
  {
    std::cout << "Setting up quadrature point data..." << std::endl;
    const unsigned int   n_q_points_cell = qf_cell.size();

    quadrature_point_history.clear();
    quadrature_point_history.initialize(triangulation.begin_active(),
                                        triangulation.end(),
                                        n_q_points_cell);
    for (typename DoFHandler<dim>::active_cell_iterator
         cell = dof_handler.begin_active();
         cell!=dof_handler.end(); ++cell)
      {
        const std::vector<std::shared_ptr<PointHistory<dim> > > lqph =
          quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points_cell, ExcInternalError());
        for (unsigned int q_point = 0; q_point < n_q_points_cell; ++q_point)
          lqph[q_point]->setup_lqp(cell->material_id(), parameters, time);
      }
}

  // @sect4{LinearElastoplasticProblem::assemble_system}

  template <int dim>
  void LinearElastoplasticProblem<dim>::assemble_system ()
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
    // const unsigned int   n_q_points_face = qf_face.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    std::vector<Tensor<2, dim> > solution_grads_u_total (qf_cell.size());

    for (typename DoFHandler<dim>::active_cell_iterator
         cell = dof_handler.begin_active();
         cell!=dof_handler.end(); ++cell)
      {
        cell_matrix = 0;
        cell_rhs = 0;

        fe_values.reinit (cell);
        fe_values[u_fe].get_function_gradients(
            solution,
            solution_grads_u_total);

        const std::vector<std::shared_ptr< PointHistory<dim> > > lqph =
        quadrature_point_history.get_data(cell);

        for (unsigned int q_point_cell=0; q_point_cell<n_q_points_cell; ++q_point_cell)
          {
            const Tensor<2,dim> &Grad_u = solution_grads_u_total[q_point_cell];
            const SymmetricTensor<2,dim> epsilon = Physics::Elasticity::Kinematics::epsilon(Grad_u);
            lqph[q_point_cell]->update_internal_equilibrium(epsilon);
            const SymmetricTensor<4,dim> K = lqph[q_point_cell]->get_K(epsilon);

            const double JxW = fe_values.JxW(q_point_cell);

            // Precompute the shape function symmetric gradients
            // (related to the variation of the small strain)
            std::vector< SymmetricTensor<2,dim> > d_epsilon (dofs_per_cell);
            for (unsigned int K=0; K<dofs_per_cell; ++K)
              d_epsilon[K] = fe_values[u_fe].symmetric_gradient(K,q_point_cell);

            for (unsigned int I=0; I<dofs_per_cell; ++I)
              {
                const SymmetricTensor<2,dim> &d_epsilon_I = d_epsilon[I];

                for (unsigned int J=I; J<dofs_per_cell; ++J)
                  {
                    const SymmetricTensor<2,dim> &d_epsilon_J = d_epsilon[J];

                    cell_matrix(I,J)
                      += contract3(d_epsilon_I,K,d_epsilon_J) * JxW;
                  }
              }
          }

        for (unsigned int I=0; I<dofs_per_cell; ++I)
          for (unsigned int J=I+1; J<dofs_per_cell; ++J)
            cell_matrix(J,I) = cell_matrix(I,J);

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
  void LinearElastoplasticProblem<dim>::apply_boundary_conditions ()
  {
    std::map<types::global_dof_index,double> boundary_values;

    if (parameters.geometry_type == "Notched cylinder")
    {
      // No X displacement on constraint on -X faces
      {
        ComponentMask component_mask_x (n_components, false);
        component_mask_x.set(0, true);
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  1,
                                                  ZeroFunction<dim>(dim),
                                                  boundary_values,
                                                  component_mask_x);
      }
      // Prescribed horizontal displacement on +X faces
      {
        ComponentMask component_mask_x (n_components, false);
        component_mask_x.set(0, true);
        const double total_displacement
        = (parameters.final_stretch-1.0)*parameters.length
        * parameters.scale
        * (time.current()/time.end());
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  2,
                                                  ConstantFunction<dim>(total_displacement,dim),
                                                  boundary_values,
                                                  component_mask_x);
      }
      // No radial displacement of +X faces
      {
        ComponentMask component_mask_yz (n_components, true);
        component_mask_yz.set(0, false);
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  2,
                                                  ZeroFunction<dim>(dim),
                                                  boundary_values,
                                                  component_mask_yz);
      }
    }
    else if (parameters.geometry_type == "Notched tensile specimen")
    {
      AssertThrow(false, ExcNotImplemented());
    }
    else
      AssertThrow(false, ExcMessage("Unknown geometry"));

    MatrixTools::apply_boundary_values (boundary_values,
                                        system_matrix,
                                        solution,
                                        system_rhs);
  }


  // @sect4{LinearElastoplasticProblem::solve}

  template <int dim>
  void LinearElastoplasticProblem<dim>::solve ()
  {
    SolverControl solver_control (system_matrix.m(), 1e-12);
    SolverCG<>    cg (solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    cg.solve (system_matrix, solution, system_rhs,
              preconditioner);

    hanging_node_constraints.distribute (solution);
  }


  // @sect4{LinearElastoplasticProblem::output_results}


  template <int dim>
  void LinearElastoplasticProblem<dim>::output_results () const
  {
    // Visual output: FEM results
    std::string filename = "solution-";
    filename += Utilities::int_to_string(time.get_timestep(),4);
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



  // @sect4{LinearElastoplasticProblem::run}

  template <int dim>
  void LinearElastoplasticProblem<dim>::run ()
  {
    make_grid();
    setup_system ();
    output_results ();

//    const bool do_grid_refinement = false;
    time.increment();
    while (time.current() < time.end()+0.01*time.get_delta_t())
      {
        std::cout
            << "Timestep " << time.get_timestep()
            << " @ time " << time.current()
            << std::endl;

        // Next we assemble the system and enforce boundary
        // conditions.
        // Here we assume that the system and fibres have
        // a fixed state, and we will assemble based on how
        // epsilon_c will update given the current state of
        // the body.
        assemble_system ();
        apply_boundary_conditions ();

        // Then we solve the linear system
        solve ();

        // Output some values to file
        output_results ();

        time.increment();
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

      LMM::LinearElastoplasticProblem<dim> elastoplastic_problem ("parameters.prm");
      elastoplastic_problem.run ();
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
