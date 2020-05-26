/* ---------------------------------------------------------------------
 * Copyright (C) 2018 by the deal.II authors and
 *                           Jean-Paul Pelteret
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
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/meshworker/copy_data.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/meshworker/scratch_data.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/physics/transformations.h>
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include <fstream>
#include <iostream>
#include <vector>

namespace LinearElastoplasticity
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
      double       scale;
      unsigned int n_global_refinement_steps;
      unsigned int n_local_refinement_steps;

      struct NotchedCylinder
      {
        double       length;
        double       radius;
        double       notch_length;
        double       notch_radius;
      } notched_cylinder;

      struct TensileSpecimen
      {
        const double length = 35.0/2;

        const Tensor<1,3> manifold_direction  = Tensor<1,3>({0,0,1});
        const Point<3> centre_manifold_id_10  = {6.0,4.0,0.0};
        const Point<3> centre_manifold_id_11  = {10.472135955,0.0,0.0};
      } tensile_specimen;

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
                          Patterns::Selection("Notched cylinder|Tensile specimen"),
                          "The geometry to be modelled");

        prm.declare_entry("Grid scale", "1e-3",
                          Patterns::Double(0),
                          "Global grid scaling factor");

        prm.declare_entry("Global refinement steps", "1",
                          Patterns::Integer(0),
                          "Number of global refinement steps");

        prm.declare_entry("Local refinement steps", "1",
                          Patterns::Integer(0),
                          "Number of initial local refinement cycles");

        prm.enter_subsection("Notched cylinder");
        {
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
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    void Geometry::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        geometry_type = prm.get("Geometry type");
        scale = prm.get_double("Grid scale");
        n_global_refinement_steps = prm.get_integer("Global refinement steps");
        n_local_refinement_steps = prm.get_integer("Local refinement steps");

        prm.enter_subsection("Notched cylinder");
        {
          notched_cylinder.length = prm.get_double("Length");
          notched_cylinder.radius = prm.get_double("Radius");
          notched_cylinder.notch_length = prm.get_double("Notch length");
          notched_cylinder.notch_radius = prm.get_double("Notch radius");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      AssertThrow(notched_cylinder.length > notched_cylinder.notch_length,
                  ExcMessage("Cannot create geometry with given dimensions"));
      AssertThrow(notched_cylinder.radius > notched_cylinder.notch_radius,
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


// @sect3{Time incrementation}

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


// @sect3{Constitutive laws}

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
    const double kappa_e; // bulk modulus
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
      // Record history of plastic variables
      epsilon_p_t1 = epsilon_p_t;
      alpha_p_t1 = alpha_p_t;
    }

    // --- Post-processing ---

    // Plastic variables:
    // Isotropic hardening variable
    double
    get_alpha_p() const
    {
      return alpha_p_t;
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


// @sect3{Quadrature point data}

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
    const Material_Base<dim> * const
    get_material() const
    {
      Assert(material, ExcInternalError());
      return &(*material);
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
    void refine_grid ();
    void setup_system ();
    void setup_qph();
    void make_constraints ();
    void assemble_system ();
    void solve ();
    void output_results () const;
    void update_end_timestep ();

    Parameters::AllParameters parameters;

    Triangulation<dim>   triangulation;
    DoFHandler<dim>      dof_handler;

    Time time;

    FESystem<dim>        fe;
    QGauss<dim>          qf_cell;
    QGauss<dim-1>        qf_face;

    AffineConstraints<double>     hanging_node_constraints;
    AffineConstraints<double>     constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double>       solution;
    Vector<double>       system_rhs;

    // Local data
    CellDataStorage<typename Triangulation<dim>::active_cell_iterator,
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
      const double notch_radius_inner = parameters.notched_cylinder.notch_radius - (parameters.notched_cylinder.radius-parameters.notched_cylinder.notch_radius);

      Triangulation<2> tria_cross_section;
      {
        const Point<2> centre;
        Triangulation<2> tria_inner_1;
        GridGenerator::hyper_ball (tria_inner_1,centre,
                                   notch_radius_inner,false);

        Triangulation<2> tria_inner_2;
        GridGenerator::hyper_shell  (tria_inner_2,centre,
                                     notch_radius_inner,
                                     parameters.notched_cylinder.notch_radius,
                                     4);
        GridTools::rotate(M_PI/4,tria_inner_2);

        Triangulation<2> tria_outer;
        GridGenerator::hyper_shell  (tria_outer,centre,
                                     parameters.notched_cylinder.notch_radius,
                                     parameters.notched_cylinder.radius,
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
        slice_coordinates.push_back(parameters.notched_cylinder.notch_length);
        slice_coordinates.push_back(2.0*parameters.notched_cylinder.notch_length);
        // Biased slices with function taken from GridGenerator::concentric_hyper_shells
        const double bias_first_slice = 2.0*parameters.notched_cylinder.notch_length;
        const double bias_last_slice = parameters.notched_cylinder.length;
        const double skewness = 2.0;
        const unsigned int n_slices = 5;
        for (unsigned int s=0; s<n_slices; ++s)
        {
          const double f = (1 - std::tanh(skewness*(1-s/static_cast<double>(n_slices)))) / std::tanh(skewness);
          AssertThrow(f >= 0.0 && f <= 1.0, ExcMessage("Bias function not in correct range."));
          const double r = bias_first_slice + (bias_last_slice-bias_first_slice)*f;
          AssertThrow(r >= bias_first_slice && r <= bias_last_slice, ExcMessage("New slice location not in correct range."));
          slice_coordinates.push_back(r);
        }
        slice_coordinates.push_back(parameters.notched_cylinder.length);
        GridGenerator::extrude_triangulation  (tria_cross_section,
                                               slice_coordinates,
                                               tria_unnotched);
        GridTools::rotate(M_PI/2,1,tria_unnotched);
      }

      std::set<typename Triangulation<dim>::active_cell_iterator> cells_to_remove;
      for (typename Triangulation<dim>::active_cell_iterator
           cell = tria_unnotched.begin_active();
           cell!=tria_unnotched.end(); ++cell)
        {
          // Remove all cells within the notch length and greater than
          // the notch radius.
          if (cell->center()[0] < parameters.notched_cylinder.notch_length)
          {
            // Outer layer of cells correspond to the notch to be removed
            for (unsigned int face=0; face < GeometryInfo<dim>::faces_per_cell; ++face)
              if (cell->face(face)->at_boundary() == true &&
                  std::abs(cell->face(face)->center()[0] - parameters.notched_cylinder.notch_length/2.0) < 1e-12)
                cells_to_remove.insert(cell);

            // Additional verification
            Point<dim> rad = cell->center();
            rad[0] = 0.0;
            const double radius = rad.norm();
            if (radius > parameters.notched_cylinder.notch_radius)
              cells_to_remove.insert(cell);
          }
        }

      AssertThrow(cells_to_remove.empty() == false, ExcMessage("Found no cells to remove."));
      GridGenerator::create_triangulation_with_removed_cells(tria_unnotched,cells_to_remove,triangulation);

      // Set boundary and manifold IDs
      triangulation.set_all_manifold_ids(2); // TFI manifold

      const double radius_for_cyl_cells = notch_radius_inner*std::cos(M_PI/4);
      for (typename Triangulation<dim>::active_cell_iterator
           cell = triangulation.begin_active();
           cell!=triangulation.end(); ++cell)
        {
          Point<dim> rad = cell->center();
          rad[0] = 0.0;
          const double radius = rad.norm();
          if (radius > radius_for_cyl_cells)
          {
            cell->set_all_manifold_ids(1); // Cylindrical manifold
          }
        }
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
              else if (std::abs(cell->face(face)->center()[0] - parameters.notched_cylinder.length) < 1e-6)
                cell->face(face)->set_boundary_id(2);
              else
                cell->face(face)->set_all_manifold_ids(1); // Cylindrical manifold
            }
          }
        }

      CylindricalManifold<dim> cylindrical_manifold (0 /*axis*/);
      CylindricalManifold<dim> cylindrical_manifold_tfi (0 /*axis*/);
      TransfiniteInterpolationManifold<dim> tfi_manifold;

      triangulation.set_manifold (1, cylindrical_manifold);
      tfi_manifold.initialize(triangulation);
      triangulation.set_manifold (2, tfi_manifold);

      triangulation.refine_global(parameters.n_global_refinement_steps);
      GridTools::scale (parameters.scale, triangulation);
    }
    else if (parameters.geometry_type == "Tensile specimen")
    {
      const std::string filename = "tensile_specimen.inp";
      std::ifstream input (filename.c_str());

      GridIn<dim> gridin;
      gridin.attach_triangulation(triangulation);
      gridin.read_abaqus(input, false);

//      GridTools::copy_boundary_to_manifold_id(triangulation,false);
      for (typename Triangulation<dim>::active_cell_iterator
           cell = triangulation.begin_active();
           cell!=triangulation.end(); ++cell)
        {
          for (unsigned int face=0; face < GeometryInfo<dim>::faces_per_cell; ++face)
          {
            if (cell->face(face)->at_boundary() == true)
            {
              if (cell->face(face)->boundary_id() == 10 || cell->face(face)->boundary_id() == 11)
                cell->face(face)->set_all_manifold_ids(cell->face(face)->boundary_id());
            }
          }
        }

      // The triangulation scaling needs to happen before the manifolds
      // with a non-trivial centre point are defined / applied.
      GridTools::scale (parameters.scale, triangulation);

      CylindricalManifold<dim> cylindrical_manifold_id_10 (parameters.tensile_specimen.manifold_direction,parameters.tensile_specimen.centre_manifold_id_10*parameters.scale);
      CylindricalManifold<dim> cylindrical_manifold_id_11 (parameters.tensile_specimen.manifold_direction,parameters.tensile_specimen.centre_manifold_id_11*parameters.scale);
      triangulation.set_manifold (10, cylindrical_manifold_id_10);
      triangulation.set_manifold (11, cylindrical_manifold_id_11);

      triangulation.refine_global(parameters.n_global_refinement_steps);
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


// @sect4{LinearElastoplasticProblem::refine_grid}

  template <int dim>
  void LinearElastoplasticProblem<dim>::refine_grid ()
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(
      dof_handler,
      QGauss<dim - 1>(fe.degree + 1),
      std::map<types::boundary_id, const Function<dim> *>(),
      solution,
      estimated_error_per_cell);
    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    estimated_error_per_cell,
                                                    0.3,
                                                    0.03);
    triangulation.execute_coarsening_and_refinement();
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

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    hanging_node_constraints,
                                    /*keep_constrained_dofs = */ true);
    sparsity_pattern.copy_from(dsp);

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


// @sect4{LinearElastoplasticProblem::setup_qph}

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
    system_matrix = 0;
    system_rhs = 0;

    using ScratchData      = MeshWorker::ScratchData<dim>;
    using CopyData         = MeshWorker::CopyData<1, 1, 1>;
    using CellIteratorType = decltype(dof_handler.begin_active());

    const ScratchData sample_scratch_data (fe, qf_cell,
                                           update_values | update_gradients |
                                           update_quadrature_points | update_JxW_values);
    CopyData          sample_copy_data (fe.dofs_per_cell);
    
    auto cell_worker = [this] (const CellIteratorType &cell,
                               ScratchData            &scratch_data,
                               CopyData               &copy_data)
    {
      const FEValues<dim> &fe_values = scratch_data.reinit(cell);
      FullMatrix<double>  &cell_matrix = copy_data.matrices[0];

      std::vector<types::global_dof_index> &local_dof_indices = copy_data.local_dof_indices[0];
      cell->get_dof_indices(local_dof_indices);

      const unsigned int dofs_per_cell = fe_values.dofs_per_cell;
      const unsigned int n_q_points_cell = fe_values.n_quadrature_points;

      std::vector<Tensor<2, dim> > solution_grads_u_total (n_q_points_cell);
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
    };

    auto copier = [this](const CopyData &copy_data)
    {
      const FullMatrix<double>  &cell_matrix = copy_data.matrices[0];
      const Vector<double>      &cell_rhs = copy_data.vectors[0];
      const std::vector<types::global_dof_index> &local_dof_indices = copy_data.local_dof_indices[0];

      constraints.distribute_local_to_global(cell_matrix, cell_rhs, 
                                             local_dof_indices, 
                                             system_matrix, system_rhs);
    };

    MeshWorker::mesh_loop(dof_handler.active_cell_iterators(),
                          cell_worker, copier,
                          sample_scratch_data, 
                          sample_copy_data,
                          MeshWorker::assemble_own_cells);
  }


// @sect4{LinearElastoplasticProblem::make_constraints}

  template <int dim>
  void LinearElastoplasticProblem<dim>::make_constraints ()
  {
    constraints.clear();

    if (parameters.geometry_type == "Notched cylinder")
    {
      // Fully constraints (pinned) vertex at centre of notched (-X) face
      {
        Point<dim> pinned_point;
        typename DoFHandler<dim>::active_cell_iterator cell =
            dof_handler.begin_active(), endc = dof_handler.end();
          for (; cell != endc; ++cell)
            {
            for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
              if (cell->vertex(v).distance(pinned_point) < 1e-9)
                {
                  for (unsigned int d=0; d<dim; ++d)
                    constraints.add_line(cell->vertex_dof_index(v,d));
                }
            }
      }
      // No X displacement on constraint on -X faces
      {
        ComponentMask component_mask_x (n_components, false);
        component_mask_x.set(0, true);
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  1,
                                                  ZeroFunction<dim>(dim),
                                                  constraints,
                                                  component_mask_x);
      }
      // Prescribed horizontal displacement on +X faces
      {
        ComponentMask component_mask_x (n_components, false);
        component_mask_x.set(0, true);
        const double total_displacement
        = (parameters.final_stretch-1.0)*parameters.notched_cylinder.length
        * parameters.scale
        * (time.current()/time.end());
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  2,
                                                  ConstantFunction<dim>(total_displacement,dim),
                                                  constraints,
                                                  component_mask_x);
      }
    }
    else if (parameters.geometry_type == "Tensile specimen")
    {
      // No X displacement on constraint on -X faces
      {
        ComponentMask component_mask_x (n_components, false);
        component_mask_x.set(0, true);
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  1,
                                                  ZeroFunction<dim>(dim),
                                                  constraints,
                                                  component_mask_x);
      }
      // No Y displacement on constraint on -Y faces
      {
        ComponentMask component_mask_y (n_components, false);
        component_mask_y.set(1, true);
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  3,
                                                  ZeroFunction<dim>(dim),
                                                  constraints,
                                                  component_mask_y);
      }
      // No Z displacement on constraint on -Z faces
      {
        ComponentMask component_mask_z (n_components, false);
        component_mask_z.set(2, true);
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  5,
                                                  ZeroFunction<dim>(dim),
                                                  constraints,
                                                  component_mask_z);
      }
      // Prescribed horizontal displacement on clamped surface
      {
        ComponentMask component_mask_x (n_components, false);
        component_mask_x.set(0, true);
        const double total_displacement
        = (parameters.final_stretch-1.0)*parameters.tensile_specimen.length
        * parameters.scale
        * (time.current()/time.end());
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  6,
                                                  ConstantFunction<dim>(total_displacement,dim),
                                                  constraints,
                                                  component_mask_x);
      }
      // No out-of-plane displacement on clamped surface
      {
        ComponentMask component_mask_yz (n_components, true);
        component_mask_yz.set(0, false);
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  6,
                                                  ZeroFunction<dim>(dim),
                                                  constraints,
                                                  component_mask_yz);
      }
    }
    else
      AssertThrow(false, ExcMessage("Unknown geometry"));

    constraints.merge(hanging_node_constraints, AffineConstraints<double>::right_object_wins);
    constraints.close();
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

    constraints.distribute(solution);
  }


// @sect4{LinearElastoplasticProblem::update_end_timestep}

  template <int dim>
  void LinearElastoplasticProblem<dim>::update_end_timestep ()
  {
    const unsigned int n_q_points_cell = qf_cell.size();

    for (typename DoFHandler<dim>::active_cell_iterator
         cell = dof_handler.begin_active();
         cell!=dof_handler.end(); ++cell)
      {
        const std::vector<std::shared_ptr< PointHistory<dim> > > lqph =
            quadrature_point_history.get_data(cell);

        for (unsigned int q_point_cell=0; q_point_cell<n_q_points_cell; ++q_point_cell)
          lqph[q_point_cell]->update_end_timestep();
      }
  }


  // @sect4{LinearElastoplasticProblem::output_results}

  template <int dim>
  class PostProcessIsotropicHardening : public DataPostprocessorScalar<dim>
  {
  public:
    PostProcessIsotropicHardening(const CellDataStorage<typename Triangulation<dim>::active_cell_iterator,
                                  PointHistory<dim> > &quadrature_point_history,
                                  const QGauss<dim>   &qf_cell);
    virtual ~PostProcessIsotropicHardening() = default;
    virtual void evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &inputs,
      std::vector<Vector<double>> &computed_quantities) const override;

  private:
    const CellDataStorage<typename Triangulation<dim>::active_cell_iterator,
    PointHistory<dim> > &quadrature_point_history;

    FullMatrix<double> projection_matrix_qpoint_to_support_point;
  };


  template <int dim>
  PostProcessIsotropicHardening<dim>::PostProcessIsotropicHardening(
      const CellDataStorage<typename Triangulation<dim>::active_cell_iterator,
      PointHistory<dim> > &quadrature_point_history,
      const QGauss<dim>   &quadrature_cell)
    : DataPostprocessorScalar<dim>("isotropic_hardening", update_values),
      quadrature_point_history (quadrature_point_history)
  {
    const FE_Q<dim> fe_projection (1);
    const QTrapez<dim> quadrature_projection;
    projection_matrix_qpoint_to_support_point = FullMatrix<double> (
        quadrature_projection.size(),
        quadrature_cell.size());

    FETools::compute_projection_from_quadrature_points_matrix
              (fe_projection,
               quadrature_projection,
               quadrature_cell,
               projection_matrix_qpoint_to_support_point);
  }

  
  template <int dim>
  void PostProcessIsotropicHardening<dim>::evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &inputs,
    std::vector<Vector<double>> &               computed_quantities) const
  {
    Assert(computed_quantities.size() == inputs.solution_values.size(),
           ExcDimensionMismatch(computed_quantities.size(),
               inputs.solution_values.size()));

    const typename DoFHandler<dim>::active_cell_iterator cell
      = inputs.template get_cell<DoFHandler<dim>>();

    // number of support points (nodes) to project to
    const unsigned int n_support_points = projection_matrix_qpoint_to_support_point.n_rows();
    // number of quadrature points to project from
    const unsigned int n_quad_points = projection_matrix_qpoint_to_support_point.n_cols();

    // component projected to the nodes
    Vector<double> component_at_node(n_support_points);
    // component at the quadrature point
    Vector<double> component_at_qp(n_quad_points);

    const std::vector<std::shared_ptr< const PointHistory<dim> > > lqph =
    quadrature_point_history.get_data(cell);
    Assert(lqph.size() == n_quad_points, ExcDimensionMismatch(lqph.size(), n_quad_points));

    // populate the vector of components at the qps
    for (unsigned int q_point = 0; q_point < n_quad_points; ++q_point)
    {
      const Material_Base<dim> * const material = lqph[q_point]->get_material();
      if (const Material_Linear_Elastoplastic_Isotropic_Hardening<dim>* const material_ep
          = dynamic_cast<const Material_Linear_Elastoplastic_Isotropic_Hardening<dim>* const>(material))
        component_at_qp(q_point) = material_ep->get_alpha_p();
      else
        component_at_qp(q_point) = 0.0;
    }

    // project from the qps -> nodes
    // component_at_node = projection_matrix_u * component_at_qp
    projection_matrix_qpoint_to_support_point.vmult(component_at_node, component_at_qp);

    Assert(computed_quantities.size() == n_support_points,
           ExcDimensionMismatch(computed_quantities.size(),
               n_support_points));
    for (unsigned int i = 0; i < n_support_points; i++)
    {
      Assert(inputs.solution_values[i].size() == dim,
          ExcDimensionMismatch(inputs.solution_values[i].size(), dim));
      (void)inputs;

      computed_quantities[i](0) = component_at_node(i);
    }
  }


  template <int dim>
  void LinearElastoplasticProblem<dim>::output_results () const
  {
    std::string filename = "solution-";
    filename += Utilities::int_to_string(time.get_timestep(),4);
    filename += ".vtu";
    std::ofstream output (filename.c_str());

    PostProcessIsotropicHardening<dim> pp_isotropic_hardening (quadrature_point_history, qf_cell);
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim,
                                  DataComponentInterpretation::component_is_part_of_vector);
    std::vector<std::string> solution_name(dim, "displacement");

    data_out.add_data_vector (solution, solution_name,
                              DataOut<dim>::type_dof_data,
                              data_component_interpretation);
    data_out.add_data_vector(solution, pp_isotropic_hardening);

    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags (flags);
    data_out.build_patches (StaticMappingQ1<dim>::mapping, fe.degree, DataOut<dim>::curved_inner_cells);
    data_out.write_vtu (output);

    static std::vector< std::pair< double, std::string >> times_and_names;
    times_and_names.emplace_back (time.current(), filename);
    std::ofstream pvd_output ("solution.pvd");
    DataOutBase::write_pvd_record (pvd_output, times_and_names);
  }


  // @sect4{LinearElastoplasticProblem::run}

  template <int dim>
  void LinearElastoplasticProblem<dim>::run ()
  {
    make_grid();
    setup_system ();
    output_results ();

    update_end_timestep();
    time.increment();

    auto solve_timestep = [&]()
    {
      // Next we assemble the system and enforce boundary
      // conditions.
      make_constraints();
      assemble_system ();

      // Then we solve the linear system
      solve ();
    };

    while (time.current() < time.end()+0.01*time.get_delta_t())
      {
        std::cout
            << "Timestep " << time.get_timestep()
            << " @ time " << time.current()
            << std::endl;

        // Compute the solution at the current timestep
        solve_timestep();

        // Perform local refinement
        if (time.get_timestep() == 1)
          for (unsigned int cycle = 0; cycle < parameters.n_local_refinement_steps; ++cycle)
          {
            std::cout
              << "Executing refinement cycle " << cycle
              << " of " << parameters.n_local_refinement_steps
              << "..." << std::endl;
            refine_grid();
            setup_system();
            solve_timestep();
          }

        // Output some values to file
        output_results ();

        update_end_timestep();
        time.increment();
      }
  }
} // namespace LinearElastoplasticity


// @sect3{The <code>main</code> function}

int main (int argc, char *argv[])
{
  try
    {
      dealii::deallog.depth_console (0);
      const unsigned int dim = 3;

      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

      LinearElastoplasticity::LinearElastoplasticProblem<dim> elastoplastic_problem ("parameters.prm");
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
