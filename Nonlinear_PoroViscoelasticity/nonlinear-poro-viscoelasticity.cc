/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2010 - 2020 by the deal.II authors and
 *                              Ester Comellas and Jean-Paul Pelteret
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

/*   Authors: Ester Comellas and Jean-Paul Pelteret,
 *           University of Erlangen-Nuremberg, 2018
 */

// We start by including all the necessary deal.II header files and some C++
// related ones. They have been discussed in detail in previous tutorial
// programs, so you need only refer to past tutorials for details.

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_point_data.h>

#include <deal.II/differentiation/ad.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_linear_operator.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/physics/transformations.h>
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include <iostream>
#include <fstream>
#include <numeric>
#include <iomanip>


// We create a namespace for everything that relates to
// the nonlinear poro-viscoelastic formulation,
// and import all the deal.II function and class names into it:
namespace NonLinearPoroViscoElasticity
{
    using namespace dealii;

// @sect3{Run-time parameters}
//
// Set up a ParameterHandler object to read in the parameter choices at run-time
// introduced by the user through the file "parameters.prm"
    namespace Parameters
    {
// @sect4{Finite Element system}
// Here we specify the polynomial order used to approximate the solution,
// both for the displacements and pressure unknowns.
// The quadrature order should be adjusted accordingly.
      struct FESystem
      {
        unsigned int poly_degree_displ;
        unsigned int poly_degree_pore;
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
          prm.declare_entry("Polynomial degree displ", "2",
                            Patterns::Integer(0),
                            "Displacement system polynomial order");

          prm.declare_entry("Polynomial degree pore", "1",
                            Patterns::Integer(0),
                            "Pore pressure system polynomial order");

          prm.declare_entry("Quadrature order", "3",
                            Patterns::Integer(0),
                            "Gauss quadrature order");
        }
        prm.leave_subsection();
      }

      void FESystem::parse_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Finite element system");
        {
          poly_degree_displ = prm.get_integer("Polynomial degree displ");
          poly_degree_pore = prm.get_integer("Polynomial degree pore");
          quad_order = prm.get_integer("Quadrature order");
        }
        prm.leave_subsection();
      }

// @sect4{Geometry}
// These parameters are related to the geometry definition and mesh generation.
// We select the type of problem to solve and introduce the desired load values.
      struct Geometry
      {
        std::string  geom_type;
        unsigned int global_refinement;
        double       scale;
        std::string  load_type;
        double       load;
        unsigned int num_cycle_sets;
        double       fluid_flow;
        double       drained_pressure;

        static void
        declare_parameters(ParameterHandler &prm);

        void
        parse_parameters(ParameterHandler &prm);
      };

      void Geometry::declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Geometry");
        {
          prm.declare_entry("Geometry type", "Ehlers_tube_step_load",
                             Patterns::Selection("Ehlers_tube_step_load"
                                                 "|Ehlers_tube_increase_load"
                                                 "|Ehlers_cube_consolidation"
                                                 "|Franceschini_consolidation"
                                                 "|Budday_cube_tension_compression"
                                                 "|Budday_cube_tension_compression_fully_fixed"
                                                 "|Budday_cube_shear_fully_fixed"),
                                "Type of geometry used. "
                                "For Ehlers verification examples see Ehlers and Eipper (1999). "
                                "For Franceschini brain consolidation see Franceschini et al. (2006)"
                                "For Budday brain examples see Budday et al. (2017)");

          prm.declare_entry("Global refinement", "1",
                            Patterns::Integer(0),
                            "Global refinement level");

          prm.declare_entry("Grid scale", "1.0",
                            Patterns::Double(0.0),
                            "Global grid scaling factor");

          prm.declare_entry("Load type", "pressure",
                            Patterns::Selection("pressure|displacement|none"),
                            "Type of loading");

          prm.declare_entry("Load value", "-7.5e+6",
                            Patterns::Double(),
                            "Loading value");

          prm.declare_entry("Number of cycle sets", "1",
                            Patterns::Integer(1,2),
                            "Number of times each set of 3 cycles is repeated, only for "
                            "Budday_cube_tension_compression and Budday_cube_tension_compression_fully_fixed. "
                            "Load value is doubled in second set, load rate is kept constant."
                            "Final time indicates end of second cycle set.");

          prm.declare_entry("Fluid flow value", "0.0",
                            Patterns::Double(),
                            "Prescribed fluid flow. Not implemented in any example yet.");

          prm.declare_entry("Drained pressure", "0.0",
                            Patterns::Double(),
                            "Increase of pressure value at drained boundary w.r.t the atmospheric pressure.");
        }
        prm.leave_subsection();
      }

      void Geometry::parse_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Geometry");
        {
          geom_type = prm.get("Geometry type");
          global_refinement = prm.get_integer("Global refinement");
          scale = prm.get_double("Grid scale");
          load_type = prm.get("Load type");
          load = prm.get_double("Load value");
          num_cycle_sets = prm.get_integer("Number of cycle sets");
          fluid_flow = prm.get_double("Fluid flow value");
          drained_pressure = prm.get_double("Drained pressure");
        }
        prm.leave_subsection();
      }

// @sect4{Materials}

// Here we select the type of material for the solid component
// and define the corresponding material parameters.
// Then we define he fluid data, including the type of
// seepage velocity definition to use.
      struct Materials
      {
        std::string  mat_type;
        double lambda;
        double mu;
        double mu1_infty;
        double mu2_infty;
        double mu3_infty;
        double alpha1_infty;
        double alpha2_infty;
        double alpha3_infty;
        double mu1_mode_1;
        double mu2_mode_1;
        double mu3_mode_1;
        double alpha1_mode_1;
        double alpha2_mode_1;
        double alpha3_mode_1;
        double viscosity_mode_1;
        std::string  fluid_type;
        double solid_vol_frac;
        double kappa_darcy;
        double init_intrinsic_perm;
        double viscosity_FR;
        double init_darcy_coef;
        double weight_FR;
        bool gravity_term;
        int gravity_direction;
        double gravity_value;
        double density_FR;
        double density_SR;
        enum SymmetricTensorEigenvectorMethod eigen_solver;

        static void
        declare_parameters(ParameterHandler &prm);

        void
        parse_parameters(ParameterHandler &prm);
      };

      void Materials::declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Material properties");
        {
          prm.declare_entry("material", "Neo-Hooke",
                            Patterns::Selection("Neo-Hooke|Ogden|visco-Ogden"),
                            "Type of material used in the problem");

          prm.declare_entry("lambda", "8.375e6",
                            Patterns::Double(0,1e100),
                            "First Lamé parameter for extension function related to compactation point in solid material [Pa].");

          prm.declare_entry("shear modulus", "5.583e6",
                            Patterns::Double(0,1e100),
                            "shear modulus for Neo-Hooke materials [Pa].");

          prm.declare_entry("eigen solver", "QL Implicit Shifts",
                            Patterns::Selection("QL Implicit Shifts|Jacobi"),
                            "The type of eigen solver to be used for Ogden and visco-Ogden models.");

          prm.declare_entry("mu1", "0.0",
                            Patterns::Double(),
                            "Shear material parameter 'mu1' for Ogden material [Pa].");

          prm.declare_entry("mu2", "0.0",
                            Patterns::Double(),
                            "Shear material parameter 'mu2' for Ogden material [Pa].");

          prm.declare_entry("mu3", "0.0",
                            Patterns::Double(),
                            "Shear material parameter 'mu1' for Ogden material [Pa].");

          prm.declare_entry("alpha1", "1.0",
                            Patterns::Double(),
                            "Stiffness material parameter 'alpha1' for Ogden material [-].");

          prm.declare_entry("alpha2", "1.0",
                            Patterns::Double(),
                            "Stiffness material parameter 'alpha2' for Ogden material [-].");

          prm.declare_entry("alpha3", "1.0",
                            Patterns::Double(),
                            "Stiffness material parameter 'alpha3' for Ogden material [-].");

          prm.declare_entry("mu1_1", "0.0",
                            Patterns::Double(),
                            "Shear material parameter 'mu1' for first viscous mode in Ogden material [Pa].");

          prm.declare_entry("mu2_1", "0.0",
                            Patterns::Double(),
                            "Shear material parameter 'mu2' for first viscous mode in Ogden material [Pa].");

          prm.declare_entry("mu3_1", "0.0",
                            Patterns::Double(),
                            "Shear material parameter 'mu1' for first viscous mode in Ogden material [Pa].");

          prm.declare_entry("alpha1_1", "1.0",
                            Patterns::Double(),
                            "Stiffness material parameter 'alpha1' for first viscous mode in Ogden material [-].");

          prm.declare_entry("alpha2_1", "1.0",
                            Patterns::Double(),
                            "Stiffness material parameter 'alpha2' for first viscous mode in Ogden material [-].");

          prm.declare_entry("alpha3_1", "1.0",
                            Patterns::Double(),
                            "Stiffness material parameter 'alpha3' for first viscous mode in Ogden material [-].");

          prm.declare_entry("viscosity_1", "1e-10",
                            Patterns::Double(1e-10,1e100),
                            "Deformation-independent viscosity parameter 'eta_1' for first viscous mode in Ogden material [-].");

          prm.declare_entry("seepage definition", "Ehlers",
                            Patterns::Selection("Markert|Ehlers"),
                            "Type of formulation used to define the seepage velocity in the problem. "
                            "Choose between Markert formulation of deformation-dependent intrinsic permeability "
                            "and Ehlers formulation of deformation-dependent Darcy flow coefficient.");

          prm.declare_entry("initial solid volume fraction", "0.67",
                            Patterns::Double(0.001,0.999),
                            "Initial porosity (solid volume fraction, 0 < n_0s < 1)");

          prm.declare_entry("kappa", "0.0",
                            Patterns::Double(0,100),
                            "Deformation-dependency control parameter for specific permeability (kappa >= 0)");

          prm.declare_entry("initial intrinsic permeability", "0.0",
                            Patterns::Double(0,1e100),
                            "Initial intrinsic permeability parameter [m^2] (isotropic permeability). To be used with Markert formulation.");

          prm.declare_entry("fluid viscosity", "0.0",
                            Patterns::Double(0, 1e100),
                            "Effective shear viscosity parameter of the fluid [Pa·s, (N·s)/m^2]. To be used with Markert formulation.");

          prm.declare_entry("initial Darcy coefficient", "1.0e-4",
                            Patterns::Double(0,1e100),
                            "Initial Darcy flow coefficient [m/s] (isotropic permeability). To be used with Ehlers formulation.");

          prm.declare_entry("fluid weight", "1.0e4",
                            Patterns::Double(0, 1e100),
                            "Effective weight of the fluid [N/m^3]. To be used with Ehlers formulation.");

          prm.declare_entry("gravity term", "false",
                            Patterns::Bool(),
                            "Gravity term considered (true) or neglected (false)");

          prm.declare_entry("fluid density", "1.0",
                            Patterns::Double(0,1e100),
                            "Real (or effective) density of the fluid");

          prm.declare_entry("solid density", "1.0",
                            Patterns::Double(0,1e100),
                            "Real (or effective) density of the solid");

          prm.declare_entry("gravity direction", "2",
                            Patterns::Integer(0,2),
                            "Direction of gravity (unit vector 0 for x, 1 for y, 2 for z)");

          prm.declare_entry("gravity value", "-9.81",
                            Patterns::Double(),
                            "Value of gravity (be careful to have consistent units!)");
        }
        prm.leave_subsection();
      }

      void Materials::parse_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Material properties");
        {
          //Solid
          mat_type = prm.get("material");
          lambda = prm.get_double("lambda");
          mu = prm.get_double("shear modulus");
          mu1_infty = prm.get_double("mu1");
          mu2_infty = prm.get_double("mu2");
          mu3_infty = prm.get_double("mu3");
          alpha1_infty = prm.get_double("alpha1");
          alpha2_infty = prm.get_double("alpha2");
          alpha3_infty = prm.get_double("alpha3");
          mu1_mode_1 = prm.get_double("mu1_1");
          mu2_mode_1 = prm.get_double("mu2_1");
          mu3_mode_1 = prm.get_double("mu3_1");
          alpha1_mode_1 = prm.get_double("alpha1_1");
          alpha2_mode_1 = prm.get_double("alpha2_1");
          alpha3_mode_1 = prm.get_double("alpha3_1");
          viscosity_mode_1 = prm.get_double("viscosity_1");
          //Fluid
          fluid_type = prm.get("seepage definition");
          solid_vol_frac = prm.get_double("initial solid volume fraction");
          kappa_darcy = prm.get_double("kappa");
          init_intrinsic_perm = prm.get_double("initial intrinsic permeability");
          viscosity_FR = prm.get_double("fluid viscosity");
          init_darcy_coef = prm.get_double("initial Darcy coefficient");
          weight_FR = prm.get_double("fluid weight");
          //Gravity effects
          gravity_term = prm.get_bool("gravity term");
          density_FR = prm.get_double("fluid density");
          density_SR = prm.get_double("solid density");
          gravity_direction = prm.get_integer("gravity direction");
          gravity_value = prm.get_double("gravity value");

          if ( (fluid_type == "Markert") && ((init_intrinsic_perm == 0.0) || (viscosity_FR == 0.0)) )
              AssertThrow(false, ExcMessage("Markert seepage velocity formulation requires the definition of "
                                            "'initial intrinsic permeability' and 'fluid viscosity' greater than 0.0."));

          if ( (fluid_type == "Ehlers") && ((init_darcy_coef == 0.0) || (weight_FR == 0.0)) )
              AssertThrow(false, ExcMessage("Ehler seepage velocity formulation requires the definition of "
                                            "'initial Darcy coefficient' and 'fluid weight' greater than 0.0."));

          const std::string eigen_solver_type = prm.get("eigen solver");
          if (eigen_solver_type == "QL Implicit Shifts")
            eigen_solver = SymmetricTensorEigenvectorMethod::ql_implicit_shifts;
          else if (eigen_solver_type == "Jacobi")
            eigen_solver = SymmetricTensorEigenvectorMethod::jacobi;
          else
          {
            AssertThrow(false, ExcMessage("Unknown eigen solver selected."));
          }
        }
        prm.leave_subsection();
      }

// @sect4{Nonlinear solver}

// We now define the tolerances and the maximum number of iterations for the
// Newton-Raphson scheme used to solve the nonlinear system of governing equations.
      struct NonlinearSolver
      {
        unsigned int max_iterations_NR;
        double       tol_f;
        double       tol_u;
        double       tol_p_fluid;

        static void
        declare_parameters(ParameterHandler &prm);

        void
        parse_parameters(ParameterHandler &prm);
      };

      void NonlinearSolver::declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Nonlinear solver");
        {
          prm.declare_entry("Max iterations Newton-Raphson", "15",
                            Patterns::Integer(0),
                            "Number of Newton-Raphson iterations allowed");

          prm.declare_entry("Tolerance force", "1.0e-8",
                            Patterns::Double(0.0),
                            "Force residual tolerance");

          prm.declare_entry("Tolerance displacement", "1.0e-6",
                            Patterns::Double(0.0),
                            "Displacement error tolerance");

          prm.declare_entry("Tolerance pore pressure", "1.0e-6",
                            Patterns::Double(0.0),
                            "Pore pressure error tolerance");
        }
        prm.leave_subsection();
      }

      void NonlinearSolver::parse_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Nonlinear solver");
        {
          max_iterations_NR = prm.get_integer("Max iterations Newton-Raphson");
          tol_f = prm.get_double("Tolerance force");
          tol_u = prm.get_double("Tolerance displacement");
          tol_p_fluid =  prm.get_double("Tolerance pore pressure");
        }
        prm.leave_subsection();
      }

// @sect4{Time}
// Here we set the timestep size $ \varDelta t $ and the simulation end-time.
      struct Time
      {
        double end_time;
        double delta_t;
        static void
        declare_parameters(ParameterHandler &prm);

        void
        parse_parameters(ParameterHandler &prm);
      };

      void Time::declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Time");
        {
          prm.declare_entry("End time", "10.0",
                            Patterns::Double(),
                            "End time");

          prm.declare_entry("Time step size", "0.002",
                            Patterns::Double(1.0e-6),
                            "Time step size. The value must be larger than the displacement error tolerance defined.");
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


// @sect4{Output}
// We can choose the frequency of the data for the output files.
      struct OutputParam
      {

        std::string  outfiles_requested;
        unsigned int timestep_output;
        std::string  outtype;

        static void
        declare_parameters(ParameterHandler &prm);

        void
        parse_parameters(ParameterHandler &prm);
      };

      void OutputParam::declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Output parameters");
        {
          prm.declare_entry("Output files", "true",
                            Patterns::Selection("true|false"),
                            "Paraview output files to generate.");
          prm.declare_entry("Time step number output", "1",
                            Patterns::Integer(0),
                            "Output data for time steps multiple of the given "
                            "integer value.");
          prm.declare_entry("Averaged results", "nodes",
                             Patterns::Selection("elements|nodes"),
                             "Output data associated with integration point values"
                             " averaged on elements or on nodes.");
        }
        prm.leave_subsection();
      }

      void OutputParam::parse_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Output parameters");
        {
          outfiles_requested = prm.get("Output files");
          timestep_output = prm.get_integer("Time step number output");
          outtype = prm.get("Averaged results");
        }
        prm.leave_subsection();
      }

// @sect4{All parameters}
// We finally consolidate all of the above structures into a single container that holds all the run-time selections.
      struct AllParameters : public FESystem,
                             public Geometry,
                             public Materials,
                             public NonlinearSolver,
                             public Time,
                             public OutputParam
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
        Geometry::declare_parameters(prm);
        Materials::declare_parameters(prm);
        NonlinearSolver::declare_parameters(prm);
        Time::declare_parameters(prm);
        OutputParam::declare_parameters(prm);
      }

      void AllParameters::parse_parameters(ParameterHandler &prm)
      {
        FESystem::parse_parameters(prm);
        Geometry::parse_parameters(prm);
        Materials::parse_parameters(prm);
        NonlinearSolver::parse_parameters(prm);
        Time::parse_parameters(prm);
        OutputParam::parse_parameters(prm);
      }
    }

// @sect3{Time class}
// A simple class to store time data.
// For simplicity we assume a constant time step size.
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

          double get_current() const
          {
            return time_current;
          }
          double get_end() const
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
          void increment_time ()
          {
            time_current += delta_t;
            ++timestep;
          }

        private:
          unsigned int timestep;
          double time_current;
          double time_end;
          const double delta_t;
    };

// @sect3{Constitutive equation for the solid component of the biphasic material}

//@sect4{Base class: generic hyperelastic material}
// The ``extra" Kirchhoff stress in the solid component is the sum of isochoric
// and a volumetric part.
// $\mathbf{\tau} = \mathbf{\tau}_E^{(\bullet)} + \mathbf{\tau}^{\textrm{vol}}$
// The deviatoric part changes depending on the type of material model selected:
// Neo-Hooken hyperelasticity, Ogden hyperelasticiy,
// or a single-mode finite viscoelasticity based on the Ogden hyperelastic model.
// In this base class we declare  it as a virtual function,
// and it will be defined for each model type in the corresponding derived class.
// We define here the volumetric component, which depends on the
// extension function $U(J_S)$ selected, and in this case is the same for all models.
// We use the function proposed by
// Ehlers & Eipper 1999 doi:10.1023/A:1006565509095
// We also define some public functions to access and update the internal variables.
    template <int dim, typename NumberType = Sacado::Fad::DFad<double> >
    class Material_Hyperelastic
    {
        public:
          Material_Hyperelastic(const Parameters::AllParameters &parameters,
                                const Time                      &time)
            :
            n_OS (parameters.solid_vol_frac),
            lambda (parameters.lambda),
            time(time),
            det_F (1.0),
            det_F_converged (1.0),
            eigen_solver (parameters.eigen_solver)
           {}
          ~Material_Hyperelastic()
          {}

          SymmetricTensor<2, dim, NumberType>
          get_tau_E(const Tensor<2,dim, NumberType> &F) const
          {
            return ( get_tau_E_base(F) + get_tau_E_ext_func(F) );
          }

          SymmetricTensor<2, dim, NumberType>
          get_Cauchy_E(const Tensor<2, dim, NumberType> &F) const
          {
              const NumberType det_F = determinant(F);
              Assert(det_F > 0, ExcInternalError());
              return get_tau_E(F)*NumberType(1/det_F);
          }

          double
          get_converged_det_F() const
          {
              return  det_F_converged;
          }

          virtual void
          update_end_timestep()
          {
              det_F_converged = det_F;
          }

          virtual void
          update_internal_equilibrium( const Tensor<2, dim, NumberType> &F )
          {
              det_F = Tensor<0,dim,double>(determinant(F));
          }

          virtual double
          get_viscous_dissipation( ) const = 0;

          const double n_OS;
          const double lambda;
          const Time  &time;
          double det_F;
          double det_F_converged;
          const enum SymmetricTensorEigenvectorMethod eigen_solver;

        protected:
          SymmetricTensor<2, dim, NumberType>
          get_tau_E_ext_func(const Tensor<2,dim, NumberType> &F) const
          {
              const NumberType det_F = determinant(F);
              Assert(det_F > 0, ExcInternalError());

              static const SymmetricTensor< 2, dim, double>
                    I (Physics::Elasticity::StandardTensors<dim>::I);
              return  ( NumberType(lambda * (1.0-n_OS)*(1.0-n_OS)
                         * (det_F/(1.0-n_OS) - det_F/(det_F-n_OS))) * I );
          }

          virtual SymmetricTensor<2, dim, NumberType>
           get_tau_E_base(const Tensor<2,dim, NumberType> &F) const = 0;
    };

//@sect4{Derived class: Neo-Hookean hyperelastic material}
    template <int dim, typename NumberType = Sacado::Fad::DFad<double> >
    class NeoHooke : public Material_Hyperelastic < dim, NumberType >
    {
        public:
            NeoHooke(const Parameters::AllParameters &parameters,
                     const Time                      &time)
            :
            Material_Hyperelastic< dim, NumberType > (parameters,time),
            mu(parameters.mu)
           {}
          virtual ~NeoHooke()
          {}

           double
           get_viscous_dissipation() const override
           {
               return 0.0;
           }

        protected:
          const double mu;

          SymmetricTensor<2, dim, NumberType>
          get_tau_E_base(const Tensor<2,dim, NumberType> &F) const override
          {
             static const SymmetricTensor< 2, dim, double>
                I (Physics::Elasticity::StandardTensors<dim>::I);

             const bool use_standard_model = true;

             if (use_standard_model)
             {
               // Standard Neo-Hooke
               return ( mu * ( symmetrize(F * transpose(F)) - I ) );
             }
             else
             {
               // Neo-Hooke in terms of principal stretches
               const SymmetricTensor<2, dim, NumberType>
                B = symmetrize(F * transpose(F));
               const std::array< std::pair< NumberType, Tensor< 1, dim, NumberType > >, dim >
                eigen_B = eigenvectors(B, this->eigen_solver);

               SymmetricTensor<2, dim, NumberType> B_ev;
               for (unsigned int d=0; d<dim; ++d)
                 B_ev += eigen_B[d].first*symmetrize(outer_product(eigen_B[d].second,eigen_B[d].second));

                return ( mu*(B_ev-I) );
             }
          }
    };

//@sect4{Derived class: Ogden hyperelastic material}
    template <int dim, typename NumberType = Sacado::Fad::DFad<double> >
    class Ogden : public Material_Hyperelastic < dim, NumberType >
    {
        public:
          Ogden(const Parameters::AllParameters &parameters,
                const Time                      &time)
          :
          Material_Hyperelastic< dim, NumberType > (parameters,time),
          mu({parameters.mu1_infty,
              parameters.mu2_infty,
              parameters.mu3_infty}),
          alpha({parameters.alpha1_infty,
                 parameters.alpha2_infty,
                 parameters.alpha3_infty})
           {}
          virtual ~Ogden()
          {}

           double
           get_viscous_dissipation() const override
           {
               return 0.0;
           }

        protected:
          std::vector<double> mu;
          std::vector<double> alpha;

          SymmetricTensor<2, dim, NumberType>
          get_tau_E_base(const Tensor<2,dim, NumberType> &F) const override
          {
            const SymmetricTensor<2, dim, NumberType>
             B = symmetrize(F * transpose(F));

            const std::array< std::pair< NumberType, Tensor< 1, dim, NumberType > >, dim >
             eigen_B = eigenvectors(B, this->eigen_solver);

            SymmetricTensor<2, dim, NumberType>  tau;
            static const SymmetricTensor< 2, dim, double>
              I (Physics::Elasticity::StandardTensors<dim>::I);

            for (unsigned int i = 0; i < 3; ++i)
            {
                for (unsigned int A = 0; A < dim; ++A)
                {
                    SymmetricTensor<2, dim, NumberType>  tau_aux1 = symmetrize(
                            outer_product(eigen_B[A].second,eigen_B[A].second));
                    tau_aux1 *= mu[i]*std::pow(eigen_B[A].first, (alpha[i]/2.) );
                    tau += tau_aux1;
                }
                SymmetricTensor<2, dim, NumberType>  tau_aux2 (I);
                tau_aux2 *= mu[i];
                tau -= tau_aux2;
            }
            return tau;
          }
    };

//@sect4{Derived class: Single-mode Ogden viscoelastic material}
// We use the finite viscoelastic model described in
// Reese & Govindjee (1998) doi:10.1016/S0020-7683(97)00217-5
// The algorithm for the implicit exponential time integration is given in
// Budday et al. (2017) doi: 10.1016/j.actbio.2017.06.024
    template <int dim, typename NumberType = Sacado::Fad::DFad<double> >
    class visco_Ogden : public Material_Hyperelastic < dim, NumberType >
    {
        public:
            visco_Ogden(const Parameters::AllParameters &parameters,
                        const Time                      &time)
            :
            Material_Hyperelastic< dim, NumberType > (parameters,time),
            mu_infty({parameters.mu1_infty,
                      parameters.mu2_infty,
                      parameters.mu3_infty}),
            alpha_infty({parameters.alpha1_infty,
                         parameters.alpha2_infty,
                         parameters.alpha3_infty}),
            mu_mode_1({parameters.mu1_mode_1,
                       parameters.mu2_mode_1,
                       parameters.mu3_mode_1}),
            alpha_mode_1({parameters.alpha1_mode_1,
                          parameters.alpha2_mode_1,
                          parameters.alpha3_mode_1}),
            viscosity_mode_1(parameters.viscosity_mode_1),
            Cinv_v_1(Physics::Elasticity::StandardTensors<dim>::I),
            Cinv_v_1_converged(Physics::Elasticity::StandardTensors<dim>::I)
            {}
            virtual ~visco_Ogden()
            {}

          void
          update_internal_equilibrium( const Tensor<2, dim, NumberType> &F ) override
          {
              Material_Hyperelastic < dim, NumberType >::update_internal_equilibrium(F);

              this->Cinv_v_1 = this->Cinv_v_1_converged;
              SymmetricTensor<2, dim, NumberType> B_e_1_tr = symmetrize(F * this->Cinv_v_1 * transpose(F));

              const std::array< std::pair< NumberType, Tensor< 1, dim, NumberType > >, dim >
                eigen_B_e_1_tr = eigenvectors(B_e_1_tr, this->eigen_solver);

              Tensor< 1, dim, NumberType > lambdas_e_1_tr;
              Tensor< 1, dim, NumberType > epsilon_e_1_tr;
              for (int a = 0; a < dim; ++a)
              {
                  lambdas_e_1_tr[a] = std::sqrt(eigen_B_e_1_tr[a].first);
                  epsilon_e_1_tr[a] = std::log(lambdas_e_1_tr[a]);
              }

             const double tolerance = 1e-8;
             double residual_check = tolerance*10.0;
             Tensor< 1, dim, NumberType > residual;
             Tensor< 2, dim, NumberType > tangent;
             static const SymmetricTensor< 2, dim, double> I(Physics::Elasticity::StandardTensors<dim>::I);
             NumberType J_e_1 = std::sqrt(determinant(B_e_1_tr));

             std::vector<NumberType> lambdas_e_1_iso(dim);
             SymmetricTensor<2, dim, NumberType> B_e_1;
             int iteration = 0;

             Tensor< 1, dim, NumberType > lambdas_e_1;
             Tensor< 1, dim, NumberType > epsilon_e_1;
             epsilon_e_1 = epsilon_e_1_tr;

              while(residual_check > tolerance)
              {
                  NumberType aux_J_e_1 = 1.0;
                  for (unsigned int a = 0; a < dim; ++a)
                  {
                      lambdas_e_1[a] = std::exp(epsilon_e_1[a]);
                      aux_J_e_1 *= lambdas_e_1[a];
                  }

                  J_e_1 = aux_J_e_1;

                  for (unsigned int a = 0; a < dim; ++a)
                      lambdas_e_1_iso[a] = lambdas_e_1[a]*std::pow(J_e_1,-1.0/dim);

                  for (unsigned int a = 0; a < dim; ++a)
                  {
                      residual[a] = get_beta_mode_1(lambdas_e_1_iso, a);
                      residual[a] *= this->time.get_delta_t()/(2.0*viscosity_mode_1);
                      residual[a] += epsilon_e_1[a];
                      residual[a] -= epsilon_e_1_tr[a];

                      for (unsigned int b = 0; b < dim; ++b)
                      {
                          tangent[a][b]  = get_gamma_mode_1(lambdas_e_1_iso, a, b);
                          tangent[a][b] *= this->time.get_delta_t()/(2.0*viscosity_mode_1);
                          tangent[a][b] += I[a][b];
                      }

                  }
                  epsilon_e_1 -= invert(tangent)*residual;

                  residual_check = 0.0;
                  for (unsigned int a = 0; a < dim; ++a)
                  {
                      if ( std::abs(residual[a]) > residual_check)
                          residual_check = std::abs(Tensor<0,dim,double>(residual[a]));
                  }
                  iteration += 1;
                  if (iteration > 15 )
                      AssertThrow(false, ExcMessage("No convergence in local Newton iteration for the "
                                                    "viscoelastic exponential time integration algorithm."));
              }

              NumberType aux_J_e_1 = 1.0;
              for (unsigned int a = 0; a < dim; ++a)
              {
                  lambdas_e_1[a] = std::exp(epsilon_e_1[a]);
                  aux_J_e_1 *= lambdas_e_1[a];
              }
              J_e_1 = aux_J_e_1;

              for (unsigned int a = 0; a < dim; ++a)
                  lambdas_e_1_iso[a] = lambdas_e_1[a]*std::pow(J_e_1,-1.0/dim);

              for (unsigned int a = 0; a < dim; ++a)
              {
                  SymmetricTensor<2, dim, NumberType>
                  B_e_1_aux = symmetrize(outer_product(eigen_B_e_1_tr[a].second,eigen_B_e_1_tr[a].second));
                  B_e_1_aux *= lambdas_e_1[a] * lambdas_e_1[a];
                  B_e_1 += B_e_1_aux;
              }

              Tensor<2, dim, NumberType>Cinv_v_1_AD = symmetrize(invert(F) * B_e_1 * invert(transpose(F)));

              this->tau_neq_1 = 0;
              for (unsigned int a = 0; a < dim; ++a)
              {
                  SymmetricTensor<2, dim, NumberType>
                  tau_neq_1_aux = symmetrize(outer_product(eigen_B_e_1_tr[a].second,eigen_B_e_1_tr[a].second));
                  tau_neq_1_aux *=  get_beta_mode_1(lambdas_e_1_iso, a);
                  this->tau_neq_1 += tau_neq_1_aux;
              }

              // Store history
              for (unsigned int a = 0; a < dim; ++a)
                  for (unsigned int b = 0; b < dim; ++b)
                      this->Cinv_v_1[a][b]= Tensor<0,dim,double>(Cinv_v_1_AD[a][b]);
          }

          void update_end_timestep() override
          {
              Material_Hyperelastic < dim, NumberType >::update_end_timestep();
              this->Cinv_v_1_converged = this->Cinv_v_1;
          }

           double get_viscous_dissipation() const override
           {
               NumberType dissipation_term = get_tau_E_neq() * get_tau_E_neq(); //Double contract the two SymmetricTensor
               dissipation_term /= (2*viscosity_mode_1);

               return dissipation_term.val();
           }

        protected:
          std::vector<double> mu_infty;
          std::vector<double> alpha_infty;
          std::vector<double> mu_mode_1;
          std::vector<double> alpha_mode_1;
          double viscosity_mode_1;
          SymmetricTensor<2, dim, double> Cinv_v_1;
          SymmetricTensor<2, dim, double> Cinv_v_1_converged;
          SymmetricTensor<2, dim, NumberType> tau_neq_1;

          SymmetricTensor<2, dim, NumberType>
          get_tau_E_base(const Tensor<2,dim, NumberType> &F) const override
          {
              return ( get_tau_E_neq() + get_tau_E_eq(F) );
          }

          SymmetricTensor<2, dim, NumberType>
          get_tau_E_eq(const Tensor<2,dim, NumberType> &F) const
          {
            const SymmetricTensor<2, dim, NumberType> B = symmetrize(F * transpose(F));

            std::array< std::pair< NumberType, Tensor< 1, dim, NumberType > >, dim > eigen_B;
            eigen_B = eigenvectors(B, this->eigen_solver);

            SymmetricTensor<2, dim, NumberType>  tau;
            static const SymmetricTensor< 2, dim, double>
              I (Physics::Elasticity::StandardTensors<dim>::I);

            for (unsigned int i = 0; i < 3; ++i)
            {
                for (unsigned int A = 0; A < dim; ++A)
                {
                    SymmetricTensor<2, dim, NumberType>  tau_aux1 = symmetrize(
                          outer_product(eigen_B[A].second,eigen_B[A].second));
                    tau_aux1 *= mu_infty[i]*std::pow(eigen_B[A].first, (alpha_infty[i]/2.) );
                    tau += tau_aux1;
                }
                SymmetricTensor<2, dim, NumberType>  tau_aux2 (I);
                tau_aux2 *= mu_infty[i];
                tau -= tau_aux2;
            }
            return tau;
          }

          SymmetricTensor<2, dim, NumberType>
          get_tau_E_neq() const
          {
              return tau_neq_1;
          }

          NumberType
          get_beta_mode_1(std::vector< NumberType > &lambda, const int &A) const
          {
              NumberType beta = 0.0;

              for (unsigned int i = 0; i < 3; ++i) //3rd-order Ogden model
              {

                  NumberType aux = 0.0;
                  for (int p = 0; p < dim; ++p)
                      aux += std::pow(lambda[p],alpha_mode_1[i]);

                  aux *= -1.0/dim;
                  aux += std::pow(lambda[A], alpha_mode_1[i]);
                  aux *= mu_mode_1[i];

                  beta  += aux;
              }
              return beta;
          }

          NumberType
          get_gamma_mode_1(std::vector< NumberType > &lambda,
                           const int                 &A,
                           const int                 &B       ) const
          {
              NumberType gamma = 0.0;

              if (A==B)
              {
                  for (unsigned int i = 0; i < 3; ++i)
                  {
                      NumberType aux = 0.0;
                      for (int p = 0; p < dim; ++p)
                          aux += std::pow(lambda[p],alpha_mode_1[i]);

                      aux *= 1.0/(dim*dim);
                      aux += 1.0/dim * std::pow(lambda[A], alpha_mode_1[i]);
                      aux *= mu_mode_1[i]*alpha_mode_1[i];

                      gamma += aux;
                  }
              }
              else
              {
                  for (unsigned int i = 0; i < 3; ++i)
                  {
                      NumberType aux = 0.0;
                      for (int p = 0; p < dim; ++p)
                          aux += std::pow(lambda[p],alpha_mode_1[i]);

                      aux *= 1.0/(dim*dim);
                      aux -= 1.0/dim * std::pow(lambda[A], alpha_mode_1[i]);
                      aux -= 1.0/dim * std::pow(lambda[B], alpha_mode_1[i]);
                      aux *= mu_mode_1[i]*alpha_mode_1[i];

                      gamma += aux;
                  }
              }

              return gamma;
          }
    };


// @sect3{Constitutive equation for the fluid component of the biphasic material}
// We consider two slightly different definitions to define the seepage velocity with a Darcy-like law.
// Ehlers & Eipper 1999, doi:10.1023/A:1006565509095
// Markert 2007, doi:10.1007/s11242-007-9107-6
// The selection of one or another is made by the user via the parameters file.
    template <int dim, typename NumberType = Sacado::Fad::DFad<double> >
    class Material_Darcy_Fluid
    {
       public:
         Material_Darcy_Fluid(const Parameters::AllParameters &parameters)
         :
         fluid_type(parameters.fluid_type),
         n_OS(parameters.solid_vol_frac),
         initial_intrinsic_permeability(parameters.init_intrinsic_perm),
         viscosity_FR(parameters.viscosity_FR),
         initial_darcy_coefficient(parameters.init_darcy_coef),
         weight_FR(parameters.weight_FR),
         kappa_darcy(parameters.kappa_darcy),
         gravity_term(parameters.gravity_term),
         density_FR(parameters.density_FR),
         gravity_direction(parameters.gravity_direction),
         gravity_value(parameters.gravity_value)
         {
           Assert(kappa_darcy >= 0, ExcInternalError());
         }
         ~Material_Darcy_Fluid()
         {}

         Tensor<1, dim, NumberType> get_seepage_velocity_current
                             (const Tensor<2,dim, NumberType> &F,
                              const Tensor<1,dim, NumberType> &grad_p_fluid) const
         {
             const NumberType det_F = determinant(F);
             Assert(det_F > 0.0, ExcInternalError());

             Tensor<2, dim, NumberType> permeability_term;

             if (fluid_type == "Markert")
                 permeability_term = get_instrinsic_permeability_current(F) / viscosity_FR;

             else if (fluid_type == "Ehlers")
                 permeability_term = get_darcy_flow_current(F) / weight_FR;

             else
                 AssertThrow(false, ExcMessage(
                   "Material_Darcy_Fluid --> Only Markert "
                   "and Ehlers formulations have been implemented."));

             return ( -1.0 * permeability_term * det_F
                      * (grad_p_fluid - get_body_force_FR_current()) );
         }

         double get_porous_dissipation(const Tensor<2,dim, NumberType> &F,
                                       const Tensor<1,dim, NumberType> &grad_p_fluid) const
         {
             NumberType dissipation_term;
             Tensor<1, dim, NumberType> seepage_velocity;
             Tensor<2, dim, NumberType> permeability_term;

             const NumberType det_F = determinant(F);
             Assert(det_F > 0.0, ExcInternalError());

             if (fluid_type == "Markert")
             {
                 permeability_term = get_instrinsic_permeability_current(F) / viscosity_FR;
                 seepage_velocity = get_seepage_velocity_current(F,grad_p_fluid);
             }
             else if (fluid_type == "Ehlers")
             {
                 permeability_term = get_darcy_flow_current(F) / weight_FR;
                 seepage_velocity = get_seepage_velocity_current(F,grad_p_fluid);
             }
             else
                 AssertThrow(false, ExcMessage(
                   "Material_Darcy_Fluid --> Only Markert and Ehlers "
                   "formulations have been implemented."));

             dissipation_term = ( invert(permeability_term) * seepage_velocity ) * seepage_velocity;
             dissipation_term *= 1.0/(det_F*det_F);
             return Tensor<0,dim,double>(dissipation_term);
         }

       protected:
         const std::string  fluid_type;
         const double n_OS;
         const double initial_intrinsic_permeability;
         const double viscosity_FR;
         const double initial_darcy_coefficient;
         const double weight_FR;
         const double kappa_darcy;
         const bool   gravity_term;
         const double density_FR;
         const int    gravity_direction;
         const double    gravity_value;

         Tensor<2, dim, NumberType>
         get_instrinsic_permeability_current(const Tensor<2,dim, NumberType> &F) const
         {
           static const SymmetricTensor< 2, dim, double>
               I (Physics::Elasticity::StandardTensors<dim>::I);
           const Tensor<2, dim, NumberType> initial_instrinsic_permeability_tensor
               = Tensor<2, dim, double>(initial_intrinsic_permeability * I);

           const NumberType det_F = determinant(F);
           Assert(det_F > 0.0, ExcInternalError());

           const NumberType fraction = (det_F - n_OS)/(1 - n_OS);
           return ( NumberType (std::pow(fraction, kappa_darcy))
                     * initial_instrinsic_permeability_tensor );
         }

         Tensor<2, dim, NumberType>
         get_darcy_flow_current(const Tensor<2,dim, NumberType> &F) const
         {
           static const SymmetricTensor< 2, dim, double>
              I (Physics::Elasticity::StandardTensors<dim>::I);
           const Tensor<2, dim, NumberType> initial_darcy_flow_tensor
              = Tensor<2, dim, double>(initial_darcy_coefficient * I);

           const NumberType det_F = determinant(F);
           Assert(det_F > 0.0, ExcInternalError());

           const NumberType fraction = (1.0 - (n_OS / det_F) )/(1.0 - n_OS);
           return ( NumberType (std::pow(fraction, kappa_darcy))
                     * initial_darcy_flow_tensor);
         }

        Tensor<1, dim, NumberType>
        get_body_force_FR_current() const
        {
            Tensor<1, dim, NumberType> body_force_FR_current;

            if (gravity_term == true)
            {
               Tensor<1, dim, NumberType> gravity_vector;
               gravity_vector[gravity_direction] = gravity_value;
               body_force_FR_current = density_FR * gravity_vector;
            }
            return body_force_FR_current;
        }
    };

// @sect3{Quadrature point history}
// As seen in step-18, the <code> PointHistory </code> class offers a method
// for storing data at the quadrature points.  Here each quadrature point
// holds a pointer to a material description.  Thus, different material models
// can be used in different regions of the domain.  Among other data, we
// choose to store the ``extra" Kirchhoff stress $\boldsymbol{\tau}_E$ and
// the dissipation values $\mathcal{D}_p$ and $\mathcal{D}_v$.
    template <int dim, typename NumberType = Sacado::Fad::DFad<double> > //double>
    class PointHistory
    {
        public:
            PointHistory()
            {}

            virtual ~PointHistory()
            {}

            void setup_lqp (const Parameters::AllParameters &parameters,
                            const Time                      &time)
            {
                if (parameters.mat_type == "Neo-Hooke")
                    solid_material.reset(new NeoHooke<dim,NumberType>(parameters,time));
                else if (parameters.mat_type == "Ogden")
                    solid_material.reset(new Ogden<dim,NumberType>(parameters,time));
                else if (parameters.mat_type == "visco-Ogden")
                    solid_material.reset(new visco_Ogden<dim,NumberType>(parameters,time));
                else
                    Assert (false, ExcMessage("Material type not implemented"));

                fluid_material.reset(new Material_Darcy_Fluid<dim,NumberType>(parameters));
            }

            SymmetricTensor<2, dim, NumberType>
            get_tau_E(const Tensor<2, dim, NumberType> &F) const
            {
                return solid_material->get_tau_E(F);
            }

            SymmetricTensor<2, dim, NumberType>
            get_Cauchy_E(const Tensor<2, dim, NumberType> &F) const
            {
                return solid_material->get_Cauchy_E(F);
            }

            double
            get_converged_det_F() const
            {
              return  solid_material->get_converged_det_F();
            }

            void
            update_end_timestep()
            {
                solid_material->update_end_timestep();
            }

            void
            update_internal_equilibrium(const Tensor<2, dim, NumberType> &F )
            {
                solid_material->update_internal_equilibrium(F);
            }

            double
            get_viscous_dissipation() const
            {
                return solid_material->get_viscous_dissipation();
            }

            Tensor<1,dim, NumberType>
            get_seepage_velocity_current (const Tensor<2,dim, NumberType> &F,
                                          const Tensor<1,dim, NumberType> &grad_p_fluid) const
             {
                 return fluid_material->get_seepage_velocity_current(F, grad_p_fluid);
             }

            double
            get_porous_dissipation(const Tensor<2,dim, NumberType> &F,
                                   const Tensor<1,dim, NumberType> &grad_p_fluid) const
            {
                return fluid_material->get_porous_dissipation(F, grad_p_fluid);
            }

            Tensor<1, dim, NumberType>
            get_overall_body_force (const Tensor<2,dim, NumberType> &F,
                                    const Parameters::AllParameters &parameters) const
            {
                Tensor<1, dim, NumberType> body_force;

                if (parameters.gravity_term == true)
                {
                    const NumberType det_F_AD = determinant(F);
                    Assert(det_F_AD > 0.0, ExcInternalError());

                    const NumberType overall_density_ref
                        = parameters.density_SR * parameters.solid_vol_frac
                          + parameters.density_FR
                          * (det_F_AD - parameters.solid_vol_frac);

                   Tensor<1, dim, NumberType> gravity_vector;
                   gravity_vector[parameters.gravity_direction] = parameters.gravity_value;
                   body_force = overall_density_ref * gravity_vector;
                }

                return body_force;
            }
        private:
            std::shared_ptr< Material_Hyperelastic<dim, NumberType> > solid_material;
            std::shared_ptr< Material_Darcy_Fluid<dim, NumberType> > fluid_material;
    };

// @sect3{Nonlinear poro-viscoelastic solid}
// The Solid class is the central class as it represents the problem at hand:
// the nonlinear poro-viscoelastic solid
    template <int dim>
    class Solid
    {
          public:
            Solid(const Parameters::AllParameters &parameters);
            virtual ~Solid();
            void run();

          protected:
            using ADNumberType = Sacado::Fad::DFad<double>;

            std::ofstream outfile;
            std::ofstream pointfile;

            struct PerTaskData_ASM;
            template<typename NumberType = double> struct ScratchData_ASM;

            //Generate mesh
            virtual void make_grid() = 0;

            //Define points for post-processing
            virtual void define_tracked_vertices(std::vector<Point<dim> > &tracked_vertices) = 0;

            //Set up the finite element system to be solved:
            void system_setup(TrilinosWrappers::MPI::BlockVector &solution_delta_OUT);

            //Extract sub-blocks from the global matrix
            void determine_component_extractors();

            // Several functions to assemble the system and right hand side matrices using multithreading.
            void assemble_system
                  (const TrilinosWrappers::MPI::BlockVector &solution_delta_OUT );
            void assemble_system_one_cell
                  (const typename DoFHandler<dim>::active_cell_iterator &cell,
                   ScratchData_ASM<ADNumberType> &scratch,
                   PerTaskData_ASM &data) const;
            void copy_local_to_global_system(const PerTaskData_ASM &data);

            // Define boundary conditions
            virtual void make_constraints(const int &it_nr);
            virtual void make_dirichlet_constraints(AffineConstraints<double> &constraints) = 0;
            virtual Tensor<1,dim> get_neumann_traction
                   (const types::boundary_id &boundary_id,
                    const Point<dim>         &pt,
                    const Tensor<1,dim>      &N) const = 0;
            virtual double get_prescribed_fluid_flow
                   (const types::boundary_id &boundary_id,
                    const Point<dim>         &pt) const = 0;
            virtual types::boundary_id
                     get_reaction_boundary_id_for_output () const = 0;
            virtual std::pair<types::boundary_id,types::boundary_id>
                     get_drained_boundary_id_for_output () const = 0;
            virtual std::vector<double> get_dirichlet_load
                    (const types::boundary_id   &boundary_id,
                     const int                  &direction) const = 0;

            // Create and update the quadrature points.
            void setup_qph();

            //Solve non-linear system using a Newton-Raphson scheme
            void solve_nonlinear_timestep(TrilinosWrappers::MPI::BlockVector &solution_delta_OUT);

            //Solve the linearized equations using a direct solver
            void solve_linear_system ( TrilinosWrappers::MPI::BlockVector &newton_update_OUT);

            //Retrieve the  solution
            TrilinosWrappers::MPI::BlockVector
            get_total_solution(const TrilinosWrappers::MPI::BlockVector &solution_delta_IN) const;

            // Store the converged values of the internal variables at the end of each timestep
            void update_end_timestep();

            //Post-processing and writing data to files
            void output_results_to_vtu(const unsigned int timestep,
                                       const double current_time,
                                       TrilinosWrappers::MPI::BlockVector solution) const;
            void output_results_to_plot(const unsigned int timestep,
                                        const double current_time,
                                        TrilinosWrappers::MPI::BlockVector solution,
                                        std::vector<Point<dim> > &tracked_vertices,
                                        std::ofstream &pointfile) const;

            // Headers and footer for the output files
            void print_console_file_header( std::ofstream &outfile) const;
            void print_plot_file_header(std::vector<Point<dim> > &tracked_vertices,
                                        std::ofstream &pointfile) const;
            void print_console_file_footer(std::ofstream &outfile) const;
            void print_plot_file_footer( std::ofstream &pointfile) const;

            // For parallel communication
            MPI_Comm                         mpi_communicator;
            const unsigned int               n_mpi_processes;
            const unsigned int               this_mpi_process;
            mutable ConditionalOStream       pcout;

            // A collection of the parameters used to describe the problem setup
            const Parameters::AllParameters &parameters;

            // Declare an instance of dealii Triangulation class (mesh)
            parallel::shared::Triangulation<dim>  triangulation;

            // Keep track of the current time and the time spent evaluating certain functions
            Time          time;
            TimerOutput   timerconsole;
            TimerOutput   timerfile;

            // A storage object for quadrature point information.
            CellDataStorage<typename Triangulation<dim>::cell_iterator, PointHistory<dim,ADNumberType> > quadrature_point_history;

            //Integers to store polynomial degree (needed for output)
            const unsigned int  degree_displ;
            const unsigned int  degree_pore;

            //Declare an instance of dealii FESystem class (finite element definition)
            const FESystem<dim> fe;

            //Declare an instance of dealii DoFHandler class (assign DoFs to mesh)
            DoFHandler<dim>     dof_handler_ref;

            //Integer to store DoFs per element (this value will be used often)
            const unsigned int  dofs_per_cell;

            //Declare an instance of dealii Extractor objects used to retrieve information from the solution vectors
            //We will use "u_fe" and "p_fluid_fe"as subscript in operator [] expressions on FEValues and FEFaceValues
            //objects to extract the components of the displacement vector and fluid pressure, respectively.
            const FEValuesExtractors::Vector u_fe;
            const FEValuesExtractors::Scalar p_fluid_fe;

            // Description of how the block-system is arranged. There are 3 blocks:
            //   0 - vector DOF displacements u
            //   1 - scalar DOF fluid pressure p_fluid
            static const unsigned int  n_blocks = 2;
            static const unsigned int  n_components = dim+1;
            static const unsigned int  first_u_component = 0;
            static const unsigned int  p_fluid_component = dim;

            enum
            {
              u_block = 0,
              p_fluid_block = 1
            };

            // Extractors
            const FEValuesExtractors::Scalar x_displacement;
            const FEValuesExtractors::Scalar y_displacement;
            const FEValuesExtractors::Scalar z_displacement;
            const FEValuesExtractors::Scalar pressure;

            // Block data
            std::vector<unsigned int> block_component;

            // DoF index data
            std::vector<IndexSet> all_locally_owned_dofs;
            IndexSet locally_owned_dofs;
            IndexSet locally_relevant_dofs;
            std::vector<IndexSet> locally_owned_partitioning;
            std::vector<IndexSet> locally_relevant_partitioning;

            std::vector<types::global_dof_index>   dofs_per_block;
            std::vector<types::global_dof_index>   element_indices_u;
            std::vector<types::global_dof_index>   element_indices_p_fluid;

            //Declare an instance of dealii QGauss class (The Gauss-Legendre family of quadrature rules for numerical integration)
            //Gauss Points in element, with n quadrature points (in each space direction <dim> )
            const QGauss<dim>                qf_cell;
            //Gauss Points on element faces (used for definition of BCs)
            const QGauss<dim - 1>            qf_face;
            //Integer to store num GPs per element (this value will be used often)
            const unsigned int               n_q_points;
            //Integer to store num GPs per face (this value will be used often)
            const unsigned int               n_q_points_f;

            //Declare an instance of dealii AffineConstraints class (linear constraints on DoFs due to hanging nodes or BCs)
            AffineConstraints<double>        constraints;

            //Declare an instance of dealii classes necessary for FE system set-up and assembly
            //Store elements of tangent matrix (indicated by SparsityPattern class) as sparse matrix (more efficient)
            TrilinosWrappers::BlockSparseMatrix tangent_matrix;
            TrilinosWrappers::BlockSparseMatrix tangent_matrix_preconditioner;
            //Right hand side vector of forces
            TrilinosWrappers::MPI::BlockVector  system_rhs;
            //Total displacement values + pressure (accumulated solution to FE system)
            TrilinosWrappers::MPI::BlockVector  solution_n;

            // Non-block system for the direct solver. We will copy the block system into these to solve the linearized system of equations.
            TrilinosWrappers::SparseMatrix tangent_matrix_nb;
            TrilinosWrappers::MPI::Vector  system_rhs_nb;

            //We define variables to store norms and update norms and normalisation factors.
            struct Errors
            {
              Errors()
                :
                norm(1.0), u(1.0), p_fluid(1.0)
              {}

              void reset()
              {
                norm = 1.0;
                u = 1.0;
                p_fluid = 1.0;
              }
              void normalise(const Errors &rhs)
              {
                if (rhs.norm != 0.0)
                  norm /= rhs.norm;
                if (rhs.u != 0.0)
                  u /= rhs.u;
                if (rhs.p_fluid != 0.0)
                  p_fluid /= rhs.p_fluid;
              }

              double norm, u, p_fluid;
            };

            //Declare several instances of the "Error" structure
            Errors error_residual, error_residual_0, error_residual_norm, error_update,
                   error_update_0, error_update_norm;

            // Methods to calculate error measures
            void get_error_residual(Errors &error_residual_OUT);
            void get_error_update
                 (const TrilinosWrappers::MPI::BlockVector &newton_update_IN,
                  Errors                                   &error_update_OUT);

            // Print information to screen
            void print_conv_header();
            void print_conv_footer();

//NOTE: In all functions, we pass by reference (&), so these functions work on the original copy (not a clone copy),
//      modifying the input variables inside the functions will change them outside the function.
    };

// @sect3{Implementation of the <code>Solid</code> class}
// @sect4{Public interface}
// We initialise the Solid class using data extracted from the parameter file.
    template <int dim>
    Solid<dim>::Solid(const Parameters::AllParameters &parameters)
        :
        mpi_communicator(MPI_COMM_WORLD),
        n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
        this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
        pcout(std::cout, this_mpi_process == 0),
        parameters(parameters),
        triangulation(mpi_communicator,Triangulation<dim>::maximum_smoothing),
        time(parameters.end_time, parameters.delta_t),
        timerconsole( mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times),
        timerfile( mpi_communicator,
                   outfile,
                   TimerOutput::summary,
                   TimerOutput::wall_times),
        degree_displ(parameters.poly_degree_displ),
        degree_pore(parameters.poly_degree_pore),
        fe( FE_Q<dim>(parameters.poly_degree_displ), dim,
            FE_Q<dim>(parameters.poly_degree_pore), 1 ),
        dof_handler_ref(triangulation),
        dofs_per_cell (fe.dofs_per_cell),
        u_fe(first_u_component),
        p_fluid_fe(p_fluid_component),
        x_displacement(first_u_component),
        y_displacement(first_u_component+1),
        z_displacement(first_u_component+2),
        pressure(p_fluid_component),
        dofs_per_block(n_blocks),
        qf_cell(parameters.quad_order),
        qf_face(parameters.quad_order),
        n_q_points (qf_cell.size()),
        n_q_points_f (qf_face.size())
        {
         Assert(dim==3, ExcMessage("This problem only works in 3 space dimensions."));
         determine_component_extractors();
        }

    //The class destructor simply clears the data held by the DOFHandler
    template <int dim>
    Solid<dim>::~Solid()
    {
        dof_handler_ref.clear();
    }

//Runs the 3D solid problem
    template <int dim>
    void Solid<dim>::run()
    {
          //The current solution increment is defined as a block vector to reflect the structure
          //of the PDE system, with multiple solution components
          TrilinosWrappers::MPI::BlockVector solution_delta;

          //Open file
          if (this_mpi_process == 0)
          {
              outfile.open("console-output.sol");
              print_console_file_header(outfile);
          }

          //Generate mesh
          make_grid();

          //Assign DOFs and create the stiffness and right-hand-side force vector
          system_setup(solution_delta);

          //Define points for post-processing
          std::vector<Point<dim> > tracked_vertices (2);
          define_tracked_vertices(tracked_vertices);
          std::vector<Point<dim>> reaction_force;

          if (this_mpi_process == 0)
          {
              pointfile.open("data-for-gnuplot.sol");
              print_plot_file_header(tracked_vertices, pointfile);
          }

          //Print results to output file
          if (parameters.outfiles_requested == "true")
          {
                output_results_to_vtu(time.get_timestep(),
                                      time.get_current(),
                                      solution_n           );
          }

          output_results_to_plot(time.get_timestep(),
                                 time.get_current(),
                                 solution_n,
                                 tracked_vertices,
                                 pointfile);

          //Increment time step (=load step)
          //NOTE: In solving the quasi-static problem, the time becomes a loading parameter,
          //i.e. we increase the loading linearly with time, making the two concepts interchangeable.
          time.increment_time();

          //Print information on screen
          pcout << "\nSolver:";
          pcout << "\n  CST     = make constraints";
          pcout << "\n  ASM_SYS = assemble system";
          pcout << "\n  SLV     = linear solver \n";

          //Print information on file
          outfile << "\nSolver:";
          outfile << "\n  CST     = make constraints";
          outfile << "\n  ASM_SYS = assemble system";
          outfile << "\n  SLV     = linear solver \n";

          while ( (time.get_end() - time.get_current()) > -1.0*parameters.tol_u )
            {
              //Initialize the current solution increment to zero
              solution_delta = 0.0;

              //Solve the non-linear system using a Newton-Rapshon scheme
              solve_nonlinear_timestep(solution_delta);

              //Add the computed solution increment to total solution
              solution_n += solution_delta;

              //Store the converged values of the internal variables
              update_end_timestep();

              //Output results
              if (( (time.get_timestep()%parameters.timestep_output) == 0 )
                   && (parameters.outfiles_requested == "true") )
              {
                      output_results_to_vtu(time.get_timestep(),
                                            time.get_current(),
                                            solution_n           );
              }

              output_results_to_plot(time.get_timestep(),
                                     time.get_current(),
                                     solution_n,
                                     tracked_vertices,
                                     pointfile);

              //Increment the time step (=load step)
              time.increment_time();
            }

          //Print the footers and close files
          if (this_mpi_process == 0)
          {
              print_plot_file_footer(pointfile);
              pointfile.close ();
              print_console_file_footer(outfile);

              //NOTE: ideally, we should close the outfile here [ >> outfile.close (); ]
              //But if we do, then the timer output will not be printed. That is why we leave it open.
          }
    }

// @sect4{Private interface}
// We define the structures needed for parallelization with Threading Building Blocks (TBB)
// Tangent matrix and right-hand side force vector assembly structures.
// PerTaskData_ASM stores local contributions
    template <int dim>
    struct Solid<dim>::PerTaskData_ASM
    {
        FullMatrix<double>        cell_matrix;
        Vector<double>            cell_rhs;
        std::vector<types::global_dof_index> local_dof_indices;

        PerTaskData_ASM(const unsigned int dofs_per_cell)
          :
          cell_matrix(dofs_per_cell, dofs_per_cell),
          cell_rhs(dofs_per_cell),
          local_dof_indices(dofs_per_cell)
        {}

        void reset()
        {
          cell_matrix = 0.0;
          cell_rhs = 0.0;
        }
    };

    // ScratchData_ASM stores larger objects used during the assembly
    template <int dim>
    template <typename NumberType>
    struct Solid<dim>::ScratchData_ASM
    {
        const TrilinosWrappers::MPI::BlockVector &solution_total;

        //Integration helper
        FEValues<dim>     fe_values_ref;
        FEFaceValues<dim> fe_face_values_ref;

        // Quadrature point solution
        std::vector<NumberType>                  local_dof_values;
        std::vector<Tensor<2, dim, NumberType> > solution_grads_u_total;
        std::vector<NumberType>                  solution_values_p_fluid_total;
        std::vector<Tensor<1, dim, NumberType> > solution_grads_p_fluid_total;
        std::vector<Tensor<1, dim, NumberType> > solution_grads_face_p_fluid_total;

        //shape function values
        std::vector<std::vector<Tensor<1,dim>>>          Nx;
        std::vector<std::vector<double>>                 Nx_p_fluid;
        //shape function gradients
        std::vector<std::vector<Tensor<2,dim, NumberType>>>          grad_Nx;
        std::vector<std::vector<SymmetricTensor<2,dim, NumberType>>> symm_grad_Nx;
        std::vector<std::vector<Tensor<1,dim, NumberType>>>          grad_Nx_p_fluid;

        ScratchData_ASM(const FiniteElement<dim> &fe_cell,
                        const QGauss<dim> &qf_cell, const UpdateFlags uf_cell,
                        const QGauss<dim - 1> & qf_face, const UpdateFlags uf_face,
                        const TrilinosWrappers::MPI::BlockVector &solution_total    )
          :
          solution_total (solution_total),
          fe_values_ref(fe_cell, qf_cell, uf_cell),
          fe_face_values_ref(fe_cell, qf_face, uf_face),
          local_dof_values(fe_cell.dofs_per_cell),
          solution_grads_u_total(qf_cell.size()),
          solution_values_p_fluid_total(qf_cell.size()),
          solution_grads_p_fluid_total(qf_cell.size()),
          solution_grads_face_p_fluid_total(qf_face.size()),
          Nx(qf_cell.size(), std::vector<Tensor<1,dim>>(fe_cell.dofs_per_cell)),
          Nx_p_fluid(qf_cell.size(), std::vector<double>(fe_cell.dofs_per_cell)),
          grad_Nx(qf_cell.size(), std::vector<Tensor<2, dim, NumberType>>(fe_cell.dofs_per_cell)),
          symm_grad_Nx(qf_cell.size(), std::vector<SymmetricTensor<2, dim, NumberType>> (fe_cell.dofs_per_cell)),
          grad_Nx_p_fluid(qf_cell.size(), std::vector<Tensor<1, dim, NumberType>>(fe_cell.dofs_per_cell))
        {}

        ScratchData_ASM(const ScratchData_ASM &rhs)
          :
          solution_total (rhs.solution_total),
          fe_values_ref(rhs.fe_values_ref.get_fe(),
                        rhs.fe_values_ref.get_quadrature(),
                        rhs.fe_values_ref.get_update_flags()),
          fe_face_values_ref(rhs.fe_face_values_ref.get_fe(),
                             rhs.fe_face_values_ref.get_quadrature(),
                             rhs.fe_face_values_ref.get_update_flags()),
          local_dof_values(rhs.local_dof_values),
          solution_grads_u_total(rhs.solution_grads_u_total),
          solution_values_p_fluid_total(rhs.solution_values_p_fluid_total),
          solution_grads_p_fluid_total(rhs.solution_grads_p_fluid_total),
          solution_grads_face_p_fluid_total(rhs.solution_grads_face_p_fluid_total),
          Nx(rhs.Nx),
          Nx_p_fluid(rhs.Nx_p_fluid),
          grad_Nx(rhs.grad_Nx),
          symm_grad_Nx(rhs.symm_grad_Nx),
          grad_Nx_p_fluid(rhs.grad_Nx_p_fluid)
        {}

        void reset()
        {
          const unsigned int n_q_points = Nx_p_fluid.size();
          const unsigned int n_dofs_per_cell = Nx_p_fluid[0].size();

          Assert(local_dof_values.size() == n_dofs_per_cell, ExcInternalError());

          for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
            {
              local_dof_values[k] = 0.0;
            }

          Assert(solution_grads_u_total.size() == n_q_points, ExcInternalError());
          Assert(solution_values_p_fluid_total.size() == n_q_points, ExcInternalError());
          Assert(solution_grads_p_fluid_total.size() == n_q_points, ExcInternalError());

          Assert(Nx.size() == n_q_points, ExcInternalError());
          Assert(grad_Nx.size() == n_q_points, ExcInternalError());
          Assert(symm_grad_Nx.size() == n_q_points, ExcInternalError());

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
              Assert( Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());
              Assert( grad_Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());
              Assert( symm_grad_Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());

              solution_grads_u_total[q_point] = 0.0;
              solution_values_p_fluid_total[q_point] = 0.0;
              solution_grads_p_fluid_total[q_point] = 0.0;

              for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
                {
                  Nx[q_point][k] = 0.0;
                  Nx_p_fluid[q_point][k] = 0.0;
                  grad_Nx[q_point][k] = 0.0;
                  symm_grad_Nx[q_point][k] = 0.0;
                  grad_Nx_p_fluid[q_point][k] = 0.0;
                }
            }

          const unsigned int n_f_q_points = solution_grads_face_p_fluid_total.size();
          Assert(solution_grads_face_p_fluid_total.size() == n_f_q_points, ExcInternalError());

          for (unsigned int f_q_point = 0; f_q_point < n_f_q_points; ++f_q_point)
              solution_grads_face_p_fluid_total[f_q_point] = 0.0;
        }
    };

    //Define the boundary conditions on the mesh
    template <int dim>
    void Solid<dim>::make_constraints(const int &it_nr_IN)
    {
        pcout     << " CST " << std::flush;
        outfile   << " CST " << std::flush;

        if (it_nr_IN > 1) return;

        const bool apply_dirichlet_bc = (it_nr_IN == 0);

        if (apply_dirichlet_bc)
        {
          constraints.clear();
          make_dirichlet_constraints(constraints);
        }
        else
        {
          for (unsigned int i=0; i<dof_handler_ref.n_dofs(); ++i)
            if (constraints.is_inhomogeneously_constrained(i) == true)
              constraints.set_inhomogeneity(i,0.0);
        }
        constraints.close();
    }

    //Set-up the FE system
    template <int dim>
    void Solid<dim>::system_setup(TrilinosWrappers::MPI::BlockVector &solution_delta_OUT)
    {
        timerconsole.enter_subsection("Setup system");
        timerfile.enter_subsection("Setup system");

        //Determine number of components per block
        std::vector<unsigned int> block_component(n_components, u_block);
        block_component[p_fluid_component] = p_fluid_block;

        // The DOF handler is initialised and we renumber the grid in an efficient manner.
        dof_handler_ref.distribute_dofs(fe);
        DoFRenumbering::Cuthill_McKee(dof_handler_ref);
        DoFRenumbering::component_wise(dof_handler_ref, block_component);

        // Count the number of DoFs in each block
        dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler_ref, block_component);

        // Setup the sparsity pattern and tangent matrix
        all_locally_owned_dofs = DoFTools::locally_owned_dofs_per_subdomain (dof_handler_ref);
        std::vector<IndexSet> all_locally_relevant_dofs
        = DoFTools::locally_relevant_dofs_per_subdomain (dof_handler_ref);

        locally_owned_dofs.clear();
        locally_owned_partitioning.clear();
        Assert(all_locally_owned_dofs.size() > this_mpi_process, ExcInternalError());
        locally_owned_dofs = all_locally_owned_dofs[this_mpi_process];

        locally_relevant_dofs.clear();
        locally_relevant_partitioning.clear();
        Assert(all_locally_relevant_dofs.size() > this_mpi_process, ExcInternalError());
        locally_relevant_dofs = all_locally_relevant_dofs[this_mpi_process];

        locally_owned_partitioning.reserve(n_blocks);
        locally_relevant_partitioning.reserve(n_blocks);

        for (unsigned int b=0; b<n_blocks; ++b)
          {
            const types::global_dof_index idx_begin
            = std::accumulate(dofs_per_block.begin(),
                              std::next(dofs_per_block.begin(),b), 0);
            const types::global_dof_index idx_end
            = std::accumulate(dofs_per_block.begin(),
                              std::next(dofs_per_block.begin(),b+1), 0);
            locally_owned_partitioning.push_back(locally_owned_dofs.get_view(idx_begin, idx_end));
            locally_relevant_partitioning.push_back(locally_relevant_dofs.get_view(idx_begin, idx_end));
          }

        //Print information on screen
        pcout  << "\nTriangulation:\n"
               << "  Number of active cells: "
               << triangulation.n_active_cells()
               << " (by partition:";
        for (unsigned int p=0; p<n_mpi_processes; ++p)
          pcout  << (p==0 ? ' ' : '+')
                 << (GridTools::count_cells_with_subdomain_association (triangulation,p));
        pcout << ")"
              << std::endl;
        pcout << "  Number of degrees of freedom: "
              << dof_handler_ref.n_dofs()
              << " (by partition:";
        for (unsigned int p=0; p<n_mpi_processes; ++p)
          pcout  << (p==0 ? ' ' : '+')
                 << (DoFTools::count_dofs_with_subdomain_association (dof_handler_ref,p));
        pcout << ")"
              << std::endl;
        pcout   << "  Number of degrees of freedom per block: "
            << "[n_u, n_p_fluid] = ["
            << dofs_per_block[u_block]
            << ", "
            << dofs_per_block[p_fluid_block]
            << "]"
            << std::endl;

        //Print information to file
        outfile  << "\nTriangulation:\n"
                 <<  "  Number of active cells: "
                << triangulation.n_active_cells()
                << " (by partition:";
        for (unsigned int p=0; p<n_mpi_processes; ++p)
          outfile << (p==0 ? ' ' : '+')
                  << (GridTools::count_cells_with_subdomain_association (triangulation,p));
        outfile << ")"
                << std::endl;
        outfile << "  Number of degrees of freedom: "
                << dof_handler_ref.n_dofs()
                << " (by partition:";
        for (unsigned int p=0; p<n_mpi_processes; ++p)
          outfile  << (p==0 ? ' ' : '+')
                   << (DoFTools::count_dofs_with_subdomain_association (dof_handler_ref,p));
        outfile << ")"
                << std::endl;
        outfile << "  Number of degrees of freedom per block: "
            << "[n_u, n_p_fluid] = ["
            << dofs_per_block[u_block]
            << ", "
            << dofs_per_block[p_fluid_block]
            << "]"
            << std::endl;

        // We optimise the sparsity pattern to reflect this structure and prevent
        // unnecessary data creation for the right-diagonal block components.
        Table<2, DoFTools::Coupling> coupling(n_components, n_components);
        for (unsigned int ii = 0; ii < n_components; ++ii)
          for (unsigned int jj = 0; jj < n_components; ++jj)

            //Identify "zero" matrix components of FE-system (The two components do not couple)
            if (((ii == p_fluid_component) && (jj < p_fluid_component))
                || ((ii < p_fluid_component) && (jj == p_fluid_component)) )
              coupling[ii][jj] = DoFTools::none;

            //The rest of components always couple
            else
              coupling[ii][jj] = DoFTools::always;

        TrilinosWrappers::BlockSparsityPattern bsp (locally_owned_partitioning,
                                                    mpi_communicator);

        DoFTools::make_sparsity_pattern (dof_handler_ref, bsp, constraints,
                                         false, this_mpi_process);
        bsp.compress();

        //Reinitialize the (sparse) tangent matrix with the given sparsity pattern.
        tangent_matrix.reinit (bsp);

        //Initialize the right hand side and solution vectors with number of DoFs
        system_rhs.reinit(locally_owned_partitioning, mpi_communicator);
        solution_n.reinit(locally_owned_partitioning, mpi_communicator);
        solution_delta_OUT.reinit(locally_owned_partitioning, mpi_communicator);

        // Non-block system
        TrilinosWrappers::SparsityPattern sp (locally_owned_dofs,
                                              mpi_communicator);
        DoFTools::make_sparsity_pattern (dof_handler_ref, sp, constraints,
                                         false, this_mpi_process);
        sp.compress();
        tangent_matrix_nb.reinit (sp);
        system_rhs_nb.reinit(locally_owned_dofs, mpi_communicator);

        //Set up the quadrature point history
        setup_qph();

        timerconsole.leave_subsection();
        timerfile.leave_subsection();
    }

    //Component extractors: used to extract sub-blocks from the global matrix
    //Description of which local element DOFs are attached to which block component
    template <int dim>
    void Solid<dim>::determine_component_extractors()
    {
        element_indices_u.clear();
        element_indices_p_fluid.clear();

        for (unsigned int k = 0; k < fe.dofs_per_cell; ++k)
          {
            const unsigned int k_group = fe.system_to_base_index(k).first.first;
            if (k_group == u_block)
              element_indices_u.push_back(k);
            else if (k_group == p_fluid_block)
              element_indices_p_fluid.push_back(k);
            else
              {
                Assert(k_group <= p_fluid_block, ExcInternalError());
              }
          }
    }

    //Set-up quadrature point history (QPH) data objects
    template <int dim>
    void Solid<dim>::setup_qph()
    {
        pcout     << "\nSetting up quadrature point data..." << std::endl;
        outfile   << "\nSetting up quadrature point data..." << std::endl;

        //Create QPH data objects.
        quadrature_point_history.initialize(triangulation.begin_active(),
                                            triangulation.end(), n_q_points);

        //Setup the initial quadrature point data using the info stored in parameters
        FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
        cell (IteratorFilters::LocallyOwnedCell(),
              dof_handler_ref.begin_active()),
        endc (IteratorFilters::LocallyOwnedCell(),
              dof_handler_ref.end());
        for (; cell!=endc; ++cell)
          {
            Assert(cell->is_locally_owned(), ExcInternalError());
            Assert(cell->subdomain_id() == this_mpi_process, ExcInternalError());

            const std::vector<std::shared_ptr<PointHistory<dim, ADNumberType> > >
                lqph = quadrature_point_history.get_data(cell);
            Assert(lqph.size() == n_q_points, ExcInternalError());

            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
              lqph[q_point]->setup_lqp(parameters, time);
          }
    }

    //Solve the non-linear system using a Newton-Raphson scheme
    template <int dim>
    void Solid<dim>::solve_nonlinear_timestep(TrilinosWrappers::MPI::BlockVector &solution_delta_OUT)
    {
        //Print the load step
        pcout  << std::endl
               << "\nTimestep "
               << time.get_timestep()
               << " @ "
               << time.get_current()
               << "s"
               << std::endl;
        outfile  << std::endl
                 << "\nTimestep "
                 << time.get_timestep()
                 << " @ "
                 << time.get_current()
                 << "s"
                 << std::endl;

        //Declare newton_update vector (solution of a Newton iteration),
        //which must have as many positions as global DoFs.
        TrilinosWrappers::MPI::BlockVector newton_update
            (locally_owned_partitioning, mpi_communicator);

        //Reset the error storage objects
        error_residual.reset();
        error_residual_0.reset();
        error_residual_norm.reset();
        error_update.reset();
        error_update_0.reset();
        error_update_norm.reset();

        print_conv_header();

        //Declare and initialize iterator for the Newton-Raphson algorithm steps
        unsigned int newton_iteration = 0;

        //Iterate until error is below tolerance or max number iterations are reached
        while(newton_iteration < parameters.max_iterations_NR)
          {
            pcout     << " " << std::setw(2) << newton_iteration << " " << std::flush;
            outfile   << " " << std::setw(2) << newton_iteration << " " << std::flush;

            //Initialize global stiffness matrix and global force vector to zero
            tangent_matrix = 0.0;
            system_rhs = 0.0;

            tangent_matrix_nb = 0.0;
            system_rhs_nb = 0.0;

            //Apply boundary conditions
            make_constraints(newton_iteration);
            assemble_system(solution_delta_OUT);

            //Compute the rhs residual (error between external and internal forces in FE system)
            get_error_residual(error_residual);

            //error_residual in first iteration is stored to normalize posterior error measures
            if (newton_iteration == 0)
              error_residual_0 = error_residual;

            // Determine the normalised residual error
            error_residual_norm = error_residual;
            error_residual_norm.normalise(error_residual_0);

            //If both errors are below the tolerances, exit the loop.
            // We need to check the residual vector directly for convergence
            // in the load steps where no external forces or displacements are imposed.
            if (  ((newton_iteration > 0)
                && (error_update_norm.u <= parameters.tol_u)
                && (error_update_norm.p_fluid <= parameters.tol_p_fluid)
                && (error_residual_norm.u <= parameters.tol_f)
                && (error_residual_norm.p_fluid  <= parameters.tol_f))
                || ( (newton_iteration > 0)
                    && system_rhs.l2_norm() <= parameters.tol_f) )
              {
                pcout   << "\n ***** CONVERGED! *****     "
                        << system_rhs.l2_norm() << "      "
                        << "  " << error_residual_norm.norm
                        << "  " << error_residual_norm.u
                        << "  " << error_residual_norm.p_fluid
                        << "        " << error_update_norm.norm
                        << "  " << error_update_norm.u
                        << "  " << error_update_norm.p_fluid
                        << "  " << std::endl;
                outfile   << "\n ***** CONVERGED! *****     "
                        << system_rhs.l2_norm() << "      "
                        << "  " << error_residual_norm.norm
                        << "  " << error_residual_norm.u
                        << "  " << error_residual_norm.p_fluid
                        << "        " << error_update_norm.norm
                        << "  " << error_update_norm.u
                        << "  " << error_update_norm.p_fluid
                        << "  " << std::endl;
                print_conv_footer();

                break;
              }

            //Solve the linearized system
            solve_linear_system(newton_update);
            constraints.distribute(newton_update);

            //Compute the displacement error
            get_error_update(newton_update, error_update);

            //error_update in first iteration is stored to normalize posterior error measures
            if (newton_iteration == 0)
              error_update_0 = error_update;

            // Determine the normalised Newton update error
            error_update_norm = error_update;
            error_update_norm.normalise(error_update_0);

            // Determine the normalised residual error
            error_residual_norm = error_residual;
            error_residual_norm.normalise(error_residual_0);

            //Print error values
            pcout    << " |   " << std::fixed << std::setprecision(3)
            << std::setw(7) << std::scientific
            << system_rhs.l2_norm()
            << "        " << error_residual_norm.norm
            << "  " << error_residual_norm.u
            << "  " << error_residual_norm.p_fluid
            << "        " << error_update_norm.norm
            << "  " << error_update_norm.u
            << "  " << error_update_norm.p_fluid
            << "  " << std::endl;

            outfile  << " |   " << std::fixed << std::setprecision(3)
            << std::setw(7) << std::scientific
            << system_rhs.l2_norm()
            << "        " << error_residual_norm.norm
            << "  " << error_residual_norm.u
            << "  " << error_residual_norm.p_fluid
            << "        " << error_update_norm.norm
            << "  " << error_update_norm.u
            << "  " << error_update_norm.p_fluid
            << "  " << std::endl;

            // Update
            solution_delta_OUT += newton_update;
            newton_update = 0.0;
            newton_iteration++;
          }

        //If maximum allowed number of iterations for Newton algorithm are reached, print non-convergence message and abort program
        AssertThrow (newton_iteration < parameters.max_iterations_NR, ExcMessage("No convergence in nonlinear solver!"));
    }

    //Prints the header for convergence info on console
    template <int dim>
    void Solid<dim>::print_conv_header()
    {
        static const unsigned int l_width = 120;

        for (unsigned int i = 0; i < l_width; ++i)
          {
            pcout     << "_";
            outfile   << "_";
          }

        pcout     << std::endl;
        outfile   << std::endl;

        pcout   << "\n       SOLVER STEP      |    SYS_RES         "
                << "RES_NORM     RES_U      RES_P           "
                << "NU_NORM     NU_U       NU_P " << std::endl;
        outfile << "\n       SOLVER STEP      |    SYS_RES         "
                << "RES_NORM     RES_U      RES_P           "
                << "NU_NORM     NU_U       NU_P " << std::endl;

        for (unsigned int i = 0; i < l_width; ++i)
          {
            pcout     << "_";
            outfile   << "_";
          }
        pcout     << std::endl << std::endl;
        outfile   << std::endl << std::endl;
    }

    //Prints the footer for convergence info on console
    template <int dim>
    void Solid<dim>::print_conv_footer()
    {
        static const unsigned int l_width = 120;

        for (unsigned int i = 0; i < l_width; ++i)
          {
            pcout     << "_";
            outfile   << "_";
          }
        pcout     << std::endl << std::endl;
        outfile   << std::endl << std::endl;

        pcout << "Relative errors:" << std::endl
              << "Displacement:  "
              << error_update.u / error_update_0.u << std::endl
              << "Force (displ): "
              << error_residual.u / error_residual_0.u << std::endl
              << "Pore pressure: "
              << error_update.p_fluid / error_update_0.p_fluid << std::endl
              << "Force (pore):  "
              << error_residual.p_fluid / error_residual_0.p_fluid << std::endl;
        outfile << "Relative errors:" << std::endl
                << "Displacement:  "
                << error_update.u / error_update_0.u << std::endl
                << "Force (displ): "
                << error_residual.u / error_residual_0.u << std::endl
                << "Pore pressure: "
                << error_update.p_fluid / error_update_0.p_fluid << std::endl
                << "Force (pore):  "
                << error_residual.p_fluid / error_residual_0.p_fluid << std::endl;
    }

    //Determine the true residual error for the problem
    template <int dim>
    void Solid<dim>::get_error_residual(Errors &error_residual_OUT)
    {
        TrilinosWrappers::MPI::BlockVector error_res(system_rhs);
        constraints.set_zero(error_res);

        error_residual_OUT.norm = error_res.l2_norm();
        error_residual_OUT.u = error_res.block(u_block).l2_norm();
        error_residual_OUT.p_fluid = error_res.block(p_fluid_block).l2_norm();
    }

    //Determine the true Newton update error for the problem
    template <int dim>
    void Solid<dim>::get_error_update
               (const TrilinosWrappers::MPI::BlockVector &newton_update_IN,
                Errors                                   &error_update_OUT)
    {
        TrilinosWrappers::MPI::BlockVector error_ud(newton_update_IN);
        constraints.set_zero(error_ud);

        error_update_OUT.norm = error_ud.l2_norm();
        error_update_OUT.u = error_ud.block(u_block).l2_norm();
        error_update_OUT.p_fluid = error_ud.block(p_fluid_block).l2_norm();
    }

    //Compute the total solution, which is valid at any Newton step. This is required as, to reduce
    //computational error, the total solution is only updated at the end of the timestep.
    template <int dim>
    TrilinosWrappers::MPI::BlockVector
    Solid<dim>::get_total_solution(const TrilinosWrappers::MPI::BlockVector &solution_delta_IN) const
    {
        // Cell interpolation -> Ghosted vector
        TrilinosWrappers::MPI::BlockVector
             solution_total (locally_owned_partitioning,
                             locally_relevant_partitioning,
                             mpi_communicator,
                             /*vector_writable = */ false);
        TrilinosWrappers::MPI::BlockVector tmp (solution_total);
        solution_total = solution_n;
        tmp = solution_delta_IN;
        solution_total += tmp;
        return solution_total;
    }

    //Compute elemental stiffness tensor and right-hand side force vector, and assemble into global ones
    template <int dim>
    void Solid<dim>::assemble_system( const TrilinosWrappers::MPI::BlockVector &solution_delta )
    {
        timerconsole.enter_subsection("Assemble system");
        timerfile.enter_subsection("Assemble system");
        pcout     << " ASM_SYS " << std::flush;
        outfile   << " ASM_SYS " << std::flush;

        const TrilinosWrappers::MPI::BlockVector solution_total(get_total_solution(solution_delta));

        //Info given to FEValues and FEFaceValues constructors, to indicate which data will be needed at each element.
        const UpdateFlags uf_cell(update_values |
                                  update_gradients |
                                  update_JxW_values);
        const UpdateFlags uf_face(update_values |
                                  update_gradients |
                                  update_normal_vectors |
                                  update_quadrature_points |
                                  update_JxW_values );

        //Setup a copy of the data structures required for the process and pass them, along with the
        //memory addresses of the assembly functions to the WorkStream object for processing
        PerTaskData_ASM per_task_data(dofs_per_cell);
        ScratchData_ASM<ADNumberType> scratch_data(fe, qf_cell, uf_cell,
                                                   qf_face, uf_face,
                                                   solution_total);

        FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
        cell (IteratorFilters::LocallyOwnedCell(),
              dof_handler_ref.begin_active()),
        endc (IteratorFilters::LocallyOwnedCell(),
              dof_handler_ref.end());
        for (; cell != endc; ++cell)
          {
            Assert(cell->is_locally_owned(), ExcInternalError());
            Assert(cell->subdomain_id() == this_mpi_process, ExcInternalError());

            assemble_system_one_cell(cell, scratch_data, per_task_data);
            copy_local_to_global_system(per_task_data);
          }
        tangent_matrix.compress(VectorOperation::add);
        system_rhs.compress(VectorOperation::add);

        tangent_matrix_nb.compress(VectorOperation::add);
        system_rhs_nb.compress(VectorOperation::add);

        timerconsole.leave_subsection();
        timerfile.leave_subsection();
    }

    //Add the local elemental contribution to the global stiffness tensor
    // We do it twice, for the block and the non-block systems
    template <int dim>
    void Solid<dim>::copy_local_to_global_system (const PerTaskData_ASM &data)
    {
        constraints.distribute_local_to_global(data.cell_matrix,
            data.cell_rhs,
            data.local_dof_indices,
            tangent_matrix,
            system_rhs);

        constraints.distribute_local_to_global(data.cell_matrix,
            data.cell_rhs,
            data.local_dof_indices,
            tangent_matrix_nb,
            system_rhs_nb);
    }

    //Compute stiffness matrix and corresponding rhs for one element
    template <int dim>
    void Solid<dim>::assemble_system_one_cell
             (const typename DoFHandler<dim>::active_cell_iterator &cell,
              ScratchData_ASM<ADNumberType>                        &scratch,
              PerTaskData_ASM                                      &data) const
    {
        Assert(cell->is_locally_owned(), ExcInternalError());

        data.reset();
        scratch.reset();
        scratch.fe_values_ref.reinit(cell);
        cell->get_dof_indices(data.local_dof_indices);

        // Setup automatic differentiation
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            // Initialise the dofs for the cell using the current solution.
            scratch.local_dof_values[k] = scratch.solution_total[data.local_dof_indices[k]];
            // Mark this cell DoF as an independent variable
            scratch.local_dof_values[k].diff(k, dofs_per_cell);
          }

        // Update the quadrature point solution
        // Compute the values and gradients of the solution in terms of the AD variables
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                const unsigned int k_group = fe.system_to_base_index(k).first.first;
                if (k_group == u_block)
                  {
                    const Tensor<2, dim> Grad_Nx_u =
                           scratch.fe_values_ref[u_fe].gradient(k, q);
                    for (unsigned int dd = 0; dd < dim; ++dd)
                      {
                        for (unsigned int ee = 0; ee < dim; ++ee)
                          {
                            scratch.solution_grads_u_total[q][dd][ee]
                             += scratch.local_dof_values[k] * Grad_Nx_u[dd][ee];
                          }
                      }
                  }
                else if  (k_group == p_fluid_block)
                  {
                    const double Nx_p = scratch.fe_values_ref[p_fluid_fe].value(k, q);
                    const Tensor<1, dim> Grad_Nx_p =
                              scratch.fe_values_ref[p_fluid_fe].gradient(k, q);

                    scratch.solution_values_p_fluid_total[q]
                             += scratch.local_dof_values[k] * Nx_p;
                    for (unsigned int dd = 0; dd < dim; ++dd)
                      {
                        scratch.solution_grads_p_fluid_total[q][dd]
                            += scratch.local_dof_values[k] * Grad_Nx_p[dd];
                      }
                  }
                else
                  Assert(k_group <= p_fluid_block, ExcInternalError());
              }
          }

        //Set up pointer "lgph" to the PointHistory object of this element
        const std::vector<std::shared_ptr<const PointHistory<dim, ADNumberType> > >
            lqph = quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());


        //Precalculate the element shape function values and gradients
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            Tensor<2, dim, ADNumberType> F_AD = scratch.solution_grads_u_total[q_point];
            F_AD += Tensor<2, dim, double>(Physics::Elasticity::StandardTensors<dim>::I);
            Assert(determinant(F_AD) > 0, ExcMessage("Invalid deformation map"));
            const Tensor<2, dim, ADNumberType> F_inv_AD = invert(F_AD);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const unsigned int i_group = fe.system_to_base_index(i).first.first;

                if (i_group == u_block)
                  {
                    scratch.Nx[q_point][i] =
                        scratch.fe_values_ref[u_fe].value(i, q_point);
                    scratch.grad_Nx[q_point][i] =
                        scratch.fe_values_ref[u_fe].gradient(i, q_point)*F_inv_AD;
                    scratch.symm_grad_Nx[q_point][i] =
                        symmetrize(scratch.grad_Nx[q_point][i]);
                  }
                else if  (i_group == p_fluid_block)
                  {
                    scratch.Nx_p_fluid[q_point][i] =
                        scratch.fe_values_ref[p_fluid_fe].value(i, q_point);
                    scratch.grad_Nx_p_fluid[q_point][i] =
                        scratch.fe_values_ref[p_fluid_fe].gradient(i, q_point)*F_inv_AD;
                  }
                else
                  Assert(i_group <= p_fluid_block, ExcInternalError());
              }
          }

        //Assemble the stiffness matrix and rhs vector
        std::vector<ADNumberType> residual_ad (dofs_per_cell, ADNumberType(0.0));
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            Tensor<2, dim, ADNumberType> F_AD = scratch.solution_grads_u_total[q_point];
            F_AD += Tensor<2, dim,double>(Physics::Elasticity::StandardTensors<dim>::I);
            const ADNumberType det_F_AD = determinant(F_AD);

            Assert(det_F_AD > 0, ExcInternalError());
            const Tensor<2, dim, ADNumberType> F_inv_AD = invert(F_AD); //inverse of def. gradient tensor

            const ADNumberType p_fluid = scratch.solution_values_p_fluid_total[q_point];

            {
              PointHistory<dim, ADNumberType> *lqph_q_point_nc =
                 const_cast<PointHistory<dim, ADNumberType>*>(lqph[q_point].get());
              lqph_q_point_nc->update_internal_equilibrium(F_AD);
            }

            //Get some info from constitutive model of solid
            static const SymmetricTensor< 2, dim, double>
                I (Physics::Elasticity::StandardTensors<dim>::I);
            const SymmetricTensor<2, dim, ADNumberType>
                tau_E = lqph[q_point]->get_tau_E(F_AD);
            SymmetricTensor<2, dim, ADNumberType> tau_fluid_vol (I);
            tau_fluid_vol *= -1.0 * p_fluid * det_F_AD;

            //Get some info from constitutive model of fluid
            const ADNumberType det_F_aux =  lqph[q_point]->get_converged_det_F();
            const double det_F_converged = Tensor<0,dim,double>(det_F_aux); //Needs to be double, not AD number
            const Tensor<1, dim, ADNumberType> overall_body_force
                = lqph[q_point]->get_overall_body_force(F_AD, parameters);

            // Define some aliases to make the assembly process easier to follow
            const std::vector<Tensor<1,dim>> &Nu = scratch.Nx[q_point];
            const std::vector<SymmetricTensor<2, dim, ADNumberType>>
                &symm_grad_Nu = scratch.symm_grad_Nx[q_point];
            const std::vector<double> &Np = scratch.Nx_p_fluid[q_point];
            const std::vector<Tensor<1, dim, ADNumberType> > &grad_Np
                = scratch.grad_Nx_p_fluid[q_point];
            const Tensor<1, dim, ADNumberType> grad_p
                = scratch.solution_grads_p_fluid_total[q_point]*F_inv_AD;
            const double JxW = scratch.fe_values_ref.JxW(q_point);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const unsigned int i_group = fe.system_to_base_index(i).first.first;

                if (i_group == u_block)
                  {
                    residual_ad[i] += symm_grad_Nu[i] * ( tau_E + tau_fluid_vol ) * JxW;
                    residual_ad[i] -= Nu[i] * overall_body_force * JxW;
                  }
                else if (i_group == p_fluid_block)
                  {
                    const Tensor<1, dim, ADNumberType> seepage_vel_current
                        = lqph[q_point]->get_seepage_velocity_current(F_AD, grad_p);
                    residual_ad[i] += Np[i] * (det_F_AD - det_F_converged) * JxW;
                    residual_ad[i] -= time.get_delta_t() * grad_Np[i]
                                      * seepage_vel_current * JxW;
                  }
                else
                  Assert(i_group <= p_fluid_block, ExcInternalError());
              }
          }

          // Assemble the Neumann contribution (external force contribution).
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face) //Loop over faces in element
            {
              if (cell->face(face)->at_boundary() == true)
                {
                  scratch.fe_face_values_ref.reinit(cell, face);

                  for (unsigned int f_q_point = 0; f_q_point < n_q_points_f; ++f_q_point)
                    {
                      const Tensor<1, dim> &N
                          = scratch.fe_face_values_ref.normal_vector(f_q_point);
                      const Point<dim>     &pt
                          = scratch.fe_face_values_ref.quadrature_point(f_q_point);
                      const Tensor<1, dim> traction
                          = get_neumann_traction(cell->face(face)->boundary_id(), pt, N);
                      const double flow
                          = get_prescribed_fluid_flow(cell->face(face)->boundary_id(), pt);

                      if ( (traction.norm() < 1e-12) && (std::abs(flow) < 1e-12) ) continue;

                      const double JxW_f = scratch.fe_face_values_ref.JxW(f_q_point);

                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          const unsigned int i_group = fe.system_to_base_index(i).first.first;

                          if ((i_group == u_block) && (traction.norm() > 1e-12))
                          {
                              const unsigned int component_i
                                = fe.system_to_component_index(i).first;
                              const double Nu_f
                                = scratch.fe_face_values_ref.shape_value(i, f_q_point);
                              residual_ad[i] -= (Nu_f * traction[component_i]) * JxW_f;
                          }
                          if ((i_group == p_fluid_block) && (std::abs(flow) > 1e-12))
                          {
                              const double Nu_p
                                = scratch.fe_face_values_ref.shape_value(i, f_q_point);
                              residual_ad[i] -= (Nu_p * flow) * JxW_f;
                          }
                        }
                    }
                }
            }

        // Linearise the residual
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            const ADNumberType &R_i = residual_ad[i];

            data.cell_rhs(i) -= R_i.val();
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              data.cell_matrix(i,j) += R_i.fastAccessDx(j);
          }
    }

    //Store the converged values of the internal variables
    template <int dim>
    void Solid<dim>::update_end_timestep()
    {
          FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
          cell (IteratorFilters::LocallyOwnedCell(),
                dof_handler_ref.begin_active()),
          endc (IteratorFilters::LocallyOwnedCell(),
                dof_handler_ref.end());
          for (; cell!=endc; ++cell)
          {
            Assert(cell->is_locally_owned(), ExcInternalError());
            Assert(cell->subdomain_id() == this_mpi_process, ExcInternalError());

            const std::vector<std::shared_ptr<PointHistory<dim, ADNumberType> > >
                lqph = quadrature_point_history.get_data(cell);
            Assert(lqph.size() == n_q_points, ExcInternalError());
            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
              lqph[q_point]->update_end_timestep();
          }
    }


     //Solve the linearized equations
     template <int dim>
     void Solid<dim>::solve_linear_system( TrilinosWrappers::MPI::BlockVector &newton_update_OUT)
     {

           timerconsole.enter_subsection("Linear solver");
           timerfile.enter_subsection("Linear solver");
           pcout     << " SLV " << std::flush;
           outfile   << " SLV " << std::flush;

           TrilinosWrappers::MPI::Vector newton_update_nb;
           newton_update_nb.reinit(locally_owned_dofs, mpi_communicator);

           SolverControl solver_control (tangent_matrix_nb.m(),
                                         1.0e-6 * system_rhs_nb.l2_norm());
           TrilinosWrappers::SolverDirect solver (solver_control);
           solver.solve(tangent_matrix_nb, newton_update_nb, system_rhs_nb);

           // Copy the non-block solution back to block system
           for (unsigned int i=0; i<locally_owned_dofs.n_elements(); ++i)
             {
               const types::global_dof_index idx_i
                              = locally_owned_dofs.nth_index_in_set(i);
               newton_update_OUT(idx_i) = newton_update_nb(idx_i);
             }
           newton_update_OUT.compress(VectorOperation::insert);

           timerconsole.leave_subsection();
           timerfile.leave_subsection();
     }

    //Class to compute gradient of the pressure
    template <int dim>
    class GradientPostprocessor : public DataPostprocessorVector<dim>
    {
        public:
          GradientPostprocessor (const unsigned int p_fluid_component)
            :
            DataPostprocessorVector<dim> ("grad_p",
                                          update_gradients),
            p_fluid_component (p_fluid_component)
          {}

          virtual ~GradientPostprocessor(){}

          virtual void
          evaluate_vector_field
               (const DataPostprocessorInputs::Vector<dim> &input_data,
                std::vector<Vector<double> >               &computed_quantities) const override
          {
            AssertDimension (input_data.solution_gradients.size(),
                             computed_quantities.size());
            for (unsigned int p=0; p<input_data.solution_gradients.size(); ++p)
              {
                AssertDimension (computed_quantities[p].size(), dim);
                for (unsigned int d=0; d<dim; ++d)
                  computed_quantities[p][d]
                    = input_data.solution_gradients[p][p_fluid_component][d];
              }
          }

        private:
          const unsigned int  p_fluid_component;
    };


      //Print results to vtu file
      template <int dim> void Solid<dim>::output_results_to_vtu
                            (const unsigned int timestep,
                             const double current_time,
                             TrilinosWrappers::MPI::BlockVector solution_IN) const
      {
        TrilinosWrappers::MPI::BlockVector solution_total(locally_owned_partitioning,
                                                          locally_relevant_partitioning,
                                                          mpi_communicator,
                                                          false);
        solution_total = solution_IN;
        Vector<double> material_id;
        material_id.reinit(triangulation.n_active_cells());
        std::vector<types::subdomain_id> partition_int(triangulation.n_active_cells());
        GradientPostprocessor<dim> gradient_postprocessor(p_fluid_component);

         //Declare local variables with number of stress components
         //& assign value according to "dim" value
         unsigned int num_comp_symm_tensor = 6;

        //Declare local vectors to store values
        // OUTPUT AVERAGED ON ELEMENTS -------------------------------------------
        std::vector<Vector<double>>cauchy_stresses_total_elements
                             (num_comp_symm_tensor,
                              Vector<double> (triangulation.n_active_cells()));
        std::vector<Vector<double>>cauchy_stresses_E_elements
                             (num_comp_symm_tensor,
                              Vector<double> (triangulation.n_active_cells()));
        std::vector<Vector<double>>stretches_elements
                             (dim,
                              Vector<double> (triangulation.n_active_cells()));
        std::vector<Vector<double>>seepage_velocity_elements
                              (dim,
                               Vector<double> (triangulation.n_active_cells()));
        Vector<double> porous_dissipation_elements
                              (triangulation.n_active_cells());
        Vector<double> viscous_dissipation_elements
                              (triangulation.n_active_cells());
        Vector<double> solid_vol_fraction_elements
                              (triangulation.n_active_cells());

        // OUTPUT AVERAGED ON NODES ----------------------------------------------
        // We need to create a new FE space with a single dof per node to avoid
        // duplication of the output on nodes for our problem with dim+1 dofs.
        FE_Q<dim> fe_vertex(1);
        DoFHandler<dim> vertex_handler_ref(triangulation);
        vertex_handler_ref.distribute_dofs(fe_vertex);
        AssertThrow(vertex_handler_ref.n_dofs() == triangulation.n_vertices(),
          ExcDimensionMismatch(vertex_handler_ref.n_dofs(),
                               triangulation.n_vertices()));

        Vector<double> counter_on_vertices_mpi
                        (vertex_handler_ref.n_dofs());
        Vector<double> sum_counter_on_vertices
                        (vertex_handler_ref.n_dofs());

        std::vector<Vector<double>>cauchy_stresses_total_vertex_mpi
                                  (num_comp_symm_tensor,
                                   Vector<double>(vertex_handler_ref.n_dofs()));
        std::vector<Vector<double>>sum_cauchy_stresses_total_vertex
                                  (num_comp_symm_tensor,
                                   Vector<double>(vertex_handler_ref.n_dofs()));
        std::vector<Vector<double>>cauchy_stresses_E_vertex_mpi
                                  (num_comp_symm_tensor,
                                   Vector<double>(vertex_handler_ref.n_dofs()));
        std::vector<Vector<double>>sum_cauchy_stresses_E_vertex
                                  (num_comp_symm_tensor,
                                   Vector<double>(vertex_handler_ref.n_dofs()));
        std::vector<Vector<double>>stretches_vertex_mpi
                                  (dim,
                                   Vector<double>(vertex_handler_ref.n_dofs()));
        std::vector<Vector<double>>sum_stretches_vertex
                                  (dim,
                                   Vector<double>(vertex_handler_ref.n_dofs()));
        Vector<double> porous_dissipation_vertex_mpi(vertex_handler_ref.n_dofs());
        Vector<double> sum_porous_dissipation_vertex(vertex_handler_ref.n_dofs());
        Vector<double> viscous_dissipation_vertex_mpi(vertex_handler_ref.n_dofs());
        Vector<double> sum_viscous_dissipation_vertex(vertex_handler_ref.n_dofs());
        Vector<double> solid_vol_fraction_vertex_mpi(vertex_handler_ref.n_dofs());
        Vector<double> sum_solid_vol_fraction_vertex(vertex_handler_ref.n_dofs());

        // We need to create a new FE space with a dim dof per node to
        // be able to ouput data on nodes in vector form
        FESystem<dim> fe_vertex_vec(FE_Q<dim>(1),dim);
        DoFHandler<dim> vertex_vec_handler_ref(triangulation);
        vertex_vec_handler_ref.distribute_dofs(fe_vertex_vec);
        AssertThrow(vertex_vec_handler_ref.n_dofs() == (dim*triangulation.n_vertices()),
          ExcDimensionMismatch(vertex_vec_handler_ref.n_dofs(),
                               (dim*triangulation.n_vertices())));

        Vector<double> seepage_velocity_vertex_vec_mpi(vertex_vec_handler_ref.n_dofs());
        Vector<double> sum_seepage_velocity_vertex_vec(vertex_vec_handler_ref.n_dofs());
        Vector<double> counter_on_vertices_vec_mpi(vertex_vec_handler_ref.n_dofs());
        Vector<double> sum_counter_on_vertices_vec(vertex_vec_handler_ref.n_dofs());
        // -----------------------------------------------------------------------

        //Declare and initialize local unit vectors (to construct tensor basis)
        std::vector<Tensor<1,dim>> basis_vectors (dim, Tensor<1,dim>() );
        for (unsigned int i=0; i<dim; ++i)
            basis_vectors[i][i] = 1;

        //Declare an instance of the material class object
        if (parameters.mat_type == "Neo-Hooke")
            NeoHooke<dim,ADNumberType> material(parameters,time);
        else if (parameters.mat_type == "Ogden")
            Ogden<dim,ADNumberType> material(parameters,time);
        else if (parameters.mat_type == "visco-Ogden")
            visco_Ogden <dim,ADNumberType>material(parameters,time);
        else
            Assert (false, ExcMessage("Material type not implemented"));

        //Define a local instance of FEValues to compute updated values required
        //to calculate stresses
        const UpdateFlags uf_cell(update_values | update_gradients |
                                  update_JxW_values);
        FEValues<dim> fe_values_ref (fe, qf_cell, uf_cell);

        //Iterate through elements (cells) and Gauss Points
        FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
          cell(IteratorFilters::LocallyOwnedCell(),
               dof_handler_ref.begin_active()),
          endc(IteratorFilters::LocallyOwnedCell(),
               dof_handler_ref.end()),
          cell_v(IteratorFilters::LocallyOwnedCell(),
                 vertex_handler_ref.begin_active()),
          cell_v_vec(IteratorFilters::LocallyOwnedCell(),
                     vertex_vec_handler_ref.begin_active());
        //start cell loop
        for (; cell!=endc; ++cell, ++cell_v, ++cell_v_vec)
        {
            Assert(cell->is_locally_owned(), ExcInternalError());
            Assert(cell->subdomain_id() == this_mpi_process, ExcInternalError());

            material_id(cell->active_cell_index())=
              static_cast<int>(cell->material_id());

            fe_values_ref.reinit(cell);

            std::vector<Tensor<2,dim>> solution_grads_u(n_q_points);
            fe_values_ref[u_fe].get_function_gradients(solution_total,
                                                       solution_grads_u);

            std::vector<double> solution_values_p_fluid_total(n_q_points);
            fe_values_ref[p_fluid_fe].get_function_values(solution_total,
                                                          solution_values_p_fluid_total);

            std::vector<Tensor<1,dim>> solution_grads_p_fluid_AD (n_q_points);
            fe_values_ref[p_fluid_fe].get_function_gradients(solution_total,
                                                             solution_grads_p_fluid_AD);

            //start gauss point loop
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            {
                const Tensor<2,dim,ADNumberType>
                  F_AD = Physics::Elasticity::Kinematics::F(solution_grads_u[q_point]);
                ADNumberType det_F_AD = determinant(F_AD);
                const double det_F = Tensor<0,dim,double>(det_F_AD);

                const std::vector<std::shared_ptr<const PointHistory<dim,ADNumberType>>>
                    lqph = quadrature_point_history.get_data(cell);
                Assert(lqph.size() == n_q_points, ExcInternalError());

                const double p_fluid = solution_values_p_fluid_total[q_point];

                //Cauchy stress
                static const SymmetricTensor<2,dim,double>
                  I (Physics::Elasticity::StandardTensors<dim>::I);
                SymmetricTensor<2,dim> sigma_E;
                const SymmetricTensor<2,dim,ADNumberType> sigma_E_AD =
                  lqph[q_point]->get_Cauchy_E(F_AD);

                for (unsigned int i=0; i<dim; ++i)
                    for (unsigned int j=0; j<dim; ++j)
                       sigma_E[i][j] = Tensor<0,dim,double>(sigma_E_AD[i][j]);

                SymmetricTensor<2,dim> sigma_fluid_vol (I);
                sigma_fluid_vol *= -p_fluid;
                const SymmetricTensor<2,dim> sigma = sigma_E + sigma_fluid_vol;

                //Volumes
                const double solid_vol_fraction = (parameters.solid_vol_frac)/det_F;

                //Green-Lagrange strain
                const Tensor<2,dim> E_strain = 0.5*(transpose(F_AD)*F_AD - I);

                //Seepage velocity
                const Tensor<2,dim,ADNumberType> F_inv = invert(F_AD);
                const Tensor<1,dim,ADNumberType> grad_p_fluid_AD =
                                          solution_grads_p_fluid_AD[q_point]*F_inv;
                const Tensor<1,dim,ADNumberType> seepage_vel_AD =
                 lqph[q_point]->get_seepage_velocity_current(F_AD, grad_p_fluid_AD);

                //Dissipations
                const double porous_dissipation =
                  lqph[q_point]->get_porous_dissipation(F_AD, grad_p_fluid_AD);
                const double viscous_dissipation =
                  lqph[q_point]->get_viscous_dissipation();

                // OUTPUT AVERAGED ON ELEMENTS -------------------------------------------
                // Both average on elements and on nodes is NOT weighted with the
                // integration point volume, i.e., we assume equal contribution of each
                // integration point to the average. Ideally, it should be weighted,
                // but I haven't invested time in getting it to work properly.
                if (parameters.outtype == "elements")
                {
                    for (unsigned int j=0; j<dim; ++j)
                    {
                        cauchy_stresses_total_elements[j](cell->active_cell_index())
                          += ((sigma*basis_vectors[j])*basis_vectors[j])/n_q_points;
                        cauchy_stresses_E_elements[j](cell->active_cell_index())
                          += ((sigma_E*basis_vectors[j])*basis_vectors[j])/n_q_points;
                        stretches_elements[j](cell->active_cell_index())
                          += std::sqrt(1.0+2.0*Tensor<0,dim,double>(E_strain[j][j]))
                             /n_q_points;
                        seepage_velocity_elements[j](cell->active_cell_index())
                          +=  Tensor<0,dim,double>(seepage_vel_AD[j])/n_q_points;
                    }

                    porous_dissipation_elements(cell->active_cell_index())
                      +=  porous_dissipation/n_q_points;
                    viscous_dissipation_elements(cell->active_cell_index())
                      +=  viscous_dissipation/n_q_points;
                    solid_vol_fraction_elements(cell->active_cell_index())
                      +=  solid_vol_fraction/n_q_points;

                    cauchy_stresses_total_elements[3](cell->active_cell_index())
                      += ((sigma*basis_vectors[0])*basis_vectors[1])/n_q_points; //sig_xy
                    cauchy_stresses_total_elements[4](cell->active_cell_index())
                      += ((sigma*basis_vectors[0])*basis_vectors[2])/n_q_points;//sig_xz
                    cauchy_stresses_total_elements[5](cell->active_cell_index())
                      += ((sigma*basis_vectors[1])*basis_vectors[2])/n_q_points;//sig_yz

                    cauchy_stresses_E_elements[3](cell->active_cell_index())
                      += ((sigma_E*basis_vectors[0])* basis_vectors[1])/n_q_points; //sig_xy
                    cauchy_stresses_E_elements[4](cell->active_cell_index())
                      += ((sigma_E*basis_vectors[0])* basis_vectors[2])/n_q_points;//sig_xz
                    cauchy_stresses_E_elements[5](cell->active_cell_index())
                      += ((sigma_E*basis_vectors[1])* basis_vectors[2])/n_q_points;//sig_yz

                }
                // OUTPUT AVERAGED ON NODES -------------------------------------------
                else if (parameters.outtype == "nodes")
                {
                  for (unsigned int v=0; v<(GeometryInfo<dim>::vertices_per_cell); ++v)
                  {
                      types::global_dof_index local_vertex_indices =
                                                    cell_v->vertex_dof_index(v, 0);
                      counter_on_vertices_mpi(local_vertex_indices) += 1;
                      for (unsigned int k=0; k<dim; ++k)
                      {
                          cauchy_stresses_total_vertex_mpi[k](local_vertex_indices)
                            += (sigma*basis_vectors[k])*basis_vectors[k];
                          cauchy_stresses_E_vertex_mpi[k](local_vertex_indices)
                            += (sigma_E*basis_vectors[k])*basis_vectors[k];
                          stretches_vertex_mpi[k](local_vertex_indices)
                            += std::sqrt(1.0+2.0*Tensor<0,dim,double>(E_strain[k][k]));

                          types::global_dof_index local_vertex_vec_indices =
                                                cell_v_vec->vertex_dof_index(v, k);
                          counter_on_vertices_vec_mpi(local_vertex_vec_indices) += 1;
                          seepage_velocity_vertex_vec_mpi(local_vertex_vec_indices)
                            += Tensor<0,dim,double>(seepage_vel_AD[k]);
                      }

                      porous_dissipation_vertex_mpi(local_vertex_indices)
                        += porous_dissipation;
                      viscous_dissipation_vertex_mpi(local_vertex_indices)
                        += viscous_dissipation;
                      solid_vol_fraction_vertex_mpi(local_vertex_indices)
                        += solid_vol_fraction;

                      cauchy_stresses_total_vertex_mpi[3](local_vertex_indices)
                        += (sigma*basis_vectors[0])*basis_vectors[1]; //sig_xy
                      cauchy_stresses_total_vertex_mpi[4](local_vertex_indices)
                        += (sigma*basis_vectors[0])*basis_vectors[2];//sig_xz
                      cauchy_stresses_total_vertex_mpi[5](local_vertex_indices)
                        += (sigma*basis_vectors[1])*basis_vectors[2]; //sig_yz

                      cauchy_stresses_E_vertex_mpi[3](local_vertex_indices)
                        += (sigma_E*basis_vectors[0])*basis_vectors[1]; //sig_xy
                      cauchy_stresses_E_vertex_mpi[4](local_vertex_indices)
                        += (sigma_E*basis_vectors[0])*basis_vectors[2];//sig_xz
                      cauchy_stresses_E_vertex_mpi[5](local_vertex_indices)
                        += (sigma_E*basis_vectors[1])*basis_vectors[2]; //sig_yz
                    }
              }
              //---------------------------------------------------------------
            } //end gauss point loop
        }//end cell loop

        // Different nodes might have different amount of contributions, e.g.,
        // corner nodes have less integration points contributing to the averaged.
        // This is why we need a counter and divide at the end, outside the cell loop.
        if (parameters.outtype == "nodes")
        {
          for (unsigned int d=0; d<(vertex_handler_ref.n_dofs()); ++d)
          {
            sum_counter_on_vertices[d] =
              Utilities::MPI::sum(counter_on_vertices_mpi[d],
                                  mpi_communicator);
            sum_porous_dissipation_vertex[d] =
              Utilities::MPI::sum(porous_dissipation_vertex_mpi[d],
                                  mpi_communicator);
            sum_viscous_dissipation_vertex[d] =
              Utilities::MPI::sum(viscous_dissipation_vertex_mpi[d],
                                  mpi_communicator);
            sum_solid_vol_fraction_vertex[d] =
              Utilities::MPI::sum(solid_vol_fraction_vertex_mpi[d],
                                  mpi_communicator);

            for (unsigned int k=0; k<num_comp_symm_tensor; ++k)
            {
              sum_cauchy_stresses_total_vertex[k][d] =
                  Utilities::MPI::sum(cauchy_stresses_total_vertex_mpi[k][d],
                                      mpi_communicator);
              sum_cauchy_stresses_E_vertex[k][d] =
                  Utilities::MPI::sum(cauchy_stresses_E_vertex_mpi[k][d],
                                      mpi_communicator);
            }
            for (unsigned int k=0; k<dim; ++k)
            {
              sum_stretches_vertex[k][d] =
                  Utilities::MPI::sum(stretches_vertex_mpi[k][d],
                                      mpi_communicator);
            }
          }

          for (unsigned int d=0; d<(vertex_vec_handler_ref.n_dofs()); ++d)
          {
              sum_counter_on_vertices_vec[d] =
                  Utilities::MPI::sum(counter_on_vertices_vec_mpi[d],
                                      mpi_communicator);
              sum_seepage_velocity_vertex_vec[d] =
                  Utilities::MPI::sum(seepage_velocity_vertex_vec_mpi[d],
                                      mpi_communicator);
          }

          for (unsigned int d=0; d<(vertex_handler_ref.n_dofs()); ++d)
          {
            if (sum_counter_on_vertices[d]>0)
            {
              for (unsigned int i=0; i<num_comp_symm_tensor; ++i)
              {
                  sum_cauchy_stresses_total_vertex[i][d] /= sum_counter_on_vertices[d];
                  sum_cauchy_stresses_E_vertex[i][d] /= sum_counter_on_vertices[d];
              }
              for (unsigned int i=0; i<dim; ++i)
              {
                  sum_stretches_vertex[i][d] /= sum_counter_on_vertices[d];
              }
              sum_porous_dissipation_vertex[d] /= sum_counter_on_vertices[d];
              sum_viscous_dissipation_vertex[d] /= sum_counter_on_vertices[d];
              sum_solid_vol_fraction_vertex[d] /= sum_counter_on_vertices[d];
            }
          }

          for (unsigned int d=0; d<(vertex_vec_handler_ref.n_dofs()); ++d)
          {
            if (sum_counter_on_vertices_vec[d]>0)
            {
              sum_seepage_velocity_vertex_vec[d] /= sum_counter_on_vertices_vec[d];
            }
          }

        }

        // Add the results to the solution to create the output file for Paraview
        DataOut<dim> data_out;
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
          comp_type(dim,
                    DataComponentInterpretation::component_is_part_of_vector);
        comp_type.push_back(DataComponentInterpretation::component_is_scalar);

        GridTools::get_subdomain_association(triangulation, partition_int);

        std::vector<std::string> solution_name(dim, "displacement");
        solution_name.push_back("pore_pressure");

        data_out.attach_dof_handler(dof_handler_ref);
        data_out.add_data_vector(solution_total,
                                 solution_name,
                                 DataOut<dim>::type_dof_data,
                                 comp_type);

        data_out.add_data_vector(solution_total,
                                 gradient_postprocessor);

        const Vector<double> partitioning(partition_int.begin(),
                                          partition_int.end());

        data_out.add_data_vector(partitioning, "partitioning");
        data_out.add_data_vector(material_id, "material_id");

        // Integration point results -----------------------------------------------------------
        if (parameters.outtype == "elements")
        {
          data_out.add_data_vector(cauchy_stresses_total_elements[0], "cauchy_xx");
          data_out.add_data_vector(cauchy_stresses_total_elements[1], "cauchy_yy");
          data_out.add_data_vector(cauchy_stresses_total_elements[2], "cauchy_zz");
          data_out.add_data_vector(cauchy_stresses_total_elements[3], "cauchy_xy");
          data_out.add_data_vector(cauchy_stresses_total_elements[4], "cauchy_xz");
          data_out.add_data_vector(cauchy_stresses_total_elements[5], "cauchy_yz");

          data_out.add_data_vector(cauchy_stresses_E_elements[0], "cauchy_E_xx");
          data_out.add_data_vector(cauchy_stresses_E_elements[1], "cauchy_E_yy");
          data_out.add_data_vector(cauchy_stresses_E_elements[2], "cauchy_E_zz");
          data_out.add_data_vector(cauchy_stresses_E_elements[3], "cauchy_E_xy");
          data_out.add_data_vector(cauchy_stresses_E_elements[4], "cauchy_E_xz");
          data_out.add_data_vector(cauchy_stresses_E_elements[5], "cauchy_E_yz");

          data_out.add_data_vector(stretches_elements[0], "stretch_xx");
          data_out.add_data_vector(stretches_elements[1], "stretch_yy");
          data_out.add_data_vector(stretches_elements[2], "stretch_zz");

          data_out.add_data_vector(seepage_velocity_elements[0], "seepage_vel_x");
          data_out.add_data_vector(seepage_velocity_elements[1], "seepage_vel_y");
          data_out.add_data_vector(seepage_velocity_elements[2], "seepage_vel_z");

          data_out.add_data_vector(porous_dissipation_elements, "dissipation_porous");
          data_out.add_data_vector(viscous_dissipation_elements, "dissipation_viscous");
          data_out.add_data_vector(solid_vol_fraction_elements, "solid_vol_fraction");
        }
        else if  (parameters.outtype == "nodes")
        {
          data_out.add_data_vector(vertex_handler_ref,
                                   sum_cauchy_stresses_total_vertex[0],
                                   "cauchy_xx");
          data_out.add_data_vector(vertex_handler_ref,
                                   sum_cauchy_stresses_total_vertex[1],
                                   "cauchy_yy");
          data_out.add_data_vector(vertex_handler_ref,
                                   sum_cauchy_stresses_total_vertex[2],
                                   "cauchy_zz");
          data_out.add_data_vector(vertex_handler_ref,
                                   sum_cauchy_stresses_total_vertex[3],
                                   "cauchy_xy");
          data_out.add_data_vector(vertex_handler_ref,
                                   sum_cauchy_stresses_total_vertex[4],
                                   "cauchy_xz");
          data_out.add_data_vector(vertex_handler_ref,
                                   sum_cauchy_stresses_total_vertex[5],
                                   "cauchy_yz");

          data_out.add_data_vector(vertex_handler_ref,
                                   sum_cauchy_stresses_E_vertex[0],
                                   "cauchy_E_xx");
          data_out.add_data_vector(vertex_handler_ref,
                                   sum_cauchy_stresses_E_vertex[1],
                                   "cauchy_E_yy");
          data_out.add_data_vector(vertex_handler_ref,
                                   sum_cauchy_stresses_E_vertex[2],
                                   "cauchy_E_zz");
          data_out.add_data_vector(vertex_handler_ref,
                                   sum_cauchy_stresses_E_vertex[3],
                                   "cauchy_E_xy");
          data_out.add_data_vector(vertex_handler_ref,
                                   sum_cauchy_stresses_E_vertex[4],
                                   "cauchy_E_xz");
          data_out.add_data_vector(vertex_handler_ref,
                                   sum_cauchy_stresses_E_vertex[5],
                                   "cauchy_E_yz");

          data_out.add_data_vector(vertex_handler_ref,
                                   sum_stretches_vertex[0],
                                   "stretch_xx");
          data_out.add_data_vector(vertex_handler_ref,
                                   sum_stretches_vertex[1],
                                   "stretch_yy");
          data_out.add_data_vector(vertex_handler_ref,
                                   sum_stretches_vertex[2],
                                   "stretch_zz");

          std::vector<DataComponentInterpretation::DataComponentInterpretation>
           comp_type_vec(dim,
                         DataComponentInterpretation::component_is_part_of_vector);
          std::vector<std::string> solution_name_vec(dim,"seepage_velocity");

          data_out.add_data_vector(vertex_vec_handler_ref,
                                   sum_seepage_velocity_vertex_vec,
                                   solution_name_vec,
                                   comp_type_vec);

          data_out.add_data_vector(vertex_handler_ref,
                                   sum_porous_dissipation_vertex,
                                   "dissipation_porous");
          data_out.add_data_vector(vertex_handler_ref,
                                   sum_viscous_dissipation_vertex,
                                   "dissipation_viscous");
          data_out.add_data_vector(vertex_handler_ref,
                                   sum_solid_vol_fraction_vertex,
                                   "solid_vol_fraction");
        }
      //---------------------------------------------------------------------

        data_out.build_patches(degree_displ);

        struct Filename
        {
          static std::string get_filename_vtu(unsigned int process,
                                              unsigned int timestep,
                                              const unsigned int n_digits = 5)
          {
            std::ostringstream filename_vtu;
            filename_vtu
            << "solution."
            << Utilities::int_to_string(process, n_digits)
            << "."
            << Utilities::int_to_string(timestep, n_digits)
            << ".vtu";
            return filename_vtu.str();
          }

          static std::string get_filename_pvtu(unsigned int timestep,
                                               const unsigned int n_digits = 5)
          {
            std::ostringstream filename_vtu;
            filename_vtu
            << "solution."
            << Utilities::int_to_string(timestep, n_digits)
            << ".pvtu";
            return filename_vtu.str();
          }

          static std::string get_filename_pvd (void)
          {
            std::ostringstream filename_vtu;
            filename_vtu
            << "solution.pvd";
            return filename_vtu.str();
          }
        };

        const std::string filename_vtu = Filename::get_filename_vtu(this_mpi_process,
                                                                    timestep);
        std::ofstream output(filename_vtu.c_str());
        data_out.write_vtu(output);

        // We have a collection of files written in parallel
        // This next set of steps should only be performed by master process
        if (this_mpi_process == 0)
        {
          // List of all files written out at this timestep by all processors
          std::vector<std::string> parallel_filenames_vtu;
          for (unsigned int p=0; p<n_mpi_processes; ++p)
          {
            parallel_filenames_vtu.push_back(Filename::get_filename_vtu(p, timestep));
          }

          const std::string filename_pvtu(Filename::get_filename_pvtu(timestep));
          std::ofstream pvtu_master(filename_pvtu.c_str());
          data_out.write_pvtu_record(pvtu_master,
                                     parallel_filenames_vtu);

          // Time dependent data master file
          static std::vector<std::pair<double,std::string>> time_and_name_history;
          time_and_name_history.push_back(std::make_pair(current_time,
                                                          filename_pvtu));
          const std::string filename_pvd(Filename::get_filename_pvd());
          std::ofstream pvd_output(filename_pvd.c_str());
          DataOutBase::write_pvd_record(pvd_output, time_and_name_history);
        }
      }


      //Print results to plotting file
      template <int dim>
      void Solid<dim>::output_results_to_plot(
                                const unsigned int timestep,
                                const double current_time,
                                TrilinosWrappers::MPI::BlockVector solution_IN,
                                std::vector<Point<dim> > &tracked_vertices_IN,
                                std::ofstream &plotpointfile) const
      {
        TrilinosWrappers::MPI::BlockVector solution_total(locally_owned_partitioning,
                                                          locally_relevant_partitioning,
                                                          mpi_communicator,
                                                          false);

        (void) timestep;
        solution_total = solution_IN;

        //Variables needed to print the solution file for plotting
        Point<dim> reaction_force;
        Point<dim> reaction_force_pressure;
        Point<dim> reaction_force_extra;
        double total_fluid_flow = 0.0;
        double total_porous_dissipation = 0.0;
        double total_viscous_dissipation = 0.0;
        double total_solid_vol = 0.0;
        double total_vol_current = 0.0;
        double total_vol_reference = 0.0;
        std::vector<Point<dim+1>> solution_vertices(tracked_vertices_IN.size());

        //Auxiliar variables needed for mpi processing
        Tensor<1,dim> sum_reaction_mpi;
        Tensor<1,dim> sum_reaction_pressure_mpi;
        Tensor<1,dim> sum_reaction_extra_mpi;
        sum_reaction_mpi = 0.0;
        sum_reaction_pressure_mpi = 0.0;
        sum_reaction_extra_mpi = 0.0;
        double sum_total_flow_mpi = 0.0;
        double sum_porous_dissipation_mpi = 0.0;
        double sum_viscous_dissipation_mpi = 0.0;
        double sum_solid_vol_mpi = 0.0;
        double sum_vol_current_mpi = 0.0;
        double sum_vol_reference_mpi = 0.0;

        //Declare an instance of the material class object
        if (parameters.mat_type == "Neo-Hooke")
            NeoHooke<dim,ADNumberType> material(parameters,time);
        else if (parameters.mat_type == "Ogden")
            Ogden<dim,ADNumberType> material(parameters, time);
        else if (parameters.mat_type == "visco-Ogden")
            visco_Ogden <dim,ADNumberType>material(parameters,time);
        else
        Assert (false, ExcMessage("Material type not implemented"));

        //Define a local instance of FEValues to compute updated values required
        //to calculate stresses
        const UpdateFlags uf_cell(update_values | update_gradients |
                                  update_JxW_values);
        FEValues<dim> fe_values_ref (fe, qf_cell, uf_cell);

        //Iterate through elements (cells) and Gauss Points
        FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
          cell(IteratorFilters::LocallyOwnedCell(),
               dof_handler_ref.begin_active()),
          endc(IteratorFilters::LocallyOwnedCell(),
               dof_handler_ref.end());
        //start cell loop
        for (; cell!=endc; ++cell)
        {
            Assert(cell->is_locally_owned(), ExcInternalError());
            Assert(cell->subdomain_id() == this_mpi_process, ExcInternalError());

            fe_values_ref.reinit(cell);

            std::vector<Tensor<2,dim>> solution_grads_u(n_q_points);
            fe_values_ref[u_fe].get_function_gradients(solution_total,
                                                       solution_grads_u);

            std::vector<double> solution_values_p_fluid_total(n_q_points);
            fe_values_ref[p_fluid_fe].get_function_values(solution_total,
                                                          solution_values_p_fluid_total);

            std::vector<Tensor<1,dim >> solution_grads_p_fluid_AD(n_q_points);
            fe_values_ref[p_fluid_fe].get_function_gradients(solution_total,
                                                             solution_grads_p_fluid_AD);

            //start gauss point loop
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            {
                const Tensor<2,dim,ADNumberType>
                  F_AD = Physics::Elasticity::Kinematics::F(solution_grads_u[q_point]);
                ADNumberType det_F_AD = determinant(F_AD);
                const double det_F = Tensor<0,dim,double>(det_F_AD);

                const std::vector<std::shared_ptr<const PointHistory<dim,ADNumberType>>>
                    lqph = quadrature_point_history.get_data(cell);
                Assert(lqph.size() == n_q_points, ExcInternalError());

                double JxW = fe_values_ref.JxW(q_point);

                //Volumes
                sum_vol_current_mpi  += det_F * JxW;
                sum_vol_reference_mpi += JxW;
                sum_solid_vol_mpi += parameters.solid_vol_frac * JxW * det_F;

                //Seepage velocity
                const Tensor<2,dim,ADNumberType> F_inv = invert(F_AD);
                const Tensor<1,dim,ADNumberType>
                  grad_p_fluid_AD =  solution_grads_p_fluid_AD[q_point]*F_inv;
                const Tensor<1,dim,ADNumberType> seepage_vel_AD
                = lqph[q_point]->get_seepage_velocity_current(F_AD, grad_p_fluid_AD);

                //Dissipations
                const double porous_dissipation =
                  lqph[q_point]->get_porous_dissipation(F_AD, grad_p_fluid_AD);
                sum_porous_dissipation_mpi += porous_dissipation * det_F * JxW;

                const double viscous_dissipation = lqph[q_point]->get_viscous_dissipation();
                sum_viscous_dissipation_mpi += viscous_dissipation * det_F * JxW;

              //---------------------------------------------------------------
            } //end gauss point loop

            // Compute reaction force on load boundary & total fluid flow across
            // drained boundary.
            // Define a local instance of FEFaceValues to compute values required
            // to calculate reaction force
            const UpdateFlags uf_face( update_values | update_gradients |
                                       update_normal_vectors | update_JxW_values );
            FEFaceValues<dim> fe_face_values_ref(fe, qf_face, uf_face);

            //start face loop
            for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
            {
                //Reaction force
                if (cell->face(face)->at_boundary() == true &&
                    cell->face(face)->boundary_id() == get_reaction_boundary_id_for_output() )
                {
                    fe_face_values_ref.reinit(cell, face);

                    //Get displacement gradients for current face
                    std::vector<Tensor<2,dim> > solution_grads_u_f(n_q_points_f);
                    fe_face_values_ref[u_fe].get_function_gradients
                                                         (solution_total,
                                                          solution_grads_u_f);

                    //Get pressure for current element
                    std::vector< double > solution_values_p_fluid_total_f(n_q_points_f);
                    fe_face_values_ref[p_fluid_fe].get_function_values
                                               (solution_total,
                                                solution_values_p_fluid_total_f);

                    //start gauss points on faces loop
                    for (unsigned int f_q_point=0; f_q_point<n_q_points_f; ++f_q_point)
                    {
                        const Tensor<1,dim> &N = fe_face_values_ref.normal_vector(f_q_point);
                        const double JxW_f = fe_face_values_ref.JxW(f_q_point);

                        //Compute deformation gradient from displacements gradient
                        //(present configuration)
                        const Tensor<2,dim,ADNumberType> F_AD =
                          Physics::Elasticity::Kinematics::F(solution_grads_u_f[f_q_point]);

                        const std::vector<std::shared_ptr<const PointHistory<dim,ADNumberType>>>
                            lqph = quadrature_point_history.get_data(cell);
                        Assert(lqph.size() == n_q_points, ExcInternalError());

                        const double p_fluid = solution_values_p_fluid_total[f_q_point];

                        //Cauchy stress
                        static const SymmetricTensor<2,dim,double>
                          I (Physics::Elasticity::StandardTensors<dim>::I);
                        SymmetricTensor<2,dim> sigma_E;
                        const SymmetricTensor<2,dim,ADNumberType> sigma_E_AD =
                          lqph[f_q_point]->get_Cauchy_E(F_AD);

                        for (unsigned int i=0; i<dim; ++i)
                            for (unsigned int j=0; j<dim; ++j)
                               sigma_E[i][j] = Tensor<0,dim,double>(sigma_E_AD[i][j]);

                        SymmetricTensor<2,dim> sigma_fluid_vol(I);
                        sigma_fluid_vol *= -1.0*p_fluid;
                        const SymmetricTensor<2,dim> sigma = sigma_E+sigma_fluid_vol;
                        sum_reaction_mpi += sigma * N * JxW_f;
                        sum_reaction_pressure_mpi += sigma_fluid_vol * N * JxW_f;
                        sum_reaction_extra_mpi += sigma_E * N * JxW_f;
                    }//end gauss points on faces loop
                }

                //Fluid flow
                if (cell->face(face)->at_boundary() == true &&
                   (cell->face(face)->boundary_id() ==
                      get_drained_boundary_id_for_output().first ||
                    cell->face(face)->boundary_id() ==
                      get_drained_boundary_id_for_output().second ) )
                {
                    fe_face_values_ref.reinit(cell, face);

                    //Get displacement gradients for current face
                    std::vector<Tensor<2,dim>> solution_grads_u_f(n_q_points_f);
                    fe_face_values_ref[u_fe].get_function_gradients
                                                            (solution_total,
                                                             solution_grads_u_f);

                    //Get pressure gradients for current face
                    std::vector<Tensor<1,dim>> solution_grads_p_f(n_q_points_f);
                    fe_face_values_ref[p_fluid_fe].get_function_gradients
                                                             (solution_total,
                                                              solution_grads_p_f);

                    //start gauss points on faces loop
                    for (unsigned int f_q_point=0; f_q_point<n_q_points_f; ++f_q_point)
                    {
                        const Tensor<1,dim> &N =
                                  fe_face_values_ref.normal_vector(f_q_point);
                        const double JxW_f = fe_face_values_ref.JxW(f_q_point);

                        //Deformation gradient and inverse from displacements gradient
                        //(present configuration)
                        const Tensor<2,dim,ADNumberType> F_AD
                            = Physics::Elasticity::Kinematics::F(solution_grads_u_f[f_q_point]);

                        const Tensor<2,dim,ADNumberType> F_inv_AD = invert(F_AD);
                        ADNumberType det_F_AD = determinant(F_AD);

                        const std::vector<std::shared_ptr<const PointHistory<dim,ADNumberType>>>
                            lqph = quadrature_point_history.get_data(cell);
                        Assert(lqph.size() == n_q_points, ExcInternalError());

                        //Seepage velocity
                        Tensor<1,dim> seepage;
                        double det_F = Tensor<0,dim,double>(det_F_AD);
                        const Tensor<1,dim,ADNumberType> grad_p
                                          = solution_grads_p_f[f_q_point]*F_inv_AD;
                        const Tensor<1,dim,ADNumberType> seepage_AD
                          = lqph[f_q_point]->get_seepage_velocity_current(F_AD, grad_p);

                        for (unsigned int i=0; i<dim; ++i)
                            seepage[i] = Tensor<0,dim,double>(seepage_AD[i]);

                        sum_total_flow_mpi += (seepage/det_F) * N * JxW_f;
                    }//end gauss points on faces loop
                }
            }//end face loop
        }//end cell loop

        //Sum the results from different MPI process and then add to the reaction_force vector
        //In theory, the solution on each surface (each cell) only exists in one MPI process
        //so, we add all MPI process, one will have the solution and the others will be zero
        for (unsigned int d=0; d<dim; ++d)
        {
            reaction_force[d] = Utilities::MPI::sum(sum_reaction_mpi[d],
                                                    mpi_communicator);
            reaction_force_pressure[d] = Utilities::MPI::sum(sum_reaction_pressure_mpi[d],
                                                             mpi_communicator);
            reaction_force_extra[d] = Utilities::MPI::sum(sum_reaction_extra_mpi[d],
                                                          mpi_communicator);
        }

        //Same for total fluid flow, and for porous and viscous dissipations
        total_fluid_flow = Utilities::MPI::sum(sum_total_flow_mpi,
                                               mpi_communicator);
        total_porous_dissipation = Utilities::MPI::sum(sum_porous_dissipation_mpi,
                                                       mpi_communicator);
        total_viscous_dissipation = Utilities::MPI::sum(sum_viscous_dissipation_mpi,
                                                        mpi_communicator);
        total_solid_vol = Utilities::MPI::sum(sum_solid_vol_mpi,
                                              mpi_communicator);
        total_vol_current = Utilities::MPI::sum(sum_vol_current_mpi,
                                                mpi_communicator);
        total_vol_reference = Utilities::MPI::sum(sum_vol_reference_mpi,
                                                  mpi_communicator);

      //  Extract solution for tracked vectors
      // Copying an MPI::BlockVector into MPI::Vector is not possible,
      // so we copy each block of MPI::BlockVector into an MPI::Vector
      // And then we copy the MPI::Vector into "normal" Vectors
        TrilinosWrappers::MPI::Vector solution_vector_u_MPI(solution_total.block(u_block));
        TrilinosWrappers::MPI::Vector solution_vector_p_MPI(solution_total.block(p_fluid_block));
        Vector<double> solution_u_vector(solution_vector_u_MPI);
        Vector<double> solution_p_vector(solution_vector_p_MPI);

        if (this_mpi_process == 0)
        {
            //Append the pressure solution vector to the displacement solution vector,
            //creating a single solution vector equivalent to the original BlockVector
            //so FEFieldFunction will work with the dof_handler_ref.
            Vector<double> solution_vector(solution_p_vector.size()
                                           +solution_u_vector.size());

            for (unsigned int d=0; d<(solution_u_vector.size()); ++d)
                solution_vector[d] = solution_u_vector[d];

            for (unsigned int d=0; d<(solution_p_vector.size()); ++d)
                solution_vector[solution_u_vector.size()+d] = solution_p_vector[d];

            Functions::FEFieldFunction<dim,Vector<double>>
            find_solution(dof_handler_ref, solution_vector);

            for (unsigned int p=0; p<tracked_vertices_IN.size(); ++p)
            {
                Vector<double> update(dim+1);
                Point<dim> pt_ref;

                pt_ref[0]= tracked_vertices_IN[p][0];
                pt_ref[1]= tracked_vertices_IN[p][1];
                pt_ref[2]= tracked_vertices_IN[p][2];

               find_solution.vector_value(pt_ref, update);

               for (unsigned int d=0; d<(dim+1); ++d)
               {
                   //For values close to zero, set to 0.0
                   if (abs(update[d])<1.5*parameters.tol_u)
                       update[d] = 0.0;
                   solution_vertices[p][d] = update[d];
               }
            }
      // Write the results to the plotting file.
      // Add two blank lines between cycles in the cyclic loading examples so GNUPLOT can detect each cycle as a different block
            if (( (parameters.geom_type == "Budday_cube_tension_compression_fully_fixed")||
                  (parameters.geom_type == "Budday_cube_tension_compression")||
                  (parameters.geom_type == "Budday_cube_shear_fully_fixed")                ) &&
                ( (abs(current_time - parameters.end_time/3.)   <0.9*parameters.delta_t)||
                  (abs(current_time - 2.*parameters.end_time/3.)<0.9*parameters.delta_t)   ) &&
                  parameters.num_cycle_sets == 1 )
            {
                plotpointfile << std::endl<< std::endl;
            }
            if (( (parameters.geom_type == "Budday_cube_tension_compression_fully_fixed")||
                  (parameters.geom_type == "Budday_cube_tension_compression")||
                  (parameters.geom_type == "Budday_cube_shear_fully_fixed")             ) &&
                ( (abs(current_time - parameters.end_time/9.)   <0.9*parameters.delta_t)||
                  (abs(current_time - 2.*parameters.end_time/9.)<0.9*parameters.delta_t)||
                  (abs(current_time - 3.*parameters.end_time/9.)<0.9*parameters.delta_t)||
                  (abs(current_time - 5.*parameters.end_time/9.)<0.9*parameters.delta_t)||
                  (abs(current_time - 7.*parameters.end_time/9.)<0.9*parameters.delta_t) ) &&
                  parameters.num_cycle_sets == 2 )
            {
                plotpointfile << std::endl<< std::endl;
            }

            plotpointfile <<  std::setprecision(6) << std::scientific;
            plotpointfile << std::setw(16) << current_time        << ","
                          << std::setw(15) << total_vol_reference << ","
                          << std::setw(15) << total_vol_current   << ","
                          << std::setw(15) << total_solid_vol     << ",";

            if (current_time == 0.0)
            {
                for (unsigned int p=0; p<tracked_vertices_IN.size(); ++p)
                {
                    for (unsigned int d=0; d<dim; ++d)
                        plotpointfile << std::setw(15) << 0.0 << ",";

                    plotpointfile << std::setw(15) << parameters.drained_pressure << ",";
                }
                for (unsigned int d=0; d<(3*dim+2); ++d)
                    plotpointfile << std::setw(15) << 0.0 << ",";

                plotpointfile << std::setw(15) << 0.0;
            }
            else
            {
                for (unsigned int p=0; p<tracked_vertices_IN.size(); ++p)
                    for (unsigned int d=0; d<(dim+1); ++d)
                        plotpointfile << std::setw(15) << solution_vertices[p][d]<< ",";

                for (unsigned int d=0; d<dim; ++d)
                    plotpointfile << std::setw(15) << reaction_force[d] << ",";

                for (unsigned int d=0; d<dim; ++d)
                    plotpointfile << std::setw(15) << reaction_force_pressure[d] << ",";

                for (unsigned int d=0; d<dim; ++d)
                    plotpointfile << std::setw(15) << reaction_force_extra[d] << ",";

                plotpointfile << std::setw(15) << total_fluid_flow << ","
                              << std::setw(15) << total_porous_dissipation<< ","
                              << std::setw(15) << total_viscous_dissipation;
            }
            plotpointfile << std::endl;
        }
      }

    //Header for console output file
    template <int dim>
    void Solid<dim>::print_console_file_header(std::ofstream &outputfile) const
    {
            outputfile << "/*-----------------------------------------------------------------------------------------";
            outputfile << "\n\n  Poro-viscoelastic formulation to solve nonlinear solid mechanics problems using deal.ii";
            outputfile << "\n\n  Problem setup by E Comellas and J-P Pelteret, University of Erlangen-Nuremberg, 2018";
            outputfile << "\n\n/*-----------------------------------------------------------------------------------------";
            outputfile << "\n\nCONSOLE OUTPUT: \n\n";
    }

    //Header for plotting output file
    template <int dim>
    void Solid<dim>::print_plot_file_header(std::vector<Point<dim> > &tracked_vertices,
                                            std::ofstream &plotpointfile) const
    {
            plotpointfile << "#\n# *** Solution history for tracked vertices -- DOF: 0 = Ux,  1 = Uy,  2 = Uz,  3 = P ***"
                          << std::endl;

            for  (unsigned int p=0; p<tracked_vertices.size(); ++p)
            {
                plotpointfile << "#        Point " << p << " coordinates:  ";
                for (unsigned int d=0; d<dim; ++d)
                  {
                    plotpointfile << tracked_vertices[p][d];
                    if (!( (p == tracked_vertices.size()-1) && (d == dim-1) ))
                        plotpointfile << ",        ";
                  }
                plotpointfile << std::endl;
            }
            plotpointfile << "#    The reaction force is the integral over the loaded surfaces in the "
                          << "undeformed configuration of the Cauchy stress times the normal surface unit vector.\n"
                          << "#    reac(p) corresponds to the volumetric part of the Cauchy stress due to the pore fluid pressure"
                          << " and reac(E) corresponds to the extra part of the Cauchy stress due to the solid contribution."
                          << std::endl
                          << "#    The fluid flow is the integral over the drained surfaces in the "
                          << "undeformed configuration of the seepage velocity times the normal surface unit vector."
                          << std::endl
                          << "# Column number:"
                          << std::endl
                          << "#";

          unsigned int columns = 24;
          for (unsigned int d=1; d<columns; ++d)
              plotpointfile << std::setw(15)<< d <<",";

            plotpointfile << std::setw(15)<< columns
                          << std::endl
                          << "#"
                          << std::right << std::setw(16) << "Time,"
                          << std::right << std::setw(16) << "ref vol,"
                          << std::right << std::setw(16) << "def vol,"
                          << std::right << std::setw(16) << "solid vol,";
            for (unsigned int p=0; p<tracked_vertices.size(); ++p)
                for (unsigned int d=0; d<(dim+1); ++d)
                    plotpointfile << std::right<< std::setw(11)
                                  <<"P" << p << "[" << d << "],";

            for (unsigned int d=0; d<dim; ++d)
                plotpointfile << std::right<< std::setw(13)
                              << "reaction [" << d << "],";

            for (unsigned int d=0; d<dim; ++d)
                plotpointfile << std::right<< std::setw(13)
                              << "reac(p) [" << d << "],";

            for (unsigned int d=0; d<dim; ++d)
                plotpointfile << std::right<< std::setw(13)
                              << "reac(E) [" << d << "],";

            plotpointfile << std::right<< std::setw(16)<< "fluid flow,"
                          << std::right<< std::setw(16)<< "porous dissip,"
                          << std::right<< std::setw(15)<< "viscous dissip"
                          << std::endl;
    }

    //Footer for console output file
    template <int dim>
    void Solid<dim>::print_console_file_footer(std::ofstream &outputfile) const
    {
           //Copy "parameters" file at end of output file.
           std::ifstream infile("parameters.prm");
           std::string content = "";
           int i;

           for(i=0 ; infile.eof()!=true ; i++)
           {
               char aux = infile.get();
               content += aux;
               if(aux=='\n') content += '#';
           }

           i--;
           content.erase(content.end()-1);
           infile.close();

           outputfile << "\n\n\n\n PARAMETERS FILE USED IN THIS COMPUTATION: \n#"
                      << std::endl
                      << content;
    }

    //Footer for plotting output file
    template <int dim>
    void Solid<dim>::print_plot_file_footer(std::ofstream &plotpointfile) const
    {
           //Copy "parameters" file at end of output file.
           std::ifstream infile("parameters.prm");
           std::string content = "";
           int i;

           for(i=0 ; infile.eof()!=true ; i++)
           {
               char aux = infile.get();
               content += aux;
               if(aux=='\n') content += '#';
           }

           i--;
           content.erase(content.end()-1);
           infile.close();

           plotpointfile << "#"<< std::endl
                         << "#"<< std::endl
                         << "# PARAMETERS FILE USED IN THIS COMPUTATION:" << std::endl
                         << "#"<< std::endl
                         << content;
    }


    // @sect3{Verification examples from Ehlers and Eipper 1999}
    // We group the definition of the geometry, boundary and loading conditions specific to
    // the verification examples from Ehlers and Eipper 1999 into specific classes.

    //@sect4{Base class: Tube geometry and boundary conditions}
    template <int dim>
    class VerificationEhlers1999TubeBase
          : public Solid<dim>
    {
        public:
          VerificationEhlers1999TubeBase (const Parameters::AllParameters &parameters)
            : Solid<dim> (parameters)
          {}

          virtual ~VerificationEhlers1999TubeBase () {}

        private:
          virtual void make_grid() override
          {
            GridGenerator::cylinder( this->triangulation,
                                     0.1,
                                     0.5);

            const double rot_angle = 3.0*numbers::PI/2.0;
            GridTools::rotate( Point<3>::unit_vector(1), rot_angle, this->triangulation);

            this->triangulation.reset_manifold(0);
            static const CylindricalManifold<dim> manifold_description_3d(2);
            this->triangulation.set_manifold (0, manifold_description_3d);
            GridTools::scale(this->parameters.scale, this->triangulation);
            this->triangulation.refine_global(std::max (1U, this->parameters.global_refinement));
            this->triangulation.reset_manifold(0);
          }

          virtual void define_tracked_vertices(std::vector<Point<dim> > &tracked_vertices) override
          {
            tracked_vertices[0][0] = 0.0*this->parameters.scale;
            tracked_vertices[0][1] = 0.0*this->parameters.scale;
            tracked_vertices[0][2] = 0.5*this->parameters.scale;

            tracked_vertices[1][0] = 0.0*this->parameters.scale;
            tracked_vertices[1][1] = 0.0*this->parameters.scale;
            tracked_vertices[1][2] = -0.5*this->parameters.scale;
          }

          virtual void make_dirichlet_constraints(AffineConstraints<double> &constraints) override
          {
            if (this->time.get_timestep() < 2)
            {
              VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                       2,
                                                       Functions::ConstantFunction<dim>(this->parameters.drained_pressure,this->n_components),
                                                       constraints,
                                                       (this->fe.component_mask(this->pressure)));
            }
            else
            {
              VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                       2,
                                                       Functions::ZeroFunction<dim>(this->n_components),
                                                       constraints,
                                                       (this->fe.component_mask(this->pressure)));
            }

            VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                      0,
                                                      Functions::ZeroFunction<dim>(this->n_components),
                                                      constraints,
                                                      (this->fe.component_mask(this->x_displacement)|
                                                       this->fe.component_mask(this->y_displacement)  ) );

            VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                      1,
                                                      Functions::ZeroFunction<dim>(this->n_components),
                                                      constraints,
                                                      (this->fe.component_mask(this->x_displacement) |
                                                       this->fe.component_mask(this->y_displacement) |
                                                       this->fe.component_mask(this->z_displacement) ));
          }

          virtual double
          get_prescribed_fluid_flow (const types::boundary_id &boundary_id,
                                     const Point<dim>         &pt) const override
          {
              (void)pt;
              (void)boundary_id;
              return 0.0;
          }

          virtual types::boundary_id
          get_reaction_boundary_id_for_output() const override
          {
              return 2;
          }

          virtual  std::pair<types::boundary_id,types::boundary_id>
          get_drained_boundary_id_for_output() const override
          {
              return std::make_pair(2,2);
          }

          virtual std::vector<double>
          get_dirichlet_load(const types::boundary_id   &boundary_id,
                             const int                  &direction) const override
          {
              std::vector<double> displ_incr(dim, 0.0);
              (void)boundary_id;
              (void)direction;
              AssertThrow(false, ExcMessage("Displacement loading not implemented for Ehlers verification examples."));

              return displ_incr;
          }
    };

    //@sect4{Derived class: Step load example}
    template <int dim>
    class VerificationEhlers1999StepLoad
          : public VerificationEhlers1999TubeBase<dim>
    {
        public:
          VerificationEhlers1999StepLoad (const Parameters::AllParameters &parameters)
            : VerificationEhlers1999TubeBase<dim> (parameters)
          {}

          virtual ~VerificationEhlers1999StepLoad () {}

        private:
            virtual Tensor<1,dim>
            get_neumann_traction (const types::boundary_id &boundary_id,
                                  const Point<dim>         &pt,
                                  const Tensor<1,dim>      &N) const override
            {
              if (this->parameters.load_type == "pressure")
              {
                if (boundary_id == 2)
                {
                  return this->parameters.load * N;
                }
              }

              (void)pt;

              return Tensor<1,dim>();
            }
    };

    //@sect4{Derived class: Load increasing example}
    template <int dim>
    class VerificationEhlers1999IncreaseLoad
          : public VerificationEhlers1999TubeBase<dim>
    {
        public:
          VerificationEhlers1999IncreaseLoad (const Parameters::AllParameters &parameters)
            : VerificationEhlers1999TubeBase<dim> (parameters)
          {}

          virtual ~VerificationEhlers1999IncreaseLoad () {}

        private:
            virtual Tensor<1,dim>
            get_neumann_traction (const types::boundary_id &boundary_id,
                                  const Point<dim>         &pt,
                                  const Tensor<1,dim>      &N) const override
            {
              if (this->parameters.load_type == "pressure")
              {
                if (boundary_id == 2)
                {
                  const double initial_load = this->parameters.load;
                  const double final_load = 20.0*initial_load;
                  const double initial_time = this->time.get_delta_t();
                  const double final_time = this->time.get_end();
                  const double current_time = this->time.get_current();
                  const double load = initial_load + (final_load-initial_load)*(current_time-initial_time)/(final_time-initial_time);
                  return load * N;
                }
              }

              (void)pt;

              return Tensor<1,dim>();
            }
    };

    //@sect4{Class: Consolidation cube}
    template <int dim>
    class VerificationEhlers1999CubeConsolidation
          : public Solid<dim>
    {
        public:
          VerificationEhlers1999CubeConsolidation (const Parameters::AllParameters &parameters)
            : Solid<dim> (parameters)
          {}

          virtual ~VerificationEhlers1999CubeConsolidation () {}

        private:
          virtual void
          make_grid() override
          {
             GridGenerator::hyper_rectangle(this->triangulation,
                                            Point<dim>(0.0, 0.0, 0.0),
                                            Point<dim>(1.0, 1.0, 1.0),
                                            true);

             GridTools::scale(this->parameters.scale, this->triangulation);
             this->triangulation.refine_global(std::max (1U, this->parameters.global_refinement));

             typename Triangulation<dim>::active_cell_iterator cell =
                     this->triangulation.begin_active(), endc = this->triangulation.end();
             for (; cell != endc; ++cell)
             {
               for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
                 if (cell->face(face)->at_boundary() == true  &&
                     cell->face(face)->center()[2] == 1.0 * this->parameters.scale)
                 {
                   if (cell->face(face)->center()[0] < 0.5 * this->parameters.scale  &&
                       cell->face(face)->center()[1] < 0.5 * this->parameters.scale)
                       cell->face(face)->set_boundary_id(100);
                   else
                       cell->face(face)->set_boundary_id(101);
                 }
             }
          }

          virtual void
          define_tracked_vertices(std::vector<Point<dim> > &tracked_vertices) override
          {
            tracked_vertices[0][0] = 0.0*this->parameters.scale;
            tracked_vertices[0][1] = 0.0*this->parameters.scale;
            tracked_vertices[0][2] = 1.0*this->parameters.scale;

            tracked_vertices[1][0] = 0.0*this->parameters.scale;
            tracked_vertices[1][1] = 0.0*this->parameters.scale;
            tracked_vertices[1][2] = 0.0*this->parameters.scale;
          }

          virtual void
          make_dirichlet_constraints(AffineConstraints<double> &constraints) override
          {
            if (this->time.get_timestep() < 2)
            {
              VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                       101,
                                                       Functions::ConstantFunction<dim>(this->parameters.drained_pressure,this->n_components),
                                                       constraints,
                                                       (this->fe.component_mask(this->pressure)));
            }
            else
            {
              VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                       101,
                                                       Functions::ZeroFunction<dim>(this->n_components),
                                                       constraints,
                                                       (this->fe.component_mask(this->pressure)));
            }

            VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                      0,
                                                      Functions::ZeroFunction<dim>(this->n_components),
                                                      constraints,
                                                      this->fe.component_mask(this->x_displacement));

            VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                      1,
                                                      Functions::ZeroFunction<dim>(this->n_components),
                                                      constraints,
                                                      this->fe.component_mask(this->x_displacement));

            VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                      2,
                                                      Functions::ZeroFunction<dim>(this->n_components),
                                                      constraints,
                                                      this->fe.component_mask(this->y_displacement));

            VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                      3,
                                                      Functions::ZeroFunction<dim>(this->n_components),
                                                      constraints,
                                                      this->fe.component_mask(this->y_displacement));

            VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                      4,
                                                      Functions::ZeroFunction<dim>(this->n_components),
                                                      constraints,
                                                      ( this->fe.component_mask(this->x_displacement) |
                                                        this->fe.component_mask(this->y_displacement) |
                                                        this->fe.component_mask(this->z_displacement) ));
          }

          virtual Tensor<1,dim>
          get_neumann_traction (const types::boundary_id &boundary_id,
                                const Point<dim>         &pt,
                                const Tensor<1,dim>      &N) const override
          {
            if (this->parameters.load_type == "pressure")
            {
              if (boundary_id == 100)
              {
                return this->parameters.load * N;
              }
            }

            (void)pt;

            return Tensor<1,dim>();
          }

          virtual double
          get_prescribed_fluid_flow (const types::boundary_id &boundary_id,
                                     const Point<dim>         &pt) const override
          {
              (void)pt;
              (void)boundary_id;
              return 0.0;
          }

          virtual types::boundary_id
          get_reaction_boundary_id_for_output() const override
          {
              return 100;
          }

          virtual  std::pair<types::boundary_id,types::boundary_id>
          get_drained_boundary_id_for_output() const override
          {
              return std::make_pair(101,101);
          }

          virtual std::vector<double>
          get_dirichlet_load(const types::boundary_id   &boundary_id,
                             const int                  &direction) const override
          {
              std::vector<double> displ_incr(dim, 0.0);
              (void)boundary_id;
              (void)direction;
              AssertThrow(false, ExcMessage("Displacement loading not implemented for Ehlers verification examples."));

              return displ_incr;
          }
    };

    //@sect4{Franceschini experiments}
    template <int dim>
    class Franceschini2006Consolidation
          : public Solid<dim>
    {
        public:
        Franceschini2006Consolidation (const Parameters::AllParameters &parameters)
            : Solid<dim> (parameters)
          {}

          virtual ~Franceschini2006Consolidation () {}

        private:
          virtual void make_grid() override
          {
            const Point<dim-1> mesh_center(0.0, 0.0);
            const double radius = 0.5;
            //const double height = 0.27;  //8.1 mm for 30 mm radius
            const double height = 0.23;  //6.9 mm for 30 mm radius
            Triangulation<dim-1> triangulation_in;
            GridGenerator::hyper_ball( triangulation_in,
                                       mesh_center,
                                       radius);

            GridGenerator::extrude_triangulation(triangulation_in,
                                                  2,
                                                  height,
                                                  this->triangulation);

            const CylindricalManifold<dim> cylinder_3d(2);
            const types::manifold_id cylinder_id = 0;


            this->triangulation.set_manifold(cylinder_id, cylinder_3d);

            for (auto cell : this->triangulation.active_cell_iterators())
            {
              for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
              {
                if (cell->face(face)->at_boundary() == true)
                {
                  if (cell->face(face)->center()[2] == 0.0)
                      cell->face(face)->set_boundary_id(1);

                  else if (cell->face(face)->center()[2] == height)
                      cell->face(face)->set_boundary_id(2);

                  else
                  {
                      cell->face(face)->set_boundary_id(0);
                      cell->face(face)->set_all_manifold_ids(cylinder_id);
                  }
                }
              }
            }

            GridTools::scale(this->parameters.scale, this->triangulation);
            this->triangulation.refine_global(std::max (1U, this->parameters.global_refinement));
          }

          virtual void define_tracked_vertices(std::vector<Point<dim> > &tracked_vertices) override
          {
            tracked_vertices[0][0] = 0.0*this->parameters.scale;
            tracked_vertices[0][1] = 0.0*this->parameters.scale;
	  //  tracked_vertices[0][2] = 0.27*this->parameters.scale;
            tracked_vertices[0][2] = 0.23*this->parameters.scale;

            tracked_vertices[1][0] = 0.0*this->parameters.scale;
            tracked_vertices[1][1] = 0.0*this->parameters.scale;
            tracked_vertices[1][2] = 0.0*this->parameters.scale;
          }

          virtual void make_dirichlet_constraints(AffineConstraints<double> &constraints) override
          {
            if (this->time.get_timestep() < 2)
            {
              VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                       1,
                                                       Functions::ConstantFunction<dim>(this->parameters.drained_pressure,this->n_components),
                                                       constraints,
                                                       (this->fe.component_mask(this->pressure)));

              VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                       2,
                                                       Functions::ConstantFunction<dim>(this->parameters.drained_pressure,this->n_components),
                                                       constraints,
                                                       (this->fe.component_mask(this->pressure)));
            }
            else
            {
              VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                       1,
                                                       Functions::ZeroFunction<dim>(this->n_components),
                                                       constraints,
                                                       (this->fe.component_mask(this->pressure)));

              VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                       2,
                                                       Functions::ZeroFunction<dim>(this->n_components),
                                                       constraints,
                                                       (this->fe.component_mask(this->pressure)));
            }

            VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                      0,
                                                      Functions::ZeroFunction<dim>(this->n_components),
                                                      constraints,
                                                      (this->fe.component_mask(this->x_displacement)|
                                                       this->fe.component_mask(this->y_displacement)  ) );

            VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                      1,
                                                      Functions::ZeroFunction<dim>(this->n_components),
                                                      constraints,
                                                      (this->fe.component_mask(this->x_displacement) |
                                                       this->fe.component_mask(this->y_displacement) |
                                                       this->fe.component_mask(this->z_displacement) ));

            VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                      2,
                                                      Functions::ZeroFunction<dim>(this->n_components),
                                                      constraints,
                                                      (this->fe.component_mask(this->x_displacement) |
                                                       this->fe.component_mask(this->y_displacement) ));
          }

          virtual double
          get_prescribed_fluid_flow (const types::boundary_id &boundary_id,
                                     const Point<dim>         &pt) const override
          {
              (void)pt;
              (void)boundary_id;
              return 0.0;
          }

          virtual types::boundary_id
          get_reaction_boundary_id_for_output() const override
          {
              return 2;
          }

          virtual  std::pair<types::boundary_id,types::boundary_id>
          get_drained_boundary_id_for_output() const override
          {
              return std::make_pair(1,2);
          }

          virtual std::vector<double>
          get_dirichlet_load(const types::boundary_id   &boundary_id,
                             const int                  &direction) const override
          {
              std::vector<double> displ_incr(dim, 0.0);
              (void)boundary_id;
              (void)direction;
              AssertThrow(false, ExcMessage("Displacement loading not implemented for Franceschini examples."));

              return displ_incr;
          }

          virtual Tensor<1,dim>
          get_neumann_traction (const types::boundary_id &boundary_id,
                                const Point<dim>         &pt,
                                const Tensor<1,dim>      &N) const override
          {
            if (this->parameters.load_type == "pressure")
            {
              if (boundary_id == 2)
              {
                return (this->parameters.load * N);
                /*
                const double final_load = this->parameters.load;
                const double final_load_time = 10 * this->time.get_delta_t();
                const double current_time = this->time.get_current();


                const double c = final_load_time / 2.0;
                const double r = 200.0 * 0.03 / c;

                const double load = final_load * std::exp(r * current_time)
                                    / ( std::exp(c * current_time) +  std::exp(r * current_time));
                return load * N;
                */
              }
            }

            (void)pt;

            return Tensor<1,dim>();
          }
    };

    // @sect3{Examples to reproduce experiments by Budday et al. 2017}
    // We group the definition of the geometry, boundary and loading conditions specific to
    // the examples to reproduce experiments by Budday et al. 2017 into specific classes.

    //@sect4{Base class: Cube geometry and loading pattern}
    template <int dim>
    class BrainBudday2017BaseCube
          : public Solid<dim>
    {
        public:
            BrainBudday2017BaseCube (const Parameters::AllParameters &parameters)
            : Solid<dim> (parameters)
          {}

          virtual ~BrainBudday2017BaseCube () {}

        private:
          virtual void
          make_grid() override
          {
            GridGenerator::hyper_cube(this->triangulation,
                                      0.0,
                                      1.0,
                                      true);

            typename Triangulation<dim>::active_cell_iterator cell =
                    this->triangulation.begin_active(), endc = this->triangulation.end();
            for (; cell != endc; ++cell)
            {
              for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
                if (cell->face(face)->at_boundary() == true  &&
                    ( cell->face(face)->boundary_id() == 0 ||
                      cell->face(face)->boundary_id() == 1 ||
                      cell->face(face)->boundary_id() == 2 ||
                      cell->face(face)->boundary_id() == 3    ) )

                      cell->face(face)->set_boundary_id(100);

            }

            GridTools::scale(this->parameters.scale, this->triangulation);
            this->triangulation.refine_global(std::max (1U, this->parameters.global_refinement));
          }

          virtual double
          get_prescribed_fluid_flow (const types::boundary_id &boundary_id,
                                     const Point<dim>         &pt) const override
          {
              (void)pt;
              (void)boundary_id;
              return 0.0;
          }

          virtual  std::pair<types::boundary_id,types::boundary_id>
          get_drained_boundary_id_for_output() const override
          {
              return std::make_pair(100,100);
          }
    };

    //@sect4{Derived class: Uniaxial boundary conditions}
    template <int dim>
    class BrainBudday2017CubeTensionCompression
          : public BrainBudday2017BaseCube<dim>
    {
        public:
          BrainBudday2017CubeTensionCompression (const Parameters::AllParameters &parameters)
            : BrainBudday2017BaseCube<dim> (parameters)
          {}

          virtual ~BrainBudday2017CubeTensionCompression () {}

        private:
          virtual void
          define_tracked_vertices(std::vector<Point<dim> > &tracked_vertices) override
          {
            tracked_vertices[0][0] = 0.5*this->parameters.scale;
            tracked_vertices[0][1] = 0.5*this->parameters.scale;
            tracked_vertices[0][2] = 1.0*this->parameters.scale;

            tracked_vertices[1][0] = 0.5*this->parameters.scale;
            tracked_vertices[1][1] = 0.5*this->parameters.scale;
            tracked_vertices[1][2] = 0.5*this->parameters.scale;
          }

          virtual void
          make_dirichlet_constraints(AffineConstraints<double> &constraints) override
          {
              if (this->time.get_timestep() < 2)
              {
                  VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                           100,
                                                           Functions::ConstantFunction<dim>(this->parameters.drained_pressure,this->n_components),
                                                           constraints,
                                                           (this->fe.component_mask(this->pressure)));
              }
              else
              {
                  VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                            100,
                                                            Functions::ZeroFunction<dim>(this->n_components),
                                                            constraints,
                                                            (this->fe.component_mask(this->pressure)));
              }
              VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                        4,
                                                        Functions::ZeroFunction<dim>(this->n_components),
                                                        constraints,
                                                        this->fe.component_mask(this->z_displacement) );

            Point<dim> fix_node(0.5*this->parameters.scale, 0.5*this->parameters.scale, 0.0);
            typename DoFHandler<dim>::active_cell_iterator
            cell = this->dof_handler_ref.begin_active(), endc = this->dof_handler_ref.end();
            for (; cell != endc; ++cell)
              for (unsigned int node = 0; node < GeometryInfo<dim>::vertices_per_cell; ++node)
              {
                  if (  (abs(cell->vertex(node)[2]-fix_node[2]) < (1e-6 * this->parameters.scale))
                    &&  (abs(cell->vertex(node)[0]-fix_node[0]) < (1e-6 * this->parameters.scale)))
                      constraints.add_line(cell->vertex_dof_index(node, 0));

                  if (  (abs(cell->vertex(node)[2]-fix_node[2]) < (1e-6 * this->parameters.scale))
                    &&  (abs(cell->vertex(node)[1]-fix_node[1]) < (1e-6 * this->parameters.scale)))
                    constraints.add_line(cell->vertex_dof_index(node, 1));
              }

            if (this->parameters.load_type == "displacement")
            {
                const std::vector<double> value = get_dirichlet_load(5,2);
                FEValuesExtractors::Scalar direction;
                direction = this->z_displacement;

                VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                           5,
                                                           Functions::ConstantFunction<dim>(value[2],this->n_components),
                                                           constraints,
                                                           this->fe.component_mask(direction));
            }
          }

          virtual Tensor<1,dim>
          get_neumann_traction (const types::boundary_id &boundary_id,
                                const Point<dim>         &pt,
                                const Tensor<1,dim>      &N) const override
          {
              if (this->parameters.load_type == "pressure")
              {
                if (boundary_id ==  5)
                {
                    const double final_load   = this->parameters.load;
                    const double current_time = this->time.get_current();
                    const double final_time   = this->time.get_end();
                    const double num_cycles   = 3.0;

                    return  final_load/2.0 * (1.0 - std::sin(numbers::PI * (2.0*num_cycles*current_time/final_time + 0.5))) * N;
                }
              }

              (void)pt;

              return Tensor<1,dim>();
            }

          virtual types::boundary_id
          get_reaction_boundary_id_for_output() const override
          {
              return 5;
          }

          virtual std::vector<double>
          get_dirichlet_load(const types::boundary_id   &boundary_id,
                             const int                  &direction) const override
          {
                std::vector<double> displ_incr(dim,0.0);

                if ( (boundary_id == 5) && (direction == 2) )
                {
                    const double final_displ  = this->parameters.load;
                    const double current_time = this->time.get_current();
                    const double final_time   = this->time.get_end();
                    const double delta_time   = this->time.get_delta_t();
                    const double num_cycles   = 3.0;
                    double current_displ = 0.0;
                    double previous_displ = 0.0;

                    if (this->parameters.num_cycle_sets == 1)
                    {
                        current_displ  = final_displ/2.0 * (1.0
                          - std::sin(numbers::PI * (2.0*num_cycles*current_time/final_time + 0.5)));
                        previous_displ = final_displ/2.0 * (1.0
                          - std::sin(numbers::PI * (2.0*num_cycles*(current_time-delta_time)/final_time + 0.5)));
                    }
                    else
                    {
                        if ( current_time <= (final_time*1.0/3.0) )
                        {
                            current_displ  = final_displ/2.0 * (1.0 - std::sin(numbers::PI *
                                                 (2.0*num_cycles*current_time/(final_time*1.0/3.0) + 0.5)));
                            previous_displ = final_displ/2.0 * (1.0 - std::sin(numbers::PI *
                                                 (2.0*num_cycles*(current_time-delta_time)/(final_time*1.0/3.0) + 0.5)));
                        }
                        else
                        {
                            current_displ  = final_displ * (1.0 - std::sin(numbers::PI *
                                                 (2.0*num_cycles*current_time / (final_time*2.0/3.0)
                                                  - (num_cycles - 0.5) )));
                            previous_displ = final_displ * (1.0 - std::sin(numbers::PI *
                                                 (2.0*num_cycles*(current_time-delta_time) / (final_time*2.0/3.0)
                                                  - (num_cycles - 0.5))));
                        }
                    }
                    displ_incr[2] = current_displ - previous_displ;
                }
                return displ_incr;
          }
    };

    //@sect4{Derived class: No lateral displacement in loading surfaces}
    template <int dim>
    class BrainBudday2017CubeTensionCompressionFullyFixed
          : public BrainBudday2017BaseCube<dim>
    {
        public:
          BrainBudday2017CubeTensionCompressionFullyFixed (const Parameters::AllParameters &parameters)
            : BrainBudday2017BaseCube<dim> (parameters)
          {}

          virtual ~BrainBudday2017CubeTensionCompressionFullyFixed () {}

        private:
          virtual void
          define_tracked_vertices(std::vector<Point<dim> > &tracked_vertices) override
          {
            tracked_vertices[0][0] = 0.5*this->parameters.scale;
            tracked_vertices[0][1] = 0.5*this->parameters.scale;
            tracked_vertices[0][2] = 1.0*this->parameters.scale;

            tracked_vertices[1][0] = 0.5*this->parameters.scale;
            tracked_vertices[1][1] = 0.5*this->parameters.scale;
            tracked_vertices[1][2] = 0.5*this->parameters.scale;
          }

          virtual void
          make_dirichlet_constraints(AffineConstraints<double> &constraints) override
          {
              if (this->time.get_timestep() < 2)
              {
                  VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                           100,
                                                           Functions::ConstantFunction<dim>(this->parameters.drained_pressure,this->n_components),
                                                           constraints,
                                                           (this->fe.component_mask(this->pressure)));
              }
              else
              {
                  VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                            100,
                                                            Functions::ZeroFunction<dim>(this->n_components),
                                                            constraints,
                                                            (this->fe.component_mask(this->pressure)));
              }

              VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                        4,
                                                        Functions::ZeroFunction<dim>(this->n_components),
                                                        constraints,
                                                      (this->fe.component_mask(this->x_displacement) |
                                                       this->fe.component_mask(this->y_displacement) |
                                                       this->fe.component_mask(this->z_displacement) ));


            if (this->parameters.load_type == "displacement")
            {
                const std::vector<double> value = get_dirichlet_load(5,2);
                FEValuesExtractors::Scalar direction;
                direction = this->z_displacement;

                VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                           5,
                                                           Functions::ConstantFunction<dim>(value[2],this->n_components),
                                                           constraints,
                                                           this->fe.component_mask(direction) );

               VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                          5,
                                                          Functions::ZeroFunction<dim>(this->n_components),
                                                          constraints,
                                                          (this->fe.component_mask(this->x_displacement) |
                                                           this->fe.component_mask(this->y_displacement) ));
            }
          }

          virtual Tensor<1,dim>
          get_neumann_traction (const types::boundary_id &boundary_id,
                                const Point<dim>         &pt,
                                const Tensor<1,dim>      &N) const override
          {
              if (this->parameters.load_type == "pressure")
              {
                if (boundary_id ==  5)
                {
                    const double final_load   = this->parameters.load;
                    const double current_time = this->time.get_current();
                    const double final_time   = this->time.get_end();
                    const double num_cycles   = 3.0;

                    return  final_load/2.0 * (1.0 - std::sin(numbers::PI * (2.0*num_cycles*current_time/final_time + 0.5))) * N;
                }
              }

              (void)pt;

              return Tensor<1,dim>();
            }

          virtual types::boundary_id
          get_reaction_boundary_id_for_output() const override
          {
              return 5;
          }

          virtual std::vector<double>
          get_dirichlet_load(const types::boundary_id   &boundary_id,
                             const int                  &direction) const override
          {
                std::vector<double> displ_incr(dim,0.0);

                if ( (boundary_id == 5) && (direction == 2) )
                {
                    const double final_displ  = this->parameters.load;
                    const double current_time = this->time.get_current();
                    const double final_time   = this->time.get_end();
                    const double delta_time   = this->time.get_delta_t();
                    const double num_cycles   = 3.0;
                    double current_displ = 0.0;
                    double previous_displ = 0.0;

                    if (this->parameters.num_cycle_sets == 1)
                    {
                        current_displ  = final_displ/2.0 * (1.0 - std::sin(numbers::PI * (2.0*num_cycles*current_time/final_time + 0.5)));
                        previous_displ = final_displ/2.0 * (1.0 - std::sin(numbers::PI * (2.0*num_cycles*(current_time-delta_time)/final_time + 0.5)));
                    }
                    else
                    {
                        if ( current_time <= (final_time*1.0/3.0) )
                        {
                            current_displ  = final_displ/2.0 * (1.0 - std::sin(numbers::PI *
                                                 (2.0*num_cycles*current_time/(final_time*1.0/3.0) + 0.5)));
                            previous_displ = final_displ/2.0 * (1.0 - std::sin(numbers::PI *
                                                 (2.0*num_cycles*(current_time-delta_time)/(final_time*1.0/3.0) + 0.5)));
                        }
                        else
                        {
                            current_displ  = final_displ * (1.0 - std::sin(numbers::PI *
                                                 (2.0*num_cycles*current_time / (final_time*2.0/3.0)
                                                  - (num_cycles - 0.5) )));
                            previous_displ = final_displ * (1.0 - std::sin(numbers::PI *
                                                 (2.0*num_cycles*(current_time-delta_time) / (final_time*2.0/3.0)
                                                  - (num_cycles - 0.5))));
                        }
                    }
                    displ_incr[2] = current_displ - previous_displ;
                }
                return displ_incr;
          }
    };

    //@sect4{Derived class: No lateral or vertical displacement in loading surface}
    template <int dim>
    class BrainBudday2017CubeShearFullyFixed
          : public BrainBudday2017BaseCube<dim>
    {
        public:
          BrainBudday2017CubeShearFullyFixed (const Parameters::AllParameters &parameters)
            : BrainBudday2017BaseCube<dim> (parameters)
          {}

          virtual ~BrainBudday2017CubeShearFullyFixed () {}

        private:
          virtual void
          define_tracked_vertices(std::vector<Point<dim> > &tracked_vertices) override
          {
            tracked_vertices[0][0] = 0.75*this->parameters.scale;
            tracked_vertices[0][1] = 0.5*this->parameters.scale;
            tracked_vertices[0][2] = 0.0*this->parameters.scale;

            tracked_vertices[1][0] = 0.25*this->parameters.scale;
            tracked_vertices[1][1] = 0.5*this->parameters.scale;
            tracked_vertices[1][2] = 0.0*this->parameters.scale;
          }

          virtual void
          make_dirichlet_constraints(AffineConstraints<double> &constraints) override
          {
              if (this->time.get_timestep() < 2)
              {
                  VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                           100,
                                                           Functions::ConstantFunction<dim>(this->parameters.drained_pressure,this->n_components),
                                                           constraints,
                                                           (this->fe.component_mask(this->pressure)));
              }
              else
              {
                  VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                            100,
                                                            Functions::ZeroFunction<dim>(this->n_components),
                                                            constraints,
                                                            (this->fe.component_mask(this->pressure)));
              }

              VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                        5,
                                                        Functions::ZeroFunction<dim>(this->n_components),
                                                        constraints,
                                                      (this->fe.component_mask(this->x_displacement) |
                                                       this->fe.component_mask(this->y_displacement) |
                                                       this->fe.component_mask(this->z_displacement) ));


            if (this->parameters.load_type == "displacement")
            {
                const std::vector<double> value = get_dirichlet_load(4,0);
                FEValuesExtractors::Scalar direction;
                direction = this->x_displacement;

                VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                           4,
                                                           Functions::ConstantFunction<dim>(value[0],this->n_components),
                                                           constraints,
                                                           this->fe.component_mask(direction));

               VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                          4,
                                                          Functions::ZeroFunction<dim>(this->n_components),
                                                          constraints,
                                                          (this->fe.component_mask(this->y_displacement) |
                                                           this->fe.component_mask(this->z_displacement) ));
            }
          }

          virtual Tensor<1,dim>
          get_neumann_traction (const types::boundary_id &boundary_id,
                                const Point<dim>         &pt,
                                const Tensor<1,dim>      &N) const override
          {
              if (this->parameters.load_type == "pressure")
              {
                if (boundary_id ==  4)
                {
                    const double final_load   = this->parameters.load;
                    const double current_time = this->time.get_current();
                    const double final_time   = this->time.get_end();
                    const double num_cycles   = 3.0;
                    const Tensor<1,3> axis ({0.0,1.0,0.0});
                    const double angle = numbers::PI;
                    static const Tensor< 2, dim, double> R(Physics::Transformations::Rotations::rotation_matrix_3d(axis,angle));

                    return  (final_load * (std::sin(2.0*(numbers::PI)*num_cycles*current_time/final_time)) * (R * N));
                }
              }

              (void)pt;

              return Tensor<1,dim>();
            }

          virtual types::boundary_id
          get_reaction_boundary_id_for_output() const override
          {
              return 4;
          }

          virtual std::vector<double>
          get_dirichlet_load(const types::boundary_id   &boundary_id,
                             const int                  &direction) const override
          {
                std::vector<double> displ_incr (dim, 0.0);

                if ( (boundary_id == 4) && (direction == 0) )
                {
                    const double final_displ  = this->parameters.load;
                    const double current_time = this->time.get_current();
                    const double final_time   = this->time.get_end();
                    const double delta_time   = this->time.get_delta_t();
                    const double num_cycles   = 3.0;
                    double current_displ = 0.0;
                    double previous_displ = 0.0;

                    if (this->parameters.num_cycle_sets == 1)
                    {
                        current_displ  = final_displ * (std::sin(2.0*(numbers::PI)*num_cycles*current_time/final_time));
                        previous_displ = final_displ * (std::sin(2.0*(numbers::PI)*num_cycles*(current_time-delta_time)/final_time));
                    }
                    else
                    {
                        AssertThrow(false, ExcMessage("Problem type not defined. Budday shear experiments implemented only for one set of cycles."));
                    }
                    displ_incr[0] = current_displ - previous_displ;
                }
                return displ_incr;
          }
    };

}

// @sect3{Main function}
// Lastly we provide the main driver function which is similar to the other tutorials.
int main (int argc, char *argv[])
{
  using namespace dealii;
  using namespace NonLinearPoroViscoElasticity;

  const unsigned int n_tbb_processes = 1;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, n_tbb_processes);

  try
    {
      Parameters::AllParameters parameters ("parameters.prm");
      if (parameters.geom_type == "Ehlers_tube_step_load")
      {
        VerificationEhlers1999StepLoad<3> solid_3d(parameters);
        solid_3d.run();
      }
      else if (parameters.geom_type == "Ehlers_tube_increase_load")
      {
        VerificationEhlers1999IncreaseLoad<3> solid_3d(parameters);
        solid_3d.run();
      }
      else if (parameters.geom_type == "Ehlers_cube_consolidation")
      {
        VerificationEhlers1999CubeConsolidation<3> solid_3d(parameters);
        solid_3d.run();
      }
      else if (parameters.geom_type == "Franceschini_consolidation")
      {
        Franceschini2006Consolidation<3> solid_3d(parameters);
        solid_3d.run();
      }
      else if (parameters.geom_type == "Budday_cube_tension_compression")
      {
        BrainBudday2017CubeTensionCompression<3> solid_3d(parameters);
        solid_3d.run();
      }
      else if (parameters.geom_type == "Budday_cube_tension_compression_fully_fixed")
      {
        BrainBudday2017CubeTensionCompressionFullyFixed<3> solid_3d(parameters);
        solid_3d.run();
      }
      else if (parameters.geom_type == "Budday_cube_shear_fully_fixed")
      {
        BrainBudday2017CubeShearFullyFixed<3> solid_3d(parameters);
        solid_3d.run();
      }
      else
      {
        AssertThrow(false, ExcMessage("Problem type not defined. Current setting: " + parameters.geom_type));
      }

    }
  catch (std::exception &exc)
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
          std::cerr << std::endl << std::endl
                    << "----------------------------------------------------"
                    << std::endl;
          std::cerr << "Exception on processing: " << std::endl << exc.what()
                    << std::endl << "Aborting!" << std::endl
                    << "----------------------------------------------------"
                    << std::endl;

          return 1;
      }
    }
  catch (...)
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
          std::cerr << std::endl << std::endl
                    << "----------------------------------------------------"
                    << std::endl;
          std::cerr << "Unknown exception!" << std::endl << "Aborting!"
                    << std::endl
                    << "----------------------------------------------------"
                    << std::endl;
          return 1;
      }
    }
  return 0;
}
