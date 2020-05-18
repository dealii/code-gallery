/* ---------------------------------------------------------------------
 * Copyright (C) 2017 by the deal.II authors and
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
 * Authors: Jean-Paul Pelteret, University of Erlangen-Nuremberg, 2017
 */

#include <deal.II/base/mpi.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/solver_selector.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/trilinos_linear_operator.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/physics/transformations.h>
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>
#include <iostream>
#include <fstream>
#include <numeric>

#include <deal.II/grid/grid_out.h>

namespace ViscoElasStripHole
{
  using namespace dealii;
  namespace LA = TrilinosWrappers;
  namespace Parameters
  {
    struct BoundaryConditions
    {
      BoundaryConditions();

      std::string driver;
      double stretch;
      double pressure;
      double load_time;

      const types::boundary_id boundary_id_minus_X;
      const types::boundary_id boundary_id_plus_X;
      const types::boundary_id boundary_id_minus_Y;
      const types::boundary_id boundary_id_plus_Y;
      const types::boundary_id boundary_id_minus_Z;
      const types::boundary_id boundary_id_plus_Z;
      const types::boundary_id boundary_id_hole;
      const types::manifold_id manifold_id_hole;

      static void
      declare_parameters(ParameterHandler &prm);
      void
      parse_parameters(ParameterHandler &prm);
    };
    BoundaryConditions::BoundaryConditions()
      :
      driver ("Neumann"),
      stretch (2.0),
      pressure(0.0),
      load_time(2.5),
      boundary_id_minus_X (1),
      boundary_id_plus_X (2),
      boundary_id_minus_Y (3),
      boundary_id_plus_Y (4),
      boundary_id_minus_Z (5),
      boundary_id_plus_Z (6),
      boundary_id_hole (10),
      manifold_id_hole (10)
    { }
    void BoundaryConditions::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary conditions");
      {
        prm.declare_entry("Driver", "Dirichlet",
                          Patterns::Selection("Dirichlet|Neumann"),
                          "Driver boundary condition for the problem");
        prm.declare_entry("Final stretch", "2.0",
                          Patterns::Double(1.0),
                          "Positive stretch applied length-ways to the strip");
        prm.declare_entry("Applied pressure", "0.0",
                          Patterns::Double(-1e3,1e3),
                          "Hydrostatic pressure applied (in the referential configuration) to the interior surface of the hole");
        prm.declare_entry("Load time", "2.5",
                          Patterns::Double(0.0),
                          "Total time over which the stretch/pressure is ramped up");
      }
      prm.leave_subsection();
    }
    void BoundaryConditions::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary conditions");
      {
        driver = prm.get("Driver");
        stretch = prm.get_double("Final stretch");
        pressure = prm.get_double("Applied pressure");
        load_time = prm.get_double("Load time");
      }
      prm.leave_subsection();
    }
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
        prm.declare_entry("Polynomial degree", "2",
                          Patterns::Integer(0),
                          "Displacement system polynomial order");
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
        poly_degree = prm.get_integer("Polynomial degree");
        quad_order = prm.get_integer("Quadrature order");
      }
      prm.leave_subsection();
    }
    struct Geometry
    {
      double length;
      double width;
      double thickness;
      double hole_diameter;
      double hole_division_fraction;
      unsigned int n_repetitions_xy;
      unsigned int n_repetitions_z;
      unsigned int global_refinement;
      double       scale;
      static void
      declare_parameters(ParameterHandler &prm);
      void
      parse_parameters(ParameterHandler &prm);
    };
    void Geometry::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        prm.declare_entry("Length", "100.0",
                          Patterns::Double(0.0),
                          "Total sample length");
        prm.declare_entry("Width", "50.0",
                          Patterns::Double(0.0),
                          "Total sample width");
        prm.declare_entry("Thickness", "5.0",
                          Patterns::Double(0.0),
                          "Total sample thickness");
        prm.declare_entry("Hole diameter", "20.0",
                          Patterns::Double(0.0),
                          "Hole diameter");
        prm.declare_entry("Hole division fraction", "0.5",
                          Patterns::Double(0.0,1.0),
                          "A geometric factor affecting the discretisation near the hole");
        prm.declare_entry("Number of subdivisions in cross-section", "2",
                          Patterns::Integer(1.0),
                          "A factor defining the number of initial grid subdivisions in the cross-section");
        prm.declare_entry("Number of subdivisions thickness", "6",
                          Patterns::Integer(1.0),
                          "A factor defining the number of initial grid subdivisions through the thickness");
        prm.declare_entry("Global refinement", "2",
                          Patterns::Integer(0),
                          "Global refinement level");
        prm.declare_entry("Grid scale", "1e-3",
                          Patterns::Double(0.0),
                          "Global grid scaling factor");
      }
      prm.leave_subsection();
    }
    void Geometry::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        length = prm.get_double("Length");
        width = prm.get_double("Width");
        thickness = prm.get_double("Thickness");
        hole_diameter = prm.get_double("Hole diameter");
        hole_division_fraction = prm.get_double("Hole division fraction");
        n_repetitions_xy = prm.get_integer("Number of subdivisions in cross-section");
        n_repetitions_z = prm.get_integer("Number of subdivisions thickness");
        global_refinement = prm.get_integer("Global refinement");
        scale = prm.get_double("Grid scale");
      }
      prm.leave_subsection();
    }
    struct Materials
    {
      double nu_e;
      double mu_e;
      double mu_v;
      double tau_v;
      static void
      declare_parameters(ParameterHandler &prm);
      void
      parse_parameters(ParameterHandler &prm);
    };
    void Materials::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material properties");
      {
        prm.declare_entry("Poisson's ratio", "0.4999",
                          Patterns::Double(-1.0,0.5),
                          "Poisson's ratio");
        prm.declare_entry("Elastic shear modulus", "80.194e6",
                          Patterns::Double(0.0),
                          "Elastic shear modulus");
        prm.declare_entry("Viscous shear modulus", "80.194e6",
                          Patterns::Double(0.0),
                          "Viscous shear modulus");
        prm.declare_entry("Viscous relaxation time", "2.0",
                          Patterns::Double(0.0),
                          "Viscous relaxation time");
      }
      prm.leave_subsection();
    }
    void Materials::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material properties");
      {
        nu_e = prm.get_double("Poisson's ratio");
        mu_e = prm.get_double("Elastic shear modulus");
        mu_v = prm.get_double("Viscous shear modulus");
        tau_v = prm.get_double("Viscous relaxation time");
      }
      prm.leave_subsection();
    }
    struct LinearSolver
    {
      std::string type_lin;
      double      tol_lin;
      double      max_iterations_lin;
      static void
      declare_parameters(ParameterHandler &prm);
      void
      parse_parameters(ParameterHandler &prm);
    };
    void LinearSolver::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Linear solver");
      {
        prm.declare_entry("Solver type", "cg",
                          Patterns::Selection(SolverSelector<LA::MPI::Vector>::get_solver_names()),
                          "Type of solver used to solve the linear system");
        prm.declare_entry("Residual", "1e-6",
                          Patterns::Double(0.0),
                          "Linear solver residual (scaled by residual norm)");
        prm.declare_entry("Max iteration multiplier", "1",
                          Patterns::Double(1.0),
                          "Linear solver iterations (multiples of the system matrix size)");
      }
      prm.leave_subsection();
    }
    void LinearSolver::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Linear solver");
      {
        type_lin = prm.get("Solver type");
        tol_lin = prm.get_double("Residual");
        max_iterations_lin = prm.get_double("Max iteration multiplier");
      }
      prm.leave_subsection();
    }
    struct NonlinearSolver
    {
      unsigned int max_iterations_NR;
      double       tol_f;
      double       tol_u;
      static void
      declare_parameters(ParameterHandler &prm);
      void
      parse_parameters(ParameterHandler &prm);
    };
    void NonlinearSolver::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        prm.declare_entry("Max iterations Newton-Raphson", "10",
                          Patterns::Integer(0),
                          "Number of Newton-Raphson iterations allowed");
        prm.declare_entry("Tolerance displacement", "1.0e-6",
                          Patterns::Double(0.0),
                          "Displacement error tolerance");
        prm.declare_entry("Tolerance force", "1.0e-9",
                          Patterns::Double(0.0),
                          "Force residual tolerance");
      }
      prm.leave_subsection();
    }
    void NonlinearSolver::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        max_iterations_NR = prm.get_integer("Max iterations Newton-Raphson");
        tol_f   = prm.get_double("Tolerance force");
        tol_u   = prm.get_double("Tolerance displacement");
      }
      prm.leave_subsection();
    }
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
        prm.declare_entry("End time", "1",
                          Patterns::Double(),
                          "End time");
        prm.declare_entry("Time step size", "0.1",
                          Patterns::Double(),
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
    struct AllParameters
      : public BoundaryConditions,
        public FESystem,
        public Geometry,
        public Materials,
        public LinearSolver,
        public NonlinearSolver,
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
      BoundaryConditions::declare_parameters(prm);
      FESystem::declare_parameters(prm);
      Geometry::declare_parameters(prm);
      Materials::declare_parameters(prm);
      LinearSolver::declare_parameters(prm);
      NonlinearSolver::declare_parameters(prm);
      Time::declare_parameters(prm);
    }
    void AllParameters::parse_parameters(ParameterHandler &prm)
    {
      BoundaryConditions::parse_parameters(prm);
      FESystem::parse_parameters(prm);
      Geometry::parse_parameters(prm);
      Materials::parse_parameters(prm);
      LinearSolver::parse_parameters(prm);
      NonlinearSolver::parse_parameters(prm);
      Time::parse_parameters(prm);
    }
  }
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
  class Material_Compressible_Three_Field_Linear_Viscoelastic
  {
  public:
    Material_Compressible_Three_Field_Linear_Viscoelastic(const double mu_e,
                                                          const double nu_e,
                                                          const double mu_v,
                                                          const double tau_v,
                                                          const Time  &time)
      :
      kappa((2.0 * mu_e * (1.0 + nu_e)) / (3.0 * (1.0 - 2.0 * nu_e))),
      mu_e(mu_e),
      mu_v(mu_v),
      tau_v(tau_v),
      time(time),
      Q_n_t(Physics::Elasticity::StandardTensors<dim>::I),
      Q_t1(Physics::Elasticity::StandardTensors<dim>::I)
    {
      Assert(kappa > 0, ExcInternalError());
    }
    ~Material_Compressible_Three_Field_Linear_Viscoelastic()
    {}

    SymmetricTensor<2,dim>
    get_tau(const Tensor<2,dim> &F,
            const double        &p_tilde) const
    {
      return get_tau_iso(F) + get_tau_vol(F,p_tilde);
    }
    SymmetricTensor<4,dim> get_Jc(const Tensor<2,dim> &F,
                                  const double        &p_tilde) const
    {
      return get_Jc_iso(F) + get_Jc_vol(F,p_tilde);
    }
    double
    get_dPsi_vol_dJ(const double &J_tilde) const
    {
      return (kappa / 2.0) * (J_tilde - 1.0 / J_tilde);
    }
    double
    get_d2Psi_vol_dJ2(const double &J_tilde) const
    {
      return ( (kappa / 2.0) * (1.0 + 1.0 / (J_tilde * J_tilde)));
    }
    void
    update_internal_equilibrium(const Tensor<2, dim> &F,
                                const double         &/*p_tilde*/,
                                const double         &/*J_tilde*/)
    {
      const double det_F = determinant(F);
      const SymmetricTensor<2,dim> C_bar = std::pow(det_F, -2.0 / dim) * Physics::Elasticity::Kinematics::C(F);
      // Linder2011 eq 54
      // Assumes first-oder backward Euler time discretisation
      Q_n_t = (1.0/(1.0 + time.get_delta_t()/tau_v))*(Q_t1 + (time.get_delta_t()/tau_v)*invert(C_bar));
    }
    void
    update_end_timestep()
    {
      Q_t1 = Q_n_t;
    }

  protected:
    const double kappa;
    const double mu_e;
    const double mu_v;
    const double tau_v;
    const Time  &time;
    SymmetricTensor<2,dim> Q_n_t; // Value of internal variable at this Newton step and timestep
    SymmetricTensor<2,dim> Q_t1; // Value of internal variable at the previous timestep

    SymmetricTensor<2, dim>
    get_tau_vol(const Tensor<2,dim> &F,
                const double        &p_tilde) const
    {
      const double det_F = determinant(F);

      return p_tilde * det_F * Physics::Elasticity::StandardTensors<dim>::I;
    }
    SymmetricTensor<2, dim>
    get_tau_iso(const Tensor<2,dim> &F) const
    {
      return Physics::Elasticity::StandardTensors<dim>::dev_P * get_tau_bar(F);
    }
    SymmetricTensor<2, dim>
    get_tau_bar(const Tensor<2,dim> &F) const
    {
      const double det_F = determinant(F);
      const Tensor<2,dim> F_bar = std::pow(det_F, -1.0 / dim) * F;
      const SymmetricTensor<2,dim> b_bar = std::pow(det_F, -2.0 / dim) * symmetrize(F * transpose(F));
      // Elastic Neo-Hookean + Linder2011 eq 47
      return mu_e * b_bar
             + mu_v * symmetrize(F_bar*static_cast<Tensor<2,dim> >(Q_n_t)*transpose(F_bar));
    }
    SymmetricTensor<4, dim> get_Jc_vol(const Tensor<2,dim> &F,
                                       const double        &p_tilde) const
    {
      const double det_F = determinant(F);
      return p_tilde * det_F
             * ( Physics::Elasticity::StandardTensors<dim>::IxI
                 - (2.0 * Physics::Elasticity::StandardTensors<dim>::S) );
    }
    SymmetricTensor<4, dim> get_Jc_iso(const Tensor<2,dim> &F) const
    {
      const SymmetricTensor<2, dim> tau_bar = get_tau_bar(F);
      const SymmetricTensor<2, dim> tau_iso = get_tau_iso(F);
      const SymmetricTensor<4, dim> tau_iso_x_I
        = outer_product(tau_iso,
                        Physics::Elasticity::StandardTensors<dim>::I);
      const SymmetricTensor<4, dim> I_x_tau_iso
        = outer_product(Physics::Elasticity::StandardTensors<dim>::I,
                        tau_iso);
      const SymmetricTensor<4, dim> c_bar = get_c_bar(F);
      return (2.0 / dim) * trace(tau_bar)
             * Physics::Elasticity::StandardTensors<dim>::dev_P
             - (2.0 / dim) * (tau_iso_x_I + I_x_tau_iso)
             + Physics::Elasticity::StandardTensors<dim>::dev_P * c_bar
             * Physics::Elasticity::StandardTensors<dim>::dev_P;
    }
    SymmetricTensor<4, dim> get_c_bar(const Tensor<2,dim> &/*F*/) const
    {
      // Elastic Neo-Hookean + Linder2011 eq 56
      return -2.0*mu_v*((time.get_delta_t()/tau_v)/(1.0 + time.get_delta_t()/tau_v))*Physics::Elasticity::StandardTensors<dim>::S;
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
    setup_lqp (const Parameters::AllParameters &parameters,
               const Time                      &time)
    {
      material.reset(new Material_Compressible_Three_Field_Linear_Viscoelastic<dim>(
                       parameters.mu_e, parameters.nu_e,
                       parameters.mu_v, parameters.tau_v,
                       time));
    }

    SymmetricTensor<2, dim>
    get_tau(const Tensor<2, dim> &F,
            const double         &p_tilde) const
    {
      return material->get_tau(F, p_tilde);
    }
    SymmetricTensor<4, dim>
    get_Jc(const Tensor<2, dim> &F,
           const double         &p_tilde) const
    {
      return material->get_Jc(F, p_tilde);
    }
    double
    get_dPsi_vol_dJ(const double &J_tilde) const
    {
      return material->get_dPsi_vol_dJ(J_tilde);
    }
    double
    get_d2Psi_vol_dJ2(const double &J_tilde) const
    {
      return material->get_d2Psi_vol_dJ2(J_tilde);
    }
    void
    update_internal_equilibrium(const Tensor<2, dim> &F,
                                const double         &p_tilde,
                                const double         &J_tilde)
    {
      material->update_internal_equilibrium(F,p_tilde,J_tilde);
    }
    void
    update_end_timestep()
    {
      material->update_end_timestep();
    }
  private:
    std::shared_ptr< Material_Compressible_Three_Field_Linear_Viscoelastic<dim> > material;
  };
  template <int dim>
  class Solid
  {
  public:
    Solid(const std::string &input_file);
    virtual
    ~Solid();
    void
    run();
  private:
    struct PerTaskData_ASM;
    struct ScratchData_ASM;
    void
    make_grid();
    void
    make_2d_quarter_plate_with_hole(Triangulation<2> &tria_2d,
                                    const double half_length,
                                    const double half_width,
                                    const double hole_radius,
                                    const unsigned int n_repetitions_xy = 1,
                                    const double hole_division_fraction = 0.25);
    void
    setup_system(LA::MPI::BlockVector &solution_delta);
    void
    determine_component_extractors();
    void
    assemble_system(const LA::MPI::BlockVector &solution_delta);
    void
    assemble_system_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                             ScratchData_ASM &scratch,
                             PerTaskData_ASM &data) const;
    void
    copy_local_to_global_system(const PerTaskData_ASM &data);
    void
    make_constraints(const int &it_nr);
    void
    setup_qph();
    void
    solve_nonlinear_timestep(LA::MPI::BlockVector &solution_delta);
    std::pair<unsigned int, double>
    solve_linear_system(LA::MPI::BlockVector &newton_update);
    LA::MPI::BlockVector
    get_solution_total(const LA::MPI::BlockVector &solution_delta) const;
    void
    update_end_timestep();
    void
    output_results(const unsigned int timestep,
                   const double       current_time) const;
    void
    compute_vertex_positions(std::vector<double> &real_time,
                             std::vector<std::vector<Point<dim> > > &tracked_vertices,
                             const LA::MPI::BlockVector &solution_total) const;

    // Parallel communication
    MPI_Comm                         mpi_communicator;
    const unsigned int               n_mpi_processes;
    const unsigned int               this_mpi_process;
    mutable ConditionalOStream       pcout;

    Parameters::AllParameters parameters;
    Triangulation<dim>        triangulation;
    Time                      time;
    mutable TimerOutput       timer;
    CellDataStorage<typename Triangulation<dim>::cell_iterator,
                    PointHistory<dim> > quadrature_point_history;
    const unsigned int               degree;
    const FESystem<dim>              fe;
    DoFHandler<dim>                  dof_handler;
    const unsigned int               dofs_per_cell;
    const FEValuesExtractors::Vector u_fe;
    const FEValuesExtractors::Scalar p_fe;
    const FEValuesExtractors::Scalar J_fe;
    static const unsigned int n_blocks = 3;
    static const unsigned int n_components = dim + 2;
    static const unsigned int first_u_component = 0;
    static const unsigned int p_component = dim;
    static const unsigned int J_component = dim + 1;
    enum
    {
      u_block = 0,
      p_block = 1,
      J_block = 2
    };
    // Block data
    std::vector<unsigned int> block_component;

    // DoF index data
    std::vector<IndexSet> all_locally_owned_dofs;
    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;
    std::vector<IndexSet> locally_owned_partitioning;
    std::vector<IndexSet> locally_relevant_partitioning;
    std::vector<types::global_dof_index> dofs_per_block;
    std::vector<types::global_dof_index> element_indices_u;
    std::vector<types::global_dof_index> element_indices_p;
    std::vector<types::global_dof_index> element_indices_J;
    const QGauss<dim>     qf_cell;
    const QGauss<dim - 1> qf_face;
    const unsigned int    n_q_points;
    const unsigned int    n_q_points_f;
    ConstraintMatrix      constraints;
    LA::BlockSparseMatrix tangent_matrix;
    LA::MPI::BlockVector  system_rhs;
    LA::MPI::BlockVector  solution_n;
    struct Errors
    {
      Errors()
        :
        norm(1.0), u(1.0), p(1.0), J(1.0)
      {}
      void reset()
      {
        norm = 1.0;
        u = 1.0;
        p = 1.0;
        J = 1.0;
      }
      void normalise(const Errors &rhs)
      {
        if (rhs.norm != 0.0)
          norm /= rhs.norm;
        if (rhs.u != 0.0)
          u /= rhs.u;
        if (rhs.p != 0.0)
          p /= rhs.p;
        if (rhs.J != 0.0)
          J /= rhs.J;
      }
      double norm, u, p, J;
    };
    Errors error_residual, error_residual_0, error_residual_norm, error_update,
           error_update_0, error_update_norm;
    void
    get_error_residual(Errors &error_residual);
    void
    get_error_update(const LA::MPI::BlockVector &newton_update,
                     Errors &error_update);
    std::pair<double, std::pair<double,double> >
    get_error_dilation(const LA::MPI::BlockVector &solution_total) const;
    void
    print_conv_header();
    void
    print_conv_footer(const LA::MPI::BlockVector &solution_delta);
  };
  template <int dim>
  Solid<dim>::Solid(const std::string &input_file)
    :
    mpi_communicator(MPI_COMM_WORLD),
    n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
    this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
    pcout(std::cout, this_mpi_process == 0),
    parameters(input_file),
    triangulation(Triangulation<dim>::maximum_smoothing),
    time(parameters.end_time, parameters.delta_t),
    timer(mpi_communicator,
          pcout,
          TimerOutput::summary,
          TimerOutput::wall_times),
    degree(parameters.poly_degree),
    fe(FE_Q<dim>(parameters.poly_degree), dim, // displacement
       FE_DGPMonomial<dim>(parameters.poly_degree - 1), 1, // pressure
       FE_DGPMonomial<dim>(parameters.poly_degree - 1), 1), // dilatation
    dof_handler(triangulation),
    dofs_per_cell (fe.dofs_per_cell),
    u_fe(first_u_component),
    p_fe(p_component),
    J_fe(J_component),
    dofs_per_block(n_blocks),
    qf_cell(parameters.quad_order),
    qf_face(parameters.quad_order),
    n_q_points (qf_cell.size()),
    n_q_points_f (qf_face.size())
  {
    Assert(dim==2 || dim==3, ExcMessage("This problem only works in 2 or 3 space dimensions."));
    determine_component_extractors();
  }
  template <int dim>
  Solid<dim>::~Solid()
  {
    dof_handler.clear();
  }
  template <int dim>
  void Solid<dim>::run()
  {
    LA::MPI::BlockVector solution_delta;

    make_grid();
    setup_system(solution_delta);
    {
      ConstraintMatrix constraints;
      constraints.close();
      const ComponentSelectFunction<dim>
      J_mask (J_component, n_components);
      VectorTools::project (dof_handler,
                            constraints,
                            QGauss<dim>(degree+2),
                            J_mask,
                            solution_n);
    }
    output_results(time.get_timestep(), time.current());
    time.increment();

    // Some points for post-processing
    std::vector<double> real_time;
    real_time.push_back(0);
    std::vector<std::vector<Point<dim> > > tracked_vertices (4);
    {
      Point<dim> p;
      p[1] = parameters.length/2.0;
      tracked_vertices[0].push_back(p*parameters.scale);
    }
    {
      Point<dim> p;
      p[1] = parameters.hole_diameter/2.0;
      tracked_vertices[1].push_back(p*parameters.scale);
    }
    {
      Point<dim> p;
      p[0] = parameters.hole_diameter/2.0;
      tracked_vertices[2].push_back(p*parameters.scale);
    }
    {
      Point<dim> p;
      p[0] = parameters.width/2.0;
      tracked_vertices[3].push_back(p*parameters.scale);
    }

    while (time.current() < time.end()+0.01*time.get_delta_t())
      {
        solve_nonlinear_timestep(solution_delta);
        solution_n += solution_delta;
        solution_delta = 0.0;
        output_results(time.get_timestep(), time.current());
        compute_vertex_positions(real_time,
                                 tracked_vertices,
                                 get_solution_total(solution_delta));
        update_end_timestep();
        time.increment();
      }

    pcout << "\n\n*** Spatial position history for tracked vertices ***" << std::endl;
    for (unsigned int t=0; t<real_time.size(); ++t)
      {
        if (t == 0)
          {
            pcout << "Time,";
            for (unsigned int p=0; p<tracked_vertices.size(); ++p)
              {
                for (unsigned int d=0; d<dim; ++d)
                  {
                    pcout << "Point " << p << " [" << d << "]";
                    if (!(p == tracked_vertices.size()-1 && d == dim-1))
                      pcout << ",";
                  }
              }
            pcout << std::endl;
          }

        pcout << std::setprecision(6);
        pcout << real_time[t] << ",";
        for (unsigned int p=0; p<tracked_vertices.size(); ++p)
          {
            Assert(tracked_vertices[p].size() == real_time.size(),
                   ExcMessage("Vertex not tracked at each timestep"));
            for (unsigned int d=0; d<dim; ++d)
              {
                pcout << tracked_vertices[p][t][d];
                if (!(p == tracked_vertices.size()-1 && d == dim-1))
                  pcout << ",";
              }
          }
        pcout << std::endl;
      }
  }
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
  template <int dim>
  struct Solid<dim>::ScratchData_ASM
  {
    const LA::MPI::BlockVector &solution_total;

    // Integration helper
    FEValues<dim>              fe_values_ref;
    FEFaceValues<dim>          fe_face_values_ref;

    // Quadrature point solution
    std::vector<Tensor<2, dim> > solution_grads_u_total;
    std::vector<double>          solution_values_p_total;
    std::vector<double>          solution_values_J_total;

    // Shape function values and gradients
    std::vector<std::vector<double> >                   Nx;
    std::vector<std::vector<Tensor<2, dim> > >          grad_Nx;
    std::vector<std::vector<SymmetricTensor<2, dim> > > symm_grad_Nx;

    ScratchData_ASM(const FiniteElement<dim> &fe_cell,
                    const QGauss<dim> &qf_cell, const UpdateFlags uf_cell,
                    const QGauss<dim - 1> & qf_face, const UpdateFlags uf_face,
                    const LA::MPI::BlockVector &solution_total)
      :
      solution_total (solution_total),
      fe_values_ref(fe_cell, qf_cell, uf_cell),
      fe_face_values_ref(fe_cell, qf_face, uf_face),
      solution_grads_u_total(qf_cell.size()),
      solution_values_p_total(qf_cell.size()),
      solution_values_J_total(qf_cell.size()),
      Nx(qf_cell.size(),
         std::vector<double>(fe_cell.dofs_per_cell)),
      grad_Nx(qf_cell.size(),
              std::vector<Tensor<2, dim> >(fe_cell.dofs_per_cell)),
      symm_grad_Nx(qf_cell.size(),
                   std::vector<SymmetricTensor<2, dim> >
                   (fe_cell.dofs_per_cell))
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
      solution_grads_u_total(rhs.solution_grads_u_total),
      solution_values_p_total(rhs.solution_values_p_total),
      solution_values_J_total(rhs.solution_values_J_total),
      Nx(rhs.Nx),
      grad_Nx(rhs.grad_Nx),
      symm_grad_Nx(rhs.symm_grad_Nx)
    {}
    void reset()
    {
      const unsigned int n_q_points = solution_grads_u_total.size();
      const unsigned int n_dofs_per_cell = Nx[0].size();

      Assert(solution_grads_u_total.size() == n_q_points,
             ExcInternalError());
      Assert(solution_values_p_total.size() == n_q_points,
             ExcInternalError());
      Assert(solution_values_J_total.size() == n_q_points,
             ExcInternalError());
      Assert(Nx.size() == n_q_points,
             ExcInternalError());
      Assert(grad_Nx.size() == n_q_points,
             ExcInternalError());
      Assert(symm_grad_Nx.size() == n_q_points,
             ExcInternalError());

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          Assert( Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());
          Assert( grad_Nx[q_point].size() == n_dofs_per_cell,
                  ExcInternalError());
          Assert( symm_grad_Nx[q_point].size() == n_dofs_per_cell,
                  ExcInternalError());

          solution_grads_u_total[q_point] = 0.0;
          solution_values_p_total[q_point] = 0.0;
          solution_values_J_total[q_point] = 0.0;
          for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
            {
              Nx[q_point][k] = 0.0;
              grad_Nx[q_point][k] = 0.0;
              symm_grad_Nx[q_point][k] = 0.0;
            }
        }
    }
  };
  template <>
  void Solid<2>::make_grid()
  {
    const int dim = 2;
    const double tol = 1e-12;
    make_2d_quarter_plate_with_hole(triangulation,
                                    parameters.length/2.0,
                                    parameters.width/2.0,
                                    parameters.hole_diameter/2.0,
                                    parameters.n_repetitions_xy,
                                    parameters.hole_division_fraction);

    // Clear boundary ID's
    for (typename Triangulation<dim>::active_cell_iterator
         cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell)
      {
        for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
          if (cell->face(face)->at_boundary())
            {
              cell->face(face)->set_all_boundary_ids(0);
            }
      }

    // Set boundary IDs and and manifolds
    const Point<dim> centre (0,0);
    for (typename Triangulation<dim>::active_cell_iterator
         cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell)
      {
        for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
          if (cell->face(face)->at_boundary())
            {
              // Set boundary IDs
              if (std::abs(cell->face(face)->center()[0] - 0.0) < tol)
                {
                  cell->face(face)->set_boundary_id(parameters.boundary_id_minus_X);
                }
              else if (std::abs(cell->face(face)->center()[0] - parameters.width/2.0) < tol)
                {
                  cell->face(face)->set_boundary_id(parameters.boundary_id_plus_X);
                }
              else if (std::abs(cell->face(face)->center()[1] - 0.0) < tol)
                {
                  cell->face(face)->set_boundary_id(parameters.boundary_id_minus_Y);
                }
              else if (std::abs(cell->face(face)->center()[1] - parameters.length/2.0) < tol)
                {
                  cell->face(face)->set_boundary_id(parameters.boundary_id_plus_Y);
                }
              else
                {
                  for (unsigned int vertex=0; vertex<GeometryInfo<dim>::vertices_per_face; ++vertex)
                    if (std::abs(cell->vertex(vertex).distance(centre) - parameters.hole_diameter/2.0) < tol)
                      {
                        cell->face(face)->set_boundary_id(parameters.boundary_id_hole);
                        break;
                      }
                }

              // Set manifold IDs
              for (unsigned int vertex=0; vertex<GeometryInfo<dim>::vertices_per_face; ++vertex)
                if (std::abs(cell->vertex(vertex).distance(centre) - parameters.hole_diameter/2.0) < tol)
                  {
                    cell->face(face)->set_manifold_id(parameters.manifold_id_hole);
                    break;
                  }
            }
      }
    static SphericalManifold<dim> spherical_manifold (centre);
    triangulation.set_manifold(parameters.manifold_id_hole,spherical_manifold);
    triangulation.refine_global(parameters.global_refinement);
    GridTools::scale(parameters.scale,triangulation);
  }
  template <>
  void Solid<3>::make_grid()
  {
    const int dim = 3;
    const double tol = 1e-12;
    Triangulation<2> tria_2d;
    make_2d_quarter_plate_with_hole(tria_2d,
                                    parameters.length/2.0,
                                    parameters.width/2.0,
                                    parameters.hole_diameter/2.0,
                                    parameters.n_repetitions_xy,
                                    parameters.hole_division_fraction);
    GridGenerator::extrude_triangulation(tria_2d,
                                         parameters.n_repetitions_z+1,
                                         parameters.thickness/2.0,
                                         triangulation);

    // Clear boundary ID's
    for (typename Triangulation<dim>::active_cell_iterator
         cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell)
      {
        for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
          if (cell->face(face)->at_boundary())
            {
              cell->face(face)->set_all_boundary_ids(0);
            }
      }

    // Set boundary IDs and and manifolds
    const Point<dim> direction (0,0,1);
    const Point<dim> centre (0,0,0);
    for (typename Triangulation<dim>::active_cell_iterator
         cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell)
      {
        for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
          if (cell->face(face)->at_boundary())
            {
              // Set boundary IDs
              if (std::abs(cell->face(face)->center()[0] - 0.0) < tol)
                {
                  cell->face(face)->set_boundary_id(parameters.boundary_id_minus_X);
                }
              else if (std::abs(cell->face(face)->center()[0] - parameters.width/2.0) < tol)
                {
                  cell->face(face)->set_boundary_id(parameters.boundary_id_plus_X);
                }
              else if (std::abs(cell->face(face)->center()[1] - 0.0) < tol)
                {
                  cell->face(face)->set_boundary_id(parameters.boundary_id_minus_Y);
                }
              else if (std::abs(cell->face(face)->center()[1] - parameters.length/2.0) < tol)
                {
                  cell->face(face)->set_boundary_id(parameters.boundary_id_plus_Y);
                }
              else if (std::abs(cell->face(face)->center()[2] - 0.0) < tol)
                {
                  cell->face(face)->set_boundary_id(parameters.boundary_id_minus_Z);
                }
              else if (std::abs(cell->face(face)->center()[2] - parameters.thickness/2.0) < tol)
                {
                  cell->face(face)->set_boundary_id(parameters.boundary_id_plus_Z);
                }
              else
                {
                  for (unsigned int vertex=0; vertex<GeometryInfo<dim>::vertices_per_face; ++vertex)
                    {
                      // Project the cell vertex to the XY plane and
                      // test the distance from the cylinder axis
                      Point<dim> vertex_proj = cell->vertex(vertex);
                      vertex_proj[2] = 0.0;
                      if (std::abs(vertex_proj.distance(centre) - parameters.hole_diameter/2.0) < tol)
                        {
                          cell->face(face)->set_boundary_id(parameters.boundary_id_hole);
                          break;
                        }
                    }
                }

              // Set manifold IDs
              for (unsigned int vertex=0; vertex<GeometryInfo<dim>::vertices_per_face; ++vertex)
                {
                  // Project the cell vertex to the XY plane and
                  // test the distance from the cylinder axis
                  Point<dim> vertex_proj = cell->vertex(vertex);
                  vertex_proj[2] = 0.0;
                  if (std::abs(vertex_proj.distance(centre) - parameters.hole_diameter/2.0) < 1e-12)
                    {
                      // Set manifold ID on face and edges
                      cell->face(face)->set_all_manifold_ids(parameters.manifold_id_hole);
                      break;
                    }
                }
            }
      }
    static CylindricalManifold<dim> cylindrical_manifold (direction,centre);
    triangulation.set_manifold(parameters.manifold_id_hole,cylindrical_manifold);
    triangulation.refine_global(parameters.global_refinement);
    GridTools::scale(parameters.scale,triangulation);
  }
  template <int dim>
  void Solid<dim>::make_2d_quarter_plate_with_hole(Triangulation<2> &tria_2d,
                                                   const double half_length,
                                                   const double half_width,
                                                   const double hole_radius,
                                                   const unsigned int n_repetitions_xy,
                                                   const double hole_division_fraction)
  {
    const double length = 2.0*half_length;
    const double width = 2.0*half_width;
    const double hole_diameter = 2.0*hole_radius;

    const double internal_width = hole_diameter + hole_division_fraction*(width - hole_diameter);
    Triangulation<2> tria_quarter_plate_hole;
    {
      Triangulation<2> tria_plate_hole;
      GridGenerator::hyper_cube_with_cylindrical_hole (tria_plate_hole,
                                                       hole_diameter/2.0,
                                                       internal_width/2.0);

      std::set<typename Triangulation<2>::active_cell_iterator > cells_to_remove;
      for (typename Triangulation<2>::active_cell_iterator
           cell = tria_plate_hole.begin_active();
           cell != tria_plate_hole.end(); ++cell)
        {
          // Remove all cells that are not in the first quadrant
          if (cell->center()[0] < 0.0 || cell->center()[1] < 0.0)
            cells_to_remove.insert(cell);
        }
      Assert(cells_to_remove.size() > 0, ExcInternalError());
      Assert(cells_to_remove.size() != tria_plate_hole.n_active_cells(), ExcInternalError());
      GridGenerator::create_triangulation_with_removed_cells(tria_plate_hole,cells_to_remove,tria_quarter_plate_hole);
    }

    Triangulation<2> tria_cut_plate;
    {
      Triangulation<2> tria_plate;
      // Subdivide the plate so that we're left one
      // cell to remove (we'll replace this with the
      // plate with the hole) and then make the
      // rest of the subdivisions so that we're left
      // with cells with a decent aspect ratio
      std::vector<std::vector<double> > step_sizes;
      {
        std::vector<double> subdivision_width;
        subdivision_width.push_back(internal_width/2.0);
        const double width_remaining = (width - internal_width)/2.0;
        const unsigned int n_subs = std::max(1.0,std::ceil(width_remaining/(internal_width/2.0)));
        Assert(n_subs>0, ExcInternalError());
        for (unsigned int s=0; s<n_subs; ++s)
          subdivision_width.push_back(width_remaining/n_subs);
        step_sizes.push_back(subdivision_width);

        const double sum_half_width = std::accumulate(subdivision_width.begin(), subdivision_width.end(), 0.0);
        Assert(std::abs(sum_half_width-width/2.0) < 1e-12, ExcInternalError());
      }
      {
        std::vector<double> subdivision_length;
        subdivision_length.push_back(internal_width/2.0);
        const double length_remaining = (length - internal_width)/2.0;
        const unsigned int n_subs = std::max(1.0,std::ceil(length_remaining/(internal_width/2.0)));
        Assert(n_subs>0, ExcInternalError());
        for (unsigned int s=0; s<n_subs; ++s)
          subdivision_length.push_back(length_remaining/n_subs);
        step_sizes.push_back(subdivision_length);

        const double sum_half_length = std::accumulate(subdivision_length.begin(), subdivision_length.end(), 0.0);
        Assert(std::abs(sum_half_length-length/2.0) < 1e-12, ExcInternalError());
      }

      GridGenerator::subdivided_hyper_rectangle(tria_plate,
                                                step_sizes,
                                                Point<2>(0.0, 0.0),
                                                Point<2>(width/2.0, length/2.0));

      std::set<typename Triangulation<2>::active_cell_iterator > cells_to_remove;
      for (typename Triangulation<2>::active_cell_iterator
           cell = tria_plate.begin_active();
           cell != tria_plate.end(); ++cell)
        {
          // Remove all cells that are in the first quadrant
          if (cell->center()[0] < internal_width/2.0 && cell->center()[1] < internal_width/2.0)
            cells_to_remove.insert(cell);
        }
      Assert(cells_to_remove.size() > 0, ExcInternalError());
      Assert(cells_to_remove.size() != tria_plate.n_active_cells(), ExcInternalError());
      GridGenerator::create_triangulation_with_removed_cells(tria_plate,cells_to_remove,tria_cut_plate);
    }

    Triangulation<2> tria_2d_not_flat;
    GridGenerator::merge_triangulations(tria_quarter_plate_hole,
                                        tria_cut_plate,
                                        tria_2d_not_flat);

    // Attach a manifold to the curved boundary and refine
    // Note: We can only guarentee that the vertices sit on
    // the curve, so we must test with their position instead
    // of the cell centre.
    const Point<2> centre_2d (0,0);
    for (typename Triangulation<2>::active_cell_iterator
         cell = tria_2d_not_flat.begin_active();
         cell != tria_2d_not_flat.end(); ++cell)
      {
        for (unsigned int face=0; face<GeometryInfo<2>::faces_per_cell; ++face)
          if (cell->face(face)->at_boundary())
            for (unsigned int vertex=0; vertex<GeometryInfo<2>::vertices_per_face; ++vertex)
              if (std::abs(cell->vertex(vertex).distance(centre_2d) - hole_diameter/2.0) < 1e-12)
                {
                  cell->face(face)->set_manifold_id(10);
                  break;
                }
      }
    SphericalManifold<2> spherical_manifold_2d (centre_2d);
    tria_2d_not_flat.set_manifold(10,spherical_manifold_2d);
    tria_2d_not_flat.refine_global(std::max (1U, n_repetitions_xy));
    tria_2d_not_flat.reset_manifold(10); // Clear manifold

    GridGenerator::flatten_triangulation(tria_2d_not_flat,tria_2d);
  }
  template <int dim>
  void Solid<dim>::setup_system(LA::MPI::BlockVector &solution_delta)
  {
    timer.enter_subsection("Setup system");
    pcout << "Setting up linear system..." << std::endl;

    // Partition triangulation
    GridTools::partition_triangulation (n_mpi_processes,
                                        triangulation);

    block_component = std::vector<unsigned int> (n_components, u_block); // Displacement
    block_component[p_component] = p_block; // Pressure
    block_component[J_component] = J_block; // Dilatation
    dof_handler.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(dof_handler);
    DoFRenumbering::component_wise(dof_handler, block_component);

    // Count DoFs in each block
    dofs_per_block.clear();
    dofs_per_block.resize(n_blocks);
    DoFTools::count_dofs_per_block(dof_handler, dofs_per_block,
                                   block_component);

    all_locally_owned_dofs = DoFTools::locally_owned_dofs_per_subdomain (dof_handler);
    std::vector<IndexSet> all_locally_relevant_dofs
      = DoFTools::locally_relevant_dofs_per_subdomain (dof_handler);

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

    pcout
        << "  Number of active cells: " << triangulation.n_active_cells()
        << " (by partition:";
    for (unsigned int p=0; p<n_mpi_processes; ++p)
      pcout
          << (p==0 ? ' ' : '+')
          << (GridTools::count_cells_with_subdomain_association (triangulation,p));
    pcout << ")" << std::endl;

    pcout
        << "  Number of degrees of freedom: " << dof_handler.n_dofs()
        << " (by partition:";
    for (unsigned int p=0; p<n_mpi_processes; ++p)
      pcout
          << (p==0 ? ' ' : '+')
          << (DoFTools::count_dofs_with_subdomain_association (dof_handler,p));
    pcout << ")" << std::endl;
    pcout
        << "  Number of degrees of freedom per block: "
        << "[n_u, n_p, n_J] = ["
        << dofs_per_block[u_block] << ", "
        << dofs_per_block[p_block] << ", "
        << dofs_per_block[J_block] << "]"
        << std::endl;


    Table<2, DoFTools::Coupling> coupling(n_components, n_components);
    for (unsigned int ii = 0; ii < n_components; ++ii)
      for (unsigned int jj = 0; jj < n_components; ++jj)
        if (((ii < p_component) && (jj == J_component))
            || ((ii == J_component) && (jj < p_component))
            || ((ii == p_component) && (jj == p_component)))
          coupling[ii][jj] = DoFTools::none;
        else
          coupling[ii][jj] = DoFTools::always;

    TrilinosWrappers::BlockSparsityPattern bsp (locally_owned_partitioning,
                                                locally_owned_partitioning,
                                                locally_relevant_partitioning,
                                                mpi_communicator);
    DoFTools::make_sparsity_pattern (dof_handler, bsp,
                                     constraints, false,
                                     this_mpi_process);
    bsp.compress();
    tangent_matrix.reinit (bsp);

    // We then set up storage vectors
    system_rhs.reinit(locally_owned_partitioning,
                      mpi_communicator);
    solution_n.reinit(locally_owned_partitioning,
                      mpi_communicator);
    solution_delta.reinit(locally_owned_partitioning,
                          mpi_communicator);
    setup_qph();
    timer.leave_subsection();
  }
  template <int dim>
  void
  Solid<dim>::determine_component_extractors()
  {
    element_indices_u.clear();
    element_indices_p.clear();
    element_indices_J.clear();
    for (unsigned int k = 0; k < fe.dofs_per_cell; ++k)
      {
        const unsigned int k_group = fe.system_to_base_index(k).first.first;
        if (k_group == u_block)
          element_indices_u.push_back(k);
        else if (k_group == p_block)
          element_indices_p.push_back(k);
        else if (k_group == J_block)
          element_indices_J.push_back(k);
        else
          {
            Assert(k_group <= J_block, ExcInternalError());
          }
      }
  }
  template <int dim>
  void Solid<dim>::setup_qph()
  {
    pcout << "Setting up quadrature point data..." << std::endl;
    quadrature_point_history.initialize(triangulation.begin_active(),
                                        triangulation.end(),
                                        n_q_points);
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    cell (IteratorFilters::SubdomainEqualTo(this_mpi_process),
          dof_handler.begin_active()),
                                   endc (IteratorFilters::SubdomainEqualTo(this_mpi_process),
                                         dof_handler.end());
    for (; cell!=endc; ++cell)
      {
        Assert(cell->subdomain_id()==this_mpi_process, ExcInternalError());
        const std::vector<std::shared_ptr<PointHistory<dim> > > lqph =
          quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          lqph[q_point]->setup_lqp(parameters, time);
      }
  }
  template <int dim>
  void
  Solid<dim>::solve_nonlinear_timestep(LA::MPI::BlockVector &solution_delta)
  {
    pcout << std::endl
          << "Timestep " << time.get_timestep() << " @ "
          << time.current() << "s of "
          << time.end() << "s" << std::endl;
    LA::MPI::BlockVector newton_update(locally_owned_partitioning,
                                       mpi_communicator);
    error_residual.reset();
    error_residual_0.reset();
    error_residual_norm.reset();
    error_update.reset();
    error_update_0.reset();
    error_update_norm.reset();
    print_conv_header();
    unsigned int newton_iteration = 0;
    for (; newton_iteration < parameters.max_iterations_NR;
         ++newton_iteration)
      {
        pcout << " " << std::setw(2) << newton_iteration << " " << std::flush;
        make_constraints(newton_iteration);
        assemble_system(solution_delta);
        get_error_residual(error_residual);
        if (newton_iteration == 0)
          error_residual_0 = error_residual;
        error_residual_norm = error_residual;
        error_residual_norm.normalise(error_residual_0);
        if (newton_iteration > 0 &&
            (error_update_norm.u <= parameters.tol_u &&
             error_residual_norm.u <= parameters.tol_f) )
          {
            pcout << " CONVERGED! " << std::endl;
            print_conv_footer(solution_delta);
            break;
          }
        const std::pair<unsigned int, double>
        lin_solver_output = solve_linear_system(newton_update);
        get_error_update(newton_update, error_update);
        if (newton_iteration == 0)
          error_update_0 = error_update;
        error_update_norm = error_update;
        error_update_norm.normalise(error_update_0);
        solution_delta += newton_update;
        newton_update = 0.0;
        pcout << " | " << std::fixed << std::setprecision(3) << std::setw(7)
              << std::scientific << lin_solver_output.first << "  "
              << lin_solver_output.second << "  " << error_residual_norm.norm
              << "  " << error_residual_norm.u << "  "
              << error_residual_norm.p << "  " << error_residual_norm.J
              << "  " << error_update_norm.norm << "  " << error_update_norm.u
              << "  " << error_update_norm.p << "  " << error_update_norm.J
              << "  " << std::endl;
      }
    AssertThrow (newton_iteration <= parameters.max_iterations_NR,
                 ExcMessage("No convergence in nonlinear solver!"));
  }
  template <int dim>
  void Solid<dim>::print_conv_header()
  {
    pcout << std::string(132,'_') << std::endl;
    pcout << "     SOLVER STEP       "
          << " |  LIN_IT   LIN_RES    RES_NORM    "
          << " RES_U     RES_P      RES_J     NU_NORM     "
          << " NU_U       NU_P       NU_J " << std::endl;
    pcout << std::string(132,'_') << std::endl;
  }
  template <int dim>
  void Solid<dim>::print_conv_footer(const LA::MPI::BlockVector &solution_delta)
  {
    pcout << std::string(132,'_') << std::endl;
    const std::pair<double,std::pair<double,double> > error_dil = get_error_dilation(get_solution_total(solution_delta));
    pcout << "Relative errors:" << std::endl
          << "Displacement:\t" << error_update.u / error_update_0.u << std::endl
          << "Force: \t\t" << error_residual.u / error_residual_0.u << std::endl
          << "Dilatation:\t" << error_dil.first << std::endl
          << "v / V_0:\t" << error_dil.second.second << " / " << error_dil.second.first
          << " = " << (error_dil.second.second/error_dil.second.first) << std::endl;
  }
  template <int dim>
  std::pair<double,std::pair<double,double> >
  Solid<dim>::get_error_dilation(const LA::MPI::BlockVector &solution_total) const
  {
    double vol_reference = 0.0;
    double vol_current = 0.0;
    double dil_L2_error = 0.0;
    FEValues<dim> fe_values_ref(fe, qf_cell,
                                update_values | update_gradients | update_JxW_values);
    std::vector<Tensor<2, dim> > solution_grads_u_total (qf_cell.size());
    std::vector<double>          solution_values_J_total (qf_cell.size());
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    cell (IteratorFilters::SubdomainEqualTo(this_mpi_process),
          dof_handler.begin_active()),
                                   endc (IteratorFilters::SubdomainEqualTo(this_mpi_process),
                                         dof_handler.end());
    for (; cell != endc; ++cell)
      {
        Assert(cell->subdomain_id()==this_mpi_process, ExcInternalError());
        fe_values_ref.reinit(cell);
        fe_values_ref[u_fe].get_function_gradients(solution_total,
                                                   solution_grads_u_total);
        fe_values_ref[J_fe].get_function_values(solution_total,
                                                solution_values_J_total);
        const std::vector<std::shared_ptr<const PointHistory<dim> > > lqph =
          quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            const double det_F_qp = determinant(Physics::Elasticity::Kinematics::F(solution_grads_u_total[q_point]));
            const double J_tilde_qp = solution_values_J_total[q_point];
            const double the_error_qp_squared = std::pow((det_F_qp - J_tilde_qp),
                                                         2);
            const double JxW = fe_values_ref.JxW(q_point);
            dil_L2_error  += the_error_qp_squared * JxW;
            vol_reference += JxW;
            vol_current   += det_F_qp * JxW;
          }
      }
    Assert(vol_current > 0.0, ExcInternalError());
    // Sum across all porcessors
    dil_L2_error  = Utilities::MPI::sum(dil_L2_error,mpi_communicator);
    vol_reference = Utilities::MPI::sum(vol_reference,mpi_communicator);
    vol_current   = Utilities::MPI::sum(vol_current,mpi_communicator);

    return std::make_pair(std::sqrt(dil_L2_error),
                          std::make_pair(vol_reference,vol_current));
  }
  template <int dim>
  void Solid<dim>::get_error_residual(Errors &error_residual)
  {
    // Construct a residual vector that has the values for all of its
    // constrained DoFs set to zero.
    LA::MPI::BlockVector error_res (system_rhs);
    constraints.set_zero(error_res);
    error_residual.norm = error_res.l2_norm();
    error_residual.u = error_res.block(u_block).l2_norm();
    error_residual.p = error_res.block(p_block).l2_norm();
    error_residual.J = error_res.block(J_block).l2_norm();
  }
  template <int dim>
  void Solid<dim>::get_error_update(const LA::MPI::BlockVector &newton_update,
                                    Errors &error_update)
  {
    // Construct a update vector that has the values for all of its
    // constrained DoFs set to zero.
    LA::MPI::BlockVector error_ud (newton_update);
    constraints.set_zero(error_ud);
    error_update.norm = error_ud.l2_norm();
    error_update.u = error_ud.block(u_block).l2_norm();
    error_update.p = error_ud.block(p_block).l2_norm();
    error_update.J = error_ud.block(J_block).l2_norm();
  }
  template <int dim>
  LA::MPI::BlockVector
  Solid<dim>::get_solution_total(const LA::MPI::BlockVector &solution_delta) const
  {
    // Cell interpolation -> Ghosted vector
    LA::MPI::BlockVector solution_total (locally_owned_partitioning,
                                         locally_relevant_partitioning,
                                         mpi_communicator,
                                         /*vector_writable = */ false);
    LA::MPI::BlockVector tmp (solution_total);
    solution_total = solution_n;
    tmp = solution_delta;
    solution_total += tmp;
    return solution_total;
  }
  template <int dim>
  void Solid<dim>::assemble_system(const LA::MPI::BlockVector &solution_delta)
  {
    timer.enter_subsection("Assemble system");
    pcout << " ASM_SYS " << std::flush;
    tangent_matrix = 0.0;
    system_rhs = 0.0;
    const LA::MPI::BlockVector solution_total(get_solution_total(solution_delta));
    const UpdateFlags uf_cell(update_values |
                              update_gradients |
                              update_JxW_values);
    const UpdateFlags uf_face(update_values |
                              update_normal_vectors |
                              update_JxW_values);
    PerTaskData_ASM per_task_data(dofs_per_cell);
    ScratchData_ASM scratch_data(fe, qf_cell, uf_cell, qf_face, uf_face, solution_total);

    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    cell (IteratorFilters::SubdomainEqualTo(this_mpi_process),
          dof_handler.begin_active()),
                                   endc (IteratorFilters::SubdomainEqualTo(this_mpi_process),
                                         dof_handler.end());
    for (; cell != endc; ++cell)
      {
        Assert(cell->subdomain_id()==this_mpi_process, ExcInternalError());
        assemble_system_one_cell(cell, scratch_data, per_task_data);
        copy_local_to_global_system(per_task_data);
      }
    tangent_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
    timer.leave_subsection();
  }
  template <int dim>
  void Solid<dim>::copy_local_to_global_system(const PerTaskData_ASM &data)
  {
    constraints.distribute_local_to_global(data.cell_matrix, data.cell_rhs,
                                           data.local_dof_indices,
                                           tangent_matrix, system_rhs);
  }
  template <int dim>
  void
  Solid<dim>::assemble_system_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                       ScratchData_ASM &scratch,
                                       PerTaskData_ASM &data) const
  {
    Assert(cell->subdomain_id()==this_mpi_process, ExcInternalError());

    data.reset();
    scratch.reset();
    scratch.fe_values_ref.reinit(cell);
    cell->get_dof_indices(data.local_dof_indices);
    const std::vector<std::shared_ptr<const PointHistory<dim> > > lqph =
      quadrature_point_history.get_data(cell);
    Assert(lqph.size() == n_q_points, ExcInternalError());

    // Update quadrature point solution
    scratch.fe_values_ref[u_fe].get_function_gradients(scratch.solution_total,
                                                       scratch.solution_grads_u_total);
    scratch.fe_values_ref[p_fe].get_function_values(scratch.solution_total,
                                                    scratch.solution_values_p_total);
    scratch.fe_values_ref[J_fe].get_function_values(scratch.solution_total,
                                                    scratch.solution_values_J_total);

    // Update shape functions and their gradients (push-forward)
    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        const Tensor<2, dim> F = Physics::Elasticity::Kinematics::F(scratch.solution_grads_u_total[q_point]);
        const Tensor<2, dim> F_inv = invert(F);

        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            const unsigned int k_group = fe.system_to_base_index(k).first.first;
            if (k_group == u_block)
              {
                scratch.grad_Nx[q_point][k] = scratch.fe_values_ref[u_fe].gradient(k, q_point)
                                              * F_inv;
                scratch.symm_grad_Nx[q_point][k] = symmetrize(scratch.grad_Nx[q_point][k]);
              }
            else if (k_group == p_block)
              scratch.Nx[q_point][k] = scratch.fe_values_ref[p_fe].value(k,
                                                                         q_point);
            else if (k_group == J_block)
              scratch.Nx[q_point][k] = scratch.fe_values_ref[J_fe].value(k,
                                                                         q_point);
            else
              Assert(k_group <= J_block, ExcInternalError());
          }
      }
    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        const SymmetricTensor<2, dim> &I = Physics::Elasticity::StandardTensors<dim>::I;
        const Tensor<2, dim> F = Physics::Elasticity::Kinematics::F(scratch.solution_grads_u_total[q_point]);
        const double det_F = determinant(F);
        const double &p_tilde  = scratch.solution_values_p_total[q_point];
        const double &J_tilde  = scratch.solution_values_J_total[q_point];
        Assert(det_F > 0, ExcInternalError());

        {
          PointHistory<dim> *lqph_q_point_nc =  const_cast<PointHistory<dim>*>(lqph[q_point].get());
          lqph_q_point_nc->update_internal_equilibrium(F,p_tilde,J_tilde);
        }

        const SymmetricTensor<2, dim> tau = lqph[q_point]->get_tau(F,p_tilde);
        const Tensor<2, dim> tau_ns (tau);
        const SymmetricTensor<4, dim> Jc = lqph[q_point]->get_Jc(F,p_tilde);
        const double dPsi_vol_dJ = lqph[q_point]->get_dPsi_vol_dJ(J_tilde);
        const double d2Psi_vol_dJ2 = lqph[q_point]->get_d2Psi_vol_dJ2(J_tilde);

        const std::vector<double> &Nx = scratch.Nx[q_point];
        const std::vector<Tensor<2, dim> > &grad_Nx = scratch.grad_Nx[q_point];
        const std::vector<SymmetricTensor<2, dim> > &symm_grad_Nx = scratch.symm_grad_Nx[q_point];
        const double JxW = scratch.fe_values_ref.JxW(q_point);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            const unsigned int component_i = fe.system_to_component_index(i).first;
            const unsigned int i_group = fe.system_to_base_index(i).first.first;
            if (i_group == u_block)
              data.cell_rhs(i) -= (symm_grad_Nx[i] * tau) * JxW;
            else if (i_group == p_block)
              data.cell_rhs(i) -= Nx[i] * (det_F - J_tilde) * JxW;
            else if (i_group == J_block)
              data.cell_rhs(i) -= Nx[i] * (dPsi_vol_dJ - p_tilde) * JxW;
            else
              Assert(i_group <= J_block, ExcInternalError());

            for (unsigned int j = 0; j <= i; ++j)
              {
                const unsigned int component_j = fe.system_to_component_index(j).first;
                const unsigned int j_group     = fe.system_to_base_index(j).first.first;
                if ((i_group == u_block) && (j_group == u_block))
                  {
                    data.cell_matrix(i, j) += symm_grad_Nx[i] * Jc // The material contribution:
                                              * symm_grad_Nx[j] * JxW;
                    if (component_i == component_j) // geometrical stress contribution
                      data.cell_matrix(i, j) += grad_Nx[i][component_i] * tau_ns
                                                * grad_Nx[j][component_j] * JxW;
                  }
                else if ((i_group == u_block) && (j_group == p_block))
                  {
                    data.cell_matrix(i, j) += (symm_grad_Nx[i] * I)
                                              * Nx[j] * det_F
                                              * JxW;
                  }
                else if ((i_group == p_block) && (j_group == u_block))
                  {
                    data.cell_matrix(i, j) += Nx[i] * det_F
                                              * (symm_grad_Nx[j] * I)
                                              * JxW;
                  }
                else if ((i_group == p_block) && (j_group == J_block))
                  data.cell_matrix(i, j) -= Nx[i] * Nx[j] * JxW;
                else if ((i_group == J_block) && (j_group == p_block))
                  data.cell_matrix(i, j) -= Nx[i] * Nx[j] * JxW;
                else if ((i_group == J_block) && (j_group == J_block))
                  data.cell_matrix(i, j) += Nx[i] * d2Psi_vol_dJ2 * Nx[j] * JxW;
                else
                  Assert((i_group <= J_block) && (j_group <= J_block),
                         ExcInternalError());
              }
          }
      }

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
        data.cell_matrix(i, j) = data.cell_matrix(j, i);

    if (parameters.driver == "Neumann")
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           ++face)
        if (cell->face(face)->at_boundary() == true
            && cell->face(face)->boundary_id() == parameters.boundary_id_plus_Y)
          {
            scratch.fe_face_values_ref.reinit(cell, face);
            for (unsigned int f_q_point = 0; f_q_point < n_q_points_f;
                 ++f_q_point)
              {
                const Tensor<1, dim> &N =
                  scratch.fe_face_values_ref.normal_vector(f_q_point);
                static const double  pressure_nom  = parameters.pressure
                                                     / (parameters.scale * parameters.scale);
                const double         time_ramp = (time.current() < parameters.load_time ?
                                                  time.current() / parameters.load_time : 1.0);
                const double         pressure  = -pressure_nom * time_ramp;
                const Tensor<1, dim> traction  = pressure * N;
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    const unsigned int i_group =
                      fe.system_to_base_index(i).first.first;
                    if (i_group == u_block)
                      {
                        const unsigned int component_i =
                          fe.system_to_component_index(i).first;
                        const double Ni =
                          scratch.fe_face_values_ref.shape_value(i,
                                                                 f_q_point);
                        const double JxW = scratch.fe_face_values_ref.JxW(
                                             f_q_point);
                        data.cell_rhs(i) += (Ni * traction[component_i])
                                            * JxW;
                      }
                  }
              }
          }
  }
  template <int dim>
  void Solid<dim>::make_constraints(const int &it_nr)
  {
    pcout << " CST " << std::flush;
    if (it_nr > 1)
      return;
    constraints.clear();
    const bool apply_dirichlet_bc = (it_nr == 0);
    const FEValuesExtractors::Scalar x_displacement(0);
    const FEValuesExtractors::Scalar y_displacement(1);
    {
      const int boundary_id = parameters.boundary_id_minus_X;
      if (apply_dirichlet_bc == true)
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 boundary_id,
                                                 ZeroFunction<dim>(n_components),
                                                 constraints,
                                                 fe.component_mask(x_displacement));
      else
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 boundary_id,
                                                 ZeroFunction<dim>(n_components),
                                                 constraints,
                                                 fe.component_mask(x_displacement));
    }
    {
      const int boundary_id = parameters.boundary_id_minus_Y;
      if (apply_dirichlet_bc == true)
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 boundary_id,
                                                 ZeroFunction<dim>(n_components),
                                                 constraints,
                                                 fe.component_mask(y_displacement));
      else
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 boundary_id,
                                                 ZeroFunction<dim>(n_components),
                                                 constraints,
                                                 fe.component_mask(y_displacement));
    }
    if (dim==3)
      {
        const FEValuesExtractors::Scalar z_displacement(2);
        {
          const int boundary_id = parameters.boundary_id_minus_Z;
          if (apply_dirichlet_bc == true)
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     boundary_id,
                                                     ZeroFunction<dim>(n_components),
                                                     constraints,
                                                     fe.component_mask(z_displacement));
          else
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     boundary_id,
                                                     ZeroFunction<dim>(n_components),
                                                     constraints,
                                                     fe.component_mask(z_displacement));
        }
        {
          const int boundary_id = parameters.boundary_id_plus_Z;
          if (apply_dirichlet_bc == true)
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     boundary_id,
                                                     ZeroFunction<dim>(n_components),
                                                     constraints,
                                                     fe.component_mask(z_displacement));
          else
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     boundary_id,
                                                     ZeroFunction<dim>(n_components),
                                                     constraints,
                                                     fe.component_mask(z_displacement));
        }
      }
    if (parameters.driver == "Dirichlet")
      {
        const int boundary_id = parameters.boundary_id_plus_Y;
        if (apply_dirichlet_bc == true)
          {

            if (time.current() < parameters.load_time+0.01*time.get_delta_t())
              {
                const double delta_length = parameters.length*(parameters.stretch - 1.0)*parameters.scale;
                const unsigned int n_stretch_steps = parameters.load_time/time.get_delta_t();
                const double delta_u_y = delta_length/2.0/n_stretch_steps;
                VectorTools::interpolate_boundary_values(dof_handler,
                                                         boundary_id,
                                                         ConstantFunction<dim>(delta_u_y,n_components),
                                                         constraints,
                                                         fe.component_mask(y_displacement));
              }
            else
              VectorTools::interpolate_boundary_values(dof_handler,
                                                       boundary_id,
                                                       ZeroFunction<dim>(n_components),
                                                       constraints,
                                                       fe.component_mask(y_displacement));
          }
        else
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   boundary_id,
                                                   ZeroFunction<dim>(n_components),
                                                   constraints,
                                                   fe.component_mask(y_displacement));
      }
    constraints.close();
  }
  template <int dim>
  std::pair<unsigned int, double>
  Solid<dim>::solve_linear_system(LA::MPI::BlockVector &newton_update)
  {
    unsigned int lin_it = 0;
    double lin_res = 0.0;

    timer.enter_subsection("Linear solver");
    pcout << " SLV " << std::flush;

    const LA::MPI::Vector &f_u = system_rhs.block(u_block);
    const LA::MPI::Vector &f_p = system_rhs.block(p_block);
    const LA::MPI::Vector &f_J = system_rhs.block(J_block);
    LA::MPI::Vector &d_u = newton_update.block(u_block);
    LA::MPI::Vector &d_p = newton_update.block(p_block);
    LA::MPI::Vector &d_J = newton_update.block(J_block);
    const auto K_uu = linear_operator<LA::MPI::Vector>(tangent_matrix.block(u_block, u_block));
    const auto K_up = linear_operator<LA::MPI::Vector>(tangent_matrix.block(u_block, p_block));
    const auto K_pu = linear_operator<LA::MPI::Vector>(tangent_matrix.block(p_block, u_block));
    const auto K_Jp = linear_operator<LA::MPI::Vector>(tangent_matrix.block(J_block, p_block));
    const auto K_JJ = linear_operator<LA::MPI::Vector>(tangent_matrix.block(J_block, J_block));

    LA::PreconditionJacobi preconditioner_K_Jp_inv;
    preconditioner_K_Jp_inv.initialize(
      tangent_matrix.block(J_block, p_block),
      LA::PreconditionJacobi::AdditionalData());
    ReductionControl solver_control_K_Jp_inv (
      tangent_matrix.block(J_block, p_block).m() * parameters.max_iterations_lin,
      1.0e-30, 1e-6);
    dealii::SolverCG<LA::MPI::Vector> solver_K_Jp_inv (solver_control_K_Jp_inv);

    const auto K_Jp_inv = inverse_operator(K_Jp,
                                           solver_K_Jp_inv,
                                           preconditioner_K_Jp_inv);
    const auto K_pJ_inv     = transpose_operator(K_Jp_inv);
    const auto K_pp_bar     = K_Jp_inv * K_JJ * K_pJ_inv;
    const auto K_uu_bar_bar = K_up * K_pp_bar * K_pu;
    const auto K_uu_con     = K_uu + K_uu_bar_bar;

    LA::PreconditionAMG preconditioner_K_con_inv;
    preconditioner_K_con_inv.initialize(
      tangent_matrix.block(u_block, u_block),
      LA::PreconditionAMG::AdditionalData(
        true /*elliptic*/,
        (parameters.poly_degree > 1 /*higher_order_elements*/)) );
    ReductionControl solver_control_K_con_inv (
      tangent_matrix.block(u_block, u_block).m() * parameters.max_iterations_lin,
      1.0e-30, parameters.tol_lin);
    dealii::SolverSelector<LA::MPI::Vector> solver_K_con_inv;
    solver_K_con_inv.select(parameters.type_lin);
    solver_K_con_inv.set_control(solver_control_K_con_inv);
    const auto K_uu_con_inv = inverse_operator(K_uu_con,
                                               solver_K_con_inv,
                                               preconditioner_K_con_inv);

    d_u     = K_uu_con_inv*(f_u - K_up*(K_Jp_inv*f_J - K_pp_bar*f_p));
    lin_it  = solver_control_K_con_inv.last_step();
    lin_res = solver_control_K_con_inv.last_value();
    timer.leave_subsection();

    timer.enter_subsection("Linear solver postprocessing");
    d_J = K_pJ_inv*(f_p - K_pu*d_u);
    d_p = K_Jp_inv*(f_J - K_JJ*d_J);
    timer.leave_subsection();

    constraints.distribute(newton_update);
    return std::make_pair(lin_it, lin_res);
  }
  template <int dim>
  void
  Solid<dim>::update_end_timestep ()
  {
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    cell (IteratorFilters::SubdomainEqualTo(this_mpi_process),
          dof_handler.begin_active()),
                                   endc (IteratorFilters::SubdomainEqualTo(this_mpi_process),
                                         dof_handler.end());
    for (; cell != endc; ++cell)
      {
        Assert(cell->subdomain_id()==this_mpi_process, ExcInternalError());
        const std::vector<std::shared_ptr<PointHistory<dim> > > lqph =
          quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          lqph[q_point]->update_end_timestep();
      }
  }

  template<int dim, class DH=DoFHandler<dim> >
  class FilteredDataOut : public DataOut<dim, DH>
  {
  public:
    FilteredDataOut (const unsigned int subdomain_id)
      :
      subdomain_id (subdomain_id)
    {}

    virtual ~FilteredDataOut() {}

    virtual typename DataOut<dim, DH>::cell_iterator
    first_cell ()
    {
      auto cell = this->dofs->begin_active();
      while ((cell != this->dofs->end()) &&
             (cell->subdomain_id() != subdomain_id))
        ++cell;
      return cell;
    }

    virtual typename DataOut<dim, DH>::cell_iterator
    next_cell (const typename DataOut<dim, DH>::cell_iterator &old_cell)
    {
      if (old_cell != this->dofs->end())
        {
          const IteratorFilters::SubdomainEqualTo predicate(subdomain_id);
          return
            ++(FilteredIterator
               <typename DataOut<dim, DH>::cell_iterator>
               (predicate,old_cell));
        }
      else
        return old_cell;
    }

  private:
    const unsigned int subdomain_id;
  };

  template <int dim>
  void Solid<dim>::output_results(const unsigned int timestep,
                                  const double       current_time) const
  {
    // Output -> Ghosted vector
    LA::MPI::BlockVector solution_total (locally_owned_partitioning,
                                         locally_relevant_partitioning,
                                         mpi_communicator,
                                         /*vector_writable = */ false);
    LA::MPI::BlockVector residual (locally_owned_partitioning,
                                   locally_relevant_partitioning,
                                   mpi_communicator,
                                   /*vector_writable = */ false);
    solution_total = solution_n;
    residual = system_rhs;
    residual *= -1.0;

    // --- Additional data ---
    Vector<double> material_id;
    Vector<double> polynomial_order;
    material_id.reinit(triangulation.n_active_cells());
    polynomial_order.reinit(triangulation.n_active_cells());
    std::vector<types::subdomain_id> partition_int (triangulation.n_active_cells());

    FilteredDataOut<dim> data_out(this_mpi_process);
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim,
                                  DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    GridTools::get_subdomain_association (triangulation, partition_int);

    // Can't use filtered iterators here because the cell
    // count "c" is incorrect for the parallel case
    unsigned int c = 0;
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell, ++c)
      {
        if (cell->subdomain_id() != this_mpi_process) continue;

        material_id(c) = static_cast<int>(cell->material_id());
      }

    std::vector<std::string> solution_name(n_components, "solution_");
    std::vector<std::string> residual_name(n_components, "residual_");
    for (unsigned int c=0; c<n_components; ++c)
      {
        if (block_component[c] == u_block)
          {
            solution_name[c] += "u";
            residual_name[c] += "u";
          }
        else if (block_component[c] == p_block)
          {
            solution_name[c] += "p";
            residual_name[c] += "p";
          }
        else if (block_component[c] == J_block)
          {
            solution_name[c] += "J";
            residual_name[c] += "J";
          }
        else
          {
            Assert(c <= J_block, ExcInternalError());
          }
      }

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution_total,
                             solution_name,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.add_data_vector(residual,
                             residual_name,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    const Vector<double> partitioning(partition_int.begin(),
                                      partition_int.end());
    data_out.add_data_vector (material_id, "material_id");
    data_out.add_data_vector (partitioning, "partitioning");
    data_out.build_patches(degree);

    struct Filename
    {
      static std::string get_filename_vtu (unsigned int process,
                                           unsigned int timestep,
                                           const unsigned int n_digits = 4)
      {
        std::ostringstream filename_vtu;
        filename_vtu
            << "solution-"
            << (std::to_string(dim) + "d")
            << "."
            << Utilities::int_to_string (process, n_digits)
            << "."
            << Utilities::int_to_string(timestep, n_digits)
            << ".vtu";
        return filename_vtu.str();
      }

      static std::string get_filename_pvtu (unsigned int timestep,
                                            const unsigned int n_digits = 4)
      {
        std::ostringstream filename_vtu;
        filename_vtu
            << "solution-"
            << (std::to_string(dim) + "d")
            << "."
            << Utilities::int_to_string(timestep, n_digits)
            << ".pvtu";
        return filename_vtu.str();
      }

      static std::string get_filename_pvd (void)
      {
        std::ostringstream filename_vtu;
        filename_vtu
            << "solution-"
            << (std::to_string(dim) + "d")
            << ".pvd";
        return filename_vtu.str();
      }
    };

    // Write out main data file
    const std::string filename_vtu = Filename::get_filename_vtu(this_mpi_process, timestep);
    std::ofstream output(filename_vtu.c_str());
    data_out.write_vtu(output);

    // Collection of files written in parallel
    // This next set of steps should only be performed
    // by master process
    if (this_mpi_process == 0)
      {
        // List of all files written out at this timestep by all processors
        std::vector<std::string> parallel_filenames_vtu;
        for (unsigned int p=0; p < n_mpi_processes; ++p)
          {
            parallel_filenames_vtu.push_back(Filename::get_filename_vtu(p, timestep));
          }

        const std::string filename_pvtu (Filename::get_filename_pvtu(timestep));
        std::ofstream pvtu_master(filename_pvtu.c_str());
        data_out.write_pvtu_record(pvtu_master,
                                   parallel_filenames_vtu);

        // Time dependent data master file
        static std::vector<std::pair<double,std::string> > time_and_name_history;
        time_and_name_history.push_back (std::make_pair (current_time,
                                                         filename_pvtu));
        const std::string filename_pvd (Filename::get_filename_pvd());
        std::ofstream pvd_output (filename_pvd.c_str());
        DataOutBase::write_pvd_record (pvd_output, time_and_name_history);
      }
  }
  template <int dim>
  void Solid<dim>::compute_vertex_positions(std::vector<double> &real_time,
                                            std::vector<std::vector<Point<dim> > > &tracked_vertices,
                                            const LA::MPI::BlockVector &solution_total) const
  {
    real_time.push_back(time.current());

    std::vector<bool> vertex_found (tracked_vertices.size(), false);
    std::vector<Tensor<1,dim> > vertex_update (tracked_vertices.size());

    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    cell (IteratorFilters::SubdomainEqualTo(this_mpi_process),
          dof_handler.begin_active()),
                                   endc (IteratorFilters::SubdomainEqualTo(this_mpi_process),
                                         dof_handler.end());
    for (; cell != endc; ++cell)
      {
        Assert(cell->subdomain_id()==this_mpi_process, ExcInternalError());
        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            for (unsigned int p=0; p<tracked_vertices.size(); ++p)
              {
                if (vertex_found[p] == true) continue;

                const Point<dim> pt_ref = tracked_vertices[p][0];
                if (cell->vertex(v).distance(pt_ref) < 1e-6*parameters.scale)
                  {
                    for (unsigned int d=0; d<dim; ++d)
                      vertex_update[p][d] = solution_total(cell->vertex_dof_index(v,u_block+d));

                    vertex_found[p] = true;
                  }
              }
          }
      }

    for (unsigned int p=0; p<tracked_vertices.size(); ++p)
      {
        const int found_on_n_processes = Utilities::MPI::sum(int(vertex_found[p]), mpi_communicator);
        Assert(found_on_n_processes>0, ExcMessage("Vertex not found on any processor"));
        Tensor<1,dim> update;
        for (unsigned int d=0; d<dim; ++d)
          update[d] = Utilities::MPI::sum(vertex_update[p][d], mpi_communicator);
        update /= found_on_n_processes;
        tracked_vertices[p].push_back(tracked_vertices[p][0] + update);
      }

  }
}
int main (int argc, char *argv[])
{
  using namespace dealii;
  using namespace ViscoElasStripHole;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  try
    {
      const unsigned int dim = 2; // Works in both 2d and 3d
      Solid<dim> solid("parameters.prm");
      solid.run();
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
