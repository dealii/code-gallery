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
          double E_e; // Young's modulus
          double nu_e; // Poisson ratio

          double mu_e; // Shear modulus
          double lambda_e; // Lame parameter

          static void
          declare_parameters(ParameterHandler &prm);

          void
          parse_parameters(ParameterHandler &prm);
        };

        void MaterialProperties::declare_parameters(ParameterHandler &prm)
        {
          prm.enter_subsection("Material properties");
          {
            prm.declare_entry("Young's modulus", "100",
                              Patterns::Double(0),
                              "Young's modulus of the specimen");

            prm.declare_entry("Poisson ratio", "10",
                              Patterns::Double(0),
                              "Poisson ratio of the specimen");
          }
          prm.leave_subsection();
        }

        void MaterialProperties::parse_parameters(ParameterHandler &prm)
        {
          prm.enter_subsection("Material properties");
          {
            E_e = prm.get_double("Young's modulus");
            nu_e = prm.get_double("Poisson ratio");
          }
          prm.leave_subsection();

          mu_e = E_e/(2.0*(1.0 + nu_e));
          lambda_e = 2.0*mu_e*nu_e/(1.0 - 2.0*nu_e);
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
  class Material_Linear_Elastoplastic
  {
  public:
    Material_Linear_Elastoplastic(const double lambda_e,
                                  const double mu_e,
//                                  const double mu_v,
//                                  const double tau_v,
                                  const Time  &time)
      :
      lambda_e(lambda_e),
      mu_e(mu_e),
//      mu_v(mu_v),
//      tau_v(tau_v),
      time(time)
//      Q_n_t(Physics::Elasticity::StandardTensors<dim>::I),
//      Q_t1(Physics::Elasticity::StandardTensors<dim>::I)
    {
//      Assert(kappa > 0, ExcInternalError());
    }
    ~Material_Linear_Elastoplastic()
    {}

    SymmetricTensor<2,dim>
    get_sigma(const SymmetricTensor<2, dim> &epsilon) const
    {
      return get_K(epsilon)*epsilon;
    }
    SymmetricTensor<4,dim> get_K(const SymmetricTensor<2, dim> &epsilon) const
    {
      (void)epsilon;
      static const SymmetricTensor<2,dim> I = unit_symmetric_tensor<dim>();

      SymmetricTensor<4,dim> K;
      for (unsigned int i=0; i < dim; ++i)
        for (unsigned int j=i; j < dim; ++j)
          for (unsigned int k=0; k < dim; ++k)
            for (unsigned int l=k; l < dim; ++l)
              {
                // Matrix contribution
                K[i][j][k][l] = lambda_e * I[i][j]*I[k][l]
                                + mu_e * (I[i][k]*I[j][l] + I[i][l]*I[j][k]);
              }

      return K;
    }
    void
    update_internal_equilibrium(const SymmetricTensor<2, dim> &epsilon)
    {
      (void)epsilon;
//      const double det_F = determinant(F);
//      const SymmetricTensor<2,dim> C_bar = std::pow(det_F, -2.0 / dim) * Physics::Elasticity::Kinematics::C(F);
//      // Linder2011 eq 54
//      // Assumes first-oder backward Euler time discretisation
//      Q_n_t = (1.0/(1.0 + time.get_delta_t()/tau_v))*(Q_t1 + (time.get_delta_t()/tau_v)*invert(C_bar));
    }
    void
    update_end_timestep()
    {
//      Q_t1 = Q_n_t;
    }

  protected:
    const double lambda_e;
    const double mu_e;
//    const double mu_v;
//    const double tau_v;

    const Time  &time;

//    SymmetricTensor<2,dim> Q_n_t; // Value of internal variable at this Newton step and timestep
//    SymmetricTensor<2,dim> Q_t1; // Value of internal variable at the previous timestep
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
      material.reset(new Material_Linear_Elastoplastic<dim>(
                       parameters.lambda_e, parameters.mu_e,
//                       parameters.mu_v, parameters.tau_v,
                       time));
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
    std::shared_ptr< Material_Linear_Elastoplastic<dim> > material;
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

    ConstraintMatrix     hanging_node_constraints;

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
    fe (FE_Q<dim>(parameters.poly_degree), dim),
    qf_cell (parameters.quad_order),
    qf_face (parameters.quad_order),
    time (parameters.end_time, parameters.delta_t),
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

    GridGenerator::cylinder (triangulation, parameters.radius, parameters.length/2.0);
    triangulation.refine_global(parameters.n_global_refinement_steps);
    GridTools::scale (parameters.scale, triangulation);
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
          lqph[q_point]->setup_lqp(parameters, time);
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
    const unsigned int   n_q_points_face = qf_face.size();

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
          const SymmetricTensor<2,dim> epsilon = Physics::Elasticity::Kinematics::epsilon(solution_grads_u_total[q_point_cell]);
            const SymmetricTensor<4,dim> K = lqph[q_point_cell]->get_K(epsilon);

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
                            K[component_I][k][component_J][l] *
                            fe_values.shape_grad(J,q_point_cell)[l]) *
                           fe_values.JxW(q_point_cell);
                  }
              }

//            for (unsigned int I=0; I<dofs_per_cell; ++I)
//              {
//                const unsigned int
//                component_I = fe.system_to_component_index(I).first;
//
//                cell_rhs(I)
//                += fe_values.shape_value(I,q_point_cell) *
//                   body_force_values[q_point_cell](component_I) *
//                   fe_values.JxW(q_point_cell);
//              }
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
  void LinearElastoplasticProblem<dim>::apply_boundary_conditions ()
  {
    std::map<types::global_dof_index,double> boundary_values;

    // Full constraint on -X faces
    {
      ComponentMask component_mask_all (n_components, true);
      component_mask_all.set(0, true);
      VectorTools::interpolate_boundary_values (dof_handler,
                                                1,
                                                ZeroFunction<dim>(dim),
                                                boundary_values,
                                                component_mask_all);
    }
    // Horizontal displacement on +X faces
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

//        // Fixed point on -X face
//        {
//          const Point<dim> fixed_point (-parameters.half_length_x,0.0,0.0);
//          std::vector<types::global_dof_index> fixed_dof_indices;
//          bool found_point_of_interest = false;
//
//          for (typename DoFHandler<dim>::active_cell_iterator
//               cell = dof_handler.begin_active(),
//               endc = dof_handler.end(); cell != endc; ++cell)
//            {
//              for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
//                {
//                  // We know that the fixed point is on the -X Dirichlet boundary
//                  if (cell->face(face)->at_boundary() == true &&
//                      cell->face(face)->boundary_id() == parameters.bid_CC_dirichlet_symm_X)
//                    {
//                      for (unsigned int face_vertex_index = 0; face_vertex_index < GeometryInfo<dim>::vertices_per_face; ++face_vertex_index)
//                        {
//                          if (cell->face(face)->vertex(face_vertex_index).distance(fixed_point) < 1e-6)
//                            {
//                              found_point_of_interest = true;
//                              for (unsigned int index_component = 0; index_component < dim; ++index_component)
//                                fixed_dof_indices.push_back(cell->face(face)->vertex_dof_index(face_vertex_index,
//                                                            index_component));
//                            }
//
//                          if (found_point_of_interest == true) break;
//                        }
//                    }
//                  if (found_point_of_interest == true) break;
//                }
//              if (found_point_of_interest == true) break;
//            }
//
//          Assert(found_point_of_interest == true, ExcMessage("Didn't find point of interest"));
//          AssertThrow(fixed_dof_indices.size() == dim, ExcMessage("Didn't find the correct number of DoFs to fix"));
//
//          for (unsigned int i=0; i < fixed_dof_indices.size(); ++i)
//            boundary_values[fixed_dof_indices[i]] = 0.0;
//        }
//      }

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
