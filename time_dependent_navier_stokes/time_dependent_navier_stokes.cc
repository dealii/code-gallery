/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2017 Jie Cheng
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 *
 * Author: Jie Cheng <chengjiehust@gmail.com>
 */

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/lac/petsc_block_sparse_matrix.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <fstream>
#include <iostream>
#include <sstream>

namespace fluid
{
  using namespace dealii;

  // @sect3{Create the triangulation}
  // The code to create triangulation is copied from
  // [Martin Kronbichler's
  // code](https://github.com/kronbichler/adaflo/blob/master/tests/flow_past_cylinder.cc)
  // with very few modifications.
  //
  // @sect4{Helper function}
  void create_triangulation_2d(Triangulation<2> &tria,
                               bool compute_in_2d = true)
  {
    SphericalManifold<2> boundary(Point<2>(0.5, 0.2));
    Triangulation<2> left, middle, right, tmp, tmp2;
    GridGenerator::subdivided_hyper_rectangle(
      left,
      std::vector<unsigned int>({3U, 4U}),
      Point<2>(),
      Point<2>(0.3, 0.41),
      false);
    GridGenerator::subdivided_hyper_rectangle(
      right,
      std::vector<unsigned int>({18U, 4U}),
      Point<2>(0.7, 0),
      Point<2>(2.5, 0.41),
      false);

    // Create middle part first as a hyper shell.
    GridGenerator::hyper_shell(middle, Point<2>(0.5, 0.2), 0.05, 0.2, 4, true);
    middle.reset_all_manifolds();
    for (Triangulation<2>::cell_iterator cell = middle.begin();
         cell != middle.end();
         ++cell)
      for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f)
        {
          bool is_inner_rim = true;
          for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_face; ++v)
            {
              Point<2> &vertex = cell->face(f)->vertex(v);
              if (std::abs(vertex.distance(Point<2>(0.5, 0.2)) - 0.05) > 1e-10)
                {
                  is_inner_rim = false;
                  break;
                }
            }
          if (is_inner_rim)
            cell->face(f)->set_manifold_id(1);
        }
    middle.set_manifold(1, boundary);
    middle.refine_global(1);

    // Then move the vertices to the points where we want them to be to create a
    // slightly asymmetric cube with a hole:
    for (Triangulation<2>::cell_iterator cell = middle.begin();
         cell != middle.end();
         ++cell)
      for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v)
        {
          Point<2> &vertex = cell->vertex(v);
          if (std::abs(vertex[0] - 0.7) < 1e-10 &&
              std::abs(vertex[1] - 0.2) < 1e-10)
            vertex = Point<2>(0.7, 0.205);
          else if (std::abs(vertex[0] - 0.6) < 1e-10 &&
                   std::abs(vertex[1] - 0.3) < 1e-10)
            vertex = Point<2>(0.7, 0.41);
          else if (std::abs(vertex[0] - 0.6) < 1e-10 &&
                   std::abs(vertex[1] - 0.1) < 1e-10)
            vertex = Point<2>(0.7, 0);
          else if (std::abs(vertex[0] - 0.5) < 1e-10 &&
                   std::abs(vertex[1] - 0.4) < 1e-10)
            vertex = Point<2>(0.5, 0.41);
          else if (std::abs(vertex[0] - 0.5) < 1e-10 &&
                   std::abs(vertex[1] - 0.0) < 1e-10)
            vertex = Point<2>(0.5, 0.0);
          else if (std::abs(vertex[0] - 0.4) < 1e-10 &&
                   std::abs(vertex[1] - 0.3) < 1e-10)
            vertex = Point<2>(0.3, 0.41);
          else if (std::abs(vertex[0] - 0.4) < 1e-10 &&
                   std::abs(vertex[1] - 0.1) < 1e-10)
            vertex = Point<2>(0.3, 0);
          else if (std::abs(vertex[0] - 0.3) < 1e-10 &&
                   std::abs(vertex[1] - 0.2) < 1e-10)
            vertex = Point<2>(0.3, 0.205);
          else if (std::abs(vertex[0] - 0.56379) < 1e-4 &&
                   std::abs(vertex[1] - 0.13621) < 1e-4)
            vertex = Point<2>(0.59, 0.11);
          else if (std::abs(vertex[0] - 0.56379) < 1e-4 &&
                   std::abs(vertex[1] - 0.26379) < 1e-4)
            vertex = Point<2>(0.59, 0.29);
          else if (std::abs(vertex[0] - 0.43621) < 1e-4 &&
                   std::abs(vertex[1] - 0.13621) < 1e-4)
            vertex = Point<2>(0.41, 0.11);
          else if (std::abs(vertex[0] - 0.43621) < 1e-4 &&
                   std::abs(vertex[1] - 0.26379) < 1e-4)
            vertex = Point<2>(0.41, 0.29);
        }

    // Refine once to create the same level of refinement as in the
    // neighboring domains:
    middle.refine_global(1);

    // Must copy the triangulation because we cannot merge triangulations with
    // refinement:
    GridGenerator::flatten_triangulation(middle, tmp2);

    // Left domain is requred in 3d only.
    if (compute_in_2d)
      {
        GridGenerator::merge_triangulations(tmp2, right, tria);
      }
    else
      {
        GridGenerator::merge_triangulations(left, tmp2, tmp);
        GridGenerator::merge_triangulations(tmp, right, tria);
      }
  }

  // @sect4{2D flow around cylinder triangulation}
  void create_triangulation(Triangulation<2> &tria)
  {
    create_triangulation_2d(tria);
    // Set the left boundary (inflow) to 0, the right boundary (outflow) to 1,
    // upper to 2, lower to 3 and the cylindrical surface to 4.
    for (Triangulation<2>::active_cell_iterator cell = tria.begin();
         cell != tria.end();
         ++cell)
      {
        for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f)
          {
            if (cell->face(f)->at_boundary())
              {
                if (std::abs(cell->face(f)->center()[0] - 2.5) < 1e-12)
                  {
                    cell->face(f)->set_all_boundary_ids(1);
                  }
                else if (std::abs(cell->face(f)->center()[0] - 0.3) < 1e-12)
                  {
                    cell->face(f)->set_all_boundary_ids(0);
                  }
                else if (std::abs(cell->face(f)->center()[1] - 0.41) < 1e-12)
                  {
                    cell->face(f)->set_all_boundary_ids(3);
                  }
                else if (std::abs(cell->face(f)->center()[1]) < 1e-12)
                  {
                    cell->face(f)->set_all_boundary_ids(2);
                  }
                else
                  {
                    cell->face(f)->set_all_boundary_ids(4);
                  }
              }
          }
      }
  }

  // @sect4{3D flow around cylinder triangulation}
  void create_triangulation(Triangulation<3> &tria)
  {
    Triangulation<2> tria_2d;
    create_triangulation_2d(tria_2d, false);
    GridGenerator::extrude_triangulation(tria_2d, 5, 0.41, tria);
    // Set the ids of the boundaries in x direction to 0 and 1; y direction to 2 and 3;
    // z direction to 4 and 5; the cylindrical surface 6.
    for (Triangulation<3>::active_cell_iterator cell = tria.begin();
         cell != tria.end();
         ++cell)
      {
        for (unsigned int f = 0; f < GeometryInfo<3>::faces_per_cell; ++f)
          {
            if (cell->face(f)->at_boundary())
              {
                if (std::abs(cell->face(f)->center()[0] - 2.5) < 1e-12)
                  {
                    cell->face(f)->set_all_boundary_ids(1);
                  }
                else if (std::abs(cell->face(f)->center()[0]) < 1e-12)
                  {
                    cell->face(f)->set_all_boundary_ids(0);
                  }
                else if (std::abs(cell->face(f)->center()[1] - 0.41) < 1e-12)
                  {
                    cell->face(f)->set_all_boundary_ids(3);
                  }
                else if (std::abs(cell->face(f)->center()[1]) < 1e-12)
                  {
                    cell->face(f)->set_all_boundary_ids(2);
                  }
                else if (std::abs(cell->face(f)->center()[2] - 0.41) < 1e-12)
                  {
                    cell->face(f)->set_all_boundary_ids(5);
                  }
                else if (std::abs(cell->face(f)->center()[2]) < 1e-12)
                  {
                    cell->face(f)->set_all_boundary_ids(4);
                  }
                else
                  {
                    cell->face(f)->set_all_boundary_ids(6);
                  }
              }
          }
      }
  }

  // @sect3{Time stepping}
  // This class is pretty much self-explanatory.
  class Time
  {
  public:
    Time(const double time_end,
         const double delta_t,
         const double output_interval,
         const double refinement_interval)
      : timestep(0),
        time_current(0.0),
        time_end(time_end),
        delta_t(delta_t),
        output_interval(output_interval),
        refinement_interval(refinement_interval)
    {
    }
    double current() const { return time_current; }
    double end() const { return time_end; }
    double get_delta_t() const { return delta_t; }
    unsigned int get_timestep() const { return timestep; }
    bool time_to_output() const;
    bool time_to_refine() const;
    void increment();

  private:
    unsigned int timestep;
    double time_current;
    const double time_end;
    const double delta_t;
    const double output_interval;
    const double refinement_interval;
  };

  bool Time::time_to_output() const
  {
    unsigned int delta = static_cast<unsigned int>(output_interval / delta_t);
    return (timestep >= delta && timestep % delta == 0);
  }

  bool Time::time_to_refine() const
  {
    unsigned int delta = static_cast<unsigned int>(refinement_interval / delta_t);
    return (timestep >= delta && timestep % delta == 0);
  }

  void Time::increment()
  {
    time_current += delta_t;
    ++timestep;
  }

  // @sect3{Boundary values}
  // Dirichlet boundary conditions for the velocity inlet and walls.
  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues() : Function<dim>(dim + 1) {}
    virtual double value(const Point<dim> &p,
                         const unsigned int component) const override;

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &values) const override;
  };

  template <int dim>
  double BoundaryValues<dim>::value(const Point<dim> &p,
                                    const unsigned int component) const
  {
    Assert(component < this->n_components,
           ExcIndexRange(component, 0, this->n_components));
    double left_boundary = (dim == 2 ? 0.3 : 0.0);
    if (component == 0 && std::abs(p[0] - left_boundary) < 1e-10)
      {
        // For a parabolic velocity profile, $U_\mathrm{avg} = 2/3
        // U_\mathrm{max}$
        // in 2D, and $U_\mathrm{avg} = 4/9 U_\mathrm{max}$ in 3D.
        // If $\nu = 0.001$, $D = 0.1$, then $Re = 100 U_\mathrm{avg}$.
        double Uavg = 1.0;
        double Umax = (dim == 2 ? 3 * Uavg / 2 : 9 * Uavg / 4);
        double value = 4 * Umax * p[1] * (0.41 - p[1]) / (0.41 * 0.41);
        if (dim == 3)
          {
            value *= 4 * p[2] * (0.41 - p[2]) / (0.41 * 0.41);
          }
        return value;
      }
    return 0;
  }

  template <int dim>
  void BoundaryValues<dim>::vector_value(const Point<dim> &p,
                                         Vector<double> &values) const
  {
    for (unsigned int c = 0; c < this->n_components; ++c)
      values(c) = BoundaryValues<dim>::value(p, c);
  }

  // @sect3{Block preconditioner}
  //
  // The block Schur preconditioner can be written as the product of three
  // matrices:
  // $
  //   P^{-1} = \begin{pmatrix} \tilde{A}^{-1} & 0\\ 0 & I\end{pmatrix}
  //            \begin{pmatrix} I & -B^T\\ 0 & I\end{pmatrix}
  //            \begin{pmatrix} I & 0\\ 0 & \tilde{S}^{-1}\end{pmatrix}
  // $
  // $\tilde{A}$ is symmetric since the convection term is eliminated from the
  // LHS.
  // $\tilde{S}^{-1}$ is the inverse of the Schur complement of $\tilde{A}$,
  // which consists of a reaction term, a diffusion term, a Grad-Div term
  // and a convection term.
  // In practice, the convection contribution is ignored, namely
  // $\tilde{S}^{-1} = -(\nu + \gamma)M_p^{-1} -
  //                   \frac{1}{\Delta{t}}{[B(diag(M_u))^{-1}B^T]}^{-1}$
  // where $M_p$ is the pressure mass, and
  // ${[B(diag(M_u))^{-1}B^T]}$ is an approximation to the Schur complement of
  // (velocity) mass matrix $BM_u^{-1}B^T$.
  //
  // Same as the tutorials, we define a vmult operation for the block
  // preconditioner
  // instead of write it as a matrix. It can be seen from the above definition,
  // the result of the vmult operation of the block preconditioner can be
  // obtained
  // from the results of the vmult operations of $M_u^{-1}$, $M_p^{-1}$,
  // $\tilde{A}^{-1}$, which can be transformed into solving three symmetric
  // linear
  // systems.
  class BlockSchurPreconditioner : public Subscriptor
  {
  public:
    BlockSchurPreconditioner(
      TimerOutput &timer,
      double gamma,
      double viscosity,
      double dt,
      const std::vector<IndexSet> &owned_partitioning,
      const PETScWrappers::MPI::BlockSparseMatrix &system,
      const PETScWrappers::MPI::BlockSparseMatrix &mass,
      PETScWrappers::MPI::BlockSparseMatrix &schur);

    void vmult(PETScWrappers::MPI::BlockVector &dst,
               const PETScWrappers::MPI::BlockVector &src) const;

  private:
    TimerOutput &timer;
    const double gamma;
    const double viscosity;
    const double dt;

    const SmartPointer<const PETScWrappers::MPI::BlockSparseMatrix>
      system_matrix;
    const SmartPointer<const PETScWrappers::MPI::BlockSparseMatrix> mass_matrix;
    // As discussed, ${[B(diag(M_u))^{-1}B^T]}$ and its inverse
    // need to be computed.
    // We can either explicitly compute it out as a matrix, or define
    // it as a class with a vmult operation.
    // The second approach saves some computation to construct the matrix,
    // but leads to slow convergence in CG solver because it is impossible
    // to apply a preconditioner. We go with the first route.
    const SmartPointer<PETScWrappers::MPI::BlockSparseMatrix> mass_schur;
  };

  // @sect4{BlockSchurPreconditioner::BlockSchurPreconditioner}
  //
  // Input parameters and system matrix, mass matrix as well as the mass schur
  // matrix are needed in the preconditioner. In addition, we pass the
  // partitioning information into this class because we need to create some
  // temporary block vectors inside.
  BlockSchurPreconditioner::BlockSchurPreconditioner(
    TimerOutput &timer,
    double gamma,
    double viscosity,
    double dt,
    const std::vector<IndexSet> &owned_partitioning,
    const PETScWrappers::MPI::BlockSparseMatrix &system,
    const PETScWrappers::MPI::BlockSparseMatrix &mass,
    PETScWrappers::MPI::BlockSparseMatrix &schur)
    : timer(timer),
      gamma(gamma),
      viscosity(viscosity),
      dt(dt),
      system_matrix(&system),
      mass_matrix(&mass),
      mass_schur(&schur)
  {
    TimerOutput::Scope timer_section(timer, "CG for Sm");
    // The schur complemete of mass matrix is actually being computed here.
    PETScWrappers::MPI::BlockVector tmp1, tmp2;
    tmp1.reinit(owned_partitioning, mass_matrix->get_mpi_communicator());
    tmp2.reinit(owned_partitioning, mass_matrix->get_mpi_communicator());
    tmp1 = 1;
    tmp2 = 0;
    // Jacobi preconditioner of matrix A is by definition ${diag(A)}^{-1}$,
    // this is exactly what we want to compute.
    PETScWrappers::PreconditionJacobi jacobi(mass_matrix->block(0, 0));
    jacobi.vmult(tmp2.block(0), tmp1.block(0));
    system_matrix->block(1, 0).mmult(
      mass_schur->block(1, 1), system_matrix->block(0, 1), tmp2.block(0));
  }

  // @sect4{BlockSchurPreconditioner::vmult}
  //
  // The vmult operation strictly follows the definition of
  // BlockSchurPreconditioner
  // introduced above. Conceptually it computes $u = P^{-1}v$.
  void BlockSchurPreconditioner::vmult(
    PETScWrappers::MPI::BlockVector &dst,
    const PETScWrappers::MPI::BlockVector &src) const
  {
    // Temporary vectors
    PETScWrappers::MPI::Vector utmp(src.block(0));
    PETScWrappers::MPI::Vector tmp(src.block(1));
    tmp = 0;
    // This block computes $u_1 = \tilde{S}^{-1} v_1$,
    // where CG solvers are used for $M_p^{-1}$ and $S_m^{-1}$.
    {
      TimerOutput::Scope timer_section(timer, "CG for Mp");
      SolverControl mp_control(src.block(1).size(),
                               1e-6 * src.block(1).l2_norm());
      PETScWrappers::SolverCG cg_mp(mp_control,
                                    mass_schur->get_mpi_communicator());
      // $-(\nu + \gamma)M_p^{-1}v_1$
      PETScWrappers::PreconditionBlockJacobi Mp_preconditioner;
      Mp_preconditioner.initialize(mass_matrix->block(1, 1));
      cg_mp.solve(
        mass_matrix->block(1, 1), tmp, src.block(1), Mp_preconditioner);
      tmp *= -(viscosity + gamma);
    }
    // $-\frac{1}{dt}S_m^{-1}v_1$
    {
      TimerOutput::Scope timer_section(timer, "CG for Sm");
      SolverControl sm_control(src.block(1).size(),
                               1e-6 * src.block(1).l2_norm());
      PETScWrappers::SolverCG cg_sm(sm_control,
                                    mass_schur->get_mpi_communicator());
      // PreconditionBlockJacobi works find on Sm if we do not refine the mesh.
      // Because after refine_mesh is called, zero entries will be created on
      // the diagonal (not sure why), which prevents PreconditionBlockJacobi
      // from being used.
      PETScWrappers::PreconditionNone Sm_preconditioner;
      Sm_preconditioner.initialize(mass_schur->block(1, 1));
      cg_sm.solve(
        mass_schur->block(1, 1), dst.block(1), src.block(1), Sm_preconditioner);
      dst.block(1) *= -1 / dt;
    }
    // Adding up these two, we get $\tilde{S}^{-1}v_1$.
    dst.block(1) += tmp;
    // Compute $v_0 - B^T\tilde{S}^{-1}v_1$ based on $u_1$.
    system_matrix->block(0, 1).vmult(utmp, dst.block(1));
    utmp *= -1.0;
    utmp += src.block(0);
    // Finally, compute the product of $\tilde{A}^{-1}$ and utmp
    // using another CG solver.
    {
      TimerOutput::Scope timer_section(timer, "CG for A");
      SolverControl a_control(src.block(0).size(),
                              1e-6 * src.block(0).l2_norm());
      PETScWrappers::SolverCG cg_a(a_control,
                                   mass_schur->get_mpi_communicator());
      // We do not use any preconditioner for this block, which is of course
      // slow,
      // only because the performance of the only two preconditioners available
      // PreconditionBlockJacobi and PreconditionBoomerAMG are even worse than
      // none.
      PETScWrappers::PreconditionNone A_preconditioner;
      A_preconditioner.initialize(system_matrix->block(0, 0));
      cg_a.solve(
        system_matrix->block(0, 0), dst.block(0), utmp, A_preconditioner);
    }
  }

  // @sect3{The incompressible Navier-Stokes solver}
  //
  // Parallel incompressible Navier Stokes equation solver using
  // implicit-explicit time scheme.
  // This program is built upon dealii tutorials step-57, step-40, step-22,
  // and step-20.
  // The system equation is written in the incremental form, and we treat
  // the convection term explicitly. Therefore the system equation is linear
  // and symmetric, which does not need to be solved with Newton's iteration.
  // The system is further stablized and preconditioned with Grad-Div method,
  // where GMRES solver is used as the outer solver.
  template <int dim>
  class InsIMEX
  {
  public:
    InsIMEX(parallel::distributed::Triangulation<dim> &);
    void run();
    ~InsIMEX() { timer.print_summary(); }

  private:
    void setup_dofs();
    void make_constraints();
    void initialize_system();
    void assemble(bool use_nonzero_constraints, bool assemble_system);
    std::pair<unsigned int, double> solve(bool use_nonzero_constraints,
                                          bool assemble_system);
    void refine_mesh(const unsigned int, const unsigned int);
    void output_results(const unsigned int) const;
    double viscosity;
    double gamma;
    const unsigned int degree;
    std::vector<types::global_dof_index> dofs_per_block;

    parallel::distributed::Triangulation<dim> &triangulation;
    FESystem<dim> fe;
    DoFHandler<dim> dof_handler;
    QGauss<dim> volume_quad_formula;
    QGauss<dim - 1> face_quad_formula;

    AffineConstraints<double> zero_constraints;
    AffineConstraints<double> nonzero_constraints;

    BlockSparsityPattern sparsity_pattern;
    // System matrix to be solved
    PETScWrappers::MPI::BlockSparseMatrix system_matrix;
    // Mass matrix is a block matrix which includes both velocity
    // mass matrix and pressure mass matrix.
    PETScWrappers::MPI::BlockSparseMatrix mass_matrix;
    // The schur complement of mass matrix is not a block matrix.
    // However, because we want to reuse the partition we created
    // for the system matrix, it is defined as a block matrix
    // where only one block is actually used.
    PETScWrappers::MPI::BlockSparseMatrix mass_schur;
    // The latest known solution.
    PETScWrappers::MPI::BlockVector present_solution;
    // The increment at a certain time step.
    PETScWrappers::MPI::BlockVector solution_increment;
    // System RHS
    PETScWrappers::MPI::BlockVector system_rhs;

    MPI_Comm mpi_communicator;

    ConditionalOStream pcout;

    // The IndexSets of owned velocity and pressure respectively.
    std::vector<IndexSet> owned_partitioning;

    // The IndexSets of relevant velocity and pressure respectively.
    std::vector<IndexSet> relevant_partitioning;

    // The IndexSet of all relevant dofs.
    IndexSet locally_relevant_dofs;

    // The BlockSchurPreconditioner for the entire system.
    std::shared_ptr<BlockSchurPreconditioner> preconditioner;

    Time time;
    mutable TimerOutput timer;
  };

  // @sect4{InsIMEX::InsIMEX}
  template <int dim>
  InsIMEX<dim>::InsIMEX(parallel::distributed::Triangulation<dim> &tria)
    : viscosity(0.001),
      gamma(0.1),
      degree(1),
      triangulation(tria),
      fe(FE_Q<dim>(degree + 1), dim, FE_Q<dim>(degree), 1),
      dof_handler(triangulation),
      volume_quad_formula(degree + 2),
      face_quad_formula(degree + 2),
      mpi_communicator(MPI_COMM_WORLD),
      pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0),
      time(1e0, 1e-3, 1e-2, 1e-2),
      timer(
        mpi_communicator, pcout, TimerOutput::never, TimerOutput::wall_times)
  {
  }

  // @sect4{InsIMEX::setup_dofs}
  template <int dim>
  void InsIMEX<dim>::setup_dofs()
  {
    // The first step is to associate DoFs with a given mesh.
    dof_handler.distribute_dofs(fe);
    // We renumber the components to have all velocity DoFs come before
    // the pressure DoFs to be able to split the solution vector in two blocks
    // which are separately accessed in the block preconditioner.
    DoFRenumbering::Cuthill_McKee(dof_handler);
    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    DoFRenumbering::component_wise(dof_handler, block_component);
    dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    // Partitioning.
    unsigned int dof_u = dofs_per_block[0];
    unsigned int dof_p = dofs_per_block[1];
    owned_partitioning.resize(2);
    owned_partitioning[0] = dof_handler.locally_owned_dofs().get_view(0, dof_u);
    owned_partitioning[1] =
      dof_handler.locally_owned_dofs().get_view(dof_u, dof_u + dof_p);
    locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
    relevant_partitioning.resize(2);
    relevant_partitioning[0] = locally_relevant_dofs.get_view(0, dof_u);
    relevant_partitioning[1] =
      locally_relevant_dofs.get_view(dof_u, dof_u + dof_p);
    pcout << "   Number of active fluid cells: "
          << triangulation.n_global_active_cells() << std::endl
          << "   Number of degrees of freedom: " << dof_handler.n_dofs() << " ("
          << dof_u << '+' << dof_p << ')' << std::endl;
  }

  // @sect4{InsIMEX::make_constraints}
  template <int dim>
  void InsIMEX<dim>::make_constraints()
  {
    // Because the equation is written in incremental form, two constraints
    // are needed: nonzero constraint and zero constraint.
    nonzero_constraints.clear();
    zero_constraints.clear();
    nonzero_constraints.reinit(locally_relevant_dofs);
    zero_constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, nonzero_constraints);
    DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints);

    // Apply Dirichlet boundary conditions on all boundaries except for the
    // outlet.
    std::vector<unsigned int> dirichlet_bc_ids;
    if (dim == 2)
      dirichlet_bc_ids = std::vector<unsigned int>{0, 2, 3, 4};
    else
      dirichlet_bc_ids = std::vector<unsigned int>{0, 2, 3, 4, 5, 6};

    FEValuesExtractors::Vector velocities(0);
    for (auto id : dirichlet_bc_ids)
      {
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 id,
                                                 BoundaryValues<dim>(),
                                                 nonzero_constraints,
                                                 fe.component_mask(velocities));
        VectorTools::interpolate_boundary_values(
          dof_handler,
          id,
          Functions::ZeroFunction<dim>(dim + 1),
          zero_constraints,
          fe.component_mask(velocities));
      }
    nonzero_constraints.close();
    zero_constraints.close();
  }

  // @sect4{InsIMEX::initialize_system}
  template <int dim>
  void InsIMEX<dim>::initialize_system()
  {
    preconditioner.reset();
    system_matrix.clear();
    mass_matrix.clear();
    mass_schur.clear();

    BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, nonzero_constraints);
    sparsity_pattern.copy_from(dsp);
    SparsityTools::distribute_sparsity_pattern(
      dsp,
      dof_handler.locally_owned_dofs(),
      mpi_communicator,
      locally_relevant_dofs);

    system_matrix.reinit(owned_partitioning, dsp, mpi_communicator);
    mass_matrix.reinit(owned_partitioning, dsp, mpi_communicator);

    // Only the $(1, 1)$ block in the mass schur matrix is used.
    // Compute the sparsity pattern for mass schur in advance.
    // The only nonzero block has the same sparsity pattern as $BB^T$.
    BlockDynamicSparsityPattern schur_dsp(dofs_per_block, dofs_per_block);
    schur_dsp.block(1, 1).compute_mmult_pattern(sparsity_pattern.block(1, 0),
                                                sparsity_pattern.block(0, 1));
    mass_schur.reinit(owned_partitioning, schur_dsp, mpi_communicator);

    // present_solution is ghosted because it is used in the
    // output and mesh refinement functions.
    present_solution.reinit(
      owned_partitioning, relevant_partitioning, mpi_communicator);
    // solution_increment is non-ghosted because the linear solver needs
    // a completely distributed vector.
    solution_increment.reinit(owned_partitioning, mpi_communicator);
    // system_rhs is non-ghosted because it is only used in the linear
    // solver and residual evaluation.
    system_rhs.reinit(owned_partitioning, mpi_communicator);
  }

  // @sect4{InsIMEX::assemble}
  //
  // Assemble the system matrix, mass matrix, and the RHS.
  // It can be used to assemble the entire system or only the RHS.
  // An additional option is added to determine whether nonzero
  // constraints or zero constraints should be used.
  // Note that we only need to assemble the LHS for twice: once with the nonzero
  // constraint
  // and once for zero constraint. But we must assemble the RHS at every time
  // step.
  template <int dim>
  void InsIMEX<dim>::assemble(bool use_nonzero_constraints,
                              bool assemble_system)
  {
    TimerOutput::Scope timer_section(timer, "Assemble system");

    if (assemble_system)
      {
        system_matrix = 0;
        mass_matrix = 0;
      }
    system_rhs = 0;

    FEValues<dim> fe_values(fe,
                            volume_quad_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);
    FEFaceValues<dim> fe_face_values(fe,
                                     face_quad_formula,
                                     update_values | update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = volume_quad_formula.size();

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_mass_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<Tensor<1, dim>> current_velocity_values(n_q_points);
    std::vector<Tensor<2, dim>> current_velocity_gradients(n_q_points);
    std::vector<double> current_velocity_divergences(n_q_points);
    std::vector<double> current_pressure_values(n_q_points);

    std::vector<double> div_phi_u(dofs_per_cell);
    std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
    std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    std::vector<double> phi_p(dofs_per_cell);

    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
         ++cell)
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);

            if (assemble_system)
              {
                local_matrix = 0;
                local_mass_matrix = 0;
              }
            local_rhs = 0;

            fe_values[velocities].get_function_values(present_solution,
                                                      current_velocity_values);

            fe_values[velocities].get_function_gradients(
              present_solution, current_velocity_gradients);

            fe_values[velocities].get_function_divergences(
              present_solution, current_velocity_divergences);

            fe_values[pressure].get_function_values(present_solution,
                                                    current_pressure_values);

            // Assemble the system matrix and mass matrix simultaneouly.
            // The mass matrix only uses the $(0, 0)$ and $(1, 1)$ blocks.
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                  {
                    div_phi_u[k] = fe_values[velocities].divergence(k, q);
                    grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                    phi_u[k] = fe_values[velocities].value(k, q);
                    phi_p[k] = fe_values[pressure].value(k, q);
                  }

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    if (assemble_system)
                      {
                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                          {
                            local_matrix(i, j) +=
                              (viscosity *
                                 scalar_product(grad_phi_u[j], grad_phi_u[i]) -
                               div_phi_u[i] * phi_p[j] -
                               phi_p[i] * div_phi_u[j] +
                               gamma * div_phi_u[j] * div_phi_u[i] +
                               phi_u[i] * phi_u[j] / time.get_delta_t()) *
                              fe_values.JxW(q);
                            local_mass_matrix(i, j) +=
                              (phi_u[i] * phi_u[j] + phi_p[i] * phi_p[j]) *
                              fe_values.JxW(q);
                          }
                      }
                    local_rhs(i) -=
                      (viscosity * scalar_product(current_velocity_gradients[q],
                                                  grad_phi_u[i]) -
                       current_velocity_divergences[q] * phi_p[i] -
                       current_pressure_values[q] * div_phi_u[i] +
                       gamma * current_velocity_divergences[q] * div_phi_u[i] +
                       current_velocity_gradients[q] *
                         current_velocity_values[q] * phi_u[i]) *
                      fe_values.JxW(q);
                  }
              }

            cell->get_dof_indices(local_dof_indices);

            const AffineConstraints<double> &constraints_used =
              use_nonzero_constraints ? nonzero_constraints : zero_constraints;
            if (assemble_system)
              {
                constraints_used.distribute_local_to_global(local_matrix,
                                                            local_rhs,
                                                            local_dof_indices,
                                                            system_matrix,
                                                            system_rhs);
                constraints_used.distribute_local_to_global(
                  local_mass_matrix, local_dof_indices, mass_matrix);
              }
            else
              {
                constraints_used.distribute_local_to_global(
                  local_rhs, local_dof_indices, system_rhs);
              }
          }
      }

    if (assemble_system)
      {
        system_matrix.compress(VectorOperation::add);
        mass_matrix.compress(VectorOperation::add);
      }
    system_rhs.compress(VectorOperation::add);
  }

  // @sect4{InsIMEX::solve}
  // Solve the linear system using FGMRES solver with block preconditioner.
  // After solving the linear system, the same AffineConstraints object as used
  // in assembly must be used again, to set the constrained value.
  // The second argument is used to determine whether the block
  // preconditioner should be reset or not.
  template <int dim>
  std::pair<unsigned int, double>
  InsIMEX<dim>::solve(bool use_nonzero_constraints, bool assemble_system)
  {
    if (assemble_system)
      {
        preconditioner.reset(new BlockSchurPreconditioner(timer,
                                                          gamma,
                                                          viscosity,
                                                          time.get_delta_t(),
                                                          owned_partitioning,
                                                          system_matrix,
                                                          mass_matrix,
                                                          mass_schur));
      }

    SolverControl solver_control(
      system_matrix.m(), 1e-8 * system_rhs.l2_norm(), true);
    // Because PETScWrappers::SolverGMRES only accepts preconditioner
    // derived from PETScWrappers::PreconditionBase,
    // we use dealii SolverFGMRES.
    GrowingVectorMemory<PETScWrappers::MPI::BlockVector> vector_memory;
    SolverFGMRES<PETScWrappers::MPI::BlockVector> gmres(solver_control,
                                                        vector_memory);

    // The solution vector must be non-ghosted
    gmres.solve(system_matrix, solution_increment, system_rhs, *preconditioner);

    const AffineConstraints<double> &constraints_used =
      use_nonzero_constraints ? nonzero_constraints : zero_constraints;
    constraints_used.distribute(solution_increment);

    return {solver_control.last_step(), solver_control.last_value()};
  }

  // @sect4{InsIMEX::run}
  template <int dim>
  void InsIMEX<dim>::run()
  {
    pcout << "Running with PETSc on "
          << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

    triangulation.refine_global(0);
    setup_dofs();
    make_constraints();
    initialize_system();

    // Time loop.
    bool refined = false;
    while (time.end() - time.current() > 1e-12)
      {
        if (time.get_timestep() == 0)
          {
            output_results(0);
          }
        time.increment();
        std::cout.precision(6);
        std::cout.width(12);
        pcout << std::string(96, '*') << std::endl
              << "Time step = " << time.get_timestep()
              << ", at t = " << std::scientific << time.current() << std::endl;
        // Resetting
        solution_increment = 0;
        // Only use nonzero constraints at the very first time step
        bool apply_nonzero_constraints = (time.get_timestep() == 1);
        // We have to assemble the LHS for the initial two time steps:
        // once using nonzero_constraints, once using zero_constraints,
        // as well as the steps imediately after mesh refinement.
        bool assemble_system = (time.get_timestep() < 3 || refined);
        refined = false;
        assemble(apply_nonzero_constraints, assemble_system);
        auto state = solve(apply_nonzero_constraints, assemble_system);
        // Note we have to use a non-ghosted vector to do the addition.
        PETScWrappers::MPI::BlockVector tmp;
        tmp.reinit(owned_partitioning, mpi_communicator);
        tmp = present_solution;
        tmp += solution_increment;
        present_solution = tmp;
        pcout << std::scientific << std::left << " GMRES_ITR = " << std::setw(3)
              << state.first << " GMRES_RES = " << state.second << std::endl;
        // Output
        if (time.time_to_output())
          {
            output_results(time.get_timestep());
          }
        if (time.time_to_refine())
          {
            refine_mesh(0, 4);
            refined = true;
          }
      }
  }

  // @sect4{InsIMEX::output_result}
  //
  template <int dim>
  void InsIMEX<dim>::output_results(const unsigned int output_index) const
  {
    TimerOutput::Scope timer_section(timer, "Output results");
    pcout << "Writing results..." << std::endl;
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.push_back("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    // vector to be output must be ghosted
    data_out.add_data_vector(present_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    // Partition
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      {
        subdomain(i) = triangulation.locally_owned_subdomain();
      }
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(degree + 1);

    std::string basename =
      "navierstokes" + Utilities::int_to_string(output_index, 6) + "-";

    std::string filename =
      basename +
      Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
      ".vtu";

    std::ofstream output(filename);
    data_out.write_vtu(output);

    static std::vector<std::pair<double, std::string>> times_and_names;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        for (unsigned int i = 0;
             i < Utilities::MPI::n_mpi_processes(mpi_communicator);
             ++i)
          {
            times_and_names.push_back(
              {time.current(),
               basename + Utilities::int_to_string(i, 4) + ".vtu"});
          }
        std::ofstream pvd_output("navierstokes.pvd");
        DataOutBase::write_pvd_record(pvd_output, times_and_names);
      }
  }

  // @sect4{InsIMEX::refine_mesh}
  //
  template <int dim>
  void InsIMEX<dim>::refine_mesh(const unsigned int min_grid_level,
                                 const unsigned int max_grid_level)
  {
    TimerOutput::Scope timer_section(timer, "Refine mesh");
    pcout << "Refining mesh..." << std::endl;

    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    FEValuesExtractors::Vector velocity(0);
    KellyErrorEstimator<dim>::estimate(dof_handler,
                                       face_quad_formula,
                                       {},
                                       present_solution,
                                       estimated_error_per_cell,
                                       fe.component_mask(velocity));
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
      triangulation, estimated_error_per_cell, 0.6, 0.4);
    if (triangulation.n_levels() > max_grid_level)
      {
        for (auto cell = triangulation.begin_active(max_grid_level);
             cell != triangulation.end();
             ++cell)
          {
            cell->clear_refine_flag();
          }
      }
    for (auto cell = triangulation.begin_active(min_grid_level);
         cell != triangulation.end_active(min_grid_level);
         ++cell)
      {
        cell->clear_coarsen_flag();
      }

    // Prepare to transfer
    parallel::distributed::SolutionTransfer<dim,
                                            PETScWrappers::MPI::BlockVector>
      trans(dof_handler);

    triangulation.prepare_coarsening_and_refinement();

    trans.prepare_for_coarsening_and_refinement(present_solution);

    // Refine the mesh
    triangulation.execute_coarsening_and_refinement();

    // Reinitialize the system
    setup_dofs();
    make_constraints();
    initialize_system();

    // Transfer solution
    // Need a non-ghosted vector for interpolation
    PETScWrappers::MPI::BlockVector tmp(solution_increment);
    tmp = 0;
    trans.interpolate(tmp);
    present_solution = tmp;
  }
}

// @sect3{main function}
//
int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace fluid;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
      create_triangulation(tria);
      InsIMEX<2> flow(tria);
      flow.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
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
      std::cerr << std::endl
                << std::endl
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
