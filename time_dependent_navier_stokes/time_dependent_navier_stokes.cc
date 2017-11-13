#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
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
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/physics/elasticity/standard_tensors.h>

#include <fstream>
#include <iostream>
#include <sstream>

namespace fluid
{
  using namespace dealii;

  // @sect3{Create the triangulation}

  // The code to create triangulation is copied from Martin Kronbichler's code
  // (https://github.com/kronbichler/adaflo/blob/master/tests/flow_past_cylinder.cc)
  // with very few modifications.
  // Helper function used in both 2d and 3d:
  void create_triangulation_2d(Triangulation<2> &tria, bool compute_in_2d = true)
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
    middle.set_manifold(0, boundary);
    middle.refine_global(1);

    // Then move the vertices to the points where we want them to be to create a
    // slightly asymmetric cube with a hole
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

  // Create 2D triangulation:
  void create_triangulation(Triangulation<2> &tria)
  {
    create_triangulation_2d(tria);
    // Set the cylinder boundary to 1, the right boundary (outflow) to 2, the rest to 0.
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
              cell->face(f)->set_all_boundary_ids(2);
            }
            else if (Point<2>(0.5, 0.2).distance(cell->face(f)->center()) <= 0.05)
              {
                cell->face(f)->set_all_manifold_ids(10);
                cell->face(f)->set_all_boundary_ids(1);
              }
            else
            {
              cell->face(f)->set_all_boundary_ids(0);
            }
          }
      }
    }
  }

  // Create 3D triangulation:
  void create_triangulation(Triangulation<3> &tria)
  {
    Triangulation<2> tria_2d;
    create_triangulation_2d(tria_2d, false);
    GridGenerator::extrude_triangulation(tria_2d, 5, 0.41, tria);
    // Set the cylinder boundary to 1, the right boundary (outflow) to 2, the rest to 0.
    for (Triangulation<3>::active_cell_iterator cell = tria.begin();
        cell != tria.end(); ++cell)
    {
      for (unsigned int f = 0; f<GeometryInfo<3>::faces_per_cell; ++f)
      {
        if (cell->face(f)->at_boundary())
        {
          if (std::abs(cell->face(f)->center()[0]-2.5) < 1e-12)
          {
            cell->face(f)->set_all_boundary_ids(2);
          }
          else if (Point<3>(0.5, 0.2, cell->face(f)->center()[2]).distance
            (cell->face(f)->center()) <= 0.05)
          {
            cell->face(f)->set_all_manifold_ids(10);
            cell->face(f)->set_all_boundary_ids(1);
          }
          else
          {
            cell->face(f)->set_all_boundary_ids(0);
          }
        }
      }
    }
  }

  // @sect3{Time stepping}
  class Time
  {
  public:
    Time(const double time_end, const double delta_t)
      : timestep(0), time_current(0.0), time_end(time_end), delta_t(delta_t)
    {
    }
    virtual ~Time() {}
    double current() const { return time_current; }
    double end() const { return time_end; }
    double get_delta_t() const { return delta_t; }
    unsigned int get_timestep() const { return timestep; }
    void increment()
    {
      time_current += delta_t;
      ++timestep;
    }

  private:
    unsigned int timestep;
    double time_current;
    const double time_end;
    const double delta_t;
  };

  // @sect3{Boundary values}

  // Dirichlet boundary conditions for the velocity inlet and walls
  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues() : Function<dim>(dim + 1) {}
    virtual double value(const Point<dim> &p,
                         const unsigned int component) const;

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &values) const;
  };

  template <int dim>
  double BoundaryValues<dim>::value(const Point<dim> &p,
                                    const unsigned int component) const
  {
    Assert(component < this->n_components,
           ExcIndexRange(component, 0, this->n_components));
    if (component == 0 && std::abs(p[0] - 0.3) < 1e-10)
      {
        double U = 1.5;
        double y = p[1];
        return 4 * U * y * (0.41 - y) / (0.41 * 0.41);
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

  // @sect3{Preconditioners}

  // The LHS of the system matrix is the same as Stokes equation for IMEX scheme.
  // A block preconditioner as in step-22 is used here.
 
  // @sect4{Inner preconditioner}

  // Adapted from step-22, used to solve for ${\tilde{A}}^{-1}$
  template <int dim>
  struct InnerPreconditioner;

  template <>
  struct InnerPreconditioner<2>
  {
    typedef SparseDirectUMFPACK type;
  };

  template <>
  struct InnerPreconditioner<3>
  {
    typedef SparseILU<double> type;
  };

  // @sect4{Inverse matrix}

  // This is used for ${\tilde{S}}^{-1}$ and ${\tilde{A}}^{-1}$, which are symmetric so we use CG
  // solver inside
  template <class MatrixType, class PreconditionerType>
  class InverseMatrix : public Subscriptor
  {
  public:
    InverseMatrix(const MatrixType &m,
                  const PreconditionerType &preconditioner);
    void vmult(Vector<double> &dst, const Vector<double> &src) const;

  private:
    const SmartPointer<const MatrixType> matrix;
    const SmartPointer<const PreconditionerType> preconditioner;
  };

  template <class MatrixType, class PreconditionerType>
  InverseMatrix<MatrixType, PreconditionerType>::InverseMatrix(
    const MatrixType &m, const PreconditionerType &preconditioner)
    : matrix(&m), preconditioner(&preconditioner)
  {
  }

  template <class MatrixType, class PreconditionerType>
  void InverseMatrix<MatrixType, PreconditionerType>::vmult(
    Vector<double> &dst, const Vector<double> &src) const
  {
    SolverControl solver_control(src.size(), 1e-6 * src.l2_norm());
    SolverCG<> cg(solver_control);
    dst = 0;
    cg.solve(*matrix, dst, src, *preconditioner);
  }

  // @sect4{Approximate Schur complement of mass matrix}

  // The Schur complement of mass matrix is written as $S_M = BM^{-1}B^T$
  // Similar to step-20, we use $B(diag(M))^{-1}B^T$ to approximate it.
  class ApproximateMassSchur : public Subscriptor
  {
  public:
    ApproximateMassSchur(const BlockSparseMatrix<double> &M);
    void vmult(Vector<double> &dst, const Vector<double> &src) const;

  private:
    const SmartPointer<const BlockSparseMatrix<double>> mass_matrix;
    mutable Vector<double> tmp1, tmp2;
  };

  ApproximateMassSchur::ApproximateMassSchur(
    const BlockSparseMatrix<double> &M)
    : mass_matrix(&M), tmp1(M.block(0, 0).m()), tmp2(M.block(0, 0).m())
  {
  }

  void ApproximateMassSchur::vmult(Vector<double> &dst,
                                         const Vector<double> &src) const
  {
    mass_matrix->block(0, 1).vmult(tmp1, src);
    mass_matrix->block(0, 0).precondition_Jacobi(tmp2, tmp1);
    mass_matrix->block(1, 0).vmult(dst, tmp2);
  }

  // @sect4{The inverse matrix of the system Schur complement}

  // The inverse of the total Schur complement is the sum of the inverse of
  // diffusion, Grad-Div term, and mass Schur complements. Note that the first
  // two components add up to $\Delta{t}(\nu + \gamma)M_p^{-1}$ as introduced in step-57,
  // in which the additional $\Delta{t}$ comes from the time discretization,
  // and the last component is obtained by wrapping a <code>InverseMatrix<\code>
  // around <code>ApproximateMassSchur<\code>.
  template <class PreconditionerSm, class PreconditionerMp>
  class SchurComplementInverse : public Subscriptor
  {
  public:
    SchurComplementInverse(
      double gamma, double viscosity, double dt,
      const InverseMatrix<ApproximateMassSchur, PreconditionerSm> &Sm_inv,
      const InverseMatrix<SparseMatrix<double>, PreconditionerMp> &Mp_inv);
    void vmult(Vector<double> &dst, const Vector<double> &src) const;
  private:
    const double gamma;
    const double viscosity;
    const double dt;
    const SmartPointer<const InverseMatrix<ApproximateMassSchur,
      PreconditionerSm>> Sm_inverse;
    const SmartPointer<const InverseMatrix<SparseMatrix<double>,
      PreconditionerMp>> Mp_inverse;
  };

  template <class PreconditionerSm, class PreconditionerMp>
  SchurComplementInverse<PreconditionerSm, PreconditionerMp>::SchurComplementInverse(
    double gamma, double viscosity, double dt,
    const InverseMatrix<ApproximateMassSchur, PreconditionerSm> &Sm_inv,
    const InverseMatrix<SparseMatrix<double>, PreconditionerMp> &Mp_inv) :
    gamma(gamma), viscosity(viscosity), dt(dt), Sm_inverse(&Sm_inv), Mp_inverse(&Mp_inv)
  {
  }

  template <class PreconditionerSm, class PreconditionerMp>
  void SchurComplementInverse<PreconditionerSm, PreconditionerMp>::vmult(
    Vector<double> &dst, const Vector<double> &src) const
  {
    Vector<double> tmp(src.size());
    Sm_inverse->vmult(dst, src);
    Mp_inverse->vmult(tmp, src);
    tmp *= (viscosity + gamma) * dt;
    dst += tmp;
  }

  // @sect4{The block Schur preconditioner}

  // The block Schur preconditioner has the same form as in step-22, which is written as
  // $P^{-1} = [\tilde{A}}^{-1}, 0; {\tilde{S}}^{-1}B{\tilde{A}}^{-1}, -{\tilde{S}}^{-1}]$
  // Note that ${\tilde{A}}^{-1}$ has contributions from the diffusion, Grad-Div and mass terms.
  // This class has three template arguments: PreconditionerA is needed for ${\tilde{A}}^{-1}$,
  // PreconditionerSm and PreconditionerMp are used in the inverse of the Schur complement
  // of $\tilde{A}$, namely ${\tilde{S}}^{-1}$.
  template <class PreconditionerA, class PreconditionerSm, class PreconditionerMp>
  class BlockSchurPreconditioner : public Subscriptor
  {
  public:
    BlockSchurPreconditioner(
      const BlockSparseMatrix<double> &system_m,
      const InverseMatrix<SparseMatrix<double>, PreconditionerA> &A_inv,
      const SchurComplementInverse<PreconditionerSm, PreconditionerMp> &S_inv);
    void vmult(BlockVector<double> &dst, const BlockVector<double> &src) const;

  private:
    const SmartPointer<const BlockSparseMatrix<double>> system_matrix;
    const SmartPointer<
      const InverseMatrix<SparseMatrix<double>, PreconditionerA>> A_inverse;
    const SmartPointer<
      const SchurComplementInverse<PreconditionerSm, PreconditionerMp>> S_inverse;
    mutable Vector<double> tmp;
  };

  template <class PreconditionerA, class PreconditionerSm, class PreconditionerMp>
  BlockSchurPreconditioner<PreconditionerA, PreconditionerSm, PreconditionerMp>::
    BlockSchurPreconditioner(
      const BlockSparseMatrix<double> &system_m,
      const InverseMatrix<SparseMatrix<double>, PreconditionerA> &A_inv,
      const SchurComplementInverse<PreconditionerSm, PreconditionerMp> &S_inv)
    : system_matrix(&system_m), A_inverse(&A_inv), S_inverse(&S_inv),
      tmp(system_matrix->block(1, 1).m())
  {
  }

  template <class PreconditionerA, class PreconditionerSm, class PreconditionerMp>
  void BlockSchurPreconditioner<PreconditionerA, PreconditionerSm, PreconditionerMp>::vmult(
    BlockVector<double> &dst, const BlockVector<double> &src) const
  {
    A_inverse->vmult(dst.block(0), src.block(0));
    system_matrix->block(1, 0).residual(tmp, dst.block(0), src.block(1));
    tmp *= -1;
    S_inverse->vmult(dst.block(1), tmp);
  }

  // @sect3{The time-dependent Navier-Stokes class template}
  template <int dim>
  class NavierStokes
  {
  public:
    NavierStokes(const unsigned int degree);
    void run();

  private:
    void setup();
    void assemble(bool assemble_lhs);

    std::pair<unsigned int, double> solve_linear_system(bool update_preconditioner);
    void output_results(const unsigned int index) const;
    void process_solution(std::ofstream& out) const;
    const ConstraintMatrix &get_constraints() const;

    double viscosity;
    double gamma;
    const unsigned int degree;
    std::vector<types::global_dof_index> dofs_per_block;

    Triangulation<dim> triangulation;
    FESystem<dim> fe;
    DoFHandler<dim> dof_handler;
    QGauss<dim> quadrature_formula;
    QGauss<dim-1> face_quadrature_formula;

    ConstraintMatrix zero_constraints;
    ConstraintMatrix nonzero_constraints;

    BlockSparsityPattern sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;
    // We need both velocity mass and pressure mass, so we use a block sparse matrix to store it.
    BlockSparseMatrix<double> mass_matrix;

    BlockVector<double> solution;
    BlockVector<double> solution_increment;
    BlockVector<double> system_rhs;

    Time time;
    mutable TimerOutput timer;

    // We use shared pointers for all the preconditioning-related stuff
    std::shared_ptr<ApproximateMassSchur> approximate_Sm;
    std::shared_ptr<PreconditionIdentity> preconditioner_Sm;
    std::shared_ptr<InverseMatrix<ApproximateMassSchur, PreconditionIdentity>> Sm_inverse;

    std::shared_ptr<SparseILU<double>> preconditioner_Mp;
    std::shared_ptr<InverseMatrix<SparseMatrix<double>, SparseILU<double>>> Mp_inverse;

    std::shared_ptr<SchurComplementInverse<PreconditionIdentity,
      SparseILU<double>>> S_inverse;

    std::shared_ptr<typename InnerPreconditioner<dim>::type> preconditioner_A;
    std::shared_ptr<InverseMatrix<SparseMatrix<double>, 
      typename InnerPreconditioner<dim>::type>> A_inverse;

    std::shared_ptr<BlockSchurPreconditioner
      <typename InnerPreconditioner<dim>::type, PreconditionIdentity, SparseILU<double>>> preconditioner;
  };

  // @sect4{NavierStokes::NavierStokes}
  template <int dim>
  NavierStokes<dim>::NavierStokes(const unsigned int degree)
    : viscosity(0.001),
      gamma(1),
      degree(degree),
      triangulation(Triangulation<dim>::maximum_smoothing),
      fe(FE_Q<dim>(degree + 1), dim, FE_Q<dim>(degree), 1),
      dof_handler(triangulation),
      quadrature_formula(degree+2),
      face_quadrature_formula(degree+2),
      time(1e-2, 1e-3),
      timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)
  {
  }

  // @sect4{NavierStokes::setup}
  template <int dim>
  void NavierStokes<dim>::setup()
  {
    timer.enter_subsection("Setup system");
    dof_handler.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(dof_handler);

    // We renumber the components to have all velocity DoFs come before
    // the pressure DoFs to be able to split the solution vector in two blocks
    // which are separately accessed
    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    DoFRenumbering::component_wise(dof_handler, block_component);

    dofs_per_block.resize(2);
    DoFTools::count_dofs_per_block(
      dof_handler, dofs_per_block, block_component);
    unsigned int dof_u = dofs_per_block[0];
    unsigned int dof_p = dofs_per_block[1];

    // The Dirichlet boundary condition is applied to boundaries 0 and 1.
    FEValuesExtractors::Vector velocities(0);
    {
      nonzero_constraints.clear();

      DoFTools::make_hanging_node_constraints(dof_handler, nonzero_constraints);
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               BoundaryValues<dim>(),
                                               nonzero_constraints,
                                               fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               1,
                                               BoundaryValues<dim>(),
                                               nonzero_constraints,
                                               fe.component_mask(velocities));
    }
    nonzero_constraints.close();

    {
      zero_constraints.clear();

      DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints);
      VectorTools::interpolate_boundary_values(
        dof_handler,
        0,
        Functions::ZeroFunction<dim>(dim + 1),
        zero_constraints,
        fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(
        dof_handler,
        1,
        Functions::ZeroFunction<dim>(dim + 1),
        zero_constraints,
        fe.component_mask(velocities));
    }
    zero_constraints.close();

    std::cout << "   Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "   Number of vertices: " << triangulation.n_vertices()
              << std::endl
              << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << " (" << dof_u << '+' << dof_p << ')' << std::endl;

    BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, nonzero_constraints);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
    mass_matrix.reinit(sparsity_pattern);

    solution.reinit(dofs_per_block);
    solution_increment.reinit(dofs_per_block);
    system_rhs.reinit(dofs_per_block);

    timer.leave_subsection();
  }

  // @sect4{NavierStokes::setup}

  // A helper function to determine which constrint to use based on the current timestep
  template <int dim>
  const ConstraintMatrix &NavierStokes<dim>::get_constraints() const
  {
    return time.get_timestep() == 0 ? nonzero_constraints : zero_constraints;
  }

  // @sect4{NavierStokes::assemble}

  // Note that we only need to assemble the LHS for twice: once with the nonzero constraint
  // and once for zero constraint. But we must assemble the RHS at every time step.
  template <int dim>
  void NavierStokes<dim>::assemble(bool assemble_lhs)
  {
    timer.enter_subsection("Assemble system");
    if (assemble_lhs)
      {
        system_matrix = 0;
        mass_matrix = 0;
      }

    system_rhs = 0;

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();

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

    typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                   endc = dof_handler.end();

    for (; cell != endc; ++cell)
      {
        fe_values.reinit(cell);

        local_matrix = 0;
        local_rhs = 0;
        local_mass_matrix = 0;

        fe_values[velocities].get_function_values(solution,
                                                  current_velocity_values);

        fe_values[velocities].get_function_gradients(
          solution, current_velocity_gradients);

        fe_values[velocities].get_function_divergences(
          solution, current_velocity_divergences);

        fe_values[pressure].get_function_values(solution,
                                                current_pressure_values);

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
                if (assemble_lhs)
                  {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        // $LHS = a((u, p), (v, q))*dt + m(u, v)
                        //      = ((grad_v, nu*grad_u) - (div_v, p) - (q, div_u))*dt +
                        //        m(u, v)$ plus Grad-Div term.
                        local_matrix(i, j) +=
                          ((viscosity *
                             scalar_product(grad_phi_u[j], grad_phi_u[i]) -
                           div_phi_u[i] * phi_p[j] - phi_p[i] * div_phi_u[j] +
                           gamma*div_phi_u[j]*div_phi_u[i]) *
                            time.get_delta_t() +
                           phi_u[i] * phi_u[j]) *
                          fe_values.JxW(q);
                        // Besides the velocity and pressure mass matrices, we also
                        // assemble $B^T$ and $B$ into the block mass matrix for convenience
                        // because we need to use them to compute the Schur complement.
                        // As a result $M = [M_u, B^T; B, M_p]$.
                        local_mass_matrix(i, j) +=
                          (phi_u[i] * phi_u[j] + phi_p[i] * phi_p[j] -
                           div_phi_u[i] * phi_p[j] - phi_p[i] * div_phi_u[j]) *
                          fe_values.JxW(q);
                      }
                  }
                // $RHS = - dt*[ a((u_prev, p_prev), (v, q)) + c(u_prev; u_prev, v)]$
                // plus Grad-Div term.
                local_rhs(i) -=
                  (viscosity * scalar_product(current_velocity_gradients[q],
                                              grad_phi_u[i]) -
                    current_velocity_divergences[q] * phi_p[i] -
                    current_pressure_values[q] * div_phi_u[i] +
                    current_velocity_gradients[q] * current_velocity_values[q] *
                      phi_u[i] +
                    gamma * current_velocity_divergences[q] * div_phi_u[i]) *
                  fe_values.JxW(q) * time.get_delta_t();
              }
          }

        cell->get_dof_indices(local_dof_indices);

        const ConstraintMatrix &constraints_used = get_constraints();

        if (assemble_lhs)
          {
            constraints_used.distribute_local_to_global(local_matrix,
                                                        local_rhs,
                                                        local_dof_indices,
                                                        system_matrix,
                                                        system_rhs);
            constraints_used.distribute_local_to_global(local_mass_matrix,
                                                        local_dof_indices,
                                                        mass_matrix);
          }
        else
          {
            constraints_used.distribute_local_to_global(
              local_rhs, local_dof_indices, system_rhs);
          }
      }
    timer.leave_subsection();
  }

  // @sect4{NavierStokes::solve_linear_system}

  // Only updates the preconditioners when we assemble the LHS of the system.
  template <int dim>
  std::pair<unsigned int, double> NavierStokes<dim>::solve_linear_system(bool update_precondition)
  {
    const ConstraintMatrix &constraints_used = get_constraints();

    if (update_precondition)
    {
      timer.enter_subsection("Precondition linear system");

      preconditioner.reset();
      A_inverse.reset();
      preconditioner_A.reset();
      S_inverse.reset();
      Mp_inverse.reset();
      preconditioner_Mp.reset();
      Sm_inverse.reset();
      preconditioner_Sm.reset();
      approximate_Sm.reset();

      approximate_Sm.reset(new ApproximateMassSchur(mass_matrix));
      preconditioner_Sm.reset(new PreconditionIdentity());
      Sm_inverse.reset(new InverseMatrix<ApproximateMassSchur, PreconditionIdentity>
        (*approximate_Sm, *preconditioner_Sm));
      preconditioner_Mp.reset(new SparseILU<double>());
      preconditioner_Mp->initialize(mass_matrix.block(1,1));
      Mp_inverse.reset(new InverseMatrix<SparseMatrix<double>, SparseILU<double>>
        (mass_matrix.block(1,1), *preconditioner_Mp)); 
      S_inverse.reset(new SchurComplementInverse<PreconditionIdentity,
        SparseILU<double>>(gamma, viscosity, time.get_delta_t(), *Sm_inverse, *Mp_inverse));
      preconditioner_A.reset(new typename InnerPreconditioner<dim>::type());
      preconditioner_A->initialize(system_matrix.block(0,0),
        typename InnerPreconditioner<dim>::type::AdditionalData());

      A_inverse.reset(new InverseMatrix<SparseMatrix<double>,
        typename InnerPreconditioner<dim>::type>(system_matrix.block(0,0), *preconditioner_A)); 
      preconditioner.reset(new BlockSchurPreconditioner<
        typename InnerPreconditioner<dim>::type, PreconditionIdentity,
        SparseILU<double>>(system_matrix, *A_inverse, *S_inverse));

      timer.leave_subsection();
    }

    // Solve with GMRES solver.
    timer.enter_subsection("Solve linear system");
    SolverControl solver_control(system_matrix.m(),
                                 1e-8 * system_rhs.l2_norm());
    GrowingVectorMemory<BlockVector<double>> vector_memory;
    SolverGMRES<BlockVector<double>>::AdditionalData gmres_data;
    gmres_data.max_n_tmp_vectors = 100;
    SolverGMRES<BlockVector<double>> gmres(
      solver_control, vector_memory, gmres_data);
    gmres.solve(system_matrix, solution_increment, system_rhs, *preconditioner);

    constraints_used.distribute(solution_increment);
    timer.leave_subsection();

    return {solver_control.last_step(), solver_control.last_value()};
  }

  // @sect4{NavierStokes::run}

  template <int dim>
  void NavierStokes<dim>::run()
  {
    create_triangulation(triangulation);
    triangulation.refine_global(2);
    setup();

    std::ofstream out("grid.eps");
    GridOut grid_out;
    grid_out.write_eps(triangulation, out);

    std::ofstream out2("force.txt");
    out2 << std::setw(13) << std::left << "Time/s"
      << std::setw(13) << std::left << " Drag" << std::setw(13)
      << std::left << " Lift" << std::endl;

    // In IMEX scheme we do not need to implement the Newton's method, what we need
    // to do at every time step is simple:
    // 1. Solve for the solution increment; 2. Update the solution.
    output_results(time.get_timestep());
    while (time.current() <= time.end())
      {
        std::cout << "*****************************************" << std::endl;
        std::cout << "Time = " << time.current() << std::endl;

        assemble(time.get_timestep() < 2);

        auto state = solve_linear_system(time.get_timestep() < 2);
        solution.add(1.0, solution_increment);

        // solution is distributed using nonzero_constraints all the time
        nonzero_constraints.distribute(solution);
        solution_increment = 0;

        std::cout << " FGMRES steps = " << state.first 
          << " residual = " << std::setw(6) << state.second << std::endl;

        time.increment();

        if (time.get_timestep() % 1 == 0)
        {
          output_results(time.get_timestep());
          process_solution(out2);
        }
      }

    out2.close();
  }

  // @sect4{NavierStokes::output_result}

  template <int dim>
  void NavierStokes<dim>::output_results(const unsigned int output_index) const
  {
    timer.enter_subsection("Output");
    std::cout << " Writing results..." << std::endl;
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.push_back("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.build_patches();

    std::ostringstream filename;
    filename << "Re100-"
             << Utilities::int_to_string(output_index, 6) << ".vtu";

    std::ofstream output(filename.str().c_str());
    data_out.write_vtu(output);
    timer.leave_subsection();
  }

  // @sect4{NavierStokes::process_solution}

  // This function is used to calculate the drag and lift coefficients on the cylinder.
  // We first calculate the traction of the fluid, which is nothing but the product of the
  // stress tensor and the normal of the cylindrical surface, and then integrate it along
  // the cylindrical surface and negate it.
  template <int dim>
  void NavierStokes<dim>::process_solution(std::ofstream& out) const
  {
    timer.enter_subsection("Process solution");
    
    Tensor<1, dim> force;

    FEFaceValues<dim> fe_face_values(fe,
                            face_quadrature_formula,
                            update_values | update_quadrature_points |
                            update_JxW_values | update_normal_vectors |
                            update_gradients);

    const unsigned int n_q_points = face_quadrature_formula.size();

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    std::vector<double> p(n_q_points);
    std::vector<SymmetricTensor<2, dim>> grad_sym_v(n_q_points);

    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
    {
      for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f)
      {
        if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == 1)
        {
          fe_face_values.reinit(cell, f);
          fe_face_values[pressure].get_function_values(solution, p);
          fe_face_values[velocities].get_function_symmetric_gradients(solution, grad_sym_v);
          for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const Tensor<1, dim> &N = fe_face_values.normal_vector(q);
            SymmetricTensor<2, dim> stress = -p[q]*Physics::Elasticity::StandardTensors<dim>::I
              + viscosity*grad_sym_v[q];
            force -= stress*N*fe_face_values.JxW(q);
          }
        }
      }
    }

    double drag_coef = 2*force[0]/(0.1);
    double lift_coef = 2*force[dim-1]/(0.1);

    out.precision(6);
    out.width(12);
     
    out << std::scientific << std::left << 
      time.current() << " " << drag_coef << " " << lift_coef << std::endl;

    timer.leave_subsection();
  }
}

// @sect3{main function}

int main()
{
  try
    {
      using namespace dealii;
      using namespace fluid;

      NavierStokes<2> flow(/* degree = */ 1);
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
