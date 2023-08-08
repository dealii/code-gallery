/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2022 by the deal.II authors and Jake J. Harmon
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
 * Author: Jake J. Harmon, 2022
 *
 */



#include <deal.II/base/function_parser.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_series.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/slepc_solver.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

// For parallelization (using WorkStream and Intel TBB)
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/work_stream.h>

#include "petscpc.h"

// For Error Estimation/Indication and Smoothness Indication
#include <deal.II/fe/fe_tools.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/smoothness_estimator.h>
// For refinement
#include <deal.II/grid/grid_refinement.h>

#include <fstream>
#include <iostream>
#include <memory>

namespace Operations
{
  /**
  Computes the curl-curl term needed to fill the stiffness matrix (specific to
  2-D)
  */
  double
  curlcurl(const dealii::FEValues<2> &fe_values,
           const unsigned int &       i,
           const unsigned int &       j,
           const unsigned int &       q_point)
  {
    auto gradu1_x1x2 = fe_values.shape_grad_component(i, q_point, 0);
    auto gradu2_x1x2 = fe_values.shape_grad_component(i, q_point, 1);

    auto gradv1_x1x2 = fe_values.shape_grad_component(j, q_point, 0);
    auto gradv2_x1x2 = fe_values.shape_grad_component(j, q_point, 1);
    return (gradu2_x1x2[0] - gradu1_x1x2[1]) *
           (gradv2_x1x2[0] - gradv1_x1x2[1]);
  }

  /**
  Computes the dot product of the shape functions needed to fill the mass matrix
  */
  template <int dim>
  inline double
  dot_term(const dealii::FEValues<dim> &fe_values,
           const unsigned int &         i,
           const unsigned int &         j,
           const unsigned int &         q_point)
  {
    double output = 0.0;
    for (unsigned int comp = 0; comp < dim; ++comp)
      {
        output += fe_values.shape_value_component(i, q_point, comp) *
                  fe_values.shape_value_component(j, q_point, comp);
      }
    return output;
  }
} // namespace Operations
/**
The Structures namespace includes the necessary functions for constructing two
examples problems, the so-called "Standard waveguide" (width/height = 2), and
the L-domain waveguide
*/
namespace Structures
{
  using namespace dealii;

  void
  create_L_waveguide(Triangulation<2> &triangulation, const double &scaling)
  {
    const unsigned int dim = 2;

    const std::vector<Point<2>> vertices = {{scaling * 0.0, scaling * 0.0},
                                            {scaling * 0.5, scaling * 0.0},
                                            {scaling * 0.0, scaling * 0.5},
                                            {scaling * 0.5, scaling * 0.5},
                                            {scaling * 0.0, scaling * 1.0},
                                            {scaling * 0.5, scaling * 1.0},
                                            {scaling * 1.0, scaling * 0.5},
                                            {scaling * 1.0, scaling * 1.0}};

    const std::vector<std::array<int, GeometryInfo<dim>::vertices_per_cell>>
      cell_vertices = {{{0, 1, 2, 3}}, {{2, 3, 4, 5}}, {{3, 6, 5, 7}}};
    const unsigned int         n_cells = cell_vertices.size();
    std::vector<CellData<dim>> cells(n_cells, CellData<dim>());
    for (unsigned int i = 0; i < n_cells; ++i)
      {
        for (unsigned int j = 0; j < cell_vertices[i].size(); ++j)
          cells[i].vertices[j] = cell_vertices[i][j];
        cells[i].material_id = 0;
      }
    triangulation.create_triangulation(vertices, cells, SubCellData());
    triangulation.refine_global(1);
  }


  void
  create_standard_waveguide(Triangulation<2> &triangulation,
                            const double &    scaling)
  {
    const unsigned int dim = 2;

    const std::vector<Point<2>> vertices = {{scaling * 0.0, scaling * 0.0},
                                            {scaling * 0.6, scaling * 0.0},
                                            {scaling * 0.0, scaling * 0.3},
                                            {scaling * 0.6, scaling * 0.3}};

    const std::vector<std::array<int, GeometryInfo<dim>::vertices_per_cell>>
                               cell_vertices = {{{0, 1, 2, 3}}};
    const unsigned int         n_cells       = cell_vertices.size();
    std::vector<CellData<dim>> cells(n_cells, CellData<dim>());
    for (unsigned int i = 0; i < n_cells; ++i)
      {
        for (unsigned int j = 0; j < cell_vertices[i].size(); ++j)
          cells[i].vertices[j] = cell_vertices[i][j];
        cells[i].material_id = 0;
      }
    triangulation.create_triangulation(vertices, cells, SubCellData());
    triangulation.refine_global(0);
  }
} // namespace Structures
/**
The Maxwell namespace includes all of the classes for solving the Maxwell
eigenvalue problem
*/
namespace Maxwell
{
  using namespace dealii;

  /*
  The "Base" class provides the universal functionality of any eigensolver,
  namely the parameters for the problem, an underlying triangulation, and
  functionality for setting the refinement cycle and to output the solution.

  In this case, and for any future class, the use of raw pointers (as opposed to
  "smart" pointers) indicates a lack of ownership. Specifically, the
  triangulation raw pointer is pointing to a triangulation that is owned (and
  created) elsewhere.
  */
  template <int dim>
  class Base
  {
  public:
    Base(const std::string &prm_file, Triangulation<dim> &coarse_grid);

    virtual unsigned int
    solve_problem() = 0; // Implemented by a derived class
    virtual void
    set_refinement_cycle(const unsigned int cycle);

    virtual void
    output_solution() = 0; // Implemented by a derived class


  protected:
    Triangulation<dim> *              triangulation;
    unsigned int                      refinement_cycle = 0;
    std::unique_ptr<ParameterHandler> parameters;
    unsigned int                      n_eigenpairs = 1;
    double                            target       = 0.0;
    unsigned int                      eigenpair_selection_scheme;
    unsigned int                      max_cycles       = 0;
    ompi_communicator_t *             mpi_communicator = PETSC_COMM_SELF;
  };

  /**
  Reads in the parameters file and the triangulation
  */
  template <int dim>
  Base<dim>::Base(const std::string &prm_file, Triangulation<dim> &coarse_grid)
    : triangulation(&coarse_grid)
    , parameters(std::make_unique<ParameterHandler>())
  {
    parameters->declare_entry(
      "Eigenpair selection scheme",
      "1",
      Patterns::Integer(0, 1),
      "The type of eigenpairs to find (0 - smallest, 1 - target)");
    parameters->declare_entry("Number of eigenvalues/eigenfunctions",
                              "1",
                              Patterns::Integer(0, 100),
                              "The number of eigenvalues/eigenfunctions "
                              "to be computed.");
    parameters->declare_entry("Target eigenvalue",
                              "1",
                              Patterns::Anything(),
                              "The target eigenvalue (if scheme == 1)");

    parameters->declare_entry("Cycles number",
                              "1",
                              Patterns::Integer(0, 1500),
                              "The number of cycles in refinement");
    parameters->parse_input(prm_file);

    eigenpair_selection_scheme =
      parameters->get_integer("Eigenpair selection scheme");

    // The project currently only supports selection by a target eigenvalue.
    // Furthermore, only one eigenpair can be computed at a time.
    assert(eigenpair_selection_scheme == 1 &&
           "Selection by a target is the only currently supported option!");
    n_eigenpairs =
      parameters->get_integer("Number of eigenvalues/eigenfunctions");
    assert(
      n_eigenpairs == 1 &&
      "Only the computation of a single eigenpair is currently supported!");

    target     = parameters->get_double("Target eigenvalue");
    max_cycles = parameters->get_integer("Cycles number");
    if (eigenpair_selection_scheme == 1)
      n_eigenpairs = 1;
  }

  template <int dim>
  void
  Base<dim>::set_refinement_cycle(const unsigned int cycle)
  {
    refinement_cycle = cycle;
  }

  /**
  Provides the central solver (derived from the base class). Virtual inheritance
  is crucial to eliminate compiler ambiguity in the case of the
  DualWeightedResidual.
  */
  template <int dim>
  class EigenSolver : public virtual Base<dim>
  {
  public:
    EigenSolver(const std::string & prm_file,
                Triangulation<dim> &coarse_grid,
                const unsigned int &minimum_degree,
                const unsigned int &maximum_degree,
                const unsigned int &starting_degree);

    virtual unsigned int
    solve_problem() override;

    virtual unsigned int
    n_dofs() const;

    template <class SolverType>
    void
    initialize_eigensolver(SolverType &eigensolver);

    virtual void
    setup_system();

    virtual void
    assemble_system();

  protected:
    const std::unique_ptr<hp::FECollection<dim>> fe_collection;
    std::unique_ptr<hp::QCollection<dim>>        quadrature_collection;
    std::unique_ptr<hp::QCollection<dim - 1>>    face_quadrature_collection;
    DoFHandler<dim>                              dof_handler;
    const unsigned int                           max_degree, min_degree;
    // for the actual solution
    std::unique_ptr<std::vector<PETScWrappers::MPI::Vector>> eigenfunctions;
    std::unique_ptr<std::vector<double>>                     eigenvalues;
    Vector<double>                                           solution;

    double *
    get_lambda_h();

    Vector<double> *
    get_solution();

    void
    convert_solution();

  private:
    AffineConstraints<double>   constraints;
    PETScWrappers::SparseMatrix stiffness_matrix, mass_matrix;
  };

  /**
  Typical constructor. Executes the constructor for the base class and creates
  the unique pointers for the fe_collection, quadrature_collection, etc.
  */
  template <int dim>
  EigenSolver<dim>::EigenSolver(const std::string & prm_file,
                                Triangulation<dim> &triangulation,
                                const unsigned int &minimum_degree,
                                const unsigned int &maximum_degree,
                                const unsigned int &starting_degree)
    : Base<dim>(prm_file, triangulation)
    , fe_collection(std::make_unique<hp::FECollection<dim>>())
    , quadrature_collection(std::make_unique<hp::QCollection<dim>>())
    , face_quadrature_collection(std::make_unique<hp::QCollection<dim - 1>>())
    , dof_handler(triangulation)
    , max_degree(maximum_degree)
    , min_degree(minimum_degree)
    , eigenfunctions(
        std::make_unique<std::vector<PETScWrappers::MPI::Vector>>())
    , eigenvalues(std::make_unique<std::vector<double>>())
  {
    for (unsigned int degree = min_degree; degree <= max_degree; ++degree)
      {
        fe_collection->push_back(FE_Nedelec<dim>(degree - 1));
        // Generate quadrature collection with sorted quadrature weights
        const QGauss<dim>  quadrature(degree + 1);
        const QSorted<dim> sorted_quadrature(quadrature);
        quadrature_collection->push_back(sorted_quadrature);

        const QGauss<dim - 1>  face_quadrature(degree + 1);
        const QSorted<dim - 1> sorted_face_quadrature(face_quadrature);
        face_quadrature_collection->push_back(sorted_face_quadrature);
      }
    // adjust the discretization
    if (starting_degree > min_degree && starting_degree <= max_degree)
      {
        const unsigned int start_diff = starting_degree - min_degree;
        typename DoFHandler<dim>::active_cell_iterator
          cell1 = dof_handler.begin_active(),
          endc1 = dof_handler.end();
        for (; cell1 < endc1; ++cell1)
          {
            cell1->set_active_fe_index(start_diff);
          }
      }
  }

  /**
  Returns the (first) eigenvalue.
  TODO: Generalize to arbitrary, valid eigenvalues
  */
  template <int dim>
  double *
  EigenSolver<dim>::get_lambda_h()
  {
    return &(*eigenvalues)[0];
  }

  /**
  Returns the (first) eigenvector.
  TODO: Generalize to arbitrary
  */
  template <int dim>
  Vector<double> *
  EigenSolver<dim>::get_solution()
  {
    return &solution;
  }

  /**
  Temporary helper function for copying the solution vector
  */
  template <int dim>
  void
  EigenSolver<dim>::convert_solution()
  {
    solution.reinit((*eigenfunctions)[0].size());
    for (unsigned int i = 0; i < solution.size(); ++i)
      solution[i] = (*eigenfunctions)[0][i];
  }

  /**
  Initializes the eigensolver according the selection scheme in the parameters
  file. Additionally, applies the necessary problem type (GHEP) and introduces
  the Shift-and-Invert spectrum transformation based on the specified target
  value
  */
  template <int dim>
  template <class SolverType>
  void
  EigenSolver<dim>::initialize_eigensolver(SolverType &eigensolver)
  {
    // From the parameters class, initialize the eigensolver...
    switch (this->eigenpair_selection_scheme)
      {
        case 1:
          eigensolver.set_which_eigenpairs(EPS_TARGET_MAGNITUDE);
          // eigensolver.set_target_eigenvalue(this->target);
          break;
        default:
          eigensolver.set_which_eigenpairs(EPS_SMALLEST_MAGNITUDE);

          break;
      }
    eigensolver.set_problem_type(EPS_GHEP);
    // apply a Shift-Invert spectrum transformation

    double shift_scalar = this->parameters->get_double("Target eigenvalue");
    //		//For the shift-and-invert transformation
    SLEPcWrappers::TransformationShiftInvert::AdditionalData additional_data(
      shift_scalar);
    SLEPcWrappers::TransformationShiftInvert spectral_transformation(
      this->mpi_communicator, additional_data);

    eigensolver.set_transformation(spectral_transformation);
    eigensolver.set_target_eigenvalue(this->target);
  }

  /**
  Solves the eigenvalue problem and applies the constraints to the
  eigenfunctions
  */
  template <int dim>
  unsigned int
  EigenSolver<dim>::solve_problem()
  {
    setup_system();
    assemble_system();

    SolverControl                    solver_control(dof_handler.n_dofs() * 10,
                                 5.0e-8,
                                 false,
                                 false);
    SLEPcWrappers::SolverKrylovSchur eigensolver(solver_control,
                                                 this->mpi_communicator);

    initialize_eigensolver(eigensolver);

    // solve the problem
    eigensolver.solve(stiffness_matrix,
                      mass_matrix,
                      *eigenvalues,
                      *eigenfunctions,
                      eigenfunctions->size());
    for (auto &entry : *eigenfunctions)
      {
        constraints.distribute(entry);
      }
    convert_solution();

    return solver_control.last_step();
  }

  template <int dim>
  unsigned int
  EigenSolver<dim>::n_dofs() const
  {
    return dof_handler.n_dofs();
  }

  /**
  Distributes the degrees of freedom and makes the necessary hanging_node
  constraints, which includes the constraints for non-uniform $p$, and for the
  Dirichlet boundary.
  */
  template <int dim>
  void
  EigenSolver<dim>::setup_system()
  {
    dof_handler.distribute_dofs(*fe_collection);
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
    constraints.close();

    eigenfunctions->resize(this->n_eigenpairs);
    eigenvalues->resize(this->n_eigenpairs);

    IndexSet eigenfunction_index_set = dof_handler.locally_owned_dofs();

    for (auto &entry : *eigenfunctions)
      {
        entry.reinit(eigenfunction_index_set, MPI_COMM_WORLD);
      }
  }

  /**
  Fills the mass and stiffness matrices
  */
  template <int dim>
  void
  EigenSolver<dim>::assemble_system()
  {
    hp::FEValues<dim> hp_fe_values(*fe_collection,
                                   *quadrature_collection,
                                   update_values | update_gradients |
                                     update_quadrature_points |
                                     update_JxW_values);
    // Prep the system matrices for the solution
    stiffness_matrix.reinit(dof_handler.n_dofs(),
                            dof_handler.n_dofs(),
                            dof_handler.max_couplings_between_dofs());
    mass_matrix.reinit(dof_handler.n_dofs(),
                       dof_handler.n_dofs(),
                       dof_handler.max_couplings_between_dofs());

    FullMatrix<double> cell_stiffness_matrix, cell_mass_matrix;
    std::vector<types::global_dof_index> local_dof_indices;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;

        cell_stiffness_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_stiffness_matrix = 0;

        cell_mass_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_mass_matrix = 0;

        hp_fe_values.reinit(cell);

        const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();

        for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points;
             ++q_point)
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    // Note that (in general) the Nedelec element is not
                    // primitive, namely that the shape functions are vectorial
                    // with components in more than one direction

                    cell_stiffness_matrix(i, j) +=
                      Operations::curlcurl(fe_values, i, j, q_point) *
                      fe_values.JxW(q_point);

                    cell_mass_matrix(i, j) +=
                      (Operations::dot_term(fe_values, i, j, q_point)) *
                      fe_values.JxW(q_point);
                  }
              }
            local_dof_indices.resize(dofs_per_cell);
            cell->get_dof_indices(local_dof_indices);
          }

        constraints.distribute_local_to_global(cell_stiffness_matrix,
                                               local_dof_indices,
                                               stiffness_matrix);
        constraints.distribute_local_to_global(cell_mass_matrix,
                                               local_dof_indices,
                                               mass_matrix);
      }
    stiffness_matrix.compress(VectorOperation::add);
    mass_matrix.compress(VectorOperation::add);

    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
      if (constraints.is_constrained(i))
        {
          stiffness_matrix.set(i, i, 10000.0);
          mass_matrix.set(i, i, 1);
        }
    // since we have just set individual elements, we need the following
    stiffness_matrix.compress(VectorOperation::insert);
    mass_matrix.compress(VectorOperation::insert);
  }

  /**
  The main PrimalSolver, which is derived from the EigenSolver class. Provides a
  limited amount of additional functionality
  */
  template <int dim>
  class PrimalSolver : public EigenSolver<dim>
  {
  public:
    PrimalSolver(const std::string & prm_file,
                 Triangulation<dim> &triangulation,
                 const unsigned int &min_degree,
                 const unsigned int &max_degree,
                 const unsigned int &starting_degree);

    virtual void
    output_solution()
      override; // Implements the output solution of the base class...
    virtual unsigned int
    n_dofs() const override;
  };

  template <int dim>
  PrimalSolver<dim>::PrimalSolver(const std::string & prm_file,
                                  Triangulation<dim> &triangulation,
                                  const unsigned int &min_degree,
                                  const unsigned int &max_degree,
                                  const unsigned int &starting_degree)
    : Base<dim>(prm_file, triangulation)
    , EigenSolver<dim>(prm_file,
                       triangulation,
                       min_degree,
                       max_degree,
                       starting_degree)
  {}

  /**
  Outputs the first eigenpair (based on the target eigenvalue)
  TODO: Generalize to multiple eigenpairs
  */
  template <int dim>
  void
  PrimalSolver<dim>::output_solution()
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(this->dof_handler);
    Vector<double> fe_degrees(this->triangulation->n_active_cells());
    for (const auto &cell : this->dof_handler.active_cell_iterators())
      fe_degrees(cell->active_cell_index()) =
        (*this->fe_collection)[cell->active_fe_index()].degree;
    data_out.add_data_vector(fe_degrees, "fe_degree");
    data_out.add_data_vector((*this->eigenfunctions)[0],
                             std::string("eigenfunction_no_") +
                               Utilities::int_to_string(0));

    std::cout << "Eigenvalue: " << (*this->eigenvalues)[0]
              << " NDoFs: " << this->dof_handler.n_dofs() << std::endl;
    std::ofstream eigenvalues_out(
      "eigenvalues-" + std::to_string(this->refinement_cycle) + ".txt");

    eigenvalues_out << std::setprecision(20) << (*this->eigenvalues)[0] << " "
                    << this->dof_handler.n_dofs() << std::endl;

    eigenvalues_out.close();


    data_out.build_patches();
    std::ofstream output("eigenvectors-" +
                         std::to_string(this->refinement_cycle) + ".vtu");
    data_out.write_vtu(output);
  }

  template <int dim>
  unsigned int
  PrimalSolver<dim>::n_dofs() const
  {
    return EigenSolver<dim>::n_dofs();
  }

  // Note, that at least for the demonstrated problem (i.e., a Hermitian problem
  // and eigenvalue QoI), the dual problem is identical to the primal problem;
  // however, it is convenient to separate them in this manner (e.g., for
  // considering functionals of the eigenfunction).
  template <int dim>
  class DualSolver : public EigenSolver<dim>
  {
  public:
    DualSolver(const std::string & prm_file,
               Triangulation<dim> &triangulation,
               const unsigned int &min_degree,
               const unsigned int &max_degree,
               const unsigned int &starting_degree);
  };

  template <int dim>
  DualSolver<dim>::DualSolver(const std::string & prm_file,
                              Triangulation<dim> &triangulation,
                              const unsigned int &min_degree,
                              const unsigned int &max_degree,
                              const unsigned int &starting_degree)
    : Base<dim>(prm_file, triangulation)
    , EigenSolver<dim>(prm_file,
                       triangulation,
                       min_degree,
                       max_degree,
                       starting_degree)
  {}

} // namespace Maxwell
/**
The second major namespace, which includes all the classes for error
estimation and error indication.
*/
namespace ErrorIndicators
{
  using namespace Maxwell;

  /**
  The DualWeightedResidual is derived from the PrimalSolver and DualSolver. In
  this case, the DualSolver is taken with a finite element space with shape
  functions of one polynomial degree higher.
  */
  template <int dim, bool report_dual>
  class DualWeightedResidual : public PrimalSolver<dim>, public DualSolver<dim>
  {
  public:
    void
    output_eigenvalue_data(std::ofstream &os);
    void
    output_qoi_error_estimates(std::ofstream &os);

    std::string
    name() const
    {
      return "DWR";
    }
    DualWeightedResidual(const std::string & prm_file,
                         Triangulation<dim> &triangulation,
                         const unsigned int &min_primal_degree,
                         const unsigned int &max_primal_degree,
                         const unsigned int &starting_primal_degree);

    virtual unsigned int
    solve_problem() override;

    virtual void
    output_solution() override;

    virtual unsigned int
    n_dofs() const override;

    void
    estimate_error(Vector<double> &error_indicators);

    DoFHandler<dim> *
    get_DoFHandler();

    DoFHandler<dim> *
    get_primal_DoFHandler();

    DoFHandler<dim> *
    get_dual_DoFHandler();

    hp::FECollection<dim> *
    get_FECollection();

    hp::FECollection<dim> *
    get_primal_FECollection();

    std::unique_ptr<std::vector<PETScWrappers::MPI::Vector>> &
    get_eigenfunctions();

    std::unique_ptr<std::vector<PETScWrappers::MPI::Vector>> &
    get_primal_eigenfunctions();

    std::unique_ptr<std::vector<double>> &
    get_primal_eigenvalues();

    std::unique_ptr<std::vector<double>> &
    get_dual_eigenvalues();

    void
    synchronize_discretization();

    unsigned int
    get_max_degree()
    {
      return PrimalSolver<dim>::fe_collection->max_degree();
    }
    double qoi_error_estimate = 0;

  private:
    void
    embed(const DoFHandler<dim> &          dof1,
          const DoFHandler<dim> &          dof2,
          const AffineConstraints<double> &constraints,
          const Vector<double> &           solution,
          Vector<double> &                 u2);

    void
    extract(const DoFHandler<dim> &          dof1,
            const DoFHandler<dim> &          dof2,
            const AffineConstraints<double> &constraints,
            const Vector<double> &           solution,
            Vector<double> &                 u2);



    /*The following FEValues objects are unique_ptrs to 1) avoid default
    constructors for these objects, and 2) automate memory management*/
    std::unique_ptr<hp::FEValues<dim>>        cell_hp_fe_values;
    std::unique_ptr<hp::FEFaceValues<dim>>    face_hp_fe_values;
    std::unique_ptr<hp::FEFaceValues<dim>>    face_hp_fe_values_neighbor;
    std::unique_ptr<hp::FESubfaceValues<dim>> subface_hp_fe_values;

    std::unique_ptr<hp::FEValues<dim>>     cell_hp_fe_values_forward;
    std::unique_ptr<hp::FEFaceValues<dim>> face_hp_fe_values_forward;
    std::unique_ptr<hp::FEFaceValues<dim>> face_hp_fe_values_neighbor_forward;
    std::unique_ptr<hp::FESubfaceValues<dim>> subface_hp_fe_values_forward;
    using FaceIntegrals =
      typename std::map<typename DoFHandler<dim>::face_iterator, double>;

    unsigned int
    solve_primal_problem();

    unsigned int
    solve_dual_problem();

    void
    normalize_solutions(Vector<double> &primal_solution,
                        Vector<double> &dual_weights);

    double
    get_global_QoI_error(Vector<double> &dual_solution,
                         Vector<double> &error_indicators);

    void
    initialize_error_estimation_data();

    void
    estimate_on_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      const Vector<double> &                                primal_solution,
      const Vector<double> &                                dual_weights,
      const double &                                        lambda_h,
      Vector<double> &                                      error_indicators,
      FaceIntegrals &                                       face_integrals);

    void
    integrate_over_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      const Vector<double> &                                primal_solution,
      const Vector<double> &                                dual_weights,
      const double &                                        lambda_h,
      Vector<double> &                                      error_indicators);

    void
    integrate_over_regular_face(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      const unsigned int &                                  face_no,
      const Vector<double> &                                primal_solution,
      const Vector<double> &                                dual_weights,
      FaceIntegrals &                                       face_integrals);

    void
    integrate_over_irregular_face(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      const unsigned int &                                  face_no,
      const Vector<double> &                                primal_solution,
      const Vector<double> &                                dual_weights,
      FaceIntegrals &                                       face_integrals);
  };

  /**
  Basic constructor, also initializes the unique pointers for evaluating the
  cell and edge residuals in the error estimate.
  */
  template <int dim, bool report_dual>
  DualWeightedResidual<dim, report_dual>::DualWeightedResidual(
    const std::string & prm_file,
    Triangulation<dim> &triangulation,
    const unsigned int &min_primal_degree,
    const unsigned int &max_primal_degree,
    const unsigned int &starting_primal_degree)
    : Base<dim>(prm_file, triangulation)
    , PrimalSolver<dim>(prm_file,
                        triangulation,
                        min_primal_degree,
                        max_primal_degree,
                        starting_primal_degree)
    , DualSolver<dim>(prm_file,
                      triangulation,
                      min_primal_degree + 1,
                      max_primal_degree + 1,
                      starting_primal_degree + 1)
  {
    initialize_error_estimation_data();
  }

  /**
  If we are "reporting" the dual solution (e.g., for the purposes of smoothness
  estimation), we must decide which dof_handler to provide.
  */
  template <int dim, bool report_dual>
  DoFHandler<dim> *
  DualWeightedResidual<dim, report_dual>::get_DoFHandler()
  {
    if (!report_dual)
      return &(PrimalSolver<dim>::dof_handler);
    else
      return &(DualSolver<dim>::dof_handler);
  }

  // See above function, but to specifically output the primal DoFHandler...
  template <int dim, bool report_dual>
  DoFHandler<dim> *
  DualWeightedResidual<dim, report_dual>::get_primal_DoFHandler()
  {
    return &(PrimalSolver<dim>::dof_handler);
  }

  // See above function, but for the FECollection
  template <int dim, bool report_dual>
  hp::FECollection<dim> *
  DualWeightedResidual<dim, report_dual>::get_FECollection()
  {
    if (!report_dual)
      return &*(PrimalSolver<dim>::fe_collection);
    else
      return &*(DualSolver<dim>::fe_collection);
  }

  // See above function, but for the primal FECollection
  template <int dim, bool report_dual>
  hp::FECollection<dim> *
  DualWeightedResidual<dim, report_dual>::get_primal_FECollection()
  {
    return &*(PrimalSolver<dim>::fe_collection);
  }

  template <int dim, bool report_dual>
  DoFHandler<dim> *
  DualWeightedResidual<dim, report_dual>::get_dual_DoFHandler()
  {
    return &(DualSolver<dim>::dof_handler);
  }

  //
  template <int dim, bool report_dual>
  std::unique_ptr<std::vector<PETScWrappers::MPI::Vector>> &
  DualWeightedResidual<dim, report_dual>::get_eigenfunctions()
  {
    if (!report_dual)
      return (PrimalSolver<dim>::eigenfunctions);
    else
      return (DualSolver<dim>::eigenfunctions);
  }

  //
  template <int dim, bool report_dual>
  std::unique_ptr<std::vector<PETScWrappers::MPI::Vector>> &
  DualWeightedResidual<dim, report_dual>::get_primal_eigenfunctions()
  {
    return (PrimalSolver<dim>::eigenfunctions);
  }

  //
  template <int dim, bool report_dual>
  std::unique_ptr<std::vector<double>> &
  DualWeightedResidual<dim, report_dual>::get_primal_eigenvalues()
  {
    return PrimalSolver<dim>::eigenvalues;
  }

  //
  template <int dim, bool report_dual>
  std::unique_ptr<std::vector<double>> &
  DualWeightedResidual<dim, report_dual>::get_dual_eigenvalues()
  {
    return DualSolver<dim>::eigenvalues;
  }

  template <int dim, bool report_dual>
  void
  DualWeightedResidual<dim, report_dual>::output_solution()
  {
    PrimalSolver<dim>::output_solution();
  }

  // Solves the primal problem
  template <int dim, bool report_dual>
  unsigned int
  DualWeightedResidual<dim, report_dual>::solve_primal_problem()
  {
    return PrimalSolver<dim>::solve_problem();
  }

  // Solves the dual problem
  template <int dim, bool report_dual>
  unsigned int
  DualWeightedResidual<dim, report_dual>::solve_dual_problem()
  {
    return DualSolver<dim>::solve_problem();
  }

  /**
  Provides the publicly accessible solve_problem function,
  which solves both the primal and dual problems
  */
  template <int dim, bool report_dual>
  unsigned int
  DualWeightedResidual<dim, report_dual>::solve_problem()
  {
    DualWeightedResidual<dim, report_dual>::solve_primal_problem();
    return DualWeightedResidual<dim, report_dual>::solve_dual_problem();
  }

  /**
  Returns the number of dofs that the primal solver requires
  */
  template <int dim, bool report_dual>
  unsigned int
  DualWeightedResidual<dim, report_dual>::n_dofs() const
  {
    return PrimalSolver<dim>::n_dofs();
  }

  /**
  This function synchronizes the expansion orders. When working with two finite
  element spaces, we must apply the modifications made to one dof_handler
  (according to the boolean report_dual) we must also make the same
  modifications to the other finite element space.
  */
  template <int dim, bool report_dual>
  void
  DualWeightedResidual<dim, report_dual>::synchronize_discretization()
  {
    /*Note: No additional checks need to be made ensuring that these operations
       are legal as these checks are made prior to entering this function (i.e.,
       if the primal attains a degree N,
        then, by construction, a degree of N+1 must be permissible for the
       dual)*/
    DoFHandler<dim> *dof1 = &(PrimalSolver<dim>::dof_handler);
    DoFHandler<dim> *dof2 = &(DualSolver<dim>::dof_handler);

    if (report_dual)
      {
        // In this case, we have modified the polynomial orders for the dual;
        // need to update the primal
        dof1 = &(DualSolver<dim>::dof_handler);
        dof2 = &(PrimalSolver<dim>::dof_handler);
      }
    typename DoFHandler<dim>::active_cell_iterator cell1 = dof1->begin_active(),
                                                   endc1 = dof1->end();
    typename DoFHandler<dim>::active_cell_iterator cell2 = dof2->begin_active();
    for (; cell1 < endc1; ++cell1, ++cell2)
      {
        cell2->set_active_fe_index(cell1->active_fe_index());
      }
  }

  /**
  Initializes the unique pointers which contain the necessary fe_values objects
  for computing the cell and edge residuals
  */
  template <int dim, bool report_dual>
  void
  DualWeightedResidual<dim, report_dual>::initialize_error_estimation_data()
  {
    // initialize the cell fe_values...
    cell_hp_fe_values = std::make_unique<hp::FEValues<dim>>(
      *DualSolver<dim>::fe_collection,
      *DualSolver<dim>::quadrature_collection,
      update_values | update_hessians | update_quadrature_points |
        update_JxW_values);
    face_hp_fe_values = std::make_unique<hp::FEFaceValues<dim>>(
      *DualSolver<dim>::fe_collection,
      *DualSolver<dim>::face_quadrature_collection,
      update_values | update_gradients | update_JxW_values |
        update_normal_vectors);
    face_hp_fe_values_neighbor = std::make_unique<hp::FEFaceValues<dim>>(
      *DualSolver<dim>::fe_collection,
      *DualSolver<dim>::face_quadrature_collection,
      update_values | update_gradients | update_JxW_values |
        update_normal_vectors);
    subface_hp_fe_values = std::make_unique<hp::FESubfaceValues<dim>>(
      *DualSolver<dim>::fe_collection,
      *DualSolver<dim>::face_quadrature_collection,
      update_gradients);
  }

  /**
  Since any scalar multiple of an eigenvector is also an eigenvector, we must
  choose some normalization strategy. For convenience in the QoI expression, the
  L2 norm is taken in this case.
  */
  template <int dim, bool report_dual>
  void
  DualWeightedResidual<dim, report_dual>::normalize_solutions(
    Vector<double> &primal_solution,
    Vector<double> &dual_weights)
  {
    double sum_primal = 0.0, sum_dual = 0.0;
    for (const auto &cell :
         DualSolver<dim>::dof_handler.active_cell_iterators())
      {
        cell_hp_fe_values->reinit(cell);

        // grab the fe_values object
        const FEValues<dim> &fe_values =
          cell_hp_fe_values->get_present_fe_values();

        std::vector<Vector<double>> cell_primal_values(
          fe_values.n_quadrature_points, Vector<double>(dim)),
          cell_dual_values(fe_values.n_quadrature_points, Vector<double>(dim));
        fe_values.get_function_values(primal_solution, cell_primal_values);
        fe_values.get_function_values(dual_weights, cell_dual_values);


        for (unsigned int p = 0; p < fe_values.n_quadrature_points; ++p)
          {
            sum_primal +=
              cell_primal_values[p] * cell_primal_values[p] * fe_values.JxW(p);
            sum_dual +=
              cell_dual_values[p] * cell_dual_values[p] * fe_values.JxW(p);
          }
      }

    primal_solution /= sqrt(sum_primal);
    dual_weights /= sqrt(sum_dual);
  }

  /**
  Serves as the main control function for estimating all of the error
  contribution estimates
  */
  template <int dim, bool report_dual>
  void
  DualWeightedResidual<dim, report_dual>::estimate_error(
    Vector<double> &error_indicators)
  {
    // The constraints could be grabbed directly, but this is simple
    AffineConstraints<double> primal_hanging_node_constraints;
    DoFTools::make_hanging_node_constraints(PrimalSolver<dim>::dof_handler,
                                            primal_hanging_node_constraints);
    primal_hanging_node_constraints.close();

    AffineConstraints<double> dual_hanging_node_constraints;
    DoFTools::make_hanging_node_constraints(DualSolver<dim>::dof_handler,
                                            dual_hanging_node_constraints);
    dual_hanging_node_constraints.close();

    // First map the primal solution to the space of the dual solution
    // This allows us to use just one set of FEValues objects (rather than one
    // set for the primal, one for dual)

    Vector<double> primal_solution(DualSolver<dim>::dof_handler.n_dofs());

    embed(PrimalSolver<dim>::dof_handler,
          DualSolver<dim>::dof_handler,
          dual_hanging_node_constraints,
          *(PrimalSolver<dim>::get_solution()),
          primal_solution);

    Vector<double> &dual_solution = *(DualSolver<dim>::get_solution());

    normalize_solutions(primal_solution, dual_solution);

    Vector<double> dual_weights(DualSolver<dim>::dof_handler.n_dofs()),
      dual_weights_interm(PrimalSolver<dim>::dof_handler.n_dofs());

    // First extract the dual solution to the space of the primal
    extract(DualSolver<dim>::dof_handler,
            PrimalSolver<dim>::dof_handler,
            primal_hanging_node_constraints,
            *(DualSolver<dim>::get_solution()),
            dual_weights_interm);

    // Now embed this back to the space of the dual solution
    embed(PrimalSolver<dim>::dof_handler,
          DualSolver<dim>::dof_handler,
          dual_hanging_node_constraints,
          dual_weights_interm,
          dual_weights);


    // Subtract this from the full dual solution
    dual_weights -= *(DualSolver<dim>::get_solution());
    dual_weights *= -1.0;

    *(DualSolver<dim>::get_solution()) -= primal_solution;

    FaceIntegrals face_integrals;
    for (const auto &cell :
         DualSolver<dim>::dof_handler.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
        face_integrals[face] = -1e20;


    for (const auto &cell :
         DualSolver<dim>::dof_handler.active_cell_iterators())
      {
        estimate_on_one_cell(cell,
                             primal_solution,
                             dual_weights,
                             *(PrimalSolver<dim>::get_lambda_h()),
                             error_indicators,
                             face_integrals);
      }
    unsigned int present_cell = 0;
    for (const auto &cell :
         DualSolver<dim>::dof_handler.active_cell_iterators())
      {
        for (const auto &face : cell->face_iterators())
          {
            Assert(face_integrals.find(face) != face_integrals.end(),
                   ExcInternalError());
            error_indicators(present_cell) -= 0.5 * face_integrals[face];
          }
        ++present_cell;
      }

    // Now, with the error indicators computed, let us produce the
    // estimate of the QoI error
    this->qoi_error_estimate =
      this->get_global_QoI_error(*(DualSolver<dim>::get_solution()),
                                 error_indicators);
    std::cout << "Estimated QoI error: " << std::setprecision(20)
              << qoi_error_estimate << std::endl;
  }


  /**
  Accumulates the error contribution estimates for one cell
  */
  template <int dim, bool report_dual>
  void
  DualWeightedResidual<dim, report_dual>::estimate_on_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    const Vector<double> &                                primal_solution,
    const Vector<double> &                                dual_weights,
    const double &                                        lambda_h,
    Vector<double> &                                      error_indicators,
    FaceIntegrals &                                       face_integrals)
  {
    integrate_over_cell(
      cell, primal_solution, dual_weights, lambda_h, error_indicators);
    for (unsigned int face_no : GeometryInfo<dim>::face_indices())
      {
        if (cell->face(face_no)->at_boundary())
          {
            face_integrals[cell->face(face_no)] = 0.0;
            continue;
          }
        if ((cell->neighbor(face_no)->has_children() == false) &&
            (cell->neighbor(face_no)->level() == cell->level()) &&
            (cell->neighbor(face_no)->index() < cell->index()))
          continue;
        if (cell->at_boundary(face_no) == false)
          if (cell->neighbor(face_no)->level() < cell->level())
            continue;
        if (cell->face(face_no)->has_children() == false)
          integrate_over_regular_face(
            cell, face_no, primal_solution, dual_weights, face_integrals);
        else
          integrate_over_irregular_face(
            cell, face_no, primal_solution, dual_weights, face_integrals);
      }
  }

  /**
  Computes the cell residual
  */
  template <int dim, bool report_dual>
  void
  DualWeightedResidual<dim, report_dual>::integrate_over_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    const Vector<double> &                                primal_solution,
    const Vector<double> &                                dual_weights,
    const double &                                        lambda_h,
    Vector<double> &                                      error_indicators)
  {
    cell_hp_fe_values->reinit(cell);
    // Grab the fe_values object
    const FEValues<dim> &fe_values = cell_hp_fe_values->get_present_fe_values();
    std::vector<std::vector<Tensor<2, dim, double>>> cell_hessians(
      fe_values.n_quadrature_points, std::vector<Tensor<2, dim, double>>(dim));
    std::vector<Vector<double>> cell_primal_values(
      fe_values.n_quadrature_points, Vector<double>(dim)),
      cell_dual_values(fe_values.n_quadrature_points, Vector<double>(dim));
    fe_values.get_function_values(primal_solution, cell_primal_values);
    fe_values.get_function_hessians(primal_solution, cell_hessians);
    fe_values.get_function_values(dual_weights, cell_dual_values);



    double sum = 0.0;
    for (unsigned int p = 0; p < fe_values.n_quadrature_points; ++p)
      {
        sum +=
          (/*x-component*/ (cell_hessians[p][1][1][0] -
                            cell_hessians[p][0][1][1]) *
             (cell_dual_values[p](0)) +
           /*y-component*/
           (cell_hessians[p][0][0][1] - cell_hessians[p][1][0][0]) *
             (cell_dual_values[p](1)) -
           lambda_h * (cell_primal_values[p](0) * cell_dual_values[p](0) +
                       cell_primal_values[p](1) * cell_dual_values[p](1))) *
          fe_values.JxW(p);
      }

    error_indicators(cell->active_cell_index()) += sum;
  }

  /**
  Computes the edge residual when there are no hanging nodes
  */
  template <int dim, bool report_dual>
  void
  DualWeightedResidual<dim, report_dual>::integrate_over_regular_face(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    const unsigned int &                                  face_no,
    const Vector<double> &                                primal_solution,
    const Vector<double> &                                dual_weights,
    FaceIntegrals &                                       face_integrals)
  {
    Assert(cell->neighbor(face_no).state() == IteratorState::valid,
           ExcInternalError());
    const unsigned int neighbor_neighbor = cell->neighbor_of_neighbor(face_no);
    const auto         neighbor          = cell->neighbor(face_no);

    const unsigned int quadrature_index =
      std::max(cell->active_fe_index(), neighbor->active_fe_index());
    face_hp_fe_values->reinit(cell, face_no, quadrature_index);
    const FEFaceValues<dim> &fe_face_values_cell =
      face_hp_fe_values->get_present_fe_values();
    std::vector<std::vector<Tensor<1, dim, double>>> cell_primal_grads(
      fe_face_values_cell.n_quadrature_points,
      std::vector<Tensor<1, dim, double>>(dim)),
      neighbor_primal_grads(fe_face_values_cell.n_quadrature_points,
                            std::vector<Tensor<1, dim, double>>(dim));
    fe_face_values_cell.get_function_gradients(primal_solution,
                                               cell_primal_grads);

    face_hp_fe_values_neighbor->reinit(neighbor,
                                       neighbor_neighbor,
                                       quadrature_index);
    const FEFaceValues<dim> &fe_face_values_cell_neighbor =
      face_hp_fe_values_neighbor->get_present_fe_values();
    fe_face_values_cell_neighbor.get_function_gradients(primal_solution,
                                                        neighbor_primal_grads);
    const unsigned int n_q_points    = fe_face_values_cell.n_quadrature_points;
    double             face_integral = 0.0;
    std::vector<Vector<double>> cell_dual_values(n_q_points,
                                                 Vector<double>(dim));
    fe_face_values_cell.get_function_values(dual_weights, cell_dual_values);
    for (unsigned int p = 0; p < n_q_points; ++p)
      {
        auto face_normal = fe_face_values_cell.normal_vector(p);

        face_integral +=
          (cell_primal_grads[p][1][0] - cell_primal_grads[p][0][1] -
           neighbor_primal_grads[p][1][0] + neighbor_primal_grads[p][0][1]) *
          (cell_dual_values[p][0] * face_normal[1] -
           cell_dual_values[p][1] * face_normal[0]) *
          fe_face_values_cell.JxW(p);
      }
    Assert(face_integrals.find(cell->face(face_no)) != face_integrals.end(),
           ExcInternalError());
    Assert(face_integrals[cell->face(face_no)] == -1e20, ExcInternalError());
    face_integrals[cell->face(face_no)] = face_integral;
  }

  /**
  Computes the residual when there are hanging nodes
  */
  template <int dim, bool report_dual>
  void
  DualWeightedResidual<dim, report_dual>::integrate_over_irregular_face(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    const unsigned int &                                  face_no,
    const Vector<double> &                                primal_solution,
    const Vector<double> &                                dual_weights,
    FaceIntegrals &                                       face_integrals)
  {
    const typename DoFHandler<dim>::face_iterator face = cell->face(face_no);
    const typename DoFHandler<dim>::cell_iterator neighbor =
      cell->neighbor(face_no);

    Assert(neighbor.state() == IteratorState::valid, ExcInternalError());
    Assert(neighbor->has_children(), ExcInternalError());
    (void)neighbor;
    const unsigned int neighbor_neighbor = cell->neighbor_of_neighbor(face_no);
    for (unsigned int subface_no = 0; subface_no < face->n_children();
         ++subface_no)
      {
        const typename DoFHandler<dim>::active_cell_iterator neighbor_child =
          cell->neighbor_child_on_subface(face_no, subface_no);
        Assert(neighbor_child->face(neighbor_neighbor) ==
                 cell->face(face_no)->child(subface_no),
               ExcInternalError());
        const unsigned int quadrature_index =
          std::max(cell->active_fe_index(), neighbor_child->active_fe_index());
        // initialize fe_subface values_cell
        subface_hp_fe_values->reinit(cell,
                                     face_no,
                                     subface_no,
                                     quadrature_index);
        const FESubfaceValues<dim> &subface_fe_values_cell =
          subface_hp_fe_values->get_present_fe_values();
        std::vector<std::vector<Tensor<1, dim, double>>> cell_primal_grads(
          subface_fe_values_cell.n_quadrature_points,
          std::vector<Tensor<1, dim, double>>(dim)),
          neighbor_primal_grads(subface_fe_values_cell.n_quadrature_points,
                                std::vector<Tensor<1, dim, double>>(dim));
        subface_fe_values_cell.get_function_gradients(primal_solution,
                                                      cell_primal_grads);
        // initialize fe_face_values_neighbor
        face_hp_fe_values_neighbor->reinit(neighbor_child,
                                           neighbor_neighbor,
                                           quadrature_index);
        const FEFaceValues<dim> &face_fe_values_neighbor =
          face_hp_fe_values_neighbor->get_present_fe_values();
        face_fe_values_neighbor.get_function_gradients(primal_solution,
                                                       neighbor_primal_grads);
        const unsigned int n_q_points =
          subface_fe_values_cell.n_quadrature_points;
        std::vector<Vector<double>> cell_dual_values(n_q_points,
                                                     Vector<double>(dim));
        face_fe_values_neighbor.get_function_values(dual_weights,
                                                    cell_dual_values);

        double face_integral = 0.0;

        for (unsigned int p = 0; p < n_q_points; ++p)
          {
            auto face_normal = face_fe_values_neighbor.normal_vector(p);
            face_integral +=
              (cell_primal_grads[p][0][1] - cell_primal_grads[p][1][0] +
               neighbor_primal_grads[p][1][0] -
               neighbor_primal_grads[p][0][1]) *
              (cell_dual_values[p][0] * face_normal[1] -
               cell_dual_values[p][1] * face_normal[0]) *
              face_fe_values_neighbor.JxW(p);
          }
        face_integrals[neighbor_child->face(neighbor_neighbor)] = face_integral;
      }
    double sum = 0.0;
    for (unsigned int subface_no = 0; subface_no < face->n_children();
         ++subface_no)
      {
        Assert(face_integrals.find(face->child(subface_no)) !=
                 face_integrals.end(),
               ExcInternalError());
        Assert(face_integrals[face->child(subface_no)] != -1e20,
               ExcInternalError());
        sum += face_integrals[face->child(subface_no)];
      }
    face_integrals[face] = sum;
  }

  template <int dim, bool report_dual>
  double
  DualWeightedResidual<dim, report_dual>::get_global_QoI_error(
    Vector<double> &dual_solution,
    Vector<double> &error_indicators)
  {
    auto dual_less_primal =
      dual_solution; // Note: We have already extracted the primal solution...


    double scaling_factor = 0.0;
    for (const auto &cell :
         DualSolver<dim>::dof_handler.active_cell_iterators())
      {
        cell_hp_fe_values->reinit(cell);
        // grab the fe_values object
        const FEValues<dim> &fe_values =
          cell_hp_fe_values->get_present_fe_values();

        std::vector<Vector<double>> cell_values(fe_values.n_quadrature_points,
                                                Vector<double>(dim));
        fe_values.get_function_values(dual_less_primal, cell_values);

        for (unsigned int p = 0; p < fe_values.n_quadrature_points; ++p)
          {
            scaling_factor +=
              (cell_values[p] * cell_values[p]) * fe_values.JxW(p);
          }
      }
    double global_QoI_error = 0.0;
    for (const auto &indicator : error_indicators)
      {
        global_QoI_error += indicator;
      }

    global_QoI_error /= (1 - 0.5 * scaling_factor);
    return global_QoI_error;
  }


  template <int dim, bool report_dual>
  void
  DualWeightedResidual<dim, report_dual>::embed(
    const DoFHandler<dim> &          dof1,
    const DoFHandler<dim> &          dof2,
    const AffineConstraints<double> &constraints,
    const Vector<double> &           solution,
    Vector<double> &                 u2)
  {
    assert(u2.size() == dof2.n_dofs() && "Incorrect input vector size!");

    u2 = 0.0;

    typename DoFHandler<dim>::active_cell_iterator cell1 = dof1.begin_active(),
                                                   endc1 = dof1.end();
    typename DoFHandler<dim>::active_cell_iterator cell2 = dof2.begin_active();

    for (; cell1 < endc1; ++cell1, ++cell2)
      {
        const auto &fe1 =
          dynamic_cast<const FE_Nedelec<dim> &>(cell1->get_fe());
        const auto &fe2 =
          dynamic_cast<const FE_Nedelec<dim> &>(cell2->get_fe());

        assert(fe1.degree < fe2.degree && "Incorrect usage of embed!");

        // Get the embedding_dofs


        std::vector<unsigned int> embedding_dofs =
          fe2.get_embedding_dofs(fe1.degree);
        const unsigned int dofs_per_cell2 = fe2.n_dofs_per_cell();


        Vector<double> local_dof_values_1;
        Vector<double> local_dof_values_2(dofs_per_cell2);

        local_dof_values_1.reinit(fe1.dofs_per_cell);
        cell1->get_dof_values(solution, local_dof_values_1);

        for (unsigned int i = 0; i < local_dof_values_1.size(); ++i)
          local_dof_values_2[embedding_dofs[i]] = local_dof_values_1[i];

        // Now set this changes to the global vector
        cell2->set_dof_values(local_dof_values_2, u2);
      }

    u2.compress(VectorOperation::insert);
    // Applies the constraints of the target finite element space
    constraints.distribute(u2);
  }

  template <int dim, bool report_dual>
  void
  DualWeightedResidual<dim, report_dual>::extract(
    const DoFHandler<dim> &          dof1,
    const DoFHandler<dim> &          dof2,
    const AffineConstraints<double> &constraints,
    const Vector<double> &           solution,
    Vector<double> &                 u2)
  {
    // Maps from fe1 to fe2
    assert(u2.size() == dof2.n_dofs() && "Incorrect input vector size!");

    u2 = 0.0;

    typename DoFHandler<dim>::active_cell_iterator cell1 = dof1.begin_active(),
                                                   endc1 = dof1.end();
    typename DoFHandler<dim>::active_cell_iterator cell2 = dof2.begin_active();

    for (; cell1 < endc1; ++cell1, ++cell2)
      {
        const auto &fe1 =
          dynamic_cast<const FE_Nedelec<dim> &>(cell1->get_fe());
        const auto &fe2 =
          dynamic_cast<const FE_Nedelec<dim> &>(cell2->get_fe());

        assert(fe1.degree > fe2.degree && "Incorrect usage of extract!");

        // Get the embedding_dofs
        std::vector<unsigned int> embedding_dofs =
          fe1.get_embedding_dofs(fe2.degree);
        const unsigned int dofs_per_cell2 = fe2.n_dofs_per_cell();


        Vector<double> local_dof_values_1;
        Vector<double> local_dof_values_2(dofs_per_cell2);

        local_dof_values_1.reinit(fe1.dofs_per_cell);
        cell1->get_dof_values(solution, local_dof_values_1);

        for (unsigned int i = 0; i < local_dof_values_2.size(); ++i)
          local_dof_values_2[i] = local_dof_values_1[embedding_dofs[i]];

        // Now set this changes to the global vector
        cell2->set_dof_values(local_dof_values_2, u2);
      }

    u2.compress(VectorOperation::insert);
    // Applies the constraints of the target finite element space
    constraints.distribute(u2);
  }
  template <int dim, bool report_dual>
  void
  DualWeightedResidual<dim, report_dual>::output_eigenvalue_data(
    std::ofstream &os)
  {
    os << (*this->get_primal_eigenvalues())[0] << " "
       << (this->get_primal_DoFHandler())->n_dofs() << " "
       << (*this->get_dual_eigenvalues())[0] << " "
       << (this->get_dual_DoFHandler())->n_dofs() << std::endl;
  }
  template <int dim, bool report_dual>
  void
  DualWeightedResidual<dim, report_dual>::output_qoi_error_estimates(
    std::ofstream &os)
  {
    os << qoi_error_estimate << std::endl;
  }

  /**
  Provides a secondary error estimator, based on the Kelly error indicator.
  Requires only the primal solver.
  */
  template <int dim>
  class KellyErrorIndicator : public PrimalSolver<dim>
  {
  public:
    std::string
    name() const
    {
      return "Kelly";
    }
    void
    output_eigenvalue_data(std::ofstream &os);
    void
    output_qoi_error_estimates(std::ofstream &);
    KellyErrorIndicator(const std::string & prm_file,
                        Triangulation<dim> &coarse_grid,
                        const unsigned int &min_degree,
                        const unsigned int &max_degree,
                        const unsigned int &starting_degree);

    virtual unsigned int
    solve_problem() override;

    virtual void
    output_solution() override;

    hp::FECollection<dim> *
    get_FECollection();

    hp::FECollection<dim> *
    get_primal_FECollection();

    std::unique_ptr<std::vector<PETScWrappers::MPI::Vector>> &
    get_eigenfunctions();

    std::unique_ptr<std::vector<PETScWrappers::MPI::Vector>> &
    get_primal_eigenfunctions();

    std::unique_ptr<std::vector<double>> &
    get_primal_eigenvalues();


    void
    synchronize_discretization();

    DoFHandler<dim> *
    get_DoFHandler();

    DoFHandler<dim> *
    get_primal_DoFHandler();

    unsigned int
    get_max_degree()
    {
      return PrimalSolver<dim>::fe_collection->max_degree();
    }
    double qoi_error_estimate = 0;

  protected:
    void
    estimate_error(Vector<double> &error_indicators);

  private:
    void
    prune_eigenpairs(const double &TOL);

#if DEAL_II_VERSION_GTE(9, 6, 0)
    std::vector<const ReadVector<PetscScalar> *>    eigenfunction_ptrs;
#else
    std::vector<const PETScWrappers::MPI::Vector *> eigenfunction_ptrs;
#endif
    std::vector<const double *>                     eigenvalue_ptrs;

    std::vector<std::shared_ptr<Vector<float>>> errors;
  };

  template <int dim>
  KellyErrorIndicator<dim>::KellyErrorIndicator(
    const std::string & prm_file,
    Triangulation<dim> &coarse_grid,
    const unsigned int &min_degree,
    const unsigned int &max_degree,
    const unsigned int &starting_degree)
    : Base<dim>(prm_file, coarse_grid)
    , PrimalSolver<dim>(prm_file,
                        coarse_grid,
                        min_degree,
                        max_degree,
                        starting_degree)
  {}

  template <int dim>
  unsigned int
  KellyErrorIndicator<dim>::solve_problem()
  {
    return PrimalSolver<dim>::solve_problem();
  }

  template <int dim>
  hp::FECollection<dim> *
  KellyErrorIndicator<dim>::get_FECollection()
  {
    return &*(PrimalSolver<dim>::fe_collection);
  }

  template <int dim>
  hp::FECollection<dim> *
  KellyErrorIndicator<dim>::get_primal_FECollection()
  {
    return &*(PrimalSolver<dim>::fe_collection);
  }

  template <int dim>
  std::unique_ptr<std::vector<PETScWrappers::MPI::Vector>> &
  KellyErrorIndicator<dim>::get_eigenfunctions()
  {
    return (PrimalSolver<dim>::eigenfunctions);
  }

  template <int dim>
  std::unique_ptr<std::vector<double>> &
  KellyErrorIndicator<dim>::get_primal_eigenvalues()
  {
    return PrimalSolver<dim>::eigenvalues;
  }

  template <int dim>
  std::unique_ptr<std::vector<PETScWrappers::MPI::Vector>> &
  KellyErrorIndicator<dim>::get_primal_eigenfunctions()
  {
    return (PrimalSolver<dim>::eigenfunctions);
  }

  template <int dim>
  DoFHandler<dim> *
  KellyErrorIndicator<dim>::get_DoFHandler()
  {
    return &(PrimalSolver<dim>::dof_handler);
  }

  template <int dim>
  DoFHandler<dim> *
  KellyErrorIndicator<dim>::get_primal_DoFHandler()
  {
    return &(PrimalSolver<dim>::dof_handler);
  }

  template <int dim>
  void
  KellyErrorIndicator<dim>::synchronize_discretization()
  {
    // This function does nothing for this error indicator
    return;
  }

  template <int dim>
  void
  KellyErrorIndicator<dim>::output_solution()
  {
    PrimalSolver<dim>::output_solution();
  }

  template <int dim>
  void
  KellyErrorIndicator<dim>::prune_eigenpairs(const double &TOL)
  {
    unsigned int count = 0;
    for (size_t eigenpair_index = 0;
         eigenpair_index < this->eigenfunctions->size();
         ++eigenpair_index)
      {
        if (count >= this->n_eigenpairs)
          break;
        if (abs((*this->eigenvalues)[eigenpair_index]) < TOL)
          continue;

        eigenfunction_ptrs.push_back(&(*this->eigenfunctions)[eigenpair_index]);
        eigenvalue_ptrs.push_back(&(*this->eigenvalues)[eigenpair_index]);
      }
  }

  template <int dim>
  void
  KellyErrorIndicator<dim>::estimate_error(Vector<double> &error_indicators)
  {
    std::cout << "Marking cells via Kelly indicator..." << std::endl;
    prune_eigenpairs(1e-9);
    // deallocate the errors vector
    errors.clear();
    for (size_t i = 0; i < eigenfunction_ptrs.size(); ++i)
      {
        errors.emplace_back(
          new Vector<float>(this->triangulation->n_active_cells()));
      }
    std::vector<Vector<float> *> estimated_error_per_cell(
      eigenfunction_ptrs.size());
    for (size_t i = 0; i < eigenfunction_ptrs.size(); ++i)
      {
        estimated_error_per_cell[i] = errors[i].get();
      }

#if DEAL_II_VERSION_GTE(9, 6, 0)
    const auto solution_view = make_array_view(eigenfunction_ptrs);
    auto error_view = make_array_view(estimated_error_per_cell);
    KellyErrorEstimator<dim>::estimate(this->dof_handler,
                                       *this->face_quadrature_collection,
                                       {},
                                       solution_view,
                                       error_view);
#else
    KellyErrorEstimator<dim>::estimate(this->dof_handler,
                                       *this->face_quadrature_collection,
                                       {},
                                       eigenfunction_ptrs,
                                       estimated_error_per_cell);
#endif

    for (auto &error_vec : errors)
      {
        auto normalized_vec = *error_vec;
        normalized_vec /= normalized_vec.l1_norm();

        for (unsigned int i = 0; i < error_indicators.size(); ++i)
          error_indicators(i) += double(normalized_vec(i));
      }
    std::cout << "...Done!" << std::endl;
  }
  template <int dim>
  void
  KellyErrorIndicator<dim>::output_eigenvalue_data(std::ofstream &os)
  {
    os << (*this->get_primal_eigenvalues())[0] << " "
       << (this->get_primal_DoFHandler())->n_dofs() << std::endl;
  }
  template <int dim>
  void
  KellyErrorIndicator<dim>::output_qoi_error_estimates(std::ofstream &)
  {
    return;
  }

} // namespace ErrorIndicators

/**
Includes all of the classes needed for smoothness estimation.
*/
namespace RegularityIndicators
{
  using namespace dealii;

  /* For the Legendre smoothness indicator*/
  /* Adapted from M. Fehling's smoothness_estimator.cc*/
  template <int dim>
  class LegendreInfo
  {};

  template <>
  class LegendreInfo<2>
  {
  public:
    std::unique_ptr<FESeries::Legendre<2>> legendre_u, legendre_v;

    hp::FECollection<2> *fe_collection = nullptr;
    DoFHandler<2> *      dof_handler   = nullptr;

    void
    initialization()
    {
      assert(fe_collection != nullptr && dof_handler != nullptr &&
             "A valid FECollection and DoFHandler must be accessible!");

      legendre_u = std::make_unique<FESeries::Legendre<2>>(
        SmoothnessEstimator::Legendre::default_fe_series(*fe_collection, 0));
      legendre_v = std::make_unique<FESeries::Legendre<2>>(
        SmoothnessEstimator::Legendre::default_fe_series(*fe_collection, 1));

      legendre_u->precalculate_all_transformation_matrices();
      legendre_v->precalculate_all_transformation_matrices();
    }

    template <class VectorType>
    void
    compute_coefficient_decay(const VectorType &   eigenfunction,
                              std::vector<double> &smoothness_indicators)
    {
      // Compute the coefficients for the u and v components of the solution
      // separately,
      Vector<float> smoothness_u(smoothness_indicators.size()),
        smoothness_v(smoothness_indicators.size());

      SmoothnessEstimator::Legendre::coefficient_decay(*legendre_u,
                                                       *dof_handler,
                                                       eigenfunction,
                                                       smoothness_u);

      SmoothnessEstimator::Legendre::coefficient_decay(*legendre_v,
                                                       *dof_handler,
                                                       eigenfunction,
                                                       smoothness_v);

      for (unsigned int i = 0; i < smoothness_indicators.size(); ++i)
        {
          smoothness_indicators[i] = std::min(smoothness_u[i], smoothness_v[i]);
        }
    }
  };

  /**
  Implements the LegendreIndicator for use with the Refiner
  */
  template <int dim>
  class LegendreIndicator
  {
  public:
    void
    attach_FE_info_and_initialize(hp::FECollection<dim> *fe_ptr,
                                  DoFHandler<dim> *      dof_ptr);

  protected:
    template <class VectorType>
    void
    estimate_smoothness(
      const std::unique_ptr<std::vector<VectorType>> &eigenfunctions,
      const unsigned int &                            index_of_goal,
      std::vector<double> &                           smoothness_indicators);

  private:
    LegendreInfo<dim> legendre;
  };

  template <int dim>
  void
  LegendreIndicator<dim>::attach_FE_info_and_initialize(
    hp::FECollection<dim> *fe_ptr,
    DoFHandler<dim> *      dof_ptr)
  {
    legendre.fe_collection = fe_ptr;
    legendre.dof_handler   = dof_ptr;
    this->legendre.initialization();
  }

  template <int dim>
  template <class VectorType>
  void
  LegendreIndicator<dim>::estimate_smoothness(
    const std::unique_ptr<std::vector<VectorType>> &eigenfunctions,
    const unsigned int &                            index_of_goal,
    std::vector<double> &                           smoothness_indicators)
  {
    this->legendre.compute_coefficient_decay((*eigenfunctions)[index_of_goal],
                                             smoothness_indicators);
  }
} // namespace RegularityIndicators

/**
The final namespace, which combines the error estimation/indication and
smoothness estimation functionality to conduct refinement.
*/
namespace Refinement
{
  using namespace dealii;
  using namespace Maxwell;

  template <int dim, class ErrorIndicator, class RegularityIndicator>
  class Refiner : public ErrorIndicator, public RegularityIndicator
  {
  public:
    Refiner(const std::string & prm_file,
            Triangulation<dim> &coarse_grid,
            const unsigned int &min_degree,
            const unsigned int &max_degree,
            const unsigned int &starting_degree);

    void
    execute_refinement(const double &smoothness_threshold_fraction);

    virtual void
    output_solution() override;

  private:
    Vector<double>      estimated_error_per_cell;
    std::vector<double> smoothness_indicators;
    std::ofstream       eigenvalues_out;
    std::ofstream       error_estimate_out;
  };

  template <int dim, class ErrorIndicator, class RegularityIndicator>
  Refiner<dim, ErrorIndicator, RegularityIndicator>::Refiner(
    const std::string & prm_file,
    Triangulation<dim> &coarse_grid,
    const unsigned int &min_degree,
    const unsigned int &max_degree,
    const unsigned int &starting_degree)
    : Base<dim>(prm_file, coarse_grid)
    , ErrorIndicator(prm_file,
                     coarse_grid,
                     min_degree,
                     max_degree,
                     starting_degree)
    , RegularityIndicator()
  {
    if (ErrorIndicator::name() == "DWR")
      {
        error_estimate_out.open("error_estimate.txt");
        error_estimate_out << std::setprecision(20);
      }

    eigenvalues_out.open("eigenvalues_" + ErrorIndicator::name() + "_out.txt");
    eigenvalues_out << std::setprecision(20);
  }

  // For generating samples of the curl of the electric field
  template <int dim>
  class CurlPostprocessor : public DataPostprocessorScalar<dim>
  {
  public:
    CurlPostprocessor()
      : DataPostprocessorScalar<dim>("Curl", update_gradients)
    {}

    virtual void
    evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &input_data,
      std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
        {
          computed_quantities[p](0) = input_data.solution_gradients[p][1][0] -
                                      input_data.solution_gradients[p][0][1];
        }
    }
  };

  /**
  Overrides the output_solution function in order to include the error
  estimation and smoothness estimation information. In the case of outputting
  the eigenfunction, the PrimalSolver result is taken.

  TODO: Extend to multiple eigenpairs
  */
  template <int dim, class ErrorIndicator, class RegularityIndicator>
  void
  Refiner<dim, ErrorIndicator, RegularityIndicator>::output_solution()
  {
    CurlPostprocessor<dim> curl_u;

    DataOut<dim> data_out;
    auto &       output_dof = *(ErrorIndicator::get_primal_DoFHandler());
    data_out.attach_dof_handler(output_dof);
    Vector<double> fe_degrees(this->triangulation->n_active_cells());
    for (const auto &cell : output_dof.active_cell_iterators())
      fe_degrees(cell->active_cell_index()) =
        (*ErrorIndicator::get_primal_FECollection())[cell->active_fe_index()]
          .degree;
    data_out.add_data_vector(fe_degrees, "fe_degree");
    //
    data_out.add_data_vector(estimated_error_per_cell, "error");
    Vector<double> smoothness_out(this->triangulation->n_active_cells());
    for (const auto &cell : output_dof.active_cell_iterators())
      {
        auto i = cell->active_cell_index();
        if (!cell->refine_flag_set() && !cell->coarsen_flag_set())
          smoothness_out(i) = -1;
        else
          smoothness_out(i) = smoothness_indicators[i];
      }
    data_out.add_data_vector(smoothness_out, "smoothness");
    data_out.add_data_vector((*ErrorIndicator::get_primal_eigenfunctions())[0],
                             std::string("eigenfunction_no_") +
                               Utilities::int_to_string(0));
    data_out.add_data_vector((*ErrorIndicator::get_primal_eigenfunctions())[0],
                             curl_u);

    ErrorIndicator::output_eigenvalue_data(eigenvalues_out);
    ErrorIndicator::output_qoi_error_estimates(error_estimate_out);

    std::cout << "Number of DoFs: " << (this->get_primal_DoFHandler())->n_dofs()
              << std::endl;


    data_out.build_patches();
    std::ofstream output("eigenvectors-" + ErrorIndicator::name() + "-" +
                         std::to_string(this->refinement_cycle) + +".vtu");
    data_out.write_vtu(output);
  }


  /**
  Solves the problem (provided by the ErrorIndicator) and estimates the
  smoothness. For cells marked for refinement, if the smoothness_threshold
  is exceeded, $p$-refinement is chosen, otherwise $h$-refinement is chosen.
  */
  template <int dim, class ErrorIndicator, class RegularityIndicator>
  void
  Refiner<dim, ErrorIndicator, RegularityIndicator>::execute_refinement(
    const double &smoothness_threshold_fraction)
  {
    // First initialize the RegularityIndicator...
    // Depending on the limits set, this may take a while
    std::cout << "Initializing RegularityIndicator..." << std::endl;
    std::cout
      << "(This may take a while if the max expansion order is set too high)"
      << std::endl;
    RegularityIndicator::attach_FE_info_and_initialize(
      ErrorIndicator::get_FECollection(), ErrorIndicator::get_DoFHandler());
    std::cout << "Done!" << std::endl << "Starting Refinement..." << std::endl;

    for (unsigned int cycle = 0; cycle <= this->max_cycles; ++cycle)
      {
        this->set_refinement_cycle(cycle);
        std::cout << "Cycle: " << cycle << std::endl;
        ErrorIndicator::solve_problem();
        this->estimated_error_per_cell.reinit(
          this->triangulation->n_active_cells());

        ErrorIndicator::estimate_error(estimated_error_per_cell);

        // Depending on the source of the error estimation/indication, these
        // values might be signed, so we address that with the following
        for (double &error_indicator : estimated_error_per_cell)
          error_indicator = std::abs(error_indicator);


        GridRefinement::refine_and_coarsen_fixed_number(
          *this->triangulation, estimated_error_per_cell, 1. / 5., 0.000);

        // Now get regularity indicators
        // For those elements which must be refined, swap to increasing $p$
        // depending on the regularity threshold...

        smoothness_indicators =
          std::vector<double>(this->triangulation->n_active_cells(),
                              std::numeric_limits<double>::max());
        if (ErrorIndicator::PrimalSolver::min_degree !=
            ErrorIndicator::PrimalSolver::max_degree)
          RegularityIndicator::estimate_smoothness(
            ErrorIndicator::get_eigenfunctions(), 0, smoothness_indicators);
        // save data
        this->output_solution();
        const double threshold_smoothness = smoothness_threshold_fraction;
        unsigned int num_refined = 0, num_coarsened = 0;
        if (ErrorIndicator::PrimalSolver::min_degree !=
            ErrorIndicator::PrimalSolver::max_degree)
          {
            for (const auto &cell :
                 ErrorIndicator::get_DoFHandler()->active_cell_iterators())
              {
                if (cell->refine_flag_set())
                  ++num_refined;
                if (cell->coarsen_flag_set())
                  ++num_coarsened;
                if (cell->refine_flag_set() &&
                    smoothness_indicators[cell->active_cell_index()] >
                      threshold_smoothness &&
                    static_cast<unsigned int>(cell->active_fe_index() + 1) <
                      ErrorIndicator::get_FECollection()->size())
                  {
                    cell->clear_refine_flag();
                    cell->set_active_fe_index(cell->active_fe_index() + 1);
                  }
                else if (cell->coarsen_flag_set() &&
                         smoothness_indicators[cell->active_cell_index()] <
                           threshold_smoothness &&
                         cell->active_fe_index() != 0)
                  {
                    cell->clear_coarsen_flag();

                    cell->set_active_fe_index(cell->active_fe_index() - 1);
                  }
                // Here we also impose a limit on how small the cells can become
                else if (cell->refine_flag_set() && cell->diameter() < 5.0e-6)
                  {
                    cell->clear_refine_flag();
                    if (static_cast<unsigned int>(cell->active_fe_index() + 1) <
                        ErrorIndicator::get_FECollection()->size())
                      cell->set_active_fe_index(cell->active_fe_index() + 1);
                  }
              }
          }

        // Check what the smallest diameter is
        double min_diameter = std::numeric_limits<double>::max();
        for (const auto &cell :
             ErrorIndicator::get_DoFHandler()->active_cell_iterators())
          if (cell->diameter() < min_diameter)
            min_diameter = cell->diameter();

        std::cout << "Min diameter: " << min_diameter << std::endl;

        ErrorIndicator::synchronize_discretization();

        (this->triangulation)->execute_coarsening_and_refinement();
      }
  }
} // namespace Refinement

int
main(int argc, char **argv)
{
  try
    {
      using namespace dealii;
      using namespace Maxwell;
      using namespace Refinement;
      using namespace ErrorIndicators;
      using namespace RegularityIndicators;


      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);


      AssertThrow(
        Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
        ExcMessage("This program can only be run in serial, use ./maxwell-hp"));

      Triangulation<2> triangulation_DWR, triangulation_Kelly;
      Structures::create_L_waveguide(triangulation_DWR, 2.0);
      Structures::create_L_waveguide(triangulation_Kelly, 2.0);

      Refiner<2, KellyErrorIndicator<2>, LegendreIndicator<2>> problem_Kelly(
        "maxwell-hp.prm",
        triangulation_Kelly,
        /*Minimum Degree*/ 2,
        /*Maximum Degree*/ 5,
        /*Starting Degree*/ 2);

      Refiner<2, DualWeightedResidual<2, false>, LegendreIndicator<2>>
        problem_DWR("maxwell-hp.prm",
                    triangulation_DWR,
                    /*Minimum Degree*/ 2,
                    /*Maximum Degree*/ 5,
                    /*Starting Degree*/ 2);

      // The threshold for the hp-decision: too small -> not enough
      // $h$-refinement, too large -> not enough $p$-refinement
      double smoothness_threshold = 0.75;

      std::cout << "Executing refinement for the Kelly strategy!" << std::endl;
      problem_Kelly.execute_refinement(smoothness_threshold);
      std::cout << "...Done with Kelly refinement strategy!" << std::endl;
      std::cout << "Executing refinement for the DWR strategy!" << std::endl;
      problem_DWR.execute_refinement(smoothness_threshold);
      std::cout << "...Done with DWR refinement strategy!" << std::endl;
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

  std::cout << std::endl << "   Job done." << std::endl;

  return 0;
}
