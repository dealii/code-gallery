/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2025 by Marco Feder, Pasquale Claudio Africa, Xinping Gui,
 * Andrea Cangiani
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 */

// The following headers provide the deal.II functionality needed in this
// example. Most of them are standard components for mesh handling, finite
// element mappings, linear algebra, and graphical output. In addition, we
// include the agglomeration-specific headers that define the data structures
// and utilities used to construct and manage polytopal agglomerates.

// deal.II base utilities.
#include <deal.II/base/exceptions.h>

// Finite element mappings.
#include <deal.II/fe/mapping_fe.h>

// Grid generation, mesh input/output, and mesh-related utilities.
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

// Linear algebra objects and sparse direct solvers.
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>

// Output of finite element data for visualization.
#include <deal.II/numerics/data_out.h>

// Agglomeration-specific headers used in this example.
#include <agglomeration_handler.h>
#include <poly_utils.h>

// C++ standard library headers.
#include <algorithm>
#include <chrono>


// We use the struct ConvergenceInfo to store the number of degrees of freedom together
// with the corresponding L2 and H1 errors, and print a simple
// convergence table to the console.
struct ConvergenceInfo
{
  ConvergenceInfo() = default;
  void
  add(const std::pair<types::global_dof_index, std::pair<double, double>>
        &dofs_and_errs)
  {
    vec_data.push_back(dofs_and_errs);
  }

  void
  print()
  {
    Assert(vec_data.size() > 0, ExcInternalError());
    std::cout << std::left << "#DoFs, L2 error, H1 error" << std::endl;

    for (const auto &dof_and_errs : vec_data)
      std::cout << std::scientific << dof_and_errs.first << ", "
                << dof_and_errs.second.first << ", "
                << dof_and_errs.second.second << std::endl;
  }

  std::vector<std::pair<types::global_dof_index, std::pair<double, double>>>
    vec_data;
};

// We will compare the performance of three different partitioning strategies:
// using METIS, using an R-tree based agglomeration, or not performing any
// partitioning at all.
enum class PartitionerType
{
  metis,
  rtree,
  no_partition
};

//We then implement the manufactured right-hand side
//   f(x, y) = 2 π² sin(π x) sin(π y),
// which corresponds to the exact solution
//   u(x, y) = sin(π x) sin(π y).
template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide()
    : Function<dim>()
  {}

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &values,
             const unsigned int /*component*/) const override
  {
    for (unsigned int i = 0; i < values.size(); ++i)
      values[i] = 2 * numbers::PI * numbers::PI *
                  std::sin(numbers::PI * points[i][0]) *
                  std::sin(numbers::PI * points[i][1]);
  }
};


// Exact solution is set as u(x,y) = sin(pi x) sin(pi y).
// It is used to impose Dirichlet boundary conditions and to evaluate
// the L2 and H1-seminorm errors. Its gradient is also provided for
// the computation of the H1 error.
template <int dim>
class ExactSolution : public Function<dim>
{
public:
  ExactSolution()
    : Function<dim>()
  {
    Assert(dim == 2, ExcNotImplemented());
  }

  virtual double
  value(const Point<dim> &p,
        const unsigned int /* component */ = 0) const override
  {
    return std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
  }

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &values,
             const unsigned int /*component*/) const override
  {
    for (unsigned int i = 0; i < values.size(); ++i)
      values[i] = this->value(points[i]);
  }

  virtual Tensor<1, dim>
  gradient(const Point<dim> &p,
           const unsigned int /* component */ = 0) const override
  {
    Tensor<1, dim> return_value;
    return_value[0] =
      numbers::PI * std::cos(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
    return_value[1] =
      numbers::PI * std::cos(numbers::PI * p[1]) * std::sin(numbers::PI * p[0]);
    return return_value;
  }
};

// The Poisson<dim> class encapsulates the solution of the model Poisson
// problem
// @f[ -\Delta u = f \quad \text{in } \Omega, \qquad u = u_D \quad \text{on } \partial\Omega. @f]
// It sets up a fine triangulation, constructs agglomerated polytopal
// cells according to the chosen partitioning strategy, assembles the
// symmetric interior penalty DG discretization on the agglomerated mesh,
// solves the resulting linear system, and finally postprocesses the
// numerical solution by writing visualization output and computing
// global error norms.
template <int dim>
class Poisson
{
private:
  void
  make_grid();
  void
  setup_agglomeration();
  void
  assemble_system();
  void
  solve();
  void
  output_results();

  Triangulation<dim>                         tria;
  MappingQ1<dim>                             mapping;
  FE_DGQ<dim>                                dg_fe; 
  std::unique_ptr<AgglomerationHandler<dim>> ah;
  AffineConstraints<double>                  constraints;
  SparsityPattern                            sparsity;
  DynamicSparsityPattern                     dsp;
  SparseMatrix<double>                       system_matrix;
  Vector<double>                             solution;
  Vector<double>                             system_rhs;
  std::unique_ptr<GridTools::Cache<dim>>     cached_tria;
  std::unique_ptr<const Function<dim>>       rhs_function;
  std::unique_ptr<const Function<dim>>       analytical_solution;

public:
  Poisson(const PartitionerType &partitioner_type = PartitionerType::rtree,
          const unsigned int                      = 0,
          const unsigned int                      = 0,
          const unsigned int fe_degree            = 1);
  void
  run();

  types::global_dof_index
  get_n_dofs() const;

  std::pair<double, double>
  get_error() const;

  PartitionerType partitioner_type;
  unsigned int    extraction_level;
  unsigned int    n_subdomains;
  double penalty_constant = 60.; // 10*(p+1)(p+d) for p = 1 and d = 2 => 60
  double l2_err;
  double semih1_err;
};


// The constructor initializes the Poisson<dim> solver with the selected partitioning
// strategy, agglomeration parameters, polynomial degree, and the
// manufactured exact solution and right-hand side.
template <int dim>
Poisson<dim>::Poisson(const PartitionerType &partitioner_type,
                      const unsigned int     extraction_level,
                      const unsigned int     n_subdomains,
                      const unsigned int     fe_degree)
  : mapping()
  , dg_fe(fe_degree)
  , partitioner_type(partitioner_type)
  , extraction_level(extraction_level)
  , n_subdomains(n_subdomains)
  , penalty_constant(10. * (fe_degree + 1) * (fe_degree + dim))
{
  // Initialize manufactured solution.
  analytical_solution = std::make_unique<ExactSolution<dim>>();
  rhs_function        = std::make_unique<const RightHandSide<dim>>();
  constraints.close();
}



// Build the fine triangulation from a Gmsh mesh, apply a global
// refinement, initialize the cache and agglomeration handler, and
// define agglomerates according to the selected partitioning strategy.

template <int dim>
void
Poisson<dim>::make_grid()
{
  GridIn<dim> grid_in;
  grid_in.attach_triangulation(tria);
  std::ifstream gmsh_file(std::string(MESH_DIR) +
                          "/unit_square_quad_unstructured.msh");
  grid_in.read_msh(gmsh_file);

  {
    GridOut        grid_out;
    std::ofstream  out("grid_input_mesh.vtu");
    grid_out.write_vtu(tria, out); // Write the input mesh (before any refinement), for documentation/figures.
  }

  tria.refine_global(5); // Refine the mesh to obtain the fine grid used for agglomeration.

  {
    GridOut        grid_out;
    std::ofstream  out("grid_fine_mesh_refined.vtu");
    grid_out.write_vtu(tria, out); // Write the refined (fine) mesh used as starting point for agglomeration.
   }


  std::cout << "Size of tria: " << tria.n_active_cells() << std::endl;
  cached_tria = std::make_unique<GridTools::Cache<dim>>(tria, mapping);
  ah          = std::make_unique<AgglomerationHandler<dim>>(*cached_tria);

  if (partitioner_type == PartitionerType::metis)
    { // Partition the triangulation with a graph partitioner.
      auto start = std::chrono::system_clock::now();
      GridTools::partition_triangulation(n_subdomains,
                                         tria,
                                         SparsityTools::Partitioner::metis);

      std::vector<
        std::vector<typename Triangulation<dim>::active_cell_iterator>>
        cells_per_subdomain(n_subdomains);
      for (const auto &cell : tria.active_cell_iterators())
        cells_per_subdomain[cell->subdomain_id()].push_back(cell);

      for (std::size_t i = 0; i < n_subdomains; ++i) // Define one agglomerate for each subdomain
        ah->define_agglomerate(cells_per_subdomain[i]);

      std::chrono::duration<double> wctduration =
        (std::chrono::system_clock::now() - start);
      std::cout << "METIS built in " << wctduration.count()
                << " seconds [wall clock]" << std::endl;
    }
  else if (partitioner_type == PartitionerType::rtree)
    { // Build agglomerates from the R-tree hierarchy

      namespace bgi = boost::geometry::index;
      static constexpr unsigned int max_elem_per_node =
        PolyUtils::constexpr_pow(2, dim);
      std::vector<std::pair<BoundingBox<dim>,
                            typename Triangulation<dim>::active_cell_iterator>>
                   boxes(tria.n_active_cells());
      unsigned int i = 0;
      for (const auto &cell : tria.active_cell_iterators())
        boxes[i++] = std::make_pair(mapping.get_bounding_box(cell), cell);

      auto start = std::chrono::system_clock::now();
      auto tree  = pack_rtree<bgi::rstar<max_elem_per_node>>(boxes);

      CellsAgglomerator<dim, decltype(tree)> agglomerator{tree,
                                                          extraction_level};
      const auto vec_agglomerates = agglomerator.extract_agglomerates();

      for (const auto &agglo : vec_agglomerates) // Flag elements for agglomeration
        ah->define_agglomerate(agglo);

      std::chrono::duration<double> wctduration =
        (std::chrono::system_clock::now() - start);
      std::cout << "R-tree agglomerates built in " << wctduration.count()
                << " seconds [wall clock]" << std::endl;
    }
  else if (partitioner_type == PartitionerType::no_partition)
    {
    }
  else
    {
      Assert(false, ExcMessage("Wrong partitioning."));
    }
  n_subdomains = ah->n_agglomerates();
  std::cout << "N subdomains = " << n_subdomains << std::endl;
}


// To finalize the agglomeration. In the no-partition case, each fine cell is declared as its own
// agglomerate. The function then distributes the degrees of freedom
// on the agglomerated mesh, builds the corresponding sparsity pattern,
// and writes a VTU file visualizing the agglomeration and the
// partitioning of the fine grid.
template <int dim>
void
Poisson<dim>::setup_agglomeration()
{
  if (partitioner_type == PartitionerType::no_partition)
    { // No partitioning means that each cell is a master cell
      for (const auto &cell : tria.active_cell_iterators())
        ah->define_agglomerate({cell});
    }

  ah->distribute_agglomerated_dofs(dg_fe);
  ah->create_agglomeration_sparsity_pattern(dsp);
  sparsity.copy_from(dsp);

  {
    std::string partitioner;
    if (partitioner_type == PartitionerType::metis)
      partitioner = "metis";
    else if (partitioner_type == PartitionerType::rtree)
      partitioner = "rtree";
    else
      partitioner = "no_partitioning";
        
      
    const std::string filename =
      "grid_" + partitioner + "_" + std::to_string(n_subdomains) + ".vtu";
    std::ofstream output(filename);
      
      
    DataOut<dim> data_out;
    data_out.attach_triangulation(tria);

    const auto &rel = ah->get_relationships();

    Vector<float> agglo_relationships(tria.n_active_cells()); // Store the agglomeration relationships on the fine grid by distinguishing master/slave cells
    for (const auto &cell : tria.active_cell_iterators())
    {
     const unsigned int i = cell->active_cell_index();
     agglo_relationships[i] = rel[i];
    }

    Vector<float> agglo_idx(tria.n_active_cells()); // Generate agglo_idx for visualization

    for (const auto &polytope : ah->polytope_iterators())
    {
     const float id = static_cast<float>(polytope->index());
     const auto &patch_of_cells = polytope->get_agglomerate();
     for (const auto &cell : patch_of_cells)
        agglo_idx[cell->active_cell_index()] = id;
    }

    data_out.add_data_vector(agglo_relationships,
                             "agglo_relationships",
                               DataOut<dim>::type_cell_data);
    data_out.add_data_vector(agglo_idx,
                               "agglo_idx",
                               DataOut<dim>::type_cell_data);

    data_out.build_patches(mapping);
    data_out.write_vtu(output);
  }
    
    
}

// Assemble the global SIPG matrix and right-hand side on the
// agglomerated mesh.
//
// It initializes the system matrix and right-hand side, sets up FEValues
// objects on polytopal cells and interfaces, and then adds the volume,
// boundary, and interior face contributions of the symmetric interior
// penalty formulation.
template <int dim>
void
Poisson<dim>::assemble_system()
{
  system_matrix.reinit(sparsity);
  solution.reinit(ah->n_dofs());
  system_rhs.reinit(ah->n_dofs());

  const unsigned int quadrature_degree      = dg_fe.get_degree() + 1;
  const unsigned int face_quadrature_degree = dg_fe.get_degree() + 1;

  ah->initialize_fe_values(QGauss<dim>(quadrature_degree),
                           update_gradients | update_JxW_values |
                             update_quadrature_points | update_JxW_values |
                             update_values,
                           QGauss<dim - 1>(face_quadrature_degree));

  const unsigned int dofs_per_cell = ah->n_dofs_per_cell();
  std::cout << "DoFs per cell: " << dofs_per_cell << std::endl;

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  // Next, we define the four dofsxdofs matrices needed to assemble jumps and
  // averages.
  FullMatrix<double> M11(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> M12(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> M21(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> M22(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &polytope : ah->polytope_iterators())
    {
      cell_matrix              = 0;
      cell_rhs                 = 0;
      const auto &agglo_values = ah->reinit(polytope);
      polytope->get_dof_indices(local_dof_indices);

      const auto         &q_points  = agglo_values.get_quadrature_points();
      const unsigned int  n_qpoints = q_points.size();
      std::vector<double> rhs(n_qpoints);
      rhs_function->value_list(q_points, rhs);

      for (unsigned int q_index : agglo_values.quadrature_point_indices())
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_matrix(i, j) += agglo_values.shape_grad(i, q_index) *
                                       agglo_values.shape_grad(j, q_index) *
                                       agglo_values.JxW(q_index);
                }
              cell_rhs(i) += agglo_values.shape_value(i, q_index) *
                             rhs[q_index] * agglo_values.JxW(q_index);
            }
        }

    
      const unsigned int n_faces = polytope->n_faces();
      AssertThrow(n_faces > 0,
                  ExcMessage(
                    "Invalid element: at least 4 faces are required."));

      auto polygon_boundary_vertices = polytope->polytope_boundary();
      for (unsigned int f = 0; f < n_faces; ++f)
        {
          if (polytope->at_boundary(f))
            { // std::cout << "at boundary!" << std::endl;
              const auto &fe_face = ah->reinit(polytope, f);

              const unsigned int dofs_per_cell = fe_face.dofs_per_cell;

              const auto &face_q_points = fe_face.get_quadrature_points();
              std::vector<double> analytical_solution_values(
                face_q_points.size());
              analytical_solution->value_list(face_q_points,
                                              analytical_solution_values,
                                              1);
              
              const auto &normals = fe_face.get_normal_vectors(); // Get normal vectors seen from each agglomeration.

              const double penalty =
                penalty_constant / std::fabs(polytope->diameter());

              for (unsigned int q_index : fe_face.quadrature_point_indices())
                {
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          cell_matrix(i, j) +=
                            (-fe_face.shape_value(i, q_index) *
                               fe_face.shape_grad(j, q_index) *
                               normals[q_index] -
                             fe_face.shape_grad(i, q_index) * normals[q_index] *
                               fe_face.shape_value(j, q_index) +
                             (penalty)*fe_face.shape_value(i, q_index) *
                               fe_face.shape_value(j, q_index)) *
                            fe_face.JxW(q_index);
                        }
                      cell_rhs(i) +=
                        (penalty * analytical_solution_values[q_index] *
                           fe_face.shape_value(i, q_index) -
                         fe_face.shape_grad(i, q_index) * normals[q_index] *
                           analytical_solution_values[q_index]) *
                        fe_face.JxW(q_index);
                    }
                }
            }
          else
            {
              const auto &neigh_polytope = polytope->neighbor(f);

              if (polytope->index() < neigh_polytope->index()) // This is necessary to loop over internal faces only once.
                {
                  unsigned int nofn =
                    polytope->neighbor_of_agglomerated_neighbor(f);

                  const auto &fe_faces =
                    ah->reinit_interface(polytope, neigh_polytope, f, nofn);
                  const auto &fe_faces0 = fe_faces.first;
                  const auto &fe_faces1 = fe_faces.second;

                  std::vector<types::global_dof_index>
                    local_dof_indices_neighbor(dofs_per_cell);

                  M11 = 0.;
                  M12 = 0.;
                  M21 = 0.;
                  M22 = 0.;

                  const auto &normals = fe_faces0.get_normal_vectors();

                  const double penalty =
                    penalty_constant / std::min(polytope->diameter(), neigh_polytope->diameter());

                  
                  for (unsigned int q_index : // M11
                       fe_faces0.quadrature_point_indices())
                    {
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                              M11(i, j) +=
                                (-0.5 * fe_faces0.shape_grad(i, q_index) *
                                   normals[q_index] *
                                   fe_faces0.shape_value(j, q_index) -
                                 0.5 * fe_faces0.shape_grad(j, q_index) *
                                   normals[q_index] *
                                   fe_faces0.shape_value(i, q_index) +
                                 (penalty)*fe_faces0.shape_value(i, q_index) *
                                   fe_faces0.shape_value(j, q_index)) *
                                fe_faces0.JxW(q_index);

                              M12(i, j) +=
                                (0.5 * fe_faces0.shape_grad(i, q_index) *
                                   normals[q_index] *
                                   fe_faces1.shape_value(j, q_index) -
                                 0.5 * fe_faces1.shape_grad(j, q_index) *
                                   normals[q_index] *
                                   fe_faces0.shape_value(i, q_index) -
                                 (penalty)*fe_faces0.shape_value(i, q_index) *
                                   fe_faces1.shape_value(j, q_index)) *
                                fe_faces1.JxW(q_index);

                              
                              M21(i, j) += // A10
                                (-0.5 * fe_faces1.shape_grad(i, q_index) *
                                   normals[q_index] *
                                   fe_faces0.shape_value(j, q_index) +
                                 0.5 * fe_faces0.shape_grad(j, q_index) *
                                   normals[q_index] *
                                   fe_faces1.shape_value(i, q_index) -
                                 (penalty)*fe_faces1.shape_value(i, q_index) *
                                   fe_faces0.shape_value(j, q_index)) *
                                fe_faces1.JxW(q_index);

                              
                              M22(i, j) += // A11
                                (0.5 * fe_faces1.shape_grad(i, q_index) *
                                   normals[q_index] *
                                   fe_faces1.shape_value(j, q_index) +
                                 0.5 * fe_faces1.shape_grad(j, q_index) *
                                   normals[q_index] *
                                   fe_faces1.shape_value(i, q_index) +
                                 (penalty)*fe_faces1.shape_value(i, q_index) *
                                   fe_faces1.shape_value(j, q_index)) *
                                fe_faces1.JxW(q_index);
                            }
                        }
                    }

                  neigh_polytope->get_dof_indices(local_dof_indices_neighbor);

                  constraints.distribute_local_to_global(M11,
                                                         local_dof_indices,
                                                         system_matrix);
                  constraints.distribute_local_to_global(
                    M12,
                    local_dof_indices,
                    local_dof_indices_neighbor,
                    system_matrix);
                  constraints.distribute_local_to_global(
                    M21,
                    local_dof_indices_neighbor,
                    local_dof_indices,
                    system_matrix);
                  constraints.distribute_local_to_global(
                    M22, local_dof_indices_neighbor, system_matrix);
                } // Loop only once through internal faces
            }
        } // Loop over faces of current cell

      // Distribute the local contributions to the global system.
      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    } // Loop over cells
}

// Solve the linear system by means of a sparse direct solver.
template <int dim>
void
Poisson<dim>::solve()
{
  SparseDirectUMFPACK A_direct;
  A_direct.initialize(system_matrix);
  A_direct.vmult(solution, system_rhs);
}


// Write VTU output and compute the global $L^2$ and $H^1$-seminorm
// errors of the agglomerated DG approximation.
template <int dim>
void
Poisson<dim>::output_results()
{
  {
    std::string partitioner;
    if (partitioner_type == PartitionerType::metis)
      partitioner = "metis";
    else if (partitioner_type == PartitionerType::rtree)
      partitioner = "rtree";
    else
      partitioner = "no_partitioning";

    const std::string filename = "interpolated_solution_" + partitioner + "_" +
                                 std::to_string(n_subdomains) + ".vtu";
    std::ofstream output(filename);

    DataOut<dim>   data_out;
    Vector<double> interpolated_solution;
    PolyUtils::interpolate_to_fine_grid(*ah,
                                        interpolated_solution,
                                        solution,
                                        true /*on_the_fly*/);
    data_out.attach_dof_handler(ah->output_dh);
    data_out.add_data_vector(interpolated_solution,
                             "u",
                             DataOut<dim>::type_dof_data);

    Vector<float> agglo_idx(tria.n_active_cells());

    // Mark fine cells belonging to the same agglomerate.
    for (const auto &polytope : ah->polytope_iterators())
      {
        const types::global_cell_index polytope_index = polytope->index();
        const auto &patch_of_cells = polytope->get_agglomerate(); // Fine cells
        for (const auto &cell : patch_of_cells) // Mark all fine cells belonging to the current agglomerate.
          agglo_idx[cell->active_cell_index()] = polytope_index;
      }

    data_out.add_data_vector(agglo_idx,
                             "agglo_idx",
                             DataOut<dim>::type_cell_data);

    data_out.build_patches(mapping);
    data_out.write_vtu(output);

    std::vector<double> errors;
    PolyUtils::compute_global_error(*ah,
                                    solution,
                                    *analytical_solution,
                                    {VectorTools::L2_norm,
                                     VectorTools::H1_seminorm},
                                    errors); // Compute the global L2 and H1-seminorm errors.
    l2_err     = errors[0];
    semih1_err = errors[1];
  }
}


// Return the number of degrees of freedom on the agglomerated mesh.
template <int dim>
inline types::global_dof_index
Poisson<dim>::get_n_dofs() const
{
  return ah->n_dofs();
}


// Return the pair consisting of the $L^2$ error and the $H^1$-seminorm error of the numerical solution.
template <int dim>
inline std::pair<double, double>
Poisson<dim>::get_error() const
{
  return std::make_pair(l2_err, semih1_err);
}


// Run the full workflow: mesh generation, agglomeration setup,
// assembly, solution, and postprocessing.
template <int dim>
void
Poisson<dim>::run()
{
  make_grid();
  setup_agglomeration();
  auto start = std::chrono::high_resolution_clock::now();
  assemble_system();
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
    std::chrono::duration_cast<std::chrono::seconds>(stop - start);

  std::cout << "Time taken by assemble_system(): " << duration.count()
            << " seconds" << std::endl;
  solve();
  output_results();
}

// Driver code.
int
main()
{
  ConvergenceInfo convergence_info;

  for (unsigned int fe_degree : {1}) //, 2, 3})
    {
      std::cout << "Running with FE degree: " << fe_degree << std::endl;
      Poisson<2> poisson_problem{PartitionerType::rtree, //  Three choices: metis, rtree and no_partition
                                 4 /* extraction_level */,
                                 91 /* n_subdomains */,
                                 fe_degree};
      poisson_problem.run();
      convergence_info.add(
        std::make_pair<types::global_dof_index, std::pair<double, double>>(
          poisson_problem.get_n_dofs(), poisson_problem.get_error()));
      std::cout << std::endl;
    }

  std::cout << "Convergence table:" << std::endl;
  convergence_info.print();
  std::cout << std::endl;

  return 0;
}
