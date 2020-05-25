/* ---------------------------------------------------------------------
 *
 * This file is part of the deal.II Code Gallery.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Ilona Ambartsumyan, Eldar Khattatov, University of Pittsburgh, 2018
 */


// @sect3{Include files}

// As usual, the list of necessary header files. There is not
// much new here, the files are included in order
// base-lac-grid-dofs-numerics followed by the C++ headers.
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <unordered_map>

// This is a header needed for the purposes of the
// multipoint flux mixed method, as it declares the
// new enhanced Raviart-Thomas finite element.
#include <deal.II/fe/fe_rt_bubbles.h>

// For the sake of readability, the classes representing
// data, i.e. RHS, BCs, permeability tensor and the exact
// solution are placed in a file data.h which is included
// here
#include "data.h"

// As always the program is in the namespace of its own with
// the deal.II classes and functions imported into it
namespace MFMFE
{
  using namespace dealii;

  // @sect3{Definition of multipoint flux assembly data structures}

  // The main idea of the MFMFE method is to perform local elimination
  // of the velocity variables in order to obtain the resulting
  // pressure system. Since in deal.II assembly happens cell-wise,
  // some extra work needs to be done in order to get the local
  // mass matrices $A_i$ and the corresponding to them $B_i$.
  namespace DataStructures
  {
    // This will be achieved by assembling cell-wise, but instead of placing
    // the terms into a global system matrix, they will populate node-associated
    // full matrices. For this, a data structure with fast lookup is crucial, hence
    // the hash table, with the keys as Point<dim>
    template <int dim>
    struct hash_points
    {
      size_t operator()(const Point<dim> &p) const
      {
        size_t h1,h2,h3;
        h1 = std::hash<double>()(p[0]);

        switch (dim)
          {
          case 1:
            return h1;
          case 2:
            h2 = std::hash<double>()(p[1]);
            return (h1 ^ h2);
          case 3:
            h2 = std::hash<double>()(p[1]);
            h3 = std::hash<double>()(p[2]);
            return (h1 ^ (h2 << 1)) ^ h3;
          default:
            Assert(false, ExcNotImplemented());
          }
      }
    };

    // Here, the actual hash-tables are defined. We use the C++ STL <code>unordered_map</code>,
    // with the hash function specified above. For convenience these are aliased as follows
    template <int dim>
    using PointToMatrixMap = std::unordered_map<Point<dim>, std::map<std::pair<types::global_dof_index,types::global_dof_index>, double>, hash_points<dim>>;

    template <int dim>
    using PointToVectorMap = std::unordered_map<Point<dim>, std::map<types::global_dof_index, double>, hash_points<dim>>;

    template <int dim>
    using PointToIndexMap = std::unordered_map<Point<dim>, std::set<types::global_dof_index>, hash_points<dim>>;

    // Next, since this particular program allows for the use of
    // multiple threads, the helper CopyData structures
    // are defined. There are two kinds of these, one is used
    // for the copying cell-wise contributions to the corresponging
    // node-associated data structures...
    template <int dim>
    struct NodeAssemblyCopyData
    {
      PointToMatrixMap<dim> cell_mat;
      PointToVectorMap<dim> cell_vec;
      PointToIndexMap<dim> local_pres_indices;
      PointToIndexMap<dim> local_vel_indices;
      std::vector<types::global_dof_index> local_dof_indices;
    };

    // ... and the other one for the actual process of
    // local velocity elimination and assembling the global
    // pressure system:
    template <int dim>
    struct NodeEliminationCopyData
    {
      FullMatrix<double> node_pres_matrix;
      Vector<double>     node_pres_rhs;
      FullMatrix<double> Ainverse;
      FullMatrix<double> pressure_matrix;
      Vector<double>     velocity_rhs;
      Vector<double>     vertex_vel_solution;
      Point<dim>         p;
    };

    // Similarly, two ScratchData classes are defined.
    // One for the assembly part, where we need
    // FEValues, FEFaceValues, Quadrature and storage
    // for the basis fuctions...
    template <int dim>
    struct NodeAssemblyScratchData
    {
      NodeAssemblyScratchData (const FiniteElement<dim> &fe,
                               const Triangulation<dim> &tria,
                               const Quadrature<dim>    &quad,
                               const Quadrature<dim-1>  &f_quad);

      NodeAssemblyScratchData (const NodeAssemblyScratchData &scratch_data);

      FEValues<dim>       fe_values;
      FEFaceValues<dim>   fe_face_values;
      std::vector<unsigned int>    n_faces_at_vertex;

      const unsigned long num_cells;

      std::vector<Tensor<2,dim>> k_inverse_values;
      std::vector<double> rhs_values;
      std::vector<double> pres_bc_values;

      std::vector<Tensor<1,dim> > phi_u;
      std::vector<double>         div_phi_u;
      std::vector<double>         phi_p;
    };

    template <int dim>
    NodeAssemblyScratchData<dim>::
    NodeAssemblyScratchData (const FiniteElement<dim> &fe,
                             const Triangulation<dim> &tria,
                             const Quadrature<dim> &quad,
                             const Quadrature<dim-1> &f_quad)
      :
      fe_values (fe,
                 quad,
                 update_values   | update_gradients |
                 update_quadrature_points | update_JxW_values),
      fe_face_values (fe,
                      f_quad,
                      update_values     | update_quadrature_points   |
                      update_JxW_values | update_normal_vectors),
      num_cells(tria.n_active_cells()),
      k_inverse_values(quad.size()),
      rhs_values(quad.size()),
      pres_bc_values(f_quad.size()),
      phi_u(fe.dofs_per_cell),
      div_phi_u(fe.dofs_per_cell),
      phi_p(fe.dofs_per_cell)
    {
      n_faces_at_vertex.resize(tria.n_vertices(), 0);
      typename Triangulation<dim>::active_face_iterator face = tria.begin_active_face(), endf = tria.end_face();

      for (; face != endf; ++face)
        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
          n_faces_at_vertex[face->vertex_index(v)] += 1;
    }

    template <int dim>
    NodeAssemblyScratchData<dim>::
    NodeAssemblyScratchData (const NodeAssemblyScratchData &scratch_data)
      :
      fe_values (scratch_data.fe_values.get_fe(),
                 scratch_data.fe_values.get_quadrature(),
                 update_values   | update_gradients |
                 update_quadrature_points | update_JxW_values),
      fe_face_values (scratch_data.fe_face_values.get_fe(),
                      scratch_data.fe_face_values.get_quadrature(),
                      update_values     | update_quadrature_points   |
                      update_JxW_values | update_normal_vectors),
      n_faces_at_vertex(scratch_data.n_faces_at_vertex),
      num_cells(scratch_data.num_cells),
      k_inverse_values(scratch_data.k_inverse_values),
      rhs_values(scratch_data.rhs_values),
      pres_bc_values(scratch_data.pres_bc_values),
      phi_u(scratch_data.phi_u),
      div_phi_u(scratch_data.div_phi_u),
      phi_p(scratch_data.phi_p)
    {}

    // ...and the other, simpler one, for the velocity elimination and recovery
    struct VertexEliminationScratchData
    {
      VertexEliminationScratchData () = default;
      VertexEliminationScratchData (const VertexEliminationScratchData &scratch_data);

      FullMatrix<double> velocity_matrix;
      Vector<double> pressure_rhs;

      Vector<double> local_pressure_solution;
      Vector<double> tmp_rhs1;
      Vector<double> tmp_rhs2;
      Vector<double> tmp_rhs3;
    };

    VertexEliminationScratchData::
    VertexEliminationScratchData (const VertexEliminationScratchData &scratch_data)
      :
      velocity_matrix(scratch_data.velocity_matrix),
      pressure_rhs(scratch_data.pressure_rhs),
      local_pressure_solution(scratch_data.local_pressure_solution),
      tmp_rhs1(scratch_data.tmp_rhs1),
      tmp_rhs2(scratch_data.tmp_rhs2),
      tmp_rhs3(scratch_data.tmp_rhs3)
    {}
  }



  // @sect3{The <code>MultipointMixedDarcyProblem</code> class template}

  // The main class, besides the constructor and destructor, has only one public member
  // <code>run()</code>, similarly to the tutorial programs. The private members can
  // be grouped into the ones that are used for the cell-wise assembly, vertex elimination,
  // pressure solve, vertex velocity recovery and postprocessing. Apart from the
  // MFMFE-specific data structures, the rest of the members should look familiar.
  template <int dim>
  class MultipointMixedDarcyProblem
  {
  public:
    MultipointMixedDarcyProblem (const unsigned int degree);
    ~MultipointMixedDarcyProblem ();
    void run (const unsigned int refine);
  private:
    void assemble_system_cell (const typename DoFHandler<dim>::active_cell_iterator &cell,
                               DataStructures::NodeAssemblyScratchData<dim>       &scratch_data,
                               DataStructures::NodeAssemblyCopyData<dim>          &copy_data);
    void copy_cell_to_node(const DataStructures::NodeAssemblyCopyData<dim> &copy_data);
    void node_assembly();
    void make_cell_centered_sp ();
    void nodal_elimination(const typename DataStructures::PointToMatrixMap<dim>::iterator &n_it,
                           DataStructures::VertexEliminationScratchData &scratch_data,
                           DataStructures::NodeEliminationCopyData<dim> &copy_data);
    void copy_node_to_system(const DataStructures::NodeEliminationCopyData<dim> &copy_data);
    void pressure_assembly ();
    void solve_pressure ();
    void velocity_assembly (const typename DataStructures::PointToMatrixMap<dim>::iterator &n_it,
                            DataStructures::VertexEliminationScratchData                 &scratch_data,
                            DataStructures::NodeEliminationCopyData<dim>               &copy_data);
    void copy_node_velocity_to_global(const DataStructures::NodeEliminationCopyData<dim> &copy_data);
    void velocity_recovery ();
    void reset_data_structures ();
    void compute_errors (const unsigned int cycle);
    void output_results (const unsigned int cycle,  const unsigned int refine);

    const unsigned int  degree;
    Triangulation<dim>  triangulation;
    FESystem<dim>       fe;
    DoFHandler<dim>     dof_handler;
    BlockVector<double> solution;

    SparsityPattern cell_centered_sp;
    SparseMatrix<double> pres_system_matrix;
    Vector<double> pres_rhs;

    std::unordered_map<Point<dim>, FullMatrix<double>, DataStructures::hash_points<dim>> pressure_matrix;
    std::unordered_map<Point<dim>, FullMatrix<double>, DataStructures::hash_points<dim>> A_inverse;
    std::unordered_map<Point<dim>, Vector<double>, DataStructures::hash_points<dim>> velocity_rhs;

    DataStructures::PointToMatrixMap<dim> node_matrix;
    DataStructures::PointToVectorMap<dim> node_rhs;

    DataStructures::PointToIndexMap<dim> pressure_indices;
    DataStructures::PointToIndexMap<dim> velocity_indices;

    unsigned long n_v, n_p;

    Vector<double> pres_solution;
    Vector<double> vel_solution;

    ConvergenceTable convergence_table;
    TimerOutput      computing_timer;
  };

  // @sect4{Constructor and destructor, <code>reset_data_structures</code>}

  // In the constructor of this class, we store the value that was
  // passed in concerning the degree of the finite elements we shall use (a
  // degree of one would mean the use of @ref FE_RT_Bubbles(1) and @ref FE_DGQ(0)),
  // and then construct the vector valued element belonging to the space $V_h^k$ described
  // in the introduction. The constructor also takes care of initializing the
  // computing timer, as it is of interest for us how well our method performs.
  template <int dim>
  MultipointMixedDarcyProblem<dim>::MultipointMixedDarcyProblem (const unsigned int degree)
    :
    degree(degree),
    fe(FE_RT_Bubbles<dim>(degree), 1,
       FE_DGQ<dim>(degree-1), 1),
    dof_handler(triangulation),
    computing_timer(std::cout, TimerOutput::summary,
                    TimerOutput::wall_times)
  {}


  // The destructor clears the <code>dof_handler</code> and
  // all of the data structures we used for the method.
  template <int dim>
  MultipointMixedDarcyProblem<dim>::~MultipointMixedDarcyProblem()
  {
    reset_data_structures ();
    dof_handler.clear();
  }


  // This method clears all the data that was used after one refinement
  // cycle.
  template <int dim>
  void MultipointMixedDarcyProblem<dim>::reset_data_structures ()
  {
    pressure_indices.clear();
    velocity_indices.clear();
    velocity_rhs.clear();
    A_inverse.clear();
    pressure_matrix.clear();
    node_matrix.clear();
    node_rhs.clear();
  }


  // @sect4{Cell-wise assembly and creation of the local, nodal-based data structures}

  // First, the function that copies local cell contributions to the corresponding nodal
  // matrices and vectors is defined. It places the values obtained from local cell integration
  // into the correct place in a matrix/vector corresponging to a specific node.
  template <int dim>
  void MultipointMixedDarcyProblem<dim>::copy_cell_to_node(const DataStructures::NodeAssemblyCopyData<dim> &copy_data)
  {
    for (auto m : copy_data.cell_mat)
      {
        for (auto p : m.second)
          node_matrix[m.first][p.first] += p.second;

        for (auto p : copy_data.cell_vec.at(m.first))
          node_rhs[m.first][p.first] += p.second;

        for (auto p : copy_data.local_pres_indices.at(m.first))
          pressure_indices[m.first].insert(p);

        for (auto p : copy_data.local_vel_indices.at(m.first))
          velocity_indices[m.first].insert(p);
      }
  }



  // Second, the function that does the cell assembly is defined. While it is
  // similar to the tutorial programs in a way it uses scrath and copy data
  // structures, the need to localize the DOFs leads to several differences.
  template <int dim>
  void MultipointMixedDarcyProblem<dim>::
  assemble_system_cell (const typename DoFHandler<dim>::active_cell_iterator &cell,
                        DataStructures::NodeAssemblyScratchData<dim> &scratch_data,
                        DataStructures::NodeAssemblyCopyData<dim>    &copy_data)
  {
    copy_data.cell_mat.clear();
    copy_data.cell_vec.clear();
    copy_data.local_vel_indices.clear();
    copy_data.local_pres_indices.clear();

    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int n_q_points      = scratch_data.fe_values.get_quadrature().size();
    const unsigned int n_face_q_points = scratch_data.fe_face_values.get_quadrature().size();

    copy_data.local_dof_indices.resize(dofs_per_cell);
    cell->get_dof_indices (copy_data.local_dof_indices);

    scratch_data.fe_values.reinit (cell);

    const KInverse<dim> k_inverse;
    const RightHandSide<dim> rhs;
    const PressureBoundaryValues<dim> pressure_bc;

    k_inverse.value_list (scratch_data.fe_values.get_quadrature_points(), scratch_data.k_inverse_values);
    rhs.value_list(scratch_data.fe_values.get_quadrature_points(), scratch_data.rhs_values);

    const FEValuesExtractors::Vector velocity (0);
    const FEValuesExtractors::Scalar pressure (dim);

    const unsigned int n_vel = dim*Utilities::pow(degree+1,dim);
    std::unordered_map<unsigned int, std::unordered_map<unsigned int, double>> div_map;

    // One, we need to be able to assemble the communication between velocity and
    // pressure variables and put it on the right place in our final, local version
    // of the B matrix. This is a little messy, as such communication is not in fact
    // local, so we do it in two steps. First, we compute all relevant LHS and RHS
    for (unsigned int q=0; q<n_q_points; ++q)
      {
        const Point<dim> p = scratch_data.fe_values.quadrature_point(q);

        for (unsigned int k=0; k<dofs_per_cell; ++k)
          {
            scratch_data.phi_u[k] = scratch_data.fe_values[velocity].value(k, q);
            scratch_data.div_phi_u[k] = scratch_data.fe_values[velocity].divergence (k, q);
            scratch_data.phi_p[k] = scratch_data.fe_values[pressure].value (k, q);
          }

        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            for (unsigned int j=n_vel; j<dofs_per_cell; ++j)
              {
                double div_term = (- scratch_data.div_phi_u[i] * scratch_data.phi_p[j]
                                   - scratch_data.phi_p[i] * scratch_data.div_phi_u[j]) * scratch_data.fe_values.JxW(q);

                if (std::abs(div_term) > 1.e-12)
                  div_map[i][j] += div_term;
              }

            double source_term = -scratch_data.phi_p[i] * scratch_data.rhs_values[q] * scratch_data.fe_values.JxW(q);

            if (std::abs(scratch_data.phi_p[i]) > 1.e-12 || std::abs(source_term) > 1.e-12)
              copy_data.cell_vec[p][copy_data.local_dof_indices[i]] += source_term;
          }
      }

    // Then, by making another pass, we compute the mass matrix terms and incorporate the
    // divergence form and RHS accordingly. This second pass, allows us to know where
    // the total contribution will be put in the nodal data structures, as with this
    // choice of quadrature rule and finite element only the basis functions corresponding
    // to the same quadrature points yield non-zero contribution.
    for (unsigned int q=0; q<n_q_points; ++q)
      {
        std::set<types::global_dof_index> vel_indices;
        const Point<dim> p = scratch_data.fe_values.quadrature_point(q);

        for (unsigned int k=0; k<dofs_per_cell; ++k)
          {
            scratch_data.phi_u[k] = scratch_data.fe_values[velocity].value(k, q);
            scratch_data.div_phi_u[k]     = scratch_data.fe_values[velocity].divergence (k, q);
            scratch_data.phi_p[k]         = scratch_data.fe_values[pressure].value (k, q);
          }

        for (unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int j=i; j<dofs_per_cell; ++j)
            {
              double mass_term = scratch_data.phi_u[i]
                                 * scratch_data.k_inverse_values[q]
                                 * scratch_data.phi_u[j]
                                 * scratch_data.fe_values.JxW(q);

              if (std::abs(mass_term) > 1.e-12)
                {
                  copy_data.cell_mat[p][std::make_pair(copy_data.local_dof_indices[i], copy_data.local_dof_indices[j])] +=
                    mass_term;
                  vel_indices.insert(i);
                  copy_data.local_vel_indices[p].insert(copy_data.local_dof_indices[j]);
                }
            }

        for (auto i : vel_indices)
          for (auto el : div_map[i])
            if (std::abs(el.second) > 1.e-12)
              {
                copy_data.cell_mat[p][std::make_pair(copy_data.local_dof_indices[i],
                                                     copy_data.local_dof_indices[el.first])] += el.second;
                copy_data.local_pres_indices[p].insert(copy_data.local_dof_indices[el.first]);
              }
      }

    // The pressure boundary conditions are computed as in step-20,
    std::map<types::global_dof_index,double> pres_bc;
    for (unsigned int face_no=0;
         face_no<GeometryInfo<dim>::faces_per_cell;
         ++face_no)
      if (cell->at_boundary(face_no))
        {
          scratch_data.fe_face_values.reinit (cell, face_no);
          pressure_bc.value_list(scratch_data.fe_face_values.get_quadrature_points(), scratch_data.pres_bc_values);

          for (unsigned int q=0; q<n_face_q_points; ++q)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                double tmp = -(scratch_data.fe_face_values[velocity].value(i, q) *
                               scratch_data.fe_face_values.normal_vector(q) *
                               scratch_data.pres_bc_values[q] *
                               scratch_data.fe_face_values.JxW(q));

                if (std::abs(tmp) > 1.e-12)
                  pres_bc[copy_data.local_dof_indices[i]] += tmp;
              }
        }

    // ...but we distribute them to the corresponding nodal data structures
    for (auto m : copy_data.cell_vec)
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        if (std::abs(pres_bc[copy_data.local_dof_indices[i]]) > 1.e-12)
          copy_data.cell_vec[m.first][copy_data.local_dof_indices[i]] += pres_bc[copy_data.local_dof_indices[i]];
  }


  // Finally, <code>node_assembly()</code> takes care of all the
  // local computations via WorkStream mechanism. Notice that the choice
  // of the quadrature rule here is dictated by the formulation of the
  // method. It has to be <code>degree+1</code> points Gauss-Lobatto
  // for the volume integrals and <code>degree</code> for the face ones,
  // as mentioned in the introduction.
  template <int dim>
  void MultipointMixedDarcyProblem<dim>::node_assembly()
  {
    TimerOutput::Scope t(computing_timer, "Nodal assembly");

    dof_handler.distribute_dofs(fe);
    DoFRenumbering::component_wise (dof_handler);
    std::vector<types::global_dof_index> dofs_per_component (dim+1);
    DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);

    QGaussLobatto<dim> quad(degree+1);
    QGauss<dim-1> face_quad(degree);

    n_v = dofs_per_component[0];
    n_p = dofs_per_component[dim];

    pres_rhs.reinit(n_p);

    WorkStream::run(dof_handler.begin_active(),
                    dof_handler.end(),
                    *this,
                    &MultipointMixedDarcyProblem::assemble_system_cell,
                    &MultipointMixedDarcyProblem::copy_cell_to_node,
                    DataStructures::NodeAssemblyScratchData<dim>(fe, triangulation,quad,face_quad),
                    DataStructures::NodeAssemblyCopyData<dim>());
  }

  // @sect4{Making the sparsity pattern}

  // Having computed all the local contributions, we actually have
  // all the information needed to make a cell-centered sparsity
  // pattern manually. We do this here, because @ref SparseMatrixEZ
  // leads to a slower solution.
  template <int dim>
  void MultipointMixedDarcyProblem<dim>::make_cell_centered_sp()
  {
    TimerOutput::Scope t(computing_timer, "Make sparsity pattern");
    DynamicSparsityPattern dsp(n_p, n_p);

    std::set<types::global_dof_index>::iterator pi_it, pj_it;
    unsigned int i, j;
    for (auto el : node_matrix)
      for (pi_it = pressure_indices[el.first].begin(), i = 0;
           pi_it != pressure_indices[el.first].end();
           ++pi_it, ++i)
        for (pj_it = pi_it, j = 0;
             pj_it != pressure_indices[el.first].end();
             ++pj_it, ++j)
          dsp.add(*pi_it - n_v, *pj_it - n_v);


    dsp.symmetrize();
    cell_centered_sp.copy_from(dsp);
    pres_system_matrix.reinit (cell_centered_sp);
  }


  // @sect4{The local elimination procedure}

  // This function finally performs the local elimination procedure.
  // Mathematically, it follows the same idea as in computing the
  // Schur complement (as mentioned in the introduction) but we do
  // so locally. Namely, local velocity DOFs are expressed in terms
  // of corresponding pressure values, and then used for the local
  // pressure systems.
  template <int dim>
  void MultipointMixedDarcyProblem<dim>::
  nodal_elimination(const typename DataStructures::PointToMatrixMap<dim>::iterator &n_it,
                    DataStructures::VertexEliminationScratchData &scratch_data,
                    DataStructures::NodeEliminationCopyData<dim> &copy_data)
  {
    unsigned int n_edges = velocity_indices.at((*n_it).first).size();
    unsigned int n_cells = pressure_indices.at((*n_it).first).size();

    scratch_data.velocity_matrix.reinit(n_edges,n_edges);
    copy_data.pressure_matrix.reinit(n_edges,n_cells);

    copy_data.velocity_rhs.reinit(n_edges);
    scratch_data.pressure_rhs.reinit(n_cells);

    {
      std::set<types::global_dof_index>::iterator vi_it, vj_it, p_it;
      unsigned int i;
      for (vi_it = velocity_indices.at((*n_it).first).begin(), i = 0;
           vi_it != velocity_indices.at((*n_it).first).end();
           ++vi_it, ++i)
        {
          unsigned int j;
          for (vj_it = velocity_indices.at((*n_it).first).begin(), j = 0;
               vj_it != velocity_indices.at((*n_it).first).end();
               ++vj_it, ++j)
            {
              scratch_data.velocity_matrix.add(i, j, node_matrix[(*n_it).first][std::make_pair(*vi_it, *vj_it)]);
              if (j != i)
                scratch_data.velocity_matrix.add(j, i, node_matrix[(*n_it).first][std::make_pair(*vi_it, *vj_it)]);
            }

          for (p_it = pressure_indices.at((*n_it).first).begin(), j = 0;
               p_it != pressure_indices.at((*n_it).first).end();
               ++p_it, ++j)
            copy_data.pressure_matrix.add(i, j, node_matrix[(*n_it).first][std::make_pair(*vi_it, *p_it)]);

          copy_data.velocity_rhs(i) += node_rhs.at((*n_it).first)[*vi_it];
        }

      for (p_it = pressure_indices.at((*n_it).first).begin(), i = 0;
           p_it != pressure_indices.at((*n_it).first).end();
           ++p_it, ++i)
        scratch_data.pressure_rhs(i) += node_rhs.at((*n_it).first)[*p_it];
    }

    copy_data.Ainverse.reinit(n_edges,n_edges);

    scratch_data.tmp_rhs1.reinit(n_edges);
    scratch_data.tmp_rhs2.reinit(n_edges);
    scratch_data.tmp_rhs3.reinit(n_cells);

    copy_data.Ainverse.invert(scratch_data.velocity_matrix);
    copy_data.node_pres_matrix.reinit(n_cells, n_cells);
    copy_data.node_pres_rhs = scratch_data.pressure_rhs;

    copy_data.node_pres_matrix = 0;
    copy_data.node_pres_matrix.triple_product(copy_data.Ainverse,
                                              copy_data.pressure_matrix,
                                              copy_data.pressure_matrix, true, false);

    copy_data.Ainverse.vmult(scratch_data.tmp_rhs1, copy_data.velocity_rhs, false);
    copy_data.pressure_matrix.Tvmult(scratch_data.tmp_rhs3, scratch_data.tmp_rhs1, false);
    copy_data.node_pres_rhs *= -1.0;
    copy_data.node_pres_rhs += scratch_data.tmp_rhs3;

    copy_data.p = (*n_it).first;
  }


  // Each node's pressure system is then distributed to a global pressure
  // system, using the indices we computed in the previous stages.
  template <int dim>
  void MultipointMixedDarcyProblem<dim>::
  copy_node_to_system(const DataStructures::NodeEliminationCopyData<dim> &copy_data)
  {
    A_inverse[copy_data.p] = copy_data.Ainverse;
    pressure_matrix[copy_data.p] = copy_data.pressure_matrix;
    velocity_rhs[copy_data.p] = copy_data.velocity_rhs;

    {
      std::set<types::global_dof_index>::iterator pi_it, pj_it;
      unsigned int i;
      for (pi_it = pressure_indices[copy_data.p].begin(), i = 0;
           pi_it != pressure_indices[copy_data.p].end();
           ++pi_it, ++i)
        {
          unsigned int j;
          for (pj_it = pressure_indices[copy_data.p].begin(), j = 0;
               pj_it != pressure_indices[copy_data.p].end();
               ++pj_it, ++j)
            pres_system_matrix.add(*pi_it - n_v, *pj_it - n_v, copy_data.node_pres_matrix(i, j));

          pres_rhs(*pi_it - n_v) += copy_data.node_pres_rhs(i);
        }
    }
  }


  // The @ref WorkStream mechanism is again used for the assembly
  // of the global system for the pressure variable, where the
  // previous functions are used to perform local computations.
  template <int dim>
  void MultipointMixedDarcyProblem<dim>::pressure_assembly()
  {
    TimerOutput::Scope t(computing_timer, "Pressure matrix assembly");

    QGaussLobatto<dim> quad(degree+1);
    QGauss<dim-1> face_quad(degree);

    pres_rhs.reinit(n_p);

    WorkStream::run(node_matrix.begin(),
                    node_matrix.end(),
                    *this,
                    &MultipointMixedDarcyProblem::nodal_elimination,
                    &MultipointMixedDarcyProblem::copy_node_to_system,
                    DataStructures::VertexEliminationScratchData(),
                    DataStructures::NodeEliminationCopyData<dim>());
  }



  // @sect4{Velocity solution recovery}

  // After solving for the pressure variable, we want to follow
  // the above procedure backwards, in order to obtain the
  // velocity solution (again, this is similar in nature to the
  // Schur complement approach, see step-20, but here it is done
  // locally at each node). We have almost everything computed and
  // stored already, including inverses of local mass matrices,
  // so the following is a relatively straightforward implementation.
  template <int dim>
  void MultipointMixedDarcyProblem<dim>::
  velocity_assembly (const typename DataStructures::PointToMatrixMap<dim>::iterator &n_it,
                     DataStructures::VertexEliminationScratchData                   &scratch_data,
                     DataStructures::NodeEliminationCopyData<dim>                   &copy_data)
  {
    unsigned int n_edges = velocity_indices.at((*n_it).first).size();
    unsigned int n_cells = pressure_indices.at((*n_it).first).size();

    scratch_data.tmp_rhs1.reinit(n_edges);
    scratch_data.tmp_rhs2.reinit(n_edges);
    scratch_data.tmp_rhs3.reinit(n_cells);
    scratch_data.local_pressure_solution.reinit(n_cells);

    copy_data.vertex_vel_solution.reinit(n_edges);

    std::set<types::global_dof_index>::iterator p_it;
    unsigned int i;

    for (p_it = pressure_indices[(*n_it).first].begin(), i = 0;
         p_it != pressure_indices[(*n_it).first].end();
         ++p_it, ++i)
      scratch_data.local_pressure_solution(i) = pres_solution(*p_it - n_v);

    pressure_matrix[(*n_it).first].vmult(scratch_data.tmp_rhs2, scratch_data.local_pressure_solution, false);
    scratch_data.tmp_rhs2 *= -1.0;
    scratch_data.tmp_rhs2+=velocity_rhs[(*n_it).first];
    A_inverse[(*n_it).first].vmult(copy_data.vertex_vel_solution, scratch_data.tmp_rhs2, false);

    copy_data.p = (*n_it).first;
  }


  // Copy nodal velocities to a global solution vector by using
  // local computations and indices from early stages.
  template <int dim>
  void MultipointMixedDarcyProblem<dim>::
  copy_node_velocity_to_global(const DataStructures::NodeEliminationCopyData<dim> &copy_data)
  {
    std::set<types::global_dof_index>::iterator vi_it;
    unsigned int i;

    for (vi_it = velocity_indices[copy_data.p].begin(), i = 0;
         vi_it != velocity_indices[copy_data.p].end();
         ++vi_it, ++i)
      vel_solution(*vi_it) += copy_data.vertex_vel_solution(i);
  }


  // Use @ref WorkStream to run everything concurrently.
  template <int dim>
  void MultipointMixedDarcyProblem<dim>::velocity_recovery()
  {
    TimerOutput::Scope t(computing_timer, "Velocity solution recovery");

    QGaussLobatto<dim> quad(degree+1);
    QGauss<dim-1> face_quad(degree);

    vel_solution.reinit(n_v);

    WorkStream::run(node_matrix.begin(),
                    node_matrix.end(),
                    *this,
                    &MultipointMixedDarcyProblem::velocity_assembly,
                    &MultipointMixedDarcyProblem::copy_node_velocity_to_global,
                    DataStructures::VertexEliminationScratchData(),
                    DataStructures::NodeEliminationCopyData<dim>());

    solution.reinit(2);
    solution.block(0) = vel_solution;
    solution.block(1) = pres_solution;
    solution.collect_sizes();
  }



  // @sect4{Pressure system solver}

  // The solver part is trivial. We use the CG solver with no
  // preconditioner for simplicity.
  template <int dim>
  void MultipointMixedDarcyProblem<dim>::solve_pressure()
  {
    TimerOutput::Scope t(computing_timer, "Pressure CG solve");

    pres_solution.reinit(n_p);

    SolverControl solver_control (static_cast<int>(2.0*n_p), 1e-10);
    SolverCG<> solver (solver_control);

    PreconditionIdentity identity;
    solver.solve(pres_system_matrix, pres_solution, pres_rhs, identity);
  }



  // @sect3{Postprocessing}

  // We have two postprocessing steps here, first one computes the
  // errors in order to populate the convergence tables. The other
  // one takes care of the output of the solutions in <code>.vtk</code>
  // format.

  // @sect4{Compute errors}

  // The implementation of this function is almost identical to step-20.
  // We use @ref ComponentSelectFunction as masks to use the right
  // solution component (velocity or pressure) and @ref integrate_difference
  // to compute the errors. Since we also want to compute Hdiv seminorm of the
  // velocity error, one must provide gradients in the <code>ExactSolution</code>
  // class implementation to avoid exceptions. The only noteworthy thing here
  // is that we again use lower order quadrature rule instead of projecting the
  // solution to an appropriate space in order to show superconvergence, which is
  // mathematically justified.
  template <int dim>
  void MultipointMixedDarcyProblem<dim>::compute_errors(const unsigned cycle)
  {
    TimerOutput::Scope t(computing_timer, "Compute errors");

    const ComponentSelectFunction<dim> pressure_mask(dim, dim+1);
    const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim), dim+1);

    ExactSolution<dim> exact_solution;

    Vector<double> cellwise_errors (triangulation.n_active_cells());

    QTrapez<1> q_trapez;
    QIterated<dim> quadrature(q_trapez,degree+2);
    QGauss<dim> quadrature_super(degree);

    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &pressure_mask);
    const double p_l2_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature_super,
                                       VectorTools::L2_norm,
                                       &pressure_mask);
    const double p_l2_mid_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &velocity_mask);
    const double u_l2_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::Hdiv_seminorm,
                                       &velocity_mask);
    const double u_hd_error = cellwise_errors.l2_norm();

    const unsigned int n_active_cells=triangulation.n_active_cells();
    const unsigned int n_dofs=dof_handler.n_dofs();

    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", n_active_cells);
    convergence_table.add_value("dofs", n_dofs);
    convergence_table.add_value("Velocity,L2", u_l2_error);
    convergence_table.add_value("Velocity,Hdiv", u_hd_error);
    convergence_table.add_value("Pressure,L2", p_l2_error);
    convergence_table.add_value("Pressure,L2-nodal", p_l2_mid_error);
  }



  // @sect4{Output results}

  // This function also follows the same idea as in step-20 tutorial
  // program. The only modification to it is the part involving
  // a convergence table.
  template <int dim>
  void MultipointMixedDarcyProblem<dim>::output_results(const unsigned int cycle, const unsigned int refine)
  {
    TimerOutput::Scope t(computing_timer, "Output results");

    std::vector<std::string> solution_names(dim, "u");
    solution_names.push_back ("p");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation (dim, DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.add_data_vector (dof_handler, solution, solution_names, interpretation);
    data_out.build_patches ();

    std::ofstream output ("solution" + std::to_string(dim) + "d-" + std::to_string(cycle) + ".vtk");
    data_out.write_vtk (output);

    convergence_table.set_precision("Velocity,L2", 3);
    convergence_table.set_precision("Velocity,Hdiv", 3);
    convergence_table.set_precision("Pressure,L2", 3);
    convergence_table.set_precision("Pressure,L2-nodal", 3);
    convergence_table.set_scientific("Velocity,L2", true);
    convergence_table.set_scientific("Velocity,Hdiv", true);
    convergence_table.set_scientific("Pressure,L2", true);
    convergence_table.set_scientific("Pressure,L2-nodal", true);
    convergence_table.set_tex_caption("cells", "\\# cells");
    convergence_table.set_tex_caption("dofs", "\\# dofs");
    convergence_table.set_tex_caption("Velocity,L2", "$ \\|\\u - \\u_h\\|_{L^2} $");
    convergence_table.set_tex_caption("Velocity,Hdiv", "$ \\|\\nabla\\cdot(\\u - \\u_h)\\|_{L^2} $");
    convergence_table.set_tex_caption("Pressure,L2", "$ \\|p - p_h\\|_{L^2} $");
    convergence_table.set_tex_caption("Pressure,L2-nodal", "$ \\|Qp - p_h\\|_{L^2} $");
    convergence_table.set_tex_format("cells", "r");
    convergence_table.set_tex_format("dofs", "r");

    convergence_table.evaluate_convergence_rates("Velocity,L2", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates("Velocity,Hdiv", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates("Pressure,L2", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates("Pressure,L2-nodal", ConvergenceTable::reduction_rate_log2);

    std::ofstream error_table_file("error" + std::to_string(dim) + "d.tex");

    if (cycle == refine-1)
      {
        convergence_table.write_text(std::cout);
        convergence_table.write_tex(error_table_file);
      }
  }



  // @sect3{Run function}

  // The driver method <code>run()</code>
  // takes care of mesh generation and arranging calls to member methods in
  // the right way. It also resets data structures and clear triangulation and
  // DOF handler as we run the method on a sequence of refinements in order
  // to record convergence rates.
  template <int dim>
  void MultipointMixedDarcyProblem<dim>::run(const unsigned int refine)
  {
    Assert(refine > 0, ExcMessage("Must at least have 1 refinement cycle!"));

    dof_handler.clear();
    triangulation.clear();
    convergence_table.clear();

    for (unsigned int cycle=0; cycle<refine; ++cycle)
      {
        if (cycle == 0)
          {
            // We first generate the hyper cube and refine it twice
            // so that we could distort the grid slightly and
            // demonstrate the method's ability to work in such a
            // case.
            GridGenerator::hyper_cube (triangulation, 0, 1);
            triangulation.refine_global(2);
            GridTools::distort_random (0.3, triangulation, true);
          }
        else
          triangulation.refine_global(1);

        node_assembly();
        make_cell_centered_sp();
        pressure_assembly();
        solve_pressure ();
        velocity_recovery ();
        compute_errors (cycle);
        output_results (cycle, refine);
        reset_data_structures ();

        computing_timer.print_summary ();
        computing_timer.reset ();
      }
  }
}


// @sect3{The <code>main</code> function}

// In the main functione we pass the order of the Finite Element as an argument
// to the constructor of the Multipoint Flux Mixed Darcy problem, and the number
// of refinement cycles as an argument for the run method.
int main ()
{
  try
    {
      using namespace dealii;
      using namespace MFMFE;

      MultithreadInfo::set_thread_limit();

      MultipointMixedDarcyProblem<2> mfmfe_problem(2);
      mfmfe_problem.run(6);
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
