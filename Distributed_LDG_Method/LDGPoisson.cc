// @sect3{LDGPoisson.cc}
// The code begins as per usual with a long list of the the included
// files from the deal.ii library.
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>


#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <fstream>
#include <iostream>

// Here's where the classes for the DG methods begin.  
// We can use either the Lagrange polynomials,
#include <deal.II/fe/fe_dgq.h> 
// or the Legendre polynomials
#include <deal.II/fe/fe_dgp.h>
// as basis functions.  I'll be using the Lagrange polynomials.
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>


// Now we have to load in the deal.II files that will allow us to use
// a distributed computing framework.
#include <deal.II/base/utilities.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>


// Additionally we load the files that will allow us to interact with
// the Trilinos library.
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_solver.h>


// The functions class contains all the defintions of the functions we 
// will use, i.e. the right hand side function, the boundary conditions 
// and the test functions.  
#include "Functions.cc"

using namespace dealii;


// Here is the main class for the Local Discontinuous Galerkin method 
// applied to Poisson's equation, we won't explain much of the 
// the class and method declarations, but dive deeper into describing the 
// functions when the are defined.  The only thing I will menion 
// about the class declaration is that  here is where I labeled 
// the different types of boundaries using enums.
template <int dim>
class LDGPoissonProblem
{
    
public:
    LDGPoissonProblem(const unsigned int degree,
                      const unsigned int n_refine);

    ~LDGPoissonProblem();

    void run();


private:
    void make_grid();

    void make_dofs();

    void assemble_system();

    void assemble_cell_terms(const FEValues<dim>    &cell_fe,
                             FullMatrix<double>     &cell_matrix,
                             Vector<double>         &cell_vector);

    void assemble_Neumann_boundary_terms(const FEFaceValues<dim>    &face_fe,
                                    FullMatrix<double>         &local_matrix,
                                    Vector<double>             &local_vector);

    void assemble_Dirichlet_boundary_terms(const FEFaceValues<dim>  &face_fe,
                                      FullMatrix<double>       &local_matrix,
                                      Vector<double>           &local_vector,
                                      const double             & h);

    void assemble_flux_terms(const FEFaceValuesBase<dim>  &fe_face_values,
                        const FEFaceValuesBase<dim>  &fe_neighbor_face_values,
                        FullMatrix<double>           &vi_ui_matrix,
                        FullMatrix<double>           &vi_ue_matrix,
                        FullMatrix<double>           &ve_ui_matrix,
                        FullMatrix<double>           &ve_ue_matrix,
                        const double                 & h);

    void distribute_local_flux_to_global(
      const FullMatrix<double> & vi_ui_matrix,
      const FullMatrix<double> & vi_ue_matrix,
      const FullMatrix<double> & ve_ui_matrix,
      const FullMatrix<double> & ve_ue_matrix,
      const std::vector<types::global_dof_index> & local_dof_indices,
      const std::vector<types::global_dof_index> & local_neighbor_dof_indices);

    void solve();

    void output_results() const;

    const unsigned int degree;
    const unsigned int n_refine;
    double penalty;
    double h_max;
    double h_min;

    enum
    {
        Dirichlet,
        Neumann
    };

    parallel::distributed::Triangulation<dim>       triangulation;
    FESystem<dim>                                   fe;
    DoFHandler<dim>                                 dof_handler;

    ConstraintMatrix                                constraints;

    SparsityPattern                                 sparsity_pattern;

    TrilinosWrappers::SparseMatrix                  system_matrix;
    TrilinosWrappers::MPI::Vector                   locally_relevant_solution;
    TrilinosWrappers::MPI::Vector                   system_rhs;

    ConditionalOStream                              pcout;
    TimerOutput                                     computing_timer;

    SolverControl                                   solver_control;
    TrilinosWrappers::SolverDirect                  solver;

    const RightHandSide<dim>              rhs_function;
    const DirichletBoundaryValues<dim>    Dirichlet_bc_function;
    const TrueSolution<dim>               true_solution;
};


// @sect4{Class constructor and destructor}
// The constructor and destructor for this class is very much like the 
// like those for step-40.  The difference being that we'll be passing 
// in an integer, <code>degree</code>, which tells us the maxiumum order 
// of the polynomial to use as well as <code>n_refine</code> which is the 
// global number of times we refine our mesh.  The other main differences 
// are that we use a FESystem object for our choice of basis 
// functions. This is reminiscent of the mixed finite element method in
// step-20, however, in our case we use a FESystem
// of the form,
//
// <code> 
// fe( FESystem<dim>(FE_DGQ<dim>(degree), dim),        1,
//       FE_DGQ<dim>(degree),                          1) 
// </code>
// 
// which tells us that the basis functions contain discontinous polynomials
// of order <code>degree</code> in each of the <code>dim</code> dimensions
// for the vector field.  For the scalar unknown we
// use a discontinuous polynomial of the order <code>degree</code>. 
// The LDG method solves for both the primary variable as well as
// its gradient, just like the mixed finite element method.  However,
// unlike the mixed method, the LDG method ues discontinuous 
// polynomials to approximate both variables.
// The other difference bewteen our constructor and that of step-40 is that
// we all instantiate our linear solver in the constructor definition.
template <int dim>
LDGPoissonProblem<dim>::
LDGPoissonProblem(const unsigned int degree,
                  const unsigned int n_refine)
    :
    degree(degree),
    n_refine(n_refine),
    triangulation(MPI_COMM_WORLD,
                 typename Triangulation<dim>::MeshSmoothing
                 (Triangulation<dim>::smoothing_on_refinement |
                  Triangulation<dim>::smoothing_on_coarsening)),
    fe( FESystem<dim>(FE_DGQ<dim>(degree), dim),        1,
       FE_DGQ<dim>(degree),                             1),
    dof_handler(triangulation),
    pcout(std::cout,
         Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    computing_timer(MPI_COMM_WORLD,
                   pcout,
                   TimerOutput::summary,
                   TimerOutput::wall_times),
    solver_control(1),
    solver(solver_control),
    rhs_function(),
    Dirichlet_bc_function()
{
}

template <int dim>
LDGPoissonProblem<dim>::
~LDGPoissonProblem()
{
    dof_handler.clear();
}

// @sect4{Make_grid}
// This function shows how to make a grid using local
// refinement and also shows how to label the boundaries
// using the defined enum.
template <int dim>
void
LDGPoissonProblem<dim>::
make_grid()
{
  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(n_refine);

  unsigned int local_refine = 2;
  for(unsigned int i =0; i <local_refine; i++)
  {
        typename Triangulation<dim>::active_cell_iterator
        cell = triangulation.begin_active(),
        endc = triangulation.end();

        // We loop over all the cells in the mesh
        // and mark the  appropriate cells for refinement.  
        // In this example we only choose cells which are
        // near $x=0$ and $x=1$ in the
        // the domain.  This was just to show that
        // the LDG method is working with local
        // refinement and discussions on building
        // more realistic refinement stategies are
        // discussed elsewhere in the deal.ii
        // documentation.
        for(; cell != endc; cell++)
        {
            if((cell->center()[1]) > 0.9 )
            {
                if((cell->center()[0] > 0.9)  || (cell->center()[0] < 0.1))
                    cell->set_refine_flag();
            }
        }
        // Now that we have marked all the cells
        // that we want to refine locally we can go ahead and
        // refine them.
        triangulation.execute_coarsening_and_refinement();
  }

  // To label the boundary faces of the mesh with their
  // type, i.e. Dirichlet or Neumann,
  // we loop over all the cells in the mesh and then over
  // all the faces of each cell.  We then have to figure out
  // which faces are on the bounadry and set all faces 
  // on the boundary to have 
  // <code>boundary_id</code> to be <code>Dirichlet</code>.
  // We remark that one could easily set more complicated
  // conditions where there are both Dirichlet or
  // Neumann boundaries.
  typename Triangulation<dim>::cell_iterator
  cell = triangulation.begin(),
  endc = triangulation.end();
  for(; cell != endc; cell++)
  {
    for(unsigned int face_no=0;
        face_no < GeometryInfo<dim>::faces_per_cell;
        face_no++)
    {
      if(cell->face(face_no)->at_boundary() )
          cell->face(face_no)->set_boundary_id(Dirichlet);
    } 
  } 
}

// @sect3{make_dofs}
// This function is responsible for distributing the degrees of 
// freedom (dofs) to the processors and allocating memory for 
// the global system matrix, <code>system_matrix</code>,
// and global right hand side vector, <code>system_rhs</code> .  
// The dofs are the unknown coefficients for the polynomial 
// approximation of our solution to Poisson's equation in the scalar 
// variable and its gradient.
template <int dim>
void
LDGPoissonProblem<dim>::
make_dofs()
{
    TimerOutput::Scope t(computing_timer, "setup");

    // The first step to setting up our linear system is to
    // distribute the degrees of freedom (dofs) across the processors,
    // this is done with the <code>distribute_dofs()</code>
    // method of the DoFHandler.  We remark
    // the same exact function call that occurs when using deal.ii 
    // on a single machine, the DoFHandler automatically knows
    // that we are distributed setting because it was instantiated
    // with a distributed triangulation!
    dof_handler.distribute_dofs(fe);

    // We now renumber the dofs so that the vector of unkonwn dofs 
    // that we are solving for, <code>locally_relevant_solution</code>
    // corresponds to a vector of the form,
    //
    // $ \left[\begin{matrix} \textbf{Q} \\  \textbf{U} \end{matrix}\right] $
    DoFRenumbering::component_wise(dof_handler);

    // Now we get the locally owned dofs, that is the dofs that our local
    // to this processor. These dofs corresponding entries in the 
    // matrix and vectors that we will write to.
    IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();

    // In additon to the locally owned dofs, we also need the the locally 
    // relevant dofs.  These are the dofs that have read access to and we
    // need in order to do computations on our processor, but, that
    // we do not have the ability to write to.
    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler,
                                            locally_relevant_dofs);

    std::vector<types::global_dof_index> dofs_per_component(dim+1);
    DoFTools::count_dofs_per_component(dof_handler, dofs_per_component);
 
    // Discontinuous Galerkin methods are fantanistic methods in part because
    // many of the limitations of traditional finite element methods no longer
    // exist.  Specifically, the need to use constraint matrices 
    // in order handle hanging nodes is no longer necessary. However,
    // we will continue to use the constraint matrices inorder to efficiently
    // distribute local computations to the global system, i.e. to the
    // <code>system_matrix</code> and <code>system_rhs</code>.  Therefore, we  
    // just instantiate the constraints matrix object, clear and close it.
    constraints.clear();
    constraints.close();


    // Just like step-40 we create a dynamic sparsity pattern
    // and distribute it to the processors.  Notice how we do not have to
    // explictly mention that we are using a FESystem for system of 
    // variables instead of a FE_DGQ for a scalar variable
    // or that we are using a discributed DoFHandler.  All these specifics 
    // are taken care of under the hood by the deal.ii library.
    //  In order to build the sparsity 
    // pattern we use the DoFTools::make_flux_sparsity_pattern function 
    // since we using a DG method and need to take into account the DG
    // fluxes in the sparsity pattern.
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof_handler,
                                         dsp);

    SparsityTools::distribute_sparsity_pattern(dsp,
            dof_handler.n_locally_owned_dofs_per_processor(),
            MPI_COMM_WORLD,
            locally_relevant_dofs);

    // Here is one area that I had to learn the hard way. The local
    // discontinuous Galerkin method like the Mixed Method is written
    // in mixed form and will lead to a block-structured matrix.  
    // In step-20 we see that we that we initialize the
    // <code>system_martrix</code>
    // such that we explicitly declare it to be block-structured.  
    // It turns out there are reasons to do this when you are going to be 
    // using a Schur complement method to solve the system of equations. 
    // While the LDG method will lead to a block-structured matrix, 
    // we do not have to explicitly declare our matrix to be one.
    // I found that most of the distributed linear solvers did not 
    // accept block structured matrices and since I was using a 
    // distributed direct solver it was unnecessary to explicitly use a 
    // block structured matrix.
    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         MPI_COMM_WORLD);

    // The final note that I will make in that this subroutine is that 
    // we initialize this processors solution and the 
    // right hand side vector the exact same was as we did in step-40.
    // We should note that the <code>locally_relevant_solution</code> solution
    // vector includes dofs that are locally relevant to our computations
    // while the <code>system_rhs</code> right hand side vector will only 
    // include dofs that are locally owned by this processor.
    locally_relevant_solution.reinit(locally_relevant_dofs,
                                     MPI_COMM_WORLD);

    system_rhs.reinit(locally_owned_dofs,
                      locally_relevant_dofs,
                      MPI_COMM_WORLD,
                      true);

    const unsigned int n_vector_field = dim * dofs_per_component[0];
    const unsigned int n_potential = dofs_per_component[1];

    pcout << "Number of active cells : "
          << triangulation.n_global_active_cells()
          << std::endl
          << "Number of degrees of freedom: "
          << dof_handler.n_dofs()
          << " (" << n_vector_field << " + " << n_potential << ")"
          << std::endl;
}

//@sect4{assemble_system}
// This is the function that will assemble the global system matrix and 
// global right hand side vector for the LDG method. It starts out 
// like many of the deal.ii tutorial codes: declaring quadrature and 
// UpdateFlags objects, as well as vectors to that will hold the
// dof indices for the cells we are working on in the global system.
template <int dim>
void
LDGPoissonProblem<dim>::
assemble_system()
{
    TimerOutput::Scope t(computing_timer, "assembly");

    QGauss<dim>         quadrature_formula(fe.degree+1);
    QGauss<dim-1>       face_quadrature_formula(fe.degree+1);

    const UpdateFlags update_flags  = update_values
                                      | update_gradients
                                      | update_quadrature_points
                                      | update_JxW_values;

    const UpdateFlags face_update_flags =   update_values
                                            | update_normal_vectors
                                            | update_quadrature_points
                                            | update_JxW_values;
                                            
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<types::global_dof_index> 
                                local_neighbor_dof_indices(dofs_per_cell);                                      

    // We first remark that we have the FEValues objects for 
    // the values of our cell basis functions as was done in most
    // other examples.  Now because we 
    // are using discontinuous Galerkin methods we also introduce a 
    // FEFaceValues object, <code>fe_face_values</code>, 
    // for evaluating the basis functions
    // on one side of an element face as well as another FEFaceValues object, 
    // <code>fe_neighbor_face_values</code>, for evaluting the basis functions 
    // on the opposite side of the face, i.e. on the neighoring element's face. 
    // In addition, we also introduce a FESubfaceValues object,
    // <code>fe_subface_values</code>, that
    // will be used for dealing with faces that have multiple refinement 
    // levels, i.e. hanging nodes. When we have to evaulate the fluxes across
    // a face that multiple refinement levels, we need to evaluate the 
    // fluxes across all its childrens' faces; we'll explain this more when 
    // the time comes.
    FEValues<dim>           fe_values(fe, quadrature_formula, update_flags);

    FEFaceValues<dim>       fe_face_values(fe,face_quadrature_formula, 
                                          face_update_flags);

    FEFaceValues<dim>       fe_neighbor_face_values(fe, 
                                                face_quadrature_formula,
                                                face_update_flags);

    FESubfaceValues<dim>    fe_subface_values(fe, face_quadrature_formula,
                                              face_update_flags);

    // Here are the local (dense) matrix and right hand side vector for 
    // the solid integrals as well as the integrals on the boundaries in the
    // local discontinuous Galerkin method.  These terms will be built for 
    // each local element in the mesh and then distributed to the global
    // system matrix and right hand side vector.
    FullMatrix<double>      local_matrix(dofs_per_cell,dofs_per_cell);
    Vector<double>          local_vector(dofs_per_cell);

    // The next four matrices are used to incorporate the flux integrals across
    // interior faces of the mesh:
    FullMatrix<double>      vi_ui_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double>      vi_ue_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double>      ve_ui_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double>      ve_ue_matrix(dofs_per_cell, dofs_per_cell);
    // As explained in the section on the LDG method we take our test
    // function to be v and multiply it on the left side of our differential 
    // equation that is on u and peform integration by parts as explain in the 
    // introduction. Using this notation for test and solution function,
    // the matrices below will then stand for:
    //
    // <code>vi_ui</code> - Taking the value of the test function from 
    //         interior of this cell's face and the solution function 
    //         from the interior of this cell.
    //
    // <code>vi_ue</code> - Taking the value of the test function from 
    //         interior of this cell's face and the solution function 
    //         from the exterior of this cell.
    //
    // <code>ve_ui</code> - Taking the value of the test function from
    //         exterior of this cell's face and the solution function 
    //         from the interior of this cell.
    //
    // <code>ve_ue</code> - Taking the value of the test function from 
    //         exterior of this cell's face and the solution function
    //         from the exterior of this cell.

    // Now that we have gotten preliminary orders out of the way, 
    // we loop over all the cells
    // and assemble the local system matrix and local right hand side vector 
    // using the DoFHandler::active_cell_iterator,
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

    for(; cell!=endc; cell++)
    {
        // Now, since we are working in a distributed setting, 
        // we can only work on cells and write to dofs in the
        //  <code>system_matrix</code>
        // and <code>rhs_vector</code>
        // that corresponds to cells that are locally owned
        // by this processor.  We note that while we can only write to locally
        // owned dofs, we will still use information from cells that are 
        // locally relevant.  This is very much the same as in step-40. 
        if(cell->is_locally_owned())
        {
            // We now assemble the local contributions to the system matrix 
            // that includes the solid integrals in the LDG method as well as 
            // the right hand side vector. This involves resetting the local
            // matrix and vector to contain all zeros, reinitializing the
            // FEValues object for this cell and then building the 
            // <code>local_matrix</code> and <code>local_rhs</code> vector.
            local_matrix = 0;
            local_vector = 0;

            fe_values.reinit(cell);
            assemble_cell_terms(fe_values,
                                local_matrix,
                                local_vector);

            // We remark that we need to get the local indices for the dofs to
            // to this cell before we begin to compute the contributions 
            // from the numerical fluxes, i.e. the boundary conditions and
            // interior fluxes.
            cell->get_dof_indices(local_dof_indices);

            // Now is where were start to loop over all the faces of the cell
            // and construct the local contribtuions from the numerical fluxes.
            // The numerical fluxes will be due to 3 contributions: the
            // interior faces, the faces on the Neumann boundary and the faces
            // on the Dirichlet boundary.  We instantate a 
            // <code>face_iterator</code> to loop
            // over all the faces of this cell and first see if the face is on 
            // the boundary. Notice how we do not reinitiaize the
            //  <code>fe_face_values</code>
            // object for the face until we know that we are actually on face 
            // that lies on the boundary of the domain. The reason for doing this 
            // is for computational efficiency; reinitializing the FEFaceValues 
            // for each face is expensive and we do not want to do it unless  we 
            // are actually going use it to do computations.  After this, we test
            // if the face 
            // is on the a Dirichlet or a Neumann segment of the boundary and 
            // call the appropriate subroutine to assemble the contributions for
            // that boundary.  Note that this assembles the flux contribution
            // in the <code>local_matrix</code> as well as the boundary 
            // condition that ends up
            // in the <code>local_vector</code>.
            for(unsigned int face_no=0;
                    face_no< GeometryInfo<dim>::faces_per_cell;
                    face_no++)
            {
                typename DoFHandler<dim>::face_iterator  face = 
                                                      cell->face(face_no);

                if(face->at_boundary() )
                {
                    fe_face_values.reinit(cell, face_no);

                    if(face->boundary_id() == Dirichlet)
                    {
                        // Notice here that in order to assemble the
                        // flux due to the penalty term for the the
                        // Dirichlet boundary condition we need the
                        // local cell diameter size and we can get
                        // that value for this specific cell with
                        // the following,
                        double h = cell->diameter();
                        assemble_Dirichlet_boundary_terms(fe_face_values,
                                                          local_matrix,
                                                          local_vector,
                                                          h);
                    }
                    else if(face->boundary_id() == Neumann)
                    {
                        assemble_Neumann_boundary_terms(fe_face_values,
                                                        local_matrix,
                                                        local_vector);
                    }
                    else
                        Assert(false, ExcNotImplemented() );
                }
                else
                {
                    // At this point we know that the face we are on is an 
                    // interior face. We can begin to assemble the interior 
                    // flux matrices, but first we want to make sure that the
                    // neighbor cell to this face is a valid cell.  Once we know
                    // that the neighbor is a valid cell then we also want to get
                    // the meighbor cell that shares this cell's face.
                    // 
                    Assert(cell->neighbor(face_no).state() == 
                                                          IteratorState::valid,
                                                          ExcInternalError());

                    typename DoFHandler<dim>::cell_iterator neighbor =
                        cell->neighbor(face_no);

                    // Now that we have the two cells whose face we want to 
                    // compute the numerical flux across, we need to know
                    // if the face has been refined, i.e. if it has children
                    // faces. This occurs when one of the cells has a
                    // different level of refinement than 
                    // the other cell.  If this is the case, then this face 
                    // has a different level of refinement than the other faces
                    // of the cell, i.e. on this face there is a hanging node. 
                    // Hanging nodes are not a problem in DG methods, the only 
                    // time we have to watch out for them is at this step
                    // and as you will see the changes we have to our make
                    // are minor. 
                    if(face->has_children())
                    {
                        // We now need to find the face of our neighbor cell 
                        // such that neighbor(neigh_face_no) = cell(face_no).
                        const unsigned int neighbor_face_no =
                            cell->neighbor_of_neighbor(face_no);

                        // Once we do this we then have to loop over all the 
                        // subfaces (children faces) of our cell's face and 
                        // compute the interior fluxes across the children faces
                        // and the neighbor's face.
                        for(unsigned int subface_no=0;
                                subface_no < face->number_of_children();
                                ++subface_no)
                        {
                            // We then get the neighbor cell's subface that 
                            // matches our cell face's subface and the
                            // specific subface number. We assert that the parent
                            // face cannot be more than one Level of
                            // refinement above the child's face.  This is
                            // because the deal.ii library does not allow 
                            // neighboring cells to have refinement levels
                            // that are more than one level in difference.
                            typename DoFHandler<dim>::cell_iterator neighbor_child =
                                     cell->neighbor_child_on_subface(face_no, 
                                                                     subface_no);

                            Assert(!neighbor_child->has_children(), 
                                    ExcInternalError());

                            // Now that we are ready to build the local flux
                            // matrices for
                            // this face we reset them e zero and                           
                            // reinitialize this <code>fe_values</code> 
                            // to this cell's subface and
                            // <code>neighbor_child</code>'s 
                            // FEFaceValues and the FESubfaceValues 
                            // objects on the appropriate faces.
                            vi_ui_matrix = 0;
                            vi_ue_matrix = 0;
                            ve_ui_matrix = 0;
                            ve_ue_matrix = 0;

                            fe_subface_values.reinit(cell, face_no, subface_no);
                            fe_neighbor_face_values.reinit(neighbor_child, 
                                                           neighbor_face_no);

                            // In addition, we get the minimum of diameters of 
                            // the two cells to include in the penalty term
                            double h = std::min(cell->diameter(), 
                                                neighbor_child->diameter());

                            // We now finally assemble the interior fluxes for 
                            // the case of a face which has been refined using 
                            // exactly the same subroutine as we do when both 
                            // cells have the same refinement level.
                            assemble_flux_terms(fe_subface_values,
                                                fe_neighbor_face_values,
                                                vi_ui_matrix,
                                                vi_ue_matrix,
                                                ve_ui_matrix,
                                                ve_ue_matrix,
                                                h);

                            // Now all that is left to be done before distribuing
                            // the local flux matrices to the global system 
                            // is get the neighbor child faces dof indices.  
                            neighbor_child->get_dof_indices(local_neighbor_dof_indices);

                            // Once we have this cells dof indices and the 
                            // neighboring cell's dof indices we can use the 
                            // ConstraintMatrix to distribute the local flux 
                            // matrices to the global system matrix.
                            // This is done through the class function
                            // <code>distribute_local_flux_to_global()</code>.
                            distribute_local_flux_to_global(
                                                      vi_ui_matrix,
                                                      vi_ue_matrix,
                                                      ve_ui_matrix,
                                                      ve_ue_matrix,
                                                      local_dof_indices,
                                                      local_neighbor_dof_indices);
                        } 
                    } 
                    else
                    {
                        // At this point we know that this cell and the neighbor
                        // of this cell are on the same refinement level and
                        // the work to assemble the interior flux matrices 
                        // is very much the same as before. Infact it is
                        // much simpler since we do not have to loop through the
                        // subfaces.  However, we do have to check that we do
                        // not compute the same contribution twice. Since we are
                        // looping over all the faces of all the cells in the mesh,
                        // we pass over each face twice.  If we do not take this 
                        // into consideration when assembling the interior flux 
                        // matrices we might compute the local interior flux matrix
                        // twice. To avoid doing this we only compute the interior 
                        // fluxes once for each face by restricting that the 
                        // following computation only occur on the on 
                        // the cell face with the lower index number.
                        if(neighbor->level() == cell->level() &&
                            neighbor->index() > cell->index() )
                        {
                            // Here we find the neighbor face such that 
                            // neighbor(neigh_face_no) = cell(face_no). 
                            // In addition we, reinitialize the FEFaceValues
                            // and neighbor cell's FEFaceValues on their
                            // respective cells' faces, as well as get the
                            // minimum diameter of this cell
                            // and the neighbor cell and assign 
                            // it to <code>h</code>.
                            const unsigned int neighbor_face_no =
                                cell->neighbor_of_neighbor(face_no);

                            vi_ui_matrix = 0;
                            vi_ue_matrix = 0;
                            ve_ui_matrix = 0;
                            ve_ue_matrix = 0;

                            fe_face_values.reinit(cell, face_no);
                            fe_neighbor_face_values.reinit(neighbor, 
                                                          neighbor_face_no);

                            double h = std::min(cell->diameter(), 
                                                neighbor->diameter());

                            // Just as before we assemble the interior fluxes
                            //  using the
                            // <code>assemble_flux_terms</code> subroutine, 
                            // get the neighbor cell's
                            // face dof indices and use the constraint matrix to
                            // distribute the local flux matrices to the global 
                            // <code>system_matrix</code> using the class 
                            // function 
                            // <code>distribute_local_flux_to_global()</code>
                            assemble_flux_terms(fe_face_values,
                                                fe_neighbor_face_values,
                                                vi_ui_matrix,
                                                vi_ue_matrix,
                                                ve_ui_matrix,
                                                ve_ue_matrix,
                                                h);

                            neighbor->get_dof_indices(local_neighbor_dof_indices);
                            
                            distribute_local_flux_to_global(
                                                    vi_ui_matrix,
                                                    vi_ue_matrix,
                                                    ve_ui_matrix,
                                                    ve_ue_matrix,
                                                    local_dof_indices,
                                                    local_neighbor_dof_indices);

                          
                        } 
                    } 
                } 
            } 


            // Now that have looped over all the faces for this 
            // cell and computed as well as disributed the local
            // flux matrices to the <code>system_matrix</code>, we 
            // can finally distribute the cell's <code>local_matrix</code>
            // and <code>local_vector</code> contribution to the 
            // global system matrix and global right hand side vector.
            // We remark that we have to wait until this point
            // to distribute the <code>local_matrix</code>
            // and <code>system_rhs</code> to the global system. 
            // The reason being that in looping over the faces
            // the faces on the boundary of the domain contribute
            // to the <code>local_matrix</code>
            // and <code>system_rhs</code>.  We could distribute
            // the local contributions for each component seperately,
            // but writing to the distributed sparse matrix and vector
            // is expensive and want to to minimize the number of times
            // we do so.
            constraints.distribute_local_to_global(local_matrix,
                                                   local_dof_indices,
                                                   system_matrix);

            constraints.distribute_local_to_global(local_vector,
                                                   local_dof_indices,
                                                   system_rhs);

        } 
    }

    // We need to synchronize assembly of our global system
    // matrix and global right hand side vector with all the other 
    // processors and  
    // use the compress() function to do this. 
    // This was discussed in detail in step-40.
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
}


// @sect4{assemble_cell_terms}
// This function deals with constructing the local matrix due to
// the solid integrals over each element and is very similar to the 
// the other examples in the deal.ii tutorials.
template<int dim>
void
LDGPoissonProblem<dim>::
assemble_cell_terms(
    const FEValues<dim>     &cell_fe,
    FullMatrix<double>      &cell_matrix,
    Vector<double>          &cell_vector)
{
    const unsigned int dofs_per_cell = cell_fe.dofs_per_cell;
    const unsigned int n_q_points    = cell_fe.n_quadrature_points;

    const FEValuesExtractors::Vector VectorField(0);
    const FEValuesExtractors::Scalar Potential(dim);

    std::vector<double>              rhs_values(n_q_points);

    // We first get the value of the right hand side function 
    // evaluated at the quadrature points in the cell.
    rhs_function.value_list(cell_fe.get_quadrature_points(),
                            rhs_values);

    // Now, we loop over the quadrature points in the
    // cell and then loop over the degrees of freedom and perform
    // quadrature to approximate the integrals.
    for(unsigned int q=0; q<n_q_points; q++)
    {
        for(unsigned int i=0; i<dofs_per_cell; i++)
        {
            const Tensor<1, dim> psi_i_field          = cell_fe[VectorField].value(i,q);
            const double         div_psi_i_field      = cell_fe[VectorField].divergence(i,q);
            const double         psi_i_potential      = cell_fe[Potential].value(i,q);
            const Tensor<1, dim> grad_psi_i_potential = cell_fe[Potential].gradient(i,q);

            for(unsigned int j=0; j<dofs_per_cell; j++)
            {
                const Tensor<1, dim> psi_j_field        = cell_fe[VectorField].value(j,q);
                const double         psi_j_potential    = cell_fe[Potential].value(j,q);

                // This computation corresponds to assembling the local system
                // matrix for the integral over an element,
                //
                // $\int_{\Omega_{e}} \left(\textbf{w} \cdot \textbf{q} 
                //              - \nabla \cdot \textbf{w} u
                //              - \nabla w \cdot \textbf{q}
                //              \right) dx $
                cell_matrix(i,j)  += ( (psi_i_field * psi_j_field)
                                       -
                                       (div_psi_i_field * psi_j_potential)
                                       -
                                       (grad_psi_i_potential * psi_j_field)
                                     ) * cell_fe.JxW(q);
            } 

            // And this local right hand vector corresponds to the integral 
            // over the element cell,
            //
            // $ \int_{\Omega_{e}} w \,  f(\textbf{x}) \, dx $
            cell_vector(i) += psi_i_potential *
                              rhs_values[q] *
                              cell_fe.JxW(q);
        } 
    } 
} 

// @sect4{assemble_Dirichlet_boundary_terms}
// Here we have the function that builds the <code>local_matrix</code>
// contribution
// and local right hand side vector, <code>local_vector</code>
// for the Dirichlet boundary condtions.
template<int dim>
void
LDGPoissonProblem<dim>::
assemble_Dirichlet_boundary_terms(
    const FEFaceValues<dim>     &face_fe,
    FullMatrix<double>          &local_matrix,
    Vector<double>              &local_vector,
    const double                & h)
{
    const unsigned int dofs_per_cell     = face_fe.dofs_per_cell;
    const unsigned int n_q_points        = face_fe.n_quadrature_points;

    const FEValuesExtractors::Vector VectorField(0);
    const FEValuesExtractors::Scalar Potential(dim);

    std::vector<double>     Dirichlet_bc_values(n_q_points);

    // In order to evaluate the flux on the Dirichlet boundary face we 
    // first get the value of the Dirichlet boundary function on the quadrature
    // points of the face.  Then we loop over all the quadrature points and 
    // degrees of freedom and approximate the integrals on the Dirichlet boundary
    // element faces.
    Dirichlet_bc_function.value_list(face_fe.get_quadrature_points(),
                                     Dirichlet_bc_values);

    for(unsigned int q=0; q<n_q_points; q++)
    {
        for(unsigned int i=0; i<dofs_per_cell; i++)
        {
            const Tensor<1, dim> psi_i_field     = face_fe[VectorField].value(i,q);
            const double         psi_i_potential = face_fe[Potential].value(i,q);

            for(unsigned int j=0; j<dofs_per_cell; j++)
            {
                const Tensor<1, dim> psi_j_field    = face_fe[VectorField].value(j,q);
                const double         psi_j_potential = face_fe[Potential].value(j,q);
        
                // We compute contribution for the flux $\widehat{q}$ on 
                // the Dirichlet boundary which enters our system matrix as,
                //
                // $ \int_{\text{face}} w \, ( \textbf{n} \cdot \textbf{q}
                //                        + \sigma u)  ds $
                local_matrix(i,j) += psi_i_potential * (
                                         face_fe.normal_vector(q) *
                                         psi_j_field
                                         +
                                         (penalty/h) *
                                         psi_j_potential) *
                                     	   face_fe.JxW(q);

            } 

            // We also compute the contribution for the flux for $\widehat{u}$
            // on the Dirichlet boundary which is the Dirichlet boundary 
            // condition function and enters the right hand side vector as
            //  
            // $\int_{\text{face}} (-\textbf{w} \cdot \textbf{n}
            //                      + \sigma w) \, u_{D} ds $
            local_vector(i) += (-1.0 * psi_i_field *
                                face_fe.normal_vector(q)
                                +
                                (penalty/h) *
                                psi_i_potential) *
                                Dirichlet_bc_values[q] *
                                face_fe.JxW(q);
        } 
    }  
} 

// @sect4{assemble_Neumann_boundary_terms}
// Here we have the function that builds the <code>local_matrix</code>
// and <code>local_vector</code> for the Neumann boundary condtions.
template<int dim>
void
LDGPoissonProblem<dim>::
assemble_Neumann_boundary_terms(
    const FEFaceValues<dim>     &face_fe,
    FullMatrix<double>          &local_matrix,
    Vector<double>              &local_vector)
{
    const unsigned int dofs_per_cell = face_fe.dofs_per_cell;
    const unsigned int n_q_points    = face_fe.n_quadrature_points;

    const FEValuesExtractors::Vector VectorField(0);
    const FEValuesExtractors::Scalar Potential(dim);

    // In order to get evaluate the flux on the Neumann boundary face we 
    // first get the value of the Neumann boundary function on the quadrature
    // points of the face.  Then we loop over all the quadrature points and 
    // degrees of freedom and approximate the integrals on the Neumann boundary
    // element faces.
    std::vector<double >    Neumann_bc_values(n_q_points);

    for(unsigned int q=0; q<n_q_points; q++)
    {
        for(unsigned int i=0; i<dofs_per_cell; i++)
        {
            const Tensor<1, dim> psi_i_field     = face_fe[VectorField].value(i,q);
            const double         psi_i_potential = face_fe[Potential].value(i,q);

            for(unsigned int j=0; j<dofs_per_cell; j++)
            {

                const double    psi_j_potential = face_fe[Potential].value(j,q);

                // We compute contribution for the flux $\widehat{u}$ on the
                // Neumann boundary which enters our system matrix as,
                //
                // $\int_{\text{face}} \textbf{w}  \cdot \textbf{n} \, u \, ds $
                local_matrix(i,j) += psi_i_field *
                                     face_fe.normal_vector(q) *
                                     psi_j_potential *
                                     face_fe.JxW(q);

            } 

            // We also compute the contribution for the flux for
            // $\widehat{q}$ on the Neumann bounary which is the
            // Neumann boundary condition and enters the right
            // hand side vector as
            //
            // $\int_{\text{face}} -w \, g_{N} \, ds$
            local_vector(i) +=  -psi_i_potential *
                                Neumann_bc_values[q] *
                                face_fe.JxW(q);
        } 
    }  
}

// @sect4{assemble_flux_terms}
// Now we finally get to the function which builds the interior fluxes.  
// This is a rather long function
// and we will describe what is going on in detail. 
template<int dim>
void
LDGPoissonProblem<dim>::
assemble_flux_terms(
    const FEFaceValuesBase<dim>     &fe_face_values,
    const FEFaceValuesBase<dim>     &fe_neighbor_face_values,
    FullMatrix<double>              &vi_ui_matrix,
    FullMatrix<double>              &vi_ue_matrix,
    FullMatrix<double>              &ve_ui_matrix,
    FullMatrix<double>              &ve_ue_matrix,
    const double                    & h)
{
    const unsigned int n_face_points      = fe_face_values.n_quadrature_points;
    const unsigned int dofs_this_cell     = fe_face_values.dofs_per_cell;
    const unsigned int dofs_neighbor_cell = fe_neighbor_face_values.dofs_per_cell;

    const FEValuesExtractors::Vector VectorField(0);
    const FEValuesExtractors::Scalar Potential(dim);

    // The first thing we do is after the boilerplate is define 
    // the unit vector $\boldsymbol \beta$ that is used in defining
    // the LDG/ALternating fluxes.
    Point<dim> beta;
    for(int i=0; i<dim; i++)
        beta(i) = 1.0;
    beta /= sqrt(beta.square() );

    // Now we loop over all the quadrature points on the element face
    // and loop over all the degrees of freedom and approximate
    // the following flux integrals.
    for(unsigned int q=0; q<n_face_points; q++)
    {
        for(unsigned int i=0; i<dofs_this_cell; i++)
        {
            const Tensor<1,dim>  psi_i_field_minus  =
                fe_face_values[VectorField].value(i,q);
            const double psi_i_potential_minus  =
                fe_face_values[Potential].value(i,q);

            for(unsigned int j=0; j<dofs_this_cell; j++)
            {
                const Tensor<1,dim> psi_j_field_minus   =
                    fe_face_values[VectorField].value(j,q);
                const double psi_j_potential_minus  =
                    fe_face_values[Potential].value(j,q);

                // We compute the flux matrix where the test function's
                // as well as the solution function's values are taken from 
                // the interior as,
                //
                // $\int_{\text{face}}
                //            \left( \frac{1}{2} \,  n^{-} 
                //            \cdot ( \textbf{w}^{-} u^{-} 
                //            + w^{-} \textbf{q}^{-}) 
                //            + \boldsymbol \beta \cdot \textbf{w}^{-} u^{-} 
                //            - w^{-} \boldsymbol \beta \cdot \textbf{q}^{-} 
                //            + \sigma w^{-} \, u^{-} \right) ds$
                vi_ui_matrix(i,j)   += (0.5 * (
                                        psi_i_field_minus *
                                        fe_face_values.normal_vector(q) *
                                        psi_j_potential_minus
                                        +
                                        psi_i_potential_minus *
                                        fe_face_values.normal_vector(q) *
                                        psi_j_field_minus )
                                        +
                                        beta *
                                        psi_i_field_minus *
                                        psi_j_potential_minus
                                        -
                                        beta *
                                        psi_i_potential_minus *
                                        psi_j_field_minus
                                        +
                                        (penalty/h) *
                                        psi_i_potential_minus *
                                        psi_j_potential_minus
                                       ) *
                                       fe_face_values.JxW(q);
            } 

            for(unsigned int j=0; j<dofs_neighbor_cell; j++)
            {
                const Tensor<1,dim> psi_j_field_plus    =
                    fe_neighbor_face_values[VectorField].value(j,q);
                const double            psi_j_potential_plus        =
                    fe_neighbor_face_values[Potential].value(j,q);

                // We compute the flux matrix where the test function is
                // from the interior of this elements face and solution 
                // function is taken from the exterior.  This corresponds
                // to the computation,
                //
                // $\int_{\text{face}}
                //              \left( \frac{1}{2} \, n^{-} \cdot 
                //              ( \textbf{w}^{-} u^{+} 
                //             + w^{-} \textbf{q}^{+}) 
                //             - \boldsymbol \beta \cdot \textbf{w}^{-} u^{+}
                //             + w^{-} \boldsymbol \beta \cdot  \textbf{q}^{+} 
                //             - \sigma w^{-} \, u^{+} \right) ds $
                vi_ue_matrix(i,j) += ( 0.5 * (
                                        psi_i_field_minus *
                                        fe_face_values.normal_vector(q) *
                                        psi_j_potential_plus
                                        +
                                        psi_i_potential_minus *
                                        fe_face_values.normal_vector(q) *
                                        psi_j_field_plus )
                                        -
                                        beta *
                                        psi_i_field_minus *
                                        psi_j_potential_plus
                                        +
                                        beta *
                                        psi_i_potential_minus *
                                        psi_j_field_plus
                                        -
                                        (penalty/h) *
                                        psi_i_potential_minus *
                                        psi_j_potential_plus
                                    ) *
                                     fe_face_values.JxW(q);
            } 
        } 

        for(unsigned int i=0; i<dofs_neighbor_cell; i++)
        {
            const Tensor<1,dim>  psi_i_field_plus =
                fe_neighbor_face_values[VectorField].value(i,q);
            const double         psi_i_potential_plus =
                fe_neighbor_face_values[Potential].value(i,q);

            for(unsigned int j=0; j<dofs_this_cell; j++)
            {
                const Tensor<1,dim> psi_j_field_minus               =
                    fe_face_values[VectorField].value(j,q);
                const double        psi_j_potential_minus       =
                    fe_face_values[Potential].value(j,q);


                // We compute the flux matrix where the test function is
                // from the exterior of this elements face and solution 
                // function is taken from the interior.  This corresponds
                // to the computation,
                //
                // $ \int_{\text{face}}
                //           \left( -\frac{1}{2}\, n^{-} \cdot 
                //              (\textbf{w}^{+} u^{-} 
                //              +  w^{+} \textbf{q}^{-} )
                //              - \boldsymbol \beta \cdot \textbf{w}^{+} u^{-}
                //              +  w^{+} \boldsymbol \beta \cdot \textbf{q}^{-}
                //              - \sigma w^{+} u^{-} \right) ds $
                ve_ui_matrix(i,j) +=  (-0.5 * (
                                        psi_i_field_plus *
                                        fe_face_values.normal_vector(q) *
                                        psi_j_potential_minus
                                        +
                                        psi_i_potential_plus *
                                        fe_face_values.normal_vector(q) *
                                        psi_j_field_minus)
                                        -
                                        beta *
                                        psi_i_field_plus *
                                        psi_j_potential_minus
                                        +
                                        beta *
                                        psi_i_potential_plus *
                                        psi_j_field_minus
                                        -
                                        (penalty/h) *
                                        psi_i_potential_plus *
                                        psi_j_potential_minus
                                        ) *
                                        fe_face_values.JxW(q);
            } 

            for(unsigned int j=0; j<dofs_neighbor_cell; j++)
            {
                const Tensor<1,dim> psi_j_field_plus =
                    fe_neighbor_face_values[VectorField].value(j,q);
                const double        psi_j_potential_plus =
                    fe_neighbor_face_values[Potential].value(j,q);

                // And lastly we compute the flux matrix where the test 
                // function and solution function are taken from the exterior
                // cell to this face.  This corresponds to the computation,
                //
                // $\int_{\text{face}}
                //             \left( -\frac{1}{2}\, n^{-} \cdot 
                //              ( \textbf{w}^{+} u^{+}
                //             + w^{+} \textbf{q}^{+} )
                //             + \boldsymbol \beta \cdot \textbf{w}^{+} u^{+} 
                //             -  w^{+} \boldsymbol \beta \cdot \textbf{q}^{+}
                //             + \sigma w^{+} u^{+} \right) ds $
                ve_ue_matrix(i,j) +=    (-0.5 * (
                                        psi_i_field_plus *
                                        fe_face_values.normal_vector(q) *
                                        psi_j_potential_plus
                                        +
                                        psi_i_potential_plus *
                                        fe_face_values.normal_vector(q) *
                                        psi_j_field_plus )
                                        +
                                        beta *
                                        psi_i_field_plus *
                                        psi_j_potential_plus
                                        -
                                        beta *
                                        psi_i_potential_plus *
                                        psi_j_field_plus
                                        +
                                        (penalty/h) *
                                        psi_i_potential_plus *
                                        psi_j_potential_plus
                                        ) *
                                        fe_face_values.JxW(q);
            } 

        } 
    } 
} 

// @sect4{distribute_local_flux_to_global}
// In this function we use the ConstraintMatrix to distribute
// the local flux matrices to the global system matrix.  
// Since I have to do this twice in assembling the 
// system matrix, I made function to do it rather than have
// repeated code.
// We remark that the reader take special note of
// the which matrices we are distributing and the order 
// in which we pass the dof indices vectors. In distributing 
// the first matrix, i.e. <code>vi_ui_matrix</code>, we are 
// taking the test function and solution function values from 
// the interior of this cell and therefore only need the 
// <code>local_dof_indices</code> since it contains the dof
// indices to this cell. When we distribute the second matrix,
// <code>vi_ue_matrix</code>, the test function is taken 
// form the inteior of
// this cell while the solution function is taken from the 
// exterior, i.e. the neighbor cell.  Notice that the order 
// degrees of freedom index vectors matrch this pattern: first
// the <code>local_dof_indices</code> which is local to 
// this cell and then
// the <code>local_neighbor_dof_indices</code> which is
// local to the neighbor's
// cell.  The order in which we pass the dof indices for the
// matrices is paramount to constructing our global system 
// matrix properly.  The ordering of the last two matrices 
// follow the same logic as the first two we discussed.
template<int dim>
void 
LDGPoissonProblem<dim>::
distribute_local_flux_to_global(
        const FullMatrix<double> & vi_ui_matrix,
        const FullMatrix<double> & vi_ue_matrix,
        const FullMatrix<double> & ve_ui_matrix,
        const FullMatrix<double> & ve_ue_matrix,
        const std::vector<types::global_dof_index> & local_dof_indices,
        const std::vector<types::global_dof_index> & local_neighbor_dof_indices)
{
  constraints.distribute_local_to_global(vi_ui_matrix,
                                         local_dof_indices,
                                         system_matrix);

  constraints.distribute_local_to_global(vi_ue_matrix,
                                         local_dof_indices,
                                         local_neighbor_dof_indices,
                                         system_matrix);

  constraints.distribute_local_to_global(ve_ui_matrix,
                                        local_neighbor_dof_indices,
                                        local_dof_indices,
                                        system_matrix);

  constraints.distribute_local_to_global(ve_ue_matrix,
                                         local_neighbor_dof_indices,
                                         system_matrix);
}



// @sect4{solve}
// As mentioned earlier I used a direct solver to solve
// the linear system of equations resulting from the LDG
// method applied to the Poisson equation. One could also
// use a iterative sovler, however, we then need to use 
// a preconditoner and that was something I did not wanted
// to get into. The uses of a direct sovler here is
// somewhat of a limitation.  The built-in distributed 
// direct solver in Trilinos reduces everything to one 
// processor, solves the system and then distributes 
// everything back out to the other processors.  However, 
// by linking to more advanced direct sovlers through 
// Trilinos one can accomplish fully distributed computations
// and not much about the following function calls will
// change. 
template<int dim>
void
LDGPoissonProblem<dim>::
solve()
{
    TimerOutput::Scope t(computing_timer, "solve");

    // As in step-40 in order to perform a linear solve
    // we need solution vector where there is no overlap across 
    // the processors and we create this by instantiating 
    // <code>completely_distributed_solution</code> solution
    // vector using
    // the copy constructor on the global system right hand 
    // side vector which itself is completely distributed vector.
    TrilinosWrappers::MPI::Vector
    completely_distributed_solution(system_rhs);

    // Now we can preform the solve on the completeley distributed 
    // right hand side vector, system matrix and the completely
    // distributed solution.
    solver.solve(system_matrix,
                 completely_distributed_solution,
                 system_rhs);

    // We now distribute the constraints of our system onto the 
    // completely solution vector, but in our case with the LDG
    // method there are none.
    constraints.distribute(completely_distributed_solution);

    // Lastly we copy the completely distributed solution vector,
    // <code>completely_distributed_solution</code>,
    // to solution vector which has some overlap between 
    // processors, <code>locally_relevant_solution</code>.
    // We need the overlapped portions of our solution
    // in order to be able to do computations using the solution
    // later in the code or in post processing.
    locally_relevant_solution = completely_distributed_solution;
}

// @sect4{output_results}
// This function deals with the writing of the reuslts in parallel
// to disk.  It is almost exactly the same as 
// in step-40 and we wont go into it.  It is noteworthy 
// that in step-40 the output is only the scalar solution,
// while in our situation, we are outputing both the scalar 
// solution as well as the vector field solution. The only
// difference between this function and the one in step-40 
// is in the <code>solution_names</code> vector where we have to add
// the gradient dimensions.  Everything else is taken care 
// of by the deal.ii library!
template<int dim>
void
LDGPoissonProblem<dim>::
output_results()    const
{
    std::vector<std::string> solution_names;
    switch(dim)
    {
    case 1:
        solution_names.push_back("u");
        solution_names.push_back("du/dx");
        break;

    case 2:
        solution_names.push_back("grad(u)_x");
        solution_names.push_back("grad(u)_y");
        solution_names.push_back("u");
        break;

    case 3:
        solution_names.push_back("grad(u)_x");
        solution_names.push_back("grad(u)_y");
        solution_names.push_back("grad(u)_z");
        solution_names.push_back("u");
        break;

    default:
        Assert(false, ExcNotImplemented() );
    }

    DataOut<dim>    data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution,
                             solution_names);

    Vector<float>   subdomain(triangulation.n_active_cells());

    for(unsigned int i=0; i<subdomain.size(); i++)
        subdomain(i) = triangulation.locally_owned_subdomain();

    data_out.add_data_vector(subdomain,"subdomain");

    data_out.build_patches();

    const std::string filename = ("solution."   +
                                  Utilities::int_to_string(
                                  triangulation.locally_owned_subdomain(),4));

    std::ofstream output((filename + ".vtu").c_str());
    data_out.write_vtu(output);

    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 )
    {
        std::vector<std::string>    filenames;
        for(unsigned int i=0;
                i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
                i++)
        {
            filenames.push_back("solution." +
                                Utilities::int_to_string(i,4) +
                                ".vtu");
        }
        std::ofstream master_output("solution.pvtu");
        data_out.write_pvtu_record(master_output, filenames);
    } 
}


// @sect4{run}
// The only public function of this class is pretty much exactly
// the same as all the other deal.ii examples except I setting
// the constant in the DG penalty ($\tilde{\sigma}}$) to be 1.
template<int dim>
void
LDGPoissonProblem<dim>::
run()
{
    penalty = 1.0;
    make_grid();
    make_dofs();
    assemble_system();
    solve();
    output_results();  
}


// @sect3{main}
// Here it the main class of our program, since it is nearly exactly
// the same as step-40 and many of the other examples I won't
// elaborate on it. 
int main(int argc, char *argv[])
{

    try {
        using namespace dealii;

        deallog.depth_console(0);

        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                            numbers::invalid_unsigned_int);

        unsigned int degree = 1;
        unsigned int n_refine = 6;
        LDGPoissonProblem<2>    Poisson(degree, n_refine);
        Poisson.run();
        
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
