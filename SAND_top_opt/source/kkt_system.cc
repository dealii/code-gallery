//
// Created by justin on 2/17/21.
//
#include "../include/kkt_system.h"
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>


#include <deal.II/lac/matrix_out.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nothing.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>


#include <deal.II/base/conditional_ostream.h>

#include "../include/input_information.h"
#include "../include/matrix_free_elasticity.h"
#include "../include/poly_pre.h"

#include <iostream>
#include <algorithm>

/// This problem initializes with a FESystem composed of 2×dim FE_Q(1) elements, and 8 FE_DGQ(0)  elements.
/// The  piecewise  constant  functions  are  for  density-related  variables,and displacement-related variables are assigned to the FE_Q(1) elements.
namespace SAND {

///Necessary functions for going between Trilinos vectors and multigrid-compatible distributed vectors.
namespace ChangeVectorTypes
{
template <typename number>
void copy_from_displacement_to_system_vector(LA::MPI::Vector                                           &out,
                                             const dealii::LinearAlgebra::distributed::Vector<number>  &in,
                                             std::map<types::global_dof_index,types::global_dof_index>  & displacement_to_system_dof_index_map)
{
//    dealii::LinearAlgebra::ReadWriteVector<double> rwv(
//                out.locally_owned_elements());
//    rwv.import(in, VectorOperation::insert);
    for (const auto &index_pair : displacement_to_system_dof_index_map)
    {
        out[index_pair.second] = in[index_pair.first];
    }
//    out.import(rwv, VectorOperation::insert);
}

template <typename number>
void copy_from_system_to_displacement_vector(dealii::LinearAlgebra::distributed::Vector<number>  &out,
                                             const LA::MPI::Vector                                           &in,
                                             std::map<types::global_dof_index,types::global_dof_index>  & displacement_to_system_dof_index_map)
{
//    dealii::LinearAlgebra::ReadWriteVector<double> rwv(
//                out.locally_owned_elements());
//    rwv.import(in, VectorOperation::insert);
    ConditionalOStream pcout (std::cout,(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 1));
    for (const auto &index_pair : displacement_to_system_dof_index_map)
    {
        out[index_pair.first] = in[index_pair.second];
    }
//    out.import(rwv, VectorOperation::insert);
}

template <typename number>
void copy(LA::MPI::Vector &                                         out,
          const dealii::LinearAlgebra::distributed::Vector<number> &in)
{
    dealii::LinearAlgebra::ReadWriteVector<double> rwv(
                out.locally_owned_elements());
    rwv.import(in, VectorOperation::insert);
    out.import(rwv, VectorOperation::insert);
}
template <typename number>
void copy(dealii::LinearAlgebra::distributed::Vector<number> &out,
          const LA::MPI::Vector &                             in)
{
    dealii::LinearAlgebra::ReadWriteVector<double> rwv;
    rwv.reinit(in);
    out.import(rwv, VectorOperation::insert);
}


} // namespace ChangeVectorTypes

///The KKTSystem class calculates the Hessian and Gradient of the Lagrangian of the system, and solves the resulting system to be used
/// as a step direction for the overarching solver.
template<int dim>
KktSystem<dim>::KktSystem()
    :
      mpi_communicator(MPI_COMM_WORLD),
      triangulation(mpi_communicator,
                    Triangulation<dim>::limit_level_difference_at_vertices,
                    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
      dof_handler(triangulation),
      dof_handler_displacement(triangulation),
      dof_handler_density(triangulation),
      /*fe should have 1 FE_DGQ<dim>(0) element for density, dim FE_Q finite elements for displacement,
                   * another dim FE_Q elements for the lagrange multiplier on the FE constraint, and 2 more FE_DGQ<dim>(0)
                   * elements for the upper and lower bound constraints */
      fe_nine(FE_DGQ<dim>(0) ^ 5,
              (FESystem<dim>(FE_Q<dim>(1) ^ dim)) ^ 2,
              FE_DGQ<dim>(0) ^ 2,
              FE_Nothing<dim>() ^ 1),
      fe_ten(FE_DGQ<dim>(0) ^ 5,
             (FESystem<dim>(FE_Q<dim>(1) ^ dim)) ^ 2,
             FE_DGQ<dim>(0) ^ 2,
             FE_DGQ<dim>(0) ^ 1),
      fe_displacement(FE_Q<dim>(1) ^ dim),
      fe_density(0),
      density_ratio(Input::volume_percentage),
      density_penalty_exponent(Input::density_penalty_exponent),
      density_filter(),
      pcout(std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
{
    fe_collection.push_back(fe_nine);
    fe_collection.push_back(fe_ten);

}


///A  function  used  once  at  the  beginning  of  the  program,  this  creates  a  matrix  H  so  that H* unfiltered density = filtered density

template<int dim>
void
KktSystem<dim>::setup_filter_matrix() {
    pcout << "IN KKT FILTER SETUP FUNCTION" << std::endl;
    density_filter.initialize(dof_handler);
}

///This triangulation matches the problem description

template<int dim>
void
KktSystem<dim>::create_triangulation() {

    ///Start by defining the sub-blocks of the DoFHandler

    std::vector<unsigned int> sub_blocks(2*dim+8, 0);

    sub_blocks[0]=0;
    sub_blocks[1]=1;
    sub_blocks[2]=2;
    sub_blocks[3]=3;
    sub_blocks[4]=4;
    for(int i=0; i<dim; i++)
    {
        sub_blocks[5+i]=5;
    }
    for(int i=0; i<dim; i++)
    {
        sub_blocks[5+dim+i]=6;
    }
    sub_blocks[5+2*dim]=7;
    sub_blocks[6+2*dim]=8;
    sub_blocks[7+2*dim]=9;

    ///MBB Beam defined here
    if (Input::geometry_base == GeometryOptions::mbb) {
        const double width = 6;
        const unsigned int width_refine = 6;
        const double height = 1;
        const unsigned int height_refine = 1;
        const double depth = 1;
        const unsigned int depth_refine = 1;
        const double downforce_y = 1;
        const double downforce_x = 3;
        const double downforce_size = .3;

        if (dim == 2)
        {
            GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                      {width_refine, height_refine},
                                                      Point<dim>(0, 0),
                                                      Point<dim>(width, height));

            triangulation.refine_global(Input::refinements);

            /*Set BCIDs   */
            for (const auto &cell: dof_handler.active_cell_iterators())
            {
                if(cell->is_locally_owned())
                {
                    cell->set_active_fe_index(0);
                    cell->set_material_id(MaterialIds::without_multiplier);
                    for (unsigned int face_number = 0;
                         face_number < GeometryInfo<dim>::faces_per_cell;
                         ++face_number)
                    {
                        if (cell->face(face_number)->at_boundary())
                        {
                            const auto center = cell->face(face_number)->center();

                            if (std::fabs(center(1) - downforce_y) < 1e-12)
                            {
                                if (std::fabs(center(0) - downforce_x) < downforce_size)
                                {
                                    cell->face(face_number)->set_boundary_id(BoundaryIds::down_force);
                                }
                                else
                                {
                                    cell->face(face_number)->set_boundary_id(BoundaryIds::no_force);
                                }
                            }
                        }
                    }
                    for (unsigned int vertex_number = 0;
                         vertex_number < GeometryInfo<dim>::vertices_per_cell;
                         ++vertex_number) {
                        if (std::abs(cell->vertex(vertex_number)(0)) + std::abs(cell->vertex(vertex_number)(1)) <
                                1e-10) {
                            cell->set_active_fe_index(1);
                            cell->set_material_id(MaterialIds::with_multiplier);
                        }
                    }
                }
            }

        }
        else if (dim == 3)
        {
            GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                      {width_refine, height_refine, depth_refine},
                                                      Point<dim>(0, 0, 0),
                                                      Point<dim>(width, height, depth));

            triangulation.refine_global(Input::refinements);

            for (const auto &cell: dof_handler.active_cell_iterators()) {
                if (cell->is_locally_owned())
                {
                    cell->set_active_fe_index(0);
                    cell->set_material_id(MaterialIds::without_multiplier);
                    for (unsigned int face_number = 0;
                         face_number < GeometryInfo<dim>::faces_per_cell;
                         ++face_number) {
                        if (cell->face(face_number)->at_boundary()) {
                            const auto center = cell->face(face_number)->center();

                            if (std::fabs(center(1) - downforce_y) < 1e-12) {
                                if (std::fabs(center(0) - downforce_x) < downforce_size) {
                                    cell->face(face_number)->set_boundary_id(BoundaryIds::down_force);
                                } else {
                                    cell->face(face_number)->set_boundary_id(BoundaryIds::no_force);
                                }
                            }
                        }
                    }
                    for (unsigned int vertex_number = 0;
                         vertex_number < GeometryInfo<dim>::vertices_per_cell;
                         ++vertex_number) {
                        if (std::abs(cell->vertex(vertex_number)(0)) + std::abs(cell->vertex(vertex_number)(1))
                                + std::abs(cell->vertex(vertex_number)(2)) < 1e-10) {
                            cell->set_active_fe_index(1);
                            cell->set_material_id(MaterialIds::with_multiplier);
                        }
                    }
                }
            }

        } else {
            throw;
        }
    ///L-shaped cantilever with re-entrant corner
    } else if (Input::geometry_base == GeometryOptions::l_shape) {
        const double width = 2;
        const unsigned int width_refine = 2;
        const double height = 2;
        const unsigned int height_refine = 2;
        const double depth = 1;
        const unsigned int depth_refine = 1;
        const double downforce_x = 2;
        const double downforce_y = 1;
        const double downforce_z = .5;
        const double downforce_size = .3;

        if (dim == 2) {
            GridGenerator::subdivided_hyper_L(triangulation,
                                              {width_refine, height_refine},
                                              Point<dim>(0, 0),
                                              Point<dim>(width, height),
                                              {-1, -1});

            triangulation.refine_global(Input::refinements);

            /*Set BCIDs   */
            for (const auto &cell: dof_handler.active_cell_iterators()) {
                if (cell->is_locally_owned())
                {
                    cell->set_active_fe_index(0);
                    cell->set_material_id(MaterialIds::without_multiplier);
                    for (unsigned int face_number = 0;
                         face_number < GeometryInfo<dim>::faces_per_cell;
                         ++face_number) {
                        if (cell->face(face_number)->at_boundary()) {
                            const auto center = cell->face(face_number)->center();

                            if (std::fabs(center(0) - downforce_x) < 1e-12) {
                                if (std::fabs(center(1) - downforce_y) < downforce_size) {
                                    cell->face(face_number)->set_boundary_id(BoundaryIds::down_force);
                                } else {
                                    cell->face(face_number)->set_boundary_id(BoundaryIds::no_force);
                                }
                            }
                        }
                    }
                    for (unsigned int vertex_number = 0;
                         vertex_number < GeometryInfo<dim>::vertices_per_cell;
                         ++vertex_number) {
                        if (std::abs(cell->vertex(vertex_number)(0)) + std::abs(cell->vertex(vertex_number)(1)) <
                                1e-10) {
                            cell->set_active_fe_index(1);
                            cell->set_material_id(MaterialIds::with_multiplier);
                        }
                    }
                }
            }

        } else if (dim == 3) {
            GridGenerator::subdivided_hyper_L(triangulation,
                                              {width_refine, height_refine, depth_refine},
                                              Point<dim>(0, 0, 0),
                                              Point<dim>(width, height, depth),
                                              {-1, -1, depth_refine});

            triangulation.refine_global(Input::refinements);

            /*Set BCIDs   */
            for (const auto &cell: dof_handler.active_cell_iterators()) {
                if(cell->is_locally_owned())
                {
                    cell->set_active_fe_index(0);
                    cell->set_material_id(MaterialIds::without_multiplier);
                    for (unsigned int face_number = 0;
                         face_number < GeometryInfo<dim>::faces_per_cell;
                         ++face_number) {
                        if (cell->face(face_number)->at_boundary()) {
                            const auto center = cell->face(face_number)->center();

                            if (std::fabs(center(0) - downforce_x) < 1e-12) {
                                if (std::fabs(center(1) - downforce_y) < downforce_size) {
                                    cell->face(face_number)->set_boundary_id(BoundaryIds::down_force);
                                    if (std::fabs(center(2) - downforce_z) < downforce_size) {
                                        cell->face(face_number)->set_boundary_id(BoundaryIds::down_force);
                                    } else {
                                        cell->face(face_number)->set_boundary_id(BoundaryIds::no_force);
                                    }
                                } else {
                                    cell->face(face_number)->set_boundary_id(BoundaryIds::no_force);
                                }
                            }
                        }
                    }
                    for (unsigned int vertex_number = 0;
                         vertex_number < GeometryInfo<dim>::vertices_per_cell;
                         ++vertex_number) {
                        if (std::abs(cell->vertex(vertex_number)(0)) + std::abs(cell->vertex(vertex_number)(1)) <
                                1e-10) {
                            cell->set_active_fe_index(1);
                            cell->set_material_id(MaterialIds::with_multiplier);
                        }
                    }
                }
            }
        } else {
            throw;
        }
    } else {
        throw;
    }

    dof_handler.distribute_dofs(fe_collection);
    DoFRenumbering::component_wise(dof_handler, sub_blocks);

    dof_handler_displacement.distribute_dofs(fe_displacement);
    dof_handler_displacement.distribute_mg_dofs();

    displacement_to_system_dof_index_map.clear();

}

/// Only individual points are given Dirichlet Boundary Conditions.
/// For example, in the MBB caes, The  bottom  corners  are  kept  in  place  in  the  y  direction
/// and the  bottom  left  also  in  the  x direction.
/// Because deal.ii is formulated to enforce boundary conditions along regions of the boundary,
/// we do this to ensure these BCs are only enforced at points.
template<int dim>
void
KktSystem<dim>::setup_boundary_values()
{
    if (Input::geometry_base == GeometryOptions::mbb)
    {
        if (dim == 2)
        {
            for (const auto &cell: dof_handler.active_cell_iterators())
            {
                if(cell->is_locally_owned())
                {
                    for (unsigned int face_number = 0;
                         face_number < GeometryInfo<dim>::faces_per_cell;
                         ++face_number)
                    {
                        if (cell->face(face_number)->at_boundary())
                        {
                            for (unsigned int vertex_number = 0;
                                 vertex_number < GeometryInfo<dim>::vertices_per_cell;
                                 ++vertex_number)
                            {
                                const auto vert = cell->vertex(vertex_number);
                                /*Find bottom left corner*/
                                if (std::fabs(vert(0) - 0) < 1e-12 && std::fabs(
                                            vert(1) - 0) < 1e-12)
                                {

                                    const unsigned int x_displacement =
                                            cell->vertex_dof_index(vertex_number, 0, cell->active_fe_index());
                                    const unsigned int y_displacement =
                                            cell->vertex_dof_index(vertex_number, 1, cell->active_fe_index());
                                    const unsigned int x_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 2, cell->active_fe_index());
                                    const unsigned int y_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 3, cell->active_fe_index());
                                    /*set bottom left BC*/
                                    boundary_values[x_displacement] = 0;
                                    boundary_values[y_displacement] = 0;
                                    boundary_values[x_displacement_multiplier] = 0;
                                    boundary_values[y_displacement_multiplier] = 0;
                                }
                                /*Find bottom right corner*/
                                if (std::fabs(vert(0) - 6) < 1e-12 && std::fabs(
                                            vert(1) - 0) < 1e-12)
                                {
//                                                                const unsigned int x_displacement =
//                                                                        cell->vertex_dof_index(vertex_number, 0, cell->active_fe_index());
                                    const unsigned int y_displacement =
                                            cell->vertex_dof_index(vertex_number, 1, cell->active_fe_index());
//                                                                const unsigned int x_displacement_multiplier =
//                                                                        cell->vertex_dof_index(vertex_number, 2, cell->active_fe_index());
                                    const unsigned int y_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 3, cell->active_fe_index());
//                                    boundary_values[x_displacement] = 0;
                                    boundary_values[y_displacement] = 0;
//                                    boundary_values[x_displacement_multiplier] = 0;
                                    boundary_values[y_displacement_multiplier] = 0;
                                }
                            }
                        }
                    }
                }

            }
            const unsigned int n_levels = triangulation.n_global_levels();
            level_dirichlet_boundary_dofs.resize(0,n_levels-1);
            level_boundary_values.resize(0,n_levels-1);
            mg_level_constraints.resize(0,n_levels-1);

            for(unsigned int level = 0; level < n_levels; ++level)
            {
                IndexSet relevant_dofs;
                DoFTools::extract_locally_relevant_level_dofs(dof_handler_displacement,
                                                      level,
                                                      relevant_dofs);
                mg_level_constraints[level].reinit(relevant_dofs);
            }


            for (auto cell=dof_handler_displacement.begin_active(n_levels-1); 
                 cell!=dof_handler_displacement.end_active(n_levels-1); 
                 ++cell)
            {
                if(cell->is_locally_owned())
                {
                    for (unsigned int face_number = 0;
                         face_number < GeometryInfo<dim>::faces_per_cell;
                         ++face_number)
                    {
                        if (cell->face(face_number)->at_boundary())
                        {
                            for (unsigned int vertex_number = 0;
                                 vertex_number < GeometryInfo<dim>::vertices_per_cell;
                                 ++vertex_number)
                            {
                                const auto vert = cell->vertex(vertex_number);
                                /*Find bottom left corner*/
                                if (std::fabs(vert(0) - 0) < 1e-12 && std::fabs(
                                        vert(1) - 0) < 1e-12)
                                {
                                     for (unsigned int level = 0; level < n_levels; ++level)
                                     {
                                         const unsigned int x_displacement =
                                                 cell->mg_vertex_dof_index(level, vertex_number, 0, cell->active_fe_index());
                                         const unsigned int y_displacement =
                                                 cell->mg_vertex_dof_index(level, vertex_number, 1, cell->active_fe_index());
                                                 
                                        /*set bottom left BC*/

                                         level_dirichlet_boundary_dofs[level].insert(x_displacement);
                                         level_dirichlet_boundary_dofs[level].insert(y_displacement);

                                         level_boundary_values[level][x_displacement] = 0;
                                         level_boundary_values[level][y_displacement] = 0;
                                     }

                                }
                                /*Find bottom right corner*/
                                if (std::fabs(vert(0) - 6) < 1e-12 && std::fabs(
                                        vert(1) - 0) < 1e-12)
                                {
                                    for (unsigned int level = 0; level < n_levels; ++level)
                                    {

//                                        const unsigned int x_displacement =
//                                                cell->mg_vertex_dof_index(level, vertex_number, 0,cell->active_fe_index());
                                        const unsigned int y_displacement =
                                                cell->mg_vertex_dof_index(level, vertex_number, 1, cell->active_fe_index());

//                                        level_dirichlet_boundary_dofs[level].insert(x_displacement);
                                        level_dirichlet_boundary_dofs[level].insert(y_displacement);
                                         
//                                        level_boundary_values[level][x_displacement] = 0;
                                        level_boundary_values[level][y_displacement] = 0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            for (unsigned int level = 0; level < n_levels; ++level)
            {
                IndexSet relevant_dofs;
                DoFTools::extract_locally_relevant_level_dofs(dof_handler_displacement,level,relevant_dofs);
                mg_level_constraints[level].add_lines(level_dirichlet_boundary_dofs[level]);
                mg_level_constraints[level].make_consistent_in_parallel(
                    dof_handler_displacement.locally_owned_mg_dofs(level),
                    relevant_dofs,
                    mpi_communicator
                );
                mg_level_constraints[level].close();
            }

        } else if (dim == 3)
        {
            pcout << "setting up BVs" << std::endl;
            for (const auto &cell: dof_handler.active_cell_iterators()) 
            {
                if(cell->is_locally_owned())
                {
                    for (unsigned int face_number = 0;
                         face_number < GeometryInfo<dim>::faces_per_cell;
                         ++face_number) 
                    {
                        if (cell->face(face_number)->at_boundary()) 
                        {
                            for (unsigned int vertex_number = 0;
                                 vertex_number < GeometryInfo<dim>::vertices_per_cell;
                                 ++vertex_number) {
                                const auto vert = cell->vertex(vertex_number);
                                /*Find bottom left corner*/
                                if (std::fabs(vert(0) - 0) < 1e-12 && 
                                    std::fabs(vert(1) - 0) < 1e-12 && 
                                    ((std::fabs(vert(2) - 0) < 1e-12) || (std::fabs(vert(2) - 1) < 1e-12))) 
                                {


                                    const unsigned int x_displacement =
                                            cell->vertex_dof_index(vertex_number, 0, cell->active_fe_index());
                                    const unsigned int y_displacement =
                                            cell->vertex_dof_index(vertex_number, 1, cell->active_fe_index());
                                    const unsigned int z_displacement =
                                            cell->vertex_dof_index(vertex_number, 2, cell->active_fe_index());
                                    const unsigned int x_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 3, cell->active_fe_index());
                                    const unsigned int y_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 4, cell->active_fe_index());
                                    const unsigned int z_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 5, cell->active_fe_index());

                                    boundary_values[x_displacement] = 0;
                                    boundary_values[y_displacement] = 0;
                                    boundary_values[z_displacement] = 0;
                                    boundary_values[x_displacement_multiplier] = 0;
                                    boundary_values[y_displacement_multiplier] = 0;
                                    boundary_values[z_displacement_multiplier] = 0;
                                }
                                /*Find bottom right corner*/
                                if (std::fabs(vert(0) - 6) < 1e-12 && 
                                    std::fabs(vert(1) - 0) < 1e-12 && 
                                    ((std::fabs(vert(2) - 0) < 1e-12) || (std::fabs(vert(2) - 1) < 1e-12))) 
                                {
                                    //                              const unsigned int x_displacement =
                                    //                                    cell->vertex_dof_index(vertex_number, 0, cell->active_fe_index());
                                    const unsigned int y_displacement =
                                            cell->vertex_dof_index(vertex_number, 1, cell->active_fe_index());
                                    const unsigned int z_displacement =
                                            cell->vertex_dof_index(vertex_number, 2, cell->active_fe_index());
                                    //                              const unsigned int x_displacement_multiplier =
                                    //                                    cell->vertex_dof_index(vertex_number, 3, cell->active_fe_index());
                                    const unsigned int y_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 4, cell->active_fe_index());
                                    const unsigned int z_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 5, cell->active_fe_index());
                                    //                              boundary_values[x_displacement] = 0;
                                    boundary_values[y_displacement] = 0;
                                    boundary_values[z_displacement] = 0;
                                    //                              boundary_values[x_displacement_multiplier] = 0;
                                    boundary_values[y_displacement_multiplier] = 0;
                                    boundary_values[z_displacement_multiplier] = 0;
                                }
                            }
                        }
                    }
                }

            }

            const unsigned int n_levels = triangulation.n_global_levels();
            level_dirichlet_boundary_dofs.resize(0,n_levels-1);
            level_boundary_values.resize(0,n_levels-1);
            mg_level_constraints.resize(0,n_levels-1);

            for(unsigned int level = 0; level < n_levels; ++level)
            {
                IndexSet relevant_dofs;
                DoFTools::extract_locally_relevant_level_dofs(dof_handler_displacement,
                                                      level,
                                                      relevant_dofs);
                mg_level_constraints[level].reinit(relevant_dofs);
            }

            for (auto cell=dof_handler_displacement.begin_active(n_levels-1);
                    cell!=dof_handler_displacement.end_active(n_levels-1);
                    ++cell)
            {
                if(cell->is_locally_owned())
                {
                    for (unsigned int face_number = 0;
                            face_number < GeometryInfo<dim>::faces_per_cell;
                            ++face_number)
                    {
                        if (cell->face(face_number)->at_boundary())
                        {
                            for (unsigned int vertex_number = 0;
                                    vertex_number < GeometryInfo<dim>::vertices_per_cell;
                                    ++vertex_number)
                            {
                                const auto vert = cell->vertex(vertex_number);
                                /*Find bottom left corner*/
                                if (std::fabs(vert(0) - 0) < 1e-12 && 
                                    std::fabs(vert(1) - 0) < 1e-12 && 
                                    ((std::fabs(vert(2) - 0) < 1e-12) || (std::fabs(vert(2) - 1) < 1e-12)))
                                {
                                    for (unsigned int level = 0; level < n_levels; ++level)
                                    {
                                        const unsigned int x_displacement =
                                                 cell->mg_vertex_dof_index(level, vertex_number, 0, cell->active_fe_index());
                                        const unsigned int y_displacement =
                                            cell->mg_vertex_dof_index(level, vertex_number, 1, cell->active_fe_index());
                                        const unsigned int z_displacement =
                                            cell->mg_vertex_dof_index(level, vertex_number, 2, cell->active_fe_index());
                                        /*set bottom left BC*/
                                        level_boundary_values[level][x_displacement] = 0;
                                        level_boundary_values[level][y_displacement] = 0;
                                        level_boundary_values[level][z_displacement] = 0;
                                        level_dirichlet_boundary_dofs[level].insert(x_displacement);
                                        level_dirichlet_boundary_dofs[level].insert(y_displacement);
                                        level_dirichlet_boundary_dofs[level].insert(z_displacement);
                                    }
                                }
                                /*Find bottom right corner*/
                                if (std::fabs(vert(0) - 6) < 1e-12 && 
                                    std::fabs(vert(1) - 0) < 1e-12 && 
                                    ((std::fabs(vert(2) - 0) < 1e-12) || 
                                     (std::fabs(vert(2) - 1) < 1e-12)))
                                {
                                    for (unsigned int level = 0; level < n_levels; ++level)
                                    {
                                        const unsigned int y_displacement =
                                                cell->mg_vertex_dof_index(level, vertex_number, 1, cell->active_fe_index());
                                        const unsigned int z_displacement =
                                                cell->mg_vertex_dof_index(level, vertex_number, 2, cell->active_fe_index());
                                        level_boundary_values[level][y_displacement] = 0;
                                        level_boundary_values[level][z_displacement] = 0;

                                        level_dirichlet_boundary_dofs[level].insert(y_displacement);
                                        level_dirichlet_boundary_dofs[level].insert(z_displacement);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            for (unsigned int level = 0; level < n_levels; ++level)
            {
                IndexSet relevant_dofs;
                DoFTools::extract_locally_relevant_level_dofs(dof_handler_displacement,level,relevant_dofs);
                mg_level_constraints[level].add_lines(level_dirichlet_boundary_dofs[level]);
                mg_level_constraints[level].make_consistent_in_parallel(
                    dof_handler_displacement.locally_owned_mg_dofs(level),
                    relevant_dofs,
                    mpi_communicator
                );
                mg_level_constraints[level].close();
            }

        }
        else
        {
            throw;
        }
    } else if (Input::geometry_base == GeometryOptions::l_shape) {
        if (dim == 2)
        {
            for (const auto &cell: dof_handler.active_cell_iterators())
            {
                if(cell->is_locally_owned())
                {
                    for (unsigned int face_number = 0;
                         face_number < GeometryInfo<dim>::faces_per_cell;
                         ++face_number) {
                        if (cell->face(face_number)->at_boundary()) {
                            for (unsigned int vertex_number = 0;
                                 vertex_number < GeometryInfo<dim>::vertices_per_cell;
                                 ++vertex_number) {
                                const auto vert = cell->vertex(vertex_number);
                                /*Find top left corner*/
                                if (std::fabs(vert(0) - 0) < 1e-12 && std::fabs(
                                            vert(1) - 2) < 1e-12) {

                                    const unsigned int x_displacement =
                                            cell->vertex_dof_index(vertex_number, 0, cell->active_fe_index());
                                    const unsigned int y_displacement =
                                            cell->vertex_dof_index(vertex_number, 1, cell->active_fe_index());
                                    const unsigned int x_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 2, cell->active_fe_index());
                                    const unsigned int y_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 3, cell->active_fe_index());
                                    /*set bottom left BC*/
                                    boundary_values[x_displacement] = 0;
                                    boundary_values[y_displacement] = 0;
                                    boundary_values[x_displacement_multiplier] = 0;
                                    boundary_values[y_displacement_multiplier] = 0;
                                }
                                /*Find top right corner*/
                                if (std::fabs(vert(0) - 1) < 1e-12 && std::fabs(
                                            vert(1) - 2) < 1e-12) {
                                    const unsigned int x_displacement =
                                            cell->vertex_dof_index(vertex_number, 0, cell->active_fe_index());
                                    const unsigned int y_displacement =
                                            cell->vertex_dof_index(vertex_number, 1, cell->active_fe_index());
                                    const unsigned int x_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 2, cell->active_fe_index());
                                    const unsigned int y_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 3, cell->active_fe_index());
                                    boundary_values[x_displacement] = 0;
                                    boundary_values[y_displacement] = 0;
                                    boundary_values[x_displacement_multiplier] = 0;
                                    boundary_values[y_displacement_multiplier] = 0;
                                }
                            }
                        }
                    }
                }
            }
            const unsigned int n_levels = triangulation.n_global_levels();
            for (unsigned int level = 0; level < n_levels; ++level)
            {
                for (auto cell=dof_handler_displacement.begin_active(level);
                     cell!=dof_handler.end_active(level);
                     ++cell)
                {
                    if(cell->is_locally_owned())
                    {
                        for (unsigned int face_number = 0;
                             face_number < GeometryInfo<dim>::faces_per_cell;
                             ++face_number)
                        {
                            if (cell->face(face_number)->at_boundary())
                            {
                                for (unsigned int vertex_number = 0;
                                     vertex_number < GeometryInfo<dim>::vertices_per_cell;
                                     ++vertex_number)
                                {
                                    const auto vert = cell->vertex(vertex_number);
                                    /*Find bottom left corner*/
                                    if (std::fabs(vert(0) - 0) < 1e-12 && std::fabs(
                                                vert(1) - 2) < 1e-12)
                                    {

                                        const unsigned int x_displacement =
                                                cell->mg_vertex_dof_index(level, vertex_number, 0, 0);
                                        const unsigned int y_displacement =
                                                cell->mg_vertex_dof_index(level, vertex_number, 1, 0);
                                        /*set bottom left BC*/
                                        level_boundary_values[level][x_displacement] = 0;
                                        level_boundary_values[level][y_displacement] = 0;

                                        level_dirichlet_boundary_dofs[level].insert(x_displacement);
                                        level_dirichlet_boundary_dofs[level].insert(y_displacement);

                                    }
                                    /*Find bottom right corner*/
                                    if (std::fabs(vert(0) - 1) < 1e-12 && std::fabs(
                                                vert(1) - 2) < 1e-12)
                                    {
                                        const unsigned int x_displacement =
                                                cell->mg_vertex_dof_index(level, vertex_number, 0, 0);
                                        const unsigned int y_displacement =
                                                cell->mg_vertex_dof_index(level, vertex_number, 1, 0);
                                        level_boundary_values[level][x_displacement] = 0;
                                        level_boundary_values[level][y_displacement] = 0;

                                        level_dirichlet_boundary_dofs[level].insert(x_displacement);
                                        level_dirichlet_boundary_dofs[level].insert(y_displacement);
                                    }
                                }
                            }
                        }
                    }
                }
                IndexSet relevant_dofs;
                DoFTools::extract_locally_relevant_level_dofs(dof_handler_displacement,level,relevant_dofs);
                mg_level_constraints[level].add_lines(level_dirichlet_boundary_dofs[level]);
                mg_level_constraints[level].make_consistent_in_parallel(
                    dof_handler_displacement.locally_owned_mg_dofs(level),
                    relevant_dofs,
                    mpi_communicator
                );
                mg_level_constraints[level].close();
            }

        }
        else if (dim == 3)
        {
            for (const auto &cell: dof_handler.active_cell_iterators()) {
                if(cell->is_locally_owned())
                {
                    for (unsigned int face_number = 0;
                         face_number < GeometryInfo<dim>::faces_per_cell;
                         ++face_number) {
                        if (cell->face(face_number)->at_boundary()) {
                            for (unsigned int vertex_number = 0;
                                 vertex_number < GeometryInfo<dim>::vertices_per_cell;
                                 ++vertex_number) {
                                const auto vert = cell->vertex(vertex_number);
                                /*Find bottom left corner*/
                                if (std::fabs(vert(0) - 0) < 1e-12 && std::fabs(
                                            vert(1) - 2) < 1e-12 && ((std::fabs(
                                                                          vert(2) - 0) < 1e-12) || (std::fabs(
                                                                                                        vert(2) - 1) < 1e-12))) {


                                    const unsigned int x_displacement =
                                            cell->vertex_dof_index(vertex_number, 0, cell->active_fe_index());
                                    const unsigned int y_displacement =
                                            cell->vertex_dof_index(vertex_number, 1, cell->active_fe_index());
                                    const unsigned int z_displacement =
                                            cell->vertex_dof_index(vertex_number, 2, cell->active_fe_index());
                                    const unsigned int x_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 3, cell->active_fe_index());
                                    const unsigned int y_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 4, cell->active_fe_index());
                                    const unsigned int z_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 5, cell->active_fe_index());

                                    boundary_values[x_displacement] = 0;
                                    boundary_values[y_displacement] = 0;
                                    boundary_values[z_displacement] = 0;
                                    boundary_values[x_displacement_multiplier] = 0;
                                    boundary_values[y_displacement_multiplier] = 0;
                                    boundary_values[z_displacement_multiplier] = 0;
                                }
                                /*Find bottom right corner*/
                                if (std::fabs(vert(0) - 1) < 1e-12 && std::fabs(
                                            vert(1) - 2) < 1e-12 && ((std::fabs(
                                                                          vert(2) - 0) < 1e-12) || (std::fabs(
                                                                                                        vert(2) - 1) < 1e-12))) {
                                    const unsigned int x_displacement =
                                            cell->vertex_dof_index(vertex_number, 0, cell->active_fe_index());
                                    const unsigned int y_displacement =
                                            cell->vertex_dof_index(vertex_number, 1, cell->active_fe_index());
                                    const unsigned int z_displacement =
                                            cell->vertex_dof_index(vertex_number, 2, cell->active_fe_index());
                                    const unsigned int x_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 3, cell->active_fe_index());
                                    const unsigned int y_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 4, cell->active_fe_index());
                                    const unsigned int z_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 5, cell->active_fe_index());
                                    boundary_values[x_displacement] = 0;
                                    boundary_values[y_displacement] = 0;
                                    boundary_values[z_displacement] = 0;
                                    boundary_values[x_displacement_multiplier] = 0;
                                    boundary_values[y_displacement_multiplier] = 0;
                                    boundary_values[z_displacement_multiplier] = 0;
                                }
                            }
                        }
                    }
                }
            }
            const unsigned int n_levels = triangulation.n_global_levels();
            for (unsigned int level = 0; level < n_levels; ++level)
            {
                for (auto cell=dof_handler_displacement.begin_active(level);
                     cell!=dof_handler.end_active(level);
                     ++cell)
                {
                    if(cell->is_locally_owned())
                    {
                        for (unsigned int face_number = 0;
                             face_number < GeometryInfo<dim>::faces_per_cell;
                             ++face_number)
                        {
                            if (cell->face(face_number)->at_boundary())
                            {
                                for (unsigned int vertex_number = 0;
                                     vertex_number < GeometryInfo<dim>::vertices_per_cell;
                                     ++vertex_number)
                                {
                                    const auto vert = cell->vertex(vertex_number);
                                    /*Find bottom left corner*/
                                    if (std::fabs(vert(0) - 0) < 1e-12 && std::fabs(
                                                vert(1) - 2) < 1e-12 && ((std::fabs(
                                                vert(2) - 0) < 1e-12) || (std::fabs(
                                                vert(2) - 1) < 1e-12)))
                                    {

                                        const unsigned int x_displacement =
                                                cell->mg_vertex_dof_index(level, vertex_number, 0, 0);
                                        const unsigned int y_displacement =
                                                cell->mg_vertex_dof_index(level, vertex_number, 1, 0);
                                        const unsigned int z_displacement =
                                                cell->mg_vertex_dof_index(level, vertex_number, 2, 0);
                                        /*set bottom left BC*/
                                        level_boundary_values[level][x_displacement] = 0;
                                        level_boundary_values[level][y_displacement] = 0;
                                        level_boundary_values[level][z_displacement] = 0;

                                        level_dirichlet_boundary_dofs[level].insert(x_displacement);
                                        level_dirichlet_boundary_dofs[level].insert(y_displacement);
                                        level_dirichlet_boundary_dofs[level].insert(z_displacement);
                                    }
                                    /*Find bottom right corner*/
                                    if (std::fabs(vert(0) - 1) < 1e-12 && std::fabs(
                                                vert(1) - 2) < 1e-12 && ((std::fabs(
                                                vert(2) - 0) < 1e-12) || (std::fabs(
                                                vert(2) - 1) < 1e-12)))
                                    {
                                        const unsigned int x_displacement =
                                                cell->mg_vertex_dof_index(level, vertex_number, 0, 0);
                                        const unsigned int y_displacement =
                                                cell->mg_vertex_dof_index(level, vertex_number, 1, 0);
                                        const unsigned int z_displacement =
                                                cell->mg_vertex_dof_index(level, vertex_number, 2, 0);
                                        level_boundary_values[level][x_displacement] = 0;
                                        level_boundary_values[level][y_displacement] = 0;
                                        level_boundary_values[level][z_displacement] = 0;

                                        level_dirichlet_boundary_dofs[level].insert(x_displacement);
                                        level_dirichlet_boundary_dofs[level].insert(y_displacement);
                                        level_dirichlet_boundary_dofs[level].insert(z_displacement);
                                    }
                                }
                            }
                        }
                    }
                }
                IndexSet relevant_dofs;
                DoFTools::extract_locally_relevant_level_dofs(dof_handler_displacement,level,relevant_dofs);
                mg_level_constraints[level].add_lines(level_dirichlet_boundary_dofs[level]);
                mg_level_constraints[level].make_consistent_in_parallel(
                    dof_handler_displacement.locally_owned_mg_dofs(level),
                    relevant_dofs,
                    mpi_communicator
                );
                mg_level_constraints[level].close();
            }

        } else {
            throw;
        }
    }


}


///This makes a giant 10-by-10 block matrix that when assembled will represents the 10 KKT equations that
/// come from this problem, and also sets up the necessary block vectors.  The
/// sparsity pattern for this matrix includes the sparsity pattern for the filter matrix. It also initializes
/// any block vectors we will use.
template<int dim>
void
KktSystem<dim>::setup_block_system() {

    //MAKE n_u and n_P

    /*Setup 10 by 10 block matrix*/
    std::vector<unsigned int> block_component(10, 2);

    block_component[0] = 0;
    block_component[5] = 1;

    const std::vector<types::global_dof_index> dofs_per_block =
            DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    const unsigned int n_p = dofs_per_block[0];
    const unsigned int n_u = dofs_per_block[1];

    pcout << "n_p:  " << n_p << "   n_u:  " << n_u << std::endl;

    IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    dsp.reinit(10, 10);
    owned_partitioning.resize(10);
    owned_partitioning[0] = dof_handler.locally_owned_dofs().get_view(0, n_p);
    owned_partitioning[1] = dof_handler.locally_owned_dofs().get_view(n_p, 2 * n_p);
    owned_partitioning[2] = dof_handler.locally_owned_dofs().get_view(2 * n_p, 3 * n_p);
    owned_partitioning[3] = dof_handler.locally_owned_dofs().get_view(3 * n_p, 4 * n_p);
    owned_partitioning[4] = dof_handler.locally_owned_dofs().get_view(4 * n_p, 5 * n_p);
    owned_partitioning[5] = dof_handler.locally_owned_dofs().get_view(5 * n_p, 5 * n_p + n_u);
    owned_partitioning[6] = dof_handler.locally_owned_dofs().get_view(5 * n_p + n_u, 5 * n_p + 2 * n_u);
    owned_partitioning[7] = dof_handler.locally_owned_dofs().get_view(5 * n_p + 2 * n_u, 6 * n_p + 2 * n_u);
    owned_partitioning[8] = dof_handler.locally_owned_dofs().get_view(6 * n_p + 2 * n_u, 7 * n_p + 2 * n_u);
    owned_partitioning[9] = dof_handler.locally_owned_dofs().get_view(7 * n_p + 2 * n_u, 7 * n_p + 2 * n_u + 1);
    relevant_partitioning.resize(10);
    relevant_partitioning[0] = locally_relevant_dofs.get_view(0, n_p);
    relevant_partitioning[1] = locally_relevant_dofs.get_view(n_p, 2 * n_p);
    relevant_partitioning[2] = locally_relevant_dofs.get_view(2 * n_p, 3 * n_p);
    relevant_partitioning[3] = locally_relevant_dofs.get_view(3 * n_p, 4 * n_p);
    relevant_partitioning[4] = locally_relevant_dofs.get_view(4 * n_p, 5 * n_p);
    relevant_partitioning[5] = locally_relevant_dofs.get_view(5 * n_p, 5 * n_p + n_u);
    relevant_partitioning[6] = locally_relevant_dofs.get_view(5 * n_p + n_u, 5 * n_p + 2 * n_u);
    relevant_partitioning[7] = locally_relevant_dofs.get_view(5 * n_p + 2 * n_u, 6 * n_p + 2 * n_u);
    relevant_partitioning[8] = locally_relevant_dofs.get_view(6 * n_p + 2 * n_u, 7 * n_p + 2 * n_u);
    relevant_partitioning[9] = locally_relevant_dofs.get_view(7 * n_p + 2 * n_u, 7 * n_p + 2 * n_u + 1);

    const std::vector<unsigned int> block_sizes = {n_p, n_p, n_p, n_p, n_p, n_u, n_u, n_p, n_p, 1};

    for (unsigned int k = 0; k < 10; k++) {
        for (unsigned int j = 0; j < 10; j++) {
            dsp.block(j, k).reinit(block_sizes[j], block_sizes[k]);
        }
    }
    dsp.collect_sizes();
    Table<2, DoFTools::Coupling> coupling(2 * dim + 8, 2 * dim + 8);
    //Coupling for density
    coupling[SolutionComponents::density<dim>][SolutionComponents::density<dim>] = DoFTools::always;

    for (unsigned int i = 0; i < dim; i++) {
        coupling[SolutionComponents::density<dim>][SolutionComponents::displacement<dim> +
                i] = DoFTools::always;
        coupling[SolutionComponents::displacement<dim> +
                i][SolutionComponents::density<dim>] = DoFTools::always;
    }

    coupling[SolutionComponents::density<dim>][SolutionComponents::unfiltered_density_multiplier<dim>] = DoFTools::always;
    coupling[SolutionComponents::unfiltered_density_multiplier<dim>][SolutionComponents::density<dim>] = DoFTools::always;

    for (unsigned int i = 0; i < dim; i++) {
        coupling[SolutionComponents::density<dim>][SolutionComponents::displacement_multiplier<dim> +
                i] = DoFTools::always;
        coupling[SolutionComponents::displacement_multiplier<dim> +
                i][SolutionComponents::density<dim>] = DoFTools::always;
    }

    //Coupling for displacement
    for (unsigned int i = 0; i < dim; i++) {

        for (unsigned int k = 0; k < dim; k++) {
            coupling[SolutionComponents::displacement<dim> + i][
                    SolutionComponents::displacement_multiplier<dim> +
                    k] = DoFTools::always;
            coupling[SolutionComponents::displacement_multiplier<dim> + k][
                    SolutionComponents::displacement<dim> +
                    i] = DoFTools::always;
        }
    }

    // coupling for unfiltered density
    coupling[SolutionComponents::unfiltered_density<dim>][SolutionComponents::density_lower_slack_multiplier<dim>] = DoFTools::always;
    coupling[SolutionComponents::density_lower_slack_multiplier<dim>][SolutionComponents::unfiltered_density<dim>] = DoFTools::always;

    coupling[SolutionComponents::unfiltered_density<dim>][SolutionComponents::density_upper_slack_multiplier<dim>] = DoFTools::always;
    coupling[SolutionComponents::density_upper_slack_multiplier<dim>][SolutionComponents::unfiltered_density<dim>] = DoFTools::always;

    coupling[SolutionComponents::unfiltered_density<dim>][SolutionComponents::unfiltered_density_multiplier<dim>] = DoFTools::always;
    coupling[SolutionComponents::unfiltered_density_multiplier<dim>][SolutionComponents::unfiltered_density<dim>] = DoFTools::always;

    //        Coupling for lower slack
    coupling[SolutionComponents::density_lower_slack<dim>][SolutionComponents::density_lower_slack<dim>] = DoFTools::always;

    coupling[SolutionComponents::density_lower_slack<dim>][SolutionComponents::density_lower_slack_multiplier<dim>] = DoFTools::always;
    coupling[SolutionComponents::density_lower_slack_multiplier<dim>][SolutionComponents::density_lower_slack<dim>] = DoFTools::always;

    //
    coupling[SolutionComponents::density_upper_slack<dim>][SolutionComponents::density_upper_slack<dim>] = DoFTools::always;
    coupling[SolutionComponents::density_upper_slack<dim>][SolutionComponents::density_upper_slack_multiplier<dim>] = DoFTools::always;
    coupling[SolutionComponents::density_upper_slack_multiplier<dim>][SolutionComponents::density_upper_slack<dim>] = DoFTools::always;

    coupling[SolutionComponents::density_upper_slack_multiplier<dim>][SolutionComponents::density_upper_slack_multiplier<dim>] = DoFTools::always;
    constraints.reinit(locally_relevant_dofs);
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler,constraints);
    constraints.close();

    system_matrix.clear();

    //            DoFTools::make_sparsity_pattern(dof_handler, coupling, dsp, constraints, false);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.locally_owned_dofs(), mpi_communicator,
                                               locally_relevant_dofs);
    //adds the row into the sparsity pattern for the total volume constraint
    // for (const auto &cell: dof_handler.active_cell_iterators())
    // {
    //     if (cell->is_locally_owned())
    //     {
    //         std::vector<types::global_dof_index> i(cell->get_fe().n_dofs_per_cell());
    //         cell->get_dof_indices(i);
    //         dsp.block(SolutionBlocks::density, SolutionBlocks::total_volume_multiplier).add(i[cell->get_fe().component_to_system_index(0, 0)], 0);
    //         dsp.block(SolutionBlocks::total_volume_multiplier, SolutionBlocks::density).add(0, i[cell->get_fe().component_to_system_index(0, 0)]);
    //     }
    // }
    // Because of the single volume multiplier element only being on one processor, this works, and the above does not.
    for (unsigned int i = 0; i<n_p; i++)
    {
        dsp.block(SolutionBlocks::density, SolutionBlocks::total_volume_multiplier).add(i,0);
        dsp.block(SolutionBlocks::total_volume_multiplier, SolutionBlocks::density).add(0,i);
    }

    /*This finds neighbors whose values would be relevant, and adds them to the sparsity pattern of the matrix*/
    setup_filter_matrix();
    for (const auto &cell : dof_handler.active_cell_iterators()) {
        if (cell->is_locally_owned())
        {
            std::vector<types::global_dof_index> i(cell->get_fe().n_dofs_per_cell());
            cell->get_dof_indices(i);
            const unsigned int cell_index = i[cell->get_fe().component_to_system_index(0, 0)];
            for (const auto &neighbor_cell_index : density_filter.find_relevant_neighbors(cell_index))
            {
                dsp.block(SolutionBlocks::unfiltered_density_multiplier,
                          SolutionBlocks::unfiltered_density).add(cell_index, neighbor_cell_index);
                dsp.block(SolutionBlocks::unfiltered_density,
                          SolutionBlocks::unfiltered_density_multiplier).add(cell_index, neighbor_cell_index);
            }
        }
    }
    
    SparsityTools::distribute_sparsity_pattern(
                dsp,
                Utilities::MPI::all_gather(mpi_communicator,
                                           dof_handler.locally_owned_dofs()),
                mpi_communicator,
                locally_relevant_dofs);
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    system_matrix.reinit(owned_partitioning, dsp, mpi_communicator);

    locally_relevant_solution.reinit(owned_partitioning, relevant_partitioning, mpi_communicator);
    distributed_solution.reinit(owned_partitioning, mpi_communicator);
    system_rhs.reinit(owned_partitioning, mpi_communicator);

    locally_relevant_solution.collect_sizes();
    distributed_solution.collect_sizes();
    system_rhs.collect_sizes();
    system_matrix.collect_sizes();
    IndexSet locally_owned_displacement_dofs = dof_handler_displacement.locally_owned_dofs();
    std::vector<types::global_dof_index> displacement_dof_indices;
    std::vector<types::global_dof_index> system_dof_indices;
    for (const auto &displacement_cell : dof_handler_displacement.active_cell_iterators())
        if (displacement_cell->is_locally_owned())
        {
            typename DoFHandler<dim>::active_cell_iterator system_cell (&displacement_cell->get_triangulation(),
                                                                        displacement_cell->level(),
                                                                        displacement_cell->index(),
                                                                        &dof_handler);

            displacement_dof_indices.resize (displacement_cell->get_fe().dofs_per_cell);
            system_dof_indices.resize (system_cell->get_fe().dofs_per_cell);

            displacement_cell->get_dof_indices (displacement_dof_indices);
            system_cell->get_dof_indices (system_dof_indices);

            for (unsigned int i=0; i<displacement_dof_indices.size(); ++i)
            {
                if(locally_owned_displacement_dofs.is_element(displacement_dof_indices[i]))
                {
                        displacement_to_system_dof_index_map[displacement_dof_indices[i]]
                            = system_dof_indices[system_cell->get_fe().component_to_system_index(
                            displacement_cell->get_fe().system_to_component_index(i).first+SolutionComponents::displacement<dim>,
                            displacement_cell->get_fe().system_to_component_index(i).second
                            )];
                }
                
            }
        }
    const types::global_dof_index disp_start_index = system_matrix.get_row_indices().block_start(
            SolutionBlocks::displacement);
    for (auto &index_pair : displacement_to_system_dof_index_map)
        index_pair.second -=disp_start_index;
    for (auto &index_pair : displacement_to_system_dof_index_map)
    {
        if(index_pair.first != index_pair.second)
        {
            std::cout << "inexact matching for index: " << index_pair.first << " and " << index_pair.second << std::endl;
        }
    }
    

}

///The  equations  describing  the newtons method for finding 0s in the KKT conditions are implemented here.
template<int dim>
void
KktSystem<dim>::assemble_block_system(const LA::MPI::BlockVector &distributed_state, const double barrier_size) {
    /*Remove any values from old iterations*/

    LA::MPI::BlockVector relevant_state(owned_partitioning, relevant_partitioning, mpi_communicator);
    relevant_state = distributed_state;

    system_matrix.reinit(owned_partitioning, dsp, mpi_communicator);
    locally_relevant_solution = 0;
    system_rhs = 0;

    QGauss<dim> nine_quadrature(fe_nine.degree + 1);
    QGauss<dim> ten_quadrature(fe_ten.degree + 1);

    hp::QCollection<dim> q_collection;
    q_collection.push_back(nine_quadrature);
    q_collection.push_back(ten_quadrature);

    hp::FEValues<dim> hp_fe_values(fe_collection,
                                   q_collection,
                                   update_values | update_quadrature_points |
                                   update_JxW_values | update_gradients);

    QGauss<dim - 1> common_face_quadrature(fe_ten.degree + 1);

    FEFaceValues<dim> fe_nine_face_values(fe_nine,
                                          common_face_quadrature,
                                          update_JxW_values |
                                          update_gradients | update_values);
    FEFaceValues<dim> fe_ten_face_values(fe_ten,
                                         common_face_quadrature,
                                         update_normal_vectors |
                                         update_values);

    FullMatrix<double> cell_matrix;
    Vector<double> cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;

    const FEValuesExtractors::Scalar densities(SolutionComponents::density<dim>);
    const FEValuesExtractors::Vector displacements(SolutionComponents::displacement<dim>);
    const FEValuesExtractors::Scalar unfiltered_densities(SolutionComponents::unfiltered_density<dim>);
    const FEValuesExtractors::Vector displacement_multipliers(SolutionComponents::displacement_multiplier<dim>);
    const FEValuesExtractors::Scalar unfiltered_density_multipliers(
                SolutionComponents::unfiltered_density_multiplier<dim>);
    const FEValuesExtractors::Scalar density_lower_slacks(SolutionComponents::density_lower_slack<dim>);
    const FEValuesExtractors::Scalar density_lower_slack_multipliers(
                SolutionComponents::density_lower_slack_multiplier<dim>);
    const FEValuesExtractors::Scalar density_upper_slacks(SolutionComponents::density_upper_slack<dim>);
    const FEValuesExtractors::Scalar density_upper_slack_multipliers(
                SolutionComponents::density_upper_slack_multiplier<dim>);
    const FEValuesExtractors::Scalar total_volume_multiplier(
                SolutionComponents::total_volume_multiplier<dim>);

    const Functions::ConstantFunction<dim> lambda(Input::material_lambda), mu(Input::material_mu);

    distributed_solution = distributed_state;
    LA::MPI::BlockVector filtered_unfiltered_density_solution = distributed_solution;
    LA::MPI::BlockVector filter_adjoint_unfiltered_density_multiplier_solution = distributed_solution;
    filtered_unfiltered_density_solution.block(SolutionBlocks::unfiltered_density) = 0;
    filter_adjoint_unfiltered_density_multiplier_solution.block(SolutionBlocks::unfiltered_density_multiplier) = 0;
    density_filter.filter_matrix.vmult(filtered_unfiltered_density_solution.block(SolutionBlocks::unfiltered_density),distributed_solution.block(SolutionBlocks::unfiltered_density));
    density_filter.filter_matrix_transpose.vmult(filter_adjoint_unfiltered_density_multiplier_solution.block(SolutionBlocks::unfiltered_density_multiplier),distributed_solution.block(SolutionBlocks::unfiltered_density_multiplier));

    LA::MPI::BlockVector relevant_filtered_unfiltered_density_solution = locally_relevant_solution;
    LA::MPI::BlockVector relevant_filter_adjoint_unfiltered_density_multiplier_solution = locally_relevant_solution;
    relevant_filtered_unfiltered_density_solution =filtered_unfiltered_density_solution;
    relevant_filter_adjoint_unfiltered_density_multiplier_solution = filter_adjoint_unfiltered_density_multiplier_solution;
    for (const auto &cell: dof_handler.active_cell_iterators()) {
        if(cell->is_locally_owned())
        {
            hp_fe_values.reinit(cell);
            const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
            cell_matrix.reinit(cell->get_fe().n_dofs_per_cell(),
                               cell->get_fe().n_dofs_per_cell());
            cell_rhs.reinit(cell->get_fe().n_dofs_per_cell());

            const unsigned int n_q_points = fe_values.n_quadrature_points;

            std::vector<double> old_density_values(n_q_points);
            std::vector<Tensor<1, dim>> old_displacement_values(n_q_points);
            std::vector<double> old_displacement_divs(n_q_points);
            std::vector<SymmetricTensor<2, dim>> old_displacement_symmgrads(
                        n_q_points);
            std::vector<Tensor<1, dim>> old_displacement_multiplier_values(
                        n_q_points);
            std::vector<double> old_displacement_multiplier_divs(n_q_points);
            std::vector<SymmetricTensor<2, dim>> old_displacement_multiplier_symmgrads(
                        n_q_points);
            std::vector<double> old_lower_slack_multiplier_values(n_q_points);
            std::vector<double> old_upper_slack_multiplier_values(n_q_points);
            std::vector<double> old_lower_slack_values(n_q_points);
            std::vector<double> old_upper_slack_values(n_q_points);
            std::vector<double> old_unfiltered_density_values(n_q_points);
            std::vector<double> old_unfiltered_density_multiplier_values(n_q_points);
            std::vector<double> filtered_unfiltered_density_values(n_q_points);
            std::vector<double> filter_adjoint_unfiltered_density_multiplier_values(n_q_points);
            std::vector<double> lambda_values(n_q_points);
            std::vector<double> mu_values(n_q_points);

            const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();

            cell_matrix = 0;
            cell_rhs = 0;
            local_dof_indices.resize(cell->get_fe().n_dofs_per_cell());
            cell->get_dof_indices(local_dof_indices);

            lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
            mu.value_list(fe_values.get_quadrature_points(), mu_values);

            fe_values[densities].get_function_values(relevant_state,
                                                     old_density_values);
            fe_values[displacements].get_function_values(relevant_state,
                                                         old_displacement_values);
            fe_values[displacements].get_function_divergences(relevant_state,
                                                              old_displacement_divs);
            fe_values[displacements].get_function_symmetric_gradients(
                        relevant_state, old_displacement_symmgrads);
            fe_values[displacement_multipliers].get_function_values(
                        relevant_state, old_displacement_multiplier_values);
            fe_values[displacement_multipliers].get_function_divergences(
                        relevant_state, old_displacement_multiplier_divs);
            fe_values[displacement_multipliers].get_function_symmetric_gradients(
                        relevant_state, old_displacement_multiplier_symmgrads);
            fe_values[density_lower_slacks].get_function_values(
                        relevant_state, old_lower_slack_values);
            fe_values[density_lower_slack_multipliers].get_function_values(
                        relevant_state, old_lower_slack_multiplier_values);
            fe_values[density_upper_slacks].get_function_values(
                        relevant_state, old_upper_slack_values);
            fe_values[density_upper_slack_multipliers].get_function_values(
                        relevant_state, old_upper_slack_multiplier_values);
            fe_values[unfiltered_densities].get_function_values(
                        relevant_state, old_unfiltered_density_values);
            fe_values[unfiltered_density_multipliers].get_function_values(
                        relevant_state, old_unfiltered_density_multiplier_values);
            fe_values[unfiltered_densities].get_function_values(
                        relevant_filtered_unfiltered_density_solution, filtered_unfiltered_density_values);
            fe_values[unfiltered_density_multipliers].get_function_values(
                        relevant_filter_adjoint_unfiltered_density_multiplier_solution,
                        filter_adjoint_unfiltered_density_multiplier_values);

            Tensor<1, dim> traction;
            traction[1] = -1;

            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    const SymmetricTensor<2, dim> displacement_phi_i_symmgrad =
                            fe_values[displacements].symmetric_gradient(i, q_point);
                    const double displacement_phi_i_div =
                            fe_values[displacements].divergence(i, q_point);

                    const SymmetricTensor<2, dim> displacement_multiplier_phi_i_symmgrad =
                            fe_values[displacement_multipliers].symmetric_gradient(i,
                                                                                   q_point);
                    const double displacement_multiplier_phi_i_div =
                            fe_values[displacement_multipliers].divergence(i,
                                                                           q_point);


                    const double density_phi_i = fe_values[densities].value(i,
                                                                            q_point);
                    const double unfiltered_density_phi_i = fe_values[unfiltered_densities].value(i,
                                                                                                  q_point);
                    const double unfiltered_density_multiplier_phi_i = fe_values[unfiltered_density_multipliers].value(
                                i, q_point);

                    const double lower_slack_multiplier_phi_i =
                            fe_values[density_lower_slack_multipliers].value(i,
                                                                             q_point);

                    const double lower_slack_phi_i =
                            fe_values[density_lower_slacks].value(i, q_point);

                    const double upper_slack_phi_i =
                            fe_values[density_upper_slacks].value(i, q_point);

                    const double upper_slack_multiplier_phi_i =
                            fe_values[density_upper_slack_multipliers].value(i,
                                                                             q_point);


                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                        const SymmetricTensor<2, dim> displacement_phi_j_symmgrad =
                                fe_values[displacements].symmetric_gradient(j,
                                                                            q_point);
                        const double displacement_phi_j_div =
                                fe_values[displacements].divergence(j, q_point);

                        const SymmetricTensor<2, dim> displacement_multiplier_phi_j_symmgrad =
                                fe_values[displacement_multipliers].symmetric_gradient(
                                    j, q_point);
                        const double displacement_multiplier_phi_j_div =
                                fe_values[displacement_multipliers].divergence(j,
                                                                               q_point);

                        const double density_phi_j = fe_values[densities].value(
                                    j, q_point);

                        const double unfiltered_density_phi_j = fe_values[unfiltered_densities].value(j,
                                                                                                      q_point);
                        const double unfiltered_density_multiplier_phi_j = fe_values[unfiltered_density_multipliers].value(
                                    j, q_point);


                        const double lower_slack_phi_j =
                                fe_values[density_lower_slacks].value(j, q_point);

                        const double upper_slack_phi_j =
                                fe_values[density_upper_slacks].value(j, q_point);

                        const double lower_slack_multiplier_phi_j =
                                fe_values[density_lower_slack_multipliers].value(j,
                                                                                 q_point);

                        const double upper_slack_multiplier_phi_j =
                                fe_values[density_upper_slack_multipliers].value(j,
                                                                                 q_point);

                        //Equation 0
                        cell_matrix(i, j) +=
                                fe_values.JxW(q_point) *
                                (
                                    -density_phi_i * unfiltered_density_multiplier_phi_j

                                    - density_penalty_exponent * (density_penalty_exponent - 1)
                                    * std::pow(
                                        old_density_values[q_point],
                                        density_penalty_exponent - 2)
                                    * density_phi_i
                                    * density_phi_j
                                    * (old_displacement_multiplier_divs[q_point] * old_displacement_divs[q_point]
                                       * lambda_values[q_point]
                                       + 2 * mu_values[q_point]
                                       * (old_displacement_symmgrads[q_point] *
                                          old_displacement_multiplier_symmgrads[q_point]))

                                    - density_penalty_exponent * std::pow(
                                        old_density_values[q_point],
                                        density_penalty_exponent - 1)
                                    * density_phi_i
                                    * (displacement_multiplier_phi_j_div * old_displacement_divs[q_point]
                                       * lambda_values[q_point]
                                       + 2 * mu_values[q_point]
                                       *
                                       (old_displacement_symmgrads[q_point] *
                                        displacement_multiplier_phi_j_symmgrad))

                                    - density_penalty_exponent * std::pow(
                                        old_density_values[q_point],
                                        density_penalty_exponent - 1)
                                    * density_phi_i
                                    * (displacement_phi_j_div * old_displacement_multiplier_divs[q_point]
                                       * lambda_values[q_point]
                                       + 2 * mu_values[q_point]
                                       * (old_displacement_multiplier_symmgrads[q_point] *
                                          displacement_phi_j_symmgrad)));
                        //Equation 1

                        cell_matrix(i, j) +=
                                fe_values.JxW(q_point) * (
                                    -density_penalty_exponent * std::pow(
                                        old_density_values[q_point],
                                        density_penalty_exponent - 1)
                                    * density_phi_j
                                    * (old_displacement_multiplier_divs[q_point] * displacement_phi_i_div
                                       * lambda_values[q_point]
                                       + 2 * mu_values[q_point]
                                       * (old_displacement_multiplier_symmgrads[q_point] *
                                          displacement_phi_i_symmgrad))

                                    - std::pow(old_density_values[q_point],
                                               density_penalty_exponent)
                                    * (displacement_multiplier_phi_j_div * displacement_phi_i_div
                                       * lambda_values[q_point]
                                       + 2 * mu_values[q_point]
                                       * (displacement_multiplier_phi_j_symmgrad * displacement_phi_i_symmgrad))

                                    );

                        //Equation 2 has to do with the filter, which is calculated elsewhere.
                        cell_matrix(i, j) +=
                                fe_values.JxW(q_point) * (
                                    -1 * unfiltered_density_phi_i * lower_slack_multiplier_phi_j
                                    + unfiltered_density_phi_i * upper_slack_multiplier_phi_j);

                        //Equation 3 - Primal Feasibility

                        cell_matrix(i, j) +=
                                fe_values.JxW(q_point) * (

                                    -1 * density_penalty_exponent * std::pow(
                                        old_density_values[q_point],
                                        density_penalty_exponent - 1)
                                    * density_phi_j
                                    * (old_displacement_divs[q_point] * displacement_multiplier_phi_i_div
                                       * lambda_values[q_point]
                                       + 2 * mu_values[q_point]
                                       * (old_displacement_symmgrads[q_point] *
                                          displacement_multiplier_phi_i_symmgrad))

                                    + -1 * std::pow(old_density_values[q_point],
                                                    density_penalty_exponent)
                                    * (displacement_phi_j_div * displacement_multiplier_phi_i_div
                                       * lambda_values[q_point]
                                       + 2 * mu_values[q_point]
                                       *
                                       (displacement_phi_j_symmgrad * displacement_multiplier_phi_i_symmgrad)));

                        //Equation 4 - more primal feasibility
                        cell_matrix(i, j) +=
                                -1 * fe_values.JxW(q_point) * lower_slack_multiplier_phi_i *
                                (unfiltered_density_phi_j - lower_slack_phi_j);

                        //Equation 5 - more primal feasibility
                        cell_matrix(i, j) +=
                                -1 * fe_values.JxW(q_point) * upper_slack_multiplier_phi_i * (
                                    -1 * unfiltered_density_phi_j - upper_slack_phi_j);

                        //Equation 6 - more primal feasibility - part with filter added later
                        cell_matrix(i, j) +=
                                -1 * fe_values.JxW(q_point) * unfiltered_density_multiplier_phi_i * (
                                    density_phi_j);

                        //Equation 7 - complementary slackness
                        cell_matrix(i, j) += fe_values.JxW(q_point) *
                                (lower_slack_phi_i * lower_slack_multiplier_phi_j
                                 + lower_slack_phi_i * lower_slack_phi_j *
                                 old_lower_slack_multiplier_values[q_point] /
                                 old_lower_slack_values[q_point]);
                        //Equation 8 - complementary slackness
                        cell_matrix(i, j) += fe_values.JxW(q_point) *
                                (upper_slack_phi_i * upper_slack_multiplier_phi_j
                                 + upper_slack_phi_i * upper_slack_phi_j *
                                 old_upper_slack_multiplier_values[q_point] /
                                 old_upper_slack_values[q_point]);
                    }

                }
            }


            MatrixTools::local_apply_boundary_values(boundary_values, local_dof_indices,
                                                     cell_matrix, cell_rhs, true);


            constraints.distribute_local_to_global(
                        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);

        }

    }
    // MPI_BARRIER(MPI_COMM_WORLD);
    system_matrix.compress(VectorOperation::add);
    system_rhs = calculate_rhs(distributed_state, barrier_size);
    double cell_measure;
    for (const auto &cell: dof_handler.active_cell_iterators()) 
    {
        if(cell->is_locally_owned())
        {
            std::vector<types::global_dof_index> i(cell->get_fe().n_dofs_per_cell());
            cell->get_dof_indices(i);
            const unsigned int cell_i = i[cell->get_fe().component_to_system_index(0, 0)];

            // typename LA::MPI::SparseMatrix::iterator iter = density_filter.filter_matrix.begin(cell_i);
            for (const unsigned int j : density_filter.find_relevant_neighbors(cell_i)) 
            {
                // unsigned int j = iter->column();
                double value = density_filter.filter_matrix(cell_i,j) * cell->measure();
                double value_transpose = density_filter.filter_matrix_transpose(cell_i,j) * cell->measure();

                system_matrix.block(SolutionBlocks::unfiltered_density_multiplier,
                                    SolutionBlocks::unfiltered_density).set(cell_i, j, value);
                system_matrix.block(SolutionBlocks::unfiltered_density,
                                    SolutionBlocks::unfiltered_density_multiplier).set(cell_i, j, value_transpose);
            }

            cell_measure = cell->measure();

            system_matrix.block(SolutionBlocks::density, SolutionBlocks::total_volume_multiplier).set(cell_i, 0,
                    cell->measure());
            system_matrix.block(SolutionBlocks::total_volume_multiplier,SolutionBlocks::density).set(0,cell_i,
                    cell->measure());
        }
    }
    system_matrix.compress(VectorOperation::insert);
    // for (const auto &cell: dof_handler.active_cell_iterators()) 
    // {
    //     if(cell->is_locally_owned())
    //     {
    //         std::vector<types::global_dof_index> i(cell->get_fe().n_dofs_per_cell());
    //         cell->get_dof_indices(i);
    //         const unsigned int cell_i = i[cell->get_fe().component_to_system_index(0, 0)];

    //         for (const unsigned int j : density_filter.find_relevant_neighbors(cell_i)) {
    //             double value = system_matrix.block(SolutionBlocks::unfiltered_density_multiplier,
    //                                 SolutionBlocks::unfiltered_density).el(j,cell_i);

    //             system_matrix.block(SolutionBlocks::unfiltered_density,
    //                                 SolutionBlocks::unfiltered_density_multiplier).set(j, cell_i, value);
    //         }
    //     }
    // }
    // system_matrix.compress(VectorOperation::insert);


    pcout << "assembled " << std::endl;

}

///For use in the filter, this calculates the objective value we are working to minimize.
template<int dim>
double
KktSystem<dim>::calculate_objective_value(const LA::MPI::BlockVector &distributed_state) const {
    /*Remove any values from old iterations*/

    locally_relevant_solution = distributed_state;


    QGauss<dim> nine_quadrature(fe_nine.degree + 1);
    QGauss<dim> ten_quadrature(fe_ten.degree + 1);

    hp::QCollection<dim> q_collection;
    q_collection.push_back(nine_quadrature);
    q_collection.push_back(ten_quadrature);

    hp::FEValues<dim> hp_fe_values(fe_collection,
                                   q_collection,
                                   update_values | update_quadrature_points |
                                   update_JxW_values | update_gradients);

    QGauss<dim - 1> common_face_quadrature(fe_ten.degree + 1);

    FEFaceValues<dim> fe_nine_face_values(fe_nine,
                                          common_face_quadrature,
                                          update_JxW_values |
                                          update_gradients | update_values);
    FEFaceValues<dim> fe_ten_face_values(fe_ten,
                                         common_face_quadrature,
                                         update_normal_vectors |
                                         update_values);

    FullMatrix<double> cell_matrix;
    Vector<double> cell_rhs;

    const FEValuesExtractors::Vector displacements(SolutionComponents::displacement<dim>);

    Tensor<1, dim> traction;
    traction[1] = -1;
    distributed_solution = distributed_state;
    double objective_value = 0;
    for (const auto &cell: dof_handler.active_cell_iterators())
    {
        if(cell->is_locally_owned())
        {
            hp_fe_values.reinit(cell);
            const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
            const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
            const unsigned int n_q_points = fe_values.n_quadrature_points;
            const unsigned int n_face_q_points = common_face_quadrature.size();

            std::vector<Tensor<1, dim>> old_displacement_values(n_q_points);
            fe_values[displacements].get_function_values(
                        locally_relevant_solution, old_displacement_values);

            for (unsigned int face_number = 0;
                 face_number < GeometryInfo<dim>::faces_per_cell;
                 ++face_number)
            {
                if (cell->face(face_number)->at_boundary() && cell->face(face_number)->boundary_id()
                        == BoundaryIds::down_force)
                {
                    for (unsigned int face_q_point = 0;
                         face_q_point < n_face_q_points; ++face_q_point) {
                        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                            if (cell->material_id() == MaterialIds::without_multiplier) {
                                fe_nine_face_values.reinit(cell, face_number);
                                objective_value += traction
                                        * fe_nine_face_values[displacements].value(i,
                                                                                   face_q_point)
                                        * fe_nine_face_values.JxW(face_q_point);
                            } else {
                                fe_ten_face_values.reinit(cell, face_number);
                                objective_value += traction
                                        * fe_ten_face_values[displacements].value(i,
                                                                                  face_q_point)
                                        * fe_ten_face_values.JxW(face_q_point);
                            }
                        }
                    }
                }
            }
        }
    }
    double objective_value_out;
    MPI_Allreduce(&objective_value, &objective_value_out, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return objective_value;
}


///As the KKT System knows which vectors correspond to the slack variables, the sum of the logs of the slacks is computed here for use in the filter.
template<int dim>
double
KktSystem<dim>::calculate_barrier_distance(const LA::MPI::BlockVector &state) const {
    double barrier_distance_log_sum = 0;
    unsigned int vect_size = state.block(SolutionBlocks::density_lower_slack).size();
    distributed_solution = state;
    for (unsigned int k = 0; k < vect_size; k++) {
        if (distributed_solution.block(SolutionBlocks::density_lower_slack).in_local_range(k))
            barrier_distance_log_sum += std::log(state.block(SolutionBlocks::density_lower_slack)[k]);
    }
    for (unsigned int k = 0; k < vect_size; k++) {
        if (distributed_solution.block(SolutionBlocks::density_upper_slack).in_local_range(k))
            barrier_distance_log_sum += std::log(state.block(SolutionBlocks::density_upper_slack)[k]);
    }
    double out_barrier_distance_log_sum;
    MPI_Allreduce(&barrier_distance_log_sum, &out_barrier_distance_log_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return out_barrier_distance_log_sum;
}

///Calculates the norm of the RHS. While not the KKT norm, we also expect this to be 0 at a minimum.
template<int dim>
double
KktSystem<dim>::calculate_rhs_norm(const LA::MPI::BlockVector &state, const double barrier_size) const {
    return calculate_rhs(state, barrier_size).l2_norm();
}


///Feasibility conditions appear on the RHS of the linear system, so I compute the RHS to find it. Could probably be combined with the objective value finding part to make it faster.
template<int dim>
double
KktSystem<dim>::calculate_feasibility(const LA::MPI::BlockVector &state, const double barrier_size) const {
    LA::MPI::BlockVector test_rhs = calculate_rhs(state, barrier_size);

    double norm = 0;

    distributed_solution = state;
    double full_prod1 =1;
    double full_prod2 = 1;

    for (unsigned int k = 0; k < state.block(SolutionBlocks::density_upper_slack).size(); k++) {
        double prod1 = 1;
        double prod2 = 1;
        if(state.block(SolutionBlocks::density_upper_slack).in_local_range(k))
        {
            prod1 = prod1 * state.block(SolutionBlocks::density_upper_slack)[k]
                    * state.block(SolutionBlocks::density_upper_slack)[k];
        }
        if(state.block(SolutionBlocks::density_lower_slack).in_local_range(k))
        {
            prod2 = prod2 *  state.block(SolutionBlocks::density_lower_slack)[k]
                    * state.block(SolutionBlocks::density_lower_slack)[k];
        }
        if(state.block(SolutionBlocks::density_upper_slack_multiplier).in_local_range(k))
        {
            prod1 = prod1 * state.block(SolutionBlocks::density_upper_slack_multiplier)[k]
                    * state.block(SolutionBlocks::density_upper_slack_multiplier)[k];
        }
        if(state.block(SolutionBlocks::density_lower_slack_multiplier).in_local_range(k))
        {
            prod2 = prod2 *  state.block(SolutionBlocks::density_lower_slack_multiplier)[k]
                    * state.block(SolutionBlocks::density_lower_slack_multiplier)[k];
        }
        MPI_Allreduce(&prod1, &full_prod1, 1, MPI_DOUBLE, MPI_PROD, MPI_COMM_WORLD);
        MPI_Allreduce(&prod2, &full_prod2, 1, MPI_DOUBLE, MPI_PROD, MPI_COMM_WORLD);
        norm = norm + full_prod1 + full_prod2;
    }

    norm += std::pow(test_rhs.block(SolutionBlocks::displacement).l2_norm(), 2);
    norm += std::pow(test_rhs.block(SolutionBlocks::density).l2_norm(), 2);
    norm += std::pow(test_rhs.block(SolutionBlocks::unfiltered_density).l2_norm(), 2);
    norm += std::pow(test_rhs.block(SolutionBlocks::displacement_multiplier).l2_norm(), 2);
    norm += std::pow(test_rhs.block(SolutionBlocks::unfiltered_density_multiplier).l2_norm(), 2);
    norm += std::pow(test_rhs.block(SolutionBlocks::total_volume_multiplier).l2_norm(), 2);
    norm += std::pow(test_rhs.block(SolutionBlocks::density_upper_slack_multiplier).l2_norm(), 2);
    norm += std::pow(test_rhs.block(SolutionBlocks::density_lower_slack_multiplier).l2_norm(), 2);

    return norm;
}

///calculates the KKT norm of the system, representing how close the program is to convergence.
template<int dim>
double
KktSystem<dim>::calculate_convergence(const LA::MPI::BlockVector &state) const {
    LA::MPI::BlockVector test_rhs = calculate_rhs(state, Input::min_barrier_size);
    double norm = 0;

    distributed_solution = state;
    double full_prod1 =1;
    double full_prod2 = 1;

    for (unsigned int k = 0; k < state.block(SolutionBlocks::density_upper_slack).size(); k++) {
        double prod1 = 1;
        double prod2 = 1;
        if(state.block(SolutionBlocks::density_upper_slack).in_local_range(k))
        {
            prod1 = prod1 * state.block(SolutionBlocks::density_upper_slack)[k]
                    * state.block(SolutionBlocks::density_upper_slack)[k];
        }
        if(state.block(SolutionBlocks::density_lower_slack).in_local_range(k))
        {
            prod2 = prod2 *  state.block(SolutionBlocks::density_lower_slack)[k]
                    * state.block(SolutionBlocks::density_lower_slack)[k];
        }
        if(state.block(SolutionBlocks::density_upper_slack_multiplier).in_local_range(k))
        {
            prod1 = prod1 * state.block(SolutionBlocks::density_upper_slack_multiplier)[k]
                    * state.block(SolutionBlocks::density_upper_slack_multiplier)[k];
        }
        if(state.block(SolutionBlocks::density_lower_slack_multiplier).in_local_range(k))
        {
            prod2 = prod2 *  state.block(SolutionBlocks::density_lower_slack_multiplier)[k]
                    * state.block(SolutionBlocks::density_lower_slack_multiplier)[k];
        }
        MPI_Allreduce(&prod1, &full_prod1, 1, MPI_DOUBLE, MPI_PROD, MPI_COMM_WORLD);
        MPI_Allreduce(&prod2, &full_prod2, 1, MPI_DOUBLE, MPI_PROD, MPI_COMM_WORLD);
        norm = norm + full_prod1 + full_prod2;
    }


    norm += std::pow(test_rhs.block(SolutionBlocks::displacement).l2_norm(), 2);
    norm += std::pow(test_rhs.block(SolutionBlocks::density).l2_norm(), 2);
    norm += std::pow(test_rhs.block(SolutionBlocks::unfiltered_density).l2_norm(), 2);
    norm += std::pow(test_rhs.block(SolutionBlocks::displacement_multiplier).l2_norm(), 2);
    norm += std::pow(test_rhs.block(SolutionBlocks::unfiltered_density_multiplier).l2_norm(), 2);
    norm += std::pow(test_rhs.block(SolutionBlocks::total_volume_multiplier).l2_norm(), 2);
    norm += std::pow(test_rhs.block(SolutionBlocks::density_upper_slack_multiplier).l2_norm(), 2);
    norm += std::pow(test_rhs.block(SolutionBlocks::density_lower_slack_multiplier).l2_norm(), 2);

    norm = std::pow(norm, .5);

    pcout << "KKT norm: " << norm << std::endl;
    return norm;
}

/// Makes the RHS of the KKT equations
template<int dim>
LA::MPI::BlockVector
KktSystem<dim>::calculate_rhs(const LA::MPI::BlockVector &distributed_state, const double barrier_size) const {
    LA::MPI::BlockVector test_rhs (system_rhs);
    LA::MPI::BlockVector state (locally_relevant_solution);
    state = distributed_state;
    test_rhs = 0.;

    QGauss<dim> nine_quadrature(fe_nine.degree + 1);
    QGauss<dim> ten_quadrature(fe_ten.degree + 1);

    hp::QCollection<dim> q_collection;
    q_collection.push_back(nine_quadrature);
    q_collection.push_back(ten_quadrature);

    hp::FEValues<dim> hp_fe_values(fe_collection,
                                   q_collection,
                                   update_values | update_quadrature_points |
                                   update_JxW_values | update_gradients);

    QGauss<dim - 1> common_face_quadrature(fe_ten.degree + 1);

    FEFaceValues<dim> fe_nine_face_values(fe_nine,
                                          common_face_quadrature,
                                          update_JxW_values |
                                          update_gradients | update_values);
    FEFaceValues<dim> fe_ten_face_values(fe_ten,
                                         common_face_quadrature,
                                         update_normal_vectors |
                                         update_values);

    FullMatrix<double> cell_matrix;
    Vector<double> cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;

    const FEValuesExtractors::Scalar densities(SolutionComponents::density<dim>);
    const FEValuesExtractors::Vector displacements(SolutionComponents::displacement<dim>);
    const FEValuesExtractors::Scalar unfiltered_densities(SolutionComponents::unfiltered_density<dim>);
    const FEValuesExtractors::Vector displacement_multipliers(SolutionComponents::displacement_multiplier<dim>);
    const FEValuesExtractors::Scalar unfiltered_density_multipliers(
                SolutionComponents::unfiltered_density_multiplier<dim>);
    const FEValuesExtractors::Scalar density_lower_slacks(SolutionComponents::density_lower_slack<dim>);
    const FEValuesExtractors::Scalar density_lower_slack_multipliers(
                SolutionComponents::density_lower_slack_multiplier<dim>);
    const FEValuesExtractors::Scalar density_upper_slacks(SolutionComponents::density_upper_slack<dim>);
    const FEValuesExtractors::Scalar density_upper_slack_multipliers(
                SolutionComponents::density_upper_slack_multiplier<dim>);
    const FEValuesExtractors::Scalar total_volume_multiplier(
                SolutionComponents::total_volume_multiplier<dim>);


    const unsigned int n_face_q_points = common_face_quadrature.size();

    const Functions::ConstantFunction<dim> lambda(1.), mu(1.);

    locally_relevant_solution = state;
    distributed_solution = state;
    LA::MPI::BlockVector filtered_unfiltered_density_solution (distributed_solution);
    LA::MPI::BlockVector filter_adjoint_unfiltered_density_multiplier_solution (distributed_solution);
    filtered_unfiltered_density_solution.block(SolutionBlocks::unfiltered_density) = 0;
    filter_adjoint_unfiltered_density_multiplier_solution.block(SolutionBlocks::unfiltered_density_multiplier) = 0;

    density_filter.filter_matrix.vmult(filtered_unfiltered_density_solution.block(SolutionBlocks::unfiltered_density),distributed_solution.block(SolutionBlocks::unfiltered_density));
    density_filter.filter_matrix_transpose.vmult(filter_adjoint_unfiltered_density_multiplier_solution.block(SolutionBlocks::unfiltered_density_multiplier),distributed_solution.block(SolutionBlocks::unfiltered_density_multiplier));

    LA::MPI::BlockVector relevant_filtered_unfiltered_density_solution (locally_relevant_solution);
    LA::MPI::BlockVector relevant_filter_adjoint_unfiltered_density_multiplier_solution (locally_relevant_solution);
    relevant_filtered_unfiltered_density_solution = filtered_unfiltered_density_solution;
    relevant_filter_adjoint_unfiltered_density_multiplier_solution = filter_adjoint_unfiltered_density_multiplier_solution;

    double old_volume_multiplier_temp = 0;
    double old_volume_multiplier;
    if(distributed_state.block(SolutionBlocks::total_volume_multiplier).in_local_range(0))
    {
        old_volume_multiplier_temp = state.block(SolutionBlocks::total_volume_multiplier)[0];
    }
    MPI_Allreduce(&old_volume_multiplier_temp, &old_volume_multiplier, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (const auto &cell: dof_handler.active_cell_iterators()) {
        if(cell->is_locally_owned())
        {
            hp_fe_values.reinit(cell);
            const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
            cell_matrix.reinit(cell->get_fe().n_dofs_per_cell(),
                               cell->get_fe().n_dofs_per_cell());
            cell_rhs.reinit(cell->get_fe().n_dofs_per_cell());

            const unsigned int n_q_points = fe_values.n_quadrature_points;

            std::vector<double> old_density_values(n_q_points);
            std::vector<Tensor<1, dim>> old_displacement_values(n_q_points);
            std::vector<double> old_displacement_divs(n_q_points);
            std::vector<SymmetricTensor<2, dim>> old_displacement_symmgrads(
                        n_q_points);
            std::vector<Tensor<1, dim>> old_displacement_multiplier_values(
                        n_q_points);
            std::vector<double> old_displacement_multiplier_divs(n_q_points);
            std::vector<SymmetricTensor<2, dim>> old_displacement_multiplier_symmgrads(
                        n_q_points);
            std::vector<double> old_lower_slack_multiplier_values(n_q_points);
            std::vector<double> old_upper_slack_multiplier_values(n_q_points);
            std::vector<double> old_lower_slack_values(n_q_points);
            std::vector<double> old_upper_slack_values(n_q_points);
            std::vector<double> old_unfiltered_density_values(n_q_points);
            std::vector<double> old_unfiltered_density_multiplier_values(n_q_points);
            std::vector<double> filtered_unfiltered_density_values(n_q_points);
            std::vector<double> filter_adjoint_unfiltered_density_multiplier_values(n_q_points);
            std::vector<double> lambda_values(n_q_points);
            std::vector<double> mu_values(n_q_points);

            const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();

            cell_matrix = 0;
            cell_rhs = 0;
            local_dof_indices.resize(cell->get_fe().n_dofs_per_cell());

            cell->get_dof_indices(local_dof_indices);

            lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
            mu.value_list(fe_values.get_quadrature_points(), mu_values);

            fe_values[densities].get_function_values(state,
                                                     old_density_values);
            fe_values[displacements].get_function_values(state,
                                                         old_displacement_values);
            fe_values[displacements].get_function_divergences(state,
                                                              old_displacement_divs);
            fe_values[displacements].get_function_symmetric_gradients(
                        state, old_displacement_symmgrads);
            fe_values[displacement_multipliers].get_function_values(
                        state, old_displacement_multiplier_values);
            fe_values[displacement_multipliers].get_function_divergences(
                        state, old_displacement_multiplier_divs);
            fe_values[displacement_multipliers].get_function_symmetric_gradients(
                        state, old_displacement_multiplier_symmgrads);
            fe_values[density_lower_slacks].get_function_values(
                        state, old_lower_slack_values);
            fe_values[density_lower_slack_multipliers].get_function_values(
                        state, old_lower_slack_multiplier_values);
            fe_values[density_upper_slacks].get_function_values(
                        state, old_upper_slack_values);
            fe_values[density_upper_slack_multipliers].get_function_values(
                        state, old_upper_slack_multiplier_values);
            fe_values[unfiltered_densities].get_function_values(
                        state, old_unfiltered_density_values);
            fe_values[unfiltered_density_multipliers].get_function_values(
                        state, old_unfiltered_density_multiplier_values);
            fe_values[unfiltered_densities].get_function_values(
                        relevant_filtered_unfiltered_density_solution, filtered_unfiltered_density_values);
            fe_values[unfiltered_density_multipliers].get_function_values(
                        relevant_filter_adjoint_unfiltered_density_multiplier_solution,
                        filter_adjoint_unfiltered_density_multiplier_values);


            Tensor<1, dim> traction;
            traction[1] = -1;

            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    const SymmetricTensor<2, dim> displacement_phi_i_symmgrad =
                            fe_values[displacements].symmetric_gradient(i, q_point);
                    const double displacement_phi_i_div =
                            fe_values[displacements].divergence(i, q_point);

                    const SymmetricTensor<2, dim> displacement_multiplier_phi_i_symmgrad =
                            fe_values[displacement_multipliers].symmetric_gradient(i,
                                                                                   q_point);
                    const double displacement_multiplier_phi_i_div =
                            fe_values[displacement_multipliers].divergence(i,
                                                                           q_point);


                    const double density_phi_i = fe_values[densities].value(i,
                                                                            q_point);
                    const double unfiltered_density_phi_i = fe_values[unfiltered_densities].value(i,
                                                                                                  q_point);
                    const double unfiltered_density_multiplier_phi_i = fe_values[unfiltered_density_multipliers].value(
                                i, q_point);

                    const double lower_slack_multiplier_phi_i =
                            fe_values[density_lower_slack_multipliers].value(i,
                                                                             q_point);

                    const double lower_slack_phi_i =
                            fe_values[density_lower_slacks].value(i, q_point);

                    const double upper_slack_phi_i =
                            fe_values[density_upper_slacks].value(i, q_point);

                    const double upper_slack_multiplier_phi_i =
                            fe_values[density_upper_slack_multipliers].value(i,
                                                                             q_point);

                    //rhs eqn 0
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                -1 * density_penalty_exponent *
                                std::pow(old_density_values[q_point], density_penalty_exponent - 1) * density_phi_i
                                * (old_displacement_multiplier_divs[q_point] * old_displacement_divs[q_point]
                                   * lambda_values[q_point]
                                   + 2 * mu_values[q_point] * (old_displacement_symmgrads[q_point]
                                                               * old_displacement_multiplier_symmgrads[q_point]))
                                - density_phi_i * old_unfiltered_density_multiplier_values[q_point]
                                + old_volume_multiplier * density_phi_i
                                );

                    //rhs eqn 1 - boundary terms counted later
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point)
                            * (
                                -1 * std::pow(old_density_values[q_point], density_penalty_exponent)
                                * (
                                    old_displacement_multiplier_divs[q_point] * displacement_phi_i_div
                                    * lambda_values[q_point]
                                    + 2 * mu_values[q_point] * (old_displacement_multiplier_symmgrads[q_point]
                                                               * displacement_phi_i_symmgrad)
                                   )
                                );

                    //rhs eqn 2
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                unfiltered_density_phi_i *
                                filter_adjoint_unfiltered_density_multiplier_values[q_point]
                                + unfiltered_density_phi_i * old_upper_slack_multiplier_values[q_point]
                                + -1 * unfiltered_density_phi_i * old_lower_slack_multiplier_values[q_point]
                                );

                    //rhs eqn 3 - boundary terms counted later
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                -1 * std::pow(old_density_values[q_point], density_penalty_exponent)
                                * (old_displacement_divs[q_point] * displacement_multiplier_phi_i_div
                                   * lambda_values[q_point]
                                   + 2 * mu_values[q_point] * (displacement_multiplier_phi_i_symmgrad
                                                               * old_displacement_symmgrads[q_point]))
                                );

                    //rhs eqn 4
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) *
                            (-1 * lower_slack_multiplier_phi_i
                             * (old_unfiltered_density_values[q_point] - old_lower_slack_values[q_point])
                             );

                    //rhs eqn 5
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                -1 * upper_slack_multiplier_phi_i
                                * (1 - old_unfiltered_density_values[q_point]
                                   - old_upper_slack_values[q_point]));

                    //rhs eqn 6
                    if (std::abs(old_density_values[q_point] - filtered_unfiltered_density_values[q_point])>1e-12)
                    {
                        cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                -1 * unfiltered_density_multiplier_phi_i
                                * (old_density_values[q_point] - filtered_unfiltered_density_values[q_point])
                                );
                    }


                    //rhs eqn 7
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) *
                            (lower_slack_phi_i *
                             (old_lower_slack_multiplier_values[q_point] -
                              barrier_size / old_lower_slack_values[q_point]));

                    //rhs eqn 8
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) *
                            (upper_slack_phi_i *
                             (old_upper_slack_multiplier_values[q_point] -
                              barrier_size / old_upper_slack_values[q_point]));

                }

            }

            for (unsigned int face_number = 0;
                 face_number < GeometryInfo<dim>::faces_per_cell;
                 ++face_number) {
                if (cell->face(face_number)->at_boundary() && cell->face(
                            face_number)->boundary_id() == BoundaryIds::down_force) {
                    for (unsigned int face_q_point = 0;
                         face_q_point < n_face_q_points; ++face_q_point) {
                        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                            if (cell->material_id() == MaterialIds::without_multiplier) {
                                fe_nine_face_values.reinit(cell, face_number);
                                cell_rhs(i) += -1
                                        * traction
                                        * fe_nine_face_values[displacements].value(i,
                                                                                   face_q_point)
                                        * fe_nine_face_values.JxW(face_q_point);

                                cell_rhs(i) += -1 * traction
                                        * fe_nine_face_values[displacement_multipliers].value(
                                            i, face_q_point)
                                        * fe_nine_face_values.JxW(face_q_point);
                            } else {
                                fe_ten_face_values.reinit(cell, face_number);
                                cell_rhs(i) += -1
                                        * traction
                                        * fe_ten_face_values[displacements].value(i,
                                                                                  face_q_point)
                                        * fe_ten_face_values.JxW(face_q_point);

                                cell_rhs(i) += -1 * traction
                                        * fe_ten_face_values[displacement_multipliers].value(
                                            i, face_q_point)
                                        * fe_ten_face_values.JxW(face_q_point);
                            }
                        }
                    }
                }
            }


            MatrixTools::local_apply_boundary_values(boundary_values, local_dof_indices,
                                                     cell_matrix, cell_rhs, true);
            constraints.distribute_local_to_global(
                        cell_rhs, local_dof_indices, test_rhs);
        }
    }

    test_rhs.compress(VectorOperation::add);
    double total_volume_temp = 0;
    double goal_volume_temp = 0;
    double total_volume, goal_volume;

    distributed_solution = state;
    for (const auto &cell: dof_handler.active_cell_iterators()) {
        if(cell->is_locally_owned())
        {
            std::vector<unsigned int> i(cell->get_fe().n_dofs_per_cell());
            cell->get_dof_indices(i);
            if (distributed_solution.block(SolutionBlocks::density).in_local_range(i[cell->get_fe().component_to_system_index(0, 0)]))
            {
                total_volume_temp += cell->measure() * state.block(SolutionBlocks::density)[i[cell->get_fe().component_to_system_index(0, 0)]];
                goal_volume_temp += cell->measure() * Input::volume_percentage;
            }
        }
    }

    MPI_Allreduce(&total_volume_temp, &total_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&goal_volume_temp, &goal_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


    if (test_rhs.block(SolutionBlocks::total_volume_multiplier).in_local_range(0))
    {
        test_rhs.block(SolutionBlocks::total_volume_multiplier)[0] = goal_volume - total_volume;
    }
    test_rhs.compress(VectorOperation::insert);

    return test_rhs;

}

///Solves the big system to get the newton step
template<int dim>
LA::MPI::BlockVector
KktSystem<dim>::solve(const LA::MPI::BlockVector &state) {
    double gmres_tolerance;
    if (Input::use_eisenstat_walker) {
        gmres_tolerance = std::max(
                    std::min(
                        .1 * system_rhs.l2_norm() / (initial_rhs_error),
                        .001
                        ),
                    Input::default_gmres_tolerance);
    }
    else {
        gmres_tolerance = Input::default_gmres_tolerance*system_rhs.l2_norm();
    }


    locally_relevant_solution=state;
    distributed_solution = state;

    SolverControl solver_control(10000, gmres_tolerance);

    // ************ BEGIN MAKING MF GMG ELASTICITY PRECONDITIONER ***************************
    using SystemMFMatrixType = MF_Elasticity_Operator<dim, 1, double>;
    using LevelMFMatrixType = MF_Elasticity_Operator<dim, 1, double>;

    elasticity_matrix_mf.clear();
    mg_matrices.clear_elements();

    std::map< types::global_dof_index, Point< dim > > support_points;
    std::map< types::global_dof_index, Point< dim > > support_points_displacement;

    MappingQGeneric<dim,dim> generic_map_displacement(1);
    MappingQGeneric<dim,dim> generic_map_1(1);
    MappingQGeneric<dim,dim> generic_map_2(1);

    hp::MappingCollection< dim, dim > hp_generic_map;

    hp_generic_map.push_back(generic_map_1);
    hp_generic_map.push_back(generic_map_2);

    DoFTools::map_dofs_to_support_points(generic_map_displacement, dof_handler_displacement, support_points_displacement);
    DoFTools::map_dofs_to_support_points(hp_generic_map, dof_handler, support_points);


    const types::global_dof_index disp_mult_start_index = system_matrix.get_row_indices().block_start(SolutionBlocks::displacement_multiplier);

    for (const auto &support_points_displacement_pair : support_points_displacement)
    {
        if (support_points_displacement_pair.second != support_points[support_points_displacement_pair.first+disp_mult_start_index])
            pcout << "d = " << support_points_displacement_pair.first << ", points are " << support_points_displacement_pair.second << " and " << support_points[support_points_displacement_pair.first+disp_mult_start_index] << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<IndexSet> locally_owned_dofs = Utilities::MPI::all_gather(mpi_communicator, dof_handler_displacement.locally_owned_dofs());
    IndexSet locally_active_dofs;
    DoFTools::extract_locally_active_dofs(dof_handler_displacement, locally_active_dofs);
    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler_displacement, locally_relevant_dofs);
   // AffineConstraints<double> temp_displacement_constraints;
    
    

    displacement_constraints.clear();
    displacement_constraints.reinit(locally_relevant_dofs); //FIXME SHOULD THIS BE RELEVANT???
    displacement_constraints.copy_from(mg_level_constraints[triangulation.n_global_levels()-1]);
    std::cout << "displacement constraint number: " << displacement_constraints.n_constraints() <<std::endl;
    displacement_constraints.close();
    {
        typename MatrixFree<dim, double>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme =
                MatrixFree<dim, double>::AdditionalData::none;
        additional_data.mapping_update_flags =
                (update_gradients | update_JxW_values | update_quadrature_points);
        std::shared_ptr<MatrixFree<dim, double>> system_mf_storage(
                    new MatrixFree<dim, double>());
        system_mf_storage->reinit(generic_map_1,
                                  dof_handler_displacement,
                                  displacement_constraints,
                                  QGauss<1>(fe_displacement.degree + 1),
                                  additional_data);
        elasticity_matrix_mf.initialize(system_mf_storage);
    }


    LinearAlgebra::distributed::Vector<double> distributed_displacement_sol;
    LinearAlgebra::distributed::Vector<double> distributed_displacement_rhs;

    elasticity_matrix_mf.initialize_dof_vector(distributed_displacement_sol);
    elasticity_matrix_mf.initialize_dof_vector(distributed_displacement_rhs);

    ChangeVectorTypes::copy_from_system_to_displacement_vector<double>(distributed_displacement_sol,distributed_solution.block(SolutionBlocks::displacement),displacement_to_system_dof_index_map);

    ChangeVectorTypes::copy_from_system_to_displacement_vector<double>(distributed_displacement_rhs,system_rhs.block(SolutionBlocks::displacement),displacement_to_system_dof_index_map);

    const unsigned int n_levels = triangulation.n_global_levels();
    mg_matrices.resize(0, n_levels - 1);

    mg_constrained_dofs.clear();
    mg_constrained_dofs.initialize(dof_handler_displacement);
    const std::set<types::boundary_id> empty_boundary_set;
    // mg_constrained_dofs.make_zero_boundary_constraints(dof_handler_displacement,empty_boundary_set);


    for (unsigned int level = 0; level < n_levels; ++level)
    {
        mg_constrained_dofs.add_user_constraints(level, mg_level_constraints[level]);
    }

    // for (unsigned int level = 0; level < n_levels; ++level)
    // {
    //     mg_level_constraints[level].print(std::cout);
    // }

    for (unsigned int level = 0; level < n_levels; ++level)
    {
        IndexSet relevant_dofs;
        DoFTools::extract_locally_relevant_level_dofs(dof_handler_displacement,
                                                      level,
                                                      relevant_dofs);

        typename MatrixFree<dim, double>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme =
                MatrixFree<dim, double>::AdditionalData::none;
        additional_data.mapping_update_flags =
                (update_gradients | update_JxW_values | update_quadrature_points);
        additional_data.mg_level = level;

        std::shared_ptr<MatrixFree<dim, double>> mg_mf_storage_level(
                    new MatrixFree<dim, double>());
        mg_mf_storage_level->reinit(generic_map_1,
                                    dof_handler_displacement,
                                    mg_level_constraints[level],
                                    QGauss<1>(fe_displacement.degree + 1),
                                    additional_data);

        mg_matrices[level].clear();
        mg_matrices[level].initialize(mg_mf_storage_level,
                                      mg_constrained_dofs,
                                      level);
    }


    //+++++++++++++++++++++++++EVALUATE MATRIX LEVEL DENSITIES HERE +++++++++++++++++++++++++++++++++++++


    dof_handler_density.distribute_dofs(fe_density);

    DoFRenumbering::component_wise(dof_handler_density);
    DoFRenumbering::hierarchical(dof_handler_density);

    dof_handler_density.distribute_mg_dofs();

    active_density_vector.reinit(dof_handler_density.locally_owned_dofs(),triangulation.get_communicator());

    ChangeVectorTypes::copy(active_density_vector,distributed_solution.block(SolutionBlocks::density));


    const unsigned int n_cells = elasticity_matrix_mf.get_matrix_free()->n_cell_batches();
    // {
        

    //     QGauss<dim> nine_quadrature(2);
    //     QGauss<dim> ten_quadrature(2);

    //     hp::QCollection<dim> q_collection;
    //     q_collection.push_back(nine_quadrature);
    //     q_collection.push_back(ten_quadrature);

    //     hp::FEValues<dim> hp_fe_values(fe_collection,
    //                                    q_collection,
    //                                    update_values | update_quadrature_points |
    //                                    update_JxW_values | update_gradients);



    //     for (const auto &cell : dof_handler.active_cell_iterators())
    //         if (cell->is_locally_owned())
    //         {

    //             hp_fe_values.reinit(cell);
    //             const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();

    //             const unsigned int dofs_per_cell = fe_values.dofs_per_cell;
    //             const unsigned int n_q_points = fe_values.n_quadrature_points;

    //             std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    //             cell->get_dof_indices (local_dof_indices);
    //             Vector<double> cell_vector (dofs_per_cell);
    //             Vector<double> local_projection (dofs_per_cell);
    //             FullMatrix<double> local_mass_matrix (dofs_per_cell, dofs_per_cell);

    //             std::vector<double> rhs_values(n_q_points);
    //             std::vector<double> old_density_values(n_q_points);

    //             const FEValuesExtractors::Scalar densities(SolutionComponents::density<dim>);
    //             fe_values[densities].get_function_values(locally_relevant_solution, old_density_values);
    //             double cell_density = old_density_values[0];

    //             for (unsigned int i=0; i<rhs_values.size(); ++i)
    //             {
    //                 rhs_values[i] = cell_density;
    //             }

    //             local_projection = cell_density;

    //             std::vector<types::global_dof_index> i(cell->get_fe().n_dofs_per_cell());
    //             cell->get_dof_indices(i);
    //             const unsigned int i_val = i[cell->get_fe().component_to_system_index(0, 0)];
    //             active_density_vector[i_val] = cell_density;


    //         }

    //     // active_density_vector.compress(VectorOperation::insert);
    // }
    // MAKE ACTIVE_CELL_DATA
    std::vector<types::global_dof_index> local_dof_indices(fe_density.dofs_per_cell);
    active_cell_data.density.reinit(TableIndices<2>(n_cells, 1));
    for (unsigned int cell=0; cell<n_cells; ++cell)
    {
        const unsigned int n_components_filled = elasticity_matrix_mf.get_matrix_free()->n_active_entries_per_cell_batch(cell);

        for (unsigned int i=0; i<n_components_filled; ++i)
        {
            typename DoFHandler<dim>::active_cell_iterator FEQ_cell =elasticity_matrix_mf.get_matrix_free()->get_cell_iterator(cell,i);
            typename DoFHandler<dim>::active_cell_iterator DG_cell(&(triangulation),
                                                                   FEQ_cell->level(),
                                                                   FEQ_cell->index(),
                                                                   &dof_handler_density);

            DG_cell->get_active_or_mg_dof_indices(local_dof_indices);

            active_cell_data.density(cell, 0)[i] = active_density_vector(local_dof_indices[0]);
        }
    }
    
    elasticity_matrix_mf.set_cell_data(active_cell_data);

    //MAKE LEVEL DENSITY VECTOR

    level_cell_data.resize(0,n_levels-1);
    level_density_vector = 0.;
    level_density_vector.resize(0,n_levels-1);

    transfer.build(dof_handler_density);

    transfer.interpolate_to_mg(dof_handler_density,
                               level_density_vector,
                               active_density_vector);

    // MAKE LEVEL_CELL_DATA
    for (unsigned int level=0; level<n_levels; ++level)
    {
        const unsigned int n_cells = mg_matrices[level].get_matrix_free()->n_cell_batches();

        level_cell_data[level].density.reinit(TableIndices<2>(n_cells, 1));
        for (unsigned int cell=0; cell<n_cells; ++cell)
        {
            const unsigned int n_components_filled = mg_matrices[level].get_matrix_free()->n_active_entries_per_cell_batch(cell);
            for (unsigned int i=0; i<n_components_filled; ++i)
            {

                typename DoFHandler<dim>::level_cell_iterator FEQ_cell = mg_matrices[level].get_matrix_free()->get_cell_iterator(cell,i);
                typename DoFHandler<dim>::level_cell_iterator DG_cell(&(triangulation),
                                                                      FEQ_cell->level(),
                                                                      FEQ_cell->index(),
                                                                      &dof_handler_density);
                DG_cell->get_active_or_mg_dof_indices(local_dof_indices);


                level_cell_data[level].density(cell, 0)[i] = level_density_vector[level](local_dof_indices[0]);
            }

        }

        // Store density tables and other data into the multigrid level matrix-free objects.

        mg_matrices[level].set_cell_data (level_cell_data[level]);

    }




    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    mg_transfer.initialize_constraints(mg_constrained_dofs);
    mg_transfer.build(dof_handler_displacement);

    smoother_data.resize(0, triangulation.n_global_levels() - 1);

    for (unsigned int level = 0; level < triangulation.n_global_levels();
         ++level)
    {
        if (level > 0)
        {
            smoother_data[level].smoothing_range     = 15.;
            smoother_data[level].degree              = 10;
            smoother_data[level].eig_cg_n_iterations = 10;
        }
        else
        {
            smoother_data[0].smoothing_range = 1e-3;
            smoother_data[0].degree          = numbers::invalid_unsigned_int;
            smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
        }
        mg_matrices[level].compute_diagonal();
        smoother_data[level].preconditioner =
                mg_matrices[level].get_matrix_diagonal_inverse();
    }

    mg_smoother.initialize(mg_matrices, smoother_data);

    mg_coarse.initialize(mg_smoother);
    mg::Matrix<LinearAlgebra::distributed::Vector<double>> mg_matrix(mg_matrices);

    mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels();
         ++level)
    {
        mg_interface_matrices[level].initialize(mg_matrices[level]);
    }
    mg::Matrix<LinearAlgebra::distributed::Vector<double>> mg_interface(mg_interface_matrices);

    Multigrid<LinearAlgebra::distributed::Vector<double>> mg(
                mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
    mg.set_edge_matrices(mg_interface, mg_interface);
    mg.set_cycle(Multigrid<LinearAlgebra::distributed::Vector<double>>::v_cycle);
    mg.set_minlevel(0);
    PreconditionMG<dim,LinearAlgebra::distributed::Vector<double>,MGTransferMatrixFree<dim, double>>
            mf_gmg_preconditioner(dof_handler_displacement, mg, mg_transfer);


// *************TEST SOLVE*************************
// time the solve

//    output(distributed_solution, 66);

//    TimerOutput t(pcout, TimerOutput::never, TimerOutput::wall_times);
//    {
//    TimerOutput::Scope t_scope(t, "Solve_mfgmg");
//    elasticity_matrix_mf.initialize_dof_vector(distributed_displacement_sol);
//    elasticity_matrix_mf.initialize_dof_vector(distributed_displacement_rhs);

//    locally_relevant_solution = system_rhs;
//    ChangeVectorTypes::copy_from_system_to_displacement_vector<double>(distributed_displacement_rhs,locally_relevant_solution.block(SolutionBlocks::displacement),displacement_to_system_dof_index_map);

//     pcout << "real rhs norm: " << system_rhs.block(SolutionBlocks::displacement).l2_norm() << std::endl;

//    locally_relevant_solution = distributed_solution;
//    ChangeVectorTypes::copy_from_system_to_displacement_vector<double>(distributed_displacement_sol,locally_relevant_solution.block(SolutionBlocks::displacement),displacement_to_system_dof_index_map);
   
//    SolverControl test_solver_control_1(20, 1e-6);
//    SolverCG<LinearAlgebra::distributed::Vector<double>> CG_Solve_1(test_solver_control_1);

//    pcout << "pre norm: " << distributed_displacement_rhs.l2_norm() << std::endl;

//    try
//    {
//         // mf_gmg_preconditioner.vmult(distributed_displacement_sol, distributed_displacement_rhs);
// //        CG_Solve.solve(elasticity_matrix_mf, distributed_displacement_sol, -1* distributed_displacement_rhs,    );
//        CG_Solve_1.solve(elasticity_matrix_mf, distributed_displacement_sol, -1* distributed_displacement_rhs, mf_gmg_preconditioner);
//    }
//    catch(std::exception &exc)
//    {
//        pcout << "mfgmg diff: " << solver_control.initial_value()/solver_control.last_value() << std::endl;
//    }

//    pcout << "mfgmg solved in " << test_solver_control_1.last_step() <<  " steps" << std::endl;
// //    try
// //    {
// // //        CG_Solve.solve(elasticity_matrix_mf, distributed_displacement_sol, -1* distributed_displacement_rhs,    PreconditionIdentity());
// //        CG_Solve_2.solve(system_matrix.block(SolutionBlocks::displacement,SolutionBlocks::displacement_multiplier), distributed_solution.block(SolutionBlocks::displacement_multiplier), system_rhs.block(SolutionBlocks::displacement), PreconditionIdentity()  );
// //    }
// //    catch(std::exception &exc)
// //    {
// //        std::cout << "solve failed in " << test_solver_control_2.last_step() <<  " steps" << std::endl;
// //        throw;
// //    }

// //    std::cout << "solved in " << test_solver_control_2.last_step() <<  " steps" << std::endl;


//    }
//    ChangeVectorTypes::copy_from_displacement_to_system_vector<double>(distributed_solution.block(SolutionBlocks::displacement), distributed_displacement_sol,displacement_to_system_dof_index_map);
//    displacement_constraints.distribute(distributed_solution.block(SolutionBlocks::displacement));

//     pcout << distributed_displacement_sol.linfty_norm() << "+++++++++++++" << std::endl;

//     int a = Utilities::MPI::n_mpi_processes(mpi_communicator);

//     ChangeVectorTypes::copy(distributed_solution.block(SolutionBlocks::density),active_density_vector);



//     TrilinosWrappers::PreconditionAMG amg_pre;
//     amg_pre.initialize(system_matrix.block(SolutionBlocks::displacement,SolutionBlocks::displacement_multiplier));
//    {
//     TimerOutput::Scope t_scope(t, "Solve_AMG");
    
//     SolverControl test_solver_control_2(50000, 1e-6);
//     SolverCG<LA::MPI::Vector> CG_Solve_2(test_solver_control_2);

//     distributed_solution.block(SolutionBlocks::displacement_multiplier) = 0.;

//      try
//    {
//        CG_Solve_2.solve(system_matrix.block(SolutionBlocks::displacement,SolutionBlocks::displacement_multiplier), distributed_solution.block(SolutionBlocks::displacement_multiplier), system_rhs.block(SolutionBlocks::displacement), amg_pre);
//    }
//    catch(std::exception &exc)
//    {
//        std::cout << "solve failed in " << test_solver_control_2.last_step() <<  " steps" << std::endl;
//        throw;
//    }

//    std::cout << "amg solved in " << test_solver_control_2.last_step() <<  " steps" << std::endl;

//    }
//     distributed_solution.block(SolutionBlocks::displacement_multiplier)=0;
//    {
//     TimerOutput::Scope t_scope(t, "Solve_CG");
//     SolverControl test_solver_control_3(50000, 1e-6);
//     SolverCG<LA::MPI::Vector> CG_Solve_3(test_solver_control_3);
//      try
//    {
//        CG_Solve_3.solve(system_matrix.block(SolutionBlocks::displacement,SolutionBlocks::displacement_multiplier), distributed_solution.block(SolutionBlocks::displacement), system_rhs.block(SolutionBlocks::displacement), PreconditionIdentity());
//    }
//    catch(std::exception &exc)
//    {
//        std::cout << "solve failed in " << test_solver_control_3.last_step() <<  " steps" << std::endl;
//        throw;
//    }

//     std::cout << "CG solved in " << test_solver_control_3.last_step() <<  " steps" << std::endl;

//    }


//    t.print_summary();
//    MPI_Abort(mpi_communicator, 1);

//    ***************END TEST SOLVE*************************



    TopOptSchurPreconditioner<dim> preconditioner(system_matrix, dof_handler, elasticity_matrix_mf, mf_gmg_preconditioner, displacement_to_system_dof_index_map);
    // pcout << "about to solve" << std::endl;
    // preconditioner.initialize(system_matrix, boundary_values, dof_handler, distributed_solution);
    // FullMatrix<double> out;
    // out.reinit(system_matrix.m(),system_matrix.n());
    // LA::MPI::BlockVector e_j (system_rhs);
    // LA::MPI::BlockVector r_j (system_rhs);
    // LA::MPI::BlockVector r2_j (system_rhs);
    // for (unsigned int j=0; j<out.n(); ++j)
    // {
    //     e_j = 0.;
    //     e_j(j) = 1;
    //     system_matrix.vmult(system_rhs,e_j);
    //     preconditioner.vmult(r2_j,system_rhs);

    //     for (unsigned int i=0; i<out.m(); ++i)
    //         out(i,j) = r2_j(i);
    // }

    // const unsigned int n = out.n();
    // const unsigned int m = out.m();
    // std::ofstream Xmat("preconditioned_mat.csv");
    // for (unsigned int i = 0; i < m; i++)
    // {
    //     Xmat << out(i, 0);
    //     for (unsigned int j = 1; j < n; j++)
    //     {
    //         Xmat << "," << out(i, j);
    //     }
    //     Xmat << "\n";
    // }
    // Xmat.close();
    

    

    switch (Input::solver_choice)
    {

        // case SolverOptions::inexact_K_with_exact_A_gmres: {


        //     preconditioner.initialize(system_matrix, boundary_values, dof_handler, distributed_solution);
        //     pcout << "preconditioner initialized" << std::endl;
        //     SolverFGMRES<LA::MPI::BlockVector> B_fgmres(solver_control);
        //     B_fgmres.solve(system_matrix, distributed_solution, system_rhs, preconditioner);
        //     pcout << solver_control.last_step() << " steps to solve with GMRES" << std::endl;
        //     break;
        // }
        case SolverOptions::inexact_K_with_inexact_A_gmres: {
            pcout << "size of rhs block 0 :  " << system_rhs.block(0).l1_norm()<< std::endl;
            pcout << "size of rhs block 1 :  " << system_rhs.block(1).l1_norm()<< std::endl;
            pcout << "size of rhs block 2 :  " << system_rhs.block(2).l1_norm()<< std::endl;
            pcout << "size of rhs block 3 :  " << system_rhs.block(3).l1_norm()<< std::endl;
            pcout << "size of rhs block 4 :  " << system_rhs.block(4).l1_norm()<< std::endl;
            pcout << "size of rhs block 5 :  " << system_rhs.block(5).l1_norm()<< std::endl;
            pcout << "size of rhs block 6 :  " << system_rhs.block(6).l1_norm()<< std::endl;
            pcout << "size of rhs block 7 :  " << system_rhs.block(7).l1_norm()<< std::endl;
            pcout << "size of rhs block 8 :  " << system_rhs.block(8).l1_norm()<< std::endl;
            pcout << "size of rhs block 9 :  " << system_rhs.block(9).l1_norm()<< std::endl;

            preconditioner.initialize(system_matrix, boundary_values, dof_handler, distributed_solution);
            pcout << "preconditioner initialized" << std::endl;
            distributed_solution = 0.;
            SolverFGMRES<LA::MPI::BlockVector> C_fgmres(solver_control);
            C_fgmres.solve(system_matrix, distributed_solution, system_rhs, preconditioner);
            pcout << solver_control.last_step() << " steps to solve with FGMRES" << std::endl;
            break;
        }
        default:
            throw;
        
    }

    constraints.distribute(distributed_solution);
    pcout << "size of distributed solution block 0 :  " << distributed_solution.block(0).l1_norm()<< std::endl;
    pcout << "size of distributed solution block 1 :  " << distributed_solution.block(1).l1_norm()<< std::endl;
    pcout << "size of distributed solution block 2 :  " << distributed_solution.block(2).l1_norm()<< std::endl;
    pcout << "size of distributed solution block 3 :  " << distributed_solution.block(3).l1_norm()<< std::endl;
    pcout << "size of distributed solution block 4 :  " << distributed_solution.block(4).l1_norm()<< std::endl;
    pcout << "size of distributed solution block 5 :  " << distributed_solution.block(5).l1_norm()<< std::endl;
    pcout << "size of distributed solution block 6 :  " << distributed_solution.block(6).l1_norm()<< std::endl;
    pcout << "size of distributed solution block 7 :  " << distributed_solution.block(7).l1_norm()<< std::endl;
    pcout << "size of distributed solution block 8 :  " << distributed_solution.block(8).l1_norm()<< std::endl;
    pcout << "size of distributed solution block 9 :  " << distributed_solution.block(9).l1_norm()<< std::endl;
    output(distributed_solution,100);
    return distributed_solution;
}

///Calculates and stores the first RHS norm for comparison with future RHS norm values
template<int dim>
void
KktSystem<dim>::calculate_initial_rhs_error() {
    initial_rhs_error = system_rhs.l2_norm();
}

///Creates an initial state vector used as an initial guess for the nonlinear solver.
template<int dim>
LA::MPI::BlockVector
KktSystem<dim>::get_initial_state() {

    std::vector<unsigned int> block_component(10, 2);
    block_component[SolutionBlocks::density] = 0;
    block_component[SolutionBlocks::displacement] = 1;
    const std::vector<types::global_dof_index> dofs_per_block =
            DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    const unsigned int n_p = dofs_per_block[0];
    const unsigned int n_u = dofs_per_block[1];
    const std::vector<unsigned int> block_sizes = {n_p, n_p, n_p, n_p, n_p, n_u, n_u, n_p, n_p, 1};

    LA::MPI::BlockVector state(owned_partitioning, mpi_communicator);
    {
        using namespace SolutionBlocks;
        state.block(density).add(density_ratio);
        state.block(unfiltered_density).add(density_ratio);
        state.block(unfiltered_density_multiplier)
                .add(density_ratio);
        state.block(density_lower_slack).add(density_ratio);
        state.block(density_lower_slack_multiplier).add(50);
        state.block(density_upper_slack).add(1 - density_ratio);
        state.block(density_upper_slack_multiplier).add(50);
        state.block(total_volume_multiplier).add(1);
        state.block(displacement).add(0);
        state.block(displacement_multiplier).add(0);
        // state.compress(VectorOperation::add);

        // RANDOM PART HERE
        // for(unsigned int k = 0; k<n_p; ++k)
        // {
        //     // std::rand(001);
        //     // assign random values to the density
        //     double r = std::rand()/double(RAND_MAX);
        //     state.block(density)[k] = r;
        //     state.block(unfiltered_density)[k] = r;
        // }

    }
    state.compress(VectorOperation::add);
    return state;
}

///Outputs the current state to a vtk file
template<int dim>
void
KktSystem<dim>::output(const LA::MPI::BlockVector &state, const unsigned int j) const {
    locally_relevant_solution = state;
    std::vector<std::string> solution_names(1, "low_slack_multiplier");
    std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(
                1, DataComponentInterpretation::component_is_scalar);
    solution_names.emplace_back("upper_slack_multiplier");
    data_component_interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
    solution_names.emplace_back("low_slack");
    data_component_interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
    solution_names.emplace_back("upper_slack");
    data_component_interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
    solution_names.emplace_back("unfiltered_density");
    data_component_interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
    for (unsigned int i = 0; i < dim; i++) {
        solution_names.emplace_back("displacement");
        data_component_interpretation.push_back(
                    DataComponentInterpretation::component_is_part_of_vector);
    }
    for (unsigned int i = 0; i < dim; i++) {
        solution_names.emplace_back("displacement_multiplier");
        data_component_interpretation.push_back(
                    DataComponentInterpretation::component_is_part_of_vector);
    }
    solution_names.emplace_back("density_multiplier");
    data_component_interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
    solution_names.emplace_back("density");
    data_component_interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
    solution_names.emplace_back("volume_multiplier");
    data_component_interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.build_patches();
    std::string output("solution" + std::to_string(j) + ".vtu");
    data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);

}



///Outputs to a 3d-printable file. Not yet usable in parallel.
template<>
void
KktSystem<2>::output_stl(const LA::MPI::BlockVector &state) {
    double height = .25;
    const int dim = 2;
    std::ofstream stlfile;
    stlfile.open("bridge.stl");
    stlfile << "solid bridge\n" << std::scientific;

    for (const auto &cell: dof_handler.active_cell_iterators()) {
        if (state.block(
                    SolutionBlocks::density)[cell->get_fe().component_to_system_index(0, 0)] > 0.5) {
            const Tensor<1, dim> edge_directions[2] = {cell->vertex(1) -
                                                       cell->vertex(0),
                                                       cell->vertex(2) -
                                                       cell->vertex(0)};
            const Tensor<2, dim> edge_tensor(
                        {{edge_directions[0][0], edge_directions[0][1]},
                         {edge_directions[1][0], edge_directions[1][1]}});
            const bool is_right_handed_cell = (determinant(edge_tensor) > 0);
            if (is_right_handed_cell) {
                /* Write one side at z = 0. */
                stlfile << "   facet normal " << 0.000000e+00 << " "
                            << 0.000000e+00 << " " << -1.000000e+00 << "\n";
                stlfile << "      outer loop\n";
                stlfile << "         vertex " << cell->vertex(0)[0] << " "
                            << cell->vertex(0)[1] << " " << 0.000000e+00 << "\n";
                stlfile << "         vertex " << cell->vertex(2)[0] << " "
                            << cell->vertex(2)[1] << " " << 0.000000e+00 << "\n";
                stlfile << "         vertex " << cell->vertex(1)[0] << " "
                            << cell->vertex(1)[1] << " " << 0.000000e+00 << "\n";
                stlfile << "      endloop\n";
                stlfile << "   endfacet\n";
                stlfile << "   facet normal " << 0.000000e+00 << " "
                            << 0.000000e+00 << " " << -1.000000e+00 << "\n";
                stlfile << "      outer loop\n";
                stlfile << "         vertex " << cell->vertex(1)[0] << " "
                            << cell->vertex(1)[1] << " " << 0.000000e+00 << "\n";
                stlfile << "         vertex " << cell->vertex(2)[0] << " "
                            << cell->vertex(2)[1] << " " << 0.000000e+00 << "\n";
                stlfile << "         vertex " << cell->vertex(3)[0] << " "
                            << cell->vertex(3)[1] << " " << 0.000000e+00 << "\n";
                stlfile << "      endloop\n";
                stlfile << "   endfacet\n";
                /* Write one side at z = height. */
                stlfile << "   facet normal " << 0.000000e+00 << " "
                            << 0.000000e+00 << " " << 1.000000e+00 << "\n";
                stlfile << "      outer loop\n";
                stlfile << "         vertex " << cell->vertex(0)[0] << " "
                            << cell->vertex(0)[1] << " " << height << "\n";
                stlfile << "         vertex " << cell->vertex(1)[0] << " "
                            << cell->vertex(1)[1] << " " << height << "\n";
                stlfile << "         vertex " << cell->vertex(2)[0] << " "
                            << cell->vertex(2)[1] << " " << height << "\n";
                stlfile << "      endloop\n";
                stlfile << "   endfacet\n";
                stlfile << "   facet normal " << 0.000000e+00 << " "
                            << 0.000000e+00 << " " << 1.000000e+00 << "\n";
                stlfile << "      outer loop\n";
                stlfile << "         vertex " << cell->vertex(1)[0] << " "
                            << cell->vertex(1)[1] << " " << height << "\n";
                stlfile << "         vertex " << cell->vertex(3)[0] << " "
                            << cell->vertex(3)[1] << " " << height << "\n";
                stlfile << "         vertex " << cell->vertex(2)[0] << " "
                            << cell->vertex(2)[1] << " " << height << "\n";
                stlfile << "      endloop\n";
                stlfile << "   endfacet\n";
            } else /* The cell has a left-handed set up */
            {
                /* Write one side at z = 0. */
                stlfile << "   facet normal " << 0.000000e+00 << " "
                            << 0.000000e+00 << " " << -1.000000e+00 << "\n";
                stlfile << "      outer loop\n";
                stlfile << "         vertex " << cell->vertex(0)[0] << " "
                            << cell->vertex(0)[1] << " " << 0.000000e+00 << "\n";
                stlfile << "         vertex " << cell->vertex(1)[0] << " "
                            << cell->vertex(1)[1] << " " << 0.000000e+00 << "\n";
                stlfile << "         vertex " << cell->vertex(2)[0] << " "
                            << cell->vertex(2)[1] << " " << 0.000000e+00 << "\n";
                stlfile << "      endloop\n";
                stlfile << "   endfacet\n";
                stlfile << "   facet normal " << 0.000000e+00 << " "
                            << 0.000000e+00 << " " << -1.000000e+00 << "\n";
                stlfile << "      outer loop\n";
                stlfile << "         vertex " << cell->vertex(1)[0] << " "
                            << cell->vertex(1)[1] << " " << 0.000000e+00 << "\n";
                stlfile << "         vertex " << cell->vertex(3)[0] << " "
                            << cell->vertex(3)[1] << " " << 0.000000e+00 << "\n";
                stlfile << "         vertex " << cell->vertex(2)[0] << " "
                            << cell->vertex(2)[1] << " " << 0.000000e+00 << "\n";
                stlfile << "      endloop\n";
                stlfile << "   endfacet\n";
                /* Write one side at z = height. */
                stlfile << "   facet normal " << 0.000000e+00 << " "
                            << 0.000000e+00 << " " << 1.000000e+00 << "\n";
                stlfile << "      outer loop\n";
                stlfile << "         vertex " << cell->vertex(0)[0] << " "
                            << cell->vertex(0)[1] << " " << height << "\n";
                stlfile << "         vertex " << cell->vertex(2)[0] << " "
                            << cell->vertex(2)[1] << " " << height << "\n";
                stlfile << "         vertex " << cell->vertex(1)[0] << " "
                            << cell->vertex(1)[1] << " " << height << "\n";
                stlfile << "      endloop\n";
                stlfile << "   endfacet\n";
                stlfile << "   facet normal " << 0.000000e+00 << " "
                            << 0.000000e+00 << " " << 1.000000e+00 << "\n";
                stlfile << "      outer loop\n";
                stlfile << "         vertex " << cell->vertex(1)[0] << " "
                            << cell->vertex(1)[1] << " " << height << "\n";
                stlfile << "         vertex " << cell->vertex(2)[0] << " "
                            << cell->vertex(2)[1] << " " << height << "\n";
                stlfile << "         vertex " << cell->vertex(3)[0] << " "
                            << cell->vertex(3)[1] << " " << height << "\n";
                stlfile << "      endloop\n";
                stlfile << "   endfacet\n";
            }
            for (unsigned int face_number = 0;
                 face_number < GeometryInfo<dim>::faces_per_cell;
                 ++face_number) {
                const typename DoFHandler<dim>::face_iterator face =
                        cell->face(face_number);
                if ((face->at_boundary()) ||
                        (!face->at_boundary() &&
                         (state.block(
                              SolutionBlocks::density)[cell->neighbor(face_number)->get_fe().component_to_system_index(0, 0)] <
                          0.5))) {
                    const Tensor<1, dim> normal_vector =
                            (face->center() - cell->center());
                    const double normal_norm = normal_vector.norm();
                    if ((face->vertex(0)[0] - face->vertex(0)[0]) *
                            (face->vertex(1)[1] - face->vertex(0)[1]) *
                            0.000000e+00 +
                            (face->vertex(0)[1] - face->vertex(0)[1]) * (0 - 0) *
                            normal_vector[0] +
                            (height - 0) *
                            (face->vertex(1)[0] - face->vertex(0)[0]) *
                            normal_vector[1] -
                            (face->vertex(0)[0] - face->vertex(0)[0]) * (0 - 0) *
                            normal_vector[1] -
                            (face->vertex(0)[1] - face->vertex(0)[1]) *
                            (face->vertex(1)[0] - face->vertex(0)[0]) *
                            normal_vector[0] -
                            (height - 0) *
                            (face->vertex(1)[1] - face->vertex(0)[1]) * 0 >
                            0) {
                        stlfile << "   facet normal "
                                    << normal_vector[0] / normal_norm << " "
                                    << normal_vector[1] / normal_norm << " "
                                    << 0.000000e+00 << "\n";
                        stlfile << "      outer loop\n";
                        stlfile << "         vertex " << face->vertex(0)[0]
                                << " " << face->vertex(0)[1] << " "
                                    << 0.000000e+00 << "\n";
                        stlfile << "         vertex " << face->vertex(0)[0]
                                << " " << face->vertex(0)[1] << " " << height
                                << "\n";
                        stlfile << "         vertex " << face->vertex(1)[0]
                                << " " << face->vertex(1)[1] << " "
                                    << 0.000000e+00 << "\n";
                        stlfile << "      endloop\n";
                        stlfile << "   endfacet\n";
                        stlfile << "   facet normal "
                                    << normal_vector[0] / normal_norm << " "
                                    << normal_vector[1] / normal_norm << " "
                                    << 0.000000e+00 << "\n";
                        stlfile << "      outer loop\n";
                        stlfile << "         vertex " << face->vertex(0)[0]
                                << " " << face->vertex(0)[1] << " " << height
                                << "\n";
                        stlfile << "         vertex " << face->vertex(1)[0]
                                << " " << face->vertex(1)[1] << " " << height
                                << "\n";
                        stlfile << "         vertex " << face->vertex(1)[0]
                                << " " << face->vertex(1)[1] << " "
                                    << 0.000000e+00 << "\n";
                        stlfile << "      endloop\n";
                        stlfile << "   endfacet\n";
                    } else {
                        stlfile << "   facet normal "
                                    << normal_vector[0] / normal_norm << " "
                                    << normal_vector[1] / normal_norm << " "
                                    << 0.000000e+00 << "\n";
                        stlfile << "      outer loop\n";
                        stlfile << "         vertex " << face->vertex(0)[0]
                                << " " << face->vertex(0)[1] << " "
                                    << 0.000000e+00 << "\n";
                        stlfile << "         vertex " << face->vertex(1)[0]
                                << " " << face->vertex(1)[1] << " "
                                    << 0.000000e+00 << "\n";
                        stlfile << "         vertex " << face->vertex(0)[0]
                                << " " << face->vertex(0)[1] << " " << height
                                << "\n";
                        stlfile << "      endloop\n";
                        stlfile << "   endfacet\n";
                        stlfile << "   facet normal "
                                    << normal_vector[0] / normal_norm << " "
                                    << normal_vector[1] / normal_norm << " "
                                    << 0.000000e+00 << "\n";
                        stlfile << "      outer loop\n";
                        stlfile << "         vertex " << face->vertex(0)[0]
                                << " " << face->vertex(0)[1] << " " << height
                                << "\n";
                        stlfile << "         vertex " << face->vertex(1)[0]
                                << " " << face->vertex(1)[1] << " "
                                    << 0.000000e+00 << "\n";
                        stlfile << "         vertex " << face->vertex(1)[0]
                                << " " << face->vertex(1)[1] << " " << height
                                << "\n";
                        stlfile << "      endloop\n";
                        stlfile << "   endfacet\n";
                    }
                }
            }
        }
    }
    stlfile << "endsolid bridge";
}

///Outputs to a 3d-printable file. Not yet usable in parallel.
template<>
void
KktSystem<3>::output_stl(const LA::MPI::BlockVector &state)
{
    std::ofstream stlfile;
    stlfile.open("bridge.stl");
    stlfile << "solid bridge\n" << std::scientific;
    const int dim = 3;
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (state.block(
                    SolutionBlocks::unfiltered_density)[cell->get_fe().component_to_system_index(0, 0)] > 0.5)
        {
            for (const auto n : cell->face_indices())
            {
                bool create_boundary = false;
                if (cell->at_boundary(n))
                {
                    create_boundary = true;
                }
                else if (state.block(
                             SolutionBlocks::unfiltered_density)[cell->neighbor(n)->get_fe().component_to_system_index(0, 0)] <= 0.5)
                {
                    create_boundary = true;
                }

                if (create_boundary)
                {
                    const auto face = cell->face(n);
                    const Tensor<1,dim> normal_vector = face->center() -
                            cell->center();
                    double normal_norm = normal_vector.norm();
                    const Tensor<1,dim> edge_vectors_1 = face->vertex(1) - face->vertex(0);
                    const Tensor<1,dim> edge_vectors_2 = face->vertex(2) - face->vertex(0);

                    const Tensor<2, dim> edge_tensor (
                                {{edge_vectors_1[0], edge_vectors_1[1],edge_vectors_1[2]},
                                 {edge_vectors_2[0], edge_vectors_2[1],edge_vectors_2[2]},
                                 {normal_vector[0], normal_vector[1], normal_vector[2]}});
                    const bool is_right_handed_cell = (determinant(edge_tensor) > 0);

                    if (is_right_handed_cell)
                    {
                        stlfile << "   facet normal "
                                    << normal_vector[0] / normal_norm << " "
                                    << normal_vector[1] / normal_norm << " "
                                    << normal_vector[2] / normal_norm << "\n";
                        stlfile << "      outer loop\n";
                        stlfile << "         vertex " << face->vertex(0)[0]
                                << " " << face->vertex(0)[1] << " "
                                    << face->vertex(0)[2] << "\n";
                        stlfile << "         vertex " << face->vertex(1)[0]
                                << " " << face->vertex(1)[1] << " "
                                    << face->vertex(1)[2] << "\n";
                        stlfile << "         vertex " << face->vertex(2)[0]
                                << " " << face->vertex(2)[1] << " "
                                    << face->vertex(2)[2] << "\n";
                        stlfile << "      endloop\n";
                        stlfile << "   endfacet\n";
                        stlfile << "   facet normal "
                                    << normal_vector[0] / normal_norm << " "
                                    << normal_vector[1] / normal_norm << " "
                                    << normal_vector[2] / normal_norm << "\n";
                        stlfile << "      outer loop\n";
                        stlfile << "         vertex " << face->vertex(1)[0]
                                << " " << face->vertex(1)[1] << " " << face->vertex(1)[2]
                                << "\n";
                        stlfile << "         vertex " << face->vertex(3)[0]
                                << " " << face->vertex(3)[1] << " " << face->vertex(3)[2]
                                << "\n";
                        stlfile << "         vertex " << face->vertex(2)[0]
                                << " " << face->vertex(2)[1] << " "
                                    << face->vertex(2)[2] << "\n";
                        stlfile << "      endloop\n";
                        stlfile << "   endfacet\n";
                    }
                    else
                    {
                        stlfile << "   facet normal "
                                    << normal_vector[0] / normal_norm << " "
                                    << normal_vector[1] / normal_norm << " "
                                    << normal_vector[2] / normal_norm << "\n";
                        stlfile << "      outer loop\n";
                        stlfile << "         vertex " << face->vertex(0)[0]
                                << " " << face->vertex(0)[1] << " "
                                    << face->vertex(0)[2] << "\n";
                        stlfile << "         vertex " << face->vertex(2)[0]
                                << " " << face->vertex(2)[1] << " "
                                    << face->vertex(2)[2] << "\n";
                        stlfile << "         vertex " << face->vertex(1)[0]
                                << " " << face->vertex(1)[1] << " "
                                    << face->vertex(1)[2] << "\n";
                        stlfile << "      endloop\n";
                        stlfile << "   endfacet\n";
                        stlfile << "   facet normal "
                                    << normal_vector[0] / normal_norm << " "
                                    << normal_vector[1] / normal_norm << " "
                                    << normal_vector[2] / normal_norm << "\n";
                        stlfile << "      outer loop\n";
                        stlfile << "         vertex " << face->vertex(1)[0]
                                << " " << face->vertex(1)[1] << " " << face->vertex(1)[2]
                                << "\n";
                        stlfile << "         vertex " << face->vertex(2)[0]
                                << " " << face->vertex(2)[1] << " " << face->vertex(2)[2]
                                << "\n";
                        stlfile << "         vertex " << face->vertex(3)[0]
                                << " " << face->vertex(3)[1] << " "
                                    << face->vertex(3)[2] << "\n";
                        stlfile << "      endloop\n";
                        stlfile << "   endfacet\n";
                    }

                }

            }
        }
    }
}
}

template class SAND::KktSystem<2>;
template class SAND::KktSystem<3>;
