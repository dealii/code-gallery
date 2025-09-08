//
// Created by justin on 5/13/21.
//
#include "../include/density_filter.h"
#include "../include/input_information.h"
#include <deal.II/base/tensor.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/cell_id.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

namespace SAND
{
    using namespace dealii;

    template<int dim>
    DensityFilter<dim>::DensityFilter() :
        mpi_communicator(MPI_COMM_WORLD),
        pcout(std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    {
    }

    ///When initialized, this function takes the current triangulation and creates a matrix corresponding to a
    /// convolution being applied to a piecewise constant function on that triangulation
    ///
    template<int dim>
    void
    DensityFilter<dim>::initialize(DoFHandler<dim> &dof_handler) {
        ///Start by making vectors to fill with information about the x,y,z coords of centers of cells
        ///
        std::vector<unsigned int> block_component(10, 2);
        block_component[SolutionBlocks::density] = 0;
        block_component[SolutionBlocks::displacement] = 1;
        const std::vector<types::global_dof_index> dofs_per_block =
                DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
        const unsigned int n_p = dofs_per_block[0];
        IndexSet local_owned = dof_handler.locally_owned_dofs().get_view(0, n_p);
        x_coord.resize(n_p);
        y_coord.resize(n_p);
        z_coord.resize(n_p);
        auto row_sum = z_coord;
        auto row_sum_full = z_coord;
        cell_m.resize(n_p);
        x_coord_part.resize(n_p);
        y_coord_part.resize(n_p);
        z_coord_part.resize(n_p);
        cell_m_part.resize(n_p);

        filter_dsp.reinit(dofs_per_block[0],
                          dofs_per_block[0]);
        filter_sparsity_pattern.copy_from(filter_dsp);

        // const auto owned_dofs = dof_handler.locally_owned_dofs().get_view(0, dofs_per_block[0]);

        // filter_matrix.reinit(owned_dofs, filter_sparsity_pattern, MPI_COMM_WORLD);


        // //identity filter
        // for (const auto &cell : dof_handler.active_cell_iterators())
        // {
        //     if(cell->is_locally_owned())
        //     {
        //         std::vector<unsigned int> i(cell->get_fe().n_dofs_per_cell());
        //         cell->get_dof_indices(i);
        //         filter_matrix.add(i[cell->get_fe().component_to_system_index(0, 0)], i[cell->get_fe().component_to_system_index(0, 0)], 1.0);
        //     }
        // }
       
        std::set<unsigned int> neighbor_ids;
        std::set<typename DoFHandler<dim>::cell_iterator> cells_to_check;
        std::set<typename DoFHandler<dim>::cell_iterator> cells_to_check_temp;
        ///finds neighbors whose values would be relevant, and adds them to the sparsity pattern of the matrix
         for (const auto &cell : dof_handler.active_cell_iterators())
         {
             if(cell->is_locally_owned())
             {
                std::vector<types::global_dof_index> i(cell->get_fe().n_dofs_per_cell());
                cell->get_dof_indices(i);
                const unsigned int i_val = i[cell->get_fe().component_to_system_index(0, 0)];
                x_coord_part[i_val] = cell->center()[0] ;
                y_coord_part[i_val] = cell->center()[1] ;
                cell_m_part[i_val] = cell->measure();
                if (dim==3)
                {
                    z_coord_part[i_val] = cell->center()[2] ;
                }
             }
         }
        MPI_Allreduce(x_coord_part.data(), x_coord.data(), n_p, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(y_coord_part.data(), y_coord.data(), n_p, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(z_coord_part.data(), z_coord.data(), n_p, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(cell_m_part.data(), cell_m.data(), n_p, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            if(cell->is_locally_owned())
            {
                std::vector<types::global_dof_index> i(cell->get_fe().n_dofs_per_cell());
                cell->get_dof_indices(i);
                for (const auto &neighbor_cell_index : find_relevant_neighbors(i[cell->get_fe().component_to_system_index(0, 0)]))
                {
                    filter_dsp.add(i[cell->get_fe().component_to_system_index(0, 0)], neighbor_cell_index);
                }
            }
        }
        filter_sparsity_pattern.copy_from(filter_dsp);
        const auto owned_dofs = dof_handler.locally_owned_dofs().get_view(4 * n_p, 5 * n_p);

        filter_matrix.reinit(owned_dofs, filter_sparsity_pattern, MPI_COMM_WORLD);
        filter_matrix_transpose.reinit(owned_dofs, filter_sparsity_pattern, MPI_COMM_WORLD);

        /// adds values to the matrix corresponding to the max radius - distance
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            if(cell->is_locally_owned())
            {
                std::vector<unsigned int> i(cell->get_fe().n_dofs_per_cell());
                cell->get_dof_indices(i);
                auto cell_index = i[cell->get_fe().component_to_system_index(0, 0)];
                double value_total = 0;
                for (const auto &neighbor_cell_index : find_relevant_neighbors(cell_index))
                {
                    double d_x = std::abs(x_coord[cell_index]-x_coord[neighbor_cell_index]);
                    double d_y = std::abs(y_coord[cell_index]-y_coord[neighbor_cell_index]);
                    double d;
                    if (dim==3)
                    {
                        double d_z = std::abs(z_coord[cell_index]-z_coord[neighbor_cell_index]);
                        d = std::pow(d_x*d_x + d_y*d_y + d_z*d_z , .5);
                    }
                    else
                    {
                        d = std::pow(d_x*d_x + d_y*d_y , .5);
                    }
                    ///value should be (max radius - distance between cells)*cell measure
                    double value = (Input::filter_r - d)*cell_m[neighbor_cell_index];
                    value_total += value;
                }
                row_sum[cell_index] = value_total;
                
            }
        }
        MPI_Allreduce(row_sum.data(), row_sum_full.data(), n_p, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            if(cell->is_locally_owned())
            {
                std::vector<unsigned int> i(cell->get_fe().n_dofs_per_cell());
                cell->get_dof_indices(i);
                auto cell_index = i[cell->get_fe().component_to_system_index(0, 0)];
                for (const auto &neighbor_cell_index : find_relevant_neighbors(cell_index))
                {
                    double d_x = std::abs(x_coord[cell_index]-x_coord[neighbor_cell_index]);
                    double d_y = std::abs(y_coord[cell_index]-y_coord[neighbor_cell_index]);
                    double d;
                    if (dim==3)
                    {
                        double d_z = std::abs(z_coord[cell_index]-z_coord[neighbor_cell_index]);
                        d = std::pow(d_x*d_x + d_y*d_y + d_z*d_z , .5);
                    }
                    else
                    {
                        d = std::pow(d_x*d_x + d_y*d_y , .5);
                    }
                    ///value should be (max radius - distance between cells)*cell measure
                    double value = (Input::filter_r - d)*cell_m[neighbor_cell_index];
                    filter_matrix.set(cell_index, neighbor_cell_index, value/row_sum_full[cell_index]);
                    filter_matrix_transpose.set(cell_index, neighbor_cell_index, value/row_sum_full[neighbor_cell_index]);
                }
            }
        }
        double sum_sum = 0;
        double x_sum = 0;
        double y_sum = 0;
        double z_sum = 0;
        for (int i=0; i< row_sum_full.size(); i++)
        {
            sum_sum += std::abs(row_sum_full[i]);
            x_sum += std::abs(x_coord[i]);
            y_sum += std::abs(y_coord[i]);
            z_sum += std::abs(z_coord[i]);
        }
        filter_matrix.compress(VectorOperation::insert);
        filter_matrix_transpose.compress(VectorOperation::insert);

        ///here we normalize the filter so it computes an average. Sum of values in a row should be 1
        // for (const auto &cell : dof_handler.active_cell_iterators())
        // {
        //     if(cell->is_locally_owned())
        //     {
        //         std::vector<types::global_dof_index> i(cell->get_fe().n_dofs_per_cell());
        //         cell->get_dof_indices(i);
        //         const int i_ind = cell->get_fe().component_to_system_index(0, 0);
        //         double denominator = 0;
        //         typename LA::MPI::SparseMatrix::iterator iter = filter_matrix.begin(
        //                 i[i_ind]);
        //         for (; iter != filter_matrix.end(i[i_ind]); iter++)
        //         {
        //             denominator = denominator + iter->value();
        //         }
        //         iter = filter_matrix.begin(i[i_ind]);
        //         for (; iter != filter_matrix.end(i[i_ind]); iter++)
        //         {
        //             iter->value() = iter->value() / denominator;
        //         }
        //     }
        // }
        // pcout << "FILTER TO 5" << std::endl;
        // LA::MPI::Vector test_density_start;
        // test_density_start.reinit(local_owned, mpi_communicator);
        // LA::MPI::Vector test_density_end;
        // test_density_end.reinit(local_owned, mpi_communicator);
        // test_density_end = 0.;
        // for (int i = 0; i<n_p; i++)
        // {
        //     if (test_density_start.in_local_range(i))
        //     {
        //         test_density_start[i] = i;
        //     }
        // }
        // test_density_start.compress(VectorOperation::insert);
        // filter_matrix.vmult(test_density_end,test_density_start);

    }

    ///This function finds which neighbors are within a certain radius of the initial cell.
    template<int dim>
    std::set<types::global_dof_index>
    DensityFilter<dim>::find_relevant_neighbors(types::global_dof_index cell_index) const
    {
        double d_x,d_y,d_z;
        std::set<types::global_dof_index> relevant_cells;
            for (unsigned int i=0; i < x_coord.size(); i++)
            {
                d_x = std::abs(x_coord[cell_index]-x_coord[i]);

                if (d_x < Input::filter_r)
                {
                    d_y = std::abs(y_coord[cell_index]-y_coord[i]);

                    if ((d_x*d_x + d_y*d_y) < (Input::filter_r*Input::filter_r))
                    {

                        if (dim == 3)
                        {
                            d_z = std::abs(z_coord[cell_index]-z_coord[i]);

                            if ((d_x*d_x + d_y*d_y + d_z*d_z) < (Input::filter_r*Input::filter_r))
                            {
                                relevant_cells.insert(i);
                            }
                        }
                        else
                        {
                            relevant_cells.insert(i);
                        }

                    }
                }
        }
        return relevant_cells;

    }

}//SAND namespace
    template class SAND::DensityFilter<2>;
    template class SAND::DensityFilter<3>;
