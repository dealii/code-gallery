#ifndef dealii__cdr_write_xmdf_output_templates_h
#define dealii__cdr_write_xmdf_output_templates_h
#include <deal.II/base/utilities.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II-cdr/write_xdmf_output.h>

namespace CDR
{
  template<int dim, typename VectorType>
  void WriteXDMFOutput::write_output(const DoFHandler<dim> &dof_handler,
                                     const VectorType      &solution,
                                     const unsigned int    time_step_n,
                                     const double          current_time)
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "u", DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.build_patches(patch_level);

    auto solution_file_name = "solution-"
      + Utilities::int_to_string(time_step_n, 9) + ".h5";
    std::string mesh_file_name;
    if (update_mesh_at_each_step)
      {
        mesh_file_name = "mesh-"
          + Utilities::int_to_string(time_step_n, 9) + ".h5";
      }
    else
      {
        mesh_file_name = "mesh.h5";
      }

    DataOutBase::DataOutFilter data_filter
      (DataOutBase::DataOutFilterFlags(true, true));
    data_out.write_filtered_data(data_filter);
    data_out.write_hdf5_parallel(data_filter, write_mesh, mesh_file_name,
                                 solution_file_name, MPI_COMM_WORLD);
    if (!update_mesh_at_each_step)
      {
        write_mesh = false;
      }

    xdmf_entries.push_back(data_out.create_xdmf_entry
                           (data_filter, mesh_file_name, solution_file_name,
                            current_time, MPI_COMM_WORLD));
    data_out.write_xdmf_file(xdmf_entries, xdmf_file_name, MPI_COMM_WORLD);
  }
}
#endif
