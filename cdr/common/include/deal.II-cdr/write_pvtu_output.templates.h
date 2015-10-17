#ifndef dealii__cdr_write_pvtu_output_templates_h
#define dealii__cdr_write_pvtu_output_templates_h
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II-cdr/write_pvtu_output.h>

#include <string>
#include <fstream>
#include <vector>

namespace CDR
{
  using namespace dealii;

  template<int dim, typename VectorType>
  void WritePVTUOutput::write_output(const DoFHandler<dim> &dof_handler,
                                     const VectorType      &solution,
                                     const unsigned int    time_step_n,
                                     const double          current_time)
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "u");

    Vector<float> subdomain (dof_handler.get_tria().n_active_cells());
    for (auto &domain : subdomain)
      {
        domain = dof_handler.get_tria().locally_owned_subdomain();
      }
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.build_patches(patch_level);

    DataOutBase::VtkFlags flags;
    flags.time = current_time;
    flags.compression_level = DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;
    data_out.set_flags(flags);

    unsigned int subdomain_n;
    if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1)
      {
        subdomain_n = 0;
      }
    else
      {
        subdomain_n = dof_handler.get_tria().locally_owned_subdomain();
      }

    std::ofstream output
      ("solution-" + Utilities::int_to_string(time_step_n) + "."
       + Utilities::int_to_string(subdomain_n, 4)
       + ".vtu");

    data_out.write_vtu(output);

    if (this_mpi_process == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
             ++i)
          filenames.push_back
            ("solution-" + Utilities::int_to_string (time_step_n) + "."
             + Utilities::int_to_string (i, 4) + ".vtu");
        std::ofstream master_output
          ("solution-" + Utilities::int_to_string(time_step_n) + ".pvtu");
        data_out.write_pvtu_record(master_output, filenames);
      }
  }
}
#endif
