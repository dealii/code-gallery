/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2015 by David Wells
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 */

#ifndef dealii__cdr_write_pvtu_output_templates_h
#define dealii__cdr_write_pvtu_output_templates_h
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II-cdr/write_pvtu_output.h>

#include <fstream>
#include <string>
#include <vector>

// Here is the implementation of the important function. This is similar to
// what is presented in step-40.
namespace CDR
{
  using namespace dealii;

  template <int dim, typename VectorType>
  void
  WritePVTUOutput::write_output(const DoFHandler<dim> &dof_handler,
                                const VectorType &     solution,
                                const unsigned int     time_step_n,
                                const double           current_time)
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "u");

    const auto &  triangulation = dof_handler.get_triangulation();
    Vector<float> subdomain(triangulation.n_active_cells());
    for (auto &domain : subdomain)
      {
        domain = triangulation.locally_owned_subdomain();
      }
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.build_patches(patch_level);

    DataOutBase::VtkFlags flags;
    flags.time = current_time;
    // While the default flag is for the best compression level, using
    // <code>best_speed</code> makes this function much faster.
    flags.compression_level = DataOutBase::CompressionLevel::best_speed;
    data_out.set_flags(flags);

    unsigned int subdomain_n;
    if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1)
      {
        subdomain_n = 0;
      }
    else
      {
        subdomain_n = triangulation.locally_owned_subdomain();
      }

    std::ofstream output("solution-" + Utilities::int_to_string(time_step_n) +
                         "." + Utilities::int_to_string(subdomain_n, 4) +
                         ".vtu");

    data_out.write_vtu(output);

    if (this_mpi_process == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i = 0;
             i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
             ++i)
          filenames.push_back("solution-" +
                              Utilities::int_to_string(time_step_n) + "." +
                              Utilities::int_to_string(i, 4) + ".vtu");
        std::ofstream master_output(
          "solution-" + Utilities::int_to_string(time_step_n) + ".pvtu");
        data_out.write_pvtu_record(master_output, filenames);
      }
  }
} // namespace CDR
#endif
