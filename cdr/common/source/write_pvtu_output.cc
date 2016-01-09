#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II-cdr/write_pvtu_output.templates.h>

namespace CDR
{
  using namespace dealii;

  WritePVTUOutput::WritePVTUOutput(const unsigned int patch_level)
    : patch_level {patch_level},
      this_mpi_process {Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)}
  {}

  template
  void WritePVTUOutput::write_output(const DoFHandler<2>  &dof_handler,
                                     const Vector<double> &solution,
                                     const unsigned int    time_step_n,
                                     const double          current_time);

  template
  void WritePVTUOutput::write_output(const DoFHandler<3>  &dof_handler,
                                     const Vector<double> &solution,
                                     const unsigned int    time_step_n,
                                     const double          current_time);

  template
  void WritePVTUOutput::write_output(const DoFHandler<2>                 &dof_handler,
                                     const TrilinosWrappers::MPI::Vector &solution,
                                     const unsigned int                   time_step_n,
                                     const double                         current_time);

  template
  void WritePVTUOutput::write_output(const DoFHandler<3>                 &dof_handler,
                                     const TrilinosWrappers::MPI::Vector &solution,
                                     const unsigned int                   time_step_n,
                                     const double                         current_time);
}
