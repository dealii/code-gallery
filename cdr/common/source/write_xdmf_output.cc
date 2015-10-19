#include <deal.II/lac/trilinos_vector.h>

#include <deal.II-cdr/write_xdmf_output.templates.h>

namespace CDR
{
  using namespace dealii;

  WriteXDMFOutput::WriteXDMFOutput(const unsigned int patch_level,
                                   const bool update_mesh_at_each_step)
    : patch_level {patch_level},
      update_mesh_at_each_step {update_mesh_at_each_step},
      xdmf_file_name {"solution.xdmf"},
      data_component_interpretation
  {DataComponentInterpretation::component_is_scalar},
  write_mesh {true}
  {}

  template
  void WriteXDMFOutput::write_output(const DoFHandler<2>  &dof_handler,
                                     const Vector<double> &solution,
                                     const unsigned int   time_step_n,
                                     const double         current_time);

  template
  void WriteXDMFOutput::write_output(const DoFHandler<2>                 &dof_handler,
                                     const TrilinosWrappers::MPI::Vector &solution,
                                     const unsigned int                  time_step_n,
                                     const double                        current_time);

  template
  void WriteXDMFOutput::write_output(const DoFHandler<3>  &dof_handler,
                                     const Vector<double> &solution,
                                     const unsigned int   time_step_n,
                                     const double         current_time);

  template
  void WriteXDMFOutput::write_output(const DoFHandler<3>                 &dof_handler,
                                     const TrilinosWrappers::MPI::Vector &solution,
                                     const unsigned int                  time_step_n,
                                     const double                        current_time);
}
