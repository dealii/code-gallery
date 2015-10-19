#ifndef dealii__cdr_write_xmdf_output_h
#define dealii__cdr_write_xmdf_output_h
#include <deal.II/base/data_out_base.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_component_interpretation.h>

#include <string>
#include <vector>


namespace CDR
{
  using namespace dealii;

  class WriteXDMFOutput
  {
  public:
    WriteXDMFOutput(const unsigned int patch_level,
                    const bool update_mesh_at_each_step = true);

    template<int dim, typename VectorType>
    void write_output(const DoFHandler<dim> &dof_handler,
                      const VectorType      &solution,
                      const unsigned int    time_step_n,
                      const double          current_time);
  private:
    const unsigned int patch_level;
    const bool update_mesh_at_each_step;
    const std::string xdmf_file_name;
    const std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation;
    std::vector<XDMFEntry> xdmf_entries;
    bool write_mesh;
  };
}
#endif
