#include "PhaseFieldSolver.h"

void PhaseFieldSolver::output_results(const unsigned int timestep_number) const {
    const Vector<double> localized_solution(old_solution);

    //using only one process to output the result
    if (this_mpi_process == 0)
    {
        DataOut<2> data_out;
        data_out.attach_dof_handler(dof_handler);

        std::vector<std::string> solution_names;
        solution_names.emplace_back ("p");
        solution_names.emplace_back ("T");

        data_out.add_data_vector(localized_solution, solution_names);
        const std::string filename =
                "solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtk";
        DataOutBase::VtkFlags vtk_flags;
        vtk_flags.compression_level =
                DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;
        data_out.set_flags(vtk_flags);
        std::ofstream output(filename);

        data_out.build_patches();
        data_out.write_vtk(output);
    }
}
