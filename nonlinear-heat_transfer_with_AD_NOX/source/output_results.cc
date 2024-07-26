/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
 * Copyright (C) 2024 by Narasimhan Swaminathan
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 */

#include "nonlinear_heat.h"

/**
 * Outputs the results to a vtu file, every #prn step.
 * @param prn
 */
void nonlinear_heat::output_results(unsigned int prn) const
{
    DataOut<2> data_out;
    data_out.attach_dof_handler(dof_handler);
    std::vector<std::string> solution_names;
    solution_names.emplace_back ("Temperature");
    data_out.add_data_vector(converged_solution, solution_names);
    data_out.build_patches();
    const std::string filename =
            "output/solution-" + Utilities::int_to_string(prn , 3) + ".vtu";
    std::ofstream output(filename);
    data_out.write_vtu(output);
}
