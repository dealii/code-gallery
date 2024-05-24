#include "nonlinear_heat.h"
#include "allheaders.h"
/**
 * This sets the boundary condition of the problem.
 * @param time Time (useful, if the boundary condition is time dependent)
 */
void nonlinear_heat::set_boundary_conditions(double time)
{
    Boundary_values_left bl_left;
    bl_left.set_time(time);
    std::map<types::global_dof_index,double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
            1,
            bl_left,
            boundary_values);

    for (auto &boundary_value: boundary_values)
        present_solution(boundary_value.first) = boundary_value.second;
}
