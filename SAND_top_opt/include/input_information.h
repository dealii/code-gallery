//
// Created by justin on 3/11/21.
//

#ifndef SAND_INPUT_INFORMATION_H
#define SAND_INPUT_INFORMATION_H
#include <deal.II/grid/tria_accessor.h>
#include "parameters_and_components.h"

namespace SAND {
    using namespace dealii;

    namespace Input
    {
        constexpr double volume_percentage = .5;

        //geometry options
        constexpr unsigned int geometry_base = GeometryOptions::mbb;

        constexpr unsigned int dim = 3;
        // constexpr unsigned int refinements = 3;
        extern unsigned int refinements = 3;

        //nonlinear algorithm options
        constexpr double initial_barrier_size = 25;
        constexpr double min_barrier_size = 1e-12;

        constexpr double fraction_to_boundary = .7;
        constexpr unsigned int max_steps=100;

        constexpr unsigned int barrier_reduction=BarrierOptions::loqo;
        constexpr double required_norm = .0001;

        //density filter options
        constexpr double filter_r = .25;

        //other options
        constexpr double density_penalty_exponent = 3;

        //output options
        constexpr bool output_full_preconditioned_matrix = false;
        constexpr bool output_full_matrix = false;
        constexpr bool output_parts_of_matrix = false;

        //Linear solver options
        constexpr unsigned int solver_choice = SolverOptions::inexact_K_with_inexact_A_gmres;
        constexpr bool use_eisenstat_walker = false;
        constexpr double default_gmres_tolerance = 1e-6;

        extern unsigned int a_inv_iterations = 5;
        extern unsigned int k_inv_iterations = 30;

        constexpr double a_rel_tol = 1e-12;
        constexpr double k_rel_tol = 1e-12;

        //Material Options
        constexpr double material_lambda = 1.;
        constexpr double material_mu = 1.;
    }
}
#endif //SAND_INPUT_INFORMATION_H