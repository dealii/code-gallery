//
// Created by justin on 3/11/21.
//

#ifndef SAND_PARAMETERS_AND_COMPONENTS_H
#define SAND_PARAMETERS_AND_COMPONENTS_H
#include <deal.II/grid/tria_accessor.h>


namespace SAND {
    using namespace dealii;

    namespace SolutionComponents {
        template<int dim> static constexpr unsigned int density_lower_slack_multiplier = 0;
        template<int dim> static constexpr unsigned int density_upper_slack_multiplier = 1;
        template<int dim> static constexpr unsigned int density_lower_slack = 2;
        template<int dim> static constexpr unsigned int density_upper_slack = 3;
        template<int dim> static constexpr unsigned int unfiltered_density = 4;
        template<int dim> static constexpr unsigned int displacement = 5;
        template<int dim> static constexpr unsigned int displacement_multiplier = 5 + dim;
        template<int dim> static constexpr unsigned int unfiltered_density_multiplier = 5 + 2 * dim;
        template<int dim> static constexpr unsigned int density = 6 + 2 * dim;
        template<int dim> static constexpr unsigned int total_volume_multiplier = 7 + 2 * dim;
    }

    namespace SolutionBlocks {
        static constexpr unsigned int density_lower_slack_multiplier = 0;
        static constexpr unsigned int density_upper_slack_multiplier = 1;
        static constexpr unsigned int density_lower_slack = 2;
        static constexpr unsigned int density_upper_slack = 3;
        static constexpr unsigned int unfiltered_density = 4;
        static constexpr unsigned int displacement = 5;
        static constexpr unsigned int displacement_multiplier = 6;
        static constexpr unsigned int unfiltered_density_multiplier = 7;
        static constexpr unsigned int density = 8;
        static constexpr unsigned int total_volume_multiplier = 9;
    }

    namespace BoundaryIds {
        static constexpr types::boundary_id no_force = 101;
        static constexpr types::boundary_id down_force = 102;
        static constexpr types::boundary_id held_still = 103;
    }

    namespace MaterialIds {
        static constexpr types::material_id with_multiplier = 10;
        static constexpr types::material_id without_multiplier = 9;
    }

    namespace SolverOptions {
        static constexpr unsigned int direct_solve = 1;
        static constexpr unsigned int exact_preconditioner_with_gmres = 2;
        static constexpr unsigned int inexact_K_with_exact_A_gmres = 3;
        static constexpr unsigned int inexact_K_with_inexact_A_gmres = 4;
    }

    namespace BarrierOptions {
        static constexpr unsigned int loqo = 1;
        static constexpr unsigned int monotone = 2;
        static constexpr unsigned int mixed = 3;
    }
    namespace GeometryOptions {
        static constexpr unsigned int mbb = 1;
        static constexpr unsigned int l_shape = 2;
    }
    static constexpr unsigned int block_number = 10;
}
#endif //SAND_PARAMETERS_AND_COMPONENTS_H
