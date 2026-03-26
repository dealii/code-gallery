/*
 * PlasticityLabProg.h
 *
 *  Created on: 09 Jul 2014
 *      Author: cerecam
 */

#ifndef PLASTICITYLABPROG_H_
#define PLASTICITYLABPROG_H_

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/base/function.h>
#include <stdexcept>

#include "DoFSystem.h"
#include "LBCSystem.h"
#include "MixedFEProjector.h"
#include "Material.h"
#include "NewtonStepSystem.h"

namespace PlasticityLab {
  using namespace dealii;

  template <int dim, typename Number = double>
  class PlasticityLabProg {
   public:
    PlasticityLabProg(Material<dim+1, Number> &);
    virtual ~PlasticityLabProg();
    void run();

   private:
    void make_grid (int);
    void make_cylindrical_grid(
      Triangulation<dim> &triangulation,
      LBCSystem<dim, Number, dim+1> &mech_lbc_system,
      LBCSystem<dim, Number, 1> &therm_lbc_system,
      int n_initial_global_refinements);
    void make_grid_();
    void make_necking_grid(
      Triangulation<dim> &triangulation,
      LBCSystem<dim, Number, dim+1> &mech_lbc_system,
      LBCSystem<dim, Number, 1> &therm_lbc_system,
      int n_initial_global_refinements);
    void make_interference_cylinder_grid(
      Triangulation<dim> &triangulation,
      LBCSystem<dim, Number, dim+1> &mech_lbc_system,
      LBCSystem<dim, Number, 1> &therm_lbc_system,
      int n_initial_global_refinements);
    void make_cylindrical_impact_grid(
      Triangulation<dim> &triangulation,
      LBCSystem<dim, Number, dim+1> &mech_lbc_system,
      LBCSystem<dim, Number, 1> &therm_lbc_system,
      int n_initial_global_refinements);
    void make_ball_in_hypershell_grid(
      Triangulation<dim> &triangulation,
      LBCSystem<dim, Number, dim+1> &mech_lbc_system,
      LBCSystem<dim, Number, 1> &therm_lbc_system,
      int n_initial_global_refinements);
    void make_hook_membrane_grid(int);

    void set_mesh_motion_LBCs(
      Triangulation<dim> &triangulation,
      LBCSystem<dim, Number, dim> &mesh_motion_lbc_system);


    Tensor<2, dim+1, Number> get_rotation_tensor(const Tensor<2, dim+1, Number> &skew_symmetric_rotation) const;
    Tensor<2, dim+1, Number> get_rotation_tensor_variation(
      const Tensor<2, dim+1, Number> &skew_symmetric_rotation,
      const Tensor<2, dim+1, Number> &skew_symmetric_rotation_variation) const;


    void remap_material_state_variables(
      const NewtonStepSystem &mesh_motion_nonlinear_system,
      const DoFSystem<dim, Number> &mesh_motion_dof_system,
      const NewtonStepSystem &mechanical_nonlinear_system,
      const DoFSystem<dim, Number> &mechanical_dof_system,
      const DoFSystem<dim, Number> &mixed_fe_dof_system,
      const std::vector< MixedFEProjector<dim, Number> > &mixed_fe_projector,
      Material<dim+1, Number> &material,
      std::unordered_map<point_index_t, Tensor<2, dim+1, Number>> &remapped_deformation_gradients);


    void remap_thermal_field(
      NewtonStepSystem &thermal_nonlinear_system,
      const DoFSystem<dim, Number> &thermal_dof_system,
      const NewtonStepSystem &mesh_motion_nonlinear_system,
      const DoFSystem<dim, Number> &mesh_motion_dof_system);


    void remap_mechanical_fields(
      NewtonStepSystem &mechanical_nonlinear_system,
      const DoFSystem<dim, Number> &mechanical_dof_system,
      const NewtonStepSystem &mesh_motion_nonlinear_system,
      const DoFSystem<dim, Number> &mesh_motion_dof_system);


    template <typename TriangulationType, typename MaterialType>
    void setup_material_data(TriangulationType &triangulation,
                             MaterialType &material);

    void setup_material_area_factors(
      const DoFSystem<dim, Number> &mesh_motion_dof_system,
      std::unordered_map<size_t, Tensor<1, dim+1, Number>> &material_area_factors);

    void update_material_area_factors(
        const NewtonStepSystem &mesh_motion_nonlinear_system,
        const DoFSystem<dim, Number> &mesh_motion_dof_system,
        std::unordered_map<size_t, Tensor<1, dim+1, Number>> &material_area_factors);

    template <typename TriangulationType>
    void setup_mixed_fe_projection_data(
      const TriangulationType &triangulation,
      std::vector< MixedFEProjector<dim, Number> > &MixedFeProjectors,
      const FiniteElement<dim> &MixedFE,
      const Quadrature<dim> &quadrature_formula);

    void assemble_mechanical_system(
      NewtonStepSystem &Newton_system,
      const DoFSystem<dim, Number> &mechanical_dof_system,
      const LBCSystem<dim, Number, dim+1> &mechanical_lbc_system,
      const NewtonStepSystem &mesh_motion_nonlinear_system,
      const DoFSystem<dim, Number> &mesh_motion_dof_system,
      const NewtonStepSystem &thermal_Newton_system,
      const DoFSystem<dim, Number> &thermal_dof_system,
      const DoFSystem<dim, Number> &mixed_fe_dof_system,
      Material<dim+1, Number> &material,
      const std::vector< MixedFEProjector<dim, Number> > &mixed_fe_projector,
      const bool fill_system_matrix = true,
      const bool update_material_state = false);

    void assemble_thermal_system(
      NewtonStepSystem &Newton_system,
      NewtonStepSystem &mechanical_nonlinear_system,
      const DoFSystem<dim, Number> &thermal_dof_system,
      const LBCSystem<dim, Number, 1> &thermal_lbc_system,
      const DoFSystem<dim, Number> &mechanical_dof_system,
      const DoFSystem<dim, Number> &mixed_fe_dof_system,
      Material<dim+1, Number> &material,
      const std::vector< MixedFEProjector<dim, Number> > &mixed_fe_projector,
      const std::unordered_map<size_t, Tensor<1, dim+1, Number>> &material_area_factors,
      const bool fill_system_matrix = true);

    void assemble_mesh_motion_system(
          NewtonStepSystem &mesh_motion_nonlinear_system,
          const DoFSystem<dim, Number> &mesh_motion_dof_system,
          const LBCSystem<dim, Number, dim> &mesh_motion_lbc_system,
          const NewtonStepSystem &deformation_nonlinear_system,
          const DoFSystem<dim, Number> &deformation_dof_system,
          const DoFSystem<dim, Number> &mixed_fe_dof_system,
          const std::vector< MixedFEProjector<dim, Number> > &mixed_fe_projector,
          const bool fill_system_matrix = true);

    void solve_system(const DoFSystem<dim, Number> &dof_system,
                      NewtonStepSystem &nonlinear_system,
                      const bool reset_solution=true);

    void prepare_output_results(DataOut<dim> &data_out,
                                const DoFSystem<dim, Number> &dof_system,
                                const NewtonStepSystem &nonlinear_system,
                                const DoFSystem<dim, Number> &thermal_dof_system,
                                const NewtonStepSystem &thermal_nonlinear_system,
                                const DoFSystem<dim, Number> &mesh_motion_dof_system,
                                const NewtonStepSystem &mesh_motion_nonlinear_system) const;

    template <typename TriangulationType>
    void write_output_results(DataOut<dim> &data_out,
                              const TriangulationType &tria,
                              const std::string &filename_base) const;

    void get_plastic_strain(
      TrilinosWrappers::MPI::Vector &plastic_strain,
      const DoFHandler<dim> &discontinuous_dof_handler,
      const Material<dim+1, Number> &material,
      const std::vector< MixedFEProjector<dim, Number> > &qp_values_projectors);

    void get_pressure(
      TrilinosWrappers::MPI::Vector &pressure,
      const DoFHandler<dim> &mixed_fe_dof_handler,
      TrilinosWrappers::MPI::Vector &von_mises_stress,
      const DoFHandler<dim> &discontinuous_dof_handler,
      NewtonStepSystem &Newton_system,
      const DoFSystem<dim, Number> &mechanical_dof_system,
      const NewtonStepSystem &thermal_Newton_system,
      const DoFSystem<dim, Number> &thermal_dof_system,
      Material<dim+1, Number> &material,
      const std::vector< MixedFEProjector<dim, Number> > &mixed_fe_projector,
      const std::vector< MixedFEProjector<dim, Number> > &qp_values_projectors);

    void solve_mechanical_step(int time_step);
    void solve_thermal_step(int time_step);

    void solve_mesh_motion_step(
          NewtonStepSystem &mesh_motion_nonlinear_system,
          const DoFSystem<dim, Number> &mesh_motion_dof_system,
          const LBCSystem<dim, Number, dim> &mesh_motion_lbc_system,
          const NewtonStepSystem &deformation_nonlinear_system,
          const DoFSystem<dim, Number> &deformation_dof_system,
          const DoFSystem<dim, Number> &mixed_fe_dof_system,
          const std::vector< MixedFEProjector<dim, Number> > &mixed_fe_projector,
          const int time_step);

    Tensor<2, dim+1, Number> get_deformation_gradient(
        const Tensor<2, dim, Number> &increment_gradient,
        const Number increment_0_over_rho) {
      Tensor<2, dim+1, Number> deformation_gradient = unit_symmetric_tensor<dim+1, Number>();
      for(unsigned int i=0; i<dim; ++i){
        for(unsigned int j=0; j<dim; ++j) {
          deformation_gradient[i][j] += increment_gradient[i][j];
        }
      }
      deformation_gradient[dim][dim] += increment_0_over_rho;
      return deformation_gradient;
    }

    Tensor<2, dim+1, Number> postprocess_tensor_dimension(
        const Tensor<2, dim, Number> &dimension_short_tensor,
        const Number entry_0_over_rho) {
      Tensor<2, dim+1, Number> postprocessed_tensor;
      for(unsigned int i=0; i<dim; ++i){
        for(unsigned int j=0; j<dim; ++j) {
          postprocessed_tensor[i][j] = dimension_short_tensor[i][j];
        }
      }
      postprocessed_tensor[dim][dim] = entry_0_over_rho;
      return postprocessed_tensor;
    }

    Tensor<1, dim+1, Number> postprocess_tensor_dimension(
          const Tensor<1, dim, Number> &dimension_short_tensor,
          const Number entry_at_dim=static_cast<Number>(0.0)) {
      Tensor<1, dim+1, Number> postprocessed_tensor;
      for(unsigned int i=0; i<dim; ++i){
        postprocessed_tensor[i] += dimension_short_tensor[i];
      }
      postprocessed_tensor[dim] = entry_at_dim;
      return postprocessed_tensor;
    }

    Tensor<1, dim+1, Number> scalar_to_angular_tensor(const Number angular_value) {
      Tensor<1, dim+1, Number> postprocessed_tensor = Tensor<1, dim+1, Number>();
      postprocessed_tensor[dim] = angular_value;
      return postprocessed_tensor;
    }

    Tensor<2, dim+1, Number> order_1_tensor_to_angular_gradient(
        const Tensor<1, dim, Number> &in_plane_gradient,
        const Number minus_entry_over_rho) {
      Tensor<2, dim+1, Number> postprocessed_tensor;
      for(unsigned int i=0; i<dim; ++i) {
        postprocessed_tensor[dim][i] = in_plane_gradient[i];
      }
      postprocessed_tensor[0][dim] = minus_entry_over_rho;
      return postprocessed_tensor;
    }

    Tensor<2, dim+1, Number> deformation_gradient_from_angular_displacement_gradient(
        const Number angular_displacement,
        const Tensor<1, dim, Number> &angular_displacement_gradient,
        const Number radius
      ) {
      if(2 != dim) {
        throw std::logic_error("Radial deformation gradient not implemented for dim!=2");
      }

      Tensor<2, dim+1, Number> result;
      const Number theta = angular_displacement / radius;
      Tensor<1, dim, Number> angular_displacmenet_over_r_squared;
      angular_displacmenet_over_r_squared[0] = angular_displacement / (radius * radius);
      const Tensor<1, dim, Number> theta_gradient = angular_displacement_gradient / radius - angular_displacmenet_over_r_squared;

      result[0][0] = std::cos(theta) - radius * std::sin(theta) * theta_gradient[0];
      result[0][1] = - radius * std::sin(theta) * theta_gradient[1];
      result[0][2] = -std::sin(theta);

      result[1][0] = 0;
      result[1][1] = 1;
      result[1][2] = 0;

      result[2][0] = std::sin(theta) + radius * std::cos(theta) * theta_gradient[0];
      result[2][1] = radius * std::cos(theta) * theta_gradient[1];
      result[2][2] = std::cos(theta);

      return result;
    }

    Tensor<2, dim+1, Number> deformation_gradient_from_angular_displacement_gradient_variations(
        const Number angular_displacement,
        const Tensor<1, dim, Number> &angular_displacement_gradient,
        const Number radius,
        const Number angular_displacement_variation,
        const Tensor<1, dim, Number> &angular_displacement_gradient_variation
      ) {
      if(2 != dim) {
        throw std::logic_error("Radial deformation gradient not implemented for dim!=2");
      }

      Tensor<2, dim+1, Number> result;
      const Number theta = angular_displacement / radius;
      const Number theta_variation = angular_displacement_variation / radius;

      Tensor<1, dim, Number> angular_displacmenet_over_r_squared;
      angular_displacmenet_over_r_squared[0] = angular_displacement / (radius * radius);
      const Tensor<1, dim, Number> theta_gradient = angular_displacement_gradient / radius - angular_displacmenet_over_r_squared;

      Tensor<1, dim, Number> angular_displacmenet_variation_over_r_squared;
      angular_displacmenet_variation_over_r_squared[0] = angular_displacement_variation / (radius * radius);
      const Tensor<1, dim, Number> theta_gradient_variation = angular_displacement_gradient_variation / radius - angular_displacmenet_variation_over_r_squared;

      result[0][0] = - std::sin(theta) * theta_variation
                     - radius * std::cos(theta) * theta_variation * theta_gradient[0]
                     - radius * std::sin(theta) * theta_gradient_variation[0];

      result[0][1] = - radius * std::cos(theta) * theta_variation * theta_gradient[1]
                     - radius * std::sin(theta) * theta_gradient_variation[1];

      result[0][2] = -std::cos(theta) * theta_variation;

      result[1][0] = 0;
      result[1][1] = 0;
      result[1][2] = 0;

      result[2][0] = std::cos(theta) * theta_variation
                     - radius * std::sin(theta) * theta_variation * theta_gradient[0]
                     + radius * std::cos(theta) * theta_gradient_variation[0];

      result[2][1] = - radius * std::sin(theta) * theta_variation * theta_gradient[1]
                     + radius * std::cos(theta) * theta_gradient_variation[1];

      result[2][2] = - std::sin(theta) * theta_variation;

      return result;
    }

    Tensor<2, dim+1, Number> rotation_tensor_to_transform_B_e(
        const Number angular_displacement,
        const Number radius
      ) {
      if(2 != dim) {
        throw std::logic_error("Radial deformation gradient not implemented for dim!=2");
      }

      Tensor<2, dim+1, Number> result;
      const Number theta = angular_displacement / radius;

      result[0][0] = std::cos(theta);
      result[0][1] = 0;
      result[0][2] = -std::sin(theta);

      result[1][0] = 0;
      result[1][1] = 1;
      result[1][2] = 0;

      result[2][0] = std::sin(theta);
      result[2][1] = 0;
      result[2][2] = std::cos(theta);

      return result;
    }

    Tensor<2, dim+1, Number> rotation_tensor_variation_to_transform_B_e(
        const Number angular_displacement,
        const Number angular_displacement_variation,
        const Number radius
      ) {
      if(2 != dim) {
        throw std::logic_error("Radial deformation gradient not implemented for dim!=2");
      }

      Tensor<2, dim+1, Number> result;
      const Number theta = angular_displacement / radius;
      const Number theta_variation = angular_displacement_variation / radius;

      result[0][0] = -std::sin(theta) * theta_variation;
      result[0][1] = 0;
      result[0][2] = -std::cos(theta) * theta_variation;

      result[1][0] = 0;
      result[1][1] = 0;
      result[1][2] = 0;

      result[2][0] = std::cos(theta) * theta_variation;
      result[2][1] = 0;
      result[2][2] = -std::sin(theta) * theta_variation;

      return result;
    }

    MPI_Comm mpi_communicator;
    ConditionalOStream pcout;

    const Number      order;
    FESystem<dim>     mech_fe;
    FE_Q<dim>         therm_fe;
    FE_DGP<dim>       mixed_var_fe;
    FESystem<dim>     mesh_motion_fe;
    MappingQ<dim>     mapping;

    parallel::distributed::Triangulation<dim> triangulation;

    DoFSystem<dim, Number> mech_dof_system;
    DoFSystem<dim, Number> therm_dof_system;
    DoFSystem<dim, Number> mixed_fe_dof_system;

    LBCSystem<dim, Number, dim+1> mech_lbc_system;
    LBCSystem<dim, Number, 1> therm_lbc_system;

    NewtonStepSystem mech_nonlinear_system;
    NewtonStepSystem therm_nonlinear_system;

    NewtonStepSystem mesh_motion_nonlinear_system;
    DoFSystem<dim, Number> mesh_motion_dof_system;
    LBCSystem<dim, Number, dim> mesh_motion_lbc_system;

    NewtonStepSystem deformation_remapping_nonlinear_system;

    QGauss<dim>      quadrature_formula;
    QGauss<dim-1>    face_quadrature_formula;

    Material<dim+1, Number>  &material;

    std::vector< MixedFEProjector<dim, Number> > mixed_FE_projectors;

    std::unordered_map<size_t, Tensor<1, dim+1, Number>> material_area_factors;

    Number time_increment = 1.0e-01; /*0.5e-6;*/ // [s]
    unsigned int output_rate = 1;
    Number time_since_start = 0;

    const Number ambient_temperature = 293.0; // [K]
    const Number rho_infty = 0.0;

    const unsigned int surface_boundary_id = 2;

    const bool COMPUTE_FORCES_PER_UNIT_AREA_IN_CURRENT_CONFIGURATION = false;
    const bool use_sigmoid_friction_law = true;

    Number global_lagrangian_penalty_factor = 1.0;

  };

  struct NewtonIterationDivergenceException : std::exception {
    const char *what() const _GLIBCXX_USE_NOEXCEPT {
      return "Newton step solution diverged!\n";
    }
  };

} /*namespace PlasticityLab*/

#endif /* PLASTICITYLABPROG_H_ */
