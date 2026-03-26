#include <sstream>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>


#include <deal.II/fe/fe_dgq.h>

#include "RotationFunction.h"
#include "ScaleZFunction.h"
#include "ScaleComponentFunction.h"
#include "PlasticityLabProg.h"

using namespace dealii;
using std::endl;

namespace PlasticityLab {

  template <int dim, typename Number>
  void PlasticityLabProg<dim, Number>::run() {
    // make_grid_();
    // make_ball_in_hypershell_grid(
    // make_cylindrical_grid(
    // make_cylindrical_impact_grid(
    make_necking_grid(
      triangulation,
      mech_lbc_system,
      therm_lbc_system,
      3);
    //   triangulation,
    //   mech_lbc_system,
    //   therm_lbc_system,
    //   3);
    // make_hook_membrane_grid(1);

    set_mesh_motion_LBCs(triangulation, mesh_motion_lbc_system);

    const Number total_elongation = 8.0; // mm
    const Number elongation_rate = 1.0; // [mm/s]
    const unsigned int n_steps = ceil(total_elongation/(elongation_rate*time_increment));

    mech_dof_system.setup_dof_system(mech_fe);
    mech_lbc_system.apply_constraints(mech_dof_system);
    mech_nonlinear_system.setup(mech_dof_system);

    therm_dof_system.setup_dof_system(therm_fe);
    therm_lbc_system.apply_constraints(therm_dof_system);
    therm_nonlinear_system.setup(therm_dof_system);

    therm_nonlinear_system.previous_deformation = ambient_temperature;

    mixed_fe_dof_system.setup_dof_system(mixed_var_fe);

    mesh_motion_dof_system.setup_dof_system(mesh_motion_fe);
    mesh_motion_lbc_system.apply_constraints(mesh_motion_dof_system);
    mesh_motion_nonlinear_system.setup(mesh_motion_dof_system);

    deformation_remapping_nonlinear_system.setup(mech_dof_system);

    setup_material_data(triangulation, material);
    setup_material_area_factors(mesh_motion_dof_system, material_area_factors);

    setup_mixed_fe_projection_data(
      triangulation, mixed_FE_projectors,
      mixed_var_fe, quadrature_formula);

    std::vector< MixedFEProjector<dim, Number> > discontinuous_projectors;
    FE_DGQ<dim> discontinuous_fe(1);

    setup_mixed_fe_projection_data(
      triangulation, discontinuous_projectors,
      discontinuous_fe, quadrature_formula);

    TrilinosWrappers::MPI::Vector pressure(
      mixed_fe_dof_system.locally_owned_dofs,
      mpi_communicator);

    DoFSystem<dim, Number> discontinuous_dof_system(triangulation, mapping);
    discontinuous_dof_system.setup_dof_system(discontinuous_fe);
    TrilinosWrappers::MPI::Vector plastic_strain(
      discontinuous_dof_system.locally_owned_dofs,
      mpi_communicator);
    TrilinosWrappers::MPI::Vector von_mises_stress(
      discontinuous_dof_system.locally_owned_dofs,
      mpi_communicator);

    std::vector< MixedFEProjector<dim, Number> > mech_fe_projectors;

    setup_mixed_fe_projection_data(
      triangulation, mech_fe_projectors,
      mech_fe, quadrature_formula);



    {
      TrilinosWrappers::MPI::Vector initial_velocity(
        mech_dof_system.locally_owned_dofs,
        mpi_communicator);

      MPI_Barrier(mpi_communicator);

      for(const auto initial_velocity_interpolation_handler: mech_lbc_system.initial_velocity_interpolation_handlers) {
        initial_velocity_interpolation_handler->interpolate(initial_velocity, mech_dof_system);
      }

      mech_nonlinear_system.previous_time_derivative = initial_velocity;
      mech_nonlinear_system.previous_time_derivative.compress(
        VectorOperation::insert);
    }


    {
      TrilinosWrappers::MPI::Vector initial_deformation(
        mech_dof_system.locally_owned_dofs,
        mpi_communicator);

      MPI_Barrier(mpi_communicator);

      for(const auto initial_deformation_interpolation_handler: mech_lbc_system.initial_deformation_interpolation_handlers) {
        initial_deformation_interpolation_handler->interpolate(initial_deformation, mech_dof_system);
      }

      mech_nonlinear_system.previous_deformation = initial_deformation;
      mech_nonlinear_system.previous_deformation.compress(
        VectorOperation::insert);
    }


    for (unsigned int timeStep = 0; timeStep < n_steps + 1; ++timeStep) {
      for(const auto increment_interpolation_handler: mech_lbc_system.increment_interpolation_handlers) {
        increment_interpolation_handler->advance_time(time_increment);
      }

      get_plastic_strain(
        plastic_strain,
        discontinuous_dof_system.dof_handler,
        material,
        discontinuous_projectors);

      get_pressure(
        pressure,
        mixed_fe_dof_system.dof_handler,
        von_mises_stress,
        discontinuous_dof_system.dof_handler,
        mech_nonlinear_system,
        mech_dof_system,
        therm_nonlinear_system,
        therm_dof_system,
        material,
        mixed_FE_projectors,
        discontinuous_projectors);

      if(0==timeStep % output_rate) {
        pcout << "\nOutputting results..." << endl;
        DataOut<dim> data_out;

        data_out.add_data_vector(
          discontinuous_dof_system.dof_handler,
          plastic_strain,
          "plastic_strain");
        data_out.build_patches();

        data_out.add_data_vector(
          mixed_fe_dof_system.dof_handler,
          pressure,
          "pressure");
        data_out.build_patches();

        data_out.add_data_vector(
          discontinuous_dof_system.dof_handler,
          von_mises_stress,
          "von_mises_stress");

        prepare_output_results(
          data_out,
          mech_dof_system,
          mech_nonlinear_system,
          therm_dof_system,
          therm_nonlinear_system,
          mesh_motion_dof_system,
          mesh_motion_nonlinear_system);

        std::ostringstream oss;
        oss << "step_" << timeStep;
        const std::string output_name = oss.str();
        write_output_results(data_out, triangulation, output_name);
      }

      if (timeStep == n_steps) break;

      pcout << "\n\nStarting time step " << timeStep << ":\n\n" << endl;

      {
        TrilinosWrappers::MPI::Vector   step_increment(
          mech_dof_system.locally_owned_dofs,
          mpi_communicator);

        MPI_Barrier(mpi_communicator);

        for(const auto increment_interpolation_handler: mech_lbc_system.increment_interpolation_handlers) {
          increment_interpolation_handler->interpolate(step_increment, mech_dof_system);
        }

        mech_nonlinear_system.current_increment = step_increment;
        mech_nonlinear_system.current_increment.compress(
          VectorOperation::insert);
      }

      solve_mesh_motion_step(
        mesh_motion_nonlinear_system,
        mesh_motion_dof_system,
        mesh_motion_lbc_system,
        mech_nonlinear_system,
        mech_dof_system,
        mixed_fe_dof_system,
        mixed_FE_projectors,
        timeStep);

      mesh_motion_nonlinear_system.advance_time(time_increment, rho_infty, false);
      mesh_motion_nonlinear_system.previous_deformation = mesh_motion_nonlinear_system.current_increment;
      mesh_motion_nonlinear_system.previous_deformation *= -1;

      std::unordered_map<point_index_t, Tensor<2, dim+1, Number>> remapped_deformation_gradients;
      remap_material_state_variables(
        mesh_motion_nonlinear_system,
        mesh_motion_dof_system,
        mech_nonlinear_system,
        mech_dof_system,
        mixed_fe_dof_system,
        mixed_FE_projectors,
        material,
        remapped_deformation_gradients);

      remap_thermal_field(
        therm_nonlinear_system,
        therm_dof_system,
        mesh_motion_nonlinear_system,
        mesh_motion_dof_system);

      remap_mechanical_fields(
        mech_nonlinear_system,
        mech_dof_system,
        mesh_motion_nonlinear_system,
        mesh_motion_dof_system);

      update_material_area_factors(
        mesh_motion_nonlinear_system,
        mesh_motion_dof_system,
        material_area_factors);

      solve_mechanical_step(timeStep);

      solve_thermal_step(timeStep);

      solve_mechanical_step(timeStep);

      // udpate material state
      pcout << "\n\t\tassembling mechanical system updating material state..." << endl;
      assemble_mechanical_system(
        mech_nonlinear_system,
        mech_dof_system,
        mech_lbc_system,
        mesh_motion_nonlinear_system,
        mesh_motion_dof_system,
        therm_nonlinear_system,
        therm_dof_system,
        mixed_fe_dof_system,
        material,
        mixed_FE_projectors,
        false,
        true);

      mech_nonlinear_system.advance_time(time_increment, rho_infty, true);
      therm_nonlinear_system.previous_deformation += therm_nonlinear_system.current_increment;
      therm_nonlinear_system.previous_deformation.compress(VectorOperation::add);

      therm_nonlinear_system.current_increment = 0;
      therm_nonlinear_system.current_increment.compress(VectorOperation::insert);

      pcout << "Next timestep..." << std::endl;

    } /*for(timeStep)*/
  }

  template<int dim, typename Number>
  void PlasticityLabProg<dim, Number>::solve_mechanical_step(int time_step) {

    TrilinosWrappers::MPI::Vector total_residual;

    for (unsigned int NewtonStep = 0; true; NewtonStep++) {

      pcout << "\n\ttime step " << time_step
            << ", Newton step " << NewtonStep << "..."
            << "\n\t\tassembling mechanical system with tangents..." << endl;

      assemble_mechanical_system(
        mech_nonlinear_system,
        mech_dof_system,
        mech_lbc_system,
        mesh_motion_nonlinear_system,
        mesh_motion_dof_system,
        therm_nonlinear_system,
        therm_dof_system,
        mixed_fe_dof_system,
        material,
        mixed_FE_projectors,
        true);

      total_residual = mech_nonlinear_system.Newton_step_residual;
      total_residual.compress(VectorOperation::insert);

      pcout << "-------------------------------------------------------------------" << endl;

      pcout << "Normalized system residual: "
            << std::sqrt(total_residual.norm_sqr())
            << " ..." << endl;
      pcout << "-------------------------------------------------------------------" << endl;

      if (std::sqrt(total_residual.norm_sqr()) <= 1e-5) {
        break;
      }

      const Number old_residual = total_residual.norm_sqr();
      Number previous_residual = old_residual;

      pcout << "solving system..." << endl;
      try {
        solve_system(mech_dof_system, mech_nonlinear_system);
      } catch (...) {
        if (std::isnan(mech_nonlinear_system.Newton_step_solution.norm_sqr())) {
          throw;
        }
        pcout << "System solution falied. Continuing with partial solution..." << endl;
      }
      mech_dof_system.nodal_constraints.distribute(
        mech_nonlinear_system.Newton_step_solution);
      const Number solution_norm = std::sqrt(
                                     mech_nonlinear_system.Newton_step_solution.norm_sqr());
      TrilinosWrappers::MPI::Vector full_step_increment(mech_nonlinear_system.current_increment);
      TrilinosWrappers::MPI::Vector temp_locally_owned_increment(mech_dof_system.locally_owned_dofs);

      Number clip_factor =
        (solution_norm <= std::sqrt(old_residual)) ?
        1.0 : sqrt(old_residual) / solution_norm;
      if (clip_factor < 1.0) pcout << "clip factor: " << clip_factor << endl;

      pcout << "doing line search..." << endl;
      bool hit_line_search_limit = false;
      for (unsigned int i = 0; true/*i < 18*/; ++i) {
        const Number alpha = std::pow(0.5, static_cast<Number>(i));
        if (i > 5 && clip_factor * alpha * solution_norm < 1e-1) {
          hit_line_search_limit = true;
          break;
        }
        if (i > 0) pcout << "\tline search step " << i << "..." << endl;

        while (true) {
          temp_locally_owned_increment = full_step_increment;
          temp_locally_owned_increment.sadd(1, -alpha * clip_factor, mech_nonlinear_system.Newton_step_solution);
          temp_locally_owned_increment.compress(VectorOperation::insert);
          mech_nonlinear_system.current_increment = temp_locally_owned_increment;
          mech_nonlinear_system.current_increment.compress(VectorOperation::insert);

          try {
            assemble_mechanical_system(
              mech_nonlinear_system,
              mech_dof_system,
              mech_lbc_system,
              mesh_motion_nonlinear_system,
              mesh_motion_dof_system,
              therm_nonlinear_system,
              therm_dof_system,
              mixed_fe_dof_system,
              material,
              mixed_FE_projectors,
              false);
          } catch (const std::runtime_error &) {
            clip_factor *= 0.125;  // using a power of 0.5; better for binary arithmetic
            pcout << "\t-------------------------------------------------------------------" << endl;
            pcout << "\tDeformation too large, causing degenerate mesh..." << endl;
            pcout << "\tupdated clip factor: " << clip_factor << endl;
            pcout << "\t-------------------------------------------------------------------" << endl;
            continue;
          }
          break;
        }

        total_residual = mech_nonlinear_system.Newton_step_residual;
        total_residual.compress(VectorOperation::insert);

        const Number current_residual = total_residual.norm_sqr();

        pcout << "\t-------------------------------------------------------------------" << endl;

        pcout << "\tNormalized system residual: "
              << std::sqrt(current_residual)
              << " ..." << endl;
        pcout << "\t-------------------------------------------------------------------" << endl;

        if (previous_residual < old_residual and current_residual >= previous_residual) {
          pcout << "\t---Accepting previous residual: " << std::sqrt(previous_residual)
                << " ..." << endl;

          temp_locally_owned_increment = full_step_increment;
          temp_locally_owned_increment.sadd(1, -2 * alpha * clip_factor, mech_nonlinear_system.Newton_step_solution);
          temp_locally_owned_increment.compress(VectorOperation::insert);
          mech_nonlinear_system.current_increment = temp_locally_owned_increment;
          mech_nonlinear_system.current_increment.compress(VectorOperation::insert);
          break;
        }
        previous_residual = current_residual;
      }

    }
  }


  template<int dim, typename Number>
  void PlasticityLabProg<dim, Number>::solve_thermal_step(int time_step) {

      TrilinosWrappers::MPI::Vector total_therm_residual;
      const Number starting_thermal_residual_squared_norm = 1.0;

      for (unsigned int NewtonStep = 0; true; NewtonStep++) {

        pcout << "\n\ttime step " << time_step
              << ", Newton step " << NewtonStep << "..."
              << "\n\t\tassembling thermal system with tangents..." << endl;

        assemble_thermal_system(
          therm_nonlinear_system,
          mech_nonlinear_system,
          therm_dof_system,
          therm_lbc_system,
          mech_dof_system,
          mixed_fe_dof_system,
          material,
          mixed_FE_projectors,
          material_area_factors,
          true);

        total_therm_residual = therm_nonlinear_system.Newton_step_residual;
        total_therm_residual.compress(VectorOperation::insert);

        pcout << "-------------------------------------------------------------------" << endl;
        pcout << "Normalized system residual (contactor): "
              << std::sqrt(total_therm_residual.norm_sqr()
                           / starting_thermal_residual_squared_norm)
              << " ..." << endl;
        pcout << "-------------------------------------------------------------------" << endl;

        if (std::sqrt(total_therm_residual.norm_sqr()
                      / starting_thermal_residual_squared_norm) <= 1e-6) {
          break;
        }

        const Number old_residual = total_therm_residual.norm_sqr();

        pcout << "solving system..." << endl;

        solve_system(therm_dof_system, therm_nonlinear_system);

        therm_dof_system.nodal_constraints.distribute(therm_nonlinear_system.Newton_step_solution);
        TrilinosWrappers::MPI::Vector full_step_increment(therm_nonlinear_system.current_increment);

        TrilinosWrappers::MPI::Vector temp_locally_owned_increment(therm_dof_system.locally_owned_dofs);


        pcout << "doing line search..." << endl;
        for (unsigned int i = 0; i < (NewtonStep > 0 ? 6 : 1); ++i) {
          const Number alpha = std::pow(0.5, static_cast<Number>(i));

          mech_nonlinear_system.current_increment.compress(VectorOperation::insert);

          temp_locally_owned_increment = full_step_increment;
          temp_locally_owned_increment.sadd(1, -alpha, therm_nonlinear_system.Newton_step_solution);
          therm_dof_system.nodal_constraints.distribute(temp_locally_owned_increment);
          temp_locally_owned_increment.compress(VectorOperation::insert);
          therm_nonlinear_system.current_increment = temp_locally_owned_increment;
          therm_nonlinear_system.current_increment.compress(VectorOperation::insert);

          assemble_thermal_system(
            therm_nonlinear_system,
            mech_nonlinear_system,
            therm_dof_system,
            therm_lbc_system,
            mech_dof_system,
            mixed_fe_dof_system,
            material,
            mixed_FE_projectors,
            material_area_factors,
            false);

          total_therm_residual = therm_nonlinear_system.Newton_step_residual;
          total_therm_residual.compress(VectorOperation::insert);

          const Number current_residual = total_therm_residual.norm_sqr();
          if (current_residual < old_residual)
            break;
        }
      }
      pcout << endl;
  }


  template<int dim, typename Number>
  void PlasticityLabProg<dim, Number>::solve_mesh_motion_step(
          NewtonStepSystem &mesh_motion_nonlinear_system,
          const DoFSystem<dim, Number> &mesh_motion_dof_system,
          const LBCSystem<dim, Number, dim> &mesh_motion_lbc_system,
          const NewtonStepSystem &deformation_nonlinear_system,
          const DoFSystem<dim, Number> &deformation_dof_system,
          const DoFSystem<dim, Number> &mixed_fe_dof_system,
          const std::vector< MixedFEProjector<dim, Number> > &mixed_fe_projector,
          const int time_step) {

    TrilinosWrappers::MPI::Vector total_residual;

    for (unsigned int NewtonStep = 0; true; NewtonStep++) {

      pcout << "\n\ttime step " << time_step
            << ", Newton step " << NewtonStep << "..."
            << "\n\t\tassembling mesh motion system with tangents..." << endl;

      assemble_mesh_motion_system(
        mesh_motion_nonlinear_system,
        mesh_motion_dof_system,
        mesh_motion_lbc_system,
        deformation_nonlinear_system,
        deformation_dof_system,
        mixed_fe_dof_system,
        mixed_fe_projector,
        true);

      total_residual = mesh_motion_nonlinear_system.Newton_step_residual;
      total_residual.compress(VectorOperation::insert);

      pcout << "-------------------------------------------------------------------" << endl;

      pcout << "Normalized system residual: "
            << std::sqrt(total_residual.norm_sqr())
            << " ..." << endl;
      pcout << "-------------------------------------------------------------------" << endl;

      if (std::sqrt(total_residual.norm_sqr()) <= 1e-4) {
        break;
      }

      const Number old_residual = total_residual.norm_sqr();
      Number previous_residual = old_residual;

      pcout << "solving system..." << endl;
      try {
        solve_system(mesh_motion_dof_system, mesh_motion_nonlinear_system);
      } catch (...) {
        if (std::isnan(mesh_motion_nonlinear_system.Newton_step_solution.norm_sqr())) {
          throw;
        }
        pcout << "System solution falied. Continuing with partial solution..." << endl;
      }
      mesh_motion_dof_system.nodal_constraints.distribute(mesh_motion_nonlinear_system.Newton_step_solution);
      const Number solution_norm = std::sqrt(mesh_motion_nonlinear_system.Newton_step_solution.norm_sqr());
      TrilinosWrappers::MPI::Vector full_step_increment(mesh_motion_nonlinear_system.current_increment);
      TrilinosWrappers::MPI::Vector temp_locally_owned_increment(mesh_motion_dof_system.locally_owned_dofs);

      Number clip_factor =
        (solution_norm <= std::sqrt(old_residual)) ? 1.0 : sqrt(old_residual) / solution_norm;
      if (clip_factor < 1.0) pcout << "clip factor: " << clip_factor << endl;

      pcout << "doing line search..." << endl;
      bool hit_line_search_limit = false;
      for (unsigned int i = 0; true/*i < 18*/; ++i) {
        const Number alpha = std::pow(0.5, static_cast<Number>(i));
        if (i > 5 && clip_factor * alpha * solution_norm < 1e-1) {
          hit_line_search_limit = true;
          break;
        }
        if (i > 0) pcout << "\tline search step " << i << "..." << endl;

        while (true) {
          temp_locally_owned_increment = full_step_increment;
          temp_locally_owned_increment.sadd(1, -alpha * clip_factor, mesh_motion_nonlinear_system.Newton_step_solution);
          temp_locally_owned_increment.compress(VectorOperation::insert);
          mesh_motion_nonlinear_system.current_increment = temp_locally_owned_increment;
          mesh_motion_nonlinear_system.current_increment.compress(VectorOperation::insert);

          try {
            assemble_mesh_motion_system(
              mesh_motion_nonlinear_system,
              mesh_motion_dof_system,
              mesh_motion_lbc_system,
              deformation_nonlinear_system,
              deformation_dof_system,
              mixed_fe_dof_system,
              mixed_fe_projector,
              false);
          } catch (const std::runtime_error &) {
            clip_factor *= 0.125;  // using a power of 0.5; better for binary arithmetic
            pcout << "\t-------------------------------------------------------------------" << endl;
            pcout << "\tDeformation too large, causing degenerate mesh..." << endl;
            pcout << "\tupdated clip factor: " << clip_factor << endl;
            pcout << "\t-------------------------------------------------------------------" << endl;
            continue;
          }
          break;
        }

        total_residual = mesh_motion_nonlinear_system.Newton_step_residual;
        total_residual.compress(VectorOperation::insert);

        const Number current_residual = total_residual.norm_sqr();

        pcout << "\t-------------------------------------------------------------------" << endl;

        pcout << "\tNormalized system residual: "
              << std::sqrt(current_residual)
              << " ..." << endl;
        pcout << "\t-------------------------------------------------------------------" << endl;

        if (previous_residual < old_residual and current_residual >= previous_residual) {
          pcout << "\t---Accepting previous residual: " << std::sqrt(previous_residual)
                << " ..." << endl;

          temp_locally_owned_increment = full_step_increment;
          temp_locally_owned_increment.sadd(1, -2 * alpha * clip_factor, mesh_motion_nonlinear_system.Newton_step_solution);
          temp_locally_owned_increment.compress(VectorOperation::insert);
          mesh_motion_nonlinear_system.current_increment = temp_locally_owned_increment;
          mesh_motion_nonlinear_system.current_increment.compress(VectorOperation::insert);
          break;
        }
        previous_residual = current_residual;
      }
    }
  }


  template <int dim>
  struct RefiningTransform
  {
    RefiningTransform(
      double height,
      double refining_fraction,
      double base=0,
      size_t dimension=1) :
        height(height),
        refining_fraction(refining_fraction),
        base(base),
        dimension(dimension) {}

    Point<dim> operator()(const Point<dim> &p) const
    {
      Point<dim> q = p;
      if ((p[dimension]-base)/(height-base) <= 0.5) {
        q[dimension] = base + refining_fraction/0.5 * (p[dimension]-base);
      } else if ((p[dimension]-base)/(height-base) > 0.5) {
        q[dimension] = base + refining_fraction * (height - base) + (1.0 - refining_fraction) / 0.5 * (p[dimension] - 0.5 * (height + base));
      }
      return q;
    }

    double height;
    double refining_fraction;
    double base;
    size_t dimension;
  };


  template <int dim, typename Number>
  void PlasticityLabProg<dim, Number>::set_mesh_motion_LBCs(
      Triangulation<dim> &triangulation,
      LBCSystem<dim, Number, dim> &mesh_motion_lbc_system) {

    std::set<types::boundary_id> all_boundary_ids;
    for(types::boundary_id id: triangulation.get_boundary_ids()) {
      all_boundary_ids.insert(id);
    }

    mesh_motion_lbc_system.no_normal_flux_constraints.push_back(std::make_pair(0, all_boundary_ids));
  }


  template <int dim, typename Number>
  void PlasticityLabProg<dim, Number>::make_cylindrical_grid(
    Triangulation<dim> &triangulation,
    LBCSystem<dim, Number, dim+1> &mech_lbc_system,
    LBCSystem<dim, Number, 1> &therm_lbc_system,
    int n_initial_global_refinements) {

    const Number initial_velocity = 1.9e5; // [mm/s]
    const Number height = 2.5*25.4; // [mm]
    const Number inner_radius = 12.5; // [mm]
    const Number radius = 12.5; // [mm]

    const Number top_coordinate = height/2;
    const Number base_coordinate = 0.0;

    const unsigned int base_repetitions = std::pow(2, n_initial_global_refinements);
    const unsigned int aspect_ratio = std::ceil(0.25 * height / radius);

    GridGenerator::subdivided_hyper_rectangle(
      triangulation,
      std::vector<unsigned int> {base_repetitions, aspect_ratio * base_repetitions},
      Point<dim>(inner_radius, base_coordinate),
      Point<dim>(inner_radius + radius, top_coordinate),
      true
    );

    for (auto &cell: triangulation.active_cell_iterators()) {
      for (const auto &face : cell->face_iterators()) {
        if(face->boundary_id() == 0 || face->boundary_id() == 1) {
          if(face->center()[1] > 0.95 * height/2) {
            face->set_boundary_id(4);
          }
        }
      }
    }

    // GridTools::transform(RefiningTransform<dim>(top_coordinate, 0.3, base_coordinate), triangulation);
    // GridTools::transform(RefiningTransform<dim>(top_coordinate, 0.4, base_coordinate), triangulation);
    // GridTools::transform(RefiningTransform<dim>/(inner_radius, 0.35, inner_radius + radius, 0), triangulation);

    // for(unsigned int i=0; i<2; i++) {
    //   for (auto &cell : triangulation.active_cell_iterators()) {
    //     for (const auto &face : cell->face_iterators()) {
    //       if (face->boundary_id() == 2) {
    //         cell->set_refine_flag();
    //         break;
    //       }
    //     }
    //   }
    //   triangulation.execute_coarsening_and_refinement();
    // }

    ComponentMask x_component_mask(dim+1, false);
    x_component_mask.set(0, true);
    ComponentMask y_component_mask(dim+1, false);
    y_component_mask.set(1, true);
    ComponentMask x_and_y_component_mask(dim+1, false);
    x_and_y_component_mask.set(0, true);
    x_and_y_component_mask.set(1, true);
    ComponentMask z_component_mask(dim+1, false);
    z_component_mask.set(dim-1, true);
    ComponentMask rho_component_mask(dim+1, false);
    rho_component_mask.set(dim, true);

    // std::map< types::boundary_id, const Function< dim, Number > * > base_constraint_function_map;
    // base_constraint_function_map.insert(
    //   std::pair<types::boundary_id, Function<dim, Number>*>(2, &mech_lbc_system.zero_function));
    // mech_lbc_system.interpolatoryConstraintAppliers.push_back(
    //   InterpolatoryConstraintApplier<dim, Number>(
    //     base_constraint_function_map,
    //     y_component_mask));

    std::map< types::boundary_id, const Function< dim, Number > * > top_constraint_function_map;
    top_constraint_function_map.insert(
      std::pair<types::boundary_id, Function<dim, Number>*>(3, &mech_lbc_system.zero_function));
    mech_lbc_system.interpolatoryConstraintAppliers.push_back(
      InterpolatoryConstraintApplier<dim, Number>(
        top_constraint_function_map,
        x_component_mask));

    std::map< types::boundary_id, const Function< dim, Number > * > clamp_constraint_function_map;
    clamp_constraint_function_map.insert(
      std::pair<types::boundary_id, Function<dim, Number>*>(4, &mech_lbc_system.zero_function));
    mech_lbc_system.interpolatoryConstraintAppliers.push_back(
      InterpolatoryConstraintApplier<dim, Number>(
        clamp_constraint_function_map,
        x_component_mask));

    if(std::abs(inner_radius) < 1e-16) {
      std::map< types::boundary_id, const Function< dim, Number > * > axial_constraint_function_map;
      axial_constraint_function_map.insert(
        std::pair<types::boundary_id, Function<dim, Number>*>(0, &mech_lbc_system.zero_function));
      mech_lbc_system.interpolatoryConstraintAppliers.push_back(
        InterpolatoryConstraintApplier<dim, Number>(
          axial_constraint_function_map,
          x_component_mask));

      std::map< types::boundary_id, const Function< dim, Number > * > axial_rotation_constraint_function_map;
      axial_rotation_constraint_function_map.insert(
        std::pair<types::boundary_id, Function<dim, Number>*>(0, &mech_lbc_system.zero_function));
      mech_lbc_system.interpolatoryConstraintAppliers.push_back(
        InterpolatoryConstraintApplier<dim, Number>(
          axial_rotation_constraint_function_map,
          rho_component_mask));
    }

    // therm_lbc_system.boundaryLoadAppliers.push_back(
    //   std::pair<int,BodyForceApplier<dim,Number> >(
    //       2, BodyForceApplier<dim,Number>(0, 22e0)));

    std::map< types::boundary_id, const Function< dim, Number > * > top_rotation_constraint_function_map;
    top_rotation_constraint_function_map.insert(
      std::pair<types::boundary_id, Function<dim, Number>*>(3, &mech_lbc_system.zero_function));
    mech_lbc_system.interpolatoryConstraintAppliers.push_back(
      InterpolatoryConstraintApplier<dim, Number>(
        top_rotation_constraint_function_map,
        rho_component_mask));

    // std::map< types::boundary_id, const Function< dim, Number > * > base_rotation_constraint_function_map;
    // base_rotation_constraint_function_map.insert(
    //   std::pair<types::boundary_id, Function<dim, Number>*>(2, &mech_lbc_system.zero_function));
    // mech_lbc_system.interpolatoryConstraintAppliers.push_back(
    //   InterpolatoryConstraintApplier<dim, Number>(
    //     base_rotation_constraint_function_map,
    //     rho_component_mask));

    // Thermal constraints
    const Number convection_coefficient = /*17.5e-6*/ 100e-6; // [J.mm^-2.s^-1.K^-1]
    therm_lbc_system.convection_BC_appliers.push_back(
      std::pair<int, ConvectionBoundaryConditionApplier<dim, Number> >(
        0,
        ConvectionBoundaryConditionApplier<dim, Number>(
          0, convection_coefficient, ambient_temperature)));
    therm_lbc_system.convection_BC_appliers.push_back(
      std::pair<int, ConvectionBoundaryConditionApplier<dim, Number> >(
        1,
        ConvectionBoundaryConditionApplier<dim, Number>(
          0, convection_coefficient, ambient_temperature)));
    therm_lbc_system.convection_BC_appliers.push_back(
      std::pair<int, ConvectionBoundaryConditionApplier<dim, Number> >(
        2,
        ConvectionBoundaryConditionApplier<dim, Number>(
          0, convection_coefficient, ambient_temperature)));
    therm_lbc_system.convection_BC_appliers.push_back(
      std::pair<int, ConvectionBoundaryConditionApplier<dim, Number> >(
        3,
        ConvectionBoundaryConditionApplier<dim, Number>(
          0, convection_coefficient, ambient_temperature)));

    // therm_lbc_system.convection_BC_appliers.push_back(
    //   std::pair<int, ConvectionBoundaryConditionApplier<dim, Number> >(
    //     2,
    //     ConvectionBoundaryConditionApplier<dim, Number>(
    //       0, 3000*convection_coefficient, 1350.0 /*a little less than melting*/)));

    // const Number total_elongation = 2 * 80.0; // mm
    // const Number elongation_rate = time_since_start < 6.0? 0.05 : 1.75; // [mm/s]
    // const unsigned int n_steps = ceil(total_elongation/(elongation_rate*time_increment));
    // auto scale_z_handler = new IncrementInterpolationHandler<dim, Number, dim+1>(
    //         new ConstantFunction<dim, Number>({0, -total_elongation/(static_cast<Number>(n_steps)/**height*/), 0}),
    //         true,
    //         y_component_mask,
    //         false,
    //         ComponentMask(dim+1, false),
    //         3,
    //         mapping);

    // mech_lbc_system.increment_interpolation_handlers.push_back(scale_z_handler);

    mech_lbc_system.boundaryLoadAppliers.push_back(
      std::pair<int,BodyForceApplier<dim,Number> >(
        3, BodyForceApplier<dim,Number>(dim-1, -40)));

    auto top_surface_unidirectional_penalty_spec = new BoundaryUnidirectionalPenaltySpec<Number>(3, 1e-2, 0, 1e4 / (time_increment * time_increment) );
    mech_lbc_system.boundary_unidirectional_penalty_specs.push_back(top_surface_unidirectional_penalty_spec);

    // auto constant_velocity_handler = new IncrementInterpolationHandler<dim, Number, dim+1>(
    //         new ScaleZFunction<dim, Number, dim+1>(-total_elongation/(static_cast<Number>(n_steps)*time_increment/**height*/), dim-1),
    //         true,
    //         y_component_mask,
    //         false,
    //         ComponentMask(dim+1, false),
    //         0,
    //         mapping);
    // mech_lbc_system.initial_velocity_interpolation_handlers.push_back(constant_velocity_handler);
    // std::map< types::boundary_id, const Function< dim, Number > * > top_constraint_function_map;
    // top_constraint_function_map.insert(
    //   std::pair<types::boundary_id, Function<dim, Number>*>(3, &mech_lbc_system.zero_function));
    // mech_lbc_system.interpolatoryConstraintAppliers.push_back(
    //   InterpolatoryConstraintApplier<dim, Number>(
    //     top_constraint_function_map,
    //     y_component_mask));

    const Number drive_speed = 0.8*314.159/3.0;  // [rad/s]

    auto scale_rotation_increment = new IncrementInterpolationHandler<dim, Number, dim+1>(
            new ScaleComponentFunction<dim, Number, dim+1>(drive_speed * time_increment, 0, dim),
            true,
            rho_component_mask,
            false,
            ComponentMask(dim+1, false),
            3,
            mapping);
    mech_lbc_system.increment_interpolation_handlers.push_back(scale_rotation_increment);

    auto constant_angular_velocity_handler = new IncrementInterpolationHandler<dim, Number, dim+1>(
            new ScaleComponentFunction<dim, Number, dim+1>(drive_speed, 0, dim),
            true,
            rho_component_mask,
            false,
            ComponentMask(dim+1, false),
            3,
            mapping);
    mech_lbc_system.initial_velocity_interpolation_handlers.push_back(constant_angular_velocity_handler);

    // mech_lbc_system.boundaryLoadAppliers.push_back(
    //  std::pair<int,BodyForceApplier<dim,Number> >(
    //     1, BodyForceApplier<dim,Number>(dim, 3.750e0)));

  } /* make_cylindrical_grid() */


  template <int dim, typename Number>
  void PlasticityLabProg<dim, Number>::make_grid_() {

  } /*make_grid_()*/

template <int dim>
struct InterferenceTaperTransform
{
  Point<dim> operator()(const Point<dim> &p) const
  {
    Point<dim> q = p;
    if (p[0]>=35 && p[0]<=300 && p[1]<=0 && p[1]>=-150)
    {
      q[0] += 5 * ((p[1] + 150) / 150) * ((p[0]-300) / (35-300));
    }
    return q;
  }
};

template <int dim>
struct NeckingTaperTransform
{
  double factor;
  NeckingTaperTransform(double factor) : factor(factor) {}

  Point<dim> operator()(const Point<dim> &p) const
  {
    Point<dim> q = p;
    q[0] += factor * q[0] * q[1];
    return q;
  }
};


  template <int dim, typename Number>
  void PlasticityLabProg<dim, Number>::make_necking_grid(
    Triangulation<dim> &triangulation,
    LBCSystem<dim, Number, dim+1> &mech_lbc_system,
    LBCSystem<dim, Number, 1> &therm_lbc_system,
    int n_initial_global_refinements) {

    const Number height = 53.334; // [mm]
    const Number radius = 6.413; // [mm]

    const Number top_coordinate = height/2;
    const Number base_coordinate = 0.0;
    const unsigned int base_repetitions = std::pow(2, n_initial_global_refinements);
    const unsigned int aspect_ratio = std::ceil(0.5 * height / radius);

    GridGenerator::subdivided_hyper_rectangle(
      triangulation,
      std::vector<unsigned int> {base_repetitions, aspect_ratio * base_repetitions},
      Point<dim>(0, base_coordinate),
      Point<dim>(radius, top_coordinate),
      true
    );

    // GridTools::transform(RefiningTransform<dim>(top_coordinate, 0.15, base_coordinate), triangulation);

    ComponentMask x_component_mask(dim+1, false);
    x_component_mask.set(0, true);
    ComponentMask y_component_mask(dim+1, false);
    y_component_mask.set(1, true);
    ComponentMask z_component_mask(dim+1, false);
    z_component_mask.set(dim-1, true);
    ComponentMask rho_component_mask(dim+1, false);
    rho_component_mask.set(dim, true);

    std::map< types::boundary_id, const Function< dim, Number > * > base_constraint_function_map;
    base_constraint_function_map.insert(
      std::pair<types::boundary_id, Function<dim, Number>*>(2, &mech_lbc_system.zero_function));
    mech_lbc_system.interpolatoryConstraintAppliers.push_back(
      InterpolatoryConstraintApplier<dim, Number>(
        base_constraint_function_map,
        y_component_mask));

    std::map< types::boundary_id, const Function< dim, Number > * > top_constraint_function_map;
    top_constraint_function_map.insert(
      std::pair<types::boundary_id, Function<dim, Number>*>(3, &mech_lbc_system.zero_function));
    mech_lbc_system.interpolatoryConstraintAppliers.push_back(
      InterpolatoryConstraintApplier<dim, Number>(
        top_constraint_function_map,
        y_component_mask));

    std::map< types::boundary_id, const Function< dim, Number > * > yz_constraint_function_map;
    yz_constraint_function_map.insert(
      std::pair<types::boundary_id, Function<dim, Number>*>(0, &mech_lbc_system.zero_function));
    mech_lbc_system.interpolatoryConstraintAppliers.push_back(
      InterpolatoryConstraintApplier<dim, Number>(
        yz_constraint_function_map,
        x_component_mask));


    const Number total_elongation = 2*8.0; // mm
    const Number elongation_rate = 2*1.0; // [mm/s]
    const unsigned int n_steps = ceil(total_elongation/(elongation_rate*time_increment));

    auto scale_z_handler = new IncrementInterpolationHandler<dim, Number, dim+1>(
            new ScaleZFunction<dim, Number, dim+1>(total_elongation/(static_cast<Number>(n_steps)*height), dim-1),
            true,
            y_component_mask,
            false,
            ComponentMask(dim+1, false),
            0,
            mapping);

    mech_lbc_system.increment_interpolation_handlers.push_back(scale_z_handler);

    std::map< types::boundary_id, const Function< dim, Number > * > top_rotation_constraint_function_map;
    top_rotation_constraint_function_map.insert(
      std::pair<types::boundary_id, Function<dim, Number>*>(3, &mech_lbc_system.zero_function));
    mech_lbc_system.interpolatoryConstraintAppliers.push_back(
      InterpolatoryConstraintApplier<dim, Number>(
        top_rotation_constraint_function_map,
        rho_component_mask));

    std::map< types::boundary_id, const Function< dim, Number > * > axial_rotation_constraint_function_map;
    axial_rotation_constraint_function_map.insert(
      std::pair<types::boundary_id, Function<dim, Number>*>(0, &mech_lbc_system.zero_function));
    mech_lbc_system.interpolatoryConstraintAppliers.push_back(
      InterpolatoryConstraintApplier<dim, Number>(
        axial_rotation_constraint_function_map,
        rho_component_mask));

    // Thermal constraints
    const Number convection_coefficient = 17.5e-6; // [J.mm^-2.s^-1.K^-1]
    therm_lbc_system.convection_BC_appliers.push_back(
      std::pair<int, ConvectionBoundaryConditionApplier<dim, Number> >(
        1,
        ConvectionBoundaryConditionApplier<dim, Number>(
          0, convection_coefficient, ambient_temperature)));
    therm_lbc_system.convection_BC_appliers.push_back(
      std::pair<int, ConvectionBoundaryConditionApplier<dim, Number> >(
        3,
        ConvectionBoundaryConditionApplier<dim, Number>(
          0, convection_coefficient, ambient_temperature)));
  } /* make_necking_grid() */


  template <int dim, typename Number>
  void PlasticityLabProg<dim, Number>::make_interference_cylinder_grid(
    Triangulation<dim> &triangulation,
    LBCSystem<dim, Number, dim+1> &mech_lbc_system,
    LBCSystem<dim, Number, 1> &therm_lbc_system,
    int n_initial_global_refinements) {

    const Number height = 300.0; // [mm]
    const Number radius = 40.0; // [mm]

    const unsigned int base_repetitions = std::pow(2, n_initial_global_refinements);
    const unsigned int aspect_ratio = std::ceil(0.5 * height / radius);

    GridGenerator::subdivided_hyper_rectangle(
      triangulation,
      std::vector<unsigned int> {base_repetitions, aspect_ratio * base_repetitions},
      Point<dim>(0, 0),
      Point<dim>(radius, height),
      true
    );

   // mech_lbc_system.boundaryLoadAppliers.push_back(
   //   std::pair<int,BodyForceApplier<dim,Number> >(
   //      1, BodyForceApplier<dim,Number>(dim, 0.8*4.50e1)));

    ComponentMask x_component_mask(dim+1, false);
    x_component_mask.set(0, true);
    ComponentMask y_component_mask(dim+1, false);
    y_component_mask.set(1, true);
    ComponentMask z_component_mask(dim+1, false);
    z_component_mask.set(dim-1, true);
    ComponentMask rho_component_mask(dim+1, false);
    rho_component_mask.set(dim, true);

    std::map< types::boundary_id, const Function< dim, Number > * > top_constraint_function_map;
    top_constraint_function_map.insert(
      std::pair<types::boundary_id, Function<dim, Number>*>(3, &mech_lbc_system.zero_function));
    mech_lbc_system.interpolatoryConstraintAppliers.push_back(
      InterpolatoryConstraintApplier<dim, Number>(
        top_constraint_function_map,
        y_component_mask));

    std::map< types::boundary_id, const Function< dim, Number > * > yz_constraint_function_map;
    yz_constraint_function_map.insert(
      std::pair<types::boundary_id, Function<dim, Number>*>(0, &mech_lbc_system.zero_function));
    mech_lbc_system.interpolatoryConstraintAppliers.push_back(
      InterpolatoryConstraintApplier<dim, Number>(
        yz_constraint_function_map,
        x_component_mask));


    const Number total_elongation = 300.0; // mm
    const Number elongation_rate = 1.0; // [mm/s]
    const unsigned int n_steps = ceil(total_elongation/(elongation_rate*time_increment));

    auto push_z_handler = new IncrementInterpolationHandler<dim, Number, dim+1>(
            new ConstantFunction<dim, Number>(std::vector< Number >{0, -total_elongation/static_cast<Number>(n_steps), 0}),
            true,
            y_component_mask,
            false,
            ComponentMask(dim+1, false),
            0,
            mapping);

    mech_lbc_system.increment_interpolation_handlers.push_back(push_z_handler);

    std::map< types::boundary_id, const Function< dim, Number > * > top_rotation_constraint_function_map;
    top_rotation_constraint_function_map.insert(
      std::pair<types::boundary_id, Function<dim, Number>*>(3, &mech_lbc_system.zero_function));
    mech_lbc_system.interpolatoryConstraintAppliers.push_back(
      InterpolatoryConstraintApplier<dim, Number>(
        top_rotation_constraint_function_map,
        rho_component_mask));

    std::map< types::boundary_id, const Function< dim, Number > * > axial_rotation_constraint_function_map;
    axial_rotation_constraint_function_map.insert(
      std::pair<types::boundary_id, Function<dim, Number>*>(0, &mech_lbc_system.zero_function));
    mech_lbc_system.interpolatoryConstraintAppliers.push_back(
      InterpolatoryConstraintApplier<dim, Number>(
        axial_rotation_constraint_function_map,
        rho_component_mask));

    const Number drive_speed = 0.2;  // [rad/s]

    auto scale_rotation_increment = new IncrementInterpolationHandler<dim, Number, dim+1>(
            new ScaleComponentFunction<dim, Number, dim+1>(drive_speed * time_increment, 0, dim),
            true,
            rho_component_mask,
            false,
            ComponentMask(dim+1, false),
            3,
            mapping);
    mech_lbc_system.increment_interpolation_handlers.push_back(scale_rotation_increment);
    auto constant_angular_velocity_handler = new IncrementInterpolationHandler<dim, Number, dim+1>(
            new ScaleComponentFunction<dim, Number, dim+1>(drive_speed, 0, dim),
            true,
            rho_component_mask,
            false,
            ComponentMask(dim+1, false),
            3,
            mapping);
    mech_lbc_system.initial_velocity_interpolation_handlers.push_back(constant_angular_velocity_handler);

    // auto constant_velocity_angular_handler = new IncrementInterpolationHandler<dim, Number, dim+1>(
    //         new ScaleComponentFunction<dim, Number, dim+1>(0.0e-3, 0, dim),
    //         true,
    //         rho_component_mask,
    //         false,
    //         ComponentMask(dim+1, false),
    //         0,
    //         mapping);

    // mech_lbc_system.initial_deformation_interpolation_handlers.push_back(constant_velocity_angular_handler);

    // Thermal constraints
    const Number convection_coefficient = 17.5e-6; // [J.mm^-2.s^-1.K^-1]
    therm_lbc_system.convection_BC_appliers.push_back(
      std::pair<int, ConvectionBoundaryConditionApplier<dim, Number> >(
        1,
        ConvectionBoundaryConditionApplier<dim, Number>(
          0, convection_coefficient, ambient_temperature)));
    therm_lbc_system.convection_BC_appliers.push_back(
      std::pair<int, ConvectionBoundaryConditionApplier<dim, Number> >(
        2,
        ConvectionBoundaryConditionApplier<dim, Number>(
          0, convection_coefficient, ambient_temperature)));
    therm_lbc_system.convection_BC_appliers.push_back(
      std::pair<int, ConvectionBoundaryConditionApplier<dim, Number> >(
        3,
        ConvectionBoundaryConditionApplier<dim, Number>(
          0, convection_coefficient, ambient_temperature)));

//    std::map< types::boundary_id, const Function< dim, Number > * > thermal_constraint_function_map;
//    thermal_constraint_function_map.insert(
//      std::pair<types::boundary_id, Function<dim, Number>*>(1, &therm_lbc_system.zero_function));
//    thermal_constraint_function_map.insert(
//      std::pair<types::boundary_id, Function<dim, Number>*>(8, &therm_lbc_system.zero_function));
//    therm_dof_system.interpolatoryConstraintAppliers.push_back(
//      InterpolatoryConstraintApplier<dim,Number>(
//        thermal_constraint_function_map,ComponentMask(1, true)));
  } /* make_interference_cylinder_grid() */


  template <int dim, typename Number>
  void PlasticityLabProg<dim, Number>::make_ball_in_hypershell_grid(
    Triangulation<dim> &triangulation,
    LBCSystem<dim, Number, dim+1> &mech_lbc_system,
    LBCSystem<dim, Number, 1> &therm_lbc_system,
    int n_initial_global_refinements) {

    const unsigned int INNER_BOUNDARY_ID = 0;

    const Number outer_radius = 60.0; // [mm]

    GridGenerator::half_hyper_ball(
      triangulation,
      Point<dim>(0, -10),
      outer_radius
    );

    for (const auto &cell : triangulation.active_cell_iterators()) {
      cell->set_boundary_id(INNER_BOUNDARY_ID);
    }

    triangulation.refine_global(n_initial_global_refinements);

   mech_lbc_system.boundaryLoadAppliers.push_back(
     std::pair<int,BodyForceApplier<dim,Number> >(
        1, BodyForceApplier<dim,Number>(dim, 0.8*4.50e1)));

    ComponentMask x_component_mask(dim+1, false);
    x_component_mask.set(0, true);
    ComponentMask y_component_mask(dim+1, false);
    y_component_mask.set(1, true);
    ComponentMask z_component_mask(dim+1, false);
    z_component_mask.set(dim-1, true);
    ComponentMask rho_component_mask(dim+1, false);
    rho_component_mask.set(dim, true);
    ComponentMask all_component_mask(dim+1, true);

    std::map< types::boundary_id, const Function< dim, Number > * > yz_constraint_function_map;
    yz_constraint_function_map.insert(
      std::pair<types::boundary_id, Function<dim, Number>*>(1, &mech_lbc_system.zero_function));
    mech_lbc_system.interpolatoryConstraintAppliers.push_back(
      InterpolatoryConstraintApplier<dim, Number>(
        yz_constraint_function_map,
        x_component_mask));

    std::map< types::boundary_id, const Function< dim, Number > * > down_constraint_function_map;
    down_constraint_function_map.insert(
      std::pair<types::boundary_id, Function<dim, Number>*>(INNER_BOUNDARY_ID, &mech_lbc_system.zero_function));
    mech_lbc_system.interpolatoryConstraintAppliers.push_back(
      InterpolatoryConstraintApplier<dim, Number>(
        down_constraint_function_map,
        y_component_mask));

    const Number total_elongation = 300.0; // mm
    const Number elongation_rate = 225.0e-2; // [mm/s]
    const unsigned int n_steps = ceil(total_elongation/(elongation_rate*time_increment));

    auto push_z_handler = new IncrementInterpolationHandler<dim, Number, dim+1>(
            new ConstantFunction<dim, Number>(std::vector< Number >{0, -total_elongation/static_cast<Number>(n_steps), 0}),
            true,
            y_component_mask,
            false,
            ComponentMask(dim+1, false),
            INNER_BOUNDARY_ID,
            mapping);

    mech_lbc_system.increment_interpolation_handlers.push_back(push_z_handler);

    std::map< types::boundary_id, const Function< dim, Number > * > top_rotation_constraint_function_map;
    top_rotation_constraint_function_map.insert(
      std::pair<types::boundary_id, Function<dim, Number>*>(0, &mech_lbc_system.zero_function));
    mech_lbc_system.interpolatoryConstraintAppliers.push_back(
      InterpolatoryConstraintApplier<dim, Number>(
        top_rotation_constraint_function_map,
        rho_component_mask));

    // Thermal constraints
    const Number convection_coefficient = 17.5e-6; // [J.mm^-2.s^-1.K^-1]
    therm_lbc_system.convection_BC_appliers.push_back(
      std::pair<int, ConvectionBoundaryConditionApplier<dim, Number> >(
        1,
        ConvectionBoundaryConditionApplier<dim, Number>(
          0, convection_coefficient, ambient_temperature)));
    therm_lbc_system.convection_BC_appliers.push_back(
      std::pair<int, ConvectionBoundaryConditionApplier<dim, Number> >(
        2,
        ConvectionBoundaryConditionApplier<dim, Number>(
          0, convection_coefficient, ambient_temperature)));
    therm_lbc_system.convection_BC_appliers.push_back(
      std::pair<int, ConvectionBoundaryConditionApplier<dim, Number> >(
        3,
        ConvectionBoundaryConditionApplier<dim, Number>(
          0, convection_coefficient, ambient_temperature)));
  } /* make_ball_in_hypershell_grid() */





  template <int dim, typename Number>
  void PlasticityLabProg<dim, Number>::make_cylindrical_impact_grid(
    Triangulation<dim> &triangulation,
    LBCSystem<dim, Number, dim+1> &mech_lbc_system,
    LBCSystem<dim, Number, 1> &therm_lbc_system,
    int n_initial_global_refinements) {

    const Number initial_velocity = 1.9e5; // [mm/s]
    const Number height = 25.4; // [mm]
    const Number radius = 3.81; // [mm]

    const unsigned int base_repetitions = std::pow(2, n_initial_global_refinements);
    const unsigned int aspect_ratio = std::ceil(0.25 * height / radius);

    GridGenerator::subdivided_hyper_rectangle(
      triangulation,
      std::vector<unsigned int> {base_repetitions, aspect_ratio * base_repetitions},
      Point<dim>(0, 0),
      Point<dim>(radius, height),
      true
    );

//    mech_lbc_system.boundaryLoadAppliers.push_back(
//      std::pair<int,BodyForceApplier<dim,Number> >(
//         6, BodyForceApplier<dim,Number>(2, -4.50)));

    ComponentMask x_component_mask(dim, false);
    x_component_mask.set(0, true);
    ComponentMask y_component_mask(3, false);
    y_component_mask.set(1, true);
    ComponentMask z_component_mask(dim, false);
    z_component_mask.set(dim-1, true);
    ComponentMask rho_component_mask(dim+1, false);
    rho_component_mask.set(dim, true);

    std::map< types::boundary_id, const Function< dim, Number > * > axial_constraint_function_map;
    axial_constraint_function_map.insert(
      std::pair<types::boundary_id, Function<dim, Number>*>(0, &mech_lbc_system.zero_function));
    mech_lbc_system.interpolatoryConstraintAppliers.push_back(
      InterpolatoryConstraintApplier<dim, Number>(
        axial_constraint_function_map,
        x_component_mask));

    std::map< types::boundary_id, const Function< dim, Number > * > base_constraint_function_map;
    base_constraint_function_map.insert(
      std::pair<types::boundary_id, Function<dim, Number>*>(2, &mech_lbc_system.zero_function));
    mech_lbc_system.interpolatoryConstraintAppliers.push_back(
      InterpolatoryConstraintApplier<dim, Number>(
        base_constraint_function_map,
        y_component_mask));

    auto constant_velocity_handler = new IncrementInterpolationHandler<dim, Number, dim+1>(
            new ConstantFunction<dim, Number>(std::vector< Number >{0, -initial_velocity, 0}),
            true,
            y_component_mask,
            false,
            ComponentMask(dim, false),
            0,
            mapping);

    mech_lbc_system.initial_velocity_interpolation_handlers.push_back(constant_velocity_handler);

    std::map< types::boundary_id, const Function< dim, Number > * > top_rotation_constraint_function_map;
    top_rotation_constraint_function_map.insert(
      std::pair<types::boundary_id, Function<dim, Number>*>(3, &mech_lbc_system.zero_function));
    mech_lbc_system.interpolatoryConstraintAppliers.push_back(
      InterpolatoryConstraintApplier<dim, Number>(
        top_rotation_constraint_function_map,
        rho_component_mask));

    std::map< types::boundary_id, const Function< dim, Number > * > axial_rotation_constraint_function_map;
    axial_rotation_constraint_function_map.insert(
      std::pair<types::boundary_id, Function<dim, Number>*>(0, &mech_lbc_system.zero_function));
    mech_lbc_system.interpolatoryConstraintAppliers.push_back(
      InterpolatoryConstraintApplier<dim, Number>(
        axial_rotation_constraint_function_map,
        rho_component_mask));

    auto constant_velocity_angular_handler = new IncrementInterpolationHandler<dim, Number, dim+1>(
            new ScaleComponentFunction<dim, Number, dim+1>(0.0e2, 0, dim),
            true,
            rho_component_mask,
            false,
            ComponentMask(dim+1, false),
            0,
            mapping);

    mech_lbc_system.initial_deformation_interpolation_handlers.push_back(constant_velocity_angular_handler);

    // Thermal constraints
    const Number convection_coefficient = 17.5e-6; // [J.mm^-2.s^-1.K^-1]
    therm_lbc_system.convection_BC_appliers.push_back(
      std::pair<int, ConvectionBoundaryConditionApplier<dim, Number> >(
        1,
        ConvectionBoundaryConditionApplier<dim, Number>(
          0, convection_coefficient, ambient_temperature)));
    therm_lbc_system.convection_BC_appliers.push_back(
      std::pair<int, ConvectionBoundaryConditionApplier<dim, Number> >(
        3,
        ConvectionBoundaryConditionApplier<dim, Number>(
          0, convection_coefficient, ambient_temperature)));

//    std::map< types::boundary_id, const Function< dim, Number > * > thermal_constraint_function_map;
//    thermal_constraint_function_map.insert(
//      std::pair<types::boundary_id, Function<dim, Number>*>(1, &therm_lbc_system.zero_function));
//    thermal_constraint_function_map.insert(
//      std::pair<types::boundary_id, Function<dim, Number>*>(8, &therm_lbc_system.zero_function));
//    therm_dof_system.interpolatoryConstraintAppliers.push_back(
//      InterpolatoryConstraintApplier<dim,Number>(
//        thermal_constraint_function_map,ComponentMask(1, true)));

  } /* make_cylindrical_impact_grid() */


  template <int dim, typename Number>
  void PlasticityLabProg<dim, Number>::make_hook_membrane_grid(
    int /*n_initial_global_refinements*/) {

  } /* make_necking_grid() */



  template <int dim, typename Number>
  Tensor<2, dim+1, Number> PlasticityLabProg<dim, Number>::get_rotation_tensor(
      const Tensor<2, dim+1, Number> &skew_symmetric_rotation) const {
    const Number theta = Constants<dim, Number>::sqrt_half() * skew_symmetric_rotation.norm();
    if (theta > 1e-8) {
      const Number h1 = (1./(theta*theta)*(1.-std::cos(theta)));
      const Number h2 = (1./theta*std::sin(theta));
      return static_cast<Tensor<2, dim+1, Number>>(unit_symmetric_tensor<dim+1, Number>())
        + h1 * skew_symmetric_rotation * skew_symmetric_rotation
        + h2 * skew_symmetric_rotation;
    } else {
      return static_cast<Tensor<2, dim+1, Number>>(unit_symmetric_tensor<dim+1, Number>());
    }
  }


  template <int dim, typename Number>
  Tensor<2, dim+1, Number> PlasticityLabProg<dim, Number>::get_rotation_tensor_variation(
      const Tensor<2, dim+1, Number> &skew_symmetric_rotation,
      const Tensor<2, dim+1, Number> &skew_symmetric_rotation_variation) const {

    const Number theta = Constants<dim, Number>::sqrt_half() * skew_symmetric_rotation.norm();
    if (theta > 1e-8) {
      const Number h1 = (1./(theta*theta)*(1.-std::cos(theta)));
      const Number h2 = (1./theta*std::sin(theta));
      const Number delta_theta = 1./(2.*theta) * scalar_product(skew_symmetric_rotation, skew_symmetric_rotation_variation);
      const Number delta_h1 = (-2./theta * (h1 - 0.5 * h2)) * delta_theta;
      const Number delta_h2 = (-1./theta * h2 + 1./theta * std::cos(theta)) * delta_theta;
      const Tensor<2, dim+1, Number> rotation_tensor_variation = delta_h1 * skew_symmetric_rotation * skew_symmetric_rotation
        + h1 * skew_symmetric_rotation_variation * skew_symmetric_rotation
        + h1 * skew_symmetric_rotation * skew_symmetric_rotation_variation
        + delta_h2 * skew_symmetric_rotation
        + h2 * skew_symmetric_rotation_variation;
      return rotation_tensor_variation;
    } else {
      return Tensor<2, dim+1, Number>();
    }
  }

}

