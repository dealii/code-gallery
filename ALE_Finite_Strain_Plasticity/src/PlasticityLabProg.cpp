/*
 * PlasticityLabProg.cpp
 *
 *  Created on: 09 Jul 2014
 *      Author: cerecam
 */

#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/lac/constraint_matrix.h>

#include "TimeRateUpdateFlags.h"
#include "TimeRateRequest.h"

#include "PlasticityLabProg.h"
#include "PlasticityLabProgDrivers.cpp"

#include "ReferencePoint.h"
#include "RemappedPoint.h"

using namespace dealii;
using std::endl;

namespace PlasticityLab {
  template <int dim, typename Number>
  PlasticityLabProg<dim, Number>::PlasticityLabProg(
      Material<dim+1, Number> &material) :
    mpi_communicator(MPI_COMM_WORLD),
    pcout(std::cout,
          (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
    order(1),
    mech_fe(FE_Q<dim>(order), dim, FE_Q<dim>(order), 2),
    therm_fe(order),
    mixed_var_fe(order-1),
    mesh_motion_fe(FE_Q<dim>(order), dim),
    mapping (order),
    triangulation(mpi_communicator),
    mech_dof_system(triangulation, mapping),
    therm_dof_system (triangulation, mapping),
    mixed_fe_dof_system(triangulation, mapping),
    mesh_motion_dof_system(triangulation, mapping),
    quadrature_formula(order+1),
    face_quadrature_formula(order+1),
    material(material) { }


  template <int dim, typename Number>
  PlasticityLabProg<dim, Number>::~PlasticityLabProg() {

  }

  template <int dim, typename Number>
  template <typename TriangulationType, typename MaterialType>
  void PlasticityLabProg<dim, Number>::setup_material_data(
    TriangulationType   &triangulation,
    MaterialType &material) {
    const unsigned int num_cells = triangulation.n_active_cells();
    triangulation.clear_user_data();
    material.setup_point_history(num_cells * quadrature_formula.size());
    unsigned int history_index = 0;
    for (typename Triangulation<dim>::active_cell_iterator
         cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell) {
      cell->set_user_index (history_index);
      history_index += quadrature_formula.size();
    }
  }

  template <int dim, typename Number>
  void PlasticityLabProg<dim, Number>::setup_material_area_factors(
      const DoFSystem<dim, Number> &mesh_motion_dof_system,
      std::unordered_map<size_t, Tensor<1, dim+1, Number>> &material_area_factors) {

    FEFaceValues<dim> fe_face_values (
      mapping,
      mesh_motion_fe,
      face_quadrature_formula,
      update_normal_vectors);
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    for (auto cell = mesh_motion_dof_system.dof_handler.begin_active();
         cell != mesh_motion_dof_system.dof_handler.end();
         ++cell) {
      if (cell->is_locally_owned()) {
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face) {
          if(cell->at_boundary(face)) {
            fe_face_values.reinit(cell, face);

            for (unsigned int q_point = 0; q_point < n_face_q_points; q_point++) {
              const unsigned int cell_index = cell->user_index() / quadrature_formula.size();
              const unsigned int surface_point_key =
                cell_index * GeometryInfo<dim>::faces_per_cell * n_face_q_points
                + face * n_face_q_points
                + q_point;
              material_area_factors[surface_point_key] = postprocess_tensor_dimension(fe_face_values.normal_vector(q_point), 0);
            }
          }
        }
      }
    }
  }

  template <int dim, typename Number>
  void PlasticityLabProg<dim, Number>::update_material_area_factors(
      const NewtonStepSystem &mesh_motion_nonlinear_system,
      const DoFSystem<dim, Number> &mesh_motion_dof_system,
      std::unordered_map<size_t, Tensor<1, dim+1, Number>> &material_area_factors) {

    FEFaceValues<dim> fe_face_values (
      mapping,
      mesh_motion_fe,
      face_quadrature_formula,
      update_values | update_gradients | update_quadrature_points);

    const unsigned int n_face_q_points = face_quadrature_formula.size();

    const FEValuesExtractors::Vector displacements (0);

    std::vector< Tensor<1, dim, Number> > mesh_motion_value_increments(n_face_q_points);
    std::vector< Tensor<2, dim, Number> > mesh_motion_gradient_increments(n_face_q_points);

    for (auto cell = mesh_motion_dof_system.dof_handler.begin_active();
         cell != mesh_motion_dof_system.dof_handler.end();
         ++cell) {
      if (cell->is_locally_owned()) {
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face) {
          if(cell->at_boundary(face)) {
            fe_face_values.reinit(cell, face);

            fe_face_values[displacements].get_function_gradients(
              mesh_motion_nonlinear_system.current_increment,
              mesh_motion_gradient_increments);

            fe_face_values[displacements].get_function_values(
              mesh_motion_nonlinear_system.current_increment,
              mesh_motion_value_increments);

            for (unsigned int q_point = 0; q_point < n_face_q_points; q_point++) {
              const unsigned int cell_index = cell->user_index() / quadrature_formula.size();
              const unsigned int surface_point_key =
                cell_index * GeometryInfo<dim>::faces_per_cell * n_face_q_points
                + face * n_face_q_points
                + q_point;

              const auto mesh_motion_gradient = get_deformation_gradient(
                                                  -mesh_motion_gradient_increments[q_point],
                                                  -mesh_motion_value_increments[q_point][0]
                                                    /fe_face_values.quadrature_point(q_point)[0]);

              material_area_factors[surface_point_key] = std::pow(determinant(mesh_motion_gradient), -1)
                                                         * transpose(mesh_motion_gradient)
                                                         * material_area_factors[surface_point_key];
            }
          }
        }
      }
    }
  }

  template <int dim, typename Number>
  template <typename TriangulationType>
  void PlasticityLabProg<dim, Number>::setup_mixed_fe_projection_data(
      const TriangulationType                      &triangulation,
      std::vector< MixedFEProjector<dim, Number> > &MixedFeProjectors,
      const FiniteElement<dim>                     &MixedFE,
      const Quadrature<dim>                        &quadrature_formula) {

    FEValues<dim> mixed_fe_values(
      mapping,
      MixedFE,
      quadrature_formula,
      update_values | update_quadrature_points | update_JxW_values);

    const unsigned int num_cells = triangulation.n_active_cells(),
                       n_q_points = quadrature_formula.size(),
                       mixed_dofs_per_cell = MixedFE.dofs_per_cell;
    MixedFeProjectors.clear();
    MixedFeProjectors.resize(num_cells);
    for (typename Triangulation<dim>::active_cell_iterator
         cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell) {
      mixed_fe_values.reinit(cell);
      MixedFeProjectors.at(cell->user_index() / n_q_points) =
        MixedFEProjector<dim, Number>(mixed_dofs_per_cell,
                                      mixed_fe_values);
    }
  }

  template<int dim, typename Number>
  void PlasticityLabProg<dim, Number>::remap_material_state_variables(
      const NewtonStepSystem &mesh_motion_nonlinear_system,
      const DoFSystem<dim, Number> &mesh_motion_dof_system,
      const NewtonStepSystem &mechanical_nonlinear_system,
      const DoFSystem<dim, Number> &mechanical_dof_system,
      const DoFSystem<dim, Number> &mixed_fe_dof_system,
      const std::vector< MixedFEProjector<dim, Number> > &mixed_fe_projector,
      Material<dim+1, Number> &material,
      std::unordered_map<point_index_t, Tensor<2, dim+1, Number>> &remapped_deformation_gradients) {

    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int dofs_per_cell = mesh_motion_fe.dofs_per_cell;
    const unsigned int mixed_dofs_per_cell = mixed_fe_dof_system.dof_handler.get_fe().dofs_per_cell;

    FEValues<dim> mesh_motion_fe_values(
      mapping,
      mesh_motion_fe,
      quadrature_formula,
      update_values  | update_gradients | update_quadrature_points | update_jacobians | update_JxW_values);

    FEValues<dim> mechanical_fe_values(
      mapping,
      mech_fe,
      quadrature_formula,
      update_values  | update_gradients);

    FEValues<dim> mixed_fe_values(
      mapping,
      mixed_fe_dof_system.dof_handler.get_fe(),
      quadrature_formula,
      update_values);

    std::vector< Tensor<1, dim, Number> > mesh_motion_value_increments(n_q_points);
    std::vector< Tensor<2, dim, Number> > mesh_motion_gradient_increments(n_q_points);

    std::vector< Tensor<1, dim, Number> > previous_deformation_values(n_q_points);
    std::vector< Tensor<2, dim, Number> > previous_deformation_gradients(n_q_points);
    std::vector< Tensor<1, dim, Number> > previous_deformation_value_at_remapped_point(1);
    std::vector< Tensor<2, dim, Number> > previous_deformation_gradient_at_remapped_point(1);

    const unsigned int material_parameter_count = material.get_material_parameter_count();

    const FEValuesExtractors::Vector displacements(0);
    const FEValuesExtractors::Scalar temperature (0);

    std::vector<ReferencePoint<dim, Number>> reference_points;
    std::vector<Point<dim, Number>> remapped_point_positions;

    for (auto cell = mesh_motion_dof_system.dof_handler.begin_active();
              cell != mesh_motion_dof_system.dof_handler.end();
              ++cell) {
      if (cell->is_locally_owned()) {
        mesh_motion_fe_values.reinit(cell);

        mesh_motion_fe_values[displacements].get_function_values(
          mesh_motion_nonlinear_system.current_increment,
          mesh_motion_value_increments);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
          const point_index_t quadrature_point_index = cell->user_index() + q_point;
          const Point<dim, Number> reference_point_position = mesh_motion_fe_values.quadrature_point(q_point);
          const Point<dim, Number> remapped_point_position = reference_point_position - mesh_motion_value_increments[q_point];

          ReferencePoint<dim, Number> reference_point;
          reference_point.mesh_motion_cell = cell;
          reference_point.q_point = q_point;
          reference_point.reference_point = reference_point_position;
          reference_point.remapped_point = remapped_point_position;

          reference_points.push_back(reference_point);
          remapped_point_positions.push_back(remapped_point_position);

        }

      }
    }

    MPI_Barrier(mpi_communicator);

    MPI_Datatype PointType;
    MPI_Type_contiguous(dim, MPI_DOUBLE, &PointType);
    MPI_Type_commit(&PointType);
    std::vector<int> displs;
    std::vector< Point<dim, Number>> received_remapped_positions;
    std::vector<RemappedPoint<dim, Number>> remapped_points;
    int nprocesses, this_process;
    int num_reference_points = reference_points.size();
    MPI_Comm_size(mpi_communicator, &nprocesses);
    std::vector<int> reference_point_counts(nprocesses);
    MPI_Allgather(
      &num_reference_points,
      1, MPI_INT,
      &reference_point_counts[0],
      1, MPI_INT,
      mpi_communicator);
    MPI_Comm_rank(mpi_communicator, &this_process);
    displs.resize(nprocesses);
    displs[0] = 0;
    for (int i = 1; i < nprocesses; ++i) {
      displs[i] = displs[i - 1] + reference_point_counts[i - 1];
    }

    const unsigned int count_received_reference_points = displs[nprocesses - 1] + reference_point_counts[nprocesses - 1];
    received_remapped_positions.resize(count_received_reference_points);
    remapped_points.resize(count_received_reference_points);
    std::vector<char> this_process_owns_remapped_point(count_received_reference_points);

    MPI_Allgatherv(
      &remapped_point_positions[0],
      num_reference_points,
      PointType,
      &received_remapped_positions[0],
      &reference_point_counts[0],
      &displs[0],
      PointType,
      mpi_communicator);

    MPI_Barrier(mpi_communicator);

    for(unsigned int received_point_id=0; received_point_id < count_received_reference_points; received_point_id++) {
      auto point = received_remapped_positions[received_point_id];
      auto cell_and_point = GridTools::find_active_cell_around_point(
                          mapping,
                          mesh_motion_dof_system.dof_handler,
                          point);
      auto cell = cell_and_point.first;
      auto unit_cell_point = cell_and_point.second;
      // here, we're assuming that `find_active_cell_around_point` returns the same
      // cell and point when called with different dof_handlers
      auto mechanical_cell_and_point = GridTools::find_active_cell_around_point(
                          mapping,
                          mechanical_dof_system.dof_handler,
                          point);
      auto mechanical_cell = mechanical_cell_and_point.first;
      auto mixed_fe_cell_and_point = GridTools::find_active_cell_around_point(
                          mapping,
                          mixed_fe_dof_system.dof_handler,
                          point);
      auto mixed_fe_cell = mixed_fe_cell_and_point.first;

      remapped_points[received_point_id].remapped_point = point;
      remapped_points[received_point_id].mesh_motion_cell = cell;
      remapped_points[received_point_id].unit_cell_point = unit_cell_point;
      remapped_points[received_point_id].field_cell = mechanical_cell;
      remapped_points[received_point_id].mixed_fe_cell = mixed_fe_cell;
      this_process_owns_remapped_point[received_point_id] = cell.state() == IteratorState::valid && cell->is_locally_owned()? 1 : 0;
    }

    MPI_Barrier(mpi_communicator);

    std::vector<char> remapped_point_candidates(num_reference_points * nprocesses);
    for (int process = 0; process < nprocesses; ++process) {
      if (reference_point_counts[process] > 0)
        MPI_Gather(
          &this_process_owns_remapped_point[displs[process]],
          reference_point_counts[process],
          MPI_C_BOOL,
          &remapped_point_candidates[0],
          reference_point_counts[process],
          MPI_C_BOOL,
          process,
          mpi_communicator);
    }

    std::vector<unsigned int> remapped_point_owning_process(num_reference_points);

    for (int i = 0; i < num_reference_points; ++i) {
      remapped_point_owning_process[i] = 0;
      for (int j = 1; j < nprocesses; ++j) {
        if (remapped_point_candidates[j * num_reference_points + i] == 1) {
          remapped_point_owning_process[i] = j;
          continue;
        }
      }
    }

    std::vector<RemappedPoint<dim, Number>> mapping_remapped_points;
    std::vector<unsigned int> mapping_reference_point_owning_process;
    std::vector<unsigned int> mapping_reference_point_index_at_remote_process;

    // This should really be a vector<bool>, but addresses of individual elements of
    // vector<bool> cannot be taken. It's a template specialization to save space
    std::vector<char> remote_remapped_point_is_accepted(num_reference_points * nprocesses);
    std::vector<char> local_remapped_point_is_accepted(count_received_reference_points);
    for (int i = 0; i < num_reference_points; ++i) {
      for (unsigned int j = 0; j < static_cast<unsigned int>(nprocesses); ++j) {
        remote_remapped_point_is_accepted[j * num_reference_points + i] = (remapped_point_owning_process[i] == j) ? 1 : 0;
      }
    }

    MPI_Barrier(mpi_communicator);

    for (int process = 0; process < nprocesses; ++process) {
      if (reference_point_counts[process] > 0) {
        MPI_Scatter(
          &remote_remapped_point_is_accepted[0],
          reference_point_counts[process],
          MPI_C_BOOL,
          &local_remapped_point_is_accepted[displs[process]],
          reference_point_counts[process],
          MPI_C_BOOL,
          process,
          mpi_communicator);
      }
    }

    std::vector<unsigned int> mapping_remote_reference_point_counts(nprocesses, 0);
    for (int process = 0; process < nprocesses; ++process) {
      for (int i = 0; i < reference_point_counts[process]; ++i) {
        if (local_remapped_point_is_accepted[displs[process] + i]) {
          RemappedPoint<dim, Number> accepted_remapped_point = remapped_points[displs[process] + i];

          mapping_remapped_points.push_back(accepted_remapped_point);
          mapping_reference_point_owning_process.push_back(process);
          mapping_reference_point_index_at_remote_process.push_back(i);
          ++mapping_remote_reference_point_counts[process];
        }
      }
    }

    MPI_Barrier(mpi_communicator);

    std::vector<unsigned int> mapping_remote_remapped_point_counts(nprocesses);
    for (int i = 0; i < nprocesses; ++i) {
      MPI_Scatter(
        &mapping_remote_reference_point_counts[0],
        1, MPI_UNSIGNED,
        &mapping_remote_remapped_point_counts[i],
        1, MPI_UNSIGNED,
        i, mpi_communicator);
    }

    {
      std::vector<std::vector<Number> > local_state_parameter_groups(nprocesses);
      std::vector<std::vector<Number> > remote_state_parameter_groups(nprocesses);

      std::vector<std::vector<Number> > local_deformation_gradient_groups(nprocesses);
      std::vector<std::vector<Number> > remote_deformation_gradient_groups(nprocesses);

      for (int i = 0; i < nprocesses; ++i) {
        local_state_parameter_groups.at(i).resize(material_parameter_count * mapping_remote_reference_point_counts.at(i));
        remote_state_parameter_groups.at(i).resize(material_parameter_count * mapping_remote_remapped_point_counts.at(i));

        local_deformation_gradient_groups.at(i).resize((dim+1) * (dim+1) * mapping_remote_reference_point_counts.at(i));
        remote_deformation_gradient_groups.at(i).resize((dim+1) * (dim+1) * mapping_remote_remapped_point_counts.at(i));
      }

      std::vector<unsigned int> next_to_process(nprocesses, 0);
      for (unsigned int i = 0; i < mapping_remapped_points.size(); ++i) {
        const unsigned int group = mapping_reference_point_owning_process.at(i);
        const RemappedPoint<dim, Number> remapped_point = mapping_remapped_points.at(i);

        Quadrature<dim> remapped_point_quadrature(
          std::vector<Point<dim, Number>> (1, remapped_point.unit_cell_point));

        FEValues<dim> remapped_point_fe_values(
          mapping,
          mesh_motion_fe,
          remapped_point_quadrature,
          update_values | update_gradients | update_quadrature_points);

        FEValues<dim> remapped_point_mechanical_fe_values(
          mapping,
          mech_fe,
          remapped_point_quadrature,
          update_values | update_gradients);

        FEValues<dim> remapped_point_mixed_fe_values(
          mapping,
          mixed_fe_dof_system.dof_handler.get_fe(),
          remapped_point_quadrature,
          update_values);

        remapped_point_fe_values.reinit(remapped_point.mesh_motion_cell);
        mesh_motion_fe_values.reinit(remapped_point.mesh_motion_cell);

        remapped_point_mechanical_fe_values.reinit(remapped_point.field_cell);
        mechanical_fe_values.reinit(remapped_point.field_cell);

        remapped_point_mixed_fe_values.reinit(remapped_point.mixed_fe_cell);

        mesh_motion_fe_values[displacements].get_function_gradients(
          mesh_motion_nonlinear_system.current_increment,
          mesh_motion_gradient_increments);

        mesh_motion_fe_values[displacements].get_function_values(
          mesh_motion_nonlinear_system.current_increment,
          mesh_motion_value_increments);

        mechanical_fe_values[displacements].get_function_gradients(
          mechanical_nonlinear_system.previous_deformation,
          previous_deformation_gradients);

        mechanical_fe_values[displacements].get_function_values(
          mechanical_nonlinear_system.previous_deformation,
          previous_deformation_values);

        remapped_point_mechanical_fe_values[displacements].get_function_gradients(
          mechanical_nonlinear_system.previous_deformation,
          previous_deformation_gradient_at_remapped_point);

        remapped_point_mechanical_fe_values[displacements].get_function_values(
          mechanical_nonlinear_system.previous_deformation,
          previous_deformation_value_at_remapped_point);

        std::vector< std::vector<Number> > material_parameters_at_q_points(
              material_parameter_count,
              std::vector<Number>(n_q_points));

        for(unsigned int q_point=0; q_point<n_q_points; q_point++) {
          const point_index_t quadrature_point_index = remapped_point.mesh_motion_cell->user_index() + q_point;

          const auto deformation_gradient =
            get_deformation_gradient(
              previous_deformation_gradients[q_point],
              previous_deformation_values[q_point][0]/mesh_motion_fe_values.quadrature_point(q_point)[0]);

          std::vector<Number> state_parameters = material.get_state_parameters(quadrature_point_index, /*deformation_gradient*/unit_symmetric_tensor<dim+1, Number>());

          for(unsigned int parameter_index=0; parameter_index<material_parameter_count; parameter_index++) {
              material_parameters_at_q_points[parameter_index][q_point] = state_parameters[parameter_index];
          }
        }

        std::vector<std::vector<Number> > projected_material_parameters_coefficients(
              material_parameter_count,
              std::vector<Number>(mixed_dofs_per_cell));

        const unsigned int cell_index = remapped_point.mesh_motion_cell->user_index() / n_q_points;
        for(unsigned int parameter_index=0; parameter_index<material_parameter_count; parameter_index++) {
          mixed_fe_projector[cell_index].project(
            &projected_material_parameters_coefficients[parameter_index],
            material_parameters_at_q_points[parameter_index]);
        }

        Vector<Number> mixed_values (mixed_dofs_per_cell);
        for (unsigned int mixed_dof = 0; mixed_dof < mixed_dofs_per_cell; ++mixed_dof) {
          mixed_values(mixed_dof) = remapped_point_mixed_fe_values.shape_value(mixed_dof, 0);
        }

        std::vector<Number> projected_state_parameters(material_parameter_count, 0);
        for(unsigned int parameter_index=0; parameter_index<material_parameter_count; parameter_index++) {
          for (unsigned int mixed_dof = 0; mixed_dof < mixed_dofs_per_cell; mixed_dof++) {
            projected_state_parameters[parameter_index] += mixed_values(mixed_dof) * projected_material_parameters_coefficients[parameter_index][mixed_dof];
          }
        }


        for(unsigned int parameter_index=0; parameter_index<material_parameter_count; parameter_index++) {
          local_state_parameter_groups.at(group).at(next_to_process.at(group)*material_parameter_count + parameter_index) = projected_state_parameters[parameter_index];
        }

        const Tensor<2, dim+1, Number> previous_deformation_gradient =
          get_deformation_gradient(
            previous_deformation_gradient_at_remapped_point[0],
            previous_deformation_value_at_remapped_point[0][0]/remapped_point_fe_values.quadrature_point(0)[0]);

        for(unsigned int dim_i=0; dim_i<dim+1; dim_i++) {
          const unsigned int current_i_index = next_to_process.at(group) * (dim+1) + dim_i;
          for(unsigned int dim_j=0; dim_j<dim+1; dim_j++) {
            local_deformation_gradient_groups.at(group).at(current_i_index * (dim+1) + dim_j) = previous_deformation_gradient[dim_i][dim_j];
          }
        }

        ++next_to_process.at(group);
      }



      enum MessageFlag {
        MATERIAL_STATE_PARAMETER,
        REMAPPED_DEFORMATION_GRADIENT
      };

      const unsigned int
        material_state_parameter_requests_offset = 0,
        remapped_deformation_gradient_requests_offset = 1,
        request_array_size = 2;

      std::vector<MPI_Request> requests_vector(2 * nprocesses * request_array_size);

      for (int i = 0; i < nprocesses; ++i) {
        MPI_Isend(
          &local_state_parameter_groups.at(i)[0],
          material_parameter_count * mapping_remote_reference_point_counts.at(i),
          MPI_DOUBLE, i, MATERIAL_STATE_PARAMETER,
          mpi_communicator,
          &requests_vector[i + nprocesses * material_state_parameter_requests_offset]);
      }

      for (int i = 0; i < nprocesses; ++i) {
        MPI_Isend(
          &local_deformation_gradient_groups.at(i)[0],
          (dim+1) * (dim+1) * mapping_remote_reference_point_counts.at(i),
          MPI_DOUBLE, i, REMAPPED_DEFORMATION_GRADIENT,
          mpi_communicator,
          &requests_vector[i + nprocesses * remapped_deformation_gradient_requests_offset]);
      }

      for (int i = 0; i < nprocesses; ++i) {
        const unsigned int row_start = i + nprocesses * request_array_size;

        MPI_Irecv(
          &remote_state_parameter_groups.at(i)[0],
          material_parameter_count * mapping_remote_remapped_point_counts.at(i),
          MPI_DOUBLE, i, MATERIAL_STATE_PARAMETER,
          mpi_communicator,
          &requests_vector[row_start + nprocesses * material_state_parameter_requests_offset]);

        MPI_Irecv(
          &remote_deformation_gradient_groups.at(i)[0],
          (dim+1) * (dim+1) * mapping_remote_remapped_point_counts.at(i),
          MPI_DOUBLE, i, REMAPPED_DEFORMATION_GRADIENT,
          mpi_communicator,
          &requests_vector[row_start + nprocesses * remapped_deformation_gradient_requests_offset]);

      }

      std::vector<MPI_Status> statuses_vector(2 * request_array_size * nprocesses);
      MPI_Waitall(
        2 * request_array_size * nprocesses,
        &requests_vector[0],
        &statuses_vector[0]);

      next_to_process.clear();
      next_to_process.resize(nprocesses, 0);
      for (unsigned int i = 0; i < reference_points.size(); ++i) {
        unsigned int group = remapped_point_owning_process.at(i);

        ReferencePoint<dim, Number> &reference_point = reference_points[i];

        mesh_motion_fe_values.reinit(reference_point.mesh_motion_cell);

        std::vector<Number> remapped_state_parameters(material_parameter_count);
        for(unsigned int state_index=0; state_index<material_parameter_count; state_index++) {
          remapped_state_parameters[state_index] = remote_state_parameter_groups.at(group).at(next_to_process.at(group) * material_parameter_count + state_index);
        }

        Tensor<2, dim+1, Number> previous_deformation_gradient;
        for(unsigned int dim_i=0; dim_i<dim+1; dim_i++) {
          const unsigned int current_i_index = next_to_process.at(group) * (dim+1) + dim_i;
          for(unsigned int dim_j=0; dim_j<dim+1; dim_j++) {
            previous_deformation_gradient[dim_i][dim_j] = remote_deformation_gradient_groups.at(group).at(current_i_index * (dim+1) + dim_j);
          }
        }

        mesh_motion_fe_values[displacements].get_function_gradients(
          mesh_motion_nonlinear_system.current_increment,
          mesh_motion_gradient_increments);

        mesh_motion_fe_values[displacements].get_function_values(
          mesh_motion_nonlinear_system.current_increment,
          mesh_motion_value_increments);

        const auto mesh_motion_gradient = get_deformation_gradient(
                                            -mesh_motion_gradient_increments[reference_point.q_point],
                                            -mesh_motion_value_increments[reference_point.q_point][0]/mesh_motion_fe_values.quadrature_point(reference_point.q_point)[0]);

        const point_index_t quadrature_point_index = reference_point.mesh_motion_cell->user_index() + reference_point.q_point;
        material.set_state_parameters(quadrature_point_index, remapped_state_parameters, /*previous_deformation_gradient **/ /*mesh_motion_gradient*/unit_symmetric_tensor<dim+1, Number>());
        remapped_deformation_gradients[quadrature_point_index] = previous_deformation_gradient;

        ++next_to_process.at(group);
      }

    }
  }

  template<int dim, typename Number>
  void PlasticityLabProg<dim, Number>::remap_thermal_field(
      NewtonStepSystem &thermal_nonlinear_system,
      const DoFSystem<dim, Number> &thermal_dof_system,
      const NewtonStepSystem &mesh_motion_nonlinear_system,
      const DoFSystem<dim, Number> &mesh_motion_dof_system) {

    const Quadrature<dim> thermal_fe_support_point_quadrature(therm_fe.get_unit_support_points());

    const unsigned int n_q_points = thermal_fe_support_point_quadrature.size();
    const unsigned int dofs_per_cell = mesh_motion_fe.dofs_per_cell;
    const unsigned int thermal_dofs_per_cell = therm_fe.dofs_per_cell;

    FEValues<dim> mesh_motion_fe_values(
      mapping,
      mesh_motion_fe,
      thermal_fe_support_point_quadrature,
      update_values  | update_quadrature_points);

    FEValues<dim> thermal_fe_values(
      mapping,
      therm_fe,
      thermal_fe_support_point_quadrature,
      update_values | update_quadrature_points);

    std::vector< Number > previous_temperatures(1);
    std::vector< Tensor<1, dim, Number> > mesh_motion_value_increments(n_q_points);
    std::vector< Tensor<2, dim, Number> > mesh_motion_gradient_increments(n_q_points);

    const FEValuesExtractors::Vector displacements(0);
    const FEValuesExtractors::Scalar temperature (0);

    std::vector<ReferencePoint<dim, Number>> reference_points;
    std::vector<Point<dim, Number>> remapped_point_positions;

    auto cell = mesh_motion_dof_system.dof_handler.begin_active();
    auto thermal_cell = thermal_dof_system.dof_handler.begin_active();
    for (; cell != mesh_motion_dof_system.dof_handler.end(); ++cell, ++thermal_cell) {
      if (cell->is_locally_owned()) {
        mesh_motion_fe_values.reinit(cell);

        mesh_motion_fe_values[displacements].get_function_values(
          mesh_motion_nonlinear_system.current_increment,
          mesh_motion_value_increments);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
          const Point<dim, Number> reference_point_position = mesh_motion_fe_values.quadrature_point(q_point);
          const Point<dim, Number> remapped_point_position = reference_point_position - mesh_motion_value_increments[q_point];

          ReferencePoint<dim, Number> reference_point;
          reference_point.mesh_motion_cell = cell;
          reference_point.field_cell = thermal_cell;
          reference_point.q_point = q_point;
          reference_point.reference_point = reference_point_position;
          reference_point.remapped_point = remapped_point_position;

          reference_points.push_back(reference_point);
          remapped_point_positions.push_back(remapped_point_position);

        }

      }
    }

    MPI_Datatype PointType;
    MPI_Type_contiguous(dim, MPI_DOUBLE, &PointType);
    MPI_Type_commit(&PointType);
    std::vector<int> displs;
    std::vector< Point<dim, Number>> received_remapped_positions;
    std::vector<RemappedPoint<dim, Number>> thermal_points;
    int nprocesses, this_process;
    int num_reference_points = reference_points.size();
    MPI_Comm_size(mpi_communicator, &nprocesses);
    std::vector<int> reference_point_counts(nprocesses);
    MPI_Allgather(
      &num_reference_points,
      1, MPI_INT,
      &reference_point_counts[0],
      1, MPI_INT,
      mpi_communicator);
    MPI_Comm_rank(mpi_communicator, &this_process);
    displs.resize(nprocesses);
    displs[0] = 0;
    for (int i = 1; i < nprocesses; ++i) {
      displs[i] = displs[i - 1] + reference_point_counts[i - 1];
    }

    const unsigned int count_received_reference_points = displs[nprocesses - 1] + reference_point_counts[nprocesses - 1];
    received_remapped_positions.resize(count_received_reference_points);
    thermal_points.resize(count_received_reference_points);
    std::vector<char> this_process_owns_remapped_point(count_received_reference_points);
    std::vector<char> this_process_owns_thermal_point(count_received_reference_points);

    MPI_Allgatherv(
      &remapped_point_positions[0],
      num_reference_points,
      PointType,
      &received_remapped_positions[0],
      &reference_point_counts[0],
      &displs[0],
      PointType,
      mpi_communicator);


    for(unsigned int received_point_id=0; received_point_id < count_received_reference_points; received_point_id++) {
      auto point = received_remapped_positions[received_point_id];
      auto thermal_cell_and_point = GridTools::find_active_cell_around_point(
                          mapping,
                          thermal_dof_system.dof_handler,
                          point);
      auto thermal_cell = thermal_cell_and_point.first;
      auto thermal_unit_cell_point = thermal_cell_and_point.second;
      thermal_points[received_point_id].field_cell = thermal_cell;
      thermal_points[received_point_id].unit_cell_point = thermal_unit_cell_point;
      thermal_points[received_point_id].remapped_point = point;
      this_process_owns_thermal_point[received_point_id] = thermal_cell.state() == IteratorState::valid && thermal_cell->is_locally_owned()? 1 : 0;
    }

    std::vector<char> thermal_point_candidates(num_reference_points * nprocesses);
    for (int process = 0; process < nprocesses; ++process) {
      if (reference_point_counts[process] > 0)
        MPI_Gather(
          &this_process_owns_thermal_point[displs[process]],
          reference_point_counts[process],
          MPI_C_BOOL,
          &thermal_point_candidates[0],
          reference_point_counts[process],
          MPI_C_BOOL,
          process,
          mpi_communicator);
    }

    std::vector<unsigned int> thermal_point_owning_process(num_reference_points);
    for (int i = 0; i < num_reference_points; ++i) {
      thermal_point_owning_process[i] = 0;
      for (int j = 1; j < nprocesses; ++j) {
        if (thermal_point_candidates[j * num_reference_points + i] == 1) {
          thermal_point_owning_process[i] = j;
          continue;
        }
      }
    }

    std::vector<RemappedPoint<dim, Number>> mapping_thermal_points;
    std::vector<unsigned int> thermal_reference_point_owning_process;
    std::vector<unsigned int> thermal_reference_point_index_at_remote_process;

    // This should really be a vector<bool>, but addresses of individual elements of
    // vector<bool> cannot be taken. It's a template specialization to save space
    std::vector<char> remote_thermal_point_is_accepted(num_reference_points * nprocesses);
    std::vector<char> local_thermal_point_is_accepted(count_received_reference_points);
    for (int i = 0; i < num_reference_points; ++i) {
      for (unsigned int j = 0; j < static_cast<unsigned int>(nprocesses); ++j) {
        remote_thermal_point_is_accepted[j * num_reference_points + i] = (thermal_point_owning_process[i] == j) ? 1 : 0;
      }
    }

    for (int process = 0; process < nprocesses; ++process) {
      if (reference_point_counts[process] > 0) {
        MPI_Scatter(
          &remote_thermal_point_is_accepted[0],
          reference_point_counts[process],
          MPI_C_BOOL,
          &local_thermal_point_is_accepted[displs[process]],
          reference_point_counts[process],
          MPI_C_BOOL,
          process,
          mpi_communicator);
      }
    }

    std::vector<unsigned int> thermal_remote_reference_point_counts(nprocesses, 0);
    for (int process = 0; process < nprocesses; ++process) {
      for (int i = 0; i < reference_point_counts[process]; ++i) {
        if (local_thermal_point_is_accepted[displs[process] + i]) {
          RemappedPoint<dim, Number> accepted_thermal_point = thermal_points[displs[process] + i];

          mapping_thermal_points.push_back(accepted_thermal_point);
          thermal_reference_point_owning_process.push_back(process);
          thermal_reference_point_index_at_remote_process.push_back(i);
          ++thermal_remote_reference_point_counts[process];
        }
      }
    }

    MPI_Barrier(mpi_communicator);

    std::vector<unsigned int> thermal_remote_remapped_point_counts(nprocesses);
    for (int i = 0; i < nprocesses; ++i) {
      MPI_Scatter(
        &thermal_remote_reference_point_counts[0],
        1, MPI_UNSIGNED,
        &thermal_remote_remapped_point_counts[i],
        1, MPI_UNSIGNED,
        i, mpi_communicator);
    }



    {
      std::vector<std::vector<Number> > local_previous_temperature_groups(nprocesses);
      std::vector<std::vector<Number> > remote_previous_temperature_groups(nprocesses);

      for (int i = 0; i < nprocesses; ++i) {
        local_previous_temperature_groups.at(i).resize(thermal_remote_reference_point_counts.at(i));
        remote_previous_temperature_groups.at(i).resize(thermal_remote_remapped_point_counts.at(i));
      }

      std::vector<unsigned int> next_to_process(nprocesses, 0);
      for (unsigned int i = 0; i < mapping_thermal_points.size(); ++i) {
        const unsigned int group = thermal_reference_point_owning_process.at(i);
        const RemappedPoint<dim, Number> thermal_point = mapping_thermal_points.at(i);

        Quadrature<dim> thermal_point_quadrature(
          std::vector<Point<dim, Number>> (1, thermal_point.unit_cell_point));

        FEValues<dim> thermal_point_fe_values(
          mapping,
          therm_fe,
          thermal_point_quadrature,
          update_values);

        thermal_point_fe_values.reinit(thermal_point.field_cell);

        thermal_point_fe_values[temperature].get_function_values(
          thermal_nonlinear_system.previous_deformation,
          previous_temperatures);

        local_previous_temperature_groups.at(group).at(next_to_process.at(group)) = previous_temperatures[0];

        ++next_to_process.at(group);
      }



      enum MessageFlag {
        PREVIOUS_TEMPERATURE
      };

      const unsigned int
        previous_temperature_requests_offset = 0,
        request_array_size = 1;

      std::vector<MPI_Request> requests_vector(2 * nprocesses * request_array_size);

      for (int i = 0; i < nprocesses; ++i) {
        MPI_Isend(
          &local_previous_temperature_groups.at(i)[0],
          thermal_remote_reference_point_counts.at(i),
          MPI_DOUBLE, i, PREVIOUS_TEMPERATURE,
          mpi_communicator,
          &requests_vector[i + nprocesses * previous_temperature_requests_offset]);
      }

      for (int i = 0; i < nprocesses; ++i) {
        const unsigned int row_start = i + nprocesses * request_array_size;

        MPI_Irecv(
          &remote_previous_temperature_groups.at(i)[0],
          thermal_remote_remapped_point_counts.at(i),
          MPI_DOUBLE, i, PREVIOUS_TEMPERATURE,
          mpi_communicator,
          &requests_vector[row_start + nprocesses * previous_temperature_requests_offset]);

      }

      std::vector<MPI_Status> statuses_vector(2 * request_array_size * nprocesses);
      MPI_Waitall(
        2 * request_array_size * nprocesses,
        &requests_vector[0],
        &statuses_vector[0]);

      TrilinosWrappers::SparsityPattern sparsity_pattern(
        thermal_dof_system.locally_owned_dofs,
        mpi_communicator);

      DoFTools::make_sparsity_pattern(
        thermal_dof_system.dof_handler, sparsity_pattern,
        AffineConstraints<Number>(),
        false,
        Utilities::MPI::this_mpi_process(mpi_communicator));
      sparsity_pattern.compress();

      TrilinosWrappers::SparseMatrix  projection_matrix(sparsity_pattern);
      TrilinosWrappers::MPI::Vector   projection_residual(thermal_dof_system.locally_owned_dofs, mpi_communicator);

      projection_matrix = 0;
      projection_residual = 0;

      projection_matrix = 0;
      projection_residual = 0;
      FullMatrix<Number> cell_matrix(thermal_dofs_per_cell, thermal_dofs_per_cell);
      Vector<Number> cell_residual(thermal_dofs_per_cell);
      next_to_process.clear();
      next_to_process.resize(nprocesses, 0);
      for (unsigned int i = 0; i < reference_points.size(); ++i) {
        unsigned int group = thermal_point_owning_process.at(i);

        cell_matrix = 0;
        cell_residual = 0;

        ReferencePoint<dim, Number> &reference_point = reference_points[i];

        thermal_fe_values.reinit(reference_point.field_cell);

        const Number remapped_previous_temperature = remote_previous_temperature_groups.at(group).at(next_to_process.at(group));
        for(unsigned int dof_i=0; dof_i<thermal_dofs_per_cell; dof_i++) {
          cell_residual(dof_i) +=
            remapped_previous_temperature
            * thermal_fe_values[temperature].value(dof_i, reference_point.q_point);
          for(unsigned int dof_j=0; dof_j<thermal_dofs_per_cell; dof_j++) {
            cell_matrix(dof_i, dof_j) +=
              thermal_fe_values[temperature].value(dof_i, reference_point.q_point)
              * thermal_fe_values[temperature].value(dof_j, reference_point.q_point);
          }
        }

        std::vector<types::global_dof_index> local_dof_indices(thermal_dofs_per_cell);
        reference_point.field_cell->get_dof_indices(local_dof_indices);

        projection_residual.add(local_dof_indices, cell_residual);
        for(unsigned int dof_i=0; dof_i<thermal_dofs_per_cell; dof_i++) {
          for(unsigned int dof_j=0; dof_j<thermal_dofs_per_cell; dof_j++) {
            projection_matrix.add(local_dof_indices[dof_i], local_dof_indices[dof_j], cell_matrix(dof_i, dof_j));
          }
        }

        ++next_to_process.at(group);
      }

      projection_matrix.compress(
        VectorOperation::add);
      projection_residual.compress(
        VectorOperation::add);

      // solve the thermal projection system
      TrilinosWrappers::PreconditionAMG preconditioner;

      std::vector<std::vector<bool> > constant_modes;
      DoFTools::extract_constant_modes(thermal_dof_system.dof_handler, ComponentMask(), constant_modes);

      TrilinosWrappers::PreconditionAMG::AdditionalData additional_data;
      additional_data.constant_modes = constant_modes;
      additional_data.elliptic = true;
      additional_data.n_cycles = 1;
      additional_data.w_cycle = false;
      additional_data.output_details = false;
      additional_data.smoother_sweeps = 2;
      additional_data.aggregation_threshold = 1e-2;
      preconditioner.initialize(projection_matrix, additional_data);

      TrilinosWrappers::MPI::Vector tmp(thermal_dof_system.locally_owned_dofs, mpi_communicator);
      const Number relative_accuracy = 1e-08;
      const Number solver_tolerance  = relative_accuracy
                                       * projection_matrix.residual(tmp, thermal_nonlinear_system.Newton_step_solution,
                                           projection_residual);
      SolverControl solver_control(projection_matrix.m(),
                                   solver_tolerance);

      SolverBicgstab<TrilinosWrappers::MPI::Vector> solver(solver_control);

      thermal_nonlinear_system.Newton_step_solution = 0;

      solver.solve(projection_matrix, thermal_nonlinear_system.Newton_step_solution,
                   projection_residual, preconditioner);

      thermal_nonlinear_system.previous_deformation = thermal_nonlinear_system.Newton_step_solution;

    }

  }


  template<int dim, typename Number>
  void PlasticityLabProg<dim, Number>::remap_mechanical_fields(
      NewtonStepSystem &mechanical_nonlinear_system,
      const DoFSystem<dim, Number> &mechanical_dof_system,
      const NewtonStepSystem &mesh_motion_nonlinear_system,
      const DoFSystem<dim, Number> &mesh_motion_dof_system) {

    const Quadrature<dim> mechanical_fe_support_point_quadrature(mech_fe.base_element(0).get_unit_support_points());

    const unsigned int n_q_points = mechanical_fe_support_point_quadrature.size();
    const unsigned int dofs_per_cell = mesh_motion_fe.dofs_per_cell;
    const unsigned int mechanical_dofs_per_cell = mech_fe.dofs_per_cell;

    FEValues<dim> mesh_motion_fe_values(
      mapping,
      mesh_motion_fe,
      mechanical_fe_support_point_quadrature,
      update_values  | update_quadrature_points);

    FEValues<dim> mechanical_fe_values(
      mapping,
      mech_fe,
      mechanical_fe_support_point_quadrature,
      update_values | update_quadrature_points);

    std::vector< Tensor<1, dim, Number> > previous_remapped_deformations(1);
    std::vector< Tensor<1, dim, Number> > previous_remapped_velocity(1);
    std::vector< Tensor<1, dim, Number> > previous_remapped_second_time_rate(1);
    std::vector< Number > previous_remapped_twist_deformations(1);
    std::vector< Number > previous_remapped_twist_velocity(1);
    std::vector< Number > previous_remapped_twist_second_time_rate(1);
    std::vector< Tensor<1, dim, Number> > mesh_motion_value_increments(n_q_points);
    std::vector< Tensor<2, dim, Number> > mesh_motion_gradient_increments(n_q_points);

    const FEValuesExtractors::Vector displacements(0);
    const FEValuesExtractors::Scalar angular_velocities(dim);

    std::vector<ReferencePoint<dim, Number>> reference_points;
    std::vector<Point<dim, Number>> remapped_point_positions;
    std::unordered_map<point_index_t, unsigned int> quadrature_point_reference_point_id;
    auto cell = mesh_motion_dof_system.dof_handler.begin_active();
    auto mechanical_cell = mechanical_dof_system.dof_handler.begin_active();
    for (; cell != mesh_motion_dof_system.dof_handler.end(); ++cell, ++mechanical_cell) {
      if (cell->is_locally_owned()) {
        mesh_motion_fe_values.reinit(cell);

        mesh_motion_fe_values[displacements].get_function_values(
          mesh_motion_nonlinear_system.current_increment,
          mesh_motion_value_increments);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
          const Point<dim, Number> reference_point_position = mesh_motion_fe_values.quadrature_point(q_point);
          const Point<dim, Number> remapped_point_position = reference_point_position - mesh_motion_value_increments[q_point];

          ReferencePoint<dim, Number> reference_point;
          reference_point.mesh_motion_cell = cell;
          reference_point.field_cell = mechanical_cell;
          reference_point.q_point = q_point;
          reference_point.reference_point = reference_point_position;
          reference_point.remapped_point = remapped_point_position;

          reference_points.push_back(reference_point);
          remapped_point_positions.push_back(remapped_point_position);

          const point_index_t quadrature_point_index = cell->user_index() + q_point;
          quadrature_point_reference_point_id[quadrature_point_index] = reference_points.size() - 1;

        }

      }
    }

    MPI_Datatype PointType;
    MPI_Type_contiguous(dim, MPI_DOUBLE, &PointType);
    MPI_Type_commit(&PointType);
    std::vector<int> displs;
    std::vector< Point<dim, Number>> received_remapped_positions;
    std::vector<RemappedPoint<dim, Number>> remapped_points;
    int nprocesses, this_process;
    int num_reference_points = reference_points.size();
    MPI_Comm_size(mpi_communicator, &nprocesses);
    std::vector<int> reference_point_counts(nprocesses);
    MPI_Allgather(
      &num_reference_points,
      1, MPI_INT,
      &reference_point_counts[0],
      1, MPI_INT,
      mpi_communicator);
    MPI_Comm_rank(mpi_communicator, &this_process);
    displs.resize(nprocesses);
    displs[0] = 0;
    for (int i = 1; i < nprocesses; ++i) {
      displs[i] = displs[i - 1] + reference_point_counts[i - 1];
    }

    const unsigned int count_received_reference_points = displs[nprocesses - 1] + reference_point_counts[nprocesses - 1];
    received_remapped_positions.resize(count_received_reference_points);
    remapped_points.resize(count_received_reference_points);
    std::vector<char> this_process_owns_remapped_point(count_received_reference_points);

    MPI_Allgatherv(
      &remapped_point_positions[0],
      num_reference_points,
      PointType,
      &received_remapped_positions[0],
      &reference_point_counts[0],
      &displs[0],
      PointType,
      mpi_communicator);


    for(unsigned int received_point_id=0; received_point_id < count_received_reference_points; received_point_id++) {
      auto point = received_remapped_positions[received_point_id];
      auto remapped_cell_and_point = GridTools::find_active_cell_around_point(
                          mapping,
                          mechanical_dof_system.dof_handler,
                          point);
      auto mechanical_cell = remapped_cell_and_point.first;
      auto mechanical_unit_cell_point = remapped_cell_and_point.second;
      remapped_points[received_point_id].field_cell = mechanical_cell;
      remapped_points[received_point_id].unit_cell_point = mechanical_unit_cell_point;
      remapped_points[received_point_id].remapped_point = point;
      this_process_owns_remapped_point[received_point_id] = mechanical_cell.state() == IteratorState::valid && mechanical_cell->is_locally_owned()? 1 : 0;
    }

    std::vector<char> remapped_point_candidates(num_reference_points * nprocesses);
    for (int process = 0; process < nprocesses; ++process) {
      if (reference_point_counts[process] > 0)
        MPI_Gather(
          &this_process_owns_remapped_point[displs[process]],
          reference_point_counts[process],
          MPI_C_BOOL,
          &remapped_point_candidates[0],
          reference_point_counts[process],
          MPI_C_BOOL,
          process,
          mpi_communicator);
    }

    std::vector<unsigned int> remapped_point_owning_process(num_reference_points);
    for (int i = 0; i < num_reference_points; ++i) {
      remapped_point_owning_process[i] = 0;
      for (int j = 1; j < nprocesses; ++j) {
        if (remapped_point_candidates[j * num_reference_points + i] == 1) {
          remapped_point_owning_process[i] = j;
          continue;
        }
      }
    }

    std::vector<RemappedPoint<dim, Number>> mapping_remapped_points;
    std::vector<unsigned int> reference_point_owning_process;
    std::vector<unsigned int> reference_point_index_at_remote_process;

    // This should really be a vector<bool>, but addresses of individual elements of
    // vector<bool> cannot be taken. It's a template specialization to save space
    std::vector<char> remote_remapped_point_is_accepted(num_reference_points * nprocesses);
    std::vector<char> local_remapped_point_is_accepted(count_received_reference_points);
    for (int i = 0; i < num_reference_points; ++i) {
      for (unsigned int j = 0; j < static_cast<unsigned int>(nprocesses); ++j) {
        remote_remapped_point_is_accepted[j * num_reference_points + i] = (remapped_point_owning_process[i] == j) ? 1 : 0;
      }
    }

    for (int process = 0; process < nprocesses; ++process) {
      if (reference_point_counts[process] > 0) {
        MPI_Scatter(
          &remote_remapped_point_is_accepted[0],
          reference_point_counts[process],
          MPI_C_BOOL,
          &local_remapped_point_is_accepted[displs[process]],
          reference_point_counts[process],
          MPI_C_BOOL,
          process,
          mpi_communicator);
      }
    }

    std::vector<unsigned int> remote_reference_point_counts(nprocesses, 0);
    for (int process = 0; process < nprocesses; ++process) {
      for (int i = 0; i < reference_point_counts[process]; ++i) {
        if (local_remapped_point_is_accepted[displs[process] + i]) {
          RemappedPoint<dim, Number> accepted_remapped_point = remapped_points[displs[process] + i];

          mapping_remapped_points.push_back(accepted_remapped_point);
          reference_point_owning_process.push_back(process);
          reference_point_index_at_remote_process.push_back(i);
          ++remote_reference_point_counts[process];
        }
      }
    }

    MPI_Barrier(mpi_communicator);

    std::vector<unsigned int> remote_remapped_point_counts(nprocesses);
    for (int i = 0; i < nprocesses; ++i) {
      MPI_Scatter(
        &remote_reference_point_counts[0],
        1, MPI_UNSIGNED,
        &remote_remapped_point_counts[i],
        1, MPI_UNSIGNED,
        i, mpi_communicator);
    }

    std::vector<std::vector<Number> > local_previous_deformation_groups(nprocesses);
    std::vector<std::vector<Number> > local_previous_velocity_groups(nprocesses);
    std::vector<std::vector<Number> > local_previous_second_time_rate_groups(nprocesses);
    std::vector<std::vector<Number> > remote_previous_deformation_groups(nprocesses);
    std::vector<std::vector<Number> > remote_previous_velocity_groups(nprocesses);
    std::vector<std::vector<Number> > remote_previous_second_time_rate_groups(nprocesses);

    for (int i = 0; i < nprocesses; ++i) {
      local_previous_deformation_groups.at(i).resize((dim+1) * remote_reference_point_counts.at(i));
      local_previous_velocity_groups.at(i).resize((dim+1) * remote_reference_point_counts.at(i));
      local_previous_second_time_rate_groups.at(i).resize((dim+1) * remote_reference_point_counts.at(i));
      remote_previous_deformation_groups.at(i).resize((dim+1) * remote_remapped_point_counts.at(i));
      remote_previous_velocity_groups.at(i).resize((dim+1) * remote_remapped_point_counts.at(i));
      remote_previous_second_time_rate_groups.at(i).resize((dim+1) * remote_remapped_point_counts.at(i));
    }

    std::vector<unsigned int> next_to_process(nprocesses, 0);
    for (unsigned int i = 0; i < mapping_remapped_points.size(); ++i) {
      const unsigned int group = reference_point_owning_process.at(i);
      const RemappedPoint<dim, Number> remapped_point = mapping_remapped_points.at(i);

      Quadrature<dim> remapped_point_quadrature(
        std::vector<Point<dim, Number>> (1, remapped_point.unit_cell_point));

      FEValues<dim> remapped_point_fe_values(
        mapping,
        mech_fe,
        remapped_point_quadrature,
        update_values);

      remapped_point_fe_values.reinit(remapped_point.field_cell);

      remapped_point_fe_values[displacements].get_function_values(
        mechanical_nonlinear_system.previous_deformation,
        previous_remapped_deformations);

      remapped_point_fe_values[displacements].get_function_values(
        mechanical_nonlinear_system.previous_time_derivative,
        previous_remapped_velocity);

      remapped_point_fe_values[displacements].get_function_values(
        mechanical_nonlinear_system.previous_second_time_derivative,
        previous_remapped_second_time_rate);

      remapped_point_fe_values[angular_velocities].get_function_values(
        mechanical_nonlinear_system.previous_deformation,
        previous_remapped_twist_deformations);

      remapped_point_fe_values[angular_velocities].get_function_values(
        mechanical_nonlinear_system.previous_time_derivative,
        previous_remapped_twist_velocity);

      remapped_point_fe_values[angular_velocities].get_function_values(
        mechanical_nonlinear_system.previous_second_time_derivative,
        previous_remapped_twist_second_time_rate);

      for(unsigned int dim_i=0; dim_i<dim; dim_i++) {
        local_previous_deformation_groups.at(group).at(next_to_process.at(group) * (dim+1) + dim_i) =
            previous_remapped_deformations[0][dim_i];
        local_previous_velocity_groups.at(group).at(next_to_process.at(group) * (dim+1) + dim_i) =
            previous_remapped_velocity[0][dim_i];
        local_previous_second_time_rate_groups.at(group).at(next_to_process.at(group) * (dim+1) + dim_i) =
            previous_remapped_second_time_rate[0][dim_i];
      }

      local_previous_deformation_groups.at(group).at(next_to_process.at(group) * (dim+1) + dim) =
          previous_remapped_twist_deformations[0];
      local_previous_velocity_groups.at(group).at(next_to_process.at(group) * (dim+1) + dim) =
          previous_remapped_twist_velocity[0];
      local_previous_second_time_rate_groups.at(group).at(next_to_process.at(group) * (dim+1) + dim) =
          previous_remapped_twist_second_time_rate[0];

      ++next_to_process.at(group);
    }

    enum MessageFlag {
      PREVIOUS_DEFORMATION,
      PREVIOUS_VELOCITY,
      PREVIOUS_SECOND_TIME_RATE
    };

    const unsigned int
      previous_deformation_requests_offset = 0,
      previous_velocity_requests_offset = 1,
      previous_second_time_rate_requests_offset = 2,
      request_array_size = 3;

    std::vector<MPI_Request> requests_vector(2 * nprocesses * request_array_size);

    for (int i = 0; i < nprocesses; ++i) {
      MPI_Isend(
        &local_previous_deformation_groups.at(i)[0],
        (dim+1) * remote_reference_point_counts.at(i),
        MPI_DOUBLE, i, PREVIOUS_DEFORMATION,
        mpi_communicator,
        &requests_vector[i + nprocesses * previous_deformation_requests_offset]);

      MPI_Isend(
        &local_previous_velocity_groups.at(i)[0],
        (dim+1) * remote_reference_point_counts.at(i),
        MPI_DOUBLE, i, PREVIOUS_VELOCITY,
        mpi_communicator,
        &requests_vector[i + nprocesses * previous_velocity_requests_offset]);

      MPI_Isend(
        &local_previous_second_time_rate_groups.at(i)[0],
        (dim+1) * remote_reference_point_counts.at(i),
        MPI_DOUBLE, i, PREVIOUS_SECOND_TIME_RATE,
        mpi_communicator,
        &requests_vector[i + nprocesses * previous_second_time_rate_requests_offset]);
    }

    for (int i = 0; i < nprocesses; ++i) {
      const unsigned int row_start = i + nprocesses * request_array_size;

      MPI_Irecv(
        &remote_previous_deformation_groups.at(i)[0],
        (dim+1) * remote_remapped_point_counts.at(i),
        MPI_DOUBLE, i, PREVIOUS_DEFORMATION,
        mpi_communicator,
        &requests_vector[row_start + nprocesses * previous_deformation_requests_offset]);

      MPI_Irecv(
        &remote_previous_velocity_groups.at(i)[0],
        (dim+1) * remote_remapped_point_counts.at(i),
        MPI_DOUBLE, i, PREVIOUS_VELOCITY,
        mpi_communicator,
        &requests_vector[row_start + nprocesses * previous_velocity_requests_offset]);

      MPI_Irecv(
        &remote_previous_second_time_rate_groups.at(i)[0],
        (dim+1) * remote_remapped_point_counts.at(i),
        MPI_DOUBLE, i, PREVIOUS_SECOND_TIME_RATE,
        mpi_communicator,
        &requests_vector[row_start + nprocesses * previous_second_time_rate_requests_offset]);

    }

    std::vector<MPI_Status> statuses_vector(2 * request_array_size * nprocesses);
    MPI_Waitall(
      2 * request_array_size * nprocesses,
      &requests_vector[0],
      &statuses_vector[0]);

    TrilinosWrappers::SparsityPattern sparsity_pattern(
      mechanical_dof_system.locally_owned_dofs,
      mpi_communicator);

    DoFTools::make_sparsity_pattern(
      mechanical_dof_system.dof_handler, sparsity_pattern,
      AffineConstraints<Number>(),
      false,
      Utilities::MPI::this_mpi_process(mpi_communicator));
    sparsity_pattern.compress();

    TrilinosWrappers::SparseMatrix  projection_matrix(sparsity_pattern);
    TrilinosWrappers::MPI::Vector   projection_residual(mechanical_dof_system.locally_owned_dofs, mpi_communicator);
    TrilinosWrappers::MPI::Vector   projection_velocity_residual(mechanical_dof_system.locally_owned_dofs, mpi_communicator);
    TrilinosWrappers::MPI::Vector   projection_second_time_rate_residual(mechanical_dof_system.locally_owned_dofs, mpi_communicator);
    TrilinosWrappers::MPI::Vector   projection_solution(mechanical_dof_system.locally_owned_dofs, mpi_communicator);

    projection_matrix = 0;
    projection_residual = 0;
    projection_velocity_residual = 0;
    projection_second_time_rate_residual = 0;

    FullMatrix<Number> cell_matrix(mechanical_dofs_per_cell, mechanical_dofs_per_cell);
    Vector<Number> cell_residual(mechanical_dofs_per_cell);
    Vector<Number> cell_velocity_residual(mechanical_dofs_per_cell);
    Vector<Number> cell_second_time_rate_residual(mechanical_dofs_per_cell);

    next_to_process.clear();
    next_to_process.resize(nprocesses, 0);
    for (unsigned int i = 0; i < reference_points.size(); ++i) {
      unsigned int group = remapped_point_owning_process.at(i);

      cell_matrix = 0;
      cell_residual = 0;
      cell_velocity_residual = 0;
      cell_second_time_rate_residual = 0;

      ReferencePoint<dim, Number> &reference_point = reference_points[i];

      mechanical_fe_values.reinit(reference_point.field_cell);

      Tensor<1, dim, Number> remapped_previous_deformation;
      Tensor<1, dim, Number> remapped_previous_velocity;
      Tensor<1, dim, Number> remapped_previous_second_time_rate;

      for(unsigned int dim_i=0; dim_i<dim; dim_i++) {
        remapped_previous_deformation[dim_i] =
            reference_point.remapped_point[dim_i]
            - reference_point.reference_point[dim_i]
            + remote_previous_deformation_groups.at(group).at(next_to_process.at(group) * (dim+1) + dim_i);
        remapped_previous_velocity[dim_i] = remote_previous_velocity_groups.at(group).at(next_to_process.at(group) * (dim+1) + dim_i);
        remapped_previous_second_time_rate[dim_i] = remote_previous_second_time_rate_groups.at(group).at(next_to_process.at(group) * (dim+1) + dim_i);
      }

      const Number remapped_previous_twist_deformation = remote_previous_deformation_groups.at(group).at(next_to_process.at(group) * (dim+1) + dim);
      const Number remapped_previous_twist_velocity = remote_previous_velocity_groups.at(group).at(next_to_process.at(group) * (dim+1) + dim);
      const Number remapped_previous_twist_second_time_rate = remote_previous_second_time_rate_groups.at(group).at(next_to_process.at(group) * (dim+1) + dim);

      for(unsigned int dof_i=0; dof_i<mechanical_dofs_per_cell; dof_i++) {
        const auto shape_value_i = postprocess_tensor_dimension(
          mechanical_fe_values[displacements].value(dof_i, reference_point.q_point),
          mechanical_fe_values[angular_velocities].value(dof_i, reference_point.q_point));

        cell_residual(dof_i) +=
          shape_value_i * postprocess_tensor_dimension(remapped_previous_deformation, remapped_previous_twist_deformation);

        cell_velocity_residual(dof_i) +=
          shape_value_i * postprocess_tensor_dimension(remapped_previous_velocity, remapped_previous_twist_velocity);

        cell_second_time_rate_residual(dof_i) +=
          shape_value_i * postprocess_tensor_dimension(remapped_previous_second_time_rate, remapped_previous_twist_second_time_rate);

        for(unsigned int dof_j=0; dof_j<mechanical_dofs_per_cell; dof_j++) {
          const auto shape_value_j = postprocess_tensor_dimension(
            mechanical_fe_values[displacements].value(dof_j, reference_point.q_point),
            mechanical_fe_values[angular_velocities].value(dof_j, reference_point.q_point));

          cell_matrix(dof_i, dof_j) += shape_value_i * shape_value_j;

        }
      }

      std::vector<types::global_dof_index> local_dof_indices(mechanical_dofs_per_cell);
      reference_point.field_cell->get_dof_indices(local_dof_indices);
      projection_residual.add(local_dof_indices, cell_residual);
      projection_velocity_residual.add(local_dof_indices, cell_velocity_residual);
      projection_second_time_rate_residual.add(local_dof_indices, cell_second_time_rate_residual);
      for(unsigned int dof_i=0; dof_i<mechanical_dofs_per_cell; dof_i++) {
        for(unsigned int dof_j=0; dof_j<mechanical_dofs_per_cell; dof_j++) {
          projection_matrix.add(local_dof_indices[dof_i], local_dof_indices[dof_j], cell_matrix(dof_i, dof_j));
        }
      }

      ++next_to_process.at(group);
    }

    projection_matrix.compress(VectorOperation::add);
    projection_residual.compress(VectorOperation::add);
    projection_velocity_residual.compress(VectorOperation::add);
    projection_second_time_rate_residual.compress(VectorOperation::add);

    // solve the projection system
    TrilinosWrappers::PreconditionAMG preconditioner;

    std::vector<std::vector<bool> > constant_modes;
    DoFTools::extract_constant_modes(mechanical_dof_system.dof_handler, ComponentMask(), constant_modes);

    TrilinosWrappers::PreconditionAMG::AdditionalData additional_data;
    additional_data.constant_modes = constant_modes;
    additional_data.elliptic = true;
    additional_data.n_cycles = 1;
    additional_data.w_cycle = false;
    additional_data.output_details = false;
    additional_data.smoother_sweeps = 2;
    additional_data.aggregation_threshold = 1e-2;
    preconditioner.initialize(projection_matrix, additional_data);

    TrilinosWrappers::MPI::Vector tmp(mechanical_dof_system.locally_owned_dofs, mpi_communicator);
    const Number relative_accuracy = 1e-08;
    const Number solver_tolerance  = relative_accuracy
                                     * projection_matrix.residual(tmp, projection_solution,
                                         projection_residual);
    SolverControl solver_control(projection_matrix.m(),
                                 solver_tolerance);

    SolverBicgstab<TrilinosWrappers::MPI::Vector> solver(solver_control);

    projection_solution = 0;

    solver.solve(projection_matrix, projection_solution,
                 projection_residual, preconditioner);

    mechanical_nonlinear_system.previous_deformation = projection_solution;

    projection_solution = 0;
    solver.solve(projection_matrix, projection_solution,
                 projection_velocity_residual, preconditioner);
    mechanical_nonlinear_system.previous_time_derivative = projection_solution;

    projection_solution = 0;
    solver.solve(projection_matrix, projection_solution,
                 projection_second_time_rate_residual, preconditioner);
    mechanical_nonlinear_system.previous_second_time_derivative = projection_solution;

  }


  template <int dim, typename Number>
  void PlasticityLabProg<dim, Number>::assemble_mechanical_system(
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
    const bool fill_system_matrix,
    const bool update_material_state) {
    FEValues<dim> fe_values(
      mapping,
      mech_fe,
      quadrature_formula,
      update_values | update_quadrature_points | update_gradients | update_JxW_values);

    FEFaceValues<dim> fe_face_values(
      mapping,
      mech_fe,
      face_quadrature_formula,
      update_values | update_quadrature_points | update_normal_vectors | update_JxW_values);

    FEValues<dim> fe_therm_values(
      mapping,
      therm_fe,
      quadrature_formula,
      update_values);

    FEValues<dim> mesh_motion_fe_values(
      mapping,
      mesh_motion_fe,
      quadrature_formula,
      update_values | update_gradients);

    FEValues<dim> mixed_fe_values(
      mapping,
      mixed_var_fe,
      quadrature_formula,
      update_values);

    const unsigned int dofs_per_cell = mech_fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int mixed_dofs_per_cell = mixed_var_fe.dofs_per_cell;
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    std::vector< Tensor<2, dim, Number> > current_displacement_gradients(n_q_points);
    std::vector< Tensor<2, dim, Number> > displacement_gradient_increments(n_q_points);

    std::vector< Tensor<1, dim, Number> > current_displacement_values(n_q_points);
    std::vector< Tensor<1, dim, Number> > displacement_value_increments(n_q_points);

    std::vector< Tensor<2, dim, Number> > displacement_gradient_previous_time_rates(n_q_points);
    std::vector< Tensor<2, dim, Number> > displacement_gradient_previous_second_time_rates(n_q_points);

    std::vector< Tensor<1, dim, Number> > displacement_increments(n_q_points);
    std::vector< Tensor<1, dim, Number> > displacement_previous_time_rates(n_q_points);
    std::vector< Tensor<1, dim, Number> > displacement_previous_second_time_rates(n_q_points);

    std::vector< Tensor<1, dim, Number> > angular_velocity_gradient_increments(n_q_points);
    std::vector< Tensor<1, dim, Number> > angular_velocity_gradient_previous_time_rates(n_q_points);
    std::vector< Tensor<1, dim, Number> > angular_velocity_gradient_previous_second_time_rates(n_q_points);

    std::vector< Tensor<1, dim, Number> > mesh_motion_value_increments(n_q_points);
    std::vector< Tensor<2, dim, Number> > mesh_motion_gradient_increments(n_q_points);

    std::vector< Number > angular_velocity_increments(n_q_points);
    std::vector< Number > angular_velocity_previous_time_rates(n_q_points);
    std::vector< Number > angular_velocity_previous_second_time_rates(n_q_points);

    std::vector< Number > current_temperature_values(n_q_points);
    std::vector< Number > updated_temperature_increments(n_q_points);
    std::vector< Number > updated_temperature_values(n_q_points);

    std::vector< Tensor<1, dim, Number> > face_displacement_value_increments(n_face_q_points);

    std::vector< Number > deformation_jacobians(n_q_points);
    std::vector< Number > previous_deformation_jacobian(n_q_points);
    std::vector< std::vector<Number> > strain_divergences(
      dofs_per_cell,
      std::vector<Number>(n_q_points));
    std::vector< std::vector<Number> > jacobian_tangents(
      dofs_per_cell,
      std::vector<Number>(n_q_points));

    std::vector< std::vector< std::vector < Number> > > strain_divergence_tangents(
      dofs_per_cell,
      std::vector< std::vector< Number> >(
        dofs_per_cell,
        std::vector<Number>(n_q_points)));

    std::vector<Number> projected_temperature_coefficients(mixed_dofs_per_cell);
    std::vector<Number> projected_Jacobian_coefficients(mixed_dofs_per_cell);
    std::vector<Number> projected_previous_temperature_coefficients(mixed_dofs_per_cell);
    std::vector<Number> projected_previous_Jacobian_coefficients(mixed_dofs_per_cell);
    std::vector<std::vector<Number> > projected_strain_divergence_coefficients(
      dofs_per_cell,
      std::vector<Number>(mixed_dofs_per_cell));
    std::vector<std::vector<Number> > projected_jacobian_tangent_coefficients(
      dofs_per_cell,
      std::vector<Number>(mixed_dofs_per_cell));
    std::vector<std::vector<std::vector<Number> > > projected_strain_divergence_tangent_coefficients(
      dofs_per_cell, std::vector< std::vector< Number> >(
        dofs_per_cell,
        std::vector<Number>(mixed_dofs_per_cell)));

    std::vector<Number> projected_strain_divergence(dofs_per_cell);
    std::vector<Number> projected_jacobian_tangent(dofs_per_cell);
    std::vector<std::vector<Number> > projected_strain_divergence_tangent(
      dofs_per_cell,
      std::vector<Number>(dofs_per_cell));

    FullMatrix<Number> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<Number> cell_residual(dofs_per_cell);

    Vector<Number> mixed_values (mixed_dofs_per_cell);

    const FEValuesExtractors::Vector displacements (0);
    const FEValuesExtractors::Scalar angular_velocity(dim);
    const FEValuesExtractors::Scalar temperature (0);

    Newton_system.Newton_step_matrix = 0;
    Newton_system.Newton_step_residual = 0;

    double alpha_m, alpha_f, gamma, beta;
    get_generalized_alpha_method_params(
        &alpha_m, &alpha_f, &gamma, &beta, rho_infty);

    const Number d_second_time_rate_d_increment = (1./(beta*time_increment*time_increment));
    const Number d_time_rate_d_increment = gamma/(beta*time_increment);

    bool kinematic_domains_are_valid = true;  // innocent until proven guilty

    auto cell = mechanical_dof_system.dof_handler.begin_active();
    auto endc = mechanical_dof_system.dof_handler.end();
    auto thermal_cell = thermal_dof_system.dof_handler.begin_active();
    auto mesh_motion_cell = mesh_motion_dof_system.dof_handler.begin_active();
    auto mixed_fe_cell = mixed_fe_dof_system.dof_handler.begin_active();
    for (; cell != endc; ++cell, ++thermal_cell, ++mesh_motion_cell, ++mixed_fe_cell) {
      if (cell->is_locally_owned()) {
        cell_matrix = 0;
        cell_residual = 0;

        fe_values.reinit (cell);
        fe_therm_values.reinit (thermal_cell);
        mixed_fe_values.reinit (mixed_fe_cell);
        mesh_motion_fe_values.reinit (mesh_motion_cell);

        fe_values[displacements].get_function_gradients(
          Newton_system.current_increment,
          displacement_gradient_increments);

        fe_values[displacements].get_function_gradients(
          Newton_system.previous_deformation,
          current_displacement_gradients);

        fe_values[displacements].get_function_values(
          Newton_system.current_increment,
          displacement_value_increments);

        fe_values[displacements].get_function_values(
          Newton_system.previous_deformation,
          current_displacement_values);

        fe_values[displacements].get_function_gradients(
          Newton_system.previous_time_derivative,
          displacement_gradient_previous_time_rates);

        fe_values[displacements].get_function_gradients(
          Newton_system.previous_second_time_derivative,
          displacement_gradient_previous_second_time_rates);

        fe_values[displacements].get_function_values(
          Newton_system.current_increment,
          displacement_increments);

        fe_values[displacements].get_function_values(
          Newton_system.previous_time_derivative,
          displacement_previous_time_rates);

        fe_values[displacements].get_function_values(
          Newton_system.previous_second_time_derivative,
          displacement_previous_second_time_rates);

        // Angular velocity
        fe_values[angular_velocity].get_function_gradients(
          Newton_system.current_increment,
          angular_velocity_gradient_increments);

        fe_values[angular_velocity].get_function_gradients(
          Newton_system.previous_time_derivative,
          angular_velocity_gradient_previous_time_rates);

        fe_values[angular_velocity].get_function_gradients(
          Newton_system.previous_second_time_derivative,
          angular_velocity_gradient_previous_second_time_rates);

        fe_values[angular_velocity].get_function_values(
          Newton_system.current_increment,
          angular_velocity_increments);

        fe_values[angular_velocity].get_function_values(
          Newton_system.previous_time_derivative,
          angular_velocity_previous_time_rates);

        fe_values[angular_velocity].get_function_values(
          Newton_system.previous_second_time_derivative,
          angular_velocity_previous_second_time_rates);

        // mesh motion
        mesh_motion_fe_values[displacements].get_function_gradients(
          mesh_motion_nonlinear_system.current_increment,
          mesh_motion_gradient_increments);

        mesh_motion_fe_values[displacements].get_function_gradients(
          mesh_motion_nonlinear_system.current_increment,
          mesh_motion_gradient_increments);

        // temperature
        fe_therm_values[temperature].get_function_values (
          thermal_Newton_system.previous_deformation,
          current_temperature_values);
        fe_therm_values[temperature].get_function_values (
          thermal_Newton_system.current_increment,
          updated_temperature_increments);

        // get vectors for projection onto mixed fe values
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
          const point_index_t quadrature_point_index = cell->user_index() + q_point;
          updated_temperature_values.at(q_point) =
            current_temperature_values.at(q_point)
            + updated_temperature_increments.at(q_point);

          const auto current_F = get_deformation_gradient(
            current_displacement_gradients[q_point],
            current_displacement_values[q_point][0]/fe_values.quadrature_point(q_point)[0]
            );
          const Number material_Jacobian = material.get_material_Jacobian(quadrature_point_index) / determinant(current_F);

          const auto mesh_motion_gradient = get_deformation_gradient(
            -mesh_motion_gradient_increments[q_point],
            -mesh_motion_value_increments[q_point][0]/fe_values.quadrature_point(q_point)[0]);

          const Tensor<2, dim+1, Number> updated_F = get_deformation_gradient(
            current_displacement_gradients[q_point] + displacement_gradient_increments[q_point],
            (current_displacement_values[q_point][0] + displacement_value_increments[q_point][0])
              /fe_values.quadrature_point(q_point)[0]
            );


          const Number Jacobian = material_Jacobian * determinant(updated_F);
          deformation_jacobians.at(q_point) = Jacobian;
          previous_deformation_jacobian.at(q_point) = material_Jacobian * determinant(current_F);

          const auto inv_updated_F = invert(updated_F);
          std::vector<Tensor<2, dim+1, Number>> rate_gradients(dofs_per_cell);
          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            rate_gradients[i] = postprocess_tensor_dimension(
              fe_values[displacements].gradient(i, q_point),
              fe_values[displacements].value(i, q_point)[0]/fe_values.quadrature_point(q_point)[0]) * inv_updated_F;
          }

          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            const Number strain_divergence_i = trace(rate_gradients[i]);
            strain_divergences[i].at(q_point) = Jacobian * strain_divergence_i;
            if (fill_system_matrix) {
              jacobian_tangents[i].at(q_point) = Jacobian * strain_divergence_i;
              for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                strain_divergence_tangents[i][j].at(q_point) =
                  Jacobian
                  * (trace(rate_gradients[i]) * trace(rate_gradients[j])
                      - trace(rate_gradients[i] * rate_gradients[j]));
              }
            }
          }
        }

        const unsigned int cell_index = cell->user_index() / n_q_points;

        mixed_fe_projector[cell_index].project(
          &projected_temperature_coefficients,
          updated_temperature_values);
        mixed_fe_projector[cell_index].project(
          &projected_Jacobian_coefficients,
          deformation_jacobians);
        mixed_fe_projector[cell_index].project(
          &projected_previous_Jacobian_coefficients,
          previous_deformation_jacobian);
        mixed_fe_projector[cell_index].project(
          &projected_previous_temperature_coefficients,
          current_temperature_values);
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          mixed_fe_projector[cell_index].project(
            &projected_strain_divergence_coefficients[i],
            strain_divergences[i]);
          if (fill_system_matrix) {
            mixed_fe_projector[cell_index].project(
              &projected_jacobian_tangent_coefficients[i],
              jacobian_tangents[i]);
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
              mixed_fe_projector[cell_index].project(
                &projected_strain_divergence_tangent_coefficients[i][j],
                strain_divergence_tangents[i][j]);
            }
          }
        }

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
          ConstitutiveModelUpdateFlags materialUpdateFlags =
            (update_pressure | update_stress_deviator);

          if (fill_system_matrix) {
            materialUpdateFlags |=
              (update_pressure_tangent | update_stress_deviator_tangent);
          }

          ConstitutiveModelRequest<dim+1, Number> previous_constitutive_request(materialUpdateFlags);

          if (update_material_state) {
            materialUpdateFlags |= update_material_point_history;
          }

          const point_index_t quadrature_point_index = cell->user_index() + q_point;
          ConstitutiveModelRequest<dim+1, Number> constitutive_request(materialUpdateFlags);

          for (unsigned int i = 0; i < mixed_dofs_per_cell; ++i) {
            mixed_values(i) = mixed_fe_values.shape_value(i, q_point);
          }

          const Number radius = fe_values.quadrature_point(q_point)[0];
          const auto current_F = get_deformation_gradient(
            current_displacement_gradients[q_point],
            current_displacement_values[q_point][0]/radius
            );

          const auto mesh_motion_gradient = get_deformation_gradient(
            -mesh_motion_gradient_increments[q_point],
            -mesh_motion_value_increments[q_point][0]/radius);

          const Tensor<2, dim+1, Number> updated_F = get_deformation_gradient(
            current_displacement_gradients[q_point] + displacement_gradient_increments[q_point],
            (current_displacement_values[q_point][0] + displacement_value_increments[q_point][0])/radius);

          const Number material_Jacobian = material.get_material_Jacobian(quadrature_point_index) / determinant(current_F);
          const Number mesh_motion_Jacobian = determinant(mesh_motion_gradient);

          const auto inv_updated_F = invert(updated_F);
          const Number Jacobian = determinant(updated_F) * material_Jacobian;
          const Number previous_Jacobian = determinant(current_F) * material_Jacobian;


          const Number DENSITY = 8.96e-9;

          const Tensor<1, dim+1, Number> displacement_increment = postprocess_tensor_dimension(displacement_increments[q_point]);
          const Tensor<1, dim+1, Number> displacement_previous_time_rate = postprocess_tensor_dimension(displacement_previous_time_rates[q_point]);
          const Tensor<1, dim+1, Number> displacement_previous_second_time_rate = postprocess_tensor_dimension(displacement_previous_second_time_rates[q_point]);
          const Tensor<2, dim+1, Number> displacement_gradient_previous_time_rate = postprocess_tensor_dimension(
            displacement_gradient_previous_time_rates[q_point], displacement_previous_time_rates[q_point][0]/radius);
          const Tensor<2, dim+1, Number> displacement_gradient_previous_second_time_rate = postprocess_tensor_dimension(
            displacement_gradient_previous_second_time_rates[q_point], displacement_previous_second_time_rates[q_point][0]/radius);

          const Tensor<1, dim+1, Number> uc_increment = scalar_to_angular_tensor(angular_velocity_increments[q_point]);
          const Tensor<1, dim+1, Number> vc_n = scalar_to_angular_tensor(angular_velocity_previous_time_rates[q_point]);
          const Tensor<1, dim+1, Number> d_vc_d_t_n = scalar_to_angular_tensor(angular_velocity_previous_second_time_rates[q_point]);

          const Tensor<1, dim+1, Number> d2_x_dt_2_n_plus_1 =
            (1./(beta*time_increment*time_increment))
            * (displacement_increment
               - time_increment * displacement_previous_time_rate
               - time_increment * time_increment * (0.5-beta) * displacement_previous_second_time_rate);

          const Tensor<1, dim+1, Number> d_x_dt_n_plus_1 =
            displacement_previous_time_rate + time_increment * ((1-gamma) * displacement_previous_second_time_rate + gamma * d2_x_dt_2_n_plus_1);

          const Tensor<2, dim+1, Number> Grad_d_2_x_d_t_2_n_plus_1 =
            (1./(beta*time_increment*time_increment))
            * (postprocess_tensor_dimension(
                displacement_gradient_increments[q_point],
                displacement_value_increments[q_point][0]/radius)
               - time_increment * displacement_gradient_previous_time_rate
               - time_increment * time_increment * (0.5-beta) * displacement_gradient_previous_second_time_rate);

          const Tensor<2, dim+1, Number> Grad_d_x_d_t_n_plus_1 =
            displacement_gradient_previous_time_rate
            + time_increment * (
                (1.-gamma) * displacement_gradient_previous_second_time_rate
                 + gamma * Grad_d_2_x_d_t_2_n_plus_1);

          const Tensor<1, dim+1, Number> d_vc_d_t_n_plus_1 =
            (1./(beta*time_increment*time_increment))
            * (uc_increment
               - time_increment * vc_n
               - time_increment * time_increment * (0.5-beta) * d_vc_d_t_n);

          const Tensor<1, dim+1, Number> vc_n_plus_1 =
            vc_n + time_increment * ((1-gamma) * d_vc_d_t_n + gamma * d_vc_d_t_n_plus_1);

          const Number thR_increment = angular_velocity_increments[q_point];
          const Number d_thR_d_t_n = angular_velocity_previous_time_rates[q_point];
          const Number d2_thR_d_t2_n = angular_velocity_previous_second_time_rates[q_point];

          const Number d2_thR_d_t2_n_plus_1 =
            (1./(beta*time_increment*time_increment))
            * (thR_increment
               - time_increment * d_thR_d_t_n
               - time_increment * time_increment * (0.5-beta) * d2_thR_d_t2_n);

          const Number d_thR_d_t_n_plus_1 =
            d_thR_d_t_n + time_increment * ((1-gamma) * d2_thR_d_t2_n + gamma * d2_thR_d_t2_n_plus_1);

          const Number one_plus_r_over_R_n = 1.0 + current_displacement_values[q_point][0]/radius;
          const Number one_plus_r_over_R_n_plus_1 = 1.0 + (current_displacement_values[q_point][0] + displacement_value_increments[q_point][0])/radius;
          const Number d_r_d_t_n_over_R = displacement_previous_time_rate[0] / radius;
          const Number d_r_d_t_n_plus_one_over_R = d_x_dt_n_plus_1[0] / radius;

          Tensor<1, dim+1, Number> e_hat_R;
          e_hat_R[0] = 1;

          const Tensor<1, dim+1, Number> acceleration_n_plus_1 =
            d2_x_dt_2_n_plus_1
            + scalar_to_angular_tensor(
                d_r_d_t_n_plus_one_over_R * d_thR_d_t_n_plus_1
                + one_plus_r_over_R_n_plus_1 * d2_thR_d_t2_n_plus_1)
            - (1.0/radius) * one_plus_r_over_R_n_plus_1 * std::pow(d_thR_d_t_n_plus_1, 2) * e_hat_R;

          const Tensor<1, dim+1, Number> acceleration_n =
            displacement_previous_second_time_rate
            + scalar_to_angular_tensor(
                d_r_d_t_n_over_R * d_thR_d_t_n
                + one_plus_r_over_R_n * d2_thR_d_t2_n)
            - (1.0/radius) * one_plus_r_over_R_n * std::pow(d_thR_d_t_n, 2) * e_hat_R;

          const Tensor<1, dim+1, Number> acceleration_n_plus_1_minus_alpha_m =
            alpha_m * acceleration_n + (1-alpha_m) * acceleration_n_plus_1;

          Tensor<2, dim+1, Number> acceleration_n_plus_1_minus_alpha_m_tangent_modulus;
          for(unsigned int i=0; i<dim; i++) {
            acceleration_n_plus_1_minus_alpha_m_tangent_modulus[i][i] += (1-alpha_m) * d_second_time_rate_d_increment;
          }
          acceleration_n_plus_1_minus_alpha_m_tangent_modulus[dim][0] +=
            (1-alpha_m) * (d_time_rate_d_increment/radius * d_thR_d_t_n_plus_1 + 1.0/radius * d2_thR_d_t2_n_plus_1);
          acceleration_n_plus_1_minus_alpha_m_tangent_modulus[0][0] +=
            (1-alpha_m) * (-(1.0/radius) * (1.0/radius) * std::pow(d_thR_d_t_n_plus_1, 2));

          acceleration_n_plus_1_minus_alpha_m_tangent_modulus[dim][dim] +=
            (1-alpha_m)
            * (d_r_d_t_n_plus_one_over_R * d_time_rate_d_increment
               + one_plus_r_over_R_n_plus_1 * d_second_time_rate_d_increment);
          acceleration_n_plus_1_minus_alpha_m_tangent_modulus[0][dim] +=
            (1-alpha_m)
            * (-(1.0/radius) * one_plus_r_over_R_n * 2 * d_thR_d_t_n * d_time_rate_d_increment);


          const auto d2_x_dt_2_n_plus_1_minus_alpha_m =
            alpha_m * displacement_previous_second_time_rate + (1-alpha_m)*d2_x_dt_2_n_plus_1;
          const Number d2_x_dt_2_tangent_1_minus_alpha_m = (1-alpha_m)*d_second_time_rate_d_increment;

          const auto d_Fc_dt_vc_n_plus_1_minus_alpha_f =
                        alpha_m * displacement_gradient_previous_time_rate * vc_n
                        + (1-alpha_m) * Grad_d_x_d_t_n_plus_1 * vc_n_plus_1;

          const auto d_Fc_dt_vc_n_plus_1_minus_alpha_f_F_tangent = (1-alpha_m) * d_time_rate_d_increment * vc_n_plus_1;
          const auto d_Fc_dt_vc_n_plus_1_minus_alpha_f_v_tangent = (1-alpha_m) * Grad_d_x_d_t_n_plus_1;

          const auto F_c_d_vc_d_t_n_plus_1_minus_alpha_m =
            alpha_m * current_F * d_vc_d_t_n
            + (1-alpha_m) * updated_F * d_vc_d_t_n_plus_1;

          const auto F_c_d_vc_d_t_n_plus_1_minus_alpha_m_F_tangent = (1-alpha_m) * d_vc_d_t_n_plus_1;
          const auto F_c_d_vc_d_t_n_plus_1_minus_alpha_m_V_tangent = (1-alpha_m) * updated_F * d_second_time_rate_d_increment;

          std::vector<Tensor<2, dim+1, Number>> rate_gradients(dofs_per_cell);
          std::vector<Tensor<2, dim+1, Number>> angular_rate_gradients(dofs_per_cell);
          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            rate_gradients[i] =
                postprocess_tensor_dimension(
                    fe_values[displacements].gradient(i, q_point),
                    fe_values[displacements].value(i, q_point)[0]/radius) * inv_updated_F;
            angular_rate_gradients[i] =
                order_1_tensor_to_angular_gradient(
                    fe_values[angular_velocity].gradient(i, q_point),
                    -fe_values[angular_velocity].value(i, q_point)/radius);
          }

          const Tensor<2, dim+1, Number> d_X_prime_d_X = deformation_gradient_from_angular_displacement_gradient(
                    -angular_velocity_increments[q_point],
                    -angular_velocity_gradient_increments[q_point],
                    radius
                  );

          const Tensor<2, dim+1, Number> rotation_to_X_prime_frame = rotation_tensor_to_transform_B_e(
                    -angular_velocity_increments[q_point],
                    radius
                  );

          const Tensor<2, dim+1, Number> inv_d_X_prime_d_X = invert(d_X_prime_d_X);

          const auto f_m_n_plus_1 = inv_d_X_prime_d_X * rotation_to_X_prime_frame;

          // std::cout << "f_r: " << inv_d_X_prime_d_X << std::endl;
          // std::cout << "R: " << previous_elastic_deformation_transformation_tensor << std::endl;
          // std::cout << "f_m_n+1: " << f_m_n_plus_1 << std::endl;

          Number projected_jacobian = 0;
          Number projected_previous_jacobian = 0;
          Number projected_temperature = 0;
          Number projected_previous_temperature = 0;
          for (unsigned int i = 0; i < mixed_dofs_per_cell; ++i) {
            projected_jacobian +=
              mixed_values(i) * projected_Jacobian_coefficients.at(i);
            projected_previous_jacobian +=
              mixed_values(i) * projected_previous_Jacobian_coefficients.at(i);
            projected_temperature +=
              mixed_values(i) * projected_temperature_coefficients.at(i);
            projected_previous_temperature +=
              mixed_values(i) * projected_previous_temperature_coefficients.at(i);
          }

          const auto unnormalized_deformation_gradient_increment = updated_F * f_m_n_plus_1 * invert(current_F);

          const auto deformation_gradient_increment =
            std::pow(determinant(unnormalized_deformation_gradient_increment),
                     -Constants<dim, Number>::one_third()) * unnormalized_deformation_gradient_increment;

          if(std::abs(determinant(deformation_gradient_increment)-1.0) > 1e-8)
            std::cout << "determinant(deformation_gradient_increment): " << determinant(deformation_gradient_increment) << std::endl;

          if(false && std::isnan(deformation_gradient_increment.norm())) {
            std::cout << "deformation_gradient_increment is nan: " << deformation_gradient_increment << std::endl;
            std::cout << "Jacobian: " << Jacobian
                      << "\nf_m_n_plus_1: " << f_m_n_plus_1
                      << "\ndet(f_m_n_plus_1): " << determinant(f_m_n_plus_1)
                      << "\nprevious_Jacobian: " << previous_Jacobian
                      << "\nupdated_F: " << updated_F
                      << "\ncurrent_F: " << current_F
                      << "\ninvert(current_F): " << invert(current_F)
                      << std::endl;
          }

          constitutive_request.set_deformation_Jacobian(projected_jacobian);
          constitutive_request.set_unprojected_deformation_Jacobian(determinant(updated_F) * material_Jacobian);
          constitutive_request.set_temperature(projected_temperature);
          constitutive_request.set_deformation_gradient(deformation_gradient_increment);
          constitutive_request.set_time_increment(time_increment);

          try {
            material.compute_constitutive_request(constitutive_request,
                                                  quadrature_point_index);
          } catch (const MaterialDomainException &exc) {
            // std::cerr << "projected_jacobian: " << projected_jacobian
            //           << "\ndeformation_gradient_increment: " << deformation_gradient_increment
            //           << "\nupdated_F: " << updated_F
            //           << "\nf_m_n_plus_1: " << f_m_n_plus_1
            //           << "\ncurrent_F: " << current_F
            //           << "\ninvert(current_F): " << invert(current_F)
            //           << "\nJacobian: " << Jacobian
            //           << "\ndeterminant(f_m_n_plus_1): " << determinant(f_m_n_plus_1)
            //           << "\nprevious_Jacobian: " << previous_Jacobian
            //           << "\n-------------------\n"
            //           << std::endl;
            // std::cerr << exc.what() << std::endl;
            kinematic_domains_are_valid = false;
            continue;
          }

          previous_constitutive_request.set_deformation_Jacobian(projected_previous_jacobian);
          previous_constitutive_request.set_temperature(projected_previous_temperature);
          previous_constitutive_request.set_deformation_gradient(unit_symmetric_tensor<dim+1, Number>());
          previous_constitutive_request.set_time_increment(time_increment);
          previous_constitutive_request.set_is_plastic(false); // the elastic strain is known, no need for predictor-corrector procedure

          material.compute_constitutive_request(previous_constitutive_request,
                                                quadrature_point_index);

          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            projected_strain_divergence[j] =
              mixed_values(0) * projected_strain_divergence_coefficients[j][0];
            if (fill_system_matrix) {
              projected_jacobian_tangent[j] =
                mixed_values(0) * projected_jacobian_tangent_coefficients[j][0];
              for (unsigned int k = 0; k < dofs_per_cell; ++k) {
                projected_strain_divergence_tangent[j][k] =
                  mixed_values(0)
                  * projected_strain_divergence_tangent_coefficients[j][k][0];
              }
            }
          }

          for (unsigned int i = 1; i < mixed_dofs_per_cell; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
              projected_strain_divergence[j] +=
                mixed_values(i) * projected_strain_divergence_coefficients[j][i];
              if (fill_system_matrix) {
                projected_jacobian_tangent[j] +=
                  mixed_values(i) * projected_jacobian_tangent_coefficients[j][i];
                for (unsigned int k = 0; k < dofs_per_cell; ++k) {
                  projected_strain_divergence_tangent[j][k] +=
                    mixed_values(i)
                    * projected_strain_divergence_tangent_coefficients[j][k][i];
                }
              }
            }
          }

          const Number RJxW = radius / material_Jacobian * fe_values.JxW(q_point);

          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            const auto strain_i = symmetrize(rate_gradients[i] + angular_rate_gradients[i] * inv_updated_F);

            // stress deviator term
            const SymmetricTensor<2, dim+1, Number> stress_deviator =
              alpha_f * previous_constitutive_request.get_stress_deviator()
              + (1-alpha_f) * constitutive_request.get_stress_deviator();
            const Number pressure =
              alpha_f * previous_constitutive_request.get_pressure()
              + (1-alpha_f) * constitutive_request.get_pressure();
            cell_residual(i) += strain_i * stress_deviator * RJxW;

            // pressure term
            cell_residual(i) +=
              (projected_strain_divergence.at(i)) * pressure * RJxW;

            // body force term
            const unsigned int
            component_i = mech_fe.system_to_component_index(i).first;
            for ( typename std::vector<BodyForceApplier<dim, Number> >::const_iterator
                  bodyForceApplier = mechanical_lbc_system.bodyLoadAppliers.begin();
                  bodyForceApplier != mechanical_lbc_system.bodyLoadAppliers.end();
                  ++bodyForceApplier) {
              cell_residual(i) +=
                bodyForceApplier->apply(
                  component_i,
                  fe_values.shape_value (i, q_point),
                  RJxW);
            }

            // inertial term
            cell_residual(i) +=
              (postprocess_tensor_dimension(fe_values[displacements].value(i, q_point)) + scalar_to_angular_tensor(fe_values[angular_velocity].value(i, q_point)))
              * DENSITY
              * acceleration_n_plus_1_minus_alpha_m * RJxW;
          }

          if (fill_system_matrix) {
            std::vector<SymmetricTensor<2, dim+1, Number> > stress_deviator_tangents(dofs_per_cell);
            std::vector<Number> pressure_tangents(dofs_per_cell);
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {

              const Tensor<2, dim+1, Number> d_X_prime_d_X_variation =
                  deformation_gradient_from_angular_displacement_gradient_variations(
                        -angular_velocity_increments[q_point],
                        -angular_velocity_gradient_increments[q_point],
                        radius,
                        -fe_values[angular_velocity].value(i, q_point),
                        -fe_values[angular_velocity].gradient(i, q_point)
                      );

              const Tensor<2, dim+1, Number> rotation_to_X_prime_frame_variation = rotation_tensor_variation_to_transform_B_e(
                        -angular_velocity_increments[q_point],
                        -fe_values[angular_velocity].value(i, q_point),
                        radius
                      );

              const auto f_m_n_plus_1_variation_inv_f_m_n_plus_1 =
                (- inv_d_X_prime_d_X * d_X_prime_d_X_variation * inv_d_X_prime_d_X * rotation_to_X_prime_frame
                  + inv_d_X_prime_d_X * rotation_to_X_prime_frame_variation) * invert(f_m_n_plus_1);

              stress_deviator_tangents[i] = (1-alpha_f) * constitutive_request.get_stress_deviator_tangent(
                                              rate_gradients[i]
                                              - Constants<dim+1, Number>::one_third()
                                                * trace(rate_gradients[i])
                                                * unit_symmetric_tensor<dim+1, Number>()
                                              + updated_F * f_m_n_plus_1_variation_inv_f_m_n_plus_1 * inv_updated_F
                                              - Constants<dim+1, Number>::one_third()
                                                * trace(f_m_n_plus_1_variation_inv_f_m_n_plus_1)
                                                * unit_symmetric_tensor<dim+1, Number>());

              pressure_tangents[i] = (1-alpha_f) * constitutive_request.get_pressure_tangent(
                                       projected_jacobian_tangent[i]);
            }
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
              for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                // stress tangent
                const Number f_int_dev_tau = symmetrize(rate_gradients[i] + angular_rate_gradients[i] * inv_updated_F) * stress_deviator_tangents[j];
                cell_matrix(i, j) += (f_int_dev_tau) * RJxW;
                // pressure_tangent
                const Number f_int_pressure =
                  projected_strain_divergence.at(i)
                  * pressure_tangents[j];
                cell_matrix(i, j) += (f_int_pressure) * RJxW;
                // geometric_tangent
                const Tensor<2, dim+1, Number> grad_ui_grad_uj = (rate_gradients[i] + angular_rate_gradients[i] * inv_updated_F) * rate_gradients[j];
                const SymmetricTensor<2, dim+1, Number> sym_grad_ui_grad_uj = symmetrize(grad_ui_grad_uj);
                const Number f_int_geom =
                  projected_strain_divergence_tangent[i][j]
                  * constitutive_request.get_pressure()
                  - sym_grad_ui_grad_uj
                  * constitutive_request.get_stress_deviator();
                cell_matrix(i, j) += (f_int_geom) * RJxW;

                // inertial tangent
                cell_matrix(i, j) +=
                  (postprocess_tensor_dimension(fe_values[displacements].value(i, q_point))
                   + scalar_to_angular_tensor(fe_values[angular_velocity].value(i, q_point)))
                  * DENSITY
                  * (acceleration_n_plus_1_minus_alpha_m_tangent_modulus
                      * (postprocess_tensor_dimension(fe_values[displacements].value(j, q_point))
                         + scalar_to_angular_tensor(fe_values[angular_velocity].value(j, q_point))))
                  * RJxW;
              }
            }
          }
        }

        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face) {
          for (auto boundaryForceSpec: mechanical_lbc_system.boundaryLoadAppliers) {
            if (cell->face(face)->boundary_id() == boundaryForceSpec.first) {
              fe_face_values.reinit(cell, face);
              for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point) {
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                  const unsigned int component_i = mech_fe.system_to_component_index(i).first;
                  cell_residual(i) +=
                    boundaryForceSpec.second.apply(
                      component_i,
                      fe_face_values.shape_value(i, q_point),
                      fe_face_values.quadrature_point(q_point)[0] * fe_face_values.JxW(q_point));
                }
              }
            }
          }

          for(const auto boundary_unidirectional_penalty_spec: mechanical_lbc_system.boundary_unidirectional_penalty_specs) {
            if (cell->face(face)->boundary_id() == boundary_unidirectional_penalty_spec->get_boundary_id()) {
              const Number reference_displacement_increment = boundary_unidirectional_penalty_spec->get_reference_displacement_increment();
              const Number residual_force = boundary_unidirectional_penalty_spec->get_residual_force();
              const Number quadratic_spring_factor = boundary_unidirectional_penalty_spec->get_quadratic_spring_factor();

              fe_face_values.reinit(cell, face);
              fe_face_values[displacements].get_function_values(
                Newton_system.current_increment,
                face_displacement_value_increments);
              for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point) {
                const Tensor<1, dim, Number> surface_normal = fe_face_values.normal_vector(q_point);
                const Number surface_displacement = face_displacement_value_increments[q_point] * surface_normal;
                const Number surface_force =
                  -residual_force
                  + surface_displacement < -reference_displacement_increment?
                    0 :
                    -0.5 * quadratic_spring_factor * std::pow(surface_displacement + reference_displacement_increment, 2);
                const Number surface_force_tangent_modulus =
                  surface_displacement < -reference_displacement_increment?
                  0 :
                  -quadratic_spring_factor * (surface_displacement + reference_displacement_increment);
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                  cell_residual(i) -=
                    fe_face_values[displacements].value(i, q_point)
                    * surface_force * surface_normal
                    * fe_face_values.quadrature_point(q_point)[0] * fe_face_values.JxW(q_point);
                  for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    cell_matrix(i, j) -=
                      fe_face_values[displacements].value(i, q_point)
                      * surface_force_tangent_modulus * (fe_face_values[displacements].value(j, q_point) * surface_normal) * surface_normal
                      * fe_face_values.quadrature_point(q_point)[0] * fe_face_values.JxW(q_point);
                  }
                }
              }
            }
          }
        }

        // const Number relative_symmetry_norm2 = cell_matrix.relative_symmetry_norm2();
        // if(relative_symmetry_norm2 > 1e-8)
        //   std::cout << "relative_symmetry_norm2: " << cell_matrix.relative_symmetry_norm2() << std::endl;

        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        cell->get_dof_indices (local_dof_indices);
        if (fill_system_matrix) {
          mechanical_dof_system.nodal_constraints.distribute_local_to_global(
            cell_matrix,
            cell_residual,
            local_dof_indices,
            Newton_system.Newton_step_matrix,
            Newton_system.Newton_step_residual,
            true);
        } else {
          mechanical_dof_system.nodal_constraints.distribute_local_to_global(
            cell_residual, local_dof_indices,
            Newton_system.Newton_step_residual);
        }
      } /* if cell is locally owned */
    } /*for (; cell!=endc; ++cell)*/

    const unsigned short local_domain_is_valid = kinematic_domains_are_valid ? 1 : 0;
    unsigned short all_kinematic_domains_are_valid;

    // did any of the processes fail to assemble?
    MPI_Allreduce(
      &local_domain_is_valid,
      &all_kinematic_domains_are_valid, 1, MPI_UNSIGNED_SHORT,
      MPI_MIN, mpi_communicator);

    if (all_kinematic_domains_are_valid < 1) {
      throw std::runtime_error("The domain is not valid...");
    }

    if (fill_system_matrix) {
      Newton_system.Newton_step_matrix.compress(VectorOperation::add);
    }
    Newton_system.Newton_step_residual.compress(VectorOperation::add);

  } /*PlasticityLabProg<dim,Number>::assemble_mechanical_system()*/


  template <int dim, typename Number>
  void PlasticityLabProg<dim, Number>::assemble_thermal_system(
    NewtonStepSystem &Newton_system,
    NewtonStepSystem &mechanical_nonlinear_system,
    const DoFSystem<dim, Number> &thermal_dof_system,
    const LBCSystem<dim, Number, 1> &thermal_lbc_system,
    const DoFSystem<dim, Number> &mechanical_dof_system,
    const DoFSystem<dim, Number> &mixed_fe_dof_system,
    Material<dim+1, Number> &material,
    const std::vector< MixedFEProjector<dim, Number> > &mixed_fe_projector,
    const std::unordered_map<size_t, Tensor<1, dim+1, Number>> &material_area_factors,
    const bool fill_system_matrix) {

    FEValues<dim> fe_values(
      mapping,
      therm_fe,
      quadrature_formula,
      update_values | update_quadrature_points |  update_gradients | update_JxW_values);

    FEFaceValues<dim> fe_face_values(
      mapping,
      therm_fe,
      face_quadrature_formula,
      update_values
      | update_quadrature_points
      | update_JxW_values);

    FEValues<dim> fe_mech_values(
      mapping,
      mech_fe,
      quadrature_formula,
      update_values | update_quadrature_points | update_gradients);

    FEFaceValues<dim> mech_fe_face_values(
      mapping,
      mech_fe,
      face_quadrature_formula,
      update_values
      | update_gradients
      | update_quadrature_points
      | update_normal_vectors
      | update_JxW_values);

    FEValues<dim> mixed_fe_values(
      mapping,
      mixed_var_fe,
      quadrature_formula,
      update_values);

    const unsigned int dofs_per_cell = therm_fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();
    const unsigned int mixed_dofs_per_cell = mixed_var_fe.dofs_per_cell;

    FullMatrix<Number> cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<Number> cell_residual (dofs_per_cell);

    Vector<Number> mixed_values (mixed_dofs_per_cell);
    std::vector<Number> weighted_updated_J_vec(mixed_dofs_per_cell),
        weighted_current_J_vec(mixed_dofs_per_cell),
        weighted_previous_J_vec(mixed_dofs_per_cell),
        weighted_updated_theta_vec(mixed_dofs_per_cell),
        weighted_previous_theta_vec(mixed_dofs_per_cell),
        weighted_J_time_rate_vec(mixed_dofs_per_cell);
    std::vector< std::vector<Number> > weighted_shape_values(
      dofs_per_cell,
      std::vector<Number>(mixed_dofs_per_cell) );

    std::vector<Number> qp_updated_J_values(n_q_points),
        qp_previous_J_values(n_q_points),
        qp_updated_theta_values(n_q_points),
        qp_previous_theta_values(n_q_points),
        qp_J_time_rates(n_q_points);
    std::vector<std::vector<Number> > qp_shape_values(
      dofs_per_cell,
      std::vector<Number>(n_q_points));

    std::vector< Tensor<1, dim, Number> > thermal_gradient_increment(n_q_points),
        current_thermal_gradient(n_q_points);

    std::vector< Number > current_temperature_values(n_q_points),
        temperature_values_increment(n_q_points);

    std::vector< Number > current_face_temperature_values(n_face_q_points),
        face_temperature_values_increment(n_face_q_points);

    std::vector< Tensor<2, dim, Number> > current_displacement_gradients(n_q_points),
        displacement_gradient_increments(n_q_points);
    std::vector< Tensor<1, dim, Number> > current_displacement_values(n_q_points),
        displacement_value_increments(n_q_points);

    std::vector< Tensor<1, dim, Number> > current_angular_velocity_gradients(n_q_points),
        angular_velocity_gradient_increments(n_q_points),
        angular_velocity_gradient_previous_time_rates(n_q_points);

    std::vector< Number > angular_velocity_increments(n_q_points),
        current_angular_velocities(n_q_points),
        angular_velocity_previous_time_rates(n_q_points);

    std::vector< Tensor<2, dim, Number> > current_face_displacement_gradients(n_face_q_points);
    std::vector< Tensor<2, dim, Number> > face_displacement_gradient_increments(n_face_q_points);

    std::vector< Tensor<1, dim, Number> > current_face_displacement_values(n_face_q_points);
    std::vector< Tensor<1, dim, Number> > face_displacement_value_increments(n_face_q_points);

    const FEValuesExtractors::Scalar temperature (0);
    const FEValuesExtractors::Vector displacements (0);
    const FEValuesExtractors::Scalar angular_velocity(dim);
    Newton_system.Newton_step_matrix = 0;
    Newton_system.Newton_step_residual = 0;
    Newton_system.Newton_step_matrix.compress(VectorOperation::insert);
    Newton_system.Newton_step_residual.compress(VectorOperation::insert);

    auto cell = thermal_dof_system.dof_handler.begin_active();
    auto endc = thermal_dof_system.dof_handler.end();
    auto mechanical_cell = mechanical_dof_system.dof_handler.begin_active();
    auto mixed_fe_cell = mixed_fe_dof_system.dof_handler.begin_active();
    for (; cell != endc; ++cell, ++mechanical_cell, ++mixed_fe_cell) {
      if (cell->is_locally_owned()) {
        cell_matrix = 0;
        cell_residual = 0;

        fe_values.reinit (cell);
        fe_mech_values.reinit (mechanical_cell);
        mixed_fe_values.reinit (mixed_fe_cell);

        fe_values[temperature].get_function_gradients(
          Newton_system.current_increment,
          thermal_gradient_increment);

        fe_values[temperature].get_function_gradients(
          Newton_system.previous_deformation,
          current_thermal_gradient);

        fe_values[temperature].get_function_values(
          Newton_system.current_increment,
          temperature_values_increment);

        fe_values[temperature].get_function_values(
          Newton_system.previous_deformation,
          current_temperature_values);

        fe_mech_values[displacements].get_function_gradients(
          mechanical_nonlinear_system.current_increment,
          displacement_gradient_increments);

        fe_mech_values[displacements].get_function_gradients(
          mechanical_nonlinear_system.previous_deformation,
          current_displacement_gradients);

        fe_mech_values[displacements].get_function_values(
          mechanical_nonlinear_system.current_increment,
          displacement_value_increments);

        fe_mech_values[displacements].get_function_values(
          mechanical_nonlinear_system.previous_deformation,
          current_displacement_values);

        // Angular velocity
        fe_mech_values[angular_velocity].get_function_gradients(
          mechanical_nonlinear_system.current_increment,
          angular_velocity_gradient_increments);

        fe_mech_values[angular_velocity].get_function_values(
          mechanical_nonlinear_system.current_increment,
          angular_velocity_increments);

        // get vectors for projection onto mixed fe values
        for (unsigned int q_point = 0; q_point < n_q_points;
             ++q_point) {

          const Number radius = fe_mech_values.quadrature_point(q_point)[0];

          const auto previous_F = get_deformation_gradient(
            current_displacement_gradients[q_point],
            current_displacement_values[q_point][0]/radius);

          const auto updated_F = get_deformation_gradient(
            current_displacement_gradients[q_point] + displacement_gradient_increments[q_point],
            (current_displacement_values[q_point][0] + displacement_value_increments[q_point][0])/radius);

          const auto deformation_gradient_increment = postprocess_tensor_dimension(
            displacement_gradient_increments[q_point],
            displacement_value_increments[q_point][0]/radius);

          const point_index_t quadrature_point_index = cell->user_index() + q_point;
          const Number material_Jacobian = material.get_material_Jacobian(quadrature_point_index) / determinant(previous_F);

          const Number Jacobian = material_Jacobian * determinant(updated_F);
          const Number previous_Jacobian = material_Jacobian * determinant(previous_F);

          qp_previous_J_values.at(q_point) = previous_Jacobian;
          qp_updated_J_values.at(q_point) = Jacobian;
          qp_J_time_rates.at(q_point) = Jacobian * trace(deformation_gradient_increment * invert(updated_F)) / time_increment;

          qp_previous_theta_values.at(q_point) = current_temperature_values.at(q_point);
          qp_updated_theta_values.at(q_point) = current_temperature_values.at(q_point) + temperature_values_increment.at(q_point);
          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            qp_shape_values.at(i).at(q_point) = fe_values[temperature].value(i, q_point);
          }
        }

        const unsigned int cell_index = cell->user_index() / n_q_points;

        mixed_fe_projector[cell_index].project(
          &weighted_updated_J_vec,
          qp_updated_J_values);
        mixed_fe_projector[cell_index].project(
          &weighted_J_time_rate_vec,
          qp_J_time_rates);
        mixed_fe_projector[cell_index].project(
          &weighted_updated_theta_vec,
          qp_updated_theta_values);
        mixed_fe_projector[cell_index].project(
          &weighted_previous_J_vec,
          qp_previous_J_values);
        mixed_fe_projector[cell_index].project(
          &weighted_previous_theta_vec,
          qp_previous_theta_values);
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          mixed_fe_projector[cell_index].project(
            &weighted_shape_values.at(i),
            qp_shape_values.at(i));
        }

        for (unsigned int q_point = 0; q_point < n_q_points;
             ++q_point) {
          const ConstitutiveModelUpdateFlags material_update_flags =
            fill_system_matrix ?
            (update_heat_flux | update_heat_flux_tangent
             | update_mechanical_dissipation
             | update_mechanical_dissipation_tangent
             | update_stored_heat | update_stored_heat_tangent)
            :
            (update_heat_flux | update_mechanical_dissipation
             | update_stored_heat);

          const ConstitutiveModelUpdateFlags heating_update_flags =
            fill_system_matrix ?
            (update_thermoelastic_heating
             | update_thermoelastic_heating_tangent)
            :
            (update_thermoelastic_heating);

          point_index_t quadrature_point_index = cell->user_index() + q_point;
          ConstitutiveModelRequest<dim+1, Number> constitutive_request(material_update_flags);
          ConstitutiveModelRequest<dim+1, Number> heating_request(heating_update_flags);

          for (unsigned int i = 0; i < mixed_dofs_per_cell; ++i) {
            mixed_values(i) = mixed_fe_values.shape_value(i, q_point);
          }

          Number projected_updated_J = 0;
          Number projected_previous_J = 0;
          Number projected_updated_theta = 0;
          Number projected_previous_theta = 0;
          Number projected_J_time_rate = 0;
          Vector<Number> projected_shape_values(dofs_per_cell);
          for (unsigned int i = 0; i < mixed_dofs_per_cell; ++i) {
            projected_updated_J +=
              mixed_values(i) * weighted_updated_J_vec.at(i);
            projected_J_time_rate +=
              mixed_values(i) * weighted_J_time_rate_vec.at(i);
            projected_updated_theta +=
              mixed_values(i) * weighted_updated_theta_vec.at(i);
            projected_previous_J +=
              mixed_values(i) * weighted_previous_J_vec.at(i);
            projected_previous_theta +=
              mixed_values(i) * weighted_previous_theta_vec.at(i);
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
              projected_shape_values(j) +=
                mixed_values(i) * weighted_shape_values.at(j).at(i);
            }
          }

          const Number d_theta_dt_n_plus_1 = temperature_values_increment[q_point] / time_increment;

          const Number d_theta_dt_tangent = 1.0 / time_increment;

          heating_request.set_deformation_Jacobian(projected_updated_J);
          heating_request.set_deformation_Jacobian_time_rate(projected_J_time_rate);
          heating_request.set_temperature(projected_updated_theta);
          heating_request.set_previous_deformation_Jacobian(projected_previous_J);
          heating_request.set_previous_temperature(projected_previous_theta);
          heating_request.set_time_increment(time_increment);

          const auto current_F = get_deformation_gradient(
            current_displacement_gradients[q_point],
            current_displacement_values[q_point][0]/fe_mech_values.quadrature_point(q_point)[0]
            );
          const auto updated_F = get_deformation_gradient(
            current_displacement_gradients[q_point]
            + displacement_gradient_increments[q_point],
            (current_displacement_values[q_point][0]
              + displacement_value_increments[q_point][0])/fe_mech_values.quadrature_point(q_point)[0]);

          const auto inv_updated_F = invert(updated_F);
          const Number material_Jacobian = material.get_material_Jacobian(quadrature_point_index) / determinant(current_F);
          const Number previous_Jacobian = determinant(current_F) * material_Jacobian;
          const Number Jacobian = determinant(updated_F) * material_Jacobian;

          const Number radius = fe_values.quadrature_point(q_point)[0];

          const Tensor<2, dim+1, Number> d_X_prime_d_X = deformation_gradient_from_angular_displacement_gradient(
                    -angular_velocity_increments[q_point],
                    -angular_velocity_gradient_increments[q_point],
                    radius
                  );

          const Tensor<2, dim+1, Number> rotation_to_X_prime_frame = rotation_tensor_to_transform_B_e(
                    -angular_velocity_increments[q_point],
                    radius
                  );

          const Tensor<2, dim+1, Number> inv_d_X_prime_d_X = invert(d_X_prime_d_X);

          const auto f_m_n_plus_1 = inv_d_X_prime_d_X * rotation_to_X_prime_frame;

          // std::cout << "f_r: " << inv_d_X_prime_d_X << std::endl;
          // std::cout << "R: " << previous_elastic_deformation_transformation_tensor << std::endl;
          // std::cout << "f_m_n+1: " << f_m_n_plus_1 << std::endl;

          const auto unnormalized_deformation_gradient_increment = updated_F * f_m_n_plus_1 * invert(current_F);

          const auto deformation_gradient_increment =
            std::pow(determinant(unnormalized_deformation_gradient_increment),
                      -Constants<dim, Number>::one_third()) * unnormalized_deformation_gradient_increment;

          const Number previous_temperature = current_temperature_values[q_point];
          const Number updated_temperature = previous_temperature + temperature_values_increment[q_point];
          const auto thermal_gradient =
            postprocess_tensor_dimension(current_thermal_gradient[q_point] + thermal_gradient_increment[q_point]) * inv_updated_F;

          constitutive_request.set_deformation_gradient(deformation_gradient_increment);
          constitutive_request.set_temperature_time_rate(d_theta_dt_n_plus_1);
          constitutive_request.set_temperature(updated_temperature);
          constitutive_request.set_thermal_gradient(thermal_gradient);
          constitutive_request.set_time_increment(time_increment);

          material.compute_constitutive_request(
            constitutive_request,
            quadrature_point_index);
          material.compute_constitutive_request(
            heating_request,
            quadrature_point_index);

          std::vector<Tensor<1, dim+1, Number>> rate_gradients(dofs_per_cell);
          std::vector<Number> rate_temperatures(dofs_per_cell);
          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            rate_gradients[i] = postprocess_tensor_dimension(fe_values[temperature].gradient(i, q_point)) * inv_updated_F;
            rate_temperatures[i] =fe_values[temperature].value(i, q_point);
          }

          const Number RJxW = fe_values.quadrature_point(q_point)[0] / material_Jacobian * fe_values.JxW(q_point);

          for (unsigned int i = 0; i < dofs_per_cell; ++i) {

            // heat flux term
            const auto heat_flux = constitutive_request.get_heat_flux();
            cell_residual(i) += rate_gradients[i]
                                * heat_flux
                                * RJxW;

            // stored heat term
            cell_residual(i) += rate_temperatures[i]
                                * constitutive_request.get_stored_heat_rate()
                                * RJxW;

            // mechanical dissipation term
            cell_residual(i) -= rate_temperatures[i]
                                * constitutive_request.get_mechanical_dissipation()
                                * RJxW;

            // elastoplastic heating term
            cell_residual(i) += (projected_shape_values(i))
                                * heating_request.get_thermo_elastic_heating()
                                * RJxW;

            for ( typename std::vector<BodyForceApplier<dim, Number> >::const_iterator
                  bodyHeatSourceApplier = thermal_lbc_system.bodyLoadAppliers.cbegin();
                  bodyHeatSourceApplier != thermal_lbc_system.bodyLoadAppliers.cend();
                  ++bodyHeatSourceApplier) {
              cell_residual(i) += bodyHeatSourceApplier->apply(
                                    0, rate_temperatures[i],
                                    fe_values.quadrature_point(q_point)[0] * fe_values.JxW(q_point));
            }
          }

          if (fill_system_matrix) {
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
              for (unsigned int j = 0; j < dofs_per_cell; ++j) {

                // heat flux tangent
                const Number f_int_q =
                  rate_gradients[i]
                  * constitutive_request.get_heat_flux_tangent(rate_gradients[j]);
                cell_matrix(i, j) += f_int_q * RJxW;

                // stored heat rate tangent
                const Number f_int_cThetaDot =
                  rate_temperatures[i]
                  * constitutive_request.get_stored_heat_rate_tangent(d_theta_dt_tangent*rate_temperatures[j]);
                cell_matrix(i, j) += f_int_cThetaDot * RJxW;

                // mechanical dissipation tangent
                const Number f_int_mech_dissipation =
                  rate_temperatures[i]
                  * constitutive_request.get_mechanical_dissipation_tangent(rate_temperatures[j]);
                cell_matrix(i, j) -= f_int_mech_dissipation * RJxW;

                // elastoplastic heating tangent
                const Number f_int_elastoplastic_heating =
                  (projected_shape_values(i))
                  * heating_request.get_thermo_elastic_heating_tangent(projected_shape_values(j));
                cell_matrix(i, j) += f_int_elastoplastic_heating * RJxW;
              }
            }
          }
        }

        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face) {
          fe_face_values.reinit(cell, face);
          mech_fe_face_values.reinit(mechanical_cell, face);

          fe_face_values[temperature].get_function_values(
            Newton_system.previous_deformation,
            current_face_temperature_values);
          fe_face_values[temperature].get_function_values(
            Newton_system.current_increment,
            face_temperature_values_increment);

          mech_fe_face_values[displacements].get_function_gradients(
            mechanical_nonlinear_system.previous_deformation,
            current_face_displacement_gradients);
          mech_fe_face_values[displacements].get_function_gradients(
            mechanical_nonlinear_system.current_increment,
            face_displacement_gradient_increments);

          mech_fe_face_values[displacements].get_function_values(
            mechanical_nonlinear_system.previous_deformation,
            current_face_displacement_values);
          mech_fe_face_values[displacements].get_function_values(
            mechanical_nonlinear_system.current_increment,
            face_displacement_value_increments);


          for ( typename std::vector<std::pair<int, BodyForceApplier<dim, Number> > >::const_iterator
                boundaryHeatSource = thermal_lbc_system.boundaryLoadAppliers.cbegin();
                boundaryHeatSource != thermal_lbc_system.boundaryLoadAppliers.cend();
                ++boundaryHeatSource) {
            if (cell->face(face)->boundary_id() == boundaryHeatSource->first) {
              for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                for (unsigned int q_point = 0;
                     q_point < face_quadrature_formula.size();
                     ++q_point) {


                  const auto updated_F = get_deformation_gradient(
                    current_face_displacement_gradients[q_point]
                    + face_displacement_gradient_increments[q_point],
                    (current_face_displacement_values[q_point][0]
                      + face_displacement_value_increments[q_point][0])/fe_mech_values.quadrature_point(q_point)[0]);


                  const Number J = determinant(updated_F);
                  const Tensor<2, dim+1, Number> inv_deformation_gradient = invert(updated_F);
                  const Tensor<1, dim+1, Number> reference_normal = postprocess_tensor_dimension(mech_fe_face_values.normal_vector(q_point), 0);
                  const Tensor<1, dim+1, Number> F_inv_transpose_N = transpose(inv_deformation_gradient) * reference_normal;
                  const Number norm_F_inv_transpose_N = (F_inv_transpose_N).norm();

                  cell_residual(i) +=
                    boundaryHeatSource->second.apply(
                      0,
                      fe_face_values.shape_value(i, q_point),
                      norm_F_inv_transpose_N * J *
                      fe_face_values.quadrature_point(q_point)[0] * fe_face_values.JxW(q_point));
                }
              }
            }
          }

          for ( typename std::vector<std::pair<int, ConvectionBoundaryConditionApplier<dim, Number> > >::const_iterator
                convectionBC = thermal_lbc_system.convection_BC_appliers.cbegin();
                convectionBC != thermal_lbc_system.convection_BC_appliers.cend();
                ++convectionBC) {
            if (cell->face(face)->boundary_id() == convectionBC->first) {
              for (unsigned int q_point = 0;
                   q_point < face_quadrature_formula.size();
                   ++q_point) {

                const unsigned int cell_index = cell->user_index() / quadrature_formula.size();
                const unsigned int surface_point_key =
                  cell_index * GeometryInfo<dim>::faces_per_cell * n_face_q_points
                  + face * n_face_q_points
                  + q_point;

                const auto updated_F = get_deformation_gradient(
                  current_face_displacement_gradients[q_point]
                  + face_displacement_gradient_increments[q_point],
                  (current_face_displacement_values[q_point][0]
                    + face_displacement_value_increments[q_point][0])/fe_mech_values.quadrature_point(q_point)[0]);

                const Number J = determinant(updated_F);
                const Tensor<2, dim+1, Number> inv_deformation_gradient = invert(updated_F);
                const Tensor<1, dim+1, Number> reference_normal = postprocess_tensor_dimension(mech_fe_face_values.normal_vector(q_point), 0);
                const Tensor<1, dim+1, Number> F_inv_transpose_N = transpose(inv_deformation_gradient) * reference_normal;
                const Number norm_F_inv_transpose_N = (F_inv_transpose_N).norm();

                const Number RJxW = fe_face_values.quadrature_point(q_point)[0] / material_area_factors.at(surface_point_key).norm() * fe_face_values.JxW(q_point);

                const Number updated_face_temperature_value =
                  current_face_temperature_values[q_point] +
                  face_temperature_values_increment[q_point];
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                  cell_residual(i) +=
                    convectionBC->second.apply(
                      0,
                      fe_face_values.shape_value(i, q_point),
                      updated_face_temperature_value,
                      RJxW);
                  if (fill_system_matrix) {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                      cell_matrix(i, j) +=
                        convectionBC->second.apply_gradient(
                          0,
                          fe_face_values.shape_value(i, q_point),
                          fe_face_values.shape_value(j, q_point),
                          RJxW);
                    }
                  }
                }
              }
            }
          }
        }


        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        cell->get_dof_indices (local_dof_indices);
        if (fill_system_matrix) {
          thermal_dof_system.nodal_constraints.distribute_local_to_global(
            cell_matrix,
            cell_residual,
            local_dof_indices,
            Newton_system.Newton_step_matrix,
            Newton_system.Newton_step_residual,
            true);
        } else {
          thermal_dof_system.nodal_constraints.distribute_local_to_global(
            cell_residual, local_dof_indices,
            Newton_system.Newton_step_residual);
        }
      } /*for (; cell!=endc; ++cell) if(cell->is_locally_owned())*/
    }

    if (fill_system_matrix) Newton_system.Newton_step_matrix.compress(VectorOperation::add);
    Newton_system.Newton_step_residual.compress(VectorOperation::add);
  }

  template <int dim, typename Number>
  void PlasticityLabProg<dim, Number>::assemble_mesh_motion_system(
        NewtonStepSystem &mesh_motion_nonlinear_system,
        const DoFSystem<dim, Number> &mesh_motion_dof_system,
        const LBCSystem<dim, Number, dim> &mesh_motion_lbc_system,
        const NewtonStepSystem &deformation_nonlinear_system,
        const DoFSystem<dim, Number> &deformation_dof_system,
        const DoFSystem<dim, Number> &mixed_fe_dof_system,
        const std::vector< MixedFEProjector<dim, Number> > &mixed_fe_projector,
        const bool fill_system_matrix) {


    const Number mesh_motion_mu = 1.0;
    const Number mesh_motion_kappa = 5.0;
    const Number cell_jacobian_exponent = -0.0;


    FEValues<dim> deformation_fe_values(
      mapping,
      mech_fe,
      quadrature_formula,
      update_values | update_gradients);

    FEValues<dim> mesh_motion_fe_values(
      mapping,
      mesh_motion_fe,
      quadrature_formula,
      update_values  | update_gradients | update_quadrature_points |
      update_jacobians | update_JxW_values);

    FEValues<dim> mixed_fe_values(
      mapping,
      mixed_var_fe,
      quadrature_formula,
      update_values);

    const unsigned int dofs_per_cell = mesh_motion_fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int mixed_dofs_per_cell = mixed_var_fe.dofs_per_cell;

    std::vector< Tensor<2, dim, Number> > mesh_motion_gradient_increments(n_q_points);
    std::vector< Tensor<1, dim, Number> > mesh_motion_value_increments(n_q_points);

    std::vector< Tensor<2, dim, Number> > current_deformation_gradients(n_q_points);
    std::vector< Tensor<2, dim, Number> > deformation_gradient_increments(n_q_points);

    std::vector< Tensor<1, dim, Number> > current_deformation_values(n_q_points);
    std::vector< Tensor<1, dim, Number> > deformation_value_increments(n_q_points);

    std::vector< Number > mesh_motion_jacobians(n_q_points);
    std::vector< std::vector<Number> > strain_divergences(dofs_per_cell, std::vector<Number>(n_q_points));
    std::vector< std::vector<Number> > jacobian_tangents(dofs_per_cell, std::vector<Number>(n_q_points));

    std::vector< std::vector< std::vector < Number> > > strain_divergence_tangents(
      dofs_per_cell,
      std::vector< std::vector< Number> >(dofs_per_cell,std::vector<Number>(n_q_points)));

    std::vector<Number> projected_Jacobian_coefficients(mixed_dofs_per_cell);
    std::vector<std::vector<Number> > projected_strain_divergence_coefficients(
      dofs_per_cell,
      std::vector<Number>(mixed_dofs_per_cell));
    std::vector<std::vector<Number> > projected_jacobian_tangent_coefficients(
      dofs_per_cell,
      std::vector<Number>(mixed_dofs_per_cell));
    std::vector<std::vector<std::vector<Number> > > projected_strain_divergence_tangent_coefficients(
      dofs_per_cell, std::vector< std::vector< Number> >(
        dofs_per_cell,
        std::vector<Number>(mixed_dofs_per_cell)));

    std::vector<Number> projected_strain_divergence(dofs_per_cell);
    std::vector<Number> projected_jacobian_tangent(dofs_per_cell);
    std::vector<std::vector<Number> > projected_strain_divergence_tangent(dofs_per_cell, std::vector<Number>(dofs_per_cell));

    FullMatrix<Number> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<Number> cell_residual(dofs_per_cell);

    Vector<Number> mixed_values (mixed_dofs_per_cell);

    const FEValuesExtractors::Vector displacements (0);
    const FEValuesExtractors::Scalar angular_velocity(dim);
    const FEValuesExtractors::Scalar temperature (0);

    mesh_motion_nonlinear_system.Newton_step_matrix = 0;
    mesh_motion_nonlinear_system.Newton_step_residual = 0;

    bool kinematic_domains_are_valid = true;  // innocent until proven guilty

    auto cell = mesh_motion_dof_system.dof_handler.begin_active();
    auto endc = mesh_motion_dof_system.dof_handler.end();
    auto deformation_cell = deformation_dof_system.dof_handler.begin_active();
    auto mixed_fe_cell = mixed_fe_dof_system.dof_handler.begin_active();

    for (; cell != endc; ++cell, ++deformation_cell, ++mixed_fe_cell) {
      if (cell->is_locally_owned()) {

        cell_matrix = 0;
        cell_residual = 0;

        mesh_motion_fe_values.reinit (cell);
        deformation_fe_values.reinit (deformation_cell);
        mixed_fe_values.reinit (mixed_fe_cell);

        deformation_fe_values[displacements].get_function_gradients(
          deformation_nonlinear_system.previous_deformation,
          current_deformation_gradients);

        deformation_fe_values[displacements].get_function_gradients(
          deformation_nonlinear_system.current_increment,
          deformation_gradient_increments);

        deformation_fe_values[displacements].get_function_values(
          deformation_nonlinear_system.previous_deformation,
          current_deformation_values);

        deformation_fe_values[displacements].get_function_values(
          deformation_nonlinear_system.current_increment,
          deformation_value_increments);

        mesh_motion_fe_values[displacements].get_function_gradients(
          mesh_motion_nonlinear_system.current_increment,
          mesh_motion_gradient_increments);

        mesh_motion_fe_values[displacements].get_function_values(
          mesh_motion_nonlinear_system.current_increment,
          mesh_motion_value_increments);

        // get vectors for projection onto mixed fe values
        for (unsigned int q_point = 0; q_point < n_q_points;
             ++q_point) {

          const auto mesh_motion_gradient = get_deformation_gradient(
                                              -mesh_motion_gradient_increments[q_point],
                                              -mesh_motion_value_increments[q_point][0]/mesh_motion_fe_values.quadrature_point(q_point)[0]);

          const Number Jacobian = determinant(mesh_motion_gradient);
          mesh_motion_jacobians.at(q_point) = Jacobian;

          const auto inv_mesh_motion_gradient = invert(mesh_motion_gradient);
          std::vector<Tensor<2, dim+1, Number>> rate_gradients(dofs_per_cell);
          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            rate_gradients[i] = postprocess_tensor_dimension(
                                  -mesh_motion_fe_values[displacements].gradient(i, q_point),
                                  -mesh_motion_fe_values[displacements].value(i, q_point)[0]
                                    /mesh_motion_fe_values.quadrature_point(q_point)[0]) * inv_mesh_motion_gradient;
          }

          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            const Number strain_divergence_i = trace(rate_gradients[i]);
            strain_divergences[i].at(q_point) = Jacobian * strain_divergence_i;
            if (fill_system_matrix) {
              jacobian_tangents[i].at(q_point) = Jacobian * strain_divergence_i;
              for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                strain_divergence_tangents[i][j].at(q_point) = Jacobian * (trace(rate_gradients[i]) * trace(rate_gradients[j]) - trace(rate_gradients[i] * rate_gradients[j]));
              }
            }
          }
        }

        const unsigned int cell_index = cell->user_index() / n_q_points;

        mixed_fe_projector[cell_index].project(
          &projected_Jacobian_coefficients,
          mesh_motion_jacobians);
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          mixed_fe_projector[cell_index].project(
            &projected_strain_divergence_coefficients[i],
            strain_divergences[i]);
          if (fill_system_matrix) {
            mixed_fe_projector[cell_index].project(
              &projected_jacobian_tangent_coefficients[i],
              jacobian_tangents[i]);
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
              mixed_fe_projector[cell_index].project(
                &projected_strain_divergence_tangent_coefficients[i][j],
                strain_divergence_tangents[i][j]);
            }
          }
        }

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {

          const point_index_t quadrature_point_index = cell->user_index() + q_point;

          const Number cell_jacobian = determinant(static_cast<Tensor <2, dim, Number>>(mesh_motion_fe_values.jacobian(q_point)));

          for (unsigned int i = 0; i < mixed_dofs_per_cell; ++i) {
            mixed_values(i) = mixed_fe_values.shape_value(i, q_point);
          }

          const auto mesh_motion_gradient = get_deformation_gradient(
                                              -mesh_motion_gradient_increments[q_point],
                                              -mesh_motion_value_increments[q_point][0]/mesh_motion_fe_values.quadrature_point(q_point)[0]);

          const auto deformation_gradient = get_deformation_gradient(
                                              current_deformation_gradients[q_point] + deformation_gradient_increments[q_point],
                                              (current_deformation_values[q_point][0] + deformation_value_increments[q_point][0])
                                              / mesh_motion_fe_values.quadrature_point(q_point)[0]);

          const auto inv_mesh_motion_gradient = invert(mesh_motion_gradient);
          const auto inv_deformation_gradient = invert(deformation_gradient);

          const Number deformation_Jacobian = determinant(deformation_gradient);
          const Number mesh_motion_Jacobian = determinant(mesh_motion_gradient);

          Number projected_jacobian = 0;
          for (unsigned int i = 0; i < mixed_dofs_per_cell; ++i) {
            projected_jacobian +=
              mixed_values(i) * projected_Jacobian_coefficients.at(i);
          }

          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            projected_strain_divergence[j] =
              mixed_values(0) * projected_strain_divergence_coefficients[j][0];
            if (fill_system_matrix) {
              projected_jacobian_tangent[j] =
                mixed_values(0) * projected_jacobian_tangent_coefficients[j][0];
              for (unsigned int k = 0; k < dofs_per_cell; ++k) {
                projected_strain_divergence_tangent[j][k] =
                  mixed_values(0)
                  * projected_strain_divergence_tangent_coefficients[j][k][0];
              }
            }
          }

          for (unsigned int i = 1; i < mixed_dofs_per_cell; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
              projected_strain_divergence[j] +=
                mixed_values(i) * projected_strain_divergence_coefficients[j][i];
              if (fill_system_matrix) {
                projected_jacobian_tangent[j] +=
                  mixed_values(i) * projected_jacobian_tangent_coefficients[j][i];
                for (unsigned int k = 0; k < dofs_per_cell; ++k) {
                  projected_strain_divergence_tangent[j][k] +=
                    mixed_values(i)
                    * projected_strain_divergence_tangent_coefficients[j][k][i];
                }
              }
            }
          }

          std::vector<Tensor<2, dim+1, Number>> rate_gradients(dofs_per_cell);
          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            rate_gradients[i] = postprocess_tensor_dimension(
                                  -mesh_motion_fe_values[displacements].gradient(i, q_point),
                                  -mesh_motion_fe_values[displacements].value(i, q_point)[0]
                                    /mesh_motion_fe_values.quadrature_point(q_point)[0]) * inv_mesh_motion_gradient;
          }

          const SymmetricTensor<2, dim+1, Number> stress_deviator =
                  std::pow(cell_jacobian, cell_jacobian_exponent) * mesh_motion_mu
                  * std::pow(mesh_motion_Jacobian, -Constants<dim, Number>::two_thirds())
                  * deviator(
                      symmetrize(
                        (deformation_gradient * mesh_motion_gradient)
                        * transpose(deformation_gradient * mesh_motion_gradient)));

          const Number pressure = mesh_motion_kappa * std::log(projected_jacobian);

          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            const auto strain_i = deformation_gradient * rate_gradients[i] * inv_deformation_gradient;

            // stress deviator term
            cell_residual(i) += symmetrize(strain_i) * stress_deviator
                                * mesh_motion_fe_values.quadrature_point(q_point)[0] * mesh_motion_fe_values.JxW(q_point);

            // pressure term
            cell_residual(i) += (projected_strain_divergence.at(i) * pressure)
                                  * mesh_motion_fe_values.quadrature_point(q_point)[0] * mesh_motion_fe_values.JxW(q_point);

            if (fill_system_matrix) {

              for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                const auto strain_j = deformation_gradient * rate_gradients[j] * inv_deformation_gradient;

                // stress tangent
                const SymmetricTensor<2, dim+1, Number> stress_deviator_tangent_j =
                    std::pow(cell_jacobian, cell_jacobian_exponent) * mesh_motion_mu
                    * std::pow(mesh_motion_Jacobian, -Constants<dim, Number>::two_thirds())
                    * deviator(
                        symmetrize(
                          2 * (strain_j - Constants<dim, Number>::one_third() * trace(strain_j) * unit_symmetric_tensor<dim+1, Number>())
                          * (deformation_gradient * mesh_motion_gradient)
                          * transpose(deformation_gradient * mesh_motion_gradient)));

                cell_matrix(i, j) += (symmetrize(strain_i) * stress_deviator_tangent_j - symmetrize(strain_i * strain_j) * stress_deviator)
                                     * mesh_motion_fe_values.quadrature_point(q_point)[0] * mesh_motion_fe_values.JxW(q_point);

                // pressure_tangent
                const Number pressure_tangent_j = mesh_motion_kappa * (1.0 / projected_jacobian) * projected_jacobian_tangent[j];
                cell_matrix(i, j) += (projected_strain_divergence.at(i) * pressure_tangent_j + projected_strain_divergence_tangent[i][j] * pressure)
                                      * mesh_motion_fe_values.quadrature_point(q_point)[0] * mesh_motion_fe_values.JxW(q_point);

              }
            }
          }
        }

        // const Number relative_symmetry_norm2 = cell_matrix.relative_symmetry_norm2();
        // if(relative_symmetry_norm2 > 1e-8)
        //   std::cout << "relative_symmetry_norm2: " << cell_matrix.relative_symmetry_norm2() << std::endl;

        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        cell->get_dof_indices (local_dof_indices);
        if (fill_system_matrix) {
          mesh_motion_dof_system.nodal_constraints.distribute_local_to_global(
            cell_matrix,
            cell_residual,
            local_dof_indices,
            mesh_motion_nonlinear_system.Newton_step_matrix,
            mesh_motion_nonlinear_system.Newton_step_residual,
            true);
        } else {
          mesh_motion_dof_system.nodal_constraints.distribute_local_to_global(
            cell_residual, local_dof_indices,
            mesh_motion_nonlinear_system.Newton_step_residual);
        }
      } /* if cell is locally owned */
    } /*for (; cell!=endc; ++cell)*/

    const unsigned short local_domain_is_valid = kinematic_domains_are_valid ? 1 : 0;
    unsigned short all_kinematic_domains_are_valid;

    // did any of the processes fail to assemble?
    MPI_Allreduce(
      &local_domain_is_valid,
      &all_kinematic_domains_are_valid, 1, MPI_UNSIGNED_SHORT,
      MPI_MIN, mpi_communicator);

    if (all_kinematic_domains_are_valid < 1) {
      throw std::runtime_error("The domain is not valid...");
    }

    if (fill_system_matrix) {
      mesh_motion_nonlinear_system.Newton_step_matrix.compress(VectorOperation::add);
    }
    mesh_motion_nonlinear_system.Newton_step_residual.compress(VectorOperation::add);

  }

  template<typename BlockType>
  class SumOfMatrices : public Subscriptor {
  public:
    SumOfMatrices(
        const BlockType &m1,
        const BlockType &m2):
      m1(m1),
      m2(m2) {
    }

    template<typename VectorType>
    void vmult(VectorType &dst, const VectorType &src) const {
      m1.vmult(dst, src);
      m2.vmult_add(dst, src);
    }

  private:
    const BlockType &m1;
    const BlockType &m2;
  };

// TODO encorporate mechanical and thermal subsystems into structs and include functions in them
  template <int dim, typename Number>
  void PlasticityLabProg<dim, Number>::solve_system(
    const DoFSystem<dim, Number> &dof_system,
    NewtonStepSystem &nonlinear_system,
    const bool reset_solution) {
    TrilinosWrappers::PreconditionAMG preconditioner;

    std::vector<std::vector<bool> > constant_modes;
    DoFTools::extract_constant_modes(dof_system.dof_handler, ComponentMask(),
                                     constant_modes);

    TrilinosWrappers::PreconditionAMG::AdditionalData additional_data;
    additional_data.constant_modes = constant_modes;
    additional_data.elliptic = true;
    additional_data.n_cycles = 1;
    additional_data.w_cycle = false;
    additional_data.output_details = false;
    additional_data.smoother_sweeps = 2;
    additional_data.aggregation_threshold = 1e-2;
    preconditioner.initialize(nonlinear_system.Newton_step_matrix, additional_data);

    TrilinosWrappers::MPI::Vector tmp(dof_system.locally_owned_dofs, mpi_communicator);
    const Number relative_accuracy = 1e-08;
    const Number solver_tolerance  = relative_accuracy
                                     * nonlinear_system.Newton_step_matrix.residual(tmp, nonlinear_system.Newton_step_solution,
                                         nonlinear_system.Newton_step_residual);
    SolverControl solver_control(nonlinear_system.Newton_step_matrix.m(),
                                 solver_tolerance);

    SolverBicgstab<TrilinosWrappers::MPI::Vector> solver(solver_control);

    if (reset_solution) {
      nonlinear_system.Newton_step_solution = 0;
      nonlinear_system.Newton_step_solution.compress(VectorOperation::insert);
    }

    solver.solve(nonlinear_system.Newton_step_matrix, nonlinear_system.Newton_step_solution,
                 nonlinear_system.Newton_step_residual, preconditioner);

    pcout << "solved in " << solver_control.last_step() << " steps to residual value of " << solver_control.last_value() << endl;
    pcout << "solution norm is: " << nonlinear_system.Newton_step_solution.l2_norm() << endl;

    dof_system.nodal_constraints.distribute (nonlinear_system.Newton_step_solution);
  } /*ElasticProblem<dim,Number>::solve_system*/


  template <int dim, typename Number>
  void PlasticityLabProg<dim, Number>::get_plastic_strain(
    TrilinosWrappers::MPI::Vector &plastic_strain,
    const DoFHandler<dim> &discontinuous_dof_handler,
    const Material<dim+1, Number> &material,
    const std::vector< MixedFEProjector<dim, Number> > &qp_values_projectors) {

    const FiniteElement<dim> &fe = discontinuous_dof_handler.get_fe();

    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int disc_dofs_per_cell = fe.dofs_per_cell;

    std::vector<Number> plastic_strain_qp_values(n_q_points);
    std::vector<Number> projected_plastic_strains(disc_dofs_per_cell);

    auto cell = discontinuous_dof_handler.begin_active();
    auto endc = discontinuous_dof_handler.end();
    for (; cell != endc; ++cell) {
      if (cell->is_locally_owned()) {
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
          const point_index_t quadrature_point_index = cell->user_index() + q_point;
          plastic_strain_qp_values[q_point] = std::exp(material.get_state_parameters(quadrature_point_index).at(0)) - 1;
        }

        const unsigned int cell_index = cell->user_index() / n_q_points;
        qp_values_projectors[cell_index].project(
          &projected_plastic_strains,
          plastic_strain_qp_values);

        std::vector<types::global_dof_index> local_dof_indices (disc_dofs_per_cell);
        cell->get_dof_indices (local_dof_indices);
        for (unsigned int dof_i = 0; dof_i < disc_dofs_per_cell; ++dof_i) {
          plastic_strain(local_dof_indices[dof_i]) = projected_plastic_strains.at(dof_i);
        }
      }
    }
  }


  template <int dim, typename Number>
  void PlasticityLabProg<dim, Number>::get_pressure(
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
    const std::vector< MixedFEProjector<dim, Number> > &qp_values_projectors) {

    FEValues<dim> fe_values(
      mapping,
      mech_fe,
      quadrature_formula,
      update_values | update_quadrature_points |  update_gradients);

    FEValues<dim> fe_therm_values(
      mapping,
      therm_fe,
      quadrature_formula,
      update_values);

    FEValues<dim> mixed_fe_values(
      mapping,
      mixed_var_fe,
      quadrature_formula,
      update_values);

    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int mixed_dofs_per_cell = mixed_var_fe.dofs_per_cell;
    const unsigned int disc_dofs_per_cell = discontinuous_dof_handler.get_fe().dofs_per_cell;


    std::vector< Tensor<2, dim, Number> > current_displacement_gradients(n_q_points);
    std::vector< Tensor<2, dim, Number> > displacement_gradient_increments(n_q_points);
    std::vector< Tensor<1, dim, Number> > current_displacement_values(n_q_points);
    std::vector< Tensor<1, dim, Number> > displacement_value_increments(n_q_points);

    std::vector< Number > current_temperature_values(n_q_points);
    std::vector< Number > updated_temperature_increments(n_q_points);
    std::vector< Number > deformation_jacobians(n_q_points);
    std::vector< Number > pressure_values(n_q_points);
    std::vector< Number > von_mises_stress_values(n_q_points);

    std::vector<Number> projected_temperature_coefficients(mixed_dofs_per_cell);
    std::vector<Number> projected_Jacobian_coefficients(mixed_dofs_per_cell);
    std::vector<Number> projected_pressure_coefficients(mixed_dofs_per_cell);
    std::vector<Number> projected_von_mises_stress_coefficients(disc_dofs_per_cell);

    Vector<Number> mixed_values(mixed_dofs_per_cell);

    const FEValuesExtractors::Vector displacements (0);
    const FEValuesExtractors::Scalar temperature (0);

    auto cell = mechanical_dof_system.dof_handler.begin_active();
    auto endc = mechanical_dof_system.dof_handler.end();
    auto thermal_cell = thermal_dof_system.dof_handler.begin_active();
    auto mixed_fe_cell = mixed_fe_dof_handler.begin_active();
    auto discontinuous_fe_cell = discontinuous_dof_handler.begin_active();
    for (; cell != endc; ++cell, ++thermal_cell, ++mixed_fe_cell, ++discontinuous_fe_cell) {
      if (cell->is_locally_owned()) {

        fe_values.reinit (cell);
        fe_therm_values.reinit (thermal_cell);
        mixed_fe_values.reinit (mixed_fe_cell);

        fe_values[displacements].get_function_gradients(
          Newton_system.current_increment,
          displacement_gradient_increments);

        fe_values[displacements].get_function_gradients(
          Newton_system.previous_deformation,
          current_displacement_gradients);

        fe_values[displacements].get_function_values(
          Newton_system.current_increment,
          displacement_value_increments);

        fe_values[displacements].get_function_values(
          Newton_system.previous_deformation,
          current_displacement_values);

        fe_therm_values[temperature].get_function_values (
          thermal_Newton_system.previous_deformation,
          current_temperature_values);

        fe_therm_values[temperature].get_function_values (
          thermal_Newton_system.current_increment,
          updated_temperature_increments);

        // get vectors for projection onto mixed fe values
        for (unsigned int q_point = 0; q_point < n_q_points;
             ++q_point) {
          updated_temperature_increments.at(q_point) += current_temperature_values.at(q_point);

          const auto current_F = get_deformation_gradient(
            current_displacement_gradients[q_point],
            current_displacement_values[q_point][0]/fe_values.quadrature_point(q_point)[0]
            );
          const auto updated_F = get_deformation_gradient(
            current_displacement_gradients[q_point]
            + displacement_gradient_increments[q_point],
            (current_displacement_values[q_point][0]
              + displacement_value_increments[q_point][0])/fe_values.quadrature_point(q_point)[0]);

          const point_index_t quadrature_point_index = cell->user_index() + q_point;
          const Number material_Jacobian = material.get_material_Jacobian(quadrature_point_index) / determinant(current_F);

          deformation_jacobians.at(q_point) = material_Jacobian * determinant(updated_F);
        }

        const unsigned int cell_index = cell->user_index() / n_q_points;

        mixed_fe_projector[cell_index].project(
          &projected_temperature_coefficients,
          updated_temperature_increments);
        mixed_fe_projector[cell_index].project(
          &projected_Jacobian_coefficients,
          deformation_jacobians);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {

          const point_index_t quadrature_point_index = cell->user_index() + q_point;
          ConstitutiveModelRequest<dim+1, Number> constitutive_request(update_pressure | update_stress_deviator);

          for (unsigned int i = 0; i < mixed_dofs_per_cell; ++i) {
            mixed_values(i) = mixed_fe_values.shape_value(i, q_point);
          }

          const auto current_F = get_deformation_gradient(
            current_displacement_gradients[q_point],
            current_displacement_values[q_point][0]/fe_values.quadrature_point(q_point)[0]
            );
          const auto updated_F = get_deformation_gradient(
            current_displacement_gradients[q_point]
            + displacement_gradient_increments[q_point],
            (current_displacement_values[q_point][0]
              + displacement_value_increments[q_point][0])/fe_values.quadrature_point(q_point)[0]);

          const auto inv_updated_F = invert(updated_F);
          const Number Jacobian = determinant(updated_F);
          const Number previous_Jacobian = determinant(current_F);

          const auto deformation_gradient_increment = std::pow(Jacobian / previous_Jacobian, -Constants<dim, Number>::one_third()) * updated_F * invert(current_F);

          Number projected_jacobian = 0;
          Number projected_temperature = 0;
          for (unsigned int i = 0; i < mixed_dofs_per_cell; ++i) {
            projected_jacobian +=
              mixed_values(i) * projected_Jacobian_coefficients.at(i);
            projected_temperature +=
              mixed_values(i) * projected_temperature_coefficients.at(i);
          }

          constitutive_request.set_deformation_Jacobian(projected_jacobian);
          constitutive_request.set_temperature(projected_temperature);
          constitutive_request.set_deformation_gradient(deformation_gradient_increment);
          constitutive_request.set_time_increment(time_increment);

          material.compute_constitutive_request(constitutive_request,
                                                quadrature_point_index);

          // pressure term
          pressure_values.at(q_point) = constitutive_request.get_pressure();
          von_mises_stress_values.at(q_point) =
            constitutive_request.get_stress_deviator().norm() / Constants<dim, Number>::sqrt2thirds();
        }

        mixed_fe_projector[cell_index].project(
          &projected_pressure_coefficients,
          pressure_values);

        qp_values_projectors[cell_index].project(
          &projected_von_mises_stress_coefficients,
          von_mises_stress_values);

        std::vector<types::global_dof_index> local_dof_indices (mixed_dofs_per_cell);
        mixed_fe_cell->get_dof_indices (local_dof_indices);
        for (unsigned int dof_i = 0; dof_i < mixed_dofs_per_cell; ++dof_i) {
          pressure(local_dof_indices[dof_i]) = projected_pressure_coefficients.at(dof_i);
        }

        std::vector<types::global_dof_index> local_discontinuous_dof_indices(disc_dofs_per_cell);
        discontinuous_fe_cell->get_dof_indices (local_discontinuous_dof_indices);
        for (unsigned int dof_i = 0; dof_i < disc_dofs_per_cell; ++dof_i) {
          von_mises_stress(local_discontinuous_dof_indices[dof_i]) =
            projected_von_mises_stress_coefficients.at(dof_i);
        }

      } /* if cell is locally owned */
    } /*for (; cell!=endc; ++cell)*/
  }

  template <int dim, typename Number>
  void PlasticityLabProg<dim, Number>::prepare_output_results(
    DataOut<dim> &data_out,
    const DoFSystem<dim, Number> &mechanical_dof_system,
    const NewtonStepSystem &mechanical_nonlinear_system,
    const DoFSystem<dim, Number> &thermal_dof_system,
    const NewtonStepSystem &thermal_nonlinear_system,
    const DoFSystem<dim, Number> &mesh_motion_dof_system,
    const NewtonStepSystem &mesh_motion_nonlinear_system) const {

    std::vector<std::string> displacement_names(dim, "displacement");
    displacement_names.emplace_back("angular_displacement");
    std::vector<std::string> velocity_names(dim, "displacement_time_rate");
    velocity_names.emplace_back("angular_velocity");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      mesh_motion_data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);

    data_out.add_data_vector(mechanical_dof_system.dof_handler,
                             mechanical_nonlinear_system.previous_deformation,
                             displacement_names,
                             data_component_interpretation);
    data_out.add_data_vector(mechanical_dof_system.dof_handler,
                             mechanical_nonlinear_system.previous_time_derivative,
                             velocity_names,
                             data_component_interpretation);
    data_out.add_data_vector(thermal_dof_system.dof_handler,
                             thermal_nonlinear_system.previous_deformation,
                             "Temperature");
    data_out.add_data_vector(mesh_motion_dof_system.dof_handler,
                             mesh_motion_nonlinear_system.previous_deformation,
                             std::vector<std::string>(dim, "mesh_motion"),
                             mesh_motion_data_component_interpretation);
    data_out.add_data_vector(mesh_motion_dof_system.dof_handler,
                             mesh_motion_nonlinear_system.previous_time_derivative,
                             std::vector<std::string>(dim, "mesh_velocity"),
                             mesh_motion_data_component_interpretation);
    data_out.build_patches(mapping, 2);
  }

  template <int dim, typename Number>
  template <typename TriangulationType>
  void PlasticityLabProg<dim, Number>::write_output_results(
    DataOut<dim> &data_out,
    const TriangulationType &tria,
    const std::string &filename_base) const {

    const std::string filename =
      (filename_base + "-"
       + Utilities::int_to_string(tria.locally_owned_subdomain(), 4));

    std::ofstream output_vtu((filename + ".vtu").c_str());
    data_out.write_vtu(output_vtu);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
      std::vector<std::string> filenames;
      for (unsigned int i = 0;
           i < Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
        filenames.push_back(filename_base + "-"
                            + Utilities::int_to_string(i, 4)
                            + ".vtu");
      std::ofstream pvtu_master_output((filename_base + ".pvtu").c_str());
      data_out.write_pvtu_record(pvtu_master_output, filenames);
      std::ofstream visit_master_output((filename_base + ".visit").c_str());
      DataOutBase::write_visit_record(visit_master_output, filenames);
    }
  } /* output_results */

  template class PlasticityLabProg<2, double>;

} /*namespace PlasticityLab*/
