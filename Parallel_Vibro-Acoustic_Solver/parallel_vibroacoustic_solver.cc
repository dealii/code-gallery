/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
 * Copyright (C) 2026 by Andreas Hegendörfer <andi.hegendoerfer@gmail.com>
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

namespace VibroAcousticProblem
{
  using namespace dealii;
  // geometric tolerance
  constexpr double geom_tol = 1e-10;

  enum class SurfaceID : dealii::types::boundary_id
  {
    // By default, boundary indicators are 0
    Default,
    SourceSide,
    ReceiverSide,
    FixedBoundary,
    ZeroPressure
  };

  enum class MaterialID : dealii::types::material_id
  {
    Concrete,
    Air
  };

  // return the isotropic loss factor
  std::complex<double>
  get_iso_loss()
  {
    return std::complex<double>{1., 0.01};
  }

  // calculation of the linear strain tensor
  template <int dim>
  inline SymmetricTensor<2, dim>
  get_strain(const FEValues<dim> &fe_values,
             const unsigned int   shape_func,
             const unsigned int   q_point)
  {
    SymmetricTensor<2, dim> tmp;

    for (unsigned int i = 0; i < dim; ++i)
      tmp[i][i] = fe_values.shape_grad_component(shape_func, q_point, i)[i];

    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = i + 1; j < dim; ++j)
        tmp[i][j] =
          (fe_values.shape_grad_component(shape_func, q_point, i)[j] +
           fe_values.shape_grad_component(shape_func, q_point, j)[i]) /
          2;

    return tmp;
  }

  // Returning the stiffness tensor of the wall
  template <int dim>
  SymmetricTensor<4, dim>
  get_stiffness_tensor()
  {
    const double E      = 31600. * 1e6;
    const double v      = 0.2;
    double       lambda = v / (1 - 2 * v) * 1 / (1 + v) * E;
    double       mu     = 0.5 * 1 / (1 + v) * E;

    SymmetricTensor<4, dim> stiffness_tensor;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        for (unsigned int k = 0; k < dim; ++k)
          for (unsigned int l = 0; l < dim; ++l)
            stiffness_tensor[i][j][k][l] =
              (((i == k) && (j == l) ? mu : 0.0) +
               ((i == l) && (j == k) ? mu : 0.0) +
               ((i == j) && (k == l) ? lambda : 0.0));
    return stiffness_tensor;
  }

  // Returning the density of the wall
  double
  get_density_structure()
  {
    return 2.275 * 1e-3;
  }

  // Returning the density of air
  double
  get_density_air()
  {
    return 1.204 * 1e-6;
  }

  // Returning the speed of sound
  double
  get_sound_speed()
  {
    return 343. * 1e3;
  }

  // Implementation of a perfectly matched layer
  template <int dim>
  class PML : public Function<dim, std::complex<double>>
  {
  public:
    explicit PML(double omega)
      : omega(omega)
      , b_pos{}
      , b_neg{}
      , t_pos{}
      , t_neg{}
      , pml_coeff_degree(2.0)
      , pml_coeff(1.e4 / omega)
    {
      Assert(dim == 3, ExcNotImplemented());

      // x-direction
      b_pos[0] = 812.0;
      b_neg[0] = std::numeric_limits<double>::lowest();
      t_pos[0] = 500.0;
      t_neg[0] = 500.0;

      // y-direction
      b_pos[1] = 1826.0;
      b_neg[1] = -1826.0;
      t_pos[1] = 500.0;
      t_neg[1] = 500.0;

      // z-direction
      b_pos[2] = 2591.0;
      b_neg[2] = -2591.0;
      t_pos[2] = 500.0;
      t_neg[2] = 500.0;
    }

    // calculates the value of the complex coordinate stretch
    void
    vector_value(const Point<dim>             &p,
                 Vector<std::complex<double>> &value) const override;

  private:
    double omega;

    std::array<double, dim> b_pos, b_neg;
    std::array<double, dim> t_pos, t_neg;

    double pml_coeff_degree;
    double pml_coeff;
  };

  template <int dim>
  void
  PML<dim>::vector_value(const Point<dim>             &p,
                         Vector<std::complex<double>> &value) const
  {
    value.reinit(dim);

    for (unsigned int d = 0; d < dim; ++d)
      {
        double coeff = 0.0;

        if (p[d] > b_pos[d])
          {
            const double x_prime = p[d] - b_pos[d];
            const double a_coeff =
              pml_coeff / std::pow(t_pos[d], pml_coeff_degree);
            coeff = a_coeff * std::pow(x_prime, pml_coeff_degree);
          }
        else if (p[d] < b_neg[d])
          {
            const double x_prime = b_neg[d] - p[d];
            const double a_coeff =
              pml_coeff / std::pow(t_neg[d], pml_coeff_degree);
            coeff = a_coeff * std::pow(x_prime, pml_coeff_degree);
          }

        // complex coordinate stretching: s = 1 + i * sigma(x)
        value[d] = std::complex<double>(1.0, coeff);
      }
  }

  // Creation of a diffuse sound field for excitation of the wall on the source
  // side
  template <int dim>
  class DiffuseSoundField : public Function<dim, std::complex<double>>
  {
  public:
    DiffuseSoundField(unsigned int N, double omega, MPI_Comm mpi_communicator)
      : Function<dim, std::complex<double>>()
      , N(N)
      , dist_0_2PI(0.0, 2. * numbers::PI)
      , dist_0_1(0.0, 1.)
      , generator(static_cast<int>(omega))
      , omega(omega)
    {
      // Only let one rank 0 create the random variables of the diffuse sound
      // field...
      const unsigned int rank =
        Utilities::MPI::this_mpi_process(mpi_communicator);
      if (rank == 0)
        {
          phi.resize(N);
          Phi.resize(N);
          phase.resize(N);
          for (unsigned int n = 0; n < N; ++n)
            {
              phi[n]   = dist_0_2PI(generator);
              Phi[n]   = std::acos(dist_0_1(generator));
              phase[n] = dist_0_2PI(generator);
            }
        }
      // ... and broadcast from rank 0 to all ranks.
      phi   = Utilities::MPI::broadcast(mpi_communicator, phi, 0);
      Phi   = Utilities::MPI::broadcast(mpi_communicator, Phi, 0);
      phase = Utilities::MPI::broadcast(mpi_communicator, phase, 0);

      // Generation of kn on all ranks
      for (unsigned int n = 0; n < N; n++)
        {
          double         scale = (omega / get_sound_speed());
          Tensor<1, dim> k;
          k[0] = std::cos(Phi[n]) * scale;
          k[1] = std::sin(Phi[n]) * std::cos(phi[n]) * scale;
          k[2] = std::sin(Phi[n]) * std::sin(phi[n]) * scale;
          k_vector.push_back(k);
        }
    }
    // Calculation of the total pressure on the source side
    virtual void
    value_list(const std::vector<Point<dim>>     &points,
               std::vector<std::complex<double>> &values,
               const unsigned int                 component = 0) const override;

    // Calculation of the gradient of the total pressure on the source side
    virtual void
    gradient_list(const std::vector<Point<dim>>                     &points,
                  std::vector<Tensor<1, dim, std::complex<double>>> &gradients,
                  const unsigned int component = 0) const override;

    // Calculation of the incident sound pressure on the source side
    void
    value_list_incidence(const std::vector<Point<dim>>     &points,
                         std::vector<std::complex<double>> &values,
                         const unsigned int component = 0) const;

  private:
    // Number of plane waves
    unsigned int N;
    // For  randmoness
    std::uniform_real_distribution<double> dist_0_2PI;
    std::uniform_real_distribution<double> dist_0_1;
    std::mt19937                           generator;

    // variables of the diffuse field
    std::vector<double>                 phi, Phi, phase, kn_x, kn_y, kn_z;
    std::vector<Tensor<1, dim, double>> k_vector;

    // angular velocity
    double omega;
  };

  template <int dim>
  void
  DiffuseSoundField<dim>::value_list(const std::vector<Point<dim>>     &points,
                                     std::vector<std::complex<double>> &values,
                                     const unsigned int component) const
  {
    AssertThrow(component == 0, ExcMessage("only component 0 is implemented"));
    values.resize(points.size());
    std::complex<double> j{0., 1.};

    for (unsigned int q = 0; q < points.size(); ++q)
      {
        const auto &q_p = points[q];
        values[q]       = {0., 0.};
        for (unsigned int n = 0; n < N; n++)
          {
            // sound waves traveling towards the wall...
            double dot_towards    = 0.;
            double dot_reflection = 0.;

            for (unsigned int d = 0; d < dim; d++)
              {
                // sound waves traveling towards the wall...
                dot_towards += k_vector[n][d] * q_p[d];
                // ... and reflections.
                if (d == 0)
                  {
                    dot_reflection -= k_vector[n][d] * q_p[d];
                  }
                else
                  {
                    dot_reflection += k_vector[n][d] * q_p[d];
                  }
              }
            // sound waves traveling towards the wall...
            values[q] += std::exp(-j * (dot_towards) + j * phase[n]);
            // ... and reflections.
            values[q] += std::exp(-j * (dot_reflection) + j * phase[n]);
          }
        values[q] *= 1. / (std::sqrt(2. * static_cast<double>(N))) * 1.e6;
      }
  }

  template <int dim>
  void
  DiffuseSoundField<dim>::value_list_incidence(
    const std::vector<Point<dim>>     &points,
    std::vector<std::complex<double>> &values,
    const unsigned int                 component) const
  {
    AssertThrow(component == 0, ExcMessage("only component 0 is implemented"));

    values.resize(points.size());
    std::complex<double> j{0., 1.};
    for (unsigned int q = 0; q < points.size(); ++q)
      {
        const auto &q_p = points[q];
        values[q]       = {0., 0.};
        for (unsigned int n = 0; n < N; n++)
          {
            // Only sound waves traveling towards the wall.
            double dot_towards = 0.;
            for (unsigned int d = 0; d < dim; d++)
              {
                // Sound waves traveling towards the wall.
                dot_towards += k_vector[n][d] * q_p[d];
              }
            // Only sound waves traveling towards the wall.
            values[q] += std::exp(-j * (dot_towards) + j * phase[n]);
          }
        values[q] *= 1. / (std::sqrt(2. * static_cast<double>(N))) * 1.e6;
      }
  }

  template <int dim>
  void
  DiffuseSoundField<dim>::gradient_list(
    const std::vector<Point<dim>>                     &points,
    std::vector<Tensor<1, dim, std::complex<double>>> &gradients,
    const unsigned int                                 component) const
  {
    AssertThrow(component == 0, ExcMessage("only component 0 is implemented"));

    gradients.resize(points.size());
    const std::complex<double> j{0.0, 1.0};
    for (unsigned int q = 0; q < points.size(); ++q)
      {
        const auto &q_p = points[q];
        for (unsigned int d = 0; d < dim; ++d)
          gradients[q][d] = 0.;

        for (unsigned int n = 0; n < N; ++n)
          {
            double dot = 0.;
            for (unsigned int d = 0; d < dim; ++d)
              dot += k_vector[n][d] * q_p[d];

            const std::complex<double> exp_term =
              std::exp(-j * dot + j * phase[n]);

            for (unsigned int d = 0; d < dim; ++d)
              gradients[q][d] += (-j * k_vector[n][d]) * exp_term *
                                 (1. / std::sqrt(2. * static_cast<double>(N))) *
                                 1.e6;
          }
      }
  }

  template <int dim>
  class HarmonicResponse
  {
  public:
    HarmonicResponse(double omega);
    void
    run(bool write_output = false);

  private:
    void
    setup_system();
    void
    assemble_system();
    void
    solve();

    // calculate the magnitude of u and p
    void
    calculate_magnitude();

    // calculation of sound power at receiver side for one cell
    double
    cell_receiver_sound_power(
      const FEFaceValuesBase<dim> &elasticity_fe_face_values,
      const FEFaceValuesBase<dim> &air_fe_face_values,
      const double                &omega);

    // calculation of sound power at sender side
    double
    incident_sound_power();
    // calculation of sound power at receiver side
    double
    receiver_sound_power();
    // Assemble the air-structure coupling terms
    // This function implements the assembly of the air-structure interface.
    void
    assemble_air_structure_interface_term(
      const FEFaceValuesBase<dim>      &elasticity_fe_face_values,
      const FEFaceValuesBase<dim>      &air_fe_face_values,
      FullMatrix<std::complex<double>> &local_interface_matrix,
      const double                     &omega);
    MPI_Comm                                  mpi_communicator;
    parallel::distributed::Triangulation<dim> triangulation;

    const FESystem<dim>   fe_structure, fe_air;
    const QGauss<dim>     quadrature_formula_structure, quadrature_formula_air;
    const QGauss<dim - 1> face_quadrature_formula_structure,
      face_quadrature_formula_air;

    hp::FECollection<dim>    fe_collection;
    hp::QCollection<dim>     q_collection;
    hp::QCollection<dim - 1> q_face_collection;

    DoFHandler<dim>                         dof_handler;
    IndexSet                                locally_owned_dofs;
    IndexSet                                locally_relevant_dofs;
    AffineConstraints<std::complex<double>> constraints;
    LinearAlgebraPETSc::MPI::SparseMatrix   system_matrix;
    LinearAlgebraPETSc::MPI::Vector         locally_relevant_solution;
    LinearAlgebraPETSc::MPI::Vector         locally_relevant_magnitude;
    LinearAlgebraPETSc::MPI::Vector         system_rhs;

    ConditionalOStream     pcout;
    const double           omega;
    DiffuseSoundField<dim> field;
    PML<dim>               pml;
  };

  template <int dim>
  HarmonicResponse<dim>::HarmonicResponse(double omega)
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement |
                      Triangulation<dim>::smoothing_on_coarsening))
    , fe_structure(FE_Q<dim>(2), dim, FE_Nothing<dim>(), 1)
    , fe_air(FE_Nothing<dim>(), dim, FE_Q<dim>(1), 1)
    , quadrature_formula_structure(fe_structure.degree + 1)
    , quadrature_formula_air(fe_air.degree + 1)
    , face_quadrature_formula_structure(fe_structure.degree + 1)
    , face_quadrature_formula_air(fe_air.degree + 1)
    , dof_handler(triangulation)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , omega(omega)
    , field(1.e3, omega, mpi_communicator)
    , pml(omega)
  {
    static_assert(dim == 3,
                  "HarmonicResponse is only implemented for dim == 3");
    fe_collection.push_back(fe_structure);
    fe_collection.push_back(fe_air);
    q_collection.push_back(quadrature_formula_structure);
    q_collection.push_back(quadrature_formula_air);
    q_face_collection.push_back(face_quadrature_formula_structure);
    q_face_collection.push_back(face_quadrature_formula_air);
  }

  template <int dim>
  void
  HarmonicResponse<dim>::setup_system()
  {
    // Set material id and active FE indices.
    for (const auto &cell : dof_handler.cell_iterators())
      {
        cell->set_material_id(static_cast<unsigned int>(MaterialID::Concrete));

        if ((cell->center()[0] - 203.) > 0.)
          {
            cell->set_material_id(static_cast<unsigned int>(MaterialID::Air));
          }
      }
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            cell->set_active_fe_index(
              static_cast<unsigned int>(MaterialID::Concrete));
          }
        if ((cell->center()[0] - 203.) > 0.)
          {
            if (cell->is_locally_owned())
              {
                cell->set_active_fe_index(
                  static_cast<unsigned int>(MaterialID::Air));
              }
          }
      }
    // Definition of FE space
    dof_handler.distribute_dofs(fe_collection);
    pcout << " Number of degrees of freedom = " << dof_handler.n_dofs()
          << std::endl;

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs.clear();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);
    locally_relevant_magnitude.reinit(locally_owned_dofs, mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    // set up contraints
    constraints.clear();
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    // Set boundary ids
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            for (unsigned int f = 0; f < cell->n_faces(); ++f)
              {
                const auto face = cell->face(f);
                const auto p    = face->center();
                if (std::abs(p[0] - 203.) < geom_tol &&
                    (cell->material_id() ==
                     static_cast<unsigned int>(MaterialID::Concrete)))
                  {
                    cell->face(f)->set_user_index(
                      static_cast<unsigned int>(SurfaceID::ReceiverSide));
                  }
                if (std::abs(p[0]) < geom_tol)
                  {
                    cell->face(f)->set_user_index(
                      static_cast<unsigned int>(SurfaceID::SourceSide) &&
                      (cell->material_id() ==
                       static_cast<unsigned int>(MaterialID::Concrete)));
                  }
                if (cell->face(f)->at_boundary())
                  {
                    if ((p[0] + geom_tol) < 203. && p[0] > geom_tol)
                      {
                        cell->face(f)->set_boundary_id(
                          static_cast<unsigned int>(SurfaceID::FixedBoundary));
                      }
                    if (std::abs(p[0] - 1312.) < geom_tol &&
                        cell->material_id() ==
                          static_cast<unsigned int>(MaterialID::Air))
                      {
                        cell->face(f)->set_boundary_id(
                          static_cast<unsigned int>(SurfaceID::ZeroPressure));
                      }
                  }
              }
          }
      }
    // The wall is fixed at its outer boundary
    const FEValuesExtractors::Vector displacement(0);
    ComponentMask                    component_mask_displacement =
      fe_collection.component_mask(displacement);
    VectorTools::interpolate_boundary_values(
      dof_handler,
      static_cast<unsigned int>(SurfaceID::FixedBoundary),
      Functions::ZeroFunction<dim, std::complex<double>>(4),
      constraints,
      component_mask_displacement);
    constraints.close();

    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, true);
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp, constraints);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               locally_owned_dofs,
                                               mpi_communicator,
                                               locally_relevant_dofs);
    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);
  }

  template <int dim>
  void
  HarmonicResponse<dim>::assemble_system()
  {
    // FE values for volume integration
    hp::FEValues<dim>     hp_fe_values(fe_collection,
                                   q_collection,
                                   update_values | update_gradients |
                                     update_quadrature_points |
                                     update_JxW_values);
    hp::FEFaceValues<dim> hp_fe_face_values(fe_collection,
                                            q_face_collection,
                                            update_values | update_JxW_values |
                                              update_normal_vectors |
                                              update_quadrature_points);

    // Common face quadrature
    const QGauss<dim - 1> common_face_quadrature(fe_collection.max_degree() +
                                                 1);
    // FE face values for surface integration
    FEFaceValues<dim>    air_fe_face_values(fe_air,
                                         common_face_quadrature,
                                         update_values | update_JxW_values |
                                           update_normal_vectors |
                                           update_quadrature_points);
    FEFaceValues<dim>    elasticity_fe_face_values(fe_structure,
                                                common_face_quadrature,
                                                update_values |
                                                  update_JxW_values |
                                                  update_normal_vectors |
                                                  update_quadrature_points);
    FESubfaceValues<dim> air_fe_sub_face_values(fe_air,
                                                common_face_quadrature,
                                                update_values |
                                                  update_JxW_values |
                                                  update_normal_vectors |
                                                  update_quadrature_points);
    FESubfaceValues<dim> elasticity_fe_sub_face_values(
      fe_structure,
      common_face_quadrature,
      update_values | update_JxW_values | update_normal_vectors |
        update_quadrature_points);

    // Local element matrix for a cell
    FullMatrix<std::complex<double>> cell_matrix;
    // Local interface matrix between air and structure DoFs
    FullMatrix<std::complex<double>> local_interface_matrix(
      fe_air.n_dofs_per_cell(), fe_structure.n_dofs_per_cell());

    // Right-hand side
    Vector<std::complex<double>>         cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<types::global_dof_index> neighbor_dof_indices;
    const FEValuesExtractors::Vector     displacement(0);
    const FEValuesExtractors::Scalar     pressure(dim);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            // Assemble air cells contributions
            if (cell->material_id() ==
                static_cast<unsigned int>(MaterialID::Air))
              {
                cell_matrix = 0;
                cell_rhs    = 0;
                hp_fe_values.reinit(cell);
                const FEValues<dim> &fe_values =
                  hp_fe_values.get_present_fe_values();

                const unsigned int dofs_per_cell = fe_air.n_dofs_per_cell();
                const unsigned int n_q_points = fe_values.n_quadrature_points;
                cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
                cell_rhs.reinit(dofs_per_cell);
                local_dof_indices.resize(dofs_per_cell);
                neighbor_dof_indices.resize(fe_structure.n_dofs_per_cell());
                const double sound_speed = get_sound_speed();

                // Wave number k
                const std::complex<double> k = (omega / sound_speed);

                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    const auto        JxW     = fe_values.JxW(q);
                    const Point<dim> &q_point = fe_values.quadrature_point(q);
                    // calucalte lambda and J for PML
                    Vector<std::complex<double>> lambda(dim);
                    pml.vector_value(q_point, lambda);
                    const std::complex<double> J =
                      lambda[0] * lambda[1] * lambda[2];

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                        const double phi_i = fe_values[pressure].value(i, q);

                        // Gradient of phi_i, which is multiplied by 1/lambda
                        Tensor<1, dim, std::complex<double>> phi_i_i =
                          fe_values[pressure].gradient(i, q);
                        for (unsigned int d = 0; d < dim; ++d)
                          phi_i_i[d] *= 1.0 / lambda[d];

                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                          {
                            const double phi_j =
                              fe_values[pressure].value(j, q);

                            // Gradient of phi_j, which is multiplied by
                            // 1/lambda
                            Tensor<1, dim, std::complex<double>> phi_j_j =
                              fe_values[pressure].gradient(j, q);
                            for (unsigned int d = 0; d < dim; ++d)
                              phi_j_j[d] *= 1.0 / lambda[d];

                            cell_matrix[i][j] += phi_i_i * phi_j_j * JxW * J;
                            cell_matrix[i][j] -= Utilities::fixed_power<2>(k) *
                                                 phi_i * phi_j * JxW * J;
                          }
                      }
                  }
                // assemble local system to global system.
                cell->get_dof_indices(local_dof_indices);
                constraints.distribute_local_to_global(cell_matrix,
                                                       cell_rhs,
                                                       local_dof_indices,
                                                       system_matrix,
                                                       system_rhs);

                // Here, the air-structure interface is considered. Similar to
                // step 46, 3 possibilities exist: The neighbor is
                // at the same refinement level and has no children, the
                // neighbor has children and the neighbor is coarser.
                for (const auto f : cell->face_indices())
                  {
                    if (!cell->at_boundary(f))
                      {
                        const auto neighbor = cell->neighbor(f);
                        if (neighbor->material_id() ==
                            static_cast<unsigned int>(MaterialID::Concrete))

                          // if (neighbor->center()[0]<203.)
                          {
                            // The neighbor is at the same refinement level and
                            // has no children.
                            if ((cell->neighbor(f)->level() == cell->level()) &&
                                (cell->neighbor(f)->has_children() == false))
                              {
                                air_fe_face_values.reinit(cell, f);
                                elasticity_fe_face_values.reinit(
                                  cell->neighbor(f),
                                  cell->neighbor_of_neighbor(f));
                                assemble_air_structure_interface_term(
                                  elasticity_fe_face_values,
                                  air_fe_face_values,
                                  local_interface_matrix,
                                  omega);
                                cell->neighbor(f)->get_dof_indices(
                                  neighbor_dof_indices);
                                constraints.distribute_local_to_global(
                                  local_interface_matrix,
                                  local_dof_indices,
                                  neighbor_dof_indices,
                                  system_matrix);
                              }
                            // The neighbor has children.
                            else if ((cell->neighbor(f)->level() ==
                                      cell->level()) &&
                                     (cell->neighbor(f)->has_children() ==
                                      true))
                              {
                                for (unsigned int subface = 0;
                                     subface < cell->face(f)->n_children();
                                     ++subface)
                                  {
                                    air_fe_sub_face_values.reinit(cell,
                                                                  f,
                                                                  subface);
                                    elasticity_fe_face_values.reinit(
                                      cell->neighbor_child_on_subface(f,
                                                                      subface),
                                      cell->neighbor_of_neighbor(f));
                                    assemble_air_structure_interface_term(
                                      elasticity_fe_face_values,
                                      air_fe_sub_face_values,
                                      local_interface_matrix,
                                      omega);
                                    cell->neighbor_child_on_subface(f, subface)
                                      ->get_dof_indices(neighbor_dof_indices);
                                    constraints.distribute_local_to_global(
                                      local_interface_matrix,
                                      local_dof_indices,
                                      neighbor_dof_indices,
                                      system_matrix);
                                  }
                              }
                            // The neighbor is coarser.
                            else if (cell->neighbor_is_coarser(f))
                              {
                                air_fe_face_values.reinit(cell, f);
                                elasticity_fe_sub_face_values.reinit(
                                  cell->neighbor(f),
                                  cell->neighbor_of_coarser_neighbor(f).first,
                                  cell->neighbor_of_coarser_neighbor(f).second);
                                assemble_air_structure_interface_term(
                                  elasticity_fe_sub_face_values,
                                  air_fe_face_values,
                                  local_interface_matrix,
                                  omega);
                                cell->neighbor(f)->get_dof_indices(
                                  neighbor_dof_indices);
                                constraints.distribute_local_to_global(
                                  local_interface_matrix,
                                  local_dof_indices,
                                  neighbor_dof_indices,
                                  system_matrix);
                              }
                          }
                      }
                  }
              }
            // Assemble structure cells contributions
            else if (cell->is_locally_owned() &&
                     cell->material_id() ==
                       static_cast<unsigned int>(MaterialID::Concrete))
              {
                cell_matrix = 0;
                cell_rhs    = 0;
                hp_fe_values.reinit(cell);
                const FEValues<dim> &fe_values =
                  hp_fe_values.get_present_fe_values();

                const unsigned int dofs_per_cell =
                  fe_structure.n_dofs_per_cell();
                const unsigned int n_q_points = fe_values.n_quadrature_points;
                cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
                cell_rhs.reinit(dofs_per_cell);
                local_dof_indices.resize(dofs_per_cell);
                neighbor_dof_indices.resize(fe_air.n_dofs_per_cell());

                const SymmetricTensor<4, dim> stiffness_tensor =
                  get_stiffness_tensor<dim>();
                const double               density  = get_density_structure();
                const std::complex<double> iso_loss = get_iso_loss();

                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    const auto JxW = fe_values.JxW(q);
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                        const Tensor<1, dim> phi_i =
                          fe_values[displacement].value(i, q);
                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                          {
                            const Tensor<1, dim> phi_j =
                              fe_values[displacement].value(j, q);
                            const SymmetricTensor<2, dim> eps_phi_i =
                              get_strain(fe_values, i, q);
                            const SymmetricTensor<2, dim> eps_phi_j =
                              get_strain(fe_values, j, q);
                            cell_matrix[i][j] += eps_phi_i * stiffness_tensor *
                                                 iso_loss * eps_phi_j * JxW;
                            cell_matrix[i][j] -=
                              Utilities::fixed_power<2>(omega) * phi_j * phi_i *
                              density * JxW;
                          }
                      }
                  }
                // Here the diffuse sound field is considered on the source side
                // of the wall.
                for (const auto face_no : cell->face_indices())
                  {
                    if (cell->face(face_no)->user_index() ==
                        static_cast<unsigned int>(SurfaceID::SourceSide))
                      {
                        hp_fe_face_values.reinit(cell, face_no);
                        const FEFaceValues<dim> &fe_face_values =
                          hp_fe_face_values.get_present_fe_values();
                        const unsigned int n_face_q_points =
                          fe_face_values.n_quadrature_points;
                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                          {
                            if (fe_structure.has_support_on_face(i, face_no))
                              {
                                std::vector<std::complex<double>> pressure(
                                  n_face_q_points);
                                const auto face_quadrature_points =
                                  fe_face_values.get_quadrature_points();
                                field.value_list(face_quadrature_points,
                                                 pressure);

                                for (unsigned int q = 0; q < n_face_q_points;
                                     ++q)
                                  {
                                    const auto JxW = fe_face_values.JxW(q);
                                    const Tensor<1, dim> phi_i =
                                      fe_face_values[displacement].value(i, q);
                                    const Tensor<1, dim, double> NormalVector =
                                      -fe_face_values.normal_vector(q);
                                    cell_rhs[i] += pressure[q] *
                                                   (NormalVector * phi_i) * JxW;
                                  }
                              }
                          }
                      }
                  }
                cell->get_dof_indices(local_dof_indices);
                constraints.distribute_local_to_global(cell_matrix,
                                                       cell_rhs,
                                                       local_dof_indices,
                                                       system_matrix,
                                                       system_rhs);
              }
          }
      }
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }

  template <int dim>
  void
  HarmonicResponse<dim>::assemble_air_structure_interface_term(
    const FEFaceValuesBase<dim>      &elasticity_fe_face_values,
    const FEFaceValuesBase<dim>      &air_fe_face_values,
    FullMatrix<std::complex<double>> &local_interface_matrix,
    const double                     &omega)
  {
    const FEValuesExtractors::Vector displacement(0);
    const FEValuesExtractors::Scalar pressure(dim);
    local_interface_matrix         = 0;
    const auto         density_air = get_density_air();
    const unsigned int n_face_quadrature_points =
      air_fe_face_values.n_quadrature_points;

    for (unsigned int q = 0; q < n_face_quadrature_points; ++q)
      {
        const Tensor<1, dim, double> normalVectorStructure =
          -air_fe_face_values.normal_vector(q);
        for (unsigned int i = 0; i < air_fe_face_values.dofs_per_cell; ++i)
          {
            const double phi_i = air_fe_face_values[pressure].value(i, q);
            for (unsigned int j = 0;
                 j < elasticity_fe_face_values.dofs_per_cell;
                 ++j)
              {
                const Tensor<1, dim> phi_j =
                  elasticity_fe_face_values[displacement].value(j, q);
                local_interface_matrix[i][j] +=
                  density_air * Utilities::fixed_power<2>(omega) * phi_i *
                  normalVectorStructure * phi_j * air_fe_face_values.JxW(q);
                local_interface_matrix[i][j] +=
                  phi_j * normalVectorStructure * phi_i *
                  elasticity_fe_face_values.JxW(q);
              }
          }
      }
  }

  // calculation of incident sound power on the source side
  template <int dim>
  double
  HarmonicResponse<dim>::incident_sound_power()
  {
    FEFaceValues<dim> structure_fe_face_values(
      fe_structure,
      face_quadrature_formula_structure,
      update_values | update_JxW_values | update_normal_vectors |
        update_quadrature_points);
    std::vector<types::global_dof_index> local_dof_indices;
    const FEValuesExtractors::Vector     displacement(0);
    double                               sound_power = 0.;
    const double                         density_air = get_density_air();
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            for (const auto face_no : cell->face_indices())
              {
                if (cell->face(face_no)->at_boundary() &&
                    (cell->face(face_no)->user_index() ==
                     static_cast<unsigned int>(SurfaceID::SourceSide)) &&
                    (cell->material_id() ==
                     static_cast<unsigned int>(MaterialID::Concrete)))
                  {
                    structure_fe_face_values.reinit(cell, face_no);
                    const unsigned int n_face_q_points =
                      structure_fe_face_values.n_quadrature_points;
                    std::vector<std::complex<double>> pressure(n_face_q_points);
                    const auto                        face_quadrature_points =
                      structure_fe_face_values.get_quadrature_points();
                    field.value_list_incidence(face_quadrature_points,
                                               pressure);
                    std::vector<Tensor<1, dim, std::complex<double>>>
                      pressure_gradients(n_face_q_points);
                    field.gradient_list(face_quadrature_points,
                                        pressure_gradients);
                    for (unsigned int q = 0; q < n_face_q_points; ++q)
                      {
                        const Tensor<1, dim, double> NormalVector =
                          -structure_fe_face_values.normal_vector(q);
                        const auto JxW = structure_fe_face_values.JxW(q);
                        std::complex<double> v_n =
                          (-1.0 / (omega * std::complex<double>(0., 1.) *
                                   density_air)) *
                          (pressure_gradients[q] * NormalVector);
                        sound_power +=
                          0.5 * std::real(pressure[q] * std::conj(v_n) * JxW);
                      }
                  }
              }
          }
      }
    return sound_power;
  }

  // Calculating the sound power on the receiver side
  template <int dim>
  double
  HarmonicResponse<dim>::receiver_sound_power()
  {
    // Common face quadrature
    const QGauss<dim - 1> common_face_quadrature(fe_collection.max_degree() +
                                                 1);
    // FE face values for surface integration
    FEFaceValues<dim>    air_fe_face_values(fe_air,
                                         common_face_quadrature,
                                         update_values | update_JxW_values |
                                           update_normal_vectors |
                                           update_quadrature_points |
                                           update_gradients);
    FEFaceValues<dim>    elasticity_fe_face_values(fe_structure,
                                                common_face_quadrature,
                                                update_values |
                                                  update_JxW_values |
                                                  update_normal_vectors |
                                                  update_quadrature_points |
                                                  update_gradients);
    FESubfaceValues<dim> air_fe_sub_face_values(fe_air,
                                                common_face_quadrature,
                                                update_values |
                                                  update_JxW_values |
                                                  update_normal_vectors |
                                                  update_quadrature_points |
                                                  update_gradients);
    FESubfaceValues<dim> elasticity_fe_sub_face_values(
      fe_structure,
      common_face_quadrature,
      update_values | update_JxW_values | update_normal_vectors |
        update_quadrature_points | update_gradients);

    double sound_power = 0.;
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            for (const auto f : cell->face_indices())
              {
                if ((cell->face(f)->user_index() ==
                     static_cast<unsigned int>(SurfaceID::ReceiverSide)) &&
                    !cell->at_boundary(f) &&
                    (cell->material_id() ==
                     static_cast<unsigned int>(MaterialID::Concrete)))
                  {
                    const auto neighbor = cell->neighbor(f);
                    // The neighbor is at the same refinement level and has no
                    // children.
                    if ((neighbor->level() == cell->level()) &&
                        (neighbor->has_children() == false) &&
                        (neighbor->material_id() ==
                         static_cast<unsigned int>(MaterialID::Air)))
                      {
                        elasticity_fe_face_values.reinit(cell, f);
                        air_fe_face_values.reinit(
                          neighbor, cell->neighbor_of_neighbor(f));
                        sound_power +=
                          cell_receiver_sound_power(elasticity_fe_face_values,
                                                    air_fe_face_values,
                                                    omega);
                      }
                    // The neighbor has children.
                    else if ((neighbor->level() == cell->level()) &&
                             (neighbor->has_children() == true))
                      {
                        for (unsigned int subface = 0;
                             subface < cell->face(f)->n_children();
                             ++subface)
                          {
                            elasticity_fe_sub_face_values.reinit(cell,
                                                                 f,
                                                                 subface);
                            air_fe_face_values.reinit(
                              cell->neighbor_child_on_subface(f, subface),
                              cell->neighbor_of_neighbor(f));
                            sound_power += cell_receiver_sound_power(
                              elasticity_fe_sub_face_values,
                              air_fe_face_values,
                              omega);
                          }
                      }
                    // The neighbor is coarser.
                    else if (cell->neighbor_is_coarser(f))
                      {
                        elasticity_fe_face_values.reinit(cell, f);
                        air_fe_sub_face_values.reinit(
                          neighbor,
                          cell->neighbor_of_coarser_neighbor(f).first,
                          cell->neighbor_of_coarser_neighbor(f).second);
                        sound_power +=
                          cell_receiver_sound_power(elasticity_fe_face_values,
                                                    air_fe_sub_face_values,
                                                    omega);
                      }
                  }
              }
          }
      }
    return sound_power;
  }

  template <int dim>
  double
  HarmonicResponse<dim>::cell_receiver_sound_power(
    const FEFaceValuesBase<dim> &elasticity_fe_face_values,
    const FEFaceValuesBase<dim> &air_fe_face_values,
    const double                &omega)
  {
    std::vector<types::global_dof_index> local_dof_indices;
    const FEValuesExtractors::Vector     displacement(0);
    const FEValuesExtractors::Scalar     pressure(dim);
    const unsigned int n_q_points  = air_fe_face_values.n_quadrature_points;
    double             sound_power = 0.;
    ;

    std::vector<std::complex<double>> local_dof_values_pressure(
      air_fe_face_values.n_quadrature_points);
    air_fe_face_values[pressure].get_function_values(locally_relevant_solution,
                                                     local_dof_values_pressure);

    std::vector<Tensor<1, dim, std::complex<double>>>
      local_dof_values_displacement(n_q_points);
    elasticity_fe_face_values[displacement].get_function_values(
      locally_relevant_solution, local_dof_values_displacement);
    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const Tensor<1, dim, double> NormalVector =
          air_fe_face_values.normal_vector(q);
        const auto           JxW = air_fe_face_values.JxW(q);
        std::complex<double> normal_velocity =
          local_dof_values_displacement[q] * NormalVector *
          std::complex<double>(0., 1.) * omega;
        sound_power +=
          0.5 *
          std::real(local_dof_values_pressure[q] * std::conj(normal_velocity)) *
          JxW;
      }
    return sound_power;
  }

  template <int dim>
  void
  HarmonicResponse<dim>::solve()
  {
    LinearAlgebraPETSc::MPI::Vector completely_distributed_solution(
      locally_owned_dofs, mpi_communicator);
    SolverControl                    solver_control;
    PETScWrappers::SparseDirectMUMPS solver(solver_control, mpi_communicator);
    solver.solve(system_matrix, completely_distributed_solution, system_rhs);
    constraints.distribute(completely_distributed_solution);
    locally_relevant_solution = completely_distributed_solution;
  }

  template <int dim>
  void
  HarmonicResponse<dim>::calculate_magnitude()
  {
    for (const auto i : locally_owned_dofs)
      {
        const std::complex<double> value = locally_relevant_solution[i];
        // For postprocessing, the real part represents the magnitude, while the
        // imaginary part vanishes.
        locally_relevant_magnitude[i] = std::abs(value);
      }
  }

  template <int dim>
  void
  HarmonicResponse<dim>::run(bool write_output)
  {
    static bool mode_output = false;
    if (!mode_output)
      {
#ifdef DEBUG
        pcout << "Debug mode" << std::endl;
#else
        pcout << "Release mode" << std::endl;
#endif
        mode_output = true;
      }

    const double frequency = omega / (numbers::PI * 2.); // frequency in Hz
    pcout << std::endl << "Frequency = " << frequency << " Hz" << std::endl;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation);
    std::ifstream input_file("./tria.inp");
    grid_in.read_abaqus(input_file);
    triangulation.refine_global(1);
    for (const auto &cell : triangulation.active_cell_iterators())
      {
        if (cell->center()[0] > 203.)
          {
            cell->set_refine_flag();
          }
      }
    triangulation.execute_coarsening_and_refinement();

    pcout << " setup_system" << std::endl;
    setup_system();
    pcout << " assemble_system" << std::endl;
    assemble_system();
    pcout << " solve" << std::endl;
    solve();
    // Calculation of the magnitude of the solution. For postprocessing, the
    // magnitude of the solutions corresponds to the real part and the imaginary
    // part vaishes.
    calculate_magnitude();
    const auto sound_power_source_side_local   = incident_sound_power();
    const auto sound_power_receiver_side_local = receiver_sound_power();
    const auto sound_power_source_side =
      dealii::Utilities::MPI::sum(sound_power_source_side_local,
                                  mpi_communicator);
    const auto sound_power_receiver_side =
      dealii::Utilities::MPI::sum(sound_power_receiver_side_local,
                                  mpi_communicator);

    const auto stl =
      10. * std::log10(sound_power_source_side / sound_power_receiver_side);
    pcout << " STL = " << stl << " dB" << std::endl;
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::ofstream outfile("STL_result.txt", std::ios::app);
        outfile << frequency << "  " << stl << std::endl;
      }

    if (write_output)
      {
        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);

        std::vector<std::string> solution_names_magnitude(dim, "magnitude_u");
        solution_names_magnitude.push_back("magnitude_p");
        std::vector<std::string> solution_names(dim, "u");
        solution_names.push_back("p");

        std::vector<DataComponentInterpretation::DataComponentInterpretation>
          interpretation = {
            DataComponentInterpretation::component_is_part_of_vector,
            DataComponentInterpretation::component_is_part_of_vector,
            DataComponentInterpretation::component_is_part_of_vector,
            DataComponentInterpretation::component_is_scalar};
        data_out.add_data_vector(locally_relevant_solution,
                                 solution_names,
                                 DataOut<dim>::type_dof_data,
                                 interpretation);


        data_out.add_data_vector(locally_relevant_magnitude,
                                 solution_names_magnitude,
                                 DataOut<dim>::type_dof_data,
                                 interpretation);

        // subdomain visualization (unchanged)
        Vector<float> subdomain(triangulation.n_active_cells());
        for (unsigned int i = 0; i < subdomain.size(); ++i)
          subdomain(i) = triangulation.locally_owned_subdomain();

        data_out.add_data_vector(subdomain, "subdomain");

        data_out.build_patches();

        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2) << frequency;
        std::filesystem::create_directories("./output/");
        data_out.write_vtu_with_pvtu_record(
          "./output/", "solution_" + ss.str(), 0, mpi_communicator, 2, 8);
      }
  }
} // namespace VibroAcousticProblem

int
main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      const unsigned int               dim = 3;
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      double                           f0  = 50.; // starting frequency in Hz
      unsigned int                     n   = 0;   // number of frequency steps
      double                           f_c = f0;  // evaluation frequency

      while (f_c < 1000.)
        {
          f_c          = f0 * std::pow(10., static_cast<double>(n) / 60.);
          double omega = f_c * 2. * numbers::PI;
          VibroAcousticProblem::HarmonicResponse<dim> elastic_problem(omega);
          elastic_problem.run(true);
          n++;
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}