/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2015 by David Wells
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 */

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II-cdr/parameters.h>
#include <deal.II-cdr/system_matrix.h>
#include <deal.II-cdr/system_matrix.templates.h>

// This file exists just to build template specializations of
// <code>create_system_matrix</code>. Even though the solver is run in
// parallel with Trilinos objects, other serial solvers can use the same
// function without recompilation by compiling everything here just one time.
namespace CDR
{
  using namespace dealii;

  template void
  create_system_matrix<2, SparseMatrix<double>>(
    const DoFHandler<2> &                              dof_handler,
    const QGauss<2> &                                  quad,
    const std::function<Tensor<1, 2>(const Point<2>)> &convection_function,
    const CDR::Parameters &                            parameters,
    const double                                       time_step,
    SparseMatrix<double> &                             system_matrix);

  template void
  create_system_matrix<3, SparseMatrix<double>>(
    const DoFHandler<3> &                              dof_handler,
    const QGauss<3> &                                  quad,
    const std::function<Tensor<1, 3>(const Point<3>)> &convection_function,
    const CDR::Parameters &                            parameters,
    const double                                       time_step,
    SparseMatrix<double> &                             system_matrix);

  template void
  create_system_matrix<2, SparseMatrix<double>>(
    const DoFHandler<2> &                              dof_handler,
    const QGauss<2> &                                  quad,
    const std::function<Tensor<1, 2>(const Point<2>)> &convection_function,
    const CDR::Parameters &                            parameters,
    const double                                       time_step,
    const AffineConstraints<double> &                  constraints,
    SparseMatrix<double> &                             system_matrix);

  template void
  create_system_matrix<3, SparseMatrix<double>>(
    const DoFHandler<3> &                              dof_handler,
    const QGauss<3> &                                  quad,
    const std::function<Tensor<1, 3>(const Point<3>)> &convection_function,
    const CDR::Parameters &                            parameters,
    const double                                       time_step,
    const AffineConstraints<double> &                  constraints,
    SparseMatrix<double> &                             system_matrix);

  template void
  create_system_matrix<2, TrilinosWrappers::SparseMatrix>(
    const DoFHandler<2> &                              dof_handler,
    const QGauss<2> &                                  quad,
    const std::function<Tensor<1, 2>(const Point<2>)> &convection_function,
    const CDR::Parameters &                            parameters,
    const double                                       time_step,
    TrilinosWrappers::SparseMatrix &                   system_matrix);

  template void
  create_system_matrix<3, TrilinosWrappers::SparseMatrix>(
    const DoFHandler<3> &                              dof_handler,
    const QGauss<3> &                                  quad,
    const std::function<Tensor<1, 3>(const Point<3>)> &convection_function,
    const CDR::Parameters &                            parameters,
    const double                                       time_step,
    TrilinosWrappers::SparseMatrix &                   system_matrix);

  template void
  create_system_matrix<2, TrilinosWrappers::SparseMatrix>(
    const DoFHandler<2> &                              dof_handler,
    const QGauss<2> &                                  quad,
    const std::function<Tensor<1, 2>(const Point<2>)> &convection_function,
    const CDR::Parameters &                            parameters,
    const double                                       time_step,
    const AffineConstraints<double> &                  constraints,
    TrilinosWrappers::SparseMatrix &                   system_matrix);

  template void
  create_system_matrix<3, TrilinosWrappers::SparseMatrix>(
    const DoFHandler<3> &                              dof_handler,
    const QGauss<3> &                                  quad,
    const std::function<Tensor<1, 3>(const Point<3>)> &convection_function,
    const CDR::Parameters &                            parameters,
    const double                                       time_step,
    const AffineConstraints<double> &                  constraints,
    TrilinosWrappers::SparseMatrix &                   system_matrix);
} // namespace CDR
