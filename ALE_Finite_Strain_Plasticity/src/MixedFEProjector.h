/*
 * MixedFEProjector.h
 *
 *  Created on: 21 Jul 2015
 *      Author: maien
 */

#ifndef MIXEDFEPROJECTOR_H_
#define MIXEDFEPROJECTOR_H_

#include <vector>

#include <deal.II/base/tensor.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/vector.h>

#include "utilities.h"

template <int dim, typename Number = double>
class MixedFEProjector {
 public:
  MixedFEProjector();
  MixedFEProjector(
    const unsigned int mixed_dofs_per_cell,
    const dealii::FEValues<dim> &mixed_fe_values);
  virtual ~MixedFEProjector();

  template <typename T>
  void project(
    std::vector<T> *coefficients_of_mixed_dofs,
    const std::vector<T> &values_at_q_points) const;

 private:
  unsigned int n_q_points;
  unsigned int mixed_dofs_per_cell;
  std::vector<std::vector<Number > > M_inv_ksi;
};

template <int dim, typename Number>
MixedFEProjector<dim, Number>::MixedFEProjector():
  n_q_points(0),
  mixed_dofs_per_cell(0),
  M_inv_ksi(0) {

}

template <int dim, typename Number>
MixedFEProjector<dim, Number>::MixedFEProjector(
  const unsigned int mixed_dofs_per_cell,
  const dealii::FEValues<dim> &mixed_fe_values)
  : n_q_points (mixed_fe_values.get_quadrature().size()),
    mixed_dofs_per_cell (mixed_dofs_per_cell),
    M_inv_ksi (n_q_points, std::vector<Number>(mixed_dofs_per_cell)) {
  dealii::FullMatrix<Number>   M_matrix(mixed_dofs_per_cell, mixed_dofs_per_cell),
         M_inv(mixed_dofs_per_cell, mixed_dofs_per_cell);
  std::vector<dealii::Vector<Number> > ksi(n_q_points, dealii::Vector<Number>(mixed_dofs_per_cell));

  M_matrix = 0;
  for (unsigned int q_point = 0; q_point < n_q_points;
       ++q_point) {
    // Prep to compute mixed primary variables (Simo & Miehe 1992)
    for (unsigned int i = 0; i < mixed_dofs_per_cell; ++i) {
      const Number i_value = mixed_fe_values.shape_value (i, q_point);
      for (unsigned int j = 0; j < mixed_dofs_per_cell; ++j) {
        const Number j_value = mixed_fe_values.shape_value (j, q_point);
        M_matrix(i, j) += i_value * j_value * mixed_fe_values.quadrature_point(q_point)[0] * mixed_fe_values.JxW(q_point);
      }
      ksi.at(q_point)[i] = i_value * mixed_fe_values.quadrature_point(q_point)[0] * mixed_fe_values.JxW(q_point);
    }
  }

  M_inv.invert(M_matrix);

  for (unsigned int q_point = 0; q_point < n_q_points;
       ++q_point) {
    dealii::Vector<Number> M_inv_ksi_at_q_point(mixed_dofs_per_cell);
    M_inv.vmult(M_inv_ksi_at_q_point, ksi.at(q_point), false);
    for (unsigned int i = 0; i < mixed_dofs_per_cell; ++i)
      M_inv_ksi[q_point][i] = M_inv_ksi_at_q_point(i);
  }
}

template <int dim, typename Number>
MixedFEProjector<dim, Number>::~MixedFEProjector() {}

template <int dim, typename Number>
template <typename T>
void MixedFEProjector<dim, Number>::project(
  std::vector<T> *coefficients_of_mixed_dofs,
  const std::vector<T> &values_at_q_points) const {
  for (unsigned int i = 0; i < mixed_dofs_per_cell; ++i) {
    coefficients_of_mixed_dofs->at(i) = M_inv_ksi[0][i] * values_at_q_points[0];
    for (unsigned int q_point = 1; q_point < n_q_points; ++q_point)
      coefficients_of_mixed_dofs->at(i) += M_inv_ksi[q_point][i] * values_at_q_points[q_point];
  }
}

#endif /* MIXEDFEPROJECTOR_H_ */
