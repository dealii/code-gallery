/*
 * PointHistory.h
 *
 *  Created on: 21 Jul 2014
 *      Author: cerecam
 */

#ifndef POINTHISTORY_H_
#define POINTHISTORY_H_

#include <deal.II/base/utilities.h>
#include <deal.II/base/function.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/trilinos_vector.h>

namespace PlasticityLab {

  template <int dim, typename Number = double>
  class PointHistory {
   public:
    PointHistory();
    virtual ~PointHistory();

    struct HardeningParameters {
      HardeningParameters(
        Number equivalent_plastic_strain,
        dealii::SymmetricTensor<2, dim, Number> kinematic_hardening):
        equivalent_plastic_strain(equivalent_plastic_strain),
        kinematic_hardening(kinematic_hardening) { }
      HardeningParameters() {}

      Number equivalent_plastic_strain;
      dealii::SymmetricTensor<2, dim, Number> kinematic_hardening;
    };

    PointHistory(const dealii::SymmetricTensor<2, dim, Number> &,
                 const HardeningParameters &,
                 Number plastic_entropy,
                 Number material_Jacobian);

    dealii::SymmetricTensor<2, dim, Number> plastic_strain;
    HardeningParameters hardening_parameters;
    Number plastic_entropy;
    Number material_Jacobian;
  };


  template <int dim, typename Number>
  PointHistory<dim, Number>::PointHistory()
    :
    plastic_strain (dealii::unit_symmetric_tensor<dim, Number>()),
    hardening_parameters(Number(0.0), Number(0.0) * dealii::unit_symmetric_tensor<dim, Number>()),
    plastic_entropy(Number(0.0)),
    material_Jacobian(1.0) {
  }

  template <int dim, typename Number>
  PointHistory<dim, Number>::
  PointHistory(const dealii::SymmetricTensor<2, dim, Number> &plastic_strain,
               const HardeningParameters &hardening_parameters,
               const Number plastic_entropy,
               const Number material_Jacobian):
    plastic_strain(plastic_strain),
    hardening_parameters(hardening_parameters),
    plastic_entropy(plastic_entropy),
    material_Jacobian(material_Jacobian) {
  }

  template <int dim, typename Number>
  PointHistory<dim, Number>::~PointHistory() {
  }

} /* namespace PlasticityLab */

#endif /* POINTHISTORY_H_ */
