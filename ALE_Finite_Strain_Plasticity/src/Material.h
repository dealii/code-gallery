/*
 * Material.h
 *
 *  Created on: 05 Jan 2015
 *      Author: maien
 */

#ifndef MATERIAL_H_
#define MATERIAL_H_

using namespace dealii;

#include "ConstitutiveModelRequest.h"
#include <stdexcept>

namespace PlasticityLab {

  typedef size_t point_index_t;

  template <int dim, typename Number = double>
  class Material {
   public:
    virtual ~Material() = 0;

    virtual void compute_constitutive_request(
      ConstitutiveModelRequest <dim, Number> &constitutive_request,
      const point_index_t &point_index) = 0;

    virtual Number get_material_Jacobian(const point_index_t &point_index) const = 0;
    virtual dealii::SymmetricTensor<2, dim, Number> get_plastic_strain(const point_index_t &point_index) const = 0;

    virtual void setup_point_history(const point_index_t point_count) = 0;

    virtual std::vector<Number> get_state_parameters(
                const point_index_t &point_index,
                const Tensor<2, dim, Number> &reference_transformation=unit_symmetric_tensor<dim>()) const = 0;

    virtual void set_state_parameters(
                const point_index_t &point_index,
                const std::vector<Number> &state_parameters,
                const Tensor<2, dim, Number> &reference_transformation) = 0;
    virtual size_t get_material_parameter_count() const = 0;

  };

  template <int dim, typename Number>
  inline Material<dim, Number>::~Material() {
  }

  class MaterialDomainException: public std::runtime_error {
   public:
    MaterialDomainException();
    MaterialDomainException(std::string s): std::runtime_error(s) {};
  };

} /* namespace PlasticityLab */

#endif /* MATERIAL_H_ */
