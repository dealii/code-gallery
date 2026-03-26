/*
 * ThermoPlasticMaterial.h
 *
 *  Created on: 05 Jan 2015
 *      Author: maien
 */

#ifndef THERMOPLASTICMATERIAL_H_
#define THERMOPLASTICMATERIAL_H_

#include "PointHistory.h"
#include "Material.h"
#include "ConstitutiveModelRequest.h"

using namespace dealii;

namespace PlasticityLab {

  template <int dim, typename ViscoplasticYieldLaw, typename Number = double>
  class ThermoPlasticMaterial : public Material<dim, Number> {
   public:
    ThermoPlasticMaterial(const Number kappa,
                          const Number mu,
                          const Number thermal_expansion_coefficient,
                          const Number thermal_conductivity,
                          const Number heat_capacity,
                          const Number dissipation_factor,
                          const ViscoplasticYieldLaw &viscoplastic_yield_law);

    virtual ~ThermoPlasticMaterial();

    void compute_constitutive_request(
      ConstitutiveModelRequest <dim, Number> &constitutive_request,
      const point_index_t &point_index);

    Number get_material_Jacobian(const point_index_t &point_index) const;
    dealii::SymmetricTensor<2, dim, Number> get_plastic_strain(const point_index_t &point_index) const;

    std::vector<Number> get_state_parameters(
            const point_index_t &point_index,
            const Tensor<2, dim, Number> &reference_transformation=unit_symmetric_tensor<dim>()) const;

    void  set_state_parameters(
            const point_index_t &point_index,
            const std::vector<Number> &state_parameters,
            const Tensor<2, dim, Number> &reference_transformation=unit_symmetric_tensor<dim>());
    size_t get_material_parameter_count() const;


    void setup_point_history (const point_index_t point_count);

   private:
    const Number kappa;
    const Number mu;

    const Number thermal_expansion_coefficient;
    const Number thermal_conductivity;
    const Number heat_capacity;
    const Number reference_temperature;

    const Number dissipation_factor;
    const ViscoplasticYieldLaw viscoplastic_yield_law;

    std::vector< PointHistory<dim, Number> > material_point_history;

    inline void
    compute_pressure(
      ConstitutiveModelRequest<dim, Number> &constitutive_request,
      const point_index_t &point_index);

    inline void
    determine_delta_gamma(Number &delta_gamma, Number &alpha_n_plus_1,
                          const Number norm_ksi_trial,
                          const Number mu_bar,
                          const Number alpha_n,
                          const Number temperature,
                          const Number time_increment,
                          const Number tol, unsigned int max_iter) const;

    void compute_stress_deviator_and_d_gamma(
            ConstitutiveModelRequest<dim, Number> &constitutive_request,
            const point_index_t &point_index);

    void compute_heat_flux(
            ConstitutiveModelRequest<dim, Number> &constitutive_request,
            const point_index_t &point_index);

    void compute_thermo_elastic_heating(
            ConstitutiveModelRequest<dim, Number> &constitutive_request,
            const point_index_t &point_index);

    void compute_stored_heat_rate(
            ConstitutiveModelRequest<dim, Number> &constitutive_request,
            const point_index_t &point_index);

  };

} /* namespace PlasticityLab */

#endif /* THERMOPLASTICMATERIAL_H_ */
