/*
 * ConstitutiveModelRequest.h
 *
 *  Created on: 03 Feb 2015
 *      Author: maien
 */

#ifndef CONSTITUTIVEMODELREQUEST_H_
#define CONSTITUTIVEMODELREQUEST_H_

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>

#include "Constants.h"
#include "ConstitModelUpdateFlags.h"
#include "TensorUtilities.h"

namespace PlasticityLab {

  template <int dim, typename Number>
  class ConstitutiveModelRequest {
   public:
    ConstitutiveModelRequest(ConstitutiveModelUpdateFlags);
    virtual ~ConstitutiveModelRequest();

    // Interface to be used by request client (FE system assembler)
    //   --request configuration stage--
    void set_deformation_gradient(const Tensor<2, dim, Number> &deformation_gradient);
    void set_deformation_Jacobian(const Number deformation_Jacobian);
    void set_unprojected_deformation_Jacobian(const Number unprojected_deformation_Jacobian);
    void set_previous_deformation_Jacobian(const Number previous_deformation_Jacobian);
    void set_deformation_Jacobian_time_rate(const Number deformation_Jacobian_time_rate);
    void set_temperature(const Number temperature);
    void set_previous_temperature(const Number previous_temperature);
    void set_temperature_time_rate(const Number temperature_time_rate);
    void set_thermal_gradient(const Tensor<1, dim, Number> &thermalGradient);
    void set_time_increment(const Number timeIncrement);

    // Interface to be used by request client (FE system assembler)
    //   --request response retrieval and interrogation stage--
    Number get_pressure();
    Number get_pressure_tangent(const Number volume_change_increment);
    SymmetricTensor<2, dim, Number> get_stress_deviator() const;
    SymmetricTensor<2, dim, Number> get_stress_deviator_tangent(const Tensor<2, dim, Number> &strain_increment) const;
    Tensor<1, dim, Number> get_heat_flux() const;
    Tensor<1, dim, Number> get_heat_flux_tangent(const Tensor<1, dim, Number> &thermal_gradient_increment) const;
    Number  get_stored_heat_rate() const;
    Number  get_stored_heat_rate_tangent(const Number temperature_increment) const;
    Number  get_elastic_entropy() const;
    bool    get_is_plastic() const;
    Number  get_elastic_entropy_tangent(const Number temperature_increment) const;
    Number  get_mechanical_dissipation() const;
    Number  get_mechanical_dissipation_tangent(const Number temperature_increment) const;
    Number  get_thermo_elastic_heating() const;
    Number  get_thermo_elastic_heating_tangent(const Number temperature_increment) const;

    // interface used by constitutive model object to perform computation
    // TODO consider hiding this interface and exposing it through adapter
    ConstitutiveModelUpdateFlags get_update_flags() const;
    Tensor<2, dim, Number> get_deformation_gradient() const;
    Number get_deformation_Jacobian() const;
    Number get_unprojected_deformation_Jacobian() const;
    Number get_previous_deformation_Jacobian() const;
    Number get_deformation_Jacobian_time_rate() const;
    Number get_temperature() const;
    Number get_previous_temperature() const;
    Number get_temperature_time_rate() const;
    Tensor<1, dim, Number> get_thermal_gradient() const;
    Number get_time_increment() const;

    void set_pressure(Number pressure);
    void set_stress_deviator(const SymmetricTensor<2, dim, Number> &stress_deviator);
    void set_heat_flux(const Tensor<1, dim, Number> &heat_flux);
    void set_stored_heat_rate(const Number stored_heat_rate);
    void set_elastic_entropy(const Number elastic_entropy);
    void set_mechanical_dissipation(const Number mechanical_dissipation);
    void set_thermo_elastic_heating(const Number thermo_elastic_heating);

    // TODO this can be changed so that smaller objects can be set and used
    //      to construct the tangents than the full moduli tensors
    void set_pressure_tangent_modulus(const Number pressure_tangent_modulus);
    void set_b_e_bar(const SymmetricTensor<2, dim, Number> &b_e_bar);
    void set_mu(const Number mu);
    void set_is_plastic(const bool is_plastic);
    void set_delta_gamma(const Number delta_gamma);
    void set_dK(const Number dK);
    void set_dH(const Number dH);
    void set_heat_flux_tangent_moduli(const SymmetricTensor<2, dim, Number> &heat_flux_tangent_modului);
    void set_stored_heat_rate_tangent_modulus(const Number stored_heat_rate_tangent_modulus);
    void set_elastic_entropy_tangent_modulus(const Number elastic_entropy_tangent_modulus);
    void set_mechanical_dissipation_tangent_modulus(const Number mechanicalDissipationTangentModulus);
    void set_thermo_elastic_heating_tangent_modulus(const Number thermo_elastic_heating_tangent_modulus);

   protected:
    ConstitutiveModelUpdateFlags update_flags;

    Tensor<2, dim, Number> deformation_gradient;
    Number deformation_Jacobian, previous_deformation_Jacobian, deformation_Jacobian_time_rate;
    Number unprojected_deformation_Jacobian;
    Number temperature, previous_temperature, temperature_time_rate;
    Tensor<1, dim, Number> thermal_gradient;

    Number pressure;
    SymmetricTensor<2, dim, Number> stress_deviator;
    Tensor<1, dim, Number> heat_flux;
    Number stored_heat_rate;
    Number  elastic_entropy;
    Number  mechanical_dissipation;
    Number  thermo_elastic_heating;

    Number time_increment;

    bool is_plastic;
    Number pressure_tangent_modulus;
    Number dK, dH, mu, delta_gamma;
    SymmetricTensor<2, dim, Number> b_e_bar;
    SymmetricTensor<2, dim, Number> heat_flux_tangent_moduli;
    Number  stored_heat_rate_tangent_modulus;
    Number  elastic_entropy_tangent_modulus;
    Number  mechanical_dissipation_tangent_modulus;
    Number  thermo_elastic_heating_tangent_modulus;
  };

  template <int dim, typename Number>
  ConstitutiveModelRequest<dim, Number>::
  ConstitutiveModelRequest(ConstitutiveModelUpdateFlags update_flags):
    update_flags(update_flags) {
      is_plastic = true;  // not necessarily elastic
  }

  template <int dim, typename Number>
  bool ConstitutiveModelRequest<dim, Number>::get_is_plastic() const {
    return is_plastic;
  }

  template <int dim, typename Number>
  ConstitutiveModelRequest<dim, Number>::~ConstitutiveModelRequest() { }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_deformation_gradient(const Tensor<2, dim, Number> &deformation_gradient) {
    this->deformation_gradient = deformation_gradient;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_deformation_Jacobian(Number deformation_Jacobian) {
    this->deformation_Jacobian = deformation_Jacobian;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_unprojected_deformation_Jacobian(Number unprojected_deformation_Jacobian) {
    this->unprojected_deformation_Jacobian = unprojected_deformation_Jacobian;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_previous_deformation_Jacobian(Number previous_deformation_Jacobian) {
    this->previous_deformation_Jacobian = previous_deformation_Jacobian;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_deformation_Jacobian_time_rate(Number deformation_Jacobian_time_rate) {
    this->deformation_Jacobian_time_rate = deformation_Jacobian_time_rate;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_temperature(const Number temperature) {
    this->temperature = temperature;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_previous_temperature(const Number previous_temperature) {
    this->previous_temperature = previous_temperature;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_temperature_time_rate(const Number temperature_time_rate) {
    this->temperature_time_rate = temperature_time_rate;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_thermal_gradient(const Tensor<1, dim, Number> &thermal_gradient) {
    this->thermal_gradient = thermal_gradient;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_time_increment(const Number time_increment) {
    this->time_increment = time_increment;
  }

  template <int dim, typename Number>
  Number ConstitutiveModelRequest<dim, Number>::get_pressure() {
    return pressure;
  }

  template <int dim, typename Number>
  Number
  ConstitutiveModelRequest<dim, Number>::
  get_pressure_tangent(const Number volume_change_increment) {
    return pressure_tangent_modulus * volume_change_increment;
  }

  template <int dim, typename Number>
  SymmetricTensor<2, dim, Number> ConstitutiveModelRequest<dim, Number>::
  get_stress_deviator() const {
    return stress_deviator;
  }

  template <int dim, typename Number>
  SymmetricTensor<2, dim, Number> ConstitutiveModelRequest<dim, Number>::
  get_stress_deviator_tangent(const Tensor<2, dim, Number> &strain_increment) const {
    const Number twothirds = Constants<dim, Number>::two_thirds();
    const auto tensor_b_e_bar = static_cast<Tensor<2,dim,Number> >(b_e_bar);
    SymmetricTensor<2, dim, Number> d_b_e_bar = symmetrize(2 * strain_increment * tensor_b_e_bar);
    SymmetricTensor<2, dim, Number> d_dev_b_e_bar = get_log_of_tensor_variation(b_e_bar, d_b_e_bar);
    SymmetricTensor<2, dim, Number> d_trial_stress_dev = mu * d_dev_b_e_bar;

    // TODO ensure that all the debugging tests were removed
    if (is_plastic) {
      // Number mu_bar = Constants<dim, Number>::one_third() * mu * trace(b_e_bar);
      // Number d_mu_bar = Constants<dim, Number>::one_third() * mu * trace(d_b_e_bar);
      // Number norm_dev_b_e_bar = (deviator(b_e_bar)).norm();
      // SymmetricTensor<2, dim, Number> dev_b_e_direction = deviator(b_e_bar) / norm_dev_b_e_bar;
      Number mu_bar = mu;
      Number d_mu_bar = 0;

      const auto epsilon_e_bar = get_log_of_tensor(b_e_bar);
      Number norm_dev_b_e_bar = (epsilon_e_bar).norm();
      SymmetricTensor<2, dim, Number> dev_b_e_direction = epsilon_e_bar / norm_dev_b_e_bar;

      SymmetricTensor<2, dim, Number> d_dev_b_e_direction =
        (1.0 / norm_dev_b_e_bar) * (d_dev_b_e_bar - dev_b_e_direction * (dev_b_e_direction * d_dev_b_e_bar));
      Number d_delta_gamma = (dev_b_e_direction * d_trial_stress_dev - 2 * d_mu_bar * delta_gamma) / (2 * mu_bar + twothirds * (dK + dH));

      return deviator(d_trial_stress_dev
                      - (  2 * mu_bar * delta_gamma * d_dev_b_e_direction
                         + 2 * mu_bar * d_delta_gamma * dev_b_e_direction
                         + 2 * d_mu_bar * delta_gamma * dev_b_e_direction));
    }

    return deviator(d_trial_stress_dev);
  }

  template <int dim, typename Number>
  Tensor<1, dim, Number>
  ConstitutiveModelRequest<dim, Number>::get_heat_flux() const {
    return heat_flux;
  }

  template <int dim, typename Number>
  Tensor<1, dim, Number>
  ConstitutiveModelRequest<dim, Number>::
  get_heat_flux_tangent(const Tensor<1, dim, Number> &thermal_gradient_increment) const {
    return heat_flux_tangent_moduli * thermal_gradient_increment;
  }

  template <int dim, typename Number>
  Number
  ConstitutiveModelRequest<dim, Number>::get_stored_heat_rate() const {
    return stored_heat_rate;
  }

  template <int dim, typename Number>
  Number ConstitutiveModelRequest<dim, Number>::
  get_stored_heat_rate_tangent(const Number temperature_increment) const {
    return stored_heat_rate_tangent_modulus * temperature_increment;
  }

  template <int dim, typename Number>
  Number ConstitutiveModelRequest<dim, Number>::
  get_elastic_entropy() const {
    return elastic_entropy;
  }

  template <int dim, typename Number>
  Number ConstitutiveModelRequest<dim, Number>::
  get_elastic_entropy_tangent(const Number temperature_increment) const {
    return elastic_entropy_tangent_modulus * time_increment;
  }

  template <int dim, typename Number>
  Number ConstitutiveModelRequest<dim, Number>::
  get_mechanical_dissipation() const {
    return mechanical_dissipation;
  }

  template <int dim, typename Number>
  Number ConstitutiveModelRequest<dim, Number>::
  get_mechanical_dissipation_tangent(const Number temperature_increment) const {
    return mechanical_dissipation_tangent_modulus * temperature_increment;
  }

  template <int dim, typename Number>
  Number ConstitutiveModelRequest<dim, Number>::
  get_thermo_elastic_heating() const {
    return thermo_elastic_heating;
  }

  template <int dim, typename Number>
  Number ConstitutiveModelRequest<dim, Number>::
  get_thermo_elastic_heating_tangent(const Number temperature_increment) const {
    return thermo_elastic_heating_tangent_modulus * temperature_increment;
  }

  template <int dim, typename Number>
  ConstitutiveModelUpdateFlags ConstitutiveModelRequest<dim, Number>::
  get_update_flags() const {
    return update_flags;
  }

  template <int dim, typename Number>
  Tensor<2, dim, Number> ConstitutiveModelRequest<dim, Number>::
  get_deformation_gradient() const {
    return deformation_gradient;
  }

  template <int dim, typename Number>
  Number ConstitutiveModelRequest<dim, Number>::
  get_deformation_Jacobian() const {
    return deformation_Jacobian;
  }

  template <int dim, typename Number>
  Number ConstitutiveModelRequest<dim, Number>::
  get_unprojected_deformation_Jacobian() const {
    return unprojected_deformation_Jacobian;
  }

  template <int dim, typename Number>
  Number ConstitutiveModelRequest<dim, Number>::
  get_previous_deformation_Jacobian() const {
    return previous_deformation_Jacobian;
  }

  template <int dim, typename Number>
  Number ConstitutiveModelRequest<dim, Number>::
  get_deformation_Jacobian_time_rate() const {
    return deformation_Jacobian_time_rate;
  }

  template <int dim, typename Number>
  Number ConstitutiveModelRequest<dim, Number>::
  get_temperature() const {
    return temperature;
  }

  template <int dim, typename Number>
  Number ConstitutiveModelRequest<dim, Number>::
  get_previous_temperature() const {
    return previous_temperature;
  }

  template <int dim, typename Number>
  Number ConstitutiveModelRequest<dim, Number>::
  get_temperature_time_rate() const {
    return temperature_time_rate;
  }

  template <int dim, typename Number>
  Tensor<1, dim, Number> ConstitutiveModelRequest<dim, Number>::
  get_thermal_gradient() const {
    return thermal_gradient;
  }

  template <int dim, typename Number>
  Number ConstitutiveModelRequest<dim, Number>::
  get_time_increment() const {
    return time_increment;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_pressure(Number pressure) {
    this->pressure = pressure;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_stress_deviator(const SymmetricTensor<2, dim, Number> &stress_deviator) {
    this->stress_deviator = stress_deviator;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_b_e_bar(const SymmetricTensor<2, dim, Number> &b_e_bar) {
    this->b_e_bar = b_e_bar;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_mu(const Number mu) {
    this->mu = mu;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_is_plastic(const bool is_plastic) {
    this->is_plastic = is_plastic;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_delta_gamma(const Number delta_gamma) {
    this->delta_gamma = delta_gamma;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_dK(const Number dK) {
    this->dK = dK;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_dH(const Number dH) {
    this->dH = dH;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_heat_flux(const Tensor<1, dim, Number> &heat_flux) {
    this->heat_flux = heat_flux;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_stored_heat_rate(const Number stored_heat_rate) {
    this->stored_heat_rate = stored_heat_rate;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_elastic_entropy(const Number elastic_entropy) {
    this->elastic_entropy = elastic_entropy;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_mechanical_dissipation(const Number mechanical_dissipation) {
    this->mechanical_dissipation = mechanical_dissipation;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_thermo_elastic_heating(const Number thermo_elastic_heating) {
    this->thermo_elastic_heating = thermo_elastic_heating;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_pressure_tangent_modulus(const Number pressure_tangent_modulus) {
    this->pressure_tangent_modulus = pressure_tangent_modulus;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_heat_flux_tangent_moduli(const SymmetricTensor<2, dim, Number> &heat_flux_tangent_modului) {
    this->heat_flux_tangent_moduli = heat_flux_tangent_modului;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_stored_heat_rate_tangent_modulus(const Number stored_heat_rate_tangent_modulus) {
    this->stored_heat_rate_tangent_modulus = stored_heat_rate_tangent_modulus;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_elastic_entropy_tangent_modulus(const Number elastic_entropy_tangent_modulus) {
    this->elastic_entropy_tangent_modulus = elastic_entropy_tangent_modulus;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_mechanical_dissipation_tangent_modulus(const Number mechanicalDissipationTangentModulus) {
    this->mechanical_dissipation_tangent_modulus = mechanicalDissipationTangentModulus;
  }

  template <int dim, typename Number>
  void ConstitutiveModelRequest<dim, Number>::
  set_thermo_elastic_heating_tangent_modulus(const Number thermo_elastic_heating_tangent_modulus) {
    this->thermo_elastic_heating_tangent_modulus = thermo_elastic_heating_tangent_modulus;
  }

} /* namespace PlasticityLab */

#endif /* CONSTITUTIVEMODELREQUEST_H_ */
