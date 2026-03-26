/*
 * ThermoPlasticMaterial.cpp
 *
 *  Created on: 05 Jan 2015
 *      Author: maien
 */

#include <math.h>
#include <sstream>

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>

#include "symmetric_tensor_entries.h"

#include "ExponentialHardeningThermoviscoplasticYieldLaw.h"
#include "JohnsonCookThermoviscoplasticYieldLaw.h"

#include "ThermoPlasticMaterial.h"
#include "Constants.h"
#include "TensorUtilities.h"

namespace PlasticityLab {

  template <int dim, typename ViscoplasticYieldLaw, typename Number>
  ThermoPlasticMaterial<dim, ViscoplasticYieldLaw, Number>::
  ThermoPlasticMaterial(
                const Number kappa,
                const Number mu,
                const Number thermal_expansion_coefficient,
                const Number thermal_conductivity,
                const Number heat_capacity,
                const Number dissipation_factor,
                const ViscoplasticYieldLaw &viscoplastic_yield_law) :
      kappa (kappa),
      mu (mu),
      thermal_expansion_coefficient(thermal_expansion_coefficient),
      thermal_conductivity(thermal_conductivity),
      heat_capacity(heat_capacity),
      reference_temperature(293.15),
      dissipation_factor(dissipation_factor),
      viscoplastic_yield_law(viscoplastic_yield_law) { }

  template <int dim, typename ViscoplasticYieldLaw, typename Number>
  ThermoPlasticMaterial<dim, ViscoplasticYieldLaw, Number>::~ThermoPlasticMaterial() {
  }

  template <int dim, typename ViscoplasticYieldLaw, typename Number>
  void ThermoPlasticMaterial<dim, ViscoplasticYieldLaw, Number>::compute_constitutive_request(
          ConstitutiveModelRequest<dim, Number> &constitutive_request,
          const point_index_t &point_index) {
    const ConstitutiveModelUpdateFlags update_flags = constitutive_request.get_update_flags();

    if (update_pressure & update_flags)
      compute_pressure(constitutive_request, point_index);

    if ((update_stress_deviator | update_mechanical_dissipation) & update_flags)
      compute_stress_deviator_and_d_gamma(constitutive_request, point_index);

    if (update_heat_flux & update_flags)
      compute_heat_flux(constitutive_request, point_index);

    if (update_thermoelastic_heating & update_flags)
      compute_thermo_elastic_heating(constitutive_request, point_index);

    if (update_stored_heat & update_flags)
      compute_stored_heat_rate(constitutive_request, point_index);
  }

  template <int dim, typename ViscoplasticYieldLaw, typename Number>
  std::vector<Number> ThermoPlasticMaterial<dim, ViscoplasticYieldLaw, Number>::get_state_parameters(
        const point_index_t &point_index,
        const Tensor<2, dim, Number> &reference_transformation) const {
    std::vector<Number> state_parameters;
    state_parameters.reserve(get_material_parameter_count());

    const PointHistory<dim, Number> &point_history = material_point_history.at(point_index);
    state_parameters.push_back(std::log(1 + point_history.hardening_parameters.equivalent_plastic_strain));

    const Number reference_transformation_Jacobian = determinant(reference_transformation);
    const auto isochoric_reference_transformation =
                  std::pow(reference_transformation_Jacobian, -Constants<dim, Number>::one_third()) * reference_transformation;

    const SymmetricTensor<2, dim, Number> log_of_b_e =
        get_log_of_tensor(
          symmetrize(
            isochoric_reference_transformation
            * static_cast<Tensor<2, dim, Number>>(point_history.plastic_strain)
            * transpose(isochoric_reference_transformation)));

    for (const auto &element :
     dealii_utils::symmetric_tensor_entries(log_of_b_e)) {
      state_parameters.push_back(element);
    }

    for (const auto &element :
     dealii_utils::symmetric_tensor_entries(
      point_history.hardening_parameters.kinematic_hardening)
    ) {
      state_parameters.push_back(element);
    }

    state_parameters.push_back(std::log(std::pow(reference_transformation_Jacobian, 1) * point_history.material_Jacobian));

    return state_parameters;
  }

  template <int dim, typename ViscoplasticYieldLaw, typename Number>
  size_t ThermoPlasticMaterial<dim, ViscoplasticYieldLaw, Number>:: get_material_parameter_count() const {
    return 1 + 2 * (dim * (dim + 1) / 2) + 1; // one scalar and two symmetric tensors and one more scalar
  }

  template <int dim, typename ViscoplasticYieldLaw, typename Number>
  void ThermoPlasticMaterial<dim, ViscoplasticYieldLaw, Number>::set_state_parameters(
        const point_index_t &point_index,
        const std::vector<Number> &state_parameters,
        const Tensor<2, dim, Number> &reference_transformation) {
    PointHistory<dim, Number> &point_history = material_point_history.at(point_index);
    size_t cursor = 0;

    point_history.hardening_parameters.equivalent_plastic_strain = std::exp(state_parameters[cursor++]) - 1;

    SymmetricTensor<2, dim, Number> log_of_b_e;

    for (auto &element :
     dealii_utils::symmetric_tensor_entries(log_of_b_e)) {
      element = state_parameters[cursor++];
    }

    const Number reference_transformation_Jacobian = determinant(reference_transformation);
    const auto inverse_isochoric_reference_transformation =
                  std::pow(reference_transformation_Jacobian, Constants<dim, Number>::one_third()) * invert(reference_transformation);

    point_history.plastic_strain = symmetrize(
        inverse_isochoric_reference_transformation
        * static_cast<Tensor<2, dim, Number>>(get_exp_of_tensor(log_of_b_e))
        * transpose(inverse_isochoric_reference_transformation));

    for (auto &element :
     dealii_utils::symmetric_tensor_entries(
      point_history.hardening_parameters.kinematic_hardening
    )) {
      element = state_parameters[cursor++];
    }

    point_history.material_Jacobian = std::pow(reference_transformation_Jacobian, -1) * std::exp(state_parameters[cursor++]);
  }

  template <int dim, typename ViscoplasticYieldLaw, typename Number>
  Number ThermoPlasticMaterial<dim, ViscoplasticYieldLaw, Number>::get_material_Jacobian(const point_index_t &point_index) const {
    return material_point_history.at(point_index).material_Jacobian;
  }


  template <int dim, typename ViscoplasticYieldLaw, typename Number>
  dealii::SymmetricTensor<2, dim, Number> ThermoPlasticMaterial<dim, ViscoplasticYieldLaw, Number>::get_plastic_strain(const point_index_t &point_index) const {
    return material_point_history.at(point_index).plastic_strain;
  }




  template <int dim, typename ViscoplasticYieldLaw, typename Number>
  void
  ThermoPlasticMaterial<dim, ViscoplasticYieldLaw, Number>::setup_point_history (const point_index_t point_count) {
    {
      std::vector< PointHistory<dim, Number> > tmp;
      tmp.swap (material_point_history);
    }
    const typename PointHistory<dim, Number>::HardeningParameters hardening_parameters(
      Number(0.0),
      Number(0.0)*dealii::unit_symmetric_tensor<dim, Number>());
    const PointHistory< dim, Number> point_history(unit_symmetric_tensor<dim, Number>(),
                                                   hardening_parameters,
                                                   Number(0.0),
                                                   Number(1.0));
    material_point_history.resize (point_count, point_history);
  }

  template <int dim, typename ViscoplasticYieldLaw, typename Number>
  inline void
  ThermoPlasticMaterial<dim, ViscoplasticYieldLaw, Number>::
  compute_pressure(ConstitutiveModelRequest<dim, Number> &constitutive_request,
                   const point_index_t &point_index) {
    const Number J = constitutive_request.get_deformation_Jacobian();
    const Number temperature = constitutive_request.get_temperature();
    if (J > 0) {
      const Number p = kappa * (J-1.0/J) - 3 * kappa * thermal_expansion_coefficient * (temperature - reference_temperature) * (1 + 1./(J*J));
      constitutive_request.set_pressure(p);
      // pressure tangent
      const Number dp = kappa * (1 + 1/(J*J)) - 3 * kappa * thermal_expansion_coefficient * (temperature - reference_temperature) * (-2 * 1./(J*J*J));
      constitutive_request.set_pressure_tangent_modulus(dp);
    } else {
      std::ostringstream convert;
      convert << "Encountered deformation gradient with non-positive determinant! " << J;
      throw  MaterialDomainException(convert.str());
    }
  }

  template <int dim, typename ViscoplasticYieldLaw, typename Number>
  void ThermoPlasticMaterial<dim, ViscoplasticYieldLaw, Number>::compute_stress_deviator_and_d_gamma(
    ConstitutiveModelRequest<dim, Number> &constitutive_request,
    const point_index_t &point_index) {
    const ConstitutiveModelUpdateFlags update_flags = constitutive_request.get_update_flags();
    const Tensor<2, dim, Number> b_e = material_point_history[point_index].plastic_strain;
    typename PointHistory<dim, Number>::HardeningParameters hardening_parameters =
      material_point_history[point_index].hardening_parameters;
    const Tensor<2, dim, Number> isochoric_deformation_gradient = constitutive_request.get_deformation_gradient();
    const Number temperature = constitutive_request.get_temperature();
    const auto b_e_bar_next = static_cast<SymmetricTensor<2, dim, Number> >(
                                symmetrize(isochoric_deformation_gradient * b_e
                                           * transpose(isochoric_deformation_gradient)));
    if(std::isnan(b_e_bar_next.norm())) {
      std::cout << "b_e_bar_next is nan: " << b_e_bar_next << std::endl;
      std::cout << "b_e: " << b_e << std::endl;
      std::cout << "isochoric_deformation_gradient: " << isochoric_deformation_gradient << std::endl;
      std::ostringstream convert;
      convert << "Encountered an elastic deviatoric tensor with NaN values! ";
      throw  MaterialDomainException(convert.str());
    }
    const auto epsilon_e_bar_next = get_log_of_tensor<>(b_e_bar_next);
    const SymmetricTensor<2, dim, Number> dev_stress_trial = deviator((0.5 * mu) * epsilon_e_bar_next);
    // const SymmetricTensor<2, dim, Number> dev_stress_trial = deviator((0.5 * mu) * b_e_bar_next);
    const SymmetricTensor<2, dim, Number> ksi_trial = dev_stress_trial
                                                      - hardening_parameters.kinematic_hardening;
    const Number norm_ksi_trial = ksi_trial.norm();
    const Number mu_bar = (0.5 * mu);
    // const Number mu_bar = (0.5 * mu) * Constants<dim, Number>::one_third()
    //                       * trace(b_e_bar_next);
    const SymmetricTensor<2, dim, Number> stress_flow_direction = ksi_trial
        / norm_ksi_trial;
    constitutive_request.set_b_e_bar(b_e_bar_next);
    constitutive_request.set_mu((0.5 * mu));
    const Number time_increment = constitutive_request.get_time_increment();

    const Number trial_yield_criterion = viscoplastic_yield_law.trial_yield_criterion(norm_ksi_trial,
                                   hardening_parameters.equivalent_plastic_strain,
                                   0.0,
                                   time_increment,
                                   temperature);

    if (0 < trial_yield_criterion && constitutive_request.get_is_plastic()) {
      constitutive_request.set_is_plastic(true);
      Number delta_gamma, alpha_n_plus_1;
      try {
        determine_delta_gamma(delta_gamma, alpha_n_plus_1, norm_ksi_trial,
                              mu_bar, hardening_parameters.equivalent_plastic_strain, temperature,
                              time_increment, 1e-04, 300);
      } catch(MaterialDomainException exc) {
        std::cout << "isochoric_deformation_gradient: " << isochoric_deformation_gradient
                  << "\nb_e: " << b_e
                  << "\nb_e_bar_next: " << b_e_bar_next
                  << "\nmu_bar: " << mu_bar
                  << std::endl;
        std::cerr << exc.what() << std::endl;
        // throw exc;
      }
      constitutive_request.set_delta_gamma(delta_gamma);

      // 4. Update back stress, plastic strain and stress
      Number K_alpha_n_plus_1, K_alpha_n, H_alpha_n_plus_1, H_alpha_n;
      Number DK_alpha_n_plus_1, DH_alpha_n_plus_1;
      viscoplastic_yield_law.hardening_values(H_alpha_n,
                                   K_alpha_n,
                                   hardening_parameters.equivalent_plastic_strain,
                                   0.0,
                                   time_increment,
                                   temperature);
      const Number y_alpha = viscoplastic_yield_law.hardening_values(
                               H_alpha_n_plus_1,
                               K_alpha_n_plus_1,
                               alpha_n_plus_1,
                               delta_gamma,
                               time_increment,
                               temperature);
      const Number d_y_alpha_d_alpha = viscoplastic_yield_law.hardening_alpha_derivatives(
                                         DH_alpha_n_plus_1,
                                         DK_alpha_n_plus_1,
                                         alpha_n_plus_1,
                                         delta_gamma,
                                         time_increment,
                                         temperature);

      if (update_stress_deviator & update_flags) {
        const SymmetricTensor<2, dim, Number> stress = deviator(dev_stress_trial
                                                                - 2 * mu_bar * delta_gamma
                                                                * stress_flow_direction);
        constitutive_request.set_stress_deviator(stress);
        constitutive_request.set_dH(DH_alpha_n_plus_1);
        constitutive_request.set_dK(DK_alpha_n_plus_1);
        if (update_material_point_history & update_flags) {
          material_point_history[point_index].hardening_parameters.equivalent_plastic_strain =
            alpha_n_plus_1;
          material_point_history[point_index].hardening_parameters.kinematic_hardening =
            hardening_parameters.kinematic_hardening
            + Constants<dim, Number>::sqrt2thirds()
            * (K_alpha_n_plus_1 - K_alpha_n)
            * stress_flow_direction;

          // // compute one_third_I_bar_e (c.f. Simo, Miehe 1992 pp 64)
          // const auto stress_over_mu = stress / (0.5 * mu);
          // const Number norm_stress_over_mu = stress_over_mu.norm();
          // const Number J_e2 = 0.5 * norm_stress_over_mu * norm_stress_over_mu;
          // const Number J_e3 = determinant(stress_over_mu);
          // const Number q = 0.5*(1-J_e3);
          // const Number sqrt_d = std::sqrt(-std::pow(Constants<dim, Number>::one_third()*J_e2, 3) + q*q);
          // const Number one_third_I_bar_e = std::pow(q + sqrt_d, Constants<dim, Number>::one_third())
          //                                  + std::pow(q - sqrt_d, Constants<dim, Number>::one_third());

          // auto b_e_bar = stress / (0.5 * mu)
          //                       + Constants<dim, Number>::one_third() * trace(b_e_bar_next)
          //                       * unit_symmetric_tensor<dim, Number>();
          auto b_e_bar = get_exp_of_tensor(stress / (0.5 * mu));

          // correct volumetric component so that return mapping does not change volume
          const Number det_b_e_bar_trial = determinant(b_e_bar_next);
          Number det_plastic_strain = determinant(b_e_bar);
          if(std::abs(det_plastic_strain - 1) > 1e-7) {
            std::cout << "det_plastic_strain: " << det_plastic_strain << std::endl;
          }
          while (std::abs(det_plastic_strain-det_b_e_bar_trial) > 1e-10) {
            const Number u = (det_b_e_bar_trial - det_plastic_strain) / (det_plastic_strain * trace(invert(b_e_bar)));
            b_e_bar += u*unit_symmetric_tensor<dim, Number>();
            det_plastic_strain = determinant(b_e_bar);
          }

          const Tensor<2, dim, Number> inverse_isochoric_deformation_gradient = invert(isochoric_deformation_gradient);
          material_point_history[point_index].plastic_strain = b_e_bar;
        }
      } /*if(update_stress_deviator & update_flags)*/
      if (update_mechanical_dissipation & update_flags) {
        const Number time_increment = constitutive_request.get_time_increment();
        const Number mechanical_dissipation =
          dissipation_factor
          * Constants<dim, Number>::sqrt2thirds()
          * y_alpha * delta_gamma
          / (1000. * time_increment); // [J.mm^-3.s^-1]
        constitutive_request.set_mechanical_dissipation(mechanical_dissipation);
        Number DH_theta, DK_theta;
        const Number d_y_alpha_d_theta =
          viscoplastic_yield_law.hardening_temperature_derivatives(DH_theta, DK_theta,
                                                        alpha_n_plus_1,
                                                        delta_gamma,
                                                        time_increment,
                                                        temperature);
        const Number mechanical_dissipation_temperature_tangent =
          dissipation_factor
          * d_y_alpha_d_theta
          * (Constants<dim, Number>::sqrt2thirds() * delta_gamma
             - y_alpha / (3 * mu_bar))
          / (1000. * time_increment);  // [J.mm^-3.s^-3.K^-1]
        constitutive_request.set_mechanical_dissipation_tangent_modulus(
          mechanical_dissipation_temperature_tangent);
      } /*if (update_mechanical_dissipation & updateFlags)*/
    } /*if ( trial yield criterion test )*/
    else {
      constitutive_request.set_delta_gamma(0.0);
      constitutive_request.set_is_plastic(false);
      if (update_stress_deviator & update_flags) {
        constitutive_request.set_stress_deviator(dev_stress_trial);
        constitutive_request.set_dK(0.0);
        constitutive_request.set_dH(0.0);
      }
      if (update_mechanical_dissipation & update_flags) {
        constitutive_request.set_mechanical_dissipation(0.0);
        constitutive_request.set_mechanical_dissipation_tangent_modulus(0.0);
      }
      // The following is only necessary for the elastic case when b_e is tracked instead of G_p
      if (update_stress_deviator & update_flags) {
        if (update_material_point_history & update_flags) {
          material_point_history[point_index].plastic_strain = b_e_bar_next;
        }
      }
    } /*else*/
    // The following is only necessary if the total Jacobian is tracked (J_c * J_m), rather than just J_m
    if (update_material_point_history & update_flags) {
      material_point_history.at(point_index).material_Jacobian = constitutive_request.get_unprojected_deformation_Jacobian();
    }
  } /*computeStressDeviatorAndDGamma()*/

  template <int dim, typename ViscoplasticYieldLaw, typename Number>
  void ThermoPlasticMaterial<dim, ViscoplasticYieldLaw, Number>::
  compute_heat_flux(
        ConstitutiveModelRequest<dim, Number> &constitutive_request,
        const point_index_t &point_index) {
    const Tensor<1, dim, Number> thermal_gradient = constitutive_request.get_thermal_gradient();
    constitutive_request.set_heat_flux(thermal_conductivity * thermal_gradient);
    constitutive_request.set_heat_flux_tangent_moduli(thermal_conductivity * unit_symmetric_tensor<dim, Number>());
  }

  template <int dim, typename ViscoplasticYieldLaw, typename Number>
  void ThermoPlasticMaterial<dim, ViscoplasticYieldLaw, Number>::
  compute_thermo_elastic_heating(
        ConstitutiveModelRequest<dim, Number> &constitutive_request,
        const point_index_t &point_index) {
    const auto point_history = material_point_history.at(point_index);
    const Number J = constitutive_request.get_deformation_Jacobian();
    const Number previous_J = constitutive_request.get_previous_deformation_Jacobian();
    const Number J_time_rate = constitutive_request.get_deformation_Jacobian_time_rate();
    const Number theta = constitutive_request.get_temperature();
    const Number previous_theta = constitutive_request.get_previous_temperature();
    const Number time_increment = constitutive_request.get_time_increment();
    if (J != 0 and previous_J != 0) {
      const Number eta = (3.0/1000.0) * kappa * thermal_expansion_coefficient * (J - 1.0/J);
      const Number previous_eta = (3.0/1000.0) * kappa * thermal_expansion_coefficient * (previous_J - 1.0/previous_J);
      constitutive_request.set_thermo_elastic_heating(-theta * (eta - previous_eta) / time_increment);
      constitutive_request.set_thermo_elastic_heating_tangent_modulus(-(eta - previous_eta) / time_increment);
    } else {
      // TODO define and throw appropriate exception: bad deformation gradient!
      constitutive_request.set_thermo_elastic_heating(0);
      constitutive_request.set_thermo_elastic_heating_tangent_modulus(0);
    }
  }

  template <int dim, typename ViscoplasticYieldLaw, typename Number>
  void ThermoPlasticMaterial<dim, ViscoplasticYieldLaw, Number>::
  compute_stored_heat_rate(
        ConstitutiveModelRequest<dim, Number> &constitutive_request,
        const point_index_t &point_index) {
    const Number stored_heat_rate = heat_capacity * constitutive_request.get_temperature_time_rate();
    constitutive_request.set_stored_heat_rate(stored_heat_rate);
    constitutive_request.set_stored_heat_rate_tangent_modulus(heat_capacity);
  }

  template <int dim, typename ViscoplasticYieldLaw, typename Number>
  inline void
  ThermoPlasticMaterial<dim, ViscoplasticYieldLaw, Number>::
  determine_delta_gamma(Number &delta_gamma, Number &alpha_n_plus_1,
                        const Number norm_ksi_trial,
                        const Number mu_bar,
                        const Number alpha_n,
                        const Number temperature,
                        const Number time_increment,
                        const Number tol,
                        const unsigned int max_iter) const {
    unsigned int k = 0;
    const Number sqrt2thirds = Constants<dim, Number>::sqrt2thirds();
    const Number two_thirds = Constants<dim, Number>::two_thirds();
    Number g_of_gamma_k, Dg_of_gamma_k;
    Number H_alpha_n, H_alpha_n_plus_1, K_alpha_n, K_alpha_n_plus_1;
    Number DK_alpha_n_plus_1, DH_alpha_n_plus_1;

    delta_gamma = 0;
    alpha_n_plus_1 = alpha_n;

    viscoplastic_yield_law.hardening_values(H_alpha_n, K_alpha_n, alpha_n, 0.0, time_increment, temperature);

    viscoplastic_yield_law.hardening_values(H_alpha_n_plus_1, K_alpha_n_plus_1, alpha_n_plus_1, delta_gamma, time_increment, temperature);
    g_of_gamma_k = norm_ksi_trial
                   - 2 * mu_bar * delta_gamma
                   - sqrt2thirds * H_alpha_n_plus_1
                   - sqrt2thirds * (K_alpha_n_plus_1 - K_alpha_n);
    do {
      viscoplastic_yield_law.hardening_alpha_derivatives(DH_alpha_n_plus_1, DK_alpha_n_plus_1, alpha_n_plus_1, delta_gamma, time_increment, temperature);
      Dg_of_gamma_k = -2 * mu_bar - two_thirds * (DK_alpha_n_plus_1 + DH_alpha_n_plus_1);

      delta_gamma = delta_gamma - g_of_gamma_k / Dg_of_gamma_k;
      alpha_n_plus_1 = alpha_n + sqrt2thirds * delta_gamma;

      viscoplastic_yield_law.hardening_values(H_alpha_n_plus_1, K_alpha_n_plus_1, alpha_n_plus_1, delta_gamma, time_increment, temperature);
      g_of_gamma_k = norm_ksi_trial
                     - 2 * mu_bar * delta_gamma
                     - sqrt2thirds * H_alpha_n_plus_1
                     - sqrt2thirds * (K_alpha_n_plus_1 - K_alpha_n);
    } while (std::fabs(g_of_gamma_k) > tol && ++k < max_iter);
    if (std::fabs(g_of_gamma_k) > tol) {
      std::ostringstream convert;
      convert << "Did not converge after " << k << " iterations. g: " << g_of_gamma_k
              << ", norm_ksi_trial: " << norm_ksi_trial
              << ", mu_bar: " << mu_bar
              << ", H_alpha_n_plus_1: " << H_alpha_n_plus_1
              << ", alpha: " << alpha_n_plus_1
              << ", delta_gamma: " << delta_gamma
              << ", temperature: " << temperature
              << ", gradient: " << Dg_of_gamma_k;
      throw MaterialDomainException(convert.str());
    }
  }

  template class ThermoPlasticMaterial<3, ExponentialHardeningThermoviscoplasticYieldLaw<double>, double>;
  template class ThermoPlasticMaterial<2, ExponentialHardeningThermoviscoplasticYieldLaw<double>, double>;

  template class ThermoPlasticMaterial<3, JohnsonCookThermoviscoplasticYieldLaw<double>, double>;
  template class ThermoPlasticMaterial<2, JohnsonCookThermoviscoplasticYieldLaw<double>, double>;

} /* namespace PlasticityLab */
