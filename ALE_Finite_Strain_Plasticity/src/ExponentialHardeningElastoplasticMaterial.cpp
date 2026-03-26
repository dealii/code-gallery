/*
 * ExponentialHardeningElastoplasticMaterial.cpp
 *
 *  Created on: 10 Jul 2014
 *      Author: cerecam
 */


#include <math.h>
#include <deal.II/base/symmetric_tensor.h>

#include "ExponentialHardeningElastoplasticMaterial.h"
#include "utilities.h"

namespace PlasticityLab {

  template <int dim, typename Number>
  ExponentialHardeningElastoplasticMaterial<dim, Number>::
  ExponentialHardeningElastoplasticMaterial
  (const Number kappa,
   const Number mu,
   const Number K_0,
   const Number K_infty,
   const Number delta,
   const Number H_bar,
   const Number beta)    :
    kappa (kappa),
    mu (mu),
    K_0(K_0),
    K_infty(K_infty),
    delta(delta),
    H_bar(H_bar),
    beta(beta),

    stress_strain_tensor_kappa (kappa
                                * outer_product(unit_symmetric_tensor<dim>(),
                                                unit_symmetric_tensor<dim>())),
    stress_strain_tensor_mu (2 * mu
                             * (identity_tensor<dim>()
                                - outer_product(unit_symmetric_tensor<dim>(),
                                                unit_symmetric_tensor<dim>()) / 3.0)) {

  }

  template <int dim, typename Number>
  ExponentialHardeningElastoplasticMaterial<dim, Number>::
  ~ExponentialHardeningElastoplasticMaterial() {
  }

  template <int dim, typename Number>
  std::vector<Number> ExponentialHardeningElastoplasticMaterial<dim, Number>::get_state_parameters(
      const point_index_t &point_index,
      const Tensor<2, dim, Number> &reference_transformation) const {
    throw NotImplementedException();
  }

  template <int dim, typename Number>
  void ExponentialHardeningElastoplasticMaterial<dim, Number>::
  set_state_parameters(
        const point_index_t &point_index,
        const std::vector<Number> &state_parameters,
        const Tensor<2, dim, Number> &reference_transformation) {
    throw NotImplementedException();
  }

  template <int dim, typename Number>
  size_t ExponentialHardeningElastoplasticMaterial<dim, Number>::
  get_material_parameter_count() const {
    throw NotImplementedException();
  }


  template <int dim, typename Number>
  Number ExponentialHardeningElastoplasticMaterial<dim, Number>::get_material_Jacobian(const point_index_t &point_index) const {
    throw NotImplementedException();
  }

  template <int dim, typename Number>
  dealii::SymmetricTensor<2, dim, Number> ExponentialHardeningElastoplasticMaterial<dim, Number>::get_plastic_strain(const point_index_t &point_index) const {
    throw NotImplementedException();
  }


  template <int dim, typename Number>
  void ExponentialHardeningElastoplasticMaterial<dim, Number>::
  compute_constitutive_request(ConstitutiveModelRequest<dim, Number> &constitutive_request,
                               const point_index_t &point_index) {
    SymmetricTensor<2, dim, Number> plastic_strain = material_point_history[point_index].plastic_strain;
    typename PointHistory<dim, Number>::HardeningParameters
    hardening_parameters = material_point_history[point_index].hardening_parameters;

    auto deformation_gradient = static_cast<SymmetricTensor<2, dim, Number> >(constitutive_request.get_deformation_gradient());
    SymmetricTensor<4, dim, Number> elastoplastic_tangent_moduli;
    SymmetricTensor<2, dim, Number> deviator_strain_tensor = deviator(deformation_gradient);

    SymmetricTensor<2, dim, Number> dev_stress_trial = 2 * mu * (deviator_strain_tensor - plastic_strain);
    SymmetricTensor<2, dim, Number>
    ksi_trial = dev_stress_trial - hardening_parameters.kinematic_hardening;

    Number norm_ksi_trial = ksi_trial.norm();
    if (trial_yield_criterion( norm_ksi_trial, hardening_parameters.equivalent_plastic_strain ) > 0) {
      Number delta_gamma, alpha_n_plus_1;
      determine_delta_gamma(delta_gamma, alpha_n_plus_1,
                            norm_ksi_trial,
                            hardening_parameters.equivalent_plastic_strain,
                            1e-10, 200);
      SymmetricTensor<2, dim, Number> stress_flow_direction = ksi_trial / norm_ksi_trial;

      // 4. Update back stress, plastic strain and stress
      const Number sqrt2thirds = sqrt((Number)2 / (Number)3);
      Number H_alpha_n_plus_1, H_alpha_n, K_alpha_n_plus_1, K_alpha_n;
      Number DH_alpha_n_plus_1, DK_alpha_n_plus_1;
      exponential_hardening_values(K_alpha_n, H_alpha_n, hardening_parameters.equivalent_plastic_strain);
      exponential_hardening_values(K_alpha_n_plus_1, H_alpha_n_plus_1, alpha_n_plus_1);
      exponential_hardening_derivatives(DK_alpha_n_plus_1, DH_alpha_n_plus_1, alpha_n_plus_1);
      if (update_material_point_history & constitutive_request.get_update_flags()) {
        material_point_history[point_index].hardening_parameters.equivalent_plastic_strain = alpha_n_plus_1;
        material_point_history[point_index].hardening_parameters.kinematic_hardening =
          hardening_parameters.kinematic_hardening
          + sqrt2thirds
          * (H_alpha_n_plus_1 - H_alpha_n)
          * stress_flow_direction;
        material_point_history[point_index].plastic_strain =
          plastic_strain + delta_gamma * stress_flow_direction;
      }

      SymmetricTensor<2, dim, Number>
      stress = kappa * trace(deformation_gradient) * unit_symmetric_tensor<dim, Number>()
               + dev_stress_trial
               - 2 * mu * delta_gamma * stress_flow_direction;

      Number theta_n_plus_1 = 1 - 2 * mu * delta_gamma / norm_ksi_trial;
      Number theta_bar_n_plus_1 = 1 / (1 + (DK_alpha_n_plus_1 + DH_alpha_n_plus_1) / (3 * mu))
                                  - (1 - theta_n_plus_1);
      const SymmetricTensor<4, dim, Number> one_prod_one =
        outer_product(unit_symmetric_tensor<dim, Number>(), unit_symmetric_tensor<dim, Number>());
      elastoplastic_tangent_moduli = kappa * one_prod_one
                                     + 2 * mu * theta_n_plus_1 * (identity_tensor<dim, Number>() - 1 / 3 * one_prod_one)
                                     - 2 * mu * theta_bar_n_plus_1 * outer_product(stress_flow_direction, stress_flow_direction);

      // TODO change code such that request update flags are respected
      constitutive_request.set_stress_deviator(stress);
//    constitutiveRequest.setStressDeviatorTangentModuli(elastoplastic_tangent_moduli);
    } /*if ( trial yield criterion test )*/
    else {
      elastoplastic_tangent_moduli = stress_strain_tensor_kappa + stress_strain_tensor_mu;
      SymmetricTensor<2, dim, Number> stress = (stress_strain_tensor_kappa + stress_strain_tensor_mu) * deformation_gradient;
      constitutive_request.set_stress_deviator(stress);
//    constitutiveRequest.setStressDeviatorTangentModuli(elastoplastic_tangent_moduli);
    }
  }

  template <int dim, typename Number>
  void
  ExponentialHardeningElastoplasticMaterial<dim, Number>::
  setup_point_history (const point_index_t point_count) {
    {
      std::vector< PointHistory<dim, Number> > tmp;
      tmp.swap (material_point_history);
    }
    material_point_history.resize (point_count);
  }

  template <int dim, typename Number>
  inline void
  ExponentialHardeningElastoplasticMaterial<dim, Number>::
  determine_delta_gamma(Number &delta_gamma, Number &alpha_n_plus_1,
                        const Number norm_ksi_trial,
                        const Number alpha_n,
                        Number tol, unsigned int max_iter) const {
    unsigned int k = 0;
    const Number sqrt2thirds = sqrt((Number)2 / (Number)3);
    Number g_of_gamma_k, Dg_of_gamma_k;
    Number K_alpha_n, K_alpha_n_plus_1, H_alpha_n, H_alpha_n_plus_1;
    Number DH_alpha_n_plus_1, DK_alpha_n_plus_1;

    delta_gamma = 0;
    alpha_n_plus_1 = alpha_n;

    exponential_hardening_values(K_alpha_n, H_alpha_n, alpha_n);

    do {
      k++;
      exponential_hardening_values(K_alpha_n_plus_1, H_alpha_n_plus_1, alpha_n_plus_1);
      g_of_gamma_k = -sqrt2thirds * K_alpha_n_plus_1 + norm_ksi_trial
                     - (2 * mu * delta_gamma + sqrt2thirds * (H_alpha_n_plus_1 - H_alpha_n));

      exponential_hardening_derivatives(DK_alpha_n_plus_1, DH_alpha_n_plus_1, alpha_n_plus_1);
      Dg_of_gamma_k = -2 * mu * (1 + (DH_alpha_n_plus_1 + DK_alpha_n_plus_1) / (3 * mu));

      delta_gamma = delta_gamma - g_of_gamma_k / Dg_of_gamma_k;
      alpha_n_plus_1 = alpha_n + sqrt2thirds * delta_gamma;

    } while (std::fabs(g_of_gamma_k) > tol && k < max_iter);
  }

  template <int dim, typename Number>
  inline void
  ExponentialHardeningElastoplasticMaterial<dim, Number>::
  exponential_hardening_values(Number &kinematic_hardening,
                               Number &isotropic_hardening,
                               const Number alpha) const {
    Number h = K_infty - (K_infty - K_0) * exp(-delta * alpha) + H_bar * alpha;
    kinematic_hardening = beta * h;
    isotropic_hardening = (1 - beta) * h;
  }

  template <int dim, typename Number>
  inline void
  ExponentialHardeningElastoplasticMaterial<dim, Number>::
  exponential_hardening_derivatives(Number &D_kinematic_hardening,
                                    Number &D_isotropic_hardening,
                                    const Number alpha) const {
    Number Dh = delta * (K_infty - K_0) * exp(-delta * alpha) + H_bar;
    D_kinematic_hardening = beta * Dh;
    D_isotropic_hardening = (1 - beta) * Dh;
  }

  template <int dim, typename Number>
  inline Number
  ExponentialHardeningElastoplasticMaterial<dim, Number>::
  trial_yield_criterion(const Number norm_ksi_trial,
                        const Number alpha) const {
    Number H_alpha, K_alpha;
    exponential_hardening_values(K_alpha, H_alpha, alpha);
    const Number sqrt2thirds = sqrt((Number)2 / (Number)3);
    Number trial_yield_criterion = norm_ksi_trial - sqrt2thirds * K_alpha;
    return trial_yield_criterion;
  }

  template class ExponentialHardeningElastoplasticMaterial<3, double>;
  template class ExponentialHardeningElastoplasticMaterial<2, double>;

} /* namespace PlasticityLab */
