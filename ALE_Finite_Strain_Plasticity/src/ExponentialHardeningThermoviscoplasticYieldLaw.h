/*
 * ExponentialHardeningThermoviscoplasticYieldLaw.h
 *
 *  Created on: 22 Nov 2019
 *      Author: maien
 */

#ifndef EXPONENTIALHARDENINGTHERMOVISCOPLASTICYIELDLAW_H_
#define EXPONENTIALHARDENINGTHERMOVISCOPLASTICYIELDLAW_H_

#include "Constants.h"

namespace PlasticityLab {

  template<typename Number>
  class ExponentialHardeningThermoviscoplasticYieldLaw {
  public:
    ExponentialHardeningThermoviscoplasticYieldLaw(
      const Number K_0,
      const Number K_infty,
      const Number delta,
      const Number H_bar,
      const Number beta,
      const Number flow_stress_softening,
      const Number hardening_softening,
      const Number reference_temperature=293.0) :
    K_0(K_0),
    K_infty(K_infty),
    delta(delta),
    H_bar(H_bar),
    beta(beta),
    flow_stress_softening(flow_stress_softening),
    hardening_softening(hardening_softening),
    reference_temperature(reference_temperature),
    viscous_hardening_factor(0.0),
    sqrt2thirds(Constants<3, Number>::sqrt2thirds()) {}

  Number hardening_values(Number &isotropic_hardening,
                               Number &kinematic_hardening,
                               const Number alpha,
                               const Number gamma,
                               const Number time_increment,
                               const Number temperature) const {
    Number h = K_0 * (1 - std::min(softening_threshold, flow_stress_softening * (temperature - reference_temperature)))
               + ((K_infty - K_0) * (1 - exp(-delta * alpha)) + H_bar * alpha) * (1 - std::min(softening_threshold, hardening_softening * (temperature - reference_temperature)))
               + viscous_hardening_factor * sqrt2thirds * gamma / time_increment;
    isotropic_hardening = beta * h;
    kinematic_hardening = (1 - beta) * h;
    return h;
  }

  Number hardening_alpha_derivatives(Number &D_isotropic_hardening,
                                          Number &D_kinematic_hardening,
                                          const Number alpha,
                                          const Number gamma,
                                          const Number time_increment,
                                          const Number temperature) const {
    Number Dh = (delta * (K_infty - K_0) * exp(-delta * alpha) + H_bar) * (1 - std::min(softening_threshold, hardening_softening * (temperature - reference_temperature)))
                + viscous_hardening_factor / time_increment;
    D_isotropic_hardening = beta * Dh;
    D_kinematic_hardening = (1 - beta) * Dh;
    return Dh;
  }

  Number hardening_temperature_derivatives(Number &D_isotropic_hardening,
                                                Number &D_kinematic_hardening,
                                                const Number alpha,
                                                const Number gamma,
                                                const Number time_increment,
                                                const Number temperature) const {
    Number Dh = (hardening_softening * (temperature - reference_temperature) < softening_threshold)?
                -flow_stress_softening * K_0
                - hardening_softening * ((K_infty - K_0) * (1 - exp(-delta * alpha)) + H_bar * alpha)
                : 0.0;
    D_isotropic_hardening = beta * Dh;
    D_kinematic_hardening = (1 - beta) * Dh;
    return Dh;
  }

  Number trial_yield_criterion(const Number norm_ksi_trial,
                        const Number alpha,
                        const Number gamma,
                        const Number time_increment,
                        const Number temperature) const {
    Number K_alpha, H_alpha;
    hardening_values(H_alpha, K_alpha, alpha, gamma, time_increment, temperature);
    const Number trial_yield_criterion = norm_ksi_trial - sqrt2thirds * (H_alpha);
    return trial_yield_criterion;
  }
  private:
    const Number K_0, K_infty, delta, H_bar; // hardening parameters (Simo & Hughes pp185)
    const Number beta; // isotropic/kinematic hardening parameter (1.0 for isotropic)
    const Number flow_stress_softening, hardening_softening; // thermal softening parameters (Simo & Miehe 1992 pp74)
    const Number reference_temperature;
    const Number viscous_hardening_factor;
    const Number sqrt2thirds;
    const Number softening_threshold = 0.98;
  };

} /* namespace PlasticityLab */

#endif /* EXPONENTIALHARDENINGTHERMOVISCOPLASTICYIELDLAW_H_ */