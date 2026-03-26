/*
 * JohnsonCookThermoviscoplasticYieldLaw.h
 *
 *  Created on: 22 Nov 2019
 *      Author: maien
 */

#ifndef JOHNSONCOOKTHERMOVISCOPLASTICYIELDLAW_H_
#define JOHNSONCOOKTHERMOVISCOPLASTICYIELDLAW_H_

#include <math.h>

#include "Constants.h"


namespace PlasticityLab {

  template<typename Number>
  class JohnsonCookThermoviscoplasticYieldLaw {
  public:
    JohnsonCookThermoviscoplasticYieldLaw(
      const Number mu,
      const Number A,
      const Number B,
      const Number C,
      const Number m,
      const Number n,
      const Number melting_temperature,
      const Number reference_strain_rate=1.0,
      const Number reference_temperature=293.0,
      const Number beta=1.0) :
    mu(mu),
    A(A),
    B(B),
    C(C),
    m(m),
    n(n),
    melting_temperature(melting_temperature),
    reference_strain_rate(reference_strain_rate),
    reference_temperature(reference_temperature),
    beta(beta),
    sqrt2thirds(Constants<2, Number>::sqrt2thirds()),
    exp_one_half(std::exp(0.5)) {}

  Number hardening_values(Number &isotropic_hardening,
                               Number &kinematic_hardening,
                               const Number alpha,
                               const Number gamma,
                               const Number time_increment,
                               const Number temperature) const {
    if(use_Carreau_viscous_law) {
      const Number creep_strain_rate_factor = std::pow(1-std::max(0., std::min(1., (temperature - reference_temperature)/(melting_temperature - reference_temperature))), 1.5);
      const Number minimum_strain_rate = creep_strain_rate_factor * epsilon_dot_0;
      const Number strain_rate = std::max(sqrt2thirds*gamma/time_increment, minimum_strain_rate);
      const Number softened_quasistatic_elastoplastic_stress = get_elastoplastic_factor(alpha) * get_softening_factor(temperature);
      const Number Carreau_viscocity = get_Carreau_viscocity(strain_rate, softened_quasistatic_elastoplastic_stress);
      const Number h = 3 * Carreau_viscocity * strain_rate;
      isotropic_hardening = beta * h;
      kinematic_hardening = (1 - beta) * h;
      return h;
    } else {
      const Number h =
          get_elastoplastic_factor(alpha)
              * get_viscosity_factor(gamma, time_increment)
              * get_softening_factor(temperature)
          + viscosity_regularization_factor * sqrt2thirds * gamma/time_increment;
      isotropic_hardening = beta * h;
      kinematic_hardening = (1 - beta) * h;
      return h;
    }
  }

  Number hardening_alpha_derivatives(Number &D_isotropic_hardening,
                                          Number &D_kinematic_hardening,
                                          const Number alpha,
                                          const Number gamma,
                                          const Number time_increment,
                                          const Number temperature) const {
    if(use_Carreau_viscous_law) {
      const Number creep_strain_rate_factor = std::pow(1-std::max(0., std::min(1., (temperature - reference_temperature)/(melting_temperature - reference_temperature))), 1.5);
      const Number minimum_strain_rate = creep_strain_rate_factor * epsilon_dot_0;
      const Number strain_rate = std::max(sqrt2thirds*gamma/time_increment, minimum_strain_rate);
      const Number softened_quasistatic_elastoplastic_stress = get_elastoplastic_factor(alpha) * get_softening_factor(temperature);
      const Number Carreau_viscocity = get_Carreau_viscocity(strain_rate, softened_quasistatic_elastoplastic_stress);
      const Number strain_rate_tangent = strain_rate > minimum_strain_rate? 1.0/time_increment : 0;
      const Number stress_tangent =
        get_elastoplastic_factor_tangent(alpha) * get_softening_factor(temperature);
      Number Carreau_viscocity_strain_rate_tangent, Carreau_viscocity_stress_tangent;
      get_Carreau_viscocity_tangents(
          strain_rate, softened_quasistatic_elastoplastic_stress,
          Carreau_viscocity_strain_rate_tangent,
          Carreau_viscocity_stress_tangent);
      // const Number Dh =
      //   3 * Carreau_viscocity * strain_rate > softened_quasistatic_elastoplastic_stress?
      //     3 * Carreau_viscocity * strain_rate_tangent
      //     + 3 * Carreau_viscocity_stress_tangent * stress_tangent * strain_rate
      //     + 3 * Carreau_viscocity_strain_rate_tangent * strain_rate_tangent * strain_rate
      //   : 0;
      const Number Dh =
        3 * Carreau_viscocity * strain_rate_tangent
        + 3 * Carreau_viscocity_stress_tangent * stress_tangent * strain_rate
        + 3 * Carreau_viscocity_strain_rate_tangent * strain_rate_tangent * strain_rate;
      D_isotropic_hardening = beta * Dh;
      D_kinematic_hardening = (1.0 - beta) * Dh;
      return Dh;
    } else {
      const Number Dh =
          get_elastoplastic_factor_tangent(alpha)
              * get_viscosity_factor(gamma, time_increment)
              * get_softening_factor(temperature)
          + get_elastoplastic_factor(alpha)
              * get_viscosity_factor_tangent(gamma, time_increment)
              * get_softening_factor(temperature)
          + viscosity_regularization_factor/time_increment;
      D_isotropic_hardening = beta * Dh;
      D_kinematic_hardening = (1.0 - beta) * Dh;
      return Dh;
    }
  }

  Number hardening_temperature_derivatives(Number &D_isotropic_hardening,
                                                Number &D_kinematic_hardening,
                                                const Number alpha,
                                                const Number gamma,
                                                const Number time_increment,
                                                const Number temperature) const {
    if(use_Carreau_viscous_law) {
      const Number creep_strain_rate_factor = std::pow(1-std::max(0., std::min(1., (temperature - reference_temperature)/(melting_temperature - reference_temperature))), 1.5);
      const Number minimum_strain_rate = creep_strain_rate_factor * epsilon_dot_0;
      const Number strain_rate = std::max(sqrt2thirds*gamma/time_increment, minimum_strain_rate);
      const Number softened_quasistatic_elastoplastic_stress = get_elastoplastic_factor(alpha) * get_softening_factor(temperature);
      const Number Carreau_viscocity = get_Carreau_viscocity(strain_rate, softened_quasistatic_elastoplastic_stress);
      const Number stress_temperature_tangent =
              get_elastoplastic_factor(alpha) * get_softening_factor_tangent(temperature);
      Number Carreau_viscocity_strain_rate_tangent, Carreau_viscocity_stress_tangent;
      get_Carreau_viscocity_tangents(
          strain_rate, softened_quasistatic_elastoplastic_stress,
          Carreau_viscocity_strain_rate_tangent,
          Carreau_viscocity_stress_tangent);
      const Number Dh = 3 * Carreau_viscocity_stress_tangent * stress_temperature_tangent * strain_rate;
      D_isotropic_hardening = beta * Dh;
      D_kinematic_hardening = (1 - beta) * Dh;
      return Dh;
    } else {
      Number Dh =
          get_elastoplastic_factor(alpha)
              * get_viscosity_factor(gamma, time_increment)
              * get_softening_factor_tangent(temperature);
      D_isotropic_hardening = beta * Dh;
      D_kinematic_hardening = (1 - beta) * Dh;
      return Dh;
    }
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

  Number get_elastoplastic_factor(const Number alpha) const {
    if(alpha > max_strain) {
      return A + B * std::pow(max_strain, n);
    }
    if(alpha >= eps) {
        return A + B * std::pow(alpha, n);
    }
    return A + B * alpha/eps * std::pow(eps, n);
  }

  Number get_viscosity_factor(const Number gamma, const Number time_increment) const {
    Number slope, intercept;
    get_small_hardening_fit(slope, intercept, time_increment);
    if(gamma >= intercept) {
        return 1.0 + C * std::log(sqrt2thirds*gamma/(time_increment*reference_strain_rate));
    } else if (gamma < 0.0) {
        return 1.0 - C * slope * sqrt2thirds * gamma * gamma;
    }
    return 1.0 + C * slope * sqrt2thirds * gamma * gamma;
  }

  Number get_softening_factor(const Number temperature) const {
    if(temperature > reference_temperature) {
      if(temperature < melting_temperature) {
        return (1.0 + softening_threshold - std::pow((temperature - reference_temperature)/(melting_temperature - reference_temperature), m));
      } else {
        return 0.0 + softening_threshold;
      }
    }
    return 1.0 + softening_threshold;
  }

  Number get_elastoplastic_factor_tangent(const Number alpha) const {
    if(alpha > max_strain) {
      return 0;
    }
    if(alpha >= eps) {
        return B * n * std::pow(alpha, n-1.0);
    }
    return B * 1.0/eps * std::pow(eps, n);
  }

  Number get_viscosity_factor_tangent(const Number gamma, const Number time_increment) const {
    Number slope, intercept;
    get_small_hardening_fit(slope, intercept, time_increment);
    if(gamma >= intercept) {
        return C / (sqrt2thirds * gamma);
    } else if (gamma < 0.0) {
        return -2 * C * slope * gamma;
    }
    return 2 * C * slope * gamma;
  }

  Number get_softening_factor_tangent(const Number temperature) const {
    if(temperature > reference_temperature) {
      if(temperature < melting_temperature) {
        return (-m/(melting_temperature - reference_temperature))
                * std::pow((temperature - reference_temperature)/(melting_temperature - reference_temperature), m-1.0);
      } else {
        return 0.0;
      }
    }
    return 0.0;
  }

  void get_small_hardening_fit(Number &slope, Number &intercept, const Number time_increment) const {
    // the log factor is annoying when below 1.0. Replace it by a parabula till it behaves.
    const Number log_factor = 1./(time_increment*reference_strain_rate);
    intercept = exp_one_half/(sqrt2thirds * log_factor);
    slope = 1./(2*intercept*intercept*sqrt2thirds);
  }

  Number get_Carreau_viscocity(Number strain_rate, Number sigma_0_theta) const {
    if(sigma_0_theta <= 0) {
      return mu_infty;
    }
    const Number g_sigma_epsilon_dot = std::pow(sigma_0_theta/(3*epsilon_dot_0*mu_0), (n_C/(1-n_C))) * (strain_rate/epsilon_dot_0);
    return std::pow(1 + std::pow(g_sigma_epsilon_dot, 2), ((1-n_C)/(2*n_C))) * (mu_0 - mu_infty) + mu_infty;
  }

  void get_Carreau_viscocity_tangents(
          Number strain_rate,
          Number sigma_0_theta,
          Number &Carreau_viscocity_strain_rate_tangent,
          Number &Carreau_viscocity_stress_tangent) const {
    if(sigma_0_theta <= 0) {
      Carreau_viscocity_strain_rate_tangent = 0;
      Carreau_viscocity_stress_tangent = 0;
      return;
    }
    const Number g_sigma_epsilon_dot = std::pow(sigma_0_theta/(3*epsilon_dot_0*mu_0), (n_C/(1-n_C))) * (strain_rate/epsilon_dot_0);
    const Number g_sigma_epsilon_dot_strain_rate_tangent = std::pow(sigma_0_theta/(3*epsilon_dot_0*mu_0), (n_C/(1-n_C))) * (1/epsilon_dot_0);
    const Number g_sigma_epsilon_dot_stress_tangent =
      (n_C / (1-n_C)) * std::pow(sigma_0_theta/(3*epsilon_dot_0*mu_0), ((2*n_C-1)/(1-n_C))) * (strain_rate/epsilon_dot_0) * (1/(3 * epsilon_dot_0 * mu_0));
    const Number Carreau_viscocity_g_tangent =
      ((1-n_C)/(2*n_C)) * std::pow(1 + std::pow(g_sigma_epsilon_dot, 2), ((1-3*n_C)/(2*n_C))) * (2*g_sigma_epsilon_dot) * (mu_0 - mu_infty);

    Carreau_viscocity_strain_rate_tangent = Carreau_viscocity_g_tangent * g_sigma_epsilon_dot_strain_rate_tangent;
    Carreau_viscocity_stress_tangent = Carreau_viscocity_g_tangent * g_sigma_epsilon_dot_stress_tangent;
  }

  private:
    const Number mu;
    const Number A;
    const Number B;
    const Number C;
    const Number m;
    const Number n;
    const Number melting_temperature;
    const Number reference_strain_rate;
    const Number reference_temperature;
    const Number beta;
    const Number sqrt2thirds;
    const Number exp_one_half;

    const Number eps = std::pow(B/mu, 1.0/(1.0-n));
    const Number softening_threshold = 0.0;
    const Number viscosity_regularization_factor = 0;
    const Number max_strain = std::numeric_limits<Number>::max();

    // Carreau fluid parameters
    const bool use_Carreau_viscous_law = false;
    const Number epsilon_dot_0 = 1;
    const Number n_C = 3;
    const Number mu_0 = 1e18;
    const Number mu_infty = 1e-4;

  };

} /* namespace PlasticityLab */

#endif /* JOHNSONCOOKTHERMOVISCOPLASTICYIELDLAW_H_ */