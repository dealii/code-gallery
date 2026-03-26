/*
 * Constants.h
 *
 *  Created on: 10 Feb 2015
 *      Author: maien
 */

#ifndef CONSTANTS_H_
#define CONSTANTS_H_

namespace PlasticityLab {


  template <int dim, typename Number>
  class Constants {
   public:
    inline static const Number one_third() {
      return static_cast<Number>(0.33333333333333333333333333333333333333333333333333);
    }

    inline static const Number sqrt2thirds() {
      return static_cast<Number>(0.81649658092772603273242802490196379732198249355222);
    }

    inline static const Number two_thirds() {
      return static_cast<Number>(0.66666666666666666666666666666666666666666666666666);
    }

    inline static const Number sqrt_half() {
      return static_cast<Number>(0.70710678118654752440084436210484903928483593768847);
    }

    inline static const Number sqrt_2() {
      return static_cast<Number>(1.41421356237309504880168872420969807856967187537694);
    }


    inline static const Number one_over_dim() {
      if(3==dim)
        return static_cast<Number>(0.33333333333333333333333333333333333333333333333333);
      else if (2==dim)
        return static_cast<Number>(0.5);
      else
        return static_cast<Number>(1./static_cast<Number>(dim));
    }

    inline static const Number two_over_dim() {
      if(3==dim)
        return static_cast<Number>(0.66666666666666666666666666666666666666666666666666);
      else if (2==dim)
        return static_cast<Number>(1.0);
      else
        return static_cast<Number>(2./static_cast<Number>(dim));
    }

  };

  inline void get_generalized_alpha_method_params(
        double *alpha_m,
        double *alpha_f,
        double *gamma,
        double *beta,
        double rho_infty
      ) {
    *alpha_m = (2. * rho_infty - 1.)/(rho_infty + 1.);
    *alpha_f = rho_infty / (rho_infty + 1.);
    *gamma = 0.5 - *alpha_m + *alpha_f;
    *beta = 0.25 * (1. - *alpha_m + *alpha_f) * (1. - *alpha_m + *alpha_f);
  }

} /* namespace PlasticityLab */

#endif /* CONSTANTS_H_ */
