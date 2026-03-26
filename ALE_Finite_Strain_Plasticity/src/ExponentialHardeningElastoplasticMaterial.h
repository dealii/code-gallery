/*
 * ExponentialHardeningElastoplasticMaterial.h
 *
 *  Created on: 10 Jul 2014
 *      Author: cerecam
 */

#ifndef EXPONENTIALHARDENINGMATERIAL_H_
#define EXPONENTIALHARDENINGMATERIAL_H_

#include "PointHistory.h"
#include "Material.h"
#include "ConstitutiveModelRequest.h"

using namespace dealii;

namespace PlasticityLab {

  template <int dim, typename Number = double>
  class ExponentialHardeningElastoplasticMaterial : public Material<dim, Number> {
   public:
    ExponentialHardeningElastoplasticMaterial(const Number E,
                                              const Number nu,
                                              const Number K_0,
                                              const Number K_infty,
                                              const Number delta,
                                              const Number H_bar,
                                              const Number beta);

    virtual ~ExponentialHardeningElastoplasticMaterial();

    void compute_constitutive_request(
      ConstitutiveModelRequest<dim, Number> &constitutive_request,
      const point_index_t &point_index);

    Number get_material_Jacobian(const point_index_t &point_index) const;
    dealii::SymmetricTensor<2, dim, Number> get_plastic_strain(const point_index_t &point_index) const;

    void setup_point_history (const point_index_t point_count);

    std::vector<Number> get_state_parameters(
              const point_index_t &point_index,
              const Tensor<2, dim, Number> &reference_transformation=unit_symmetric_tensor<dim>()) const;

    void  set_state_parameters(
              const point_index_t &point_index,
              const std::vector<Number> &state_parameters,
              const Tensor<2, dim, Number> &reference_transformation);
    size_t get_material_parameter_count() const;

   private:
    const Number kappa;
    const Number mu;

    const Number K_0, K_infty, delta, H_bar; // hardening parameters (Simo & Hughes pp185)
    const Number beta;               // isotropic/kinematic hardening parameter


    const SymmetricTensor<4, dim, Number> stress_strain_tensor_kappa;
    const SymmetricTensor<4, dim, Number> stress_strain_tensor_mu;

    std::vector< PointHistory< dim, Number> > material_point_history;

    inline void
    determine_delta_gamma(Number &delta_gamma, Number &alpha_n_plus_1,
                          const Number norm_ksi_trial,
                          const Number alpha_n,
                          Number tol, unsigned int max_iter) const;

    inline void
    exponential_hardening_values(Number &kinematic_hardening,
                                 Number &isotropic_hardening,
                                 const Number alpha) const;

    inline void
    exponential_hardening_derivatives(Number &D_kinematic_hardening,
                                      Number &D_isotropic_hardening,
                                      const Number alpha) const;

    inline Number
    trial_yield_criterion(const Number norm_ksi_trial,
                          const Number alpha) const;

  };

} /* namespace PlasticityLab */

#endif /* EXPONENTIALHARDENINGMATERIAL_H_ */
