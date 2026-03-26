/*
 * TensorUtilities.h
 *
 *  Created on: 06 Nov 2020
 *      Author: maien
 */

#ifndef TENSOR_UTILITIES_H_
#define TENSOR_UTILITIES_H_

#include<exception>

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>


using namespace dealii;

namespace PlasticityLab {
  template <int dim, typename Number>
  SymmetricTensor<2, dim, Number> get_log_of_tensor(
      const SymmetricTensor<2, dim, Number> &symmetric_stretch_rate) {
    try{
      const auto eig_vals_vecs = eigenvectors(symmetric_stretch_rate);
      SymmetricTensor<2, dim, Number> result =
        std::log(eig_vals_vecs[0].first)
        * symmetrize(
            outer_product(
              eig_vals_vecs[0].second,
              eig_vals_vecs[0].second));
      for(unsigned int d=1; d<dim; ++d) {
        result +=
          std::log(eig_vals_vecs[d].first)
          * symmetrize(
              outer_product(
                eig_vals_vecs[d].second,
                eig_vals_vecs[d].second));
      }
      return result;
    } catch(std::exception& e) {
      std::cout << "Could not get log of tensor: " << symmetric_stretch_rate << std::endl;
      return symmetric_stretch_rate;
    }

  }

  template <int dim, typename Number>
  SymmetricTensor<2, dim, Number> get_log_of_tensor_variation(
        const SymmetricTensor<2, dim, Number> &symmetric_stretch_rate,
        const SymmetricTensor<2, dim, Number> &symmetric_stretch_rate_variation) {
    const double epsilon = 1e-10;
    return (1./epsilon) * (get_log_of_tensor(symmetric_stretch_rate + epsilon * symmetric_stretch_rate_variation) - get_log_of_tensor(symmetric_stretch_rate));
  }

  template <int dim, typename Number>
  SymmetricTensor<2, dim, Number> get_exp_of_tensor(
      const SymmetricTensor<2, dim, Number> &symmetric_stretch_rate) {
    try{
      const auto eig_vals_vecs = eigenvectors(symmetric_stretch_rate);
      SymmetricTensor<2, dim, Number> result =
        std::exp(eig_vals_vecs[0].first)
        * symmetrize(
            outer_product(
              eig_vals_vecs[0].second,
              eig_vals_vecs[0].second));
      for(unsigned int d=1; d<dim; ++d) {
        result +=
          std::exp(eig_vals_vecs[d].first)
          * symmetrize(
              outer_product(
                eig_vals_vecs[d].second,
                eig_vals_vecs[d].second));
      }
      return result;
    } catch(std::exception& e) {
      std::cout << "Could not get exp of tensor: " << symmetric_stretch_rate << std::endl;
      return symmetric_stretch_rate;
    }
  }

  template <int dim, typename Number>
  SymmetricTensor<2, dim, Number> get_exp_of_tensor_variation(
        const SymmetricTensor<2, dim, Number> &symmetric_stretch_rate,
        const SymmetricTensor<2, dim, Number> &symmetric_stretch_rate_variation) {
    const Number epsilon = 1e-10;
    return (1./epsilon) * (get_exp_of_tensor(symmetric_stretch_rate + epsilon * symmetric_stretch_rate_variation) - get_exp_of_tensor(symmetric_stretch_rate));
  }
}

#endif  // TENSOR_UTILITIES_H_