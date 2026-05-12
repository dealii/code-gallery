/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2026 by Jonas Plank
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 */
#pragma once
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>

#include <iostream>
#include <vector>

namespace RandomField
{
    using namespace dealii;
    /* We assume here that our auto-covariance function is exponential, but not Gaussian.
    This is both modeling choice and mathematically reasonable in this case.
    We also assume a variance of 1 here.  */
    template <int dim>
    class KLExpansion
    {
        public:
        KLExpansion(const unsigned int n_terms, double L, double l, double mu); 

        double compute_kl_expansion(const Point<dim>& p, std::vector<double> &samples);
        
        private:

        // computes our coefficients

        // computes our eigenvalues based on the previously computed frequencies
        void compute_lambda_i(); 
        // computes norm values such that our eigenfunctions are normed to one.
        void compute_alpha_i();  
        // compute the frequencies
        void compute_omega_i(); 

        // functions we need for the computation of omega_i
        double f_even(double sol);     
        double f_odd(double sol);       
        double grad_f_even(double sol); 
        double grad_f_odd(double sol);  

        //newton since equation is nonlinear
        void newton_even(unsigned int index);  
        void newton_odd(unsigned int index);   

        //number of KL expansion terms
        unsigned int n_terms_;
        // domain length
        double domain_length_;
        // correlation length
        double correlation_length_;
        // important parameters for the construction of the KL-Expansion in x-direction
        std::vector<double> omega_i;
        std::vector<double> lambda_i;
        std::vector<double> alpha_i;

        // constant mean of our field.
        double mu_;
    };
}
