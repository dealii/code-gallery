/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2026 by Jonas Plank
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 */
#include "../include/KL_expansion.h"
#include <cmath>
#include <math.h>

template <int dim>
RandomField::KLExpansion<dim>::KLExpansion(const unsigned int n_terms, double domain_length, double correlation_length, double mu)
: n_terms_(n_terms)
, domain_length_(domain_length)
, correlation_length_(correlation_length)
, mu_(mu)
{
    compute_omega_i();
    compute_alpha_i();
    compute_lambda_i();
}

template <int dim>
double RandomField::KLExpansion<dim>::compute_kl_expansion(const Point<dim>& p, std::vector<double> &samples)
{
    double val = mu_;
    if constexpr (dim == 1)
    {
        for(unsigned int i = 0; i<n_terms_; i++)
        {
            // even
            if((i+1)%2 == 0)
            {
                val += std::sqrt(lambda_i[i])*samples[i]*alpha_i[i]*std::sin(omega_i[i]*(p[0]-domain_length_/2));
            }
            //odd
            else 
            {
                val += std::sqrt(lambda_i[i])*samples[i]*alpha_i[i]*std::cos(omega_i[i]*(p[0]-domain_length_/2));
            }
        }
    }
    if constexpr (dim == 2)
    {
        for(unsigned int i = 0; i<n_terms_; i++)
        {
            for(unsigned int j = 0; j<n_terms_; j++)
            {
                //both even
                if((i+1)%2==0 && (j+1)%2==0)
                {
                    val+=std::sqrt(lambda_i[i]*lambda_i[j])*samples[i*n_terms_+j]*alpha_i[i]*alpha_i[j]*std::sin(omega_i[i]*(p[0]-domain_length_/2))*std::sin(omega_i[j]*(p[1]-domain_length_/2));
                }
                //both odd
                else if((i+1)%2==1 && (j+1)%2==1)
                {
                    val+=std::sqrt(lambda_i[i]*lambda_i[j])*samples[i*n_terms_+j]*alpha_i[i]*alpha_i[j]*std::cos(omega_i[i]*(p[0]-domain_length_/2))*std::cos(omega_i[j]*(p[1]-domain_length_/2));
                }
                //i odd
                else if((i+1)%2==1 && (j+1)%2==0)
                {
                    val+=std::sqrt(lambda_i[i]*lambda_i[j])*samples[i*n_terms_+j]*alpha_i[i]*alpha_i[j]*std::cos(omega_i[i]*(p[0]-domain_length_/2))*std::sin(omega_i[j]*(p[1]-domain_length_/2));
                }
                //j odd
                else if((i+1)%2==0 && (j+1)%2==1)
                {
                    val+=std::sqrt(lambda_i[i]*lambda_i[j])*samples[i*n_terms_+j]*alpha_i[i]*alpha_i[j]*std::sin(omega_i[i]*(p[0]-domain_length_/2))*std::cos(omega_i[j]*(p[1]-domain_length_/2));
                }

            }
        }
    }

    return val;
}

template <int dim>
void RandomField::KLExpansion<dim>::compute_omega_i()
{
    for(unsigned int i = 1; i<=n_terms_; i++)
    {
        if(i%2 == 0)
        {
            newton_even(i);
        }
        else
        {
            newton_odd(i);
        }
    }
}

template <int dim>
void RandomField::KLExpansion<dim>::compute_alpha_i()
{
    for(unsigned int i = 1; i<=n_terms_; i++)
    {
        if(i%2 == 0)
        {
            double w_i = omega_i[i-1];
            double sqrt_val = domain_length_/2 -std::sin(w_i*domain_length_)/(2*w_i);
            alpha_i.push_back(1/(std::sqrt(sqrt_val)));
        }
        else
        {
            double w_i = omega_i[i-1];
            double sqrt_val = domain_length_/2 +std::sin(w_i*domain_length_)/(2*w_i);
            alpha_i.push_back(1/(std::sqrt(sqrt_val)));
        }
    }
}

template <int dim>
void RandomField::KLExpansion<dim>::compute_lambda_i()
{
    for(unsigned int i = 1; i<=n_terms_; i++)
    {
        double w_i = omega_i[i-1];
        double lambda = 2*correlation_length_/(1+w_i*w_i*correlation_length_*correlation_length_);
        lambda_i.push_back(lambda);
    }

}

template <int dim>
void RandomField::KLExpansion<dim>::newton_even(unsigned int index)
{
    double a = M_PI / domain_length_ * (index - 1);
    double b = M_PI / domain_length_ * index;

    double x = a + 0.1 * (b - a);  

    for (int i = 0; i < 50; ++i)
    {
        double fx  = f_even(x);
        double dfx = grad_f_even(x);

        if (std::abs(dfx) < 1e-14) break; 

        double step = fx / dfx;
        double x_new = x - step;

        int backtrack_count = 0;
        while ((x_new <= a || x_new >= b) && backtrack_count < 10) 
        {
            step *= 0.5;
            x_new = x - step;
            backtrack_count++;
        }

        if (x_new <= a || x_new >= b) break;


        if (!std::isfinite(x_new)) break;
        if (std::abs(x_new - x) < 1e-12) 
        {
            x = x_new; 
            break;
        }

        x = x_new;
    }
    omega_i.push_back(x);
}

template <int dim>
void RandomField::KLExpansion<dim>::newton_odd(unsigned int index)
{
    double x = M_PI / domain_length_ * (static_cast<double>(index) - 0.2);

    for (int i = 0; i < 50; ++i)
    {
        double fx  = f_odd(x);
        double dfx = grad_f_odd(x);

        if (std::abs(dfx) < 1e-12) break;

        double x_new = x - fx / dfx;

        if (std::abs(x_new - x) < 1e-10) break;
        if (!std::isfinite(x_new)) break;

        x = x_new;
    }
    omega_i.push_back(x);
}


template <int dim>
double RandomField::KLExpansion<dim>::f_odd(double x)
{
    return 1.0 / correlation_length_ - x * std::tan(x * domain_length_ / 2.0);
}

template <int dim>
double RandomField::KLExpansion<dim>::f_even(double x)
{
    return (1.0 / correlation_length_) * std::tan(x * domain_length_ / 2.0) + x;
}

template <int dim>
double RandomField::KLExpansion<dim>::grad_f_odd(double x)
{
    const double L = domain_length_;
    const double t = std::tan(x * L / 2.0);
    const double sec2 = 1.0 / std::cos(x * L / 2.0);
    return -t - x * (L / 2.0) * sec2 * sec2;
}

template <int dim>
double RandomField::KLExpansion<dim>::grad_f_even(double x)
{
    const double L = domain_length_;
    const double sec2 = 1.0 / std::cos(x * L / 2.0);
    return (1.0 / correlation_length_) * (L / 2.0) * sec2 * sec2 + 1.0;
}

template class RandomField::KLExpansion<1>;
template class RandomField::KLExpansion<2>;


