/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2026 by Jonas Plank
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 */

#include "../include/mlmc.h"
#include <cmath>

template <int dim>
MultilevelMonteCarlo::MLMC<dim>::MLMC(unsigned int oneD_samples)
: rng(std::random_device{}())   
, dist(0.0, 1.0)                 // mean = 0, stddev = 1
{
    if constexpr (dim == 1) num_samples = oneD_samples;
    else if constexpr (dim == 2) num_samples = oneD_samples*oneD_samples;
}

template <int dim>
std::vector<double> MultilevelMonteCarlo::MLMC<dim>::generate_samples()
{
    std::vector<double> samples(num_samples);

    for (auto &s : samples)
    {
        s = dist(rng);
    }
    return samples;
}

template<int dim>
void MultilevelMonteCarlo::MLMC<dim>::add_sample(double rvalue)
{
    results.push_back(rvalue);
}

template <int dim>
double MultilevelMonteCarlo::MLMC<dim>::compute_mean()
{
    double mean = 0.0;
    for(unsigned int i = 0; i<results.size(); i++)
    {
        mean+=results[i];
    }
    return mean/results.size();
}

template <int dim>
double MultilevelMonteCarlo::MLMC<dim>::compute_variance()
{
    double mean = compute_mean();
    double var = 0.0;
    for(unsigned int i = 0; i<results.size(); i++)
    {
        var += std::pow((results[i]-mean),2);
    }

    return var / (results.size() - 1);
}

template <int dim>
void MultilevelMonteCarlo::MLMC<dim>::clear_samples()
{
    results.clear();
}

template class MultilevelMonteCarlo::MLMC<1>;
template class MultilevelMonteCarlo::MLMC<2>;

