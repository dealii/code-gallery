/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2026 by Jonas Plank
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 */
#include "../include/random_permeability.h"

template <int dim>
RandomField::RandomPermeability<dim>::RandomPermeability(std::vector<double> &xi, unsigned int n_terms, double domain_length, double correlation_length, double mu)
: kl_expansion(n_terms, domain_length, correlation_length, mu)
, samples(xi)
{}

template <int dim>
void RandomField::RandomPermeability<dim>::overwrite_samples(const std::vector<double> &next_sample)
{
    samples = next_sample;
}

template <int dim>
double RandomField::RandomPermeability<dim>::value(const Point<dim>& p)
{
    return std::exp(kl_expansion.compute_kl_expansion(p, samples));
}

template class RandomField::RandomPermeability<1>;
template class RandomField::RandomPermeability<2>;

