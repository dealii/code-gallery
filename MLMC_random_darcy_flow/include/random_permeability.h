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
#include "KL_expansion.h"

namespace RandomField
{
    using namespace dealii;

    /*
    This class is really our interface between the KL expansion 
    and the FEM code. */
    template <int dim>
    class RandomPermeability : public Function<dim>
    {
        public:
        RandomPermeability(std::vector<double> &first_sample, unsigned int n_terms, double domain_length, double correlation_length, double mu);  

        // overwrite the samples
        void overwrite_samples(const std::vector<double> &next_sample);  

        // our evaluator 
        double value(const Point<dim>& p);  

        private:
        KLExpansion<dim> kl_expansion;
        std::vector<double> samples;
    };
}

