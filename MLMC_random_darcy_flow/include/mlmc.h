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
#include <vector>
#include <random>

namespace MultilevelMonteCarlo
{
    template <int dim>
    class MLMC 
    {
        public:
        MLMC(unsigned int oneD_samples);                                   

        // generates the required number of samples
        std::vector<double> generate_samples();                            

        // store the samples for the computation of mean and variance
        void add_sample(double rvalue);                                     

        double compute_mean();

        double compute_variance();
        //removes all samples after reached convergence on a level
        void clear_samples();

        private:
        std::mt19937 rng;                       // Mersenne Twister engine
        std::normal_distribution<double> dist;  // Normal distribution
        unsigned int num_samples;

        // parameters for error computation
        std::vector<double> results;
    };
}

