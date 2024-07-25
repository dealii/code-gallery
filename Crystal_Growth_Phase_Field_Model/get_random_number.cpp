/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
 * Copyright (C) 2024 by Umair Hussain
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 */

#include "PhaseFieldSolver.h"
#include <random>

float PhaseFieldSolver::get_random_number()
{
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(-0.5, 0.5); // returns a random number in the range of -0.5 to 0.5
    return dis(e);
}
