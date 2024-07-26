/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
 * Copyright (C) 2024 by Narasimhan Swaminathan
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 */

#include "nonlinear_heat.h"

/**
 * Returns the initial conditions.
 * @param p Point
 * @param comp component
 * @return
 */
double Initialcondition::value(const Point<2> & /*p*/, const unsigned int /*comp*/) const
{
    /**
     * In the current case, we assume that the initial conditions are zero everywhere.
     */
    return 0.0;
}
