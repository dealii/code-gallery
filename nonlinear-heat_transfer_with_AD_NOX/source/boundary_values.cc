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
 * Returns the boundary value for a given #time. Here, we simply return a constant value
 * 100 at the left end of the domain.
 * @param p A point (2D)
 * @return Value at the boundary
 */
double Boundary_values_left::value(const Point<2> & /*p*/, const unsigned int /*comp*/) const
{
    return 100;

    /**
     * To linearly ramp the temperature at the left end to 100 over the
     * entire time span, use the below line. See step-23 as to how the time
     * variable can be used.
     */
    // nonlinear_heat nlheat;
    // double total_time = nlheat.tot_time;
    // return this->get_time() * 100.0/total_time;
}
