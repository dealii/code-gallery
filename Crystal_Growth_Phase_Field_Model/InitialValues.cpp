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

void InitialValues::vector_value(const Point<2> &p,
                                 Vector<double> & values) const
{
    values(0)= 0.0; //Initial p value of domain
    values(1)= 0.2; //Initial temperature of domain
}
