/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
 * Copyright (C) 2024 by Umair Hussain
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 */

#include <iostream>
#include "PhaseFieldSolver.h"

int main(int argc, char **argv) {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    PhaseFieldSolver phasefieldsolver;
    phasefieldsolver.run();
    return 0;
}
