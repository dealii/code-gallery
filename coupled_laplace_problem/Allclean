## -----------------------------------------------------------------------------
##
## SPDX-License-Identifier: LGPL-2.1-or-later
## Copyright (C) 2020 by David Schneider
## Copyright (C) 2020 by Benjamin Uekermann
##
## This file is part of the deal.II code gallery.
##
## -----------------------------------------------------------------------------

#!/bin/sh
cd ${0%/*} || exit 1    # Run from this directory

echo "Cleaning..."

# Participant 1: coupled_laplace_problem
Participant1="coupled_laplace_problem"

# Participant 2: fancy_boundary_condition
Participant2="fancy_boundary_condition"

# Remove vtk result files
rm -fv solution-*.vtk

# Remove the preCICE-related log files
echo "Deleting the preCICE log files..."
rm -fv \
    precice-*.log \
    precice-*-events.json
    
rm -rfv precice-run
rm -fv .${Participant1}-${Participant2}.address

echo "Cleaning complete!"
#------------------------------------------------------------------------------
