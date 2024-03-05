## -----------------------------------------------------------------------------
##
## SPDX-License-Identifier: LGPL-2.1-or-later
## Copyright (C) 2020 by Ester Comellas
## Copyright (C) 2020 by Jean-Paul Pelteret
##
## This file is part of the deal.II code gallery.
##
## -----------------------------------------------------------------------------

#GNUPLOT file to plot graphs of biomechanical brain tests in Budday et al 2017
# **** cycles must be separated in the .sol file by two blank lines ****
# UNITS CONSIDERED: g - mm - s

#To see color names:
# >> show palette colornames

#To see line style, line width, etc.:
# >> set terminal wxt enhanced dashed
# >> test

clear
set datafile separator ","
set term png size 400,300 enhanced font 'Verdana,12'


unset colorbox
set style line 1  lw 1.5  lc rgb "black"   dt 1
set style line 2  lw 1.5  lc rgb "blue"    dt 1
set style line 3  lw 1.5  lc rgb "red"     dt 1
set style line 4  lw 1.5  lc rgb "dark-turquoise"  dt 1
set style line 5  lw 1.5  lc rgb "dark-salmon"     dt 1

set style line 100 lt 1  lw 1  lc rgb "gray" dt 3
set style line 101 lt 1  lw 1  lc rgb "red"

set notitle
set grid xtics ytics ls 100

# -------------------- PROBLEM DATA --------------------------------------------------------------------------------------
#length of specimen along loading direction to compute stretch [mm]
length = 5.0

#area of loading surface to computre reaction force [mm²]
area = length * length                    #cube
#area = 3.1416 * (length/2) * (length/2)  #cylinder

#length to compute pressure gradient [mm]
radius = length/2

#pressure at drained boundary to compute pressure gradient [Pa]
atm_pressure = 0.0


#dynamic (shear) fluid viscosity [Pa·s]
mu = 0.89e-3
#effective fluid density [g/mm³]
rho = 0.997e-3
#kinematic fluid viscosity [mm²/s]
nu = mu/rho

#time step to compute numerical dissipation [s]
timestep = 1.0


# -------------------- Stress vs stretch ---------------------------------------------------------------------------------
stretch(x) = (length + x) / length
nominal(y) =  y / (area*1000)           #[10⁻⁶ N / mm² = Pa]

set xlabel 'stretch [-]' font 'Verdana,14'
set ylabel 'nominal stress [kPa]' font 'Verdana,14'
unset y2label
set key on box inside top left font 'Verdana,12' spacing 1.0 maxcols 1 width -0.2

#set xrange [] reverse                    #compression
set xrange [] noreverse          #tension
set autoscale xy

unset y2tics
set xtics autofreq rotate by 45 right
set ytics autofreq

set output "brain_cube_stress-stretch-poro-SHEAR.png"
plot 'data-for-gnuplot.sol' index 0 using (stretch($5)):(nominal($13)) every 1  with lines ls 1 title '1^{st} cycle', \
     'data-for-gnuplot.sol' index 1 using (stretch($5)):(nominal($13)) every 1  with lines ls 2 title '2^{nd} cycle', \
     'data-for-gnuplot.sol' index 2 using (stretch($5)):(nominal($13)) every 1  with lines ls 3 title '3^{rd} cycle'

# -------------------- solid Stress vs stretch ---------------------------------------------------------------------------------
stretch(x) = (length + x) / length
nominal(y) =  y / (area*1000)           #[10⁻⁶ N / mm² = Pa]

set xlabel 'stretch [-]' font 'Verdana,14'
set ylabel 'solid nominal stress [kPa]' font 'Verdana,14'
unset y2label
set key on box inside top left font 'Verdana,12' spacing 1.0 maxcols 1 width -0.2

#set xrange [] reverse                    #compression
set xrange [] noreverse          #tension
set autoscale xy

unset y2tics
set xtics autofreq rotate by 45 right
set ytics autofreq

set output "brain_cube_solid-stress-stretch-poro-SHEAR.png"
plot 'data-for-gnuplot.sol' index 0 using (stretch($5)):(nominal($19)) every 1  with lines ls 1 title '1^{st} cycle', \
     'data-for-gnuplot.sol' index 1 using (stretch($5)):(nominal($19)) every 1  with lines ls 2 title '2^{nd} cycle', \
     'data-for-gnuplot.sol' index 2 using (stretch($5)):(nominal($19)) every 1  with lines ls 3 title '3^{rd} cycle'

# -------------------- Accumulated exiting fluid vs time  --------------------------------------------------------------------------
set xrange [] noreverse
set yrange [] noreverse
set xtics 30 norotate nooffset
set ytics 1.5
set autoscale xy
set format x "%3.0f"
set format y "%3.1f"

set xlabel 'time [s]' font 'Verdana,14'
set ylabel sprintf("accumulated fluid\n outside the sample [mm^3]") font 'Verdana,14'

unset key
a=0
y_axis(y) = (a=a+y*timestep,a)

set output "brain_cube_fluid-time-poro-SHEAR.png"
plot 'data-for-gnuplot.sol' index 0 using ($1):(y_axis($22)) every 1  with lines ls 1 title '1^{st} cycle', \
     'data-for-gnuplot.sol' index 1 using ($1):(y_axis($22)) every 1  with lines ls 2 title '2^{nd} cycle', \
     'data-for-gnuplot.sol' index 2 using ($1):(y_axis($22)) every 1  with lines ls 3 title '3^{rd} cycle'

# -------------------- Accumulated exiting fluid vs reaction pressure  ---------------------------------------------------------------
set xlabel 'fluid reaction pressure [kPa]' font 'Verdana,14'
set ylabel sprintf("accumulated fluid\n outside the sample [mm^3]") font 'Verdana,14'
set key on box outside above right font 'Verdana,12' spacing 1.0 maxcols 1 width -0.2

set autoscale xy
set format x "%3.1f"
set ytics 1.5
set xtics 1
a=0
y_axis(y) = (a=a+y*timestep,a)

set output "brain_cube_fluid-reac_p-poro-SHEAR.png"
plot 'data-for-gnuplot.sol' index 0 using ($18/1000):(y_axis($22)) every 1  with lines ls 1 title '1^{st} cycle', \
     'data-for-gnuplot.sol' index 1 using ($18/1000):(y_axis($22)) every 1  with lines ls 2 title '2^{nd} cycle', \
     'data-for-gnuplot.sol' index 2 using ($18/1000):(y_axis($22)) every 1  with lines ls 3 title '3^{rd} cycle'

# -------------------- Pressure and displ. vs time -------------------------------------------------------------------------
set term png size 600,380 enhanced font 'Verdana,12'

set format x "%3.0f"
set format y "%3.1f"
set xtics 30 norotate nooffset

set xlabel 'time [s]' font 'Verdana,14'
set ylabel 'displacement [mm]' font 'Verdana,14'
set y2label 'pressure [kPa]' font 'Verdana,14'
set ytics 0.5 nomirror
set y2tics autofreq
set key width -5
set output "brain_cube_pres-displ-vs-time-poro-SHEAR.png"
plot 'data-for-gnuplot.sol' using ($1):($8)/1000 every 1  with lines ls 5 axes x1y2 title 'pressure at central column', \
     'data-for-gnuplot.sol' using ($1):($5) every 1  with lines ls 4 axes x1y1 title 'vertical displacement of top surface'
