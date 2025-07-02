## -----------------------------------------------------------------------------
##
## SPDX-License-Identifier: LGPL-2.1-or-later
## Copyright (C) 2020 by Ester Comellas
## Copyright (C) 2020 by Jean-Paul Pelteret
##
## This file is part of the deal.II code gallery.
##
## -----------------------------------------------------------------------------

# GNUPLOT file to plot consolidation curves

#To see color names:
# >> show palette colornames

#To see line style, line width, etc.:
# >> set terminal wxt enhanced dashed
# >> test
clear
reset

file_base = "base_nos0.8_k40_Kos8.0e-11_mu0.89e-3/data-for-gnuplot.sol"
Umax_base = 1.958069e-02

file_nos1 = "nos0.7_k40_Kos8.0e-11_mu0.89e-3/data-for-gnuplot.sol"
Umax_nos1 = 1.967018e-02
file_nos2 = "nos0.9_k40_Kos8.0e-11_mu0.89e-3/data-for-gnuplot.sol"
Umax_nos2 = 1.931683e-02

file_k1 = "nos0.8_k1_Kos8.0e-11_mu0.89e-3/data-for-gnuplot.sol"
Umax_k1 = 1.958069e-02
file_k2 = "nos0.8_k80_Kos8.0e-11_mu0.89e-3/data-for-gnuplot.sol"
Umax_k2 = 1.958069e-02



file_mu1 = "nos0.8_k40_Kos8.0e-11_mu1.78e-4/data-for-gnuplot.sol"
Umax_mu1 = 1.958069e-02
file_mu2 = "nos0.8_k40_Kos8.0e-11_mu4.45e-3/data-for-gnuplot.sol"
Umax_mu2 = 1.953904e-02

file_Kos1 = "nos0.8_k40_Kos1.6e-11_mu0.89e-3/data-for-gnuplot.sol"
Umax_Kos1 = 1.953904e-02
file_Kos2 = "nos0.8_k40_Kos4.0e-10_mu0.89e-3/data-for-gnuplot.sol"
Umax_Kos2 = 1.958069e-02



# -------------------------------------------------------------------------------
unset colorbox
unset grid

set style line 1  lt 1 lw 2 lc rgb "black"   ps 0.7 dt 1
set style line 2  lt 2 lw 2 lc rgb "gray25"  ps 0.7 dt 2
set style line 3  lt 4 lw 2 lc rgb "gray50"  ps 0.7 dt 3
set style line 4  lt 10 lw 2 lc rgb "gray75"  ps 0.7

set style line 5  lt 2  lw 2   lc rgb "gray70"
set style line 6  lt 2  lw 2   lc rgb "gray40"
set style line 7  lt 2  lw 2   lc rgb "gray10"


# ---------  CONSOLIDATION CURVE  -------------------------------------------
#Final specimen shortening

set term png size 450,300 enhanced font 'Verdana,12'
unset key
set key box inside top right font 'Verdana,12' width 1.0

set logscale x
set xrange [0.1:100]
set yrange [1.1:0.0] reverse

set xlabel 'time [min]' font 'Verdana,14'
set ylabel 'consolidation ratio' font 'Verdana,14'

set output "consolidation-effect-nos.png"
plot file_nos1 using (($1)/60):(-($7)/Umax_nos1) with lines ls 5 title "n_{0s}=0.7",\
     file_base using (($1)/60):(-($7)/Umax_base) with lines ls 6 title "n_{0s}=0.8",\
     file_nos2 using (($1)/60):(-($7)/Umax_nos2) with lines ls 7 title "n_{0s}=0.9"

set output "consolidation-effect-k.png"
plot file_k1   using (($1)/60):(-($7)/Umax_k1)   with lines ls 5 title "{/Symbol k}=1",\
     file_base using (($1)/60):(-($7)/Umax_base) with lines ls 6 title "{/Symbol k}=40",\
     file_k2   using (($1)/60):(-($7)/Umax_k2)   with lines ls 7 title "{/Symbol k}=80"

set output "consolidation-effect-Kos.png"
plot file_Kos1 using (($1)/60):(-($7)/Umax_Kos1) with lines ls 5 title "K_{0s}= 16 nm^2",\
     file_base using (($1)/60):(-($7)/Umax_base) with lines ls 6 title "K_{0s}= 80 nm^2",\
     file_Kos2 using (($1)/60):(-($7)/Umax_Kos2) with lines ls 7 title "K_{0s}=400 nm^2"

set output "consolidation-effect-mu.png"
plot file_mu1  using (($1)/60):(-($7)/Umax_mu1)  with lines ls 5 title "{/Symbol m}^{FR}=0.178 mmPa·s",\
     file_base using (($1)/60):(-($7)/Umax_base) with lines ls 6 title "{/Symbol m}^{FR}=0.890 mmPa·s",\
     file_mu2  using (($1)/60):(-($7)/Umax_mu2)  with lines ls 7 title "{/Symbol m}^{FR}=4.450 mmPa·s"
