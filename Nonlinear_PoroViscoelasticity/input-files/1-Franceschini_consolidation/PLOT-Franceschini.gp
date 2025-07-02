## -----------------------------------------------------------------------------
##
## SPDX-License-Identifier: LGPL-2.1-or-later
## Copyright (C) 2020 by Ester Comellas
## Copyright (C) 2020 by Jean-Paul Pelteret
##
## This file is part of the deal.II code gallery.
##
## -----------------------------------------------------------------------------

# GNUPLOT file to plot consolidation curve Francheschini (2006)

#To see color names:
# >> show palette colornames

#To see line style, line width, etc.:
# >> set terminal wxt enhanced dashed
# >> test
clear
reset
set datafile separator ","

# -------------------------------------------------------------------------------
# DATA EXTRACTED WITH ENGAUGE FROM FRANCESCHINI PAPER COPIED HERE
# -------------------------------------------------------------------------------
$dataFranceschiniFig2Ce << EOD
#minutes, consolidation ratio
0.083718,0.0738636
0.167816,0.127841
0.251772,0.15625
0.336392,0.193182
0.424147,0.21875
0.504685,0.238636
0.589026,0.252841
0.674309,0.269886
0.757172,0.284091
0.833952,0.301136
0.918517,0.318182
1.01166,0.323864
1.11424,0.335227
1.18072,0.34375
1.27557,0.349432
1.37804,0.355114
1.51778,0.360795
1.6397,0.366477
1.77142,0.375
1.87711,0.383523
2.19081,0.403409
2.55693,0.426136
2.87114,0.448864
3.22396,0.471591
3.48294,0.488636
3.91095,0.505682
4.56453,0.536932
5.32734,0.559659
5.98199,0.585227
6.58858,0.605114
7.25668,0.625
7.83963,0.644886
8.63459,0.659091
9.14975,0.676136
10.0776,0.6875
11.9911,0.71875
13.995,0.744318
16.0213,0.761364
18.341,0.784091
19.8144,0.798295
22.2493,0.815341
24.0366,0.838068
26.474,0.857955
28.0535,0.872159
30.8982,0.886364
36.0618,0.897727
40.4932,0.909091
46.3562,0.917614
51.0568,0.931818
56.2341,0.940341
61.9364,0.948864
66.9119,0.954545
72.287,0.960227
78.094,0.965909
84.3674,0.971591
114.922,0.985795
144.902,0.988636
175.779,0.994318
209.156,0.994318
239.439,0.994318
268.862,0.994318
296.126,0.994318
326.154,0.994318
352.355,1.00284
380.66,0.997159
411.239,0.994318
EOD

fileNH = "data-for-gnuplot.sol"
# -------------------------------------------------------------------------------
set term epslatex size 9cm,6cm color colortext standalone
set output "Franceschini-consolidation.tex"

unset colorbox
unset grid

set style line 1  lt 7 lw 2 lc rgb "black"   ps 0.7 dt 1
set style line 2  lt 2 lw 2 lc rgb "gray25"  ps 0.7 dt 2
set style line 3  lt 6 lw 2 lc rgb "gray50"  ps 0.7 dt 3
set style line 4  lt 10 lw 2 lc rgb "gray75"  ps 0.7

set style line 5  lt 2  lw 5   lc rgb "dark-blue"

# ---------  CONSOLIDATION CURVE  -------------------------------------------
#Final specimen shortening                                                                         ****CHANGE IF NEEDED***
Umax = 8.1e-2

unset key

# ---------  EPSLATEX  -------------------------------------------
set key box out vert right above box width -6.0 height 1.0

set logscale x
set xrange [0.01:1000]
set yrange [1.1:-0.1] reverse

set xlabel 'time [min]'
set ylabel 'consolidation ratio'

plot '$dataFranceschiniFig2Ce' with points  ls 3 title 'experimental data', \
      fileNH using (($1)/60):(-($7)/Umax) with lines ls 5 title 'nonlinear poroelasticity'



# ---------  PNG  -------------------------------------------

set term png size 450,300 enhanced font 'Verdana,12'
set key box out vert right above box font 'Verdana,12' width -4.0

set logscale x
set xrange [0.01:1000]
set yrange [1.1:-0.1] reverse

set xlabel 'time [min]' font 'Verdana,14'
set ylabel 'consolidation ratio' font 'Verdana,14'

set output "Franceschini-consolidation.png"
plot '$dataFranceschiniFig2Ce' with points  ls 1 title 'experimental data', \
      fileNH using (($1)/60):(-($7)/Umax) with lines ls 5 title 'nonlinear poroelasticity'
