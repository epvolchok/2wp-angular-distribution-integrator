#!/usr/bin/gnuplot
set term pdf enhanced butt size 10cm,7cm font 'Helvetica ,11'
set output 'PD.pdf'
unset key
set xlabel "$\alpha$"
set ylabel "P, GW"


plot './Results/ResPower' u 1:2 w l lw 2 lt rgb 'black'
