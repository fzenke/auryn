#!/usr/bin/gnuplot

set out 'benchmark_results.pdf'
set term pdf size 7.0, 2.4

unset key

set border 3 
set xtics nomirror out
set ytics nomirror out
set xtics rotate # by -45

set boxwidth 0.8



set boxwidth 1.0
set style histogram errorbars gap 1.0 lw 1
set style data histograms
# set style fill solid border -1

# set yrange [0:]

revname(x) = sprintf("%s",x[1:6])

set multiplot layout 1,3

set label 1 at screen 0.001, screen 0.1 'Commit:' left
set title 'Vogels \& Abbott benchmark'
set ylabel 'Wall time (s)'
plot 'benchmark_results.dat' using  3:4:xticlabels(revname(strcol(2))) lc -1

unset ylabel 
unset label 1


set title 'Zenke et al. benchmark, single'
plot 'benchmark_results.dat' using  5:6:xticlabels(revname(strcol(2))) lc -1
set title 'Zenke et al. benchmark, parallel'
plot 'benchmark_results.dat' using  7:8:xticlabels(revname(strcol(2))) lc -1
unset multiplot
