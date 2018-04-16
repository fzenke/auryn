![Auryn logo](https://github.com/fzenke/auryn/blob/master/doc/auryn_logo_small.png "Auryn logo")

Auryn 
=====

Auryn is Simulator for recurrent spiking neural networks with synaptic
plasticity. It comes with the GPLv3 (please see COPYING).

* For examples and documentation visit https://www.fzenke.net/auryn/
* Please reporte issues here https://github.com/fzenke/auryn/issues
* For questions and support https://www.fzenke.net/auryn/forum/

Quick start
-----------

Note, Auryn needs a C++ compiler, the boost libraries (www.boost.org) with MPI
support installed. To download and compile the examples under Linux try:

```
sudo apt-get install cmake git build-essential libboost-all-dev
git clone https://github.com/fzenke/auryn.git && cd auryn/build/release
./bootstrap.sh && make
```

Run a first network simulation
------------------------------

```
cd examples
./sim_coba_benchmark --dir .
```
will run the Vogels Abbott benchmark, a balanced network model with conductance based synapses.
Spiking activity is written to files with the extension 'ras'. 

If you have gnuplot installed, you can visualize the output of the simulation follows:
```
echo "set xlabel 'Time [s]'; plot [1:2] 'coba.0.e.ras' with dots lc rgb 'black'" | gnuplot -p
```

![Spike raster plot](http://www.fzenke.net/auryn/lib/exe/fetch.php?cache=&media=coba_ras.png "coba ras")

Here time in seconds is plotted on the x-asis and neuron id on the y-axis.



Install as a library
--------------------

To install Auryn as a library run:
```
sudo make install
```
which will put it under `/usr/local/` or for a local install
```
make DESTDIR=./your/dir/ install
```


Citing Auryn
------------

If you find Auryn useful and you use it, please cite:

Zenke, F. and Gerstner, W., 2014.  Limits to high-speed simulations of spiking
neural networks using general-purpose computers.  Front Neuroinform 8, 76. 
doi: 10.3389/fninf.2014.00076

url: http://journal.frontiersin.org/Journal/10.3389/fninf.2014.00076/abstract

Bibtex:
```
@article{zenke_limits_2014,
	title = {Limits to high-speed simulations of spiking neural networks using general-purpose computers},
	author = {Zenke, Friedemann and Gerstner, Wulfram},
	journal = {Front Neuroinform},
	year = {2014},
	volume = {8},
	url = {http://journal.frontiersin.org/Journal/10.3389/fninf.2014.00076/abstract},
	doi = {10.3389/fninf.2014.00076}
}
```



License & Copyright 
-------------------

Copyright 2014-2018 Friedemann Zenke

Auryn is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Auryn is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Auryn.  If not, see <http://www.gnu.org/licenses/>.
