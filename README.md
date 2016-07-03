Auryn 
=====

This is the README file that comes with your version of Auryn.

![Auryn logo](http://www.fzenke.net/uploads/images/logo_trans_small.png "Auryn logo")

Auryn is a source package used to create highly specialized and optimized code
to simulate recurrent spiking neural networks with spike timing dependent
plasticity (STDP). It comes with the GPLv3 (please see COPYING).


Quick start
-----------

To download and compile the examples try:

```
sudo apt-get install cmake git build-essential libboost-all-dev
git clone https://github.com/fzenke/auryn.git && cd auryn 
mkdir build && cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release && make
```

Documentation & Installation/Use
--------------------------------

Please visit the wiki at http://www.fzenke.net/auryn/


Requirements
------------

Auryn needs the boost libraries (www.boost.org) with MPI support installed 
in development versions to compile.


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


Copyright 2014-2015 Friedemann Zenke.
Copying and distribution of this file, with or without modification, are
permitted provided the copyright notice and this notice are preserved.
