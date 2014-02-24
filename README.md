Auryn 
=====

This is the README file that comes with your version of Auryn.

Auryn is a source package used to create highly specialized and optimized code
to simulate recurrent spiking neural networks with spike timing dependent
plasticity (STDP). It comes with the GPLv3 (please see COPYING).


Documentation
-------------

Please visit http://www.fzenke.net/auryn/


Installation/Use
----------------

The simulation environment consists of a bunch of source files that are 
contained in the ./src directory no installation is required. The Auryn 
simulator is directly compiled into the simulation file which is a
default C++ program. You will find some example programs in ./sim. To build
them you find an example Makefile under ./build/home which you can modify to
your needs. As for now Auryn does not come with the autoconf/automake toolchain
due to problems with the inclusion of the necessary MPI libraries.


Requirements
------------

Auryn needs the boost libraries (www.boost.org) with MPI support and GSL 
(http://www.gnu.org/software/gsl/) installed in development versions to
compile.





-- Friedemann Zenke, Jan 26 2014





Copyright 2014 Friedemann Zenke.
Copying and distribution of this file, with or without modification, are
permitted provided the copyright notice and this notice are preserved.
