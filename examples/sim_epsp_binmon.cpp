/* 
* Copyright 2014-2016 Friedemann Zenke
*
* This file is part of Auryn, a simulation package for plastic
* spiking neural networks.
* 
* Auryn is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* Auryn is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with Auryn.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "auryn.h"

#define N 1

/*!\file 
 *
 * \brief Example simulation which simulates one Poisson input onto a single postsynaptic neuron and records the membrane potential 
 * */

using namespace auryn;

namespace po = boost::program_options;

int main(int ac, char* av[]) 
{

	int errcode = 0;
	char strbuf [255];
	string simname = "out_epsp";
	string tmpstr;
	AurynWeight w = 1.0;

	// BEGIN Global definitions
	auryn_init( ac, av );
	sys->set_simulation_name(simname);
	// END Global definitions

	// define input group
	PoissonGroup * poisson = new PoissonGroup(N,1.);

	// define receiving group
	IFGroup * neuron = new IFGroup(1);

	// define connection
	IdentityConnection * con = new IdentityConnection(poisson,neuron,w,GLUT);

	// define monitors
	BinaryStateMonitor * stmon = new BinaryStateMonitor( neuron, 0, "mem", sys->fn("bmem"), 1e-3 );
	BinaryStateMonitor * amon = new BinaryStateMonitor( neuron, 0, "g_ampa", sys->fn("bampa") );
	BinaryStateMonitor * nmon = new BinaryStateMonitor( neuron, 0, "g_nmda", sys->fn("bnmda") );

	// run simulation
	logger->msg("Running ...",PROGRESS);
	sys->run(10);

	if (errcode)
		auryn_abort(errcode);
	return errcode;

	logger->msg("Freeing ...",PROGRESS,true);
	auryn_free();
}
