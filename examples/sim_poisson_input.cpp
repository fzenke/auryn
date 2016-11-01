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

/*!\file 
 *
 * \brief Simulates exc and inh Poisson inputs to a small population of
 * AdEx neurons and records their membrane potentials.
 * */

using namespace auryn;

namespace po = boost::program_options;

int main(int ac, char* av[]) 
{

	int errcode = 0;
	char strbuf [255];
	string simname = "poisson_input";
	auryn_init( ac, av, ".", simname );

	// define neuron group
	const int size_output = 10;
	AdExGroup* neurons = new AdExGroup(size_output);
	// alternatively you can also use an
	// IzhikevichGroup * neurons = new IzhikevichGroup(1);

	// define Poisson input group
	PoissonGroup * poisson_exc = new PoissonGroup(800, 10);
	PoissonGroup * poisson_inh = new PoissonGroup(200, 10);

	// connect sparsely
	const double weight = 0.2;
	const double sparseness = 0.1;
	SparseConnection * con_exc = new SparseConnection( poisson_exc, neurons, weight, sparseness );
	SparseConnection * con_inh = new SparseConnection( poisson_inh, neurons, weight, sparseness, GABA);

	// define monitors
	SpikeMonitor * smon = new SpikeMonitor( neurons, sys->fn("ras") );
	for ( int i = 0 ; i < size_output ; ++i ) {
		VoltageMonitor * vmon = new VoltageMonitor( neurons, i, sys->fn("neurons",i,"mem") );
	}

	// run simulation
	logger->msg("Running ...",PROGRESS);

	const double simtime = 1.0;
	// simulate
	sys->run(simtime);

	if (errcode)
		auryn_abort(errcode);

	logger->msg("Freeing ...",PROGRESS,true);
	auryn_free();
	return errcode;
}
