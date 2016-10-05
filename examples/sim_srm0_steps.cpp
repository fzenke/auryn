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
 * \brief Simulates multiple step currents of increasing amplitude to 100 SRM0 neurons
 * */

using namespace auryn;

namespace po = boost::program_options;

int main(int ac, char* av[]) 
{

	int errcode = 0;
	char strbuf [255];
	string simname = "srm0steps";
	auryn_init( ac, av, ".", simname );

	// define neuron group
	SRM0Group* neuron = new SRM0Group(100);

	// define current input 
	CurrentInjector * curinject = new CurrentInjector(neuron, "mem");

	// define monitors
	SpikeMonitor * smon = new SpikeMonitor( neuron, sys->fn("ras") );
	VoltageMonitor * vmon = new VoltageMonitor( neuron, 0, sys->fn("mem") );
	StateMonitor * wmon = new StateMonitor( neuron, 0, "w", sys->fn("w") );
	PopulationRateMonitor * prmon = new PopulationRateMonitor( neuron, sys->fn("prate"), 10e-3 );

	// run simulation
	logger->msg("Running ...",PROGRESS);

	const double simtime = 0.2;
	// simulate
	sys->run(simtime);

	// simulate current steps of increasing size
	for ( int i = 0 ; i < 10 ; ++i ) {
		// turn current on
		curinject->set_all_currents(0.2*i); // current is in arbitrary units

		// simulate 
		sys->run(simtime); 

		// turn current off
		curinject->set_all_currents(0.0); 

		// simulate 
		sys->run(simtime); 
	}

	if (errcode)
		auryn_abort(errcode);

	logger->msg("Freeing ...",PROGRESS,true);
	auryn_free();
	return errcode;
}
