/* 
* Copyright 2014-2019 Friedemann Zenke
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
 * \brief Simulates multiple step currents of increasing amplitude to a NBG neuron
 * */

using namespace auryn;

namespace po = boost::program_options;

int main(int ac, char* av[]) 
{
	int errcode = 0;
	char strbuf [255];
	string simname = "nbg";
	auryn_init( ac, av, ".", simname );

	// define neuron group
	NaudGroup* neuron = new NaudGroup(1);

	// define current input 
	CurrentInjector * curr_inject1 = new CurrentInjector(neuron, "mem");
	CurrentInjector * curr_inject2 = new CurrentInjector(neuron, "Vd");

	// define monitors
	SpikeMonitor * smon = new SpikeMonitor( neuron, sys->fn("ras") );
	VoltageMonitor * vmon   = new VoltageMonitor( neuron, 0, sys->fn("mem") );
	StateMonitor * smon_vd  = new StateMonitor( neuron, 0, "Vd", sys->fn("Vd") );
	StateMonitor * smon_m   = new StateMonitor( neuron, 0, "thr", sys->fn("thr") );
	StateMonitor * smon_ws  = new StateMonitor( neuron, 0, "wsoma", sys->fn("wsoma") );

	// run simulation
	logger->msg("Running ...",PROGRESS);

	const double simtime = 200e-3;
	// simulate
	sys->run(simtime);

	// simulate current steps of increasing size
	const float s0 = 0.2;
	for ( int i = 0 ; i < 16 ; ++i ) {
		// turn current on
		// if ( i%3 == 0 ) {
		// 	curr_inject1->set_current(0,-s0*i); // current is in arbitrary units
		// }

		// if ( i%3 == 1 ) {
		// 	curr_inject2->set_current(0,-s0*i); // current is in arbitrary units
		// }

		if ( i%3 == 2 ) {
			curr_inject1->set_current(0,s0*i); // current is in arbitrary units
			curr_inject2->set_current(0,s0*i); // current is in arbitrary units
		}

		// simulate 
		sys->run(simtime); 

		// turn current off
		curr_inject1->set_current(0,0.0); 
		curr_inject2->set_current(0,0.0); 

		// simulate 
		sys->run(2*simtime); 
	}

	sys->run(5); 

	if (errcode)
		auryn_abort(errcode);

	logger->msg("Freeing ...",PROGRESS,true);
	auryn_free();
	return errcode;
}
