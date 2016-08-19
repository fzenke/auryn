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
 * \brief Example simulation which simulates a neuron during current step injection
 * */

using namespace auryn;

namespace po = boost::program_options;

int main(int ac, char* av[]) 
{

	int errcode = 0;
	char strbuf [255];
	string simname = "step_current";
	string tmpstr;
	AurynWeight w = 1.0;

	auryn_init(ac, av);

	// define neuron group
	IzhikevichGroup * neuron = new IzhikevichGroup(1);

	// define current input 
	CurrentInjector * curinject = new CurrentInjector(neuron, "mem");

	// define monitors
	SpikeMonitor * smon = new SpikeMonitor( neuron, sys->fn("ras") );
	StateMonitor * vmon = new StateMonitor( neuron, 0, "mem", sys->fn("mem") );

	// run simulation
	logger->msg("Running ...",PROGRESS);

	// simulate 1 second
	sys->run(0.1);

	// turn current on
	curinject->set_current(0,10.0);

	// simulate 1 second
	sys->run(0.1);

	if (errcode)
		auryn_abort(errcode);

	// clean up
	logger->msg("Freeing ...",PROGRESS,true);
	auryn_free();

	return errcode;
}
