/* 
* Copyright 2014-2020 Friedemann Zenke
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
 * \brief Example simulation which demonstrates the use of FileCurrentInjector 
 * */

using namespace auryn;

namespace po = boost::program_options;

int main(int ac, char* av[]) 
{

	int errcode = 0;
	string simname = "fcur_inject";
	string logfile = simname;
	string tmpstr;

	// BEGIN Global definitions
	auryn_init( ac, av );
	sys->set_simulation_name(simname);
	// END Global definitions

	// define receiving group
	IFGroup * neurons = new IFGroup(3);

	// define monitors
	VoltageMonitor * vmon0 = new VoltageMonitor( neurons, 0, sys->fn("neuron",0,"mem"), 1e-3 );
	VoltageMonitor * vmon1 = new VoltageMonitor( neurons, 1, sys->fn("neuron",1,"mem"), 1e-3 );

	// define FileCurrentInjector
	FileCurrentInjector * injector = new FileCurrentInjector( neurons, "inject_current.txt", "mem" );
	injector->set_scale(1.2); // Scales all current values by this factor

	// To repeat the input current enable looping by uncommenting the following lines
	// injector->loop = true;
	// injector->set_loop_grid(10.0); // align time series to 1s grid (should be larger or equal to the input time series)

	// To only inject current in some neurons uncomment the following code 
	// injector->mode = LIST;
	// injector->target_neuron_ids->push_back(0);
	// injector->target_neuron_ids->push_back(2);

	// run simulation
	logger->msg("Running ...",PROGRESS);
	sys->run(15.0);

	if (errcode)
		auryn_abort(errcode);

	logger->msg("Freeing ...",PROGRESS,true);
	auryn_free();
	return errcode;
}
