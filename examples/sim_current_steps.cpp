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
 * \brief Simulates multiple step currents of increasing amplitude to a neuron
 * */

using namespace auryn;

namespace po = boost::program_options;
namespace mpi = boost::mpi;

int main(int ac, char* av[]) 
{

	int errcode = 0;
	char strbuf [255];
	string simname = "current_steps";
	string tmpstr;
	AurynWeight w = 1.0;

	// BEGIN Global definitions
	mpi::environment env(ac, av);
	mpi::communicator world;
	communicator = &world;

	try
	{
		sprintf(strbuf, "current_steps.%d.log", world.rank());
		string logfile = strbuf;
		logger = new Logger(logfile,world.rank(),PROGRESS,EVERYTHING);
	}
	catch ( AurynOpenFileException excpt )
	{
		std::cerr << "Cannot proceed without log file. Exiting all ranks ..." << '\n';
		env.abort(1);
	}

	sys = new System(&world);
	sys->set_simulation_name(simname);
	// END Global definitions

	// define neuron group
	AdExGroup * neuron = new AdExGroup(1);
	// alternatively you can also use an
	// IzhikevichGroup * neuron = new IzhikevichGroup(1);


	// define current input 
	CurrentInjector * curinject = new CurrentInjector(neuron, "mem");

	// define monitors
	SpikeMonitor * smon = new SpikeMonitor( neuron, sys->fn("ras") );
	VoltageMonitor * vmon = new VoltageMonitor( neuron, 0, sys->fn("mem") );
	StateMonitor * wmon = new StateMonitor( neuron, 0, "w", sys->fn("w") );

	// run simulation
	logger->msg("Running ...",PROGRESS);

	const double simtime = 0.2;
	// simulate
	sys->run(simtime);

	// simulate current steps of increasing size
	for ( int i = 0 ; i < 10 ; ++i ) {
		// turn current on
		curinject->set_current(0,1.0*i); // current is in arbitrary units

		// simulate 
		sys->run(simtime); 

		// turn current off
		curinject->set_current(0,0.0); 

		// simulate 
		sys->run(simtime); 
	}

	// clean up
	logger->msg("Freeing ...",PROGRESS,true);
	delete sys;

	if (errcode)
		env.abort(errcode);
	return errcode;
}
