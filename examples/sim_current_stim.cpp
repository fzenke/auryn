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


/*! \brief Example illustrating the use of PoissonStimulator
 *
 * This example shows how one can add independent random Poisson currents to
 * each neuron in a Neurongroup. The example also works for NormalStimulator
 * in which the random currents follow a normal distribution
 */

#include "auryn.h"

using namespace auryn;

namespace po = boost::program_options;
namespace mpi = boost::mpi;

int main(int ac, char* av[]) 
{

	int errcode = 0;
	char strbuf [255];
	string outputfile = "out_current_stim";
	string tmpstr;

	// BEGIN Global definitions
	mpi::environment env(ac, av);
	mpi::communicator world;
	communicator = &world;

	try
	{
		sprintf(strbuf, "out_current_stim.%d.log", world.rank());
		string logfile = strbuf;
		logger = new Logger(logfile,world.rank(),PROGRESS,EVERYTHING);
	}
	catch ( AurynOpenFileException excpt )
	{
		std::cerr << "Cannot proceed without log file. Exiting all ranks ..." << '\n';
		env.abort(1);
	}

	sys = new System(&world);
	// END Global definitions
	
	IFGroup * neurons = new IFGroup(2);
	AurynWeight weight = 1e-2; // weight in mV if the target of the stimulator is "mem"
	AurynFloat rate = 100.0;  // 
	PoissonStimulator * stim = new PoissonStimulator(neurons, rate, weight);


	tmpstr = outputfile;
	tmpstr += ".mem0";
	VoltageMonitor * vmon0 = new VoltageMonitor( neurons, 0, tmpstr.c_str() );

	tmpstr = outputfile;
	tmpstr += ".mem1";
	VoltageMonitor * vmon1 = new VoltageMonitor( neurons, 1, tmpstr.c_str() );


	logger->msg("Running ...",PROGRESS);
	sys->run(1);

	logger->msg("Freeing ...",PROGRESS,true);
	delete sys;

	if (errcode)
		env.abort(errcode);
	return errcode;
}
