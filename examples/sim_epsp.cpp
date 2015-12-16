/* 
* Copyright 2014 Friedemann Zenke
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

using namespace auryn;

namespace po = boost::program_options;
namespace mpi = boost::mpi;

int main(int ac, char* av[]) 
{

	int errcode = 0;
	char strbuf [255];
	string outputfile = "out_epsp";
	string tmpstr;
	AurynWeight w = 1.0;

	// BEGIN Global definitions
	mpi::environment env(ac, av);
	mpi::communicator world;
	communicator = &world;

	try
	{
		sprintf(strbuf, "out_epsp.%d.log", world.rank());
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
	
	PoissonGroup * poisson = new PoissonGroup(N,1.);
	IFGroup * neuron = new IFGroup(1);

	IdentityConnection * con = new IdentityConnection(poisson,neuron,w,GLUT);

	tmpstr = outputfile;
	tmpstr += ".ras";
	SpikeMonitor * smon = new SpikeMonitor( neuron, tmpstr.c_str() );

	tmpstr = outputfile;
	tmpstr += ".mem";
	VoltageMonitor * vmon = new VoltageMonitor( neuron, 0, tmpstr.c_str() );

	tmpstr = outputfile;
	tmpstr += ".ampa";
	AmpaMonitor * amon = new AmpaMonitor( neuron, 0, tmpstr.c_str() );

	tmpstr = outputfile;
	tmpstr += ".nmda";
	NmdaMonitor * nmon = new NmdaMonitor( neuron, 0, tmpstr.c_str() );

	logger->msg("Running ...",PROGRESS);
	sys->run(10);

	logger->msg("Freeing ...",PROGRESS,true);
	delete sys;

	if (errcode)
		env.abort(errcode);
	return errcode;
}
