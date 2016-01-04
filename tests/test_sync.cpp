/* 
* Copyright 2014-2015 Friedemann Zenke
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
*
* If you are using Auryn or parts of it for your work please cite:
* Zenke, F. and Gerstner, W., 2014. Limits to high-speed simulations 
* of spiking neural networks using general-purpose computers. 
* Front Neuroinform 8, 76. doi: 10.3389/fninf.2014.00076
*/

#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <string>

#include <boost/program_options.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>

#include "auryn_global.h"
#include "auryn_definitions.h"
#include "System.h"
#include "Logger.h"
#include "PoissonGroup.h"
#include "IFGroup.h"
#include "P09Connection.h"
#include "SpikeMonitor.h"
#include "DelayedSpikeMonitor.h"
#include "PopulationRateMonitor.h"
#include "RateChecker.h"

using namespace std;

namespace po = boost::program_options;
namespace mpi = boost::mpi;

/*! \brief Small simultion to test the functionality of SyncBuffer at a high level.
 *
 * Run the test using for instance
 * mpirun -n 3 ./test_sync
 *
 * Then in gnuplot review the output as follows:
 * plot '< peep.sh test_sync.a.0.ras' w d, '< peep.sh test_sync.a.1.ras' w d, '< peep.sh test_sync.a.2.ras' w d, '< peep.sh test_sync.a.1.dras' using ($1-0.0008):2 w d lc -1
 *
 * In the last third all spikes should be covered by black dots. Vice versa, when reversing the order all black dots should be covered by colored dots.
 */
int main(int ac, char* av[]) 
{

	string dir = "./";
	string file_prefix = "test_sync";

	char strbuf [255];
	string msg;

	NeuronID size = 1000;
	NeuronID seed = 1;
	double kappa = 5.;
	double simtime = 1.;

	int errcode = 0;

    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("simtime", po::value<double>(), "simulation time")
            ("kappa", po::value<double>(), "poisson group rate")
            ("size", po::value<int>(), "poisson group size")
            ("seed", po::value<int>(), "random seed")
        ;

        po::variables_map vm;        
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);    

        if (vm.count("help")) {
            cout << desc << "\n";
            return 1;
        }


        if (vm.count("kappa")) {
            cout << "kappa set to " 
                 << vm["kappa"].as<double>() << ".\n";
			kappa = vm["kappa"].as<double>();
        } 

        if (vm.count("simtime")) {
            cout << "simtime set to " 
                 << vm["simtime"].as<double>() << ".\n";
			simtime = vm["simtime"].as<double>();
        } 

        if (vm.count("size")) {
            cout << "size set to " 
                 << vm["size"].as<int>() << ".\n";
			size = vm["size"].as<int>();
        } 

        if (vm.count("seed")) {
            cout << "seed set to " 
                 << vm["seed"].as<int>() << ".\n";
			seed = vm["seed"].as<int>();
        } 
    }
    catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
        cerr << "Exception of unknown type!\n";
    }

	// BEGIN Global stuff
	mpi::environment env(ac, av);
	mpi::communicator world;
	communicator = &world;

	sprintf(strbuf, "%s/%s.%d.log", dir.c_str(), file_prefix.c_str(), world.rank());
	string logfile = strbuf;
	logger = new Logger(logfile,world.rank(),PROGRESS,EVERYTHING);

	sys = new System(&world);
	// END Global stuff

	PoissonGroup * poisson = new PoissonGroup(size,kappa);
	PoissonGroup * poisson2 = new PoissonGroup(size,kappa);

	IFGroup * neurons = new IFGroup(100);

	P09Connection * con = new P09Connection(
			poisson,neurons,
			0.1,0.1);

	P09Connection * con2 = new P09Connection(
			poisson2,neurons,
			0.1,0.1);

	sprintf(strbuf, "%s/%s.a.%d.ras", dir.c_str(), file_prefix.c_str(), world.rank() );
	SpikeMonitor * smon_e = new SpikeMonitor( poisson, strbuf, size);

	sprintf(strbuf, "%s/%s.a.%d.dras", dir.c_str(), file_prefix.c_str(), world.rank() );
	DelayedSpikeMonitor * dsmon_e = new DelayedSpikeMonitor( poisson, strbuf, size);

	sprintf(strbuf, "%s/%s.b.%d.ras", dir.c_str(), file_prefix.c_str(), world.rank() );
	SpikeMonitor * smon_e2 = new SpikeMonitor( poisson2, strbuf, size);

	sprintf(strbuf, "%s/%s.b.%d.dras", dir.c_str(), file_prefix.c_str(), world.rank() );
	DelayedSpikeMonitor * dsmon_e2 = new DelayedSpikeMonitor( poisson2, strbuf, size);

	sprintf(strbuf, "%s/%s.%d.prate", dir.c_str(), file_prefix.c_str(), world.rank() );
	PopulationRateMonitor * pmon_e = new PopulationRateMonitor( poisson, strbuf, 1.0 );

	RateChecker * chk = new RateChecker( poisson , -1 , 20.*kappa , 10);
	if (!sys->run(simtime,false)) 
			errcode = 1;

	logger->msg("Freeing ...",PROGRESS,true);
	delete sys;

	if (errcode)
		env.abort(errcode);
	return errcode;
}
