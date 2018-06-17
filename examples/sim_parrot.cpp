/* 
* Copyright 2014-2018 Friedemann Zenke
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


/*!\file 
 *
 * \brief Example simulation which illustrates the use of ParrotGroup
 */

#include "auryn.h"

using namespace auryn;

namespace po = boost::program_options;

int main(int ac, char* av[]) 
{

	string dir = "./";
	string file_prefix = "parrot";

	string msg;

	NeuronID size = 1000;
	unsigned int seed = 1;
	double kappa = 5.;
	double simtime = 1.;

	int errcode = 0;

    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("simtime", po::value<double>(), "simulation time")
            ("kappa", po::value<double>(), "poisson group rate")
            ("dir", po::value<string>(), "output directory")
            ("size", po::value<int>(), "poisson group size")
            ("seed", po::value<int>(), "random seed")
        ;

        po::variables_map vm;        
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);    

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 1;
        }

        if (vm.count("kappa")) {
            std::cout << "kappa set to " 
                 << vm["kappa"].as<double>() << ".\n";
			kappa = vm["kappa"].as<double>();
        } 

        if (vm.count("dir")) {
            std::cout << "dir set to " 
                 << vm["dir"].as<string>() << ".\n";
			dir = vm["dir"].as<string>();
        } 

        if (vm.count("simtime")) {
            std::cout << "simtime set to " 
                 << vm["simtime"].as<double>() << ".\n";
			simtime = vm["simtime"].as<double>();
        } 

        if (vm.count("size")) {
            std::cout << "size set to " 
                 << vm["size"].as<int>() << ".\n";
			size = vm["size"].as<int>();
        } 

        if (vm.count("seed")) {
            std::cout << "seed set to " 
                 << vm["seed"].as<int>() << ".\n";
			seed = vm["seed"].as<int>();
        } 
    }
    catch(std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
        std::cerr << "Exception of unknown type!\n";
    }

	auryn_init(ac, av);
	sys->set_master_seed(seed);
	logger->set_logfile_loglevel(EVERYTHING);

	PoissonGroup * poisson = new PoissonGroup(size,kappa);
	ParrotGroup * parrot = new ParrotGroup(poisson);
	parrot->set_delay(20); // sets the axonal delay in units of auryn_timeteps.

	// Record original spikes before axonal delay 
	SpikeMonitor * smon1 = new SpikeMonitor( poisson, sys->fn("poisson","ras"), size);

	DelayedSpikeMonitor * dmon1 = new DelayedSpikeMonitor( poisson, sys->fn("poisson2","ras"), size);

	// Record parrot spikes before axonal delay
	DelayedSpikeMonitor * smon2 = new DelayedSpikeMonitor( parrot, sys->fn("parrot","ras"), size);

	// Record original spikes before axonal delay 
	DelayedSpikeMonitor * dmon2 = new DelayedSpikeMonitor( parrot, sys->fn("parrot2","ras"), size);

	DelayedSpikeMonitor * dsmon1 = new DelayedSpikeMonitor( poisson, sys->fn("poisson","dras"), size);
	DelayedSpikeMonitor * dsmon2 = new DelayedSpikeMonitor( parrot, sys->fn("parrot","dras"), size);

	RateChecker * chk = new RateChecker( poisson , -1 , 20.*kappa , 10);
	if (!sys->run(simtime,false)) 
			errcode = 1;

	if (errcode)
		auryn_abort(errcode);
	logger->msg("Freeing ...",PROGRESS,true);
	auryn_free();
	return errcode;
}
