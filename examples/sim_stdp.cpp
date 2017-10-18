/* 
* Copyright 2014-2017 Friedemann Zenke
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
 * \brief Example simulation of a single postsynaptic neuron with Poisson input and pair-based additive STDP.
 *
 * The default parameters are set such that postsynaptic firing rates stabilize.
 */

#include "auryn.h"

using namespace auryn;

namespace po = boost::program_options;

int main(int ac, char* av[]) 
{

	string dir = ".";
	string file_prefix = "poisson";

	char strbuf [255];
	string msg;

	double simtime = 100.;

	NeuronID nbinputs = 1000;
	NeuronID size = 10;
	unsigned int seed = 1;
	double kappa = 5.;
	AurynWeight winit = 0.01;

	double eta = 1.0e-3; // learning rate
	double tau_pre = 20e-3; // stdp window decay (pre-post part)
	double tau_post = 20e-3; // stdp window decay (post-pre part)

	int errcode = 0;

    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("dir", po::value<string>(), "output directory")
            ("simtime", po::value<double>(), "simulation time")
            ("eta", po::value<double>(), "learning rate")
            ("winit", po::value<double>(), "initial weight")
            ("kappa", po::value<double>(), "presynaptic firing rate")
            ("nbinputs", po::value<int>(), "number of Poisson inputs")
            ("size", po::value<int>(), "number of neurons")
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

        if (vm.count("eta")) {
            std::cout << "eta set to " 
                 << vm["eta"].as<double>() << ".\n";
			eta = vm["eta"].as<double>();
        } 

        if (vm.count("winit")) {
            std::cout << "winit set to " 
                 << vm["winit"].as<double>() << ".\n";
			winit = vm["winit"].as<double>();
        } 

        if (vm.count("nbinputs")) {
            std::cout << "nbinputs set to " 
                 << vm["nbinputs"].as<int>() << ".\n";
			nbinputs = vm["nbinputs"].as<int>();
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

	// Init Auryn
	auryn_init(ac, av, dir, "sim_stdp");
	sys->set_master_seed(seed);

	// Neurons
	PoissonGroup * poisson = new PoissonGroup(nbinputs, kappa);
	TIFGroup * neurons = new TIFGroup(size);

	// point online rate monitor to neurons
	sys->set_online_rate_monitor_id(neurons->get_uid());

	// Connections
	float sparseness = 1.0;
	STDPConnection * stdp_con = new STDPConnection(poisson, neurons, winit, sparseness, tau_pre, tau_post );
	stdp_con->A = -1.20*tau_post/tau_pre*eta; // post-pre
	stdp_con->B = eta; // pre-post
	stdp_con->set_min_weight(0.0);
	stdp_con->set_max_weight(1.0);

	// Monitors
	// Record output firing rate (sample every 1s)
	PopulationRateMonitor * pmon_e = new PopulationRateMonitor( neurons, sys->fn("prate"), 1.0 );

	// Record mean synaptic weight
	WeightStatsMonitor * wsmon = new WeightStatsMonitor( stdp_con, sys->fn("msyn") );

	// Record individual synaptic weights (sample every 10s)
	WeightMonitor * wmon = new WeightMonitor( stdp_con, sys->fn("syn"), 10.0 );
	wmon->add_equally_spaced(100); // record 100 synapses

	// Check that rates do not drop below 1e-3Hz or increase beyond 100
	RateChecker * chk = new RateChecker( neurons, 1e-3 , 100 , 10);

	// Run simulation
	if (!sys->run(simtime)) errcode = 1;

	// Close Auryn
	logger->msg("Freeing ...",PROGRESS,true);
	auryn_free();
	return errcode;
}
