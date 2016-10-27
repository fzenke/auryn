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

/*!\file 
 *
 * \brief Implementation of the Brunel (2000) network with added STDP as implemented in NEST 
 *
 * This simulates the network from Brunel, N. (2000). Dynamics of sparsely
 * connected networks of excitatory and inhibitory spiking neurons. J Comput
 * Neurosci 8, 183–208.
 * as implemented in the NEST examples 
 * (https://github.com/nest/nest-simulator/tree/master/examples/nest).
 *
 * This file contains the network without plasticity. However, the same network
 * was implemented with and without weight-dependent STDP for performance
 * comparison with NEST. The results are published in Zenke, F., and Gerstner,
 * W. (2014). Limits to high-speed simulations of spiking neural networks using
 * general-purpose computers. Front Neuroinform 8, 76.
 *
 */

#include "auryn.h"

using namespace auryn;

namespace po = boost::program_options;

int main(int ac,char *av[]) {
	string dir = ".";

	std::stringstream oss;
	string strbuf ;
	string msg;

	NeuronID ne = 8000;
	NeuronID ni = 2000;

	NeuronID nrec = 50;

	double w = 0.1e-3; // 0.1mV PSC size
	double wext = 0.1e-3; 
	double sparseness = 0.1;
	double simtime = 1.;

	double lambda = 1e-2; 
	// For the benchmark this value was changed to 1e-9 to 
	// avoid frequency changes during the simulation. Otherwise
	// lambda should be in the order of 1e-2 - 1e-3.
	double gamma = 5.0;
	double poisson_rate = 20.0e3;

	string load = "";
	string save = "";

	string fwmat_ee = "";
	string fwmat_ei = "";
	string fwmat_ie = "";
	string fwmat_ii = "";

	int errcode = 0;

	

    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("simtime", po::value<double>(), "duration of simulation")
            ("gamma", po::value<double>(), "gamma factor for inhibitory weight")
            ("lambda", po::value<double>(), "learning rate")
            ("nu", po::value<double>(), "the external firing rate nu")
            ("dir", po::value<string>(), "dir from file")
            ("load", po::value<string>(), "load from file")
            ("save", po::value<string>(), "save to file")
            ("fee", po::value<string>(), "file with EE connections")
            ("fei", po::value<string>(), "file with EI connections")
            ("fie", po::value<string>(), "file with IE connections")
            ("fii", po::value<string>(), "file with II connections")
        ;

        po::variables_map vm;        
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);    

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 1;
        }

        if (vm.count("simtime")) {
			simtime = vm["simtime"].as<double>();
        } 

        if (vm.count("gamma")) {
			gamma = vm["gamma"].as<double>();
        } 

        if (vm.count("lambda")) {
			lambda = vm["lambda"].as<double>();
        } 

        if (vm.count("nu")) {
			poisson_rate = vm["nu"].as<double>();
        } 

        if (vm.count("dir")) {
			dir = vm["dir"].as<string>();
        } 

        if (vm.count("load")) {
			load = vm["load"].as<string>();
        } 

        if (vm.count("save")) {
			save = vm["save"].as<string>();
        } 

        if (vm.count("fee")) {
			fwmat_ee = vm["fee"].as<string>();
        } 

        if (vm.count("fie")) {
			fwmat_ie = vm["fie"].as<string>();
        } 

        if (vm.count("fei")) {
			fwmat_ei = vm["fei"].as<string>();
        } 

        if (vm.count("fii")) {
			fwmat_ii = vm["fii"].as<string>();
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
	oss << dir  << "/brunel." << sys->mpi_rank() << ".";
	string outputfile = oss.str();

	logger->msg("Setting up neuron groups ...",PROGRESS,true);
	IafPscDeltaGroup * neurons_e = new IafPscDeltaGroup( ne );
	neurons_e->set_tau_mem(20.0e-3);
	neurons_e->set_tau_ref(2.0e-3);
	neurons_e->e_rest = 0e-3;
	neurons_e->e_reset = 10e-3;
	neurons_e->thr = 20e-3;

	IafPscDeltaGroup * neurons_i = new IafPscDeltaGroup( ni );
	neurons_i->set_tau_mem(20.0e-3);
	neurons_i->set_tau_ref(2.0e-3);
	neurons_i->e_rest = 0e-3;
	neurons_i->e_reset = 10e-3;
	neurons_i->thr = 20e-3;

	logger->msg("Setting up Poisson input ...",PROGRESS,true);
	// The traditional way to implement the network is with 
	// independent Poisson noise.
	PoissonStimulator * pstim_e
		= new PoissonStimulator( neurons_e, poisson_rate, wext );
	PoissonStimulator * pstim_i
		= new PoissonStimulator( neurons_i, poisson_rate, wext );

	// The following would give correlated poisson noise from a single
	// population of Poisson Neurons.
	// PoissonGroup * poisson 
	// 	= new PoissonGroup( ne, poisson_rate );
	// SparseConnection * cone 
	// 	= new SparseConnection(poisson,neurons_e, w, sparseness, MEM );
	// SparseConnection * coni 
	// 	= new SparseConnection(poisson,neurons_i, w, sparseness, MEM );

	// This would be a solution where independend Poisson spikes
	// are used from two PoissonGroups.
	// PoissonGroup * pstim_e 
	// 	= new PoissonGroup( ne, poisson_rate*ne*sparseness );
	// IdentityConnection * ide 
	// 	= new IdentityConnection(pstim_e,neurons_e, w, MEM );
	// PoissonGroup * pstim_i  
	// 	= new PoissonGroup( ni, poisson_rate*ne*sparseness );
	// IdentityConnection * idi
	// 	= new IdentityConnection(pstim_i,neurons_i, w, MEM );
	

    logger->msg("Setting up E connections ...",PROGRESS,true);
	STDPwdConnection * con_ee 
		= new STDPwdConnection( 
				neurons_e,
				neurons_e,
				w,
				sparseness
				);
	con_ee->set_transmitter(MEM);
	con_ee->set_name("E->E");
	con_ee->set_max_weight(3*w);
	con_ee->set_alpha(2.02);
	con_ee->set_lambda(lambda); 

	SparseConnection * con_ei 
		= new SparseConnection( neurons_e,neurons_i,w,sparseness,MEM);

	logger->msg("Setting up I connections ...",PROGRESS,true);
	SparseConnection * con_ii 
		= new SparseConnection( neurons_i,neurons_i,-gamma*w,sparseness,MEM);
	SparseConnection * con_ie 
		= new SparseConnection( neurons_i,neurons_e,-gamma*w,sparseness,MEM);

	msg = "Setting up monitors ...";
	logger->msg(msg,PROGRESS,true);

	std::stringstream filename;
	filename << outputfile << "e.ras";
	SpikeMonitor * smon_e = new SpikeMonitor( neurons_e, filename.str().c_str(), nrec);

	filename.str("");
	filename.clear();
	filename << outputfile << "i.ras";
	SpikeMonitor * smon_i = new SpikeMonitor( neurons_i, filename.str().c_str(), nrec);

	// filename.str("");
	// filename.clear();
	// filename << outputfile << "syn";
	// WeightMonitor * wmon = new WeightMonitor( con_ee, filename.str() );
	// wmon->add_equally_spaced(1000);

	// filename.str("");
	// filename.clear();
	// filename << outputfile << "mem";
	// StateMonitor * smon = new StateMonitor( neurons_e, 13, "mem", filename.str() );

	RateChecker * chk = new RateChecker( neurons_e , 0.1 , 1000. , 100e-3);

	if ( !load.empty() ) {
		sys->load_network_state(load);
	}

	if ( !fwmat_ee.empty() ) con_ee->load_from_complete_file(fwmat_ee);
	if ( !fwmat_ei.empty() ) con_ei->load_from_complete_file(fwmat_ei);
	if ( !fwmat_ie.empty() ) con_ie->load_from_complete_file(fwmat_ie);
	if ( !fwmat_ii.empty() ) con_ii->load_from_complete_file(fwmat_ii);

	// con_ee->prune();
	// con_ei->prune();
	// con_ie->prune();
	// con_ii->prune();

	// logger->msg("Running sanity check ...",PROGRESS,true);
	con_ee->sanity_check();
	con_ei->sanity_check();
	con_ie->sanity_check();
	con_ii->sanity_check();

	logger->msg("Simulating ..." ,PROGRESS,true);
	if (!sys->run(simtime,true)) 
			errcode = 1;

	if ( !save.empty() ) {
		sys->save_network_state(save);
	}


	if (errcode)
		auryn_abort(errcode);


	logger->msg("Freeing ..." ,PROGRESS,true);
	auryn_free();
	return errcode;
}
