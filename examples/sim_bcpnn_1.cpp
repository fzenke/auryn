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
*
* Author: Anders Lansner, code adapted from sim_stdp
*/


/*!\file 
 *
 * \brief Example simulation of a a pre- and postsynaptic neurongroup
 * with Poisson input and pair-based additive Bcpnn.
 *
 * 
 * Output files:
 *  * prspikes.txt : Spikes of presynaptic neurons
 *  * pospikes.txt : Spikes of postsynaptic neurons
 *  * prrate.txt : Population firing rate of presynaptic neurons
 *  * porate.txt : Population firing rate of postsynaptic neurons
 *  * zi.txt : z-trace of presynaptic neuron 5.
 *  * zj.txt : z-trace of postsynaptic neuron 5.
 *  * pi.txt : p-trace of presynaptic neuron 5.
 *  * pj.txt : p-trace of postsynaptic neuron 5.
 *  * bj.txt : bias of postsynaptic neuron 5.
 *  * pij.txt : p-trace of connection between presynaptic neuron 5 and postsynaptic neuron 5.
 *  * wij.txt : weight of connection between presynaptic neuron 5 and postsynaptic neuron 5.
 *  * vmem_pr.txt : Membrane potential of presynaptic neuron 5.
 *  * vmem_po.txt : Membrane potential of postsynaptic neuron 5.
 *
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

	double simtime = 10.;

	NeuronID nbinputs = 100;
	NeuronID size = 25;
	unsigned int seed = 1;
	double kappa = 20;
	AurynFloat sparseness = 0.5;
	AurynFloat npostsyn = 100;
	AurynWeight winit = 0.025;
	bool with_bcpnn = false;
	bool nomon = false;
	int ipre = 5;
	int ipost = 5;

	double tau_pre = 20e-3; // stdp window decay (pre-post part)
	double tau_z_pr = 25e-3; // Bcpnn tau_zi decay
	double tau_z_po = 10e-3; // Bcpnn tau_zj decay
	double tau_p = 0.2;

	int errcode = 0;

    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("dir", po::value<string>(), "output directory")
            ("simtime", po::value<double>(), "simulation time")
            // ("sparseness", po::value<double>(), "sparseness")
            ("with_bcpnn", po::value<bool>(), "BcpnnConnection used")
            ("winit", po::value<double>(), "initial weight")
            ("kappa", po::value<double>(), "presynaptic firing rate")
            ("nbinputs", po::value<int>(), "number of Poisson inputs")
            ("size", po::value<int>(), "number of neurons")
            ("npostsyn", po::value<int>(), "number of synapses on postynaptic neuron")
            ("ipre", po::value<int>(), "presynaptic neuron to monitor")
            ("ipost", po::value<int>(), "postsynaptic neuron to monitor")
            ("seed", po::value<int>(), "random seed")
            ("nomon", po::value<bool>(), "if 'true' no monitorint of state variables")
			;

        po::variables_map vm;        
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);    

        if (vm.count("help")) {
            return 1;
        }

        if (vm.count("kappa")) {
			kappa = vm["kappa"].as<double>();
        } 

        if (vm.count("dir")) {
			dir = vm["dir"].as<string>();
        } 

        if (vm.count("simtime")) {
			simtime = vm["simtime"].as<double>();
        } 

        // if (vm.count("sparseness")) {
		// 	sparseness = vm["sparseness"].as<double>();
        // } 

        if (vm.count("with_bcpnn")) {
			with_bcpnn = vm["with_bcpnn"].as<bool>();
        } 

        if (vm.count("winit")) {
			winit = vm["winit"].as<double>();
        } 

        if (vm.count("nbinputs")) {
			nbinputs = vm["nbinputs"].as<int>();
        } 

        if (vm.count("size")) {
			size = vm["size"].as<int>();
        } 

        if (vm.count("npostsyn")) {
			npostsyn = vm["npostsyn"].as<int>();
        } 

        if (vm.count("ipre")) {
			ipre = vm["ipre"].as<int>();
        } 

        if (vm.count("ipost")) {
			ipost = vm["ipost"].as<int>();
        } 

        if (vm.count("seed")) {
			seed = vm["seed"].as<int>();
        } 

        if (vm.count("nomon")) {
			nomon = vm["nomon"].as<bool>();
        } 
    }
    catch(std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
        std::cerr << "Exception of unknown type!\n";
    }

	if (npostsyn>nbinputs) {
		std::cerr << "ERROR in main: npostsyn>nbinputs" << std::endl;
		exit(0);
	}

	// Init Auryn
	auryn_init(ac, av, dir, "sim_bcpnn");
	sys->set_master_seed(seed);

	logger->set_logfile_loglevel(EVERYTHING);

	sparseness = (float)npostsyn/nbinputs;
	if (sys->mpi_rank()==0)
		std::cerr << "sparseness = " << sparseness << std::endl;

	// Neurons
	PoissonGroup * poisson = new PoissonGroup(nbinputs,kappa);

	float refractory_period = 5e-3; // Refractory time of TIF neurons = 0.005 sec

	TIFGroup * poneurons = new TIFGroup(size);

	SparseConnection *sp_con = new SparseConnection(poisson,poneurons,winit,sparseness);

	BcpnnConnection *bcpnn_con;
	if (with_bcpnn) {
		bcpnn_con = new BcpnnConnection(poisson,poneurons,0,sparseness,
										tau_pre,tau_z_pr,tau_z_po,tau_p,refractory_period);
		bcpnn_con->set_wgain(1e-16);
		bcpnn_con->set_bgain(1e-16);
	}

	// Monitors

	if (not nomon) {

		// point online rate monitor to neurons
		// sys->set_online_rate_monitor_id(poneurons->get_uid());

		if (nbinputs<ipre) {
			logger->msg("ERROR in main: nbinputs<ipre",PROGRESS,true);
			auryn_abort(4711);
		}
		if (size<ipost) {
			logger->msg("ERROR in main: size<ipost",PROGRESS,true);
			auryn_abort(4712);
		}

		if (with_bcpnn) {
			StateMonitor * zi_mon = new StateMonitor(poisson->get_pre_trace(tau_z_pr),ipre,sys->fn("zi"));
			StateMonitor * zj_mon = new StateMonitor(poneurons->get_post_trace(tau_z_po),ipost,sys->fn("zj"));

			StateMonitor * pi_mon = new StateMonitor(poisson->get_state_vector("tr_p_pre"),
													 ipre,sys->fn("pi"));
			StateMonitor * pj_mon = new StateMonitor(poneurons->get_state_vector("tr_p_post"),
													 ipost,sys->fn("pj"));
			StateMonitor * bias_mon = new StateMonitor(poneurons->get_state_vector("w"),ipost,sys->fn("bj"));

			// Record individual synaptic weights (sample every 10 ms)
			WeightMonitor * pijmon = new WeightMonitor(bcpnn_con,ipre,ipost,sys->fn("pij"),0.01,SINGLE,1);
			WeightMonitor * wijmon = new WeightMonitor(bcpnn_con,ipre,ipost,sys->fn("wij"),0.01,SINGLE,0);
		}

		// // Record spikes
		SpikeMonitor * smon_pr = new SpikeMonitor(poisson,sys->fn("prspikes"));
		SpikeMonitor * smon_po = new SpikeMonitor(poneurons,sys->fn("pospikes"));

		VoltageMonitor * vmon_po = new VoltageMonitor(poneurons,ipost,sys->fn("vmem_po"));
	
		// // Record input firing rate (sample every 1s)
		PopulationRateMonitor * pmon_in = new PopulationRateMonitor(poisson,sys->fn("prrate"), 0.005 );
	
		// // Record output firing rate (sample every 1s)
		PopulationRateMonitor * pmon_ut = new PopulationRateMonitor(poneurons,sys->fn("porate"), 0.005 );
	
	}

	MPI_Barrier(MPI::COMM_WORLD);

	double start = MPI::Wtime();

	// Run simulation
	if (!sys->run(simtime)) errcode = 1;

	int nsyn,gnsyn;
	/* Get total number of non-plastic synapses */
	nsyn = sp_con->get_nonzero();
	MPI_Gather(&nsyn,1,MPI_INT,&gnsyn,1,MPI_INT,0,*sys->get_com());

	int bcpnn_nsyn,bcpnn_gnsyn;
	if (with_bcpnn) {
		/* Get total number of bcpnn-synapses */
		bcpnn_nsyn = bcpnn_con->get_nonzero();
		MPI_Gather(&bcpnn_nsyn,1,MPI_INT,&bcpnn_gnsyn,1,MPI_INT,0,*sys->get_com());
	}
	
	if (sys->mpi_rank()==0) {
		std::cerr << "Execution time = " << MPI::Wtime() - start << " sec\n";

		std::cerr << "N:o non-plastic weights = " << gnsyn << std::endl;
		
		if (with_bcpnn)
			std::cerr << "N:o bcpnn weights = " << bcpnn_gnsyn << std::endl;
	}

	// Close Auryn
	logger->msg("Freeing ...",PROGRESS,true);
	auryn_free();
	return errcode;
}
