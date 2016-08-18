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

/*!\file 
 *
 * \brief This simulation illustrates inhibitory synaptic plasticity as modeled
 * in Vogels et al. (2011) in a larger 200k cell network
 *
 * This simulation illustrates inhibitory synapitc plasticity as modeled in our
 * paper: Vogels, T.P., Sprekeler, H., Zenke, F., Clopath, C., and Gerstner, W.
 * (2011). Inhibitory Plasticity Balances Excitation and Inhibition in Sensory
 * Pathways and Memory Networks. Science 334, 1569–1573.
 * 
 * Note that this is a parallel implementation of this network which requires a
 * larger axonal delay than in the original paper. In this example the delay is
 * 0.8ms which corresponds to Auryn's MINDELAY.
 *
 * */


#define NE 160000 //!< Number of excitatory neurons
#define NI 40000  //!< Number of inhibitory neurons

using namespace auryn;
namespace po = boost::program_options;

int main(int ac, char* av[]) 
{
	double w = 0.0035;
	double w_ext = w ;
	double gamma = 10. ;
	double wmax = 1000*gamma*w;

	double sparseness = 0.02;

	double eta = 1e-5 ;
	double kappa = 3. ;
	double tau_stdp = 20e-3 ;
	bool stdp_active = true;
	bool poisson_stim = false;
	bool record_voltage = false;
	double winh = -1;
	double wei = 1;
	double chi = 2.;
	double lambda = 1.;
	double bg_current = 2.0e-2;//1.01e-2;

	double sparseness_afferents = 0.05;

	bool quiet = false;
	bool save = false;
	double simtime = 1000. ;
	NeuronID record_neuron = 30;
	// handle command line options
	
	string simname = "isp_big";
	string infilename = "";
	string patfilename = "";
	string outputfile = "";
	string dir = ".";
	string stimfile = "";
	string strbuf ;
	string msg;

	int errcode = 0;

    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("quiet", "quiet mode")
            ("save", "save state at the end for loading")
            ("voltage", "activate voltage monitor")
            ("load", po::value<string>(), "input weight matrix")
            ("pat", po::value<string>(), "pattern file")
            ("dir", po::value<string>(), "output dir")
            ("stimfile", po::value<string>(), "stimulus file")
            ("eta", po::value<double>(), "learning rate")
            ("kappa", po::value<double>(), "target rate")
            ("simtime", po::value<double>(), "simulation time")
            ("active", po::value<bool>(), "toggle learning")
            ("poisson", po::value<bool>(), "toggle poisson stimulus")
            ("winh", po::value<double>(), "inhibitory weight multiplier")
            ("wei", po::value<double>(), "ei weight multiplier")
            ("chi", po::value<double>(), "chi recall parameter")
            ("lambda", po::value<double>(), "lambda storage parameter")
            ("sparseness", po::value<double>(), "sparseness of connections")
        ;

        po::variables_map vm;        
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);    

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 1;
        }

        if (vm.count("quiet")) {
			quiet = true;
        } 

        if (vm.count("save")) {
			save = true;
        } 

        if (vm.count("voltage")) {
			record_voltage = true;
        } 

        if (vm.count("load")) {
			infilename = vm["load"].as<string>();
        } 

        if (vm.count("pat")) {
			patfilename = vm["pat"].as<string>();
        } 

        if (vm.count("dir")) {
			dir = vm["dir"].as<string>();
        } 

        if (vm.count("stimfile")) {
			stimfile = vm["stimfile"].as<string>();
        } 

        if (vm.count("eta")) {
			eta = vm["eta"].as<double>();
        } 

        if (vm.count("kappa")) {
			kappa = vm["kappa"].as<double>();
        } 

        if (vm.count("simtime")) {
			simtime = vm["simtime"].as<double>();
        } 

        if (vm.count("active")) {
			stdp_active = vm["active"].as<bool>();
        } 

        if (vm.count("poisson")) {
			poisson_stim = vm["poisson"].as<bool>();
        } 


        if (vm.count("winh")) {
			winh = vm["winh"].as<double>();
        } 

        if (vm.count("wei")) {
			wei = vm["wei"].as<double>();
        } 

        if (vm.count("chi")) {
			chi = vm["chi"].as<double>();
        } 

        if (vm.count("lambda")) {
			lambda = vm["lambda"].as<double>();
        } 

        if (vm.count("sparseness")) {
			sparseness = vm["sparseness"].as<double>();
        } 

    }
    catch(std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
        std::cerr << "Exception of unknown type!\n";
    }

	// BEGIN Global definitions
	auryn_init( ac, av );
	sys->set_simulation_name(simname);
	// END Global definitions


	logger->msg("Setting up neuron groups ...",PROGRESS,true);
	TIFGroup * neurons_e = new TIFGroup(NE);
	TIFGroup * neurons_i = new TIFGroup(NI);
	neurons_e->random_mem(-60e-3,5e-3);
	// neurons_e->random_nmda(2,1);
	neurons_i->random_mem(-60e-3,5e-3);


	logger->msg("Setting up I connections ...",PROGRESS,true);
	SparseConnection * con_ei = new SparseConnection(neurons_e,neurons_i,wei*w,sparseness,GLUT,"EI connection");
	// con_ei->random_data(wei*w,wei*0.3*w);
	SparseConnection * con_ii = new SparseConnection(neurons_i,neurons_i,gamma*w,sparseness,GABA,"II connection");
	// con_ii->random_data(gamma*w,gamma*0.3*w);

	IdentityConnection * con_exte;
	if ( !stimfile.empty() && poisson_stim==true) {
		logger->msg("Setting up StimulusGroup ...",PROGRESS,true);
		StimulusGroup * stimgroup = new StimulusGroup(NE,stimfile,"",SEQUENTIAL,100);
		con_exte = new IdentityConnection(stimgroup,neurons_e,0.2,GLUT,"external input");
		stimgroup->set_mean_on_period(0.5);
		stimgroup->set_mean_off_period(20.);

		logger->msg("Enabling stimulus monitors ...",PROGRESS,true);
		strbuf = outputfile;
		strbuf += ".stimact";
        PatternMonitor * patmon = new PatternMonitor( stimgroup, strbuf.c_str(),patfilename.c_str());
	}

	logger->msg("Setting up E connections ...",PROGRESS,true);
	
	SparseConnection * con_ee 
		= new SparseConnection(neurons_e,neurons_e,w,sparseness,GLUT,"EE connection");

	SparseConnection * con_ie 
		= new SparseConnection(neurons_i,neurons_e,gamma*w,sparseness,GABA,"IE connection");

	// SymmetricSTDPConnection * con_ie 
	// 	= new SymmetricSTDPConnection(
	// 			neurons_i,neurons_e,
	// 			gamma*w,sparseness,
	// 			gamma*eta,kappa,tau_stdp,wmax,
	// 			GABA,"IE connection");
	
	if (!infilename.empty()) {
		logger->msg("Loading previous network state...",PROGRESS,true);
		sys->load_network_state(infilename);
	} 

	if (winh>=0)
		con_ie->set_all(winh);

	if (!patfilename.empty()) {
		logger->msg("Loading patterns ...",PROGRESS,true);
		con_ee->load_patterns(patfilename,lambda*w);

		logger->msg("Enabling pattern monitors ...",PROGRESS,true);
		strbuf = outputfile;
		strbuf += ".patact";
        PatternMonitor * patmon = new PatternMonitor(neurons_e, strbuf.c_str(),patfilename.c_str());
	}

	logger->msg("Setting up monitors ...",PROGRESS,true);
	strbuf = outputfile;
	strbuf += ".e.ras";
	SpikeMonitor * smon_e = new SpikeMonitor( neurons_e, strbuf );

	strbuf = outputfile;
	strbuf += ".i.ras";
	SpikeMonitor * smon_i = new SpikeMonitor( neurons_i, strbuf );


	if ( record_voltage ) {
		strbuf = outputfile;
		strbuf += ".mem";
			VoltageMonitor * vmon = new VoltageMonitor( neurons_e, record_neuron, strbuf.c_str() );

		strbuf = outputfile;
		strbuf += ".ampa";
			StateMonitor * amon = new StateMonitor( neurons_e, record_neuron, "g_ampa", strbuf.c_str() );

		strbuf = outputfile;
		strbuf += ".gaba";
			StateMonitor * gmon = new StateMonitor( neurons_e, record_neuron, "g_gaba", strbuf.c_str() );
	}

	RateChecker * chk = new RateChecker( neurons_e , 0.001 , 1000. , 100e-3);

	for ( int j = 0; j < (NE) ; j++ ) {
	  neurons_e->set_bg_current(j,bg_current);
	}

	for ( int j = 0; j < (NI) ; j++ ) {
	  neurons_i->set_bg_current(j,bg_current);
	}
	
	// We don't have an efficient way of setting input
	// currents from a file. So we do int manually here.
	if (!stimfile.empty() && poisson_stim==false) {
		logger->msg("Loading pattern ..." ,PROGRESS,true);
		char buffer[256];
		std::ifstream fin(stimfile.c_str());
		if (!fin) {
			std::cout << "There was a problem opening file "
			<< stimfile
			<< " for reading."
			<< std::endl;
			logger->msg("There was a problem opening file." ,ERROR,true);
			return 1;
		}

		while ( fin.getline(buffer,256) ) { 
			std::stringstream iss( buffer );
			NeuronID id;
			iss >> id;

			if (poisson_stim==false) {
				neurons_e->set_bg_current(id,chi*bg_current);
			}

		} 
		fin.close();
		
		logger->msg("Running ..." ,PROGRESS,true);
		sys->run(2);

		logger->msg("Resetting ..." ,PROGRESS,true);
		for ( int j = 0; j < (NE) ; j++ ) {
		  neurons_e->set_bg_current(j,bg_current);
		}

	}

	logger->msg("Simulating ..." ,PROGRESS,true);
	// con_ie->stdp_active = stdp_active;

	if (!sys->run(simtime,true)) 
			errcode = 1;

	if (save) {
		logger->msg("Saving network state ..." ,PROGRESS,true);
		sys->save_network_state(outputfile);
	}

	if (errcode)
		auryn_abort(errcode);

	logger->msg("Freeing ..." ,PROGRESS,true);
	auryn_free();

	return errcode;
}
