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

/*! \brief Example simulation that simulates a balanced network with 
 * triplet STDP with homeostatic scaling 
 *
 * This simulation code is based on the code used in Zenke, F., Hennequin, G.,
 * and Gerstner, W. (2013). Synaptic Plasticity in Neural Networks Needs
 * Homeostasis with a Fast Rate Detector. PLoS Comput Biol 9, e1003330.
 * */

#include "auryn.h"

#define N_REC_WEIGHTS 5000
#define NE 20000
#define NI 5000 


using namespace auryn;

namespace po = boost::program_options;

int main(int ac, char* av[]) 
{

	double w = 0.16;
	double w_ext = w;
	double wmax = 1.0;

	double w_ee = w;
	double w_ei = w;

	double gamma = 1.0;
	double w_ie = gamma;
	double w_ii = gamma;


	double sparseness = 0.05;
	double kappa = 3;

	bool quiet = false;
	bool wmatdump = false;
	double simtime = 3600.;
	double stimtime = simtime;
	double wmat_interval = 600.;
	double wstim = 0.1;

	double stimfreq = 50;

	bool corr = false;
	bool adapt = false;
	bool noisyweights = false;
	bool fast = false;

	double tau_hom = 50.;
	double eta = 1;
	double beta_scaling = 3600; // scaling time constant

	int n_strengthen = 0;

	string dir = "/tmp";
	string stimfile = "";
	string label = "";
	string infilename = "";

	const char * file_prefix = "bg_scaling";
	char strbuf [255];
	string msg;

	int errcode = 0;

    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("quiet", "quiet mode")
            ("load", po::value<string>(), "input weight matrix")
            ("wmat", "wmat dump mode")
            ("eta", po::value<double>(), "learning rate")
            ("scaling", po::value<double>(), "learning rate")
            ("tau_hom", po::value<double>(), "homeostatic time constant")
            ("kappa", po::value<double>(), "target rate")
            ("simtime", po::value<double>(), "simulation time")
            ("dir", po::value<string>(), "output dir")
            ("label", po::value<string>(), "output label")
            ("we", po::value<double>(), "we")
            ("strengthen", po::value<int>(), "connections to strengthen by 10")
            ("stimfile", po::value<string>(), "stimulus ras file")
            ("wstim", po::value<double>(), "weight of stimulus connections")
            ("stimtime", po::value<double>(), "time of stimulus on")
            ("adapt", "adapting excitatory neurons")
            ("corr", "add correlated inputs")
            ("stimfreq", po::value<double>(), "CorrelatedPoissonGroup frequency default = 100")
            ("noisyweights", "enables noisyweights for mean field checks")
            ("fast", "turn off some of the monitors to run faster")
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

        if (vm.count("load")) {
            std::cout << "load from matrix " 
                 << vm["load"].as<string>() << ".\n";
			infilename = vm["load"].as<string>();
        } 

        if (vm.count("wmat")) {
			wmatdump = true;
			std::cout << "wmat dump mode" << std::endl;
        } 

        if (vm.count("eta")) {
            std::cout << "eta set to " 
                 << vm["eta"].as<double>() << ".\n";
			eta = vm["eta"].as<double>();
        } 

        if (vm.count("scaling")) {
            std::cout << "scaling set to " 
                 << vm["scaling"].as<double>() << ".\n";
			beta_scaling = vm["scaling"].as<double>();
        } 

        if (vm.count("tau_hom")) {
            std::cout << "tau_hom set to " 
                 << vm["tau_hom"].as<double>() << ".\n";
			tau_hom = vm["tau_hom"].as<double>();
        } 

        if (vm.count("kappa")) {
            std::cout << "kappa set to " 
                 << vm["kappa"].as<double>() << ".\n";
			kappa = vm["kappa"].as<double>();
        } 

        if (vm.count("simtime")) {
            std::cout << "simtime set to " 
                 << vm["simtime"].as<double>() << ".\n";
			simtime = vm["simtime"].as<double>();
			stimtime = simtime;
        } 

        if (vm.count("corr")) {
            std::cout << "enabling corr " << std::endl;
			corr = true;
        } 

        if (vm.count("stimfreq")) {
            std::cout << "stimfreq set to " 
                 << vm["stimfreq"].as<double>() << ".\n";
			stimfreq = vm["stimfreq"].as<double>();
        } 

        if (vm.count("dir")) {
            std::cout << "dir set to " 
                 << vm["dir"].as<string>() << ".\n";
			dir = vm["dir"].as<string>();
        } 

        if (vm.count("label")) {
            std::cout << "label set to " 
                 << vm["label"].as<string>() << ".\n";
			label = vm["label"].as<string>();
        } 

        if (vm.count("we")) {
            std::cout << "we set to " 
                 << vm["we"].as<double>() << ".\n";
			w_ee = vm["we"].as<double>();
        } 

        if (vm.count("strengthen")) {
            std::cout << "strengthen set to " 
                 << vm["strengthen"].as<int>() << ".\n";
			n_strengthen = vm["strengthen"].as<int>();
        } 

        if (vm.count("stimfile")) {
            std::cout << "stimfile set to " 
                 << vm["stimfile"].as<string>() << ".\n";
			stimfile = vm["stimfile"].as<string>();
        } 

        if (vm.count("wstim")) {
            std::cout << "wstim set to " 
                 << vm["wstim"].as<double>() << ".\n";
			wstim = vm["wstim"].as<double>();
        } 

        if (vm.count("stimtime")) {
            std::cout << "stimtime set to " 
                 << vm["stimtime"].as<double>() << ".\n";
			stimtime = vm["stimtime"].as<double>();
        } 

        if (vm.count("adapt")) {
            std::cout << "adaptation on " << std::endl;
			adapt = true;
        } 

        if (vm.count("noisyweights")) {
            std::cout << "noisyweights on " << std::endl;
			noisyweights = true;
        } 

        if (vm.count("fast")) {
            std::cout << "fast on " << std::endl;
			fast = true;
        } 
    }
    catch(std::exception& e) {
		std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
		std::cerr << "Exception of unknown type!\n";
    }


	// double primetime = 3.0/(beta_scaling*eta*1.24488e-5);
	double primetime = 3.0*tau_hom;


	auryn_init(ac, av);

	if (!infilename.empty()) {
		std::stringstream iss;
		iss << infilename << "." << sys->mpi_rank();
		infilename = iss.str();
	}

	logger->msg("Setting up neuron groups ...",PROGRESS,true);


	NeuronGroup * neurons_e;
	if ( adapt )
		neurons_e = new AIFGroup(20000);
	else
		neurons_e = new IFGroup(20000);
	IFGroup * neurons_i = new IFGroup(5000);

	// initialize membranes
	neurons_i->set_tau_mem(10e-3);
	neurons_e->random_mem(-60e-3,10e-3);
	neurons_i->random_mem(-60e-3,10e-3);


	SpikingGroup * poisson = new PoissonGroup(2500,2);
	// SparseConnection * con_exte = new SparseConnection(poisson, neurons_e, w_ext, sparseness, GLUT);
	TripletScalingConnection * con_exte = new TripletScalingConnection(poisson, neurons_e, w_ext, sparseness, tau_hom, eta, kappa, beta_scaling, wmax, GLUT);



	msg = "Setting up I connections ...";
	logger->msg(msg,PROGRESS,true);
	SparseConnection * con_ie = new SparseConnection(neurons_i,neurons_e,
			w_ie,sparseness,GABA);
	SparseConnection * con_ii = new SparseConnection(neurons_i,neurons_i,
			w_ii,sparseness,GABA);

	msg =  "Setting up E connections ...";
	logger->msg(msg,PROGRESS,true);
	SparseConnection * con_ei = new SparseConnection(neurons_e,neurons_i,
			w_ei, sparseness,GLUT);

	TripletScalingConnection * con_ee;

	con_ee = new TripletScalingConnection(neurons_e, neurons_e,
		w_ee, sparseness, tau_hom, eta, kappa, beta_scaling, wmax, GLUT);
	if ( noisyweights )
		con_ee->random_data(w_ee,w_ee/4);
	for ( int i = 0 ; i < n_strengthen ; ++i ) {
		con_ee->set_data(i,i*(wmax/n_strengthen));
	}

	msg = "Initializing traces ...";
	logger->msg(msg,PROGRESS,true);
	con_ee->set_hom_trace(kappa);

	// TODO
	// con_ee->w->set_col(0,2*w_ee);

	msg = "Setting up monitors ...";
	logger->msg(msg,PROGRESS,true);

	if (wmatdump) {
		sprintf(strbuf, "%s/%s_e%.2et%.2f%s.%d.weight", dir.c_str(), file_prefix, beta_scaling, tau_hom, label.c_str(), sys->mpi_rank());
		WeightMatrixMonitor * wmatmon = new WeightMatrixMonitor( con_ee, strbuf , wmat_interval );
	}

	if ( !fast ) {
		sprintf(strbuf, "%s/%s_e%.2et%.2f%s.%d.%c.spk", dir.c_str(), file_prefix, beta_scaling, tau_hom, label.c_str(), sys->mpi_rank(), 'e');
		BinarySpikeMonitor * smon_e = new BinarySpikeMonitor( neurons_e, strbuf , 2500);

		sprintf(strbuf, "%s/%s_e%.2et%.2f%s.%d.%c.prate", dir.c_str(), file_prefix, beta_scaling, tau_hom, label.c_str(), sys->mpi_rank(), 'e');
		PopulationRateMonitor * pmon_e = new PopulationRateMonitor( neurons_e, strbuf, 0.2 );
	}

	sprintf(strbuf, "%s/%s_e%.2et%.2f%s.%d.syn", dir.c_str(), file_prefix, beta_scaling, tau_hom, label.c_str(), sys->mpi_rank());
	WeightMonitor * wmon = new WeightMonitor( con_ee, strbuf, 0.1 ); 
	wmon->add_equally_spaced(20);

	RateChecker * chk = new RateChecker( neurons_e , -0.1 , 20.*kappa , 100e-3);



	FileInputGroup * filegroup;
	SparseConnection * con_stim;
	if (!stimfile.empty()) {
		msg = "Setting up stimulus ...";
		logger->msg(msg,PROGRESS,true);
		filegroup = new FileInputGroup(2000,stimfile.c_str(),true,1);
		con_stim = new SparseConnection( filegroup, neurons_e, GLUT);
		con_stim->set_name("Stimulus Connection");
		con_stim->allocate_manually(4*500*100*sparseness);
		con_stim->connect_block_random(wstim, sparseness, 0, 500, 0, 100);
		con_stim->connect_block_random(wstim, sparseness, 500, 1000, 100, 200);
		con_stim->connect_block_random(wstim, sparseness, 1000, 1500, 200, 300);
		con_stim->connect_block_random(wstim, sparseness, 1500, 2000, 300, 400);
		con_stim->finalize();

		logger->msg("Saving weight matrix ...",PROGRESS,true);
		sprintf(strbuf, "%s/%s_e%.2et%.2f%s.%d_stim.wmat", dir.c_str(), file_prefix, beta_scaling, tau_hom, label.c_str(), sys->mpi_rank());
		con_stim->write_to_file(strbuf);
	}


    // set up correlated input
	const NeuronID size = 500;
	const NeuronID psize = 100;
	const NeuronID ngroups = 3;

	std::stringstream oss;
	oss << " Activating correlated input ... ";
	logger->msg(oss.str(),PROGRESS,true);

	CorrelatedPoissonGroup * corr_e = new CorrelatedPoissonGroup(psize*ngroups,stimfreq,psize,5);
	const double ampl = 5.0;
	corr_e->set_amplitude(ampl);
	corr_e->set_target_amplitude(ampl);
	corr_e->set_timescale(100e-3);
	// corr->set_offset(2);

	// SparseConnection * con_corr_e = new SparseConnection(corr_e,neurons_e,corr_file.c_str(),GLUT);
	TripletScalingConnection * con_corr_e = new TripletScalingConnection(corr_e, neurons_e, w, sparseness, tau_hom,eta,kappa,beta_scaling,wmax,GLUT);
	con_corr_e->set_all(0.0);
	for ( int i = 0 ; i < ngroups ; ++i ) {
		con_corr_e->set_block( i*psize, (i+1)*psize, i*psize, (i+1)*psize, w_ext );
		con_exte->set_block( i*psize, (i+1)*psize, i*psize, (i+1)*psize, 1e-3 );
	}


	sprintf(strbuf, "%s/%s_e%.2et%.2f%s.%d.c.syn", dir.c_str(), file_prefix, beta_scaling, tau_hom, label.c_str(), sys->mpi_rank());
	WeightMonitor * wmonc = new WeightMonitor( con_corr_e, strbuf, 0.1 ); 
	wmonc->add_equally_spaced(20);

	// disabling external random input
	// con_exte->set_block(0,2500,0,size,0.0);

	// set up Weight monitor

	sprintf(strbuf, "%s/%s_e%.2et%.2f%s.%d.%c.spk", dir.c_str(), file_prefix, beta_scaling, tau_hom, label.c_str(), sys->mpi_rank(), 'c');
	BinarySpikeMonitor * smon_c = new BinarySpikeMonitor( corr_e, strbuf , size );


	if (primetime>0) {
		msg = "Priming ...";
		// con_ee->set_beta(400);
		logger->msg(msg,PROGRESS,true);
		con_ee->stdp_active = false;
		con_exte->stdp_active = false;
		con_corr_e->stdp_active = false;
		sys->run(primetime,true);
	}

	logger->msg("Simulating ...",PROGRESS,true);
	con_ee->set_beta(beta_scaling);
	con_exte->set_beta(beta_scaling);
	con_corr_e->set_beta(beta_scaling);
	if (eta > 0) {
		con_ee->stdp_active = true;
		con_exte->stdp_active = true;
		con_corr_e->stdp_active = true;
	}

	if (!sys->run(simtime,true)) 
			errcode = 1;



	logger->msg("Saving neurons state ...",PROGRESS,true);
	sprintf(strbuf, "%s/%s_e%.2et%.2f%s.%d.e.nstate", dir.c_str(), file_prefix, beta_scaling, tau_hom, label.c_str(), sys->mpi_rank());
	neurons_e->write_to_file(strbuf);
	sprintf(strbuf, "%s/%s_e%.2et%.2f%s.%d.i.nstate", dir.c_str(), file_prefix, beta_scaling, tau_hom, label.c_str(), sys->mpi_rank());
	neurons_i->write_to_file(strbuf);

	logger->msg("Saving weight matrix ...",PROGRESS,true);
	sprintf(strbuf, "%s/%s_e%.2et%.2f%s.%d.wmat", dir.c_str(), file_prefix, beta_scaling, tau_hom, label.c_str(), sys->mpi_rank());
	con_ee->write_to_file(strbuf);

	// save lifetime
	sprintf(strbuf, "%s/%s_e%.2et%.2f%s.%d.lifetime", dir.c_str(), file_prefix, beta_scaling, tau_hom, label.c_str(), sys->mpi_rank());
	std::ofstream killfile;
	killfile.open(strbuf);
	killfile << sys->get_time()-primetime << std::endl;
	killfile.close();

	if (errcode)
		auryn_abort(errcode);

	logger->msg("Freeing ...",PROGRESS,true);
	auryn_free();
	return errcode;
}
