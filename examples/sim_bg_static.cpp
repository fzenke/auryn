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

	NeuronID ne = 20000;
	NeuronID ni = ne/4;

	double sparseness = 0.05;

	bool quiet = false;
	bool scaling = false;
	bool wmatdump = false;
	double tau_chk = 100e-3;
	double simtime = 3600.;
	double stimtime = simtime;
	double wmat_interval = 600.;

	double ampa_nmda_ratio = 1.0;
	double wstim = 0.1;

	NeuronID psize = 200;
	NeuronID hsize = 100;
	NeuronID offset = 0;

	string patfile = "";
	string prefile = "";
	string currentfile = "";

	double stimfreq = 10;

	int  plen = 3;
	int  hlen = 3;

	double ampl = 1.0;
	bool recall = false;
	bool decay = false;
	bool adapt = false;
	bool noisyweights = false;
	bool switchweights = false;
	bool ei_plastic = false;

	double bg_rate = 2;
	bool fast = false;
	AurynWeight wdecay = w;
	double tau_decay = 3600.;


	string dir = ".";
	string stimfile = "";
	string infilename = "";

	const char * file_prefix = "bg_static";
	char strbuf [255];
	string msg;

	int errcode = 0;

    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("quiet", "quiet mode")
            ("bgrate", po::value<double>(), "PoissonGroup external firing rate")
            ("sparseness", po::value<double>(), "overall network sparseness")
            ("simtime", po::value<double>(), "simulation time")
            ("dir", po::value<string>(), "output dir")
            ("ne", po::value<int>(), "no of exc units")
            ("adapt", "adapting excitatory neurons")
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

        if (vm.count("scaling")) {
			scaling = true;
        } 

        if (vm.count("load")) {
            std::cout << "load from matrix " 
                 << vm["load"].as<string>() << ".\n";
			infilename = vm["load"].as<string>();
        } 

        if (vm.count("patfile")) {
            std::cout << "PatternFile is " 
                 << vm["patfile"].as<string>() << ".\n";
			patfile = vm["patfile"].as<string>();
        } 

        if (vm.count("prefile")) {
            std::cout << "Preload patternfile is " 
                 << vm["prefile"].as<string>() << ".\n";
			prefile = vm["prefile"].as<string>();
        } 

        if (vm.count("wmat")) {
			wmatdump = true;
			std::cout << "wmat dump mode" << std::endl;
        } 

        if (vm.count("bgrate")) {
            std::cout << "bgrate set to " 
                 << vm["bgrate"].as<double>() << ".\n";
			bg_rate = vm["bgrate"].as<double>();
        } 

        if (vm.count("sparseness")) {
            std::cout << "sparseness set to " 
                 << vm["sparseness"].as<double>() << ".\n";
			sparseness = vm["sparseness"].as<double>();
        } 

        if (vm.count("simtime")) {
            std::cout << "simtime set to " 
                 << vm["simtime"].as<double>() << ".\n";
			simtime = vm["simtime"].as<double>();
			stimtime = simtime;
        } 

        if (vm.count("dir")) {
            std::cout << "dir set to " 
                 << vm["dir"].as<string>() << ".\n";
			dir = vm["dir"].as<string>();
        } 

        if (vm.count("wee")) {
            std::cout << "wee set to " 
                 << vm["wee"].as<double>() << ".\n";
			w_ee = vm["wee"].as<double>();
        } 

        if (vm.count("wei")) {
            std::cout << "wei set to " 
                 << vm["wei"].as<double>() << ".\n";
			w_ei = vm["wei"].as<double>();
        } 

        if (vm.count("wie")) {
            std::cout << "wie set to " 
                 << vm["wie"].as<double>() << ".\n";
			w_ie = vm["wie"].as<double>();
        } 

        if (vm.count("wii")) {
            std::cout << "wii set to " 
                 << vm["wii"].as<double>() << ".\n";
			w_ii = vm["wii"].as<double>();
        } 

        if (vm.count("ampa")) {
            std::cout << "ampa set to " 
                 << vm["ampa"].as<double>() << ".\n";
			ampa_nmda_ratio = vm["ampa"].as<double>();
        } 


        if (vm.count("ne")) {
            std::cout << "ne set to " 
                 << vm["ne"].as<int>() << ".\n";
			ne = vm["ne"].as<int>();
			ni = ne/4;
        } 

        if (vm.count("stimfile")) {
            std::cout << "stimfile set to " 
                 << vm["stimfile"].as<string>() << ".\n";
			stimfile = vm["stimfile"].as<string>();
        } 

        if (vm.count("chk")) {
            std::cout << "chk set to " 
                 << vm["chk"].as<double>() << ".\n";
			tau_chk = vm["chk"].as<double>();
        } 

        if (vm.count("adapt")) {
            std::cout << "adaptation on " << std::endl;
			adapt = true;
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


	auryn_init(ac, av);

	if (!infilename.empty()) {
		std::stringstream iss;
		iss << infilename << "." << sys->mpi_rank();
		infilename = iss.str();
	}

	logger->msg("Setting up neuron groups ...",PROGRESS,true);


	NeuronGroup * neurons_e;
	if ( adapt ) {
		neurons_e = new AIFGroup(ne);
		((AIFGroup*)neurons_e)->set_ampa_nmda_ratio(ampa_nmda_ratio);
		((AIFGroup*)neurons_e)->dg_adapt1=1.0;
	} else {
		neurons_e = new IFGroup(ne);
		((IFGroup*)neurons_e)->set_ampa_nmda_ratio(ampa_nmda_ratio);
	}
	IFGroup * neurons_i = new IFGroup(ni);

	// initialize membranes
	neurons_i->set_tau_mem(10e-3);
	neurons_e->random_mem(-60e-3,10e-3);
	neurons_i->random_mem(-60e-3,10e-3);

	((IFGroup*)neurons_i)->set_ampa_nmda_ratio(ampa_nmda_ratio);


	SpikingGroup * poisson = new PoissonGroup(2500,bg_rate);
	SparseConnection * con_exte = new SparseConnection(poisson, neurons_e, w_ext, sparseness, GLUT);

	msg = "Setting up I connections ...";
	logger->msg(msg,PROGRESS,true);
	SparseConnection * con_ie = new SparseConnection(neurons_i,neurons_e,
			w_ie,sparseness,GABA);
	SparseConnection * con_ii = new SparseConnection(neurons_i,neurons_i,
			w_ii,sparseness,GABA);

	msg =  "Setting up E connections ...";
	logger->msg(msg,PROGRESS,true);
	SparseConnection * con_ei = new SparseConnection(neurons_e,neurons_i,
			w_ee,sparseness,GLUT);
	SparseConnection * con_ee = new SparseConnection(neurons_e,neurons_e,
			w_ei,sparseness,GLUT);

	msg = "Setting up monitors ...";
	logger->msg(msg,PROGRESS,true);


	if ( !fast ) {
		sprintf(strbuf, "%s/bg_static.%d.ras", dir.c_str(), sys->mpi_rank());
		SpikeMonitor * smon_e = new SpikeMonitor( neurons_e, strbuf , 2500);
	}

	sprintf(strbuf, "%s/bg_static.%d.prate", dir.c_str(), sys->mpi_rank());
	PopulationRateMonitor * pmon_e = new PopulationRateMonitor( neurons_e, strbuf, 1.0 );

	sprintf(strbuf, "%s/bg_static.%d.rt", dir.c_str(), sys->mpi_rank());
	RealTimeMonitor * rtmon = new RealTimeMonitor( strbuf );

	RateChecker * chk = new RateChecker( neurons_e , 0.1 , 20. , tau_chk);

	logger->msg("Simulating ...",PROGRESS,true);

	if (!sys->run(simtime,true)) 
			errcode = 1;


	if (errcode) {
		auryn_abort(errcode);
	}

	logger->msg("Freeing ...",PROGRESS,true);
	auryn_free();

	return errcode;
}
