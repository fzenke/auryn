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

#define NE 20000
#define NI 20000/4

using namespace auryn;

namespace po = boost::program_options;
namespace mpi = boost::mpi;

int main(int ac,char *av[]) {
	string dir = ".";

	string infilename = "";
	string ratemodfile = "./ratemod.dat";
	string simname = "dense";
	string strbuf ;
	string msg;

	double w = 0.2;
	double wext = 0.22;
	double gamma = 5;
	double sparseness = 0.1;
	double simtime = 1000.;

	double tau_stdp = 20e-3;

	double eta = 0;
	double wmax = 10*gamma*w;
	double kappa = 0.2;

	int errcode = 0;

	

    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("simtime", po::value<double>(), "simulation time in seconds")
            ("dir", po::value<string>(), "dir for output files")
            ("load", po::value<string>(), "basename to load network from")
            ("ratefile", po::value<string>(), "file containing the timeseries with the rate modulation")
        ;

        po::variables_map vm;        
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);    

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 1;
        }

        if (vm.count("load")) {
			infilename = vm["load"].as<string>();
        } 

        if (vm.count("simtime")) {
			simtime = vm["simtime"].as<double>();
        } 

        if (vm.count("dir")) {
			dir = vm["dir"].as<string>();
        } 

        if (vm.count("ratefile")) {
			ratemodfile = vm["ratefile"].as<string>();
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
	mpi::environment env(ac, av);
	mpi::communicator world;
	communicator = &world;

	std::stringstream oss;
	oss << dir  << "/" << simname << "." << world.rank() << ".";
	string outputfile = oss.str();

	char tmp [255];
	std::stringstream logfile;
	logfile << outputfile << "log";
	logger = new Logger(logfile.str(),world.rank());

	sys = new System(&world);
	// END Global definitions


	logger->msg("Setting up neuron groups ...",PROGRESS,true);
	IFGroup * neurons_e = new IFGroup( NE);
	neurons_e->set_ampa_nmda_ratio(1.0);
	neurons_e->random_nmda(0.1,1);
	IFGroup * neurons_i = new IFGroup( NI);
	neurons_i->set_ampa_nmda_ratio(1.0);
	neurons_i->set_tau_mem(10e-3);

	FileModulatedPoissonGroup * poisson 
		= new FileModulatedPoissonGroup(2000,ratemodfile);
	SparseConnection * con_exte 
		= new SparseConnection( poisson, neurons_e, wext, sparseness, GLUT);
	SparseConnection * con_exti 
		= new SparseConnection( poisson, neurons_i, wext, sparseness, GLUT);

	logger->msg("Setting up connections ...",PROGRESS,true);

	SparseConnection * con_ee = new SparseConnection(  
			neurons_e,
			neurons_e,
			w,sparseness,
			GLUT);
	SparseConnection * con_ei = new SparseConnection( 
			neurons_e,
			neurons_i,
			w,sparseness,
			GLUT);
	SparseConnection * con_ii = new SparseConnection( 
			neurons_i,
			neurons_i, 
			0.9*gamma*w,sparseness,
			GABA);
	SparseConnection * con_ie = new SparseConnection( 
			neurons_i,
			neurons_e, 
			0.9*gamma*w,sparseness,
			GABA);


	if (!infilename.empty()) {
		logger->msg("Loading network state ...",PROGRESS,true);
		sys->load_network_state(outputfile);
	}

	msg = "Setting up monitors ...";
	logger->msg(msg,PROGRESS,true);

	std::stringstream filename;
	filename << outputfile << "e.ras";
	SpikeMonitor * smon_e = new SpikeMonitor( neurons_e, filename.str().c_str() );

	filename.str("");
	filename.clear();
	filename << outputfile << "i.ras";
	SpikeMonitor * smon_i = new SpikeMonitor( neurons_i, filename.str().c_str() );

	filename.str("");
	filename.clear();
	filename << outputfile << "p.ras";
	SpikeMonitor * smon_p = new SpikeMonitor( poisson, filename.str().c_str() );

	filename.str("");
	filename.clear();
	filename << outputfile << "e.prate";
	PopulationRateMonitor * pmon_e = new PopulationRateMonitor( neurons_e, filename.str().c_str(), 1.0 );

	filename.str("");
	filename.clear();
	filename << outputfile << "e.mem";
	VoltageMonitor * vmon = new VoltageMonitor( neurons_e, 123, filename.str().c_str() );

	RateChecker * chk = new RateChecker( neurons_e , -1.0 , 40. , 100e-3);

	for (int j = 0; j<1000 ; j++) {
	  neurons_e->tadd(j,5.);
	}

	logger->msg("Simulating ..." ,PROGRESS,true);
	if (!sys->run(simtime,true)) 
			errcode = 1;

	logger->msg("Saving network state ..." ,PROGRESS,true);
	sys->save_network_state(outputfile);

	logger->msg("Freeing ..." ,PROGRESS,true);
	delete sys;

	if (errcode)
		env.abort(errcode);

	return errcode;
}
