/* 
 * Copyright 2018 - 2022 Anders Lansner
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
 * \brief Implementation of a spiking attractor network with Bcpnn plasticity
 *
 * This simulates one hypercolumn with minicolumns and basket
 * cells. Parameters set according to Fiebig F and Lansner (2017): A
 * Spiking Working Memory Model Based on Hebbian Short-Term
 * Potentiation, J Neurosci 37(1): 83-96
 *
 */

#include "auryn.h"

using namespace auryn;

namespace po = boost::program_options;

std::string dir = "./";
std::string  monitor = "minicols.";

std::stringstream oss;
std::string msg;

unsigned int seed = 1;

NeuronID nhcu = 1;
NeuronID nmcu = 1;
NeuronID npy = 30; // N:o pyramidal cells per minicolumn
NeuronID nba = 24; // N:o basket cells per hypercolumn
NeuronID ndb = 1; // N:o double bouguet cells per minicolumn
NeuronID nvp = 1; // N:o vip cells per minicolumn
NeuronID npo = 30; // N:o poisson cells per minicolumn

int nrec = nhcu * nmcu * npy;

std::vector<Connection *> corr_connections;

double g_leak = 14e-9;
double c_mem = 281e-12;
double refractory_period = 0; // Otherwise e_reset does not work
double deltat = 3e-3;
double e_rest = -70e-3;
double e_thr = -55e-3;
double e_reset = -80e-3;
double tau_ampa = 0.005;
double tau_gaba = 0.005;
double tau_ad = 500e-3;

double U = 0.25;
double tau_d = 0.500;
double tau_f = 0.050;

double py_b = 86e-12;
double ba_b = 0;
double db_b = 0;
double vp_b = 0;

double py_win = 0.030;

double py_wbg = 0.030;
double ba_wbg = 0.0225;
double db_wbg = 0.0225;
double vp_wbg = 0.0225;
	
int runmode = 0;
double simtime = 5;

double poisson_rate = 100.0;
double injcur = 10.0;
double injdur = 0.05;

std::string load = "";
std::string save = "";

std::string outputfile = oss.str();

int errcode = 0;

int processargs(int ac,char *av[]) {

    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("seed", po::value<unsigned int>(), "master seed value")
            ("ref_per", po::value<double>(), "refractory period of neurons")
            ("tau_d", po::value<double>(), "depression tau for depressing/facilitating synapse")
            ("tau_f", po::value<double>(), "facilitation tau for depressing/facilitating synapse")
            ("U", po::value<double>(), "Utilization value for depressing/facilitating synapse")
            ("tau_ampa", po::value<double>(), "tau value for ampa current")
            ("tau_gaba", po::value<double>(), "tau value for gaba current")
            ("tau_ad", po::value<double>(), "tau value for adaptation")
            ("py_win", po::value<double>(), "weight of poisson input to pyramidal cells")
            ("py_wbg", po::value<double>(), "weight from background poisson to pyramidal cells")
            ("ba_wbg", po::value<double>(), "weight from background poisson to basket cells")
            ("db_wbg", po::value<double>(), "weight from background poisson to double bouquet cells")
            ("vp_wbg", po::value<double>(), "weight from background poisson to vip cells")
            ("py_b", po::value<double>(), "b value for pyramidal cells")
            ("ba_b", po::value<double>(), "b value for basket cells")
            ("db_b", po::value<double>(), "b value for double bouquet cells")
            ("vp_b", po::value<double>(), "b value for vip cells")
            ("runmode", po::value<int>(), "Run mode")
            ("simtime", po::value<double>(), "duration of simulation")
            ("nhcu", po::value<int>(), "n:o hypercolumns in network")
            ("nmcu", po::value<int>(), "n:o minicolumns per hypercolumn")
            ("npy", po::value<int>(), "n:o pyramidal cells per minicolumn")
            ("nba", po::value<int>(), "n:o basket cells per hypercolumn")
            ("ndb", po::value<int>(), "n:o double bouquet cells per minicolumn")
            ("nvp", po::value<int>(), "n:o vip cells per minicolumn")
            ("nrec", po::value<int>(), "n:o spike recorded cells")

            ("poisson_rate", po::value<double>(), "the background poisson firing rate")
			("injcur", po::value<double>(), "amount of injected current")
			("injdur", po::value<double>(), "duration of injected current")
            ("dir", po::value<std::string>(), "dir from file")
            (" monitor", po::value<std::string>(), "if "" produce no output files")
            ("load", po::value<std::string>(), "load from file")
            ("save", po::value<std::string>(), "save to file")
			;

        po::variables_map vm;        
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);    

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 1;
        }

        if (vm.count("seed")) {
			seed = vm["seed"].as<unsigned int>();
        } 

        if (vm.count("ref_per")) {
			refractory_period = vm["ref_per"].as<double>();
        } 

        if (vm.count("tau_d")) {
			tau_d = vm["tau_d"].as<double>();
        } 

        if (vm.count("tau_f")) {
			tau_f = vm["tau_f"].as<double>();
        } 

        if (vm.count("U")) {
			U = vm["U"].as<double>();
        } 

        if (vm.count("tau_ampa")) {
			tau_ampa = vm["tau_ampa"].as<double>();
        } 

        if (vm.count("tau_gaba")) {
			tau_gaba = vm["tau_gaba"].as<double>();
        } 

        if (vm.count("tau_ad")) {
			tau_ad = vm["tau_ad"].as<double>();
        } 

        if (vm.count("py_win")) {
			py_win = vm["py_win"].as<double>();
        } 

        if (vm.count("py_wbg")) {
			py_wbg = vm["py_wbg"].as<double>();
        } 

        if (vm.count("ba_wbg")) {
			ba_wbg = vm["ba_wbg"].as<double>();
        } 

        if (vm.count("db_wbg")) {
			db_wbg = vm["db_wbg"].as<double>();
        } 

        if (vm.count("vp_wbg")) {
			vp_wbg = vm["vp_wbg"].as<double>();
        } 

        if (vm.count("py_b")) {
			py_b = vm["py_b"].as<double>();
        } 

        if (vm.count("ba_b")) {
			ba_b = vm["ba_b"].as<double>();
        } 

        if (vm.count("db_b")) {
			db_b = vm["db_b"].as<double>();
        } 

        if (vm.count("vp_b")) {
			vp_b = vm["vp_b"].as<double>();
        } 

        if (vm.count("runmode")) {
			runmode = vm["runmode"].as<int>();
        } 

        if (vm.count("simtime")) {
			simtime = vm["simtime"].as<double>();
        } 

        if (vm.count("nhcu")) {
			nhcu = vm["nhcu"].as<int>();
        } 

        if (vm.count("nmcu")) {
			nmcu = vm["nmcu"].as<int>();
        } 

        if (vm.count("npy")) {
			npy = vm["npy"].as<int>();
        } 

        if (vm.count("nba")) {
			nba = vm["nba"].as<int>();
        } 

        if (vm.count("ndb")) {
			ndb = vm["ndb"].as<int>();
        } 

        if (vm.count("nvp")) {
			nvp = vm["nvp"].as<int>();
        } 

        if (vm.count("nrec")) {
			nrec = -vm["nrec"].as<int>();
        } 

        if (vm.count("poisson_rate")) {
			poisson_rate = vm["poisson_rate"].as<double>();
        } 

        if (vm.count("injcur")) {
			injcur = vm["injcur"].as<double>();
        } 

        if (vm.count("injdur")) {
			injdur = vm["injdur"].as<double>();
        } 

        if (vm.count("dir")) {
			dir = vm["dir"].as<std::string>();
        } 

        if (vm.count(" monitor")) {
			monitor = vm[" monitor"].as<std::string>();
        } 

        if (vm.count("load")) {
			load = vm["load"].as<std::string>();
        } 

        if (vm.count("save")) {
			save = vm["save"].as<std::string>();
        } 

    }
    catch(std::exception& e) {
		std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
		std::cerr << "Exception of unknown type!\n";
    }

	if (nrec<0) nrec = -nrec; else nrec = npy * nmcu * nhcu;

}

AdExGroup *setup_py_cells(int n) {
	AdExGroup *py_cells = new AdExGroup(n);
	py_cells->set_refractory_period(refractory_period);
	py_cells->set_g_leak(g_leak);
	py_cells->set_c_mem(c_mem);
	py_cells->set_delta_t(deltat*g_leak); // According to set_ method in AdExGroup
	py_cells->set_e_rest(e_rest);
	py_cells->set_e_thr(e_thr);
	py_cells->set_e_reset(e_reset); // Effective only when refractor_period = 0
	py_cells->set_tau_w(tau_ad);
	py_cells->set_a(0);
	py_cells->set_b(py_b);
	py_cells->set_tau_ampa(0.005);
	py_cells->set_tau_gaba(0.005);
	py_cells->set_e_rev_gaba(-80e-3);

	return py_cells;
}

AdExGroup *setup_ba_cells(int n) {
	AdExGroup *ba_cells = new AdExGroup(n);
	ba_cells->set_refractory_period(refractory_period);
	ba_cells->set_g_leak(g_leak);
	ba_cells->set_c_mem(c_mem);
	ba_cells->set_delta_t(deltat*g_leak);
	ba_cells->set_e_rest(e_rest);
	ba_cells->set_e_thr(e_thr);
	ba_cells->set_e_reset(e_reset);
	ba_cells->set_tau_w(tau_ad);
	ba_cells->set_a(0);
	ba_cells->set_b(ba_b); // Typically 0.0
	ba_cells->set_tau_ampa(0.005);
	ba_cells->set_tau_gaba(0.005);
	ba_cells->set_e_rev_gaba(-80e-3);

	return ba_cells;
}

AdExGroup *setup_db_cells(int n) {
	AdExGroup *db_cells = new AdExGroup(n);
	db_cells->set_refractory_period(refractory_period);
	db_cells->set_g_leak(g_leak);
	db_cells->set_c_mem(c_mem);
	db_cells->set_delta_t(deltat*g_leak);
	db_cells->set_e_rest(e_rest);
	db_cells->set_e_thr(e_thr);
	db_cells->set_e_reset(e_reset);

	db_cells->set_tau_w(tau_ad);
	db_cells->set_a(0);
	db_cells->set_b(db_b);
	db_cells->set_tau_ampa(0.005);
	db_cells->set_tau_gaba(0.005);
	db_cells->set_e_rev_gaba(-80e-3);


	return db_cells;
}

AdExGroup *setup_vp_cells(int n) {
	AdExGroup *vp_cells = new AdExGroup(n);
	vp_cells->set_refractory_period(refractory_period);
	vp_cells->set_g_leak(g_leak);
	vp_cells->set_c_mem(c_mem);
	vp_cells->set_delta_t(deltat*g_leak);
	vp_cells->set_e_rest(e_rest);
	vp_cells->set_e_thr(e_thr);
	vp_cells->set_e_reset(e_reset);

	vp_cells->set_tau_w(tau_ad);
	vp_cells->set_a(0);
	vp_cells->set_b(vp_b);
	vp_cells->set_tau_ampa(0.005);
	vp_cells->set_tau_gaba(0.005);
	vp_cells->set_e_rev_gaba(-80e-3);

	return vp_cells;
}

int test0() {

	/* Testing pairwise synaptic PSP:s between single neurons:
	   py1-->ba1, py1->db1, ba2-->py2, db2-->vp1, vp2-->py3 */ 

	logger->msg("Setting up neuron groups ...",PROGRESS,true);
	
	nhcu = 1; nmcu = 1;
	npy = 0; nba = 0; ndb = 0; nvp = 0;

	/* Spiking cells */
	AdExGroup *py1_cells = setup_py_cells(1); ++npy;
	AdExGroup *ba2_cells = setup_ba_cells(1); ++nba;
	AdExGroup *db2_cells = setup_db_cells(1); ++ndb;
	AdExGroup *vp2_cells = setup_vp_cells(1); ++nvp;

	e_thr = -0.020;

	/* PSP cells */
	AdExGroup *py2_cells = setup_py_cells(1); ++npy;
	AdExGroup *py3_cells = setup_py_cells(1); ++npy;
	AdExGroup *py4_cells = setup_py_cells(1); ++npy;
	AdExGroup *py5_cells = setup_py_cells(1); ++npy;
	AdExGroup *ba1_cells = setup_ba_cells(1); ++nba;
	AdExGroup *db1_cells = setup_db_cells(1); ++ndb;
	AdExGroup *db3_cells = setup_db_cells(1); ++ndb;
	AdExGroup *vp1_cells = setup_vp_cells(1); ++nvp;

	logger->msg("Setting up current injections ...",PROGRESS,true);

	CurrentInjector * curinj_py1 = new CurrentInjector(py1_cells);
	CurrentInjector * curinj_ba2 = new CurrentInjector(ba2_cells);
	CurrentInjector * curinj_db2 = new CurrentInjector(db2_cells);
	CurrentInjector * curinj_vp2 = new CurrentInjector(vp2_cells);

	logger->msg("Setting up all connections ...",PROGRESS,true);

	FILE *wfile = fopen("weights.wij","r");
	int nf;
	float wpypyL,wpypyG,wpyba,wbapy,wdbpy,wpydbG,wpyvp,wvpdb,pypydensL,pypydensG,pybadens,bapydens,
		dbpydens,pydbdensG,pyvpdens,vpdbdens;
	nf = fscanf(wfile,"%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f ",
				&pypydensL,&wpypyL,&pypydensG,&wpypyG,&pybadens,&wpyba,&bapydens,&wbapy,
				&dbpydens,&wdbpy,&pydbdensG,&wpydbG,&pyvpdens,&wpyvp,&vpdbdens,&wvpdb);
	fclose(wfile);

	STPConnection *py_pyL = new STPConnection(py1_cells,py2_cells,wpypyL,1.0,GLUT);
	// py_pyL->set_tau_d(tau_d); py_pyL->set_tau_f(tau_f); py_pyL->set_ujump(U);
	// STPConnection *py_pyG = new STPConnection(py1_cells,py3_cells,wpypyG,1,GLUT);
	// py_pyG->set_tau_d(tau_d); py_pyG->set_tau_f(tau_f); py_pyG->set_ujump(U);
	// SparseConnection *py_ba = new SparseConnection(py1_cells,ba1_cells,wpyba,1,GLUT);
	// SparseConnection *ba_py = new SparseConnection(ba2_cells,py4_cells,wbapy,1,GABA);
	// STPConnection *py_db = new STPConnection(py1_cells,db1_cells,wpydbG,1,GLUT);
	// py_db->set_tau_d(tau_d); py_db->set_tau_f(tau_f); py_db->set_ujump(U);
	// SparseConnection *db_py = new SparseConnection(db2_cells,py5_cells,wdbpy,1,GABA);
	// SparseConnection *py_vp = new SparseConnection(py1_cells,vp1_cells,wpyvp,1,GLUT);
	// SparseConnection *vp_db = new SparseConnection(vp2_cells,db3_cells,wvpdb,1,GABA);

	if (monitor!="") {
		VoltageMonitor *vmon_py1 = new VoltageMonitor(py1_cells,0,sys->fn("py1.vmem"));
		VoltageMonitor *vmon_py2 = new VoltageMonitor(py2_cells,0,sys->fn("py2.vmem"));
		VoltageMonitor *vmon_py3 = new VoltageMonitor(py3_cells,0,sys->fn("py3.vmem"));
		VoltageMonitor *vmon_py4 = new VoltageMonitor(py4_cells,0,sys->fn("py4.vmem"));
		VoltageMonitor *vmon_py5 = new VoltageMonitor(py5_cells,0,sys->fn("py5.vmem"));
		VoltageMonitor *vmon_ba1 = new VoltageMonitor(ba1_cells,0,sys->fn("ba1.vmem"));
		VoltageMonitor *vmon_ba2 = new VoltageMonitor(ba2_cells,0,sys->fn("ba2.vmem"));
		VoltageMonitor *vmon_db1 = new VoltageMonitor(db1_cells,0,sys->fn("db1.vmem"));
		VoltageMonitor *vmon_db2 = new VoltageMonitor(db2_cells,0,sys->fn("db2.vmem"));
		VoltageMonitor *vmon_db3 = new VoltageMonitor(db3_cells,0,sys->fn("db3.vmem"));
		VoltageMonitor *vmon_vp1 = new VoltageMonitor(vp1_cells,0,sys->fn("vp1.vmem"));
		VoltageMonitor *vmon_vp2 = new VoltageMonitor(vp2_cells,0,sys->fn("vp3.vmem"));
    }

	std::ofstream outfile;
	string fname = "minicols.data";
	outfile.open(fname.c_str(),std::ios::out);
	if (!outfile) {
		std::cerr << "Can't open output file " << fname.c_str() << std::endl;
		throw AurynOpenFileException();
	}
	outfile << nhcu*nmcu*npy << " " << nhcu*nba << " " << nhcu*nmcu*ndb << " " << nhcu*nmcu*nvp << std::endl;
	outfile.close();

	logger->msg("Simulating ..." ,PROGRESS,true);

	MPI_Barrier(MPI::COMM_WORLD);

	sys->run(0.1);

	curinj_py1->set_all_currents(injcur);
	curinj_ba2->set_all_currents(injcur);
	curinj_db2->set_all_currents(injcur);
	curinj_vp2->set_all_currents(injcur);

	sys->run(injdur);

	curinj_py1->set_all_currents(0);
	curinj_ba2->set_all_currents(0);
	curinj_db2->set_all_currents(0);
	curinj_vp2->set_all_currents(0);

	sys->run(0.2 - injdur);

	// if ( !save.empty() ) {
	// 	sys->save_network_state(save);
    // }

	if (errcode)
		auryn_abort(errcode);

}

int test1(bool usedens) {

	/* Testing pairwise synaptic PSP:s between single neurons:
	   py1-->ba1, py1->db1, ba2-->py2, db2-->vp1, vp2-->py3 */ 

	logger->msg("Setting up neuron groups ...",PROGRESS,true);

	e_thr = -0.020;

	AdExGroup *py1_cells = setup_py_cells(1);
	AdExGroup *py2_cells = setup_py_cells(1);
	AdExGroup *py3_cells = setup_py_cells(1);
	AdExGroup *py4_cells = setup_py_cells(1);
	AdExGroup *py5_cells = setup_py_cells(1);
	
	AdExGroup *ba1_cells = setup_ba_cells(1);
	AdExGroup *ba2_cells = setup_ba_cells(1);

	AdExGroup *db1_cells = setup_db_cells(1);
	AdExGroup *db2_cells = setup_db_cells(1);
	AdExGroup *db3_cells = setup_db_cells(1);

	AdExGroup *vp1_cells = setup_vp_cells(1);
	AdExGroup *vp2_cells = setup_vp_cells(1);

	logger->msg("Setting up current injections ...",PROGRESS,true);

	CurrentInjector * curinj_py1 = new CurrentInjector(py1_cells);
	CurrentInjector * curinj_ba2 = new CurrentInjector(ba2_cells);
	CurrentInjector * curinj_db2 = new CurrentInjector(db2_cells);
	CurrentInjector * curinj_vp2 = new CurrentInjector(vp2_cells);

	logger->msg("Setting up all connections ...",PROGRESS,true);

	FILE *wfile = fopen("weights.wij","r");
	int nf;
	float wpypyL,wpypyG,wpyba,wbapy,wdbpy,wpydbG,wpyvp,wvpdb,pypydensL,pypydensG,pybadens,bapydens,
		dbpydens,pydbdensG,pyvpdens,vpdbdens;
	nf = fscanf(wfile,"%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f ",
				&pypydensL,&wpypyL,&pypydensG,&wpypyG,&pybadens,&wpyba,&bapydens,&wbapy,
				&dbpydens,&wdbpy,&pydbdensG,&wpydbG,&pyvpdens,&wpyvp,&vpdbdens,&wvpdb);
	fclose(wfile);

	if (usedens) {
		if (pypydensL>0) {
			STPConnection *py_pyL = new STPConnection(py1_cells,py2_cells,wpypyL,pypydensL,GLUT);
			py_pyL->set_tau_d(tau_d); py_pyL->set_tau_f(tau_f); py_pyL->set_ujump(U);
		}
		if (pypydensG>0) {
			STPConnection *py_pyG = new STPConnection(py1_cells,py3_cells,wpypyG,pypydensG,GLUT);
			py_pyG->set_tau_d(tau_d); py_pyG->set_tau_f(tau_f); py_pyG->set_ujump(U);
		}
		if (pybadens>0)
			SparseConnection *py_ba = new SparseConnection(py1_cells,ba1_cells,wpyba,pybadens,GLUT);
		if (bapydens>0)
			SparseConnection *ba_py = new SparseConnection(ba2_cells,py4_cells,wbapy,bapydens,GABA);
		if (pydbdensG>0) {
			STPConnection *py_db = new STPConnection(py1_cells,db1_cells,wpydbG,pydbdensG,GLUT);
			py_db->set_tau_d(tau_d); py_db->set_tau_f(tau_f); py_db->set_ujump(U);
		}
		if (dbpydens>0)
			SparseConnection *db_py = new SparseConnection(db2_cells,py5_cells,wdbpy,dbpydens,GABA);
		if (pyvpdens>0)
			SparseConnection *py_vp = new SparseConnection(py1_cells,vp1_cells,wpyvp,pyvpdens,GLUT);
		if (vpdbdens>0)
			SparseConnection *vp_db = new SparseConnection(vp2_cells,db3_cells,wvpdb,vpdbdens,GABA);
	} else {
		STPConnection *py_pyL = new STPConnection(py1_cells,py2_cells,wpypyL,1,GLUT);
		py_pyL->set_tau_d(tau_d); py_pyL->set_tau_f(tau_f); py_pyL->set_ujump(U);
		STPConnection *py_pyG = new STPConnection(py1_cells,py3_cells,wpypyG,1,GLUT);
		py_pyG->set_tau_d(tau_d); py_pyG->set_tau_f(tau_f); py_pyG->set_ujump(U);
		SparseConnection *py_ba = new SparseConnection(py1_cells,ba1_cells,wpyba,1,GLUT);
		SparseConnection *ba_py = new SparseConnection(ba2_cells,py4_cells,wbapy,1,GABA);
		STPConnection *py_db = new STPConnection(py1_cells,db1_cells,wpydbG,1,GLUT);
		py_db->set_tau_d(tau_d); py_db->set_tau_f(tau_f); py_db->set_ujump(U);
		SparseConnection *db_py = new SparseConnection(db2_cells,py5_cells,wdbpy,1,GABA);
		SparseConnection *py_vp = new SparseConnection(py1_cells,vp1_cells,wpyvp,1,GLUT);
		SparseConnection *vp_db = new SparseConnection(vp2_cells,db3_cells,wvpdb,1,GABA);
	}
	
	if (monitor!="") {
	    VoltageMonitor *vmon_py1 = new VoltageMonitor(py1_cells,0,sys->fn("py1.vmem"));
		VoltageMonitor *vmon_py2 = new VoltageMonitor(py2_cells,0,sys->fn("py2.vmem"));
		VoltageMonitor *vmon_py3 = new VoltageMonitor(py3_cells,0,sys->fn("py3.vmem"));
		VoltageMonitor *vmon_py4 = new VoltageMonitor(py4_cells,0,sys->fn("py4.vmem"));
		VoltageMonitor *vmon_py5 = new VoltageMonitor(py5_cells,0,sys->fn("py5.vmem"));
		VoltageMonitor *vmon_ba1 = new VoltageMonitor(ba1_cells,0,sys->fn("ba1.vmem"));
		VoltageMonitor *vmon_ba2 = new VoltageMonitor(ba2_cells,0,sys->fn("ba2.vmem"));
		VoltageMonitor *vmon_db1 = new VoltageMonitor(db1_cells,0,sys->fn("db1.vmem"));
		VoltageMonitor *vmon_db2 = new VoltageMonitor(db2_cells,0,sys->fn("db2.vmem"));
		VoltageMonitor *vmon_db3 = new VoltageMonitor(db3_cells,0,sys->fn("db3.vmem"));
		VoltageMonitor *vmon_vp1 = new VoltageMonitor(vp1_cells,0,sys->fn("vp1.vmem"));
		VoltageMonitor *vmon_vp2 = new VoltageMonitor(vp2_cells,0,sys->fn("vp3.vmem"));
    }

	std::ofstream outfile;
	string fname = "minicols.data";
	outfile.open(fname.c_str(),std::ios::out);
	if (!outfile) {
		std::cerr << "Can't open output file " << fname.c_str() << std::endl;
		throw AurynOpenFileException();
	}
	outfile << nhcu*nmcu*npy << " " << nhcu*nba << " " << nhcu*nmcu*ndb << " " << nhcu*nmcu*nvp << std::endl;
	outfile.close();

	logger->msg("Simulating ..." ,PROGRESS,true);

	MPI_Barrier(MPI::COMM_WORLD);

	double start = MPI_Wtime();

	sys->run(0.1);
	curinj_py1->set_all_currents(injcur);
	curinj_ba2->set_all_currents(injcur);
	curinj_db2->set_all_currents(injcur);
	curinj_vp2->set_all_currents(injcur);

	sys->run(injdur);
	curinj_py1->set_all_currents(0);
	curinj_ba2->set_all_currents(0);
	curinj_db2->set_all_currents(0);
	curinj_vp2->set_all_currents(0);
	sys->run(0.2 - injdur);

	if ( !save.empty() ) {
		sys->save_network_state(save);
    }

	if (errcode)
		auryn_abort(errcode);

}

int test3() {

	logger->msg("Setting up neuron groups ...",PROGRESS,true);

	AdExGroup *py_cells = setup_py_cells(nhcu*nmcu*npy);
	AdExGroup *ba_cells = setup_ba_cells(nhcu*nba);
	AdExGroup *db_cells = setup_db_cells(nhcu*nmcu*ndb);
	AdExGroup *vp_cells = setup_vp_cells(nhcu*nmcu*nvp);

	PoissonGroup *poisson = new PoissonGroup(npo,poisson_rate);

	SparseConnection *con_stim_py_inp = new SparseConnection(poisson,py_cells,py_win,0.5,GLUT);

	SparseConnection *con_stim_py_exc = new SparseConnection(poisson,py_cells,py_wbg,0.5,GLUT);
	SparseConnection *con_stim_py_inh = new SparseConnection(poisson,py_cells,py_wbg,0.5,GABA);
	SparseConnection *con_stim_ba_exc = new SparseConnection(poisson,ba_cells,ba_wbg,0.5,GLUT);
	SparseConnection *con_stim_ba_inh = new SparseConnection(poisson,ba_cells,ba_wbg,0.5,GABA);
	SparseConnection *con_stim_db_exc = new SparseConnection(poisson,db_cells,db_wbg,0.5,GLUT);
	SparseConnection *con_stim_db_inh = new SparseConnection(poisson,db_cells,db_wbg,0.5,GABA);
	SparseConnection *con_stim_vp_exc = new SparseConnection(poisson,vp_cells,vp_wbg,0.5,GLUT);
	SparseConnection *con_stim_vp_inh = new SparseConnection(poisson,vp_cells,vp_wbg,0.5,GABA);
	
	logger->msg("Setting up local py->py connections ...",PROGRESS,true);
	STPConnection *py_pyL = new STPConnection(py_cells,py_cells,"WpypyLx.wij",GLUT);
	py_pyL->set_tau_d(tau_d); py_pyL->set_tau_f(tau_f); py_pyL->set_ujump(U);

	logger->msg("Setting up global py->py connections ...",PROGRESS,true);
	STPConnection *py_pyG = new STPConnection(py_cells,py_cells,"WpypyGx.wij",GLUT);
	py_pyG->set_tau_d(tau_d); py_pyG->set_tau_f(tau_f); py_pyG->set_ujump(U);

	logger->msg("Setting up py->ba connections ...",PROGRESS,true);
	SparseConnection *py_ba = new SparseConnection(py_cells,ba_cells,"Wpybax.wij",GLUT);

	logger->msg("Setting up ba->py connections ...",PROGRESS,true);
	SparseConnection *ba_py = new SparseConnection(ba_cells,py_cells,"Wbapyx.wij",GABA);

	logger->msg("Setting up py->db connections ...",PROGRESS,true);
	STPConnection *py_db = new STPConnection(py_cells,db_cells,"WpydbGx.wij",GLUT);
	py_db->set_tau_d(tau_d); py_db->set_tau_f(tau_f); py_db->set_ujump(U);

	logger->msg("Setting up db->py connections ...",PROGRESS,true);
	SparseConnection *db_py = new SparseConnection(db_cells,py_cells,"Wdbpyx.wij",GABA);

	logger->msg("Setting up py->vp connections ...",PROGRESS,true);
	SparseConnection *py_vp = new SparseConnection(py_cells,vp_cells,"Wpyvpx.wij",GLUT);

	logger->msg("Setting up vp->db connections ...",PROGRESS,true);
	SparseConnection *vp_db = new SparseConnection(vp_cells,db_cells,"Wvpdbx.wij",GABA);

	if ( monitor!="") {
	    std::stringstream fname_py,fname_ba,fname_db,fname_vp;

	    fname_py << outputfile << "py.ras";
	    SpikeMonitor *smon_py = new SpikeMonitor(py_cells,fname_py.str().c_str(),nrec);
	    fname_ba << outputfile << "ba.ras";
	    SpikeMonitor *smon_ba = new SpikeMonitor(ba_cells,fname_ba.str().c_str(),nrec);
	    fname_db << outputfile << "db.ras";
	    SpikeMonitor *smon_db = new SpikeMonitor(db_cells,fname_db.str().c_str(),nrec);
	    fname_vp << outputfile << "vp.ras";
	    SpikeMonitor *smon_vp = new SpikeMonitor(vp_cells,fname_vp.str().c_str(),nrec);

		VoltageMonitor *vmon_py1 = new VoltageMonitor(py_cells,0,sys->fn("py1.vmem"));
		VoltageMonitor *vmon_py2 = new VoltageMonitor(py_cells,nhcu*nmcu*npy-1,sys->fn("py2.vmem"));
		VoltageMonitor *vmon_ba1 = new VoltageMonitor(ba_cells,0,sys->fn("ba1.vmem"));
		VoltageMonitor *vmon_ba2 = new VoltageMonitor(ba_cells,nhcu*nba-1,sys->fn("ba2.vmem"));
		VoltageMonitor *vmon_db1 = new VoltageMonitor(db_cells,0,sys->fn("db1.vmem"));
		VoltageMonitor *vmon_db2 = new VoltageMonitor(db_cells,nhcu*nmcu*ndb-1,sys->fn("db2.vmem"));
		VoltageMonitor *vmon_vp1 = new VoltageMonitor(vp_cells,0,sys->fn("vp1.vmem"));
		VoltageMonitor *vmon_vp2 = new VoltageMonitor(vp_cells,nhcu*nmcu*nvp-1,sys->fn("vp2.vmem"));

	    // Record firing rates (sample every 50 ms)
#ifdef POISSON
	    PopulationRateMonitor *pmon_po = new PopulationRateMonitor(poisson,sys->fn("po_rate"),0.005);
#endif // POISSON
	    PopulationRateMonitor *pmon_py = new PopulationRateMonitor(py_cells,sys->fn("py_rate"),0.005);
	    PopulationRateMonitor *pmon_ba = new PopulationRateMonitor(ba_cells,sys->fn("ba_rate"),0.005);
	    PopulationRateMonitor *pmon_db = new PopulationRateMonitor(db_cells,sys->fn("db_rate"),0.005);
	    PopulationRateMonitor *pmon_vp = new PopulationRateMonitor(vp_cells,sys->fn("vp_rate"),0.005);

	}

	std::ofstream outfile;
	string fname = "minicols.data";
	outfile.open(fname.c_str(),std::ios::out);
	if (!outfile) {
		std::cerr << "Can't open output file " << fname.c_str() << std::endl;
		throw AurynOpenFileException();
	}
	outfile << nhcu*nmcu*npy << " " << nhcu*nba << " " << nhcu*nmcu*ndb << " " << nhcu*nmcu*nvp << std::endl;
	outfile.close();

	logger->msg("Simulating ..." ,PROGRESS,true);

	MPI_Barrier(MPI::COMM_WORLD);

	double start = MPI_Wtime();

	if (!sys->run(simtime,true)) 
		errcode = 1;

	// // stimgroup1->active = false;

	// if (!sys->run(simtime,true)) 
	// 		errcode = 1;

	if ( !save.empty() ) {
		sys->save_network_state(save);
	}


	if (errcode)
		auryn_abort(errcode);

	int nsyn_pypyL,gnsyn_pypyL,nsyn_pyba,gnsyn_pyba,nsyn_bapy,gnsyn_bapy,nsyn_pydb,gnsyn_pydb,nsyn_dbpy,gnsyn_dbpy,nsyn_pyvp,gnsyn_pyvp,nsyn_vpdb,gnsyn_vpdb;
	/* Get total number of different types of synapses */
	nsyn_pypyL = py_pyL->get_nonzero();
	MPI_Reduce(&nsyn_pypyL,&gnsyn_pypyL,1,MPI_INT,MPI_SUM,0,*sys->get_com());

	nsyn_pyba = py_ba->get_nonzero();
	MPI_Reduce(&nsyn_pyba,&gnsyn_pyba,1,MPI_INT,MPI_SUM,0,*sys->get_com());

	nsyn_bapy = ba_py->get_nonzero();
	MPI_Reduce(&nsyn_bapy,&gnsyn_bapy,1,MPI_INT,MPI_SUM,0,*sys->get_com());
	
	nsyn_pydb = py_db->get_nonzero();
	MPI_Reduce(&nsyn_pydb,&gnsyn_pydb,1,MPI_INT,MPI_SUM,0,*sys->get_com());

	nsyn_dbpy = db_py->get_nonzero();
	MPI_Reduce(&nsyn_dbpy,&gnsyn_dbpy,1,MPI_INT,MPI_SUM,0,*sys->get_com());
	
	nsyn_pyvp = py_vp->get_nonzero();
	MPI_Reduce(&nsyn_pyvp,&gnsyn_pyvp,1,MPI_INT,MPI_SUM,0,*sys->get_com());

	nsyn_vpdb = db_py->get_nonzero();
	MPI_Reduce(&nsyn_vpdb,&gnsyn_vpdb,1,MPI_INT,MPI_SUM,0,*sys->get_com());
	
	if (sys->mpi_rank()==0) {
		std::cerr << "Execution time = " << MPI::Wtime() - start << " sec\n";

		std::cerr << "N:o py->py local weights = " << gnsyn_pypyL << std::endl;
		
		std::cerr << "N:o py->ba weights = " << gnsyn_pyba << std::endl;
		
		std::cerr << "N:o ba->py weights = " << gnsyn_bapy << std::endl;
		
		std::cerr << "N:o py->db weights = " << gnsyn_pydb << std::endl;
		
		std::cerr << "N:o db->py weights = " << gnsyn_dbpy << std::endl;
		
		std::cerr << "N:o py->vp weights = " << gnsyn_pyvp << std::endl;
		
		std::cerr << "N:o vp->db weights = " << gnsyn_vpdb << std::endl;
		
	    std::cerr << "Maximum send buffer size: " << sys->get_max_send_buffer_size() << std::endl;

	}

}

AurynWeight *getrndweight(SparseConnection *spcon,NeuronID &i,NeuronID &j) {

	/* Find one connected pair randomly */

	AurynWeight *w;
	int prN = spcon->src->get_pre_size(),poN = spcon->dst->get_post_size();

	while ((w = spcon->get_ptr(i = rand()%prN,j=rand()%poN))==NULL) ;

}

AurynWeight *getrndsourceweight(SparseConnection *spcon,NeuronID &i,NeuronID j) {

	/* Find one connected presynaptic target randomly */

	AurynWeight *w;
	int prN = spcon->src->get_pre_size(),poN = spcon->dst->get_post_size();

	while ((w = spcon->get_ptr(i = rand()%prN,j))==NULL) ;

}

AurynWeight *getrndtargetweight(SparseConnection *spcon,NeuronID i,NeuronID &j) {

	/* Find one connected postsynaptic target randomly */

	AurynWeight *w;
	int prN = spcon->src->get_pre_size(),poN = spcon->dst->get_post_size();

	while ((w = spcon->get_ptr(i,j=rand()%poN))==NULL) {
	}

}

int test4() {

	/* Here the same networks as in test3 is probed for synaptic interactions */

	logger->msg("Setting up neuron groups ...",PROGRESS,true);

	AdExGroup *py_cells = setup_py_cells(nhcu*nmcu*npy);
	AdExGroup *ba_cells = setup_ba_cells(nhcu*nba);
	AdExGroup *db_cells = setup_db_cells(nhcu*nmcu*ndb);
	AdExGroup *vp_cells = setup_vp_cells(nhcu*nmcu*nvp);

	PoissonGroup *poisson = new PoissonGroup(npo,poisson_rate);

	SparseConnection *con_stim_py_inp = new SparseConnection(poisson,py_cells,py_win,0.5,GLUT);

	SparseConnection *con_stim_py_exc = new SparseConnection(poisson,py_cells,py_wbg,0.5,GLUT);
	SparseConnection *con_stim_py_inh = new SparseConnection(poisson,py_cells,py_wbg,0.5,GABA);
	SparseConnection *con_stim_ba_exc = new SparseConnection(poisson,ba_cells,ba_wbg,0.5,GLUT);
	SparseConnection *con_stim_ba_inh = new SparseConnection(poisson,ba_cells,ba_wbg,0.5,GABA);
	SparseConnection *con_stim_db_exc = new SparseConnection(poisson,db_cells,db_wbg,0.5,GLUT);
	SparseConnection *con_stim_db_inh = new SparseConnection(poisson,db_cells,db_wbg,0.5,GABA);
	SparseConnection *con_stim_vp_exc = new SparseConnection(poisson,vp_cells,vp_wbg,0.5,GLUT);
	SparseConnection *con_stim_vp_inh = new SparseConnection(poisson,vp_cells,vp_wbg,0.5,GABA);
	
	logger->msg("Setting up local py->py connections ...",PROGRESS,true);
	STPConnection *py_pyL = new STPConnection(py_cells,py_cells,"WpypyLx.wij",GLUT);
	py_pyL->set_tau_d(tau_d); py_pyL->set_tau_f(tau_f); py_pyL->set_ujump(U);

	logger->msg("Setting up global py->py connections ...",PROGRESS,true);
	STPConnection *py_pyG = new STPConnection(py_cells,py_cells,"WpypyGx.wij",GLUT);
	py_pyG->set_tau_d(tau_d); py_pyG->set_tau_f(tau_f); py_pyG->set_ujump(U);

	logger->msg("Setting up py->ba connections ...",PROGRESS,true);
	SparseConnection *py_ba = new SparseConnection(py_cells,ba_cells,"Wpybax.wij",GLUT);

	logger->msg("Setting up ba->py connections ...",PROGRESS,true);
	SparseConnection *ba_py = new SparseConnection(ba_cells,py_cells,"Wbapyx.wij",GABA);

	logger->msg("Setting up py->db connections ...",PROGRESS,true);
	STPConnection *py_db = new STPConnection(py_cells,db_cells,"WpydbGx.wij",GLUT);
	py_db->set_tau_d(tau_d); py_db->set_tau_f(tau_f); py_db->set_ujump(U);

	logger->msg("Setting up db->py connections ...",PROGRESS,true);
	SparseConnection *db_py = new SparseConnection(db_cells,py_cells,"Wdbpyx.wij",GABA);

	logger->msg("Setting up py->vp connections ...",PROGRESS,true);
	SparseConnection *py_vp = new SparseConnection(py_cells,vp_cells,"Wpyvpx.wij",GLUT);

	logger->msg("Setting up vp->db connections ...",PROGRESS,true);
	SparseConnection *vp_db = new SparseConnection(vp_cells,db_cells,"Wvpdbx.wij",GABA);

	logger->msg("Setting up probes for monitoring ...",PROGRESS,true);

	NeuronID pypyG_i,pypyG_j,py_db_i,py_db_j,py_db_k;
	AurynWeight *w;
	w = getrndweight(py_pyG,pypyG_i,pypyG_j);
	std::cout << "pypyG: i = " << pypyG_i << " j = " << pypyG_j << " monitored\n";
	VoltageMonitor *vmon_pypyG_i = new VoltageMonitor(py_cells,pypyG_i,sys->fn("pypyG_i.vmem"));
	VoltageMonitor *vmon_pypyG_j = new VoltageMonitor(py_cells,pypyG_j,sys->fn("pypyG_j.vmem"));

	/* Do all the pairwise (as in test0) first, then perhaps py_i-->db_j-->py_k */

	// VoltageMonitor *vmon_pydbpy = new VoltageMonitor(db_cells,0,sys->fn("pydbpy_j.vmem"));
	// w = getrndsourceweight(py_db,i,0);
	// w = getrndtargetweight(db_py,0,k);
	// std::cout << "py_db_py: i = " << i << " j = " << 0 << " k = " << j << " monitored\n";
	// VoltageMonitor *vmon_pydb_i = new VoltageMonitor(py_cells,i,sys->fn("py_db_i.vmem"));
	// VoltageMonitor *vmon_pydb_j = new VoltageMonitor(db_cells,0,sys->fn("py_db_j.vmem"));
	// VoltageMonitor *vmon_dbpy_k = new VoltageMonitor(py_cells,k,sys->fn("db_py_k.vmem"));

	CurrentInjector * curinj_1 = new CurrentInjector(py_cells);
	// CurrentInjector * curinj_2 = new CurrentInjector(py_cells);
	// CurrentInjector * curinj_3 = new CurrentInjector(db_cells);
	// CurrentInjector * curinj_4 = new CurrentInjector(ba_cells);
	// CurrentInjector * curinj_5 = new CurrentInjector(vp_cells);

	logger->msg("Simulating ..." ,PROGRESS,true);

	MPI_Barrier(MPI::COMM_WORLD);

	sys->run(0.1);

	curinj_1->set_current(pypyG_i,injcur);
	// curinj_py2->set_current(i,injcur);
	// curinj_py1->set_current(i,injcur);

	sys->run(injdur);

	curinj_1->set_all_currents(0);

	sys->run(0.2 - injdur);

	if ( !save.empty() ) {
		sys->save_network_state(save);
	}

	if (errcode)
		auryn_abort(errcode);

}

int main(int ac,char *av[]) {

	processargs(ac,av);

	auryn_init(ac, av);
	sys->set_master_seed(seed);
	sys->set_simulation_name("minicols");
	oss << dir  <<  monitor << sys->mpi_rank() << ".";
	outputfile = oss.str();

	switch (runmode) {
	case 0: test0(); break;
	case 1: test1(false); break;
	case 2: test1(true); break;
	case 3: test3(); break;
	case 4: test4(); break;
	}	 

	if (sys->mpi_rank()==0) {
		std::cout << "N:o pyramidal cells = " << nhcu*nmcu*npy<< " (" << npy*nmcu << " per hcu)" << std::endl;
		std::cout << "N:o basket cells = " << nhcu*nba << " (" << nba << " per hcu)" << std::endl;

		std::cout << "N:o double bouquet cells = " << nhcu*nmcu*ndb << " (" << ndb*nmcu << " per hcu)" << std::endl;

		std::cout << "N:o vip cells = " << nhcu*nmcu*nvp << " (" << nvp*nmcu << " per hcu)" << std::endl;
	}

	if (sys->mpi_rank()==0) std::cout << std::endl;

	auryn_free();

	return errcode;
}
