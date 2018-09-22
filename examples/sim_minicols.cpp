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
double tau_nmda = 0.150;
double ampa_nmda_ratio = 120;
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
	
int runmode = 2;
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
            ("tau_nmda", po::value<double>(), "tau value for nmda current")
            ("ampa_nmda_ratio", po::value<double>(), "ratio between g_ampa and g_nmda ")
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

        if (vm.count("tau_nmda")) {
			tau_nmda = vm["tau_nmda"].as<double>();
        } 

        if (vm.count("ampa_nmda_ratio")) {
			ampa_nmda_ratio = vm["ampa_nmda_ratio"].as<double>();
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

AdExGroupNB *setup_py_cells(int n) {
	AdExGroupNB *py_cells = new AdExGroupNB(n);
	py_cells->set_refractory_period(refractory_period);
	py_cells->set_g_leak(g_leak);
	py_cells->set_c_mem(c_mem);
	py_cells->set_delta_t(deltat*g_leak); // According to set_ method in AdExGroupNB
	py_cells->set_e_rest(e_rest);
	py_cells->set_e_thr(e_thr);
	py_cells->set_e_reset(e_reset); // Effective only when refractor_period = 0
	py_cells->set_tau_w(tau_ad);
	py_cells->set_a(0);
	py_cells->set_b(py_b);
	py_cells->set_tau_ampa(tau_ampa);
	// py_cells->set_tau_nmda(tau_nmda);
	py_cells->set_tau_gaba(tau_gaba);
	py_cells->set_e_rev_gaba(-80e-3);

	return py_cells;
}

AdExGroupNB *setup_ba_cells(int n) {
	AdExGroupNB *ba_cells = new AdExGroupNB(n);
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
	ba_cells->set_tau_ampa(tau_ampa);
	ba_cells->set_tau_gaba(tau_gaba);
	ba_cells->set_e_rev_gaba(-80e-3);

	return ba_cells;
}

AdExGroupNB *setup_db_cells(int n) {
	AdExGroupNB *db_cells = new AdExGroupNB(n);
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
	db_cells->set_tau_ampa(tau_ampa);
	//	db_cells->set_tau_nmda(tau_nmda);
	db_cells->set_tau_gaba(tau_gaba);
	db_cells->set_e_rev_gaba(-80e-3);


	return db_cells;
}

AdExGroupNB *setup_vp_cells(int n) {
	AdExGroupNB *vp_cells = new AdExGroupNB(n);
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
	vp_cells->set_tau_ampa(tau_ampa);
	vp_cells->set_tau_gaba(tau_gaba);
	vp_cells->set_e_rev_gaba(-80e-3);

	return vp_cells;
}

int test1() {

	logger->msg("Setting up neuron groups ...",PROGRESS,true);

	AdExGroupNB *py_cells = setup_py_cells(nhcu*nmcu*npy);
	AdExGroupNB *ba_cells = setup_ba_cells(nhcu*nba);
	AdExGroupNB *db_cells = setup_db_cells(nhcu*nmcu*ndb);
	AdExGroupNB *vp_cells = setup_vp_cells(nhcu*nmcu*nvp);

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
	
	logger->msg("Setting up local py->py (ampa) connections ...",PROGRESS,true);
	STPConnection *py_pyLampa = new STPConnection(py_cells,py_cells,"WpypyLx.wij",GLUT);
	py_pyLampa->set_tau_d(tau_d); py_pyLampa->set_tau_f(tau_f); py_pyLampa->set_ujump(U);
	logger->msg("Setting up local py->py (nmda) connections ...",PROGRESS,true);
	SparseConnection *py_pyLnmda = new SparseConnection(py_cells,py_cells,"WpypyLx.wij",NMDA);
	py_pyLnmda->scale_all(1/ampa_nmda_ratio);

	logger->msg("Setting up global py->py (ampa) connections ...",PROGRESS,true);
	STPConnection *py_pyGampa = new STPConnection(py_cells,py_cells,"WpypyGx.wij",GLUT);
	py_pyGampa->set_tau_d(tau_d); py_pyGampa->set_tau_f(tau_f); py_pyGampa->set_ujump(U);
	logger->msg("Setting up global py->py (nmda) connections ...",PROGRESS,true);
	SparseConnection *py_pyGnmda = new SparseConnection(py_cells,py_cells,"WpypyGx.wij",NMDA);
	py_pyGnmda->scale_all(1/ampa_nmda_ratio);

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

	int nsyn_pypyLampa,gnsyn_pypyLampa,nsyn_pypyGampa,gnsyn_pypyGampa,
	  nsyn_pypyLnmda,gnsyn_pypyLnmda,nsyn_pypyGnmda,gnsyn_pypyGnmda,
	  nsyn_pyba,gnsyn_pyba,nsyn_bapy,gnsyn_bapy,
	  nsyn_pydb,gnsyn_pydb,nsyn_dbpy,gnsyn_dbpy,
	  nsyn_pyvp,gnsyn_pyvp,nsyn_vpdb,gnsyn_vpdb;

	/* Get total number of different types of synapses */
	nsyn_pypyLampa = py_pyLampa->get_nonzero();
	MPI_Reduce(&nsyn_pypyLampa,&gnsyn_pypyLampa,1,MPI_INT,MPI_SUM,0,*sys->get_com());

	nsyn_pypyGampa = py_pyGampa->get_nonzero();
	MPI_Reduce(&nsyn_pypyGampa,&gnsyn_pypyGampa,1,MPI_INT,MPI_SUM,0,*sys->get_com());

	nsyn_pypyLnmda = py_pyLnmda->get_nonzero();
	MPI_Reduce(&nsyn_pypyLnmda,&gnsyn_pypyLnmda,1,MPI_INT,MPI_SUM,0,*sys->get_com());

	nsyn_pypyGnmda = py_pyGnmda->get_nonzero();
	MPI_Reduce(&nsyn_pypyGnmda,&gnsyn_pypyGnmda,1,MPI_INT,MPI_SUM,0,*sys->get_com());

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

		std::cerr << "N:o py->py local ampa weights = " << gnsyn_pypyLampa << std::endl;
		
		std::cerr << "N:o py->py local ampa weights = " << gnsyn_pypyGampa << std::endl;
		
		std::cerr << "N:o py->py local nmda weights = " << gnsyn_pypyLnmda << std::endl;
		
		std::cerr << "N:o py->py local nmda weights = " << gnsyn_pypyGnmda << std::endl;
		
		std::cerr << "N:o py->ba weights = " << gnsyn_pyba << std::endl;
		
		std::cerr << "N:o ba->py weights = " << gnsyn_bapy << std::endl;
		
		std::cerr << "N:o py->db weights = " << gnsyn_pydb << std::endl;
		
		std::cerr << "N:o db->py weights = " << gnsyn_dbpy << std::endl;
		
		std::cerr << "N:o py->vp weights = " << gnsyn_pyvp << std::endl;
		
		std::cerr << "N:o vp->db weights = " << gnsyn_vpdb << std::endl;
		
	    std::cerr << "Maximum send buffer size: " << sys->get_max_send_buffer_size() << std::endl;

	}

}

bool getrndpair(SparseConnection *spcon,NeuronID &i,NeuronID &j) {

	/* Find one connected pair randomly */

	int prN = spcon->src->get_pre_size(),poN = spcon->dst->get_post_size(),n = 0;
	std::vector<neuron_pair> vpdb_pair;

	int k = -1;
	do {
	    i = rand()%prN;
	    vpdb_pair = spcon->get_post_partners(i);
	    if (vpdb_pair.size()==0) continue;
	    k = vpdb_pair[rand()%vpdb_pair.size()].j;
	} while (n++<1000 and k<0) ;

	j = k;
	//	std::cout << "i = " << i << " j = " << j << " n = " << n << " l = " << vpdb_pair.size() << std::endl;

	return 0<=k;

}


void tofile(SparseConnection *spcon,int row1,int row2,int col1,int col2) {

    FILE *fp = fopen("wfile.txt","w");
    AurynWeight w;

    for (int r=row1; r<row2; r++) {
	for (int c=col1; c<col2; c++) {
	    w = spcon->get(r,c);
	    fprintf(fp,"%f ",w);
	    printf("r = %d c = %d w = %f\n",r,c,w);
	}
	fprintf(fp,"\n");
    }
    fclose(fp);

}

int test2() {

	/* Here the same networks as in test1 is probed for synaptic interactions */

	logger->msg("Setting up neuron groups ...",PROGRESS,true);

	AdExGroupNB *py_cells = setup_py_cells(nhcu*nmcu*npy);
	AdExGroupNB *ba_cells = setup_ba_cells(nhcu*nba);
	AdExGroupNB *db_cells = setup_db_cells(nhcu*nmcu*ndb);
	AdExGroupNB *vp_cells = setup_vp_cells(nhcu*nmcu*nvp);

	/* NOTE: According to the manual on Loading connections from file,
	   the file given to SparseConnection "should only contain the
	   elements of the connection matrix that are also stored on
	   the respective rank where the code is issued. This is not
	   the case now ... */

	logger->msg("Setting up local py->py (ampa) connections ...",PROGRESS,true);
	STPConnection *py_pyLampa = new STPConnection(py_cells,py_cells,"WpypyLx.wij",GLUT);
	py_pyLampa->set_tau_d(tau_d); py_pyLampa->set_tau_f(tau_f); py_pyLampa->set_ujump(U);
	logger->msg("Setting up local py->py (nmda) connections ...",PROGRESS,true);
	SparseConnection *py_pyLnmda = new SparseConnection(py_cells,py_cells,"WpypyLx.wij",NMDA);
	py_pyLnmda->scale_all(1/ampa_nmda_ratio);

	logger->msg("Setting up global py->py (ampa) connections ...",PROGRESS,true);
	STPConnection *py_pyGampa = new STPConnection(py_cells,py_cells,"WpypyGx.wij",GLUT);
	py_pyGampa->set_tau_d(tau_d); py_pyGampa->set_tau_f(tau_f); py_pyGampa->set_ujump(U);
	logger->msg("Setting up global py->py (nmda) connections ...",PROGRESS,true);
	SparseConnection *py_pyGnmda = new SparseConnection(py_cells,py_cells,"WpypyGx.wij",NMDA);
	py_pyGnmda->scale_all(1/ampa_nmda_ratio);

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

	NeuronID pypyG_pr,pypyG_po,pypyL_pr,pypyL_po;
	bool w;
	w = getrndpair(py_pyGampa,pypyG_pr,pypyG_po);
	if (w) {
	    std::cout << "pypyG: pr = " << pypyG_pr << " po = " << pypyG_po << " w = " << w << " monitored\n";
	    VoltageMonitor *vmon_pypyG_pr = new VoltageMonitor(py_cells,pypyG_pr,sys->fn("pypyG_pr.vmem"));
	    VoltageMonitor *vmon_pypyG_po = new VoltageMonitor(py_cells,pypyG_po,sys->fn("pypyG_po.vmem"));
	}

	// Should perhaps check that pypyL_pr is not measured from above 

	w = getrndpair(py_pyLampa,pypyL_pr,pypyL_po);
	if (w) {
	    std::cout << "pypyL: pr = " << pypyL_pr << " po = " << pypyL_po << " w = " << w << " monitored\n";
	    VoltageMonitor *vmon_pypyL_pr = new VoltageMonitor(py_cells,pypyL_pr,sys->fn("pypyL_pr.vmem"));
	    VoltageMonitor *vmon_pypyL_po = new VoltageMonitor(py_cells,pypyL_po,sys->fn("pypyL_po.vmem"));
	}

	NeuronID pyba_pr,pyba_po,bapy_pr,bapy_po;

	w = getrndpair(py_ba,pyba_pr,pyba_po);
	if (w) {
	    std::cout << "pyba: pr = " << pyba_pr << " po = " << pyba_po << " w = " << w << " monitored\n";
	    VoltageMonitor *vmon_pyba_pr = new VoltageMonitor(py_cells,pyba_pr,sys->fn("pyba_pr.vmem"));
	    VoltageMonitor *vmon_pyba_po = new VoltageMonitor(ba_cells,pyba_po,sys->fn("pyba_po.vmem"));
	}
	
	w = getrndpair(ba_py,bapy_pr,bapy_po);
	if (w) {
	    std::cout << "bapy: pr = " << bapy_pr << " po = " << bapy_po << " w = " << w << " monitored\n";
	    VoltageMonitor *vmon_bapy_pr = new VoltageMonitor(ba_cells,bapy_pr,sys->fn("bapy_pr.vmem"));
	    VoltageMonitor *vmon_bapy_po = new VoltageMonitor(py_cells,bapy_po,sys->fn("bapy_po.vmem"));
	}

	NeuronID pydb_pr,pydb_po,dbpy_pr,dbpy_po;

	w = getrndpair(py_db,pydb_pr,pydb_po);
	if (w) {
	    std::cout << "pydb: pr = " << pydb_pr << " po = " << pydb_po << " w = " << w << " monitored\n";
	    VoltageMonitor *vmon_pydb_pr = new VoltageMonitor(py_cells,pydb_pr,sys->fn("pydb_pr.vmem"));
	    VoltageMonitor *vmon_pydb_po = new VoltageMonitor(db_cells,pydb_po,sys->fn("pydb_po.vmem"));
	}
	
	w = getrndpair(db_py,dbpy_pr,dbpy_po);
	if (w) {
	    std::cout << "dbpy: pr = " << dbpy_pr << " po = " << dbpy_po << " w = " << w << " monitored\n";
	    VoltageMonitor *vmon_dbpy_pr = new VoltageMonitor(db_cells,dbpy_pr,sys->fn("dbpy_pr.vmem"));
	    VoltageMonitor *vmon_dbpy_po = new VoltageMonitor(py_cells,dbpy_po,sys->fn("dbpy_po.vmem"));
	}

	NeuronID pyvp_pr,pyvp_po,vpdb_pr,vpdb_po;

	w = getrndpair(py_vp,pyvp_pr,pyvp_po);

	if (w) {
	    std::cout << "pyvp: pr = " << pyvp_pr << " po = " << pyvp_po << " w = " << w << " monitored\n";
	    VoltageMonitor *vmon_pyvp_pr = new VoltageMonitor(py_cells,pyvp_pr,sys->fn("pyvp_pr.vmem"));
	    VoltageMonitor *vmon_pyvp_po = new VoltageMonitor(vp_cells,pyvp_po,sys->fn("pyvp_po.vmem"));
	}

	w = getrndpair(vp_db,vpdb_pr,vpdb_po);
	if (w) {
	    std::cout << "vpdb: pr = " << vpdb_pr << " po = " << vpdb_po << " w = " << w << " monitored\n";
	    VoltageMonitor *vmon_vpdb_pr = new VoltageMonitor(vp_cells,vpdb_pr,sys->fn("vpdb_pr.vmem"));
	    VoltageMonitor *vmon_vpdb_po = new VoltageMonitor(db_cells,vpdb_po,sys->fn("vpdb_po.vmem"));
	}
	
	/* Current injections */

	CurrentInjector * curinj_py1 = new CurrentInjector(py_cells);
	CurrentInjector * curinj_py2 = new CurrentInjector(py_cells);

	CurrentInjector * curinj_py3 = new CurrentInjector(py_cells);
	CurrentInjector * curinj_ba1 = new CurrentInjector(ba_cells);

	CurrentInjector * curinj_py4 = new CurrentInjector(py_cells);
	CurrentInjector * curinj_db1 = new CurrentInjector(db_cells);

	CurrentInjector * curinj_py5 = new CurrentInjector(py_cells);
	CurrentInjector * curinj_vp1 = new CurrentInjector(vp_cells);

	logger->msg("Simulating ..." ,PROGRESS,true);

	MPI_Barrier(MPI::COMM_WORLD);

	sys->run(0.1);

	curinj_py1->set_current(pypyG_pr,injcur);
	curinj_py2->set_current(pypyL_pr,injcur);

	curinj_py3->set_current(pyba_pr,injcur);
	curinj_ba1->set_current(bapy_pr,injcur);

	curinj_py4->set_current(pydb_pr,injcur);
	curinj_db1->set_current(dbpy_pr,injcur);

	curinj_py5->set_current(pyvp_pr,injcur);
	curinj_vp1->set_current(vpdb_pr,injcur);

	sys->run(injdur);

	curinj_py1->set_all_currents(0);
	curinj_py2->set_all_currents(0);

	curinj_py3->set_all_currents(0);
	curinj_ba1->set_all_currents(0);

	curinj_py4->set_all_currents(0);
	curinj_db1->set_all_currents(0);

	curinj_py5->set_all_currents(0);
	curinj_vp1->set_all_currents(0);
	
	sys->run(simtime - 0.1 - injdur);

	if ( !save.empty() ) {
		sys->save_network_state(save);
	}

	if (errcode)
		auryn_abort(errcode);

}

int main(int ac,char *av[]) {

	processargs(ac,av);

	auryn_init(ac, av);

	std::cout << "seed = " << seed << " --> ";

	if (seed<0) {
	    seed = time(0);
	    srand(seed);
	    printf("seed = %u\n",seed);
	} else
		srand(seed);
	
	sys->set_master_seed(seed);

	sys->set_simulation_name("minicols");
	oss << dir  <<  monitor << sys->mpi_rank() << ".";
	outputfile = oss.str();

	std::cout << "runmode = " << runmode << std::endl;

	switch (runmode) {
	case 1: test1(); break;
	case 2: test2(); break;
	}	 

	if (sys->mpi_rank()==0) {
		std::cout << "\nN:o pyramidal cells = " << nhcu*nmcu*npy<< " (" << npy*nmcu << " per hcu)" << std::endl;
		std::cout << "N:o basket cells = " << nhcu*nba << " (" << nba << " per hcu)" << std::endl;

		std::cout << "N:o double bouquet cells = " << nhcu*nmcu*ndb << " (" << ndb*nmcu << " per hcu)" << std::endl;

		std::cout << "N:o vip cells = " << nhcu*nmcu*nvp << " (" << nvp*nmcu << " per hcu)" << std::endl;
	}

	if (sys->mpi_rank()==0) std::cout << std::endl;

	auryn_free();

	return errcode;
}
