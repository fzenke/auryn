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
 * This simulates one hypercolumn with minicolumns and basket cells.
 *
 */

#include "auryn.h"

using namespace auryn;

namespace po = boost::program_options;

int main(int ac,char *av[]) {
	std::string dir = "./";
	std::string ras = "";

	std::stringstream oss;
	std::string msg;

	unsigned int seed = 1;

	NeuronID nhcu = 1;
	NeuronID nmcu = 1;
	NeuronID npy = 60; // N:o pyramidal cells per minicolumn
	NeuronID nba = 16; // N:o basket cells per minicolumn
	NeuronID ndb = 1; // N:o double bouguet cells per minicolumn
	NeuronID nvp = 1; // N:o vip cells per minicolumn
	NeuronID npo = 60; // N:o poisson cells per minicolumn

	int nrec = nhcu * nmcu * npy;

	std::vector<Connection *> corr_connections;

	double py_wbg = 0.030;
	double ba_wbg = 0.0225;
	double db_wbg = 0.0225;
	double vp_wbg = 0.0225;
	
	double tau_ad = 144e-3;
	double py_b = 0.02e-9;
	double ba_b = 0;
	double db_b = 0;
	double vp_b = 0;

	double simtime = 5;

	double poisson_rate = 100.0;

	std::string load = "";
	std::string save = "";

	int errcode = 0;

    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("seed", po::value<unsigned int>(), "master seed value")
            ("tau_ad", po::value<double>(), "tau value for adaptation")
            ("py_wbg", po::value<double>(), "weight from background poisson to pyramidalcells")
            ("ba_wbg", po::value<double>(), "weight from background poisson to basket cells")
            ("db_wbg", po::value<double>(), "weight from background poisson to double bouquet cells")
            ("vp_wbg", po::value<double>(), "weight from background poisson to vip cells")
            ("py_b", po::value<double>(), "b value for pyramidal cells")
            ("ba_b", po::value<double>(), "b value for basket cells")
            ("db_b", po::value<double>(), "b value for double bouquet cells")
            ("vp_b", po::value<double>(), "b value for vip cells")
            ("simtime", po::value<double>(), "duration of simulation")
            ("nhcu", po::value<int>(), "n:o hypercolumns in network")
            ("nmcu", po::value<int>(), "n:o minicolumns per hypercolumn")
            ("npy", po::value<int>(), "n:o pyramidal cells per minicolumn")
            ("nba", po::value<int>(), "n:o basket cells per minicolumn")
            ("ndb", po::value<int>(), "n:o double bouquet cells per minicolumn")
            ("nvp", po::value<int>(), "n:o vip cells per minicolumn")
            ("nrec", po::value<int>(), "n:o spike recorded cells")

            ("poisson_rate", po::value<double>(), "the background poisson firing rate")
            ("dir", po::value<std::string>(), "dir from file")
            ("ras", po::value<std::string>(), "if not "" produce spike raster and rate files")
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

        if (vm.count("tau_ad")) {
			tau_ad = vm["tau_ad"].as<double>();
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

        if (vm.count("dir")) {
			dir = vm["dir"].as<std::string>();
        } 

        if (vm.count("ras")) {
			ras = vm["ras"].as<std::string>();
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

	if (nrec<0) nrec = -nrec; // else nrec = npy * nmcu * nhcu;

	auryn_init(ac, av);
	sys->set_master_seed(seed);
	sys->set_simulation_name("minicol");
	oss << dir  << ras << sys->mpi_rank() << ".";
	std::string outputfile = oss.str();

	if (sys->mpi_rank()==0) {
		std::cout << "N:o pyramidal cells = " << nhcu*nmcu*npy<< " (" << npy*nmcu << " per hcu)" << std::endl;
		std::cout << "N:o basket cells = " << nhcu*nmcu*nba << " (" << nba*nmcu << " per hcu)" << std::endl;

		std::cout << "N:o double bouquet cells = " << nhcu*nmcu*ndb << " (" << ndb*nmcu << " per hcu)" << std::endl;

		std::cout << "N:o vip cells = " << nhcu*nmcu*nvp << " (" << nvp*nmcu << " per hcu)" << std::endl;
	}

	logger->msg("Setting up neuron groups ...",PROGRESS,true);

	AdExGroup *py_cells = new AdExGroup(nhcu*nmcu*npy);
	py_cells->set_refractory_period(0.005);
	py_cells->set_tau_w(tau_ad);
	py_cells->set_a(0);
	py_cells->set_b(py_b);

	AdExGroup *ba_cells = new AdExGroup(nhcu*nmcu*nba);
	ba_cells->set_refractory_period(0.005);
	ba_cells->set_tau_w(tau_ad);
	ba_cells->set_a(0);
	ba_cells->set_b(ba_b);

	AdExGroup *db_cells = new AdExGroup(nhcu*nmcu*ndb);
	db_cells->set_refractory_period(0.005);
	db_cells->set_tau_w(tau_ad);
	db_cells->set_a(0);
	db_cells->set_b(db_b);

	AdExGroup *vp_cells = new AdExGroup(nhcu*nmcu*nvp);
	vp_cells->set_refractory_period(0.005);
	vp_cells->set_tau_w(tau_ad);
	vp_cells->set_a(0);
	vp_cells->set_b(vp_b);

	std::cout << "Poisson rate " << poisson_rate;

	PoissonGroup *poisson = new PoissonGroup(npo,poisson_rate);
	SparseConnection *con_stim_py = new SparseConnection(poisson,py_cells,py_wbg,0.5,GLUT);
	SparseConnection *con_stim_ba = new SparseConnection(poisson,ba_cells,ba_wbg,0.5,GLUT);
	SparseConnection *con_stim_db = new SparseConnection(poisson,db_cells,db_wbg,0.5,GLUT);
	SparseConnection *con_stim_vp = new SparseConnection(poisson,vp_cells,vp_wbg,0.5,GLUT);

	logger->msg("Setting up local py->py connections ...",PROGRESS,true);
	SparseConnection *py_pyL = new SparseConnection(py_cells,py_cells,"WpypyLx.wij",GLUT);

	logger->msg("Setting up global py->py connections ...",PROGRESS,true);
	SparseConnection *py_pyG = new SparseConnection(py_cells,py_cells,"WpypyGx.wij",GLUT);

	logger->msg("Setting up py->ba connections ...",PROGRESS,true);
	SparseConnection *py_ba = new SparseConnection(py_cells,ba_cells,"Wpybax.wij",GLUT);

	logger->msg("Setting up ba->py connections ...",PROGRESS,true);
	SparseConnection *ba_py = new SparseConnection(ba_cells,py_cells,"Wbapyx.wij",GABA);

	logger->msg("Setting up py->db connections ...",PROGRESS,true);
	SparseConnection *py_db = new SparseConnection(py_cells,db_cells,"Wpydbx.wij",GLUT);

	logger->msg("Setting up db->py connections ...",PROGRESS,true);
	SparseConnection *db_py = new SparseConnection(db_cells,py_cells,"Wdbpyx.wij",GABA);

	logger->msg("Setting up py->vp connections ...",PROGRESS,true);
	SparseConnection *py_vp = new SparseConnection(py_cells,vp_cells,"Wpyvpx.wij",GLUT);

	logger->msg("Setting up vp->db connections ...",PROGRESS,true);
	SparseConnection *vp_db = new SparseConnection(vp_cells,db_cells,"Wvpdbx.wij",GABA);

	if (ras!="") {
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
		VoltageMonitor *vmon_py2 = new VoltageMonitor(py_cells,nmcu*npy-1,sys->fn("py2.vmem"));
		VoltageMonitor *vmon_ba1 = new VoltageMonitor(ba_cells,0,sys->fn("ba1.vmem"));
		VoltageMonitor *vmon_ba2 = new VoltageMonitor(ba_cells,nmcu*nba-1,sys->fn("ba2.vmem"));
		VoltageMonitor *vmon_db1 = new VoltageMonitor(db_cells,0,sys->fn("db1.vmem"));
		VoltageMonitor *vmon_db2 = new VoltageMonitor(db_cells,nmcu*ndb-1,sys->fn("db2.vmem"));
		VoltageMonitor *vmon_vp1 = new VoltageMonitor(vp_cells,0,sys->fn("vp1.vmem"));
		VoltageMonitor *vmon_vp = new VoltageMonitor(vp_cells,nmcu*nvp-1,sys->fn("vp2.vmem"));

	    // Record firing rates (sample every 50 ms)
	    PopulationRateMonitor *pmon_po = new PopulationRateMonitor(poisson,sys->fn("po_rate"),0.05);
	    PopulationRateMonitor *pmon_py = new PopulationRateMonitor(py_cells,sys->fn("py_rate"),0.05);
	    PopulationRateMonitor *pmon_ba = new PopulationRateMonitor(ba_cells,sys->fn("ba_rate"),0.05);
	    PopulationRateMonitor *pmon_db = new PopulationRateMonitor(db_cells,sys->fn("db_rate"),0.05);
	    PopulationRateMonitor *pmon_vp = new PopulationRateMonitor(vp_cells,sys->fn("vp_rate"),0.05);

	}

	std::ofstream outfile;
	string fname = "minicols.data";
	outfile.open(fname.c_str(),std::ios::out);
	if (!outfile) {
		std::cerr << "Can't open output file " << fname.c_str() << std::endl;
	  throw AurynOpenFileException();
	}
	outfile << nhcu*nmcu*npy << " " << nhcu*nmcu*nba << " " << nhcu*nmcu*ndb << " " << nhcu*nmcu*nvp << std::endl;
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

	auryn_free();

	return errcode;
}
