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
 * This simulates the network from Fiebig et al. (2017). A Spiking Working Memory Model Based on Hebbian Short-Term Potentiation. J. Neuroscience, 37(1): 83–96.
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

	unsigned int seed = 0;

	NeuronID nhcu = 7;
	NeuronID nmcu = 5;
	NeuronID npyr = 30; // N:o pyramidal cells per minicolumn
	NeuronID nbas = nmcu*npyr/5; // N:o basket cells per hypercolumn
	NeuronID ndbc = 1; // N:o double bouquet cells per minicolumn (assuming a hypercol has 5 minicols

	std::vector<Connection *> corr_connections;

	NeuronID nrec = 200;

	double py_wbg = 0.16;
	double ba_wbg = 0.16;
	double db_wbg = 0.16;
	
	double tau_ad = 144e-3;
	double py_b = 0.08e-9;
	double ba_b = 0.08e-9;
	double db_b = 0.08e-9;

	double simtime = 10.;

	double poisson_rate = 5.0;

	std::string load = "";
	std::string save = "";

	int errcode = 0;

    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("seed", po::value<unsigned int>(), "master seed value")
            ("tau_ad", po::value<double>(), "tau value for adaptation")
            ("py_wbg", po::value<double>(), "weight from background poisson to double bouquet cells")
            ("ba_wbg", po::value<double>(), "weight from background poisson to double bouquet cells")
            ("db_wbg", po::value<double>(), "weight from background poisson to double bouquet cells")
            ("py_b", po::value<double>(), "b value for pyramidal cells")
            ("ba_b", po::value<double>(), "b value for basket cells")
            ("db_b", po::value<double>(), "b value for double bouquet cells")
            ("simtime", po::value<double>(), "duration of simulation")
            ("nhcu", po::value<int>(), "n:o hypercolumns in network")
            ("nmcu", po::value<int>(), "n:o minicolumns per hypercolumn")
            ("npyr", po::value<int>(), "n:o pyramidal cells per minicolumn")
            ("nbas", po::value<int>(), "n:o basket cells per hypercolumn")
            ("ndbc", po::value<int>(), "n:o double bouquet cells per hypercolumn")

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

        if (vm.count("ba_b")) {
			ba_b = vm["ba_b"].as<double>();
        } 

        if (vm.count("db_b")) {
			db_b = vm["db_b"].as<double>();
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

        if (vm.count("npyr")) {
	    npyr = vm["npyr"].as<int>();
        } 

        if (vm.count("nbas")) {
	    nbas = vm["nbas"].as<int>();
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

	auryn_init(ac, av);
	sys->set_master_seed(seed);

	oss << dir  << ras << sys->mpi_rank() << ".";
	std::string outputfile = oss.str();

	if (sys->mpi_rank()==0) {
		std::cout << "N:o pyramidal cells = " << nhcu*nmcu*npyr<< " (" << npyr*nmcu << " per hcu)" << std::endl;
		std::cout << "N:o basket cells = " << nhcu*nbas << " (" << nbas << " per hcu)" << std::endl;
		std::cout << "N:o double bouquet cells = " << nhcu*nmcu*ndbc << " (" << nmcu*ndbc << " per hcu)" << std::endl;
	}

	logger->msg("Setting up neuron groups ...",PROGRESS,true);

	AdExGroup *py_cells = new AdExGroup(nhcu*nmcu*npyr);
	py_cells->set_refractory_period(0.005);
	py_cells->set_tau_w(tau_ad);
	py_cells->set_a(4e-9);
	py_cells->set_b(py_b);

	AdExGroup *ba_cells = new AdExGroup(nhcu*nbas);
	ba_cells->set_refractory_period(0.005);
	ba_cells->set_tau_w(tau_ad);
	ba_cells->set_a(0);
	ba_cells->set_b(ba_b);

	AdExGroup *db_cells = new AdExGroup(nhcu*nmcu*ndbc);
	db_cells->set_refractory_period(0.005);
	db_cells->set_tau_w(tau_ad);
	db_cells->set_a(4e-9);
	db_cells->set_b(db_b);

	double onperiod = 1;
	double offperiod = 1;

	std::string patfile = "pypatx.stim";

#ifdef STIMULUS
	Setting up stimulation

	logger->msg("Preparing stimulus ...",PROGRESS,true);

	StimulusGroup *stimgroup1 = new StimulusGroup(py_cells->get_size(),patfile,"",SEQUENTIAL,
												  0.0);
	stimgroup1->set_mean_on_period(onperiod);
	stimgroup1->set_mean_off_period(offperiod);

	stimgroup1->randomintervals = false;
	stimgroup1->randomintensities = false;

	IdentityConnection *con_stim1 = new IdentityConnection(stimgroup1,py_cells,py_wbg);

	stimgroup1->active = false;
#endif // STIMULUS
	
	std::cout << "Poisson rate " << poisson_rate;

	PoissonGroup *poisson = new PoissonGroup(py_cells->get_size(),poisson_rate);
	SparseConnection *con_stim_py = new SparseConnection(poisson,py_cells,py_wbg,0.5,GLUT);
	SparseConnection *con_stim_ba = new SparseConnection(poisson,ba_cells,ba_wbg,0.5,GLUT);
	SparseConnection *con_stim_db = new SparseConnection(poisson,db_cells,db_wbg,0.5,GLUT);

	logger->msg("Setting up global py->py connections ...",PROGRESS,true);
	SparseConnection *py_pyG = new SparseConnection(py_cells,py_cells,"WpypyGx.wij",GLUT);

	logger->msg("Setting up local py->py connections ...",PROGRESS,true);
	SparseConnection *py_pyL = new SparseConnection(py_cells,py_cells,"WpypyLx.wij",GLUT);

	logger->msg("Setting up py->ba connections ...",PROGRESS,true);
	SparseConnection *py_ba = new SparseConnection(py_cells,ba_cells,"Wpybax.wij",GLUT);

	logger->msg("Setting up ba->py connections ...",PROGRESS,true);
	SparseConnection *ba_py = new SparseConnection(ba_cells,py_cells,"Wbapyx.wij",GABA);

	logger->msg("Setting up py->db connections ...",PROGRESS,true);
	SparseConnection *py_db = new SparseConnection(py_cells,db_cells,"Wpydbx.wij",GLUT);

	logger->msg("Setting up py->db connections ...",PROGRESS,true);	logger->msg("Setting up db->py connections ...",PROGRESS,true);
	SparseConnection *db_py = new SparseConnection(db_cells,py_cells,"Wdbpyx.wij",GABA);

	if (ras!="") {
	    std::stringstream fname_py,fname_ba,fname_db;

	    fname_py << outputfile << "py.ras";
	    SpikeMonitor *smon_py = new SpikeMonitor(py_cells,fname_py.str().c_str(),nrec);
	    fname_ba << outputfile << "ba.ras";
	    SpikeMonitor *smon_ba = new SpikeMonitor(ba_cells,fname_ba.str().c_str(),nrec);
	    fname_db << outputfile << "db.ras";
	    SpikeMonitor *smon_db = new SpikeMonitor(db_cells,fname_db.str().c_str(),nrec);

		VoltageMonitor *vmon_py1 = new VoltageMonitor(py_cells,0,sys->fn("py1.vmem"));
		VoltageMonitor *vmon_py2 = new VoltageMonitor(py_cells,npyr,sys->fn("py2.vmem"));
		VoltageMonitor *vmon_ba = new VoltageMonitor(ba_cells,5,sys->fn("ba.vmem"));
		VoltageMonitor *vmon_db = new VoltageMonitor(db_cells,ndbc,sys->fn("db.vmem"));

	    // Record firing rates (sample every 50 ms)
	    PopulationRateMonitor *pmon_py = new PopulationRateMonitor(py_cells,sys->fn("py_rate"),0.05);
	    PopulationRateMonitor *pmon_ba = new PopulationRateMonitor(ba_cells,sys->fn("ba_rate"),0.05);
	    PopulationRateMonitor *pmon_db = new PopulationRateMonitor(db_cells,sys->fn("db_rate"),0.05);

	}

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

	int nsyn_pyba,gnsyn_pyba,nsyn_bapy,gnsyn_bapy,nsyn_pypyL,gnsyn_pypyL,nsyn_pypyG,gnsyn_pypyG,
		nsyn_pydb,gnsyn_pydb,nsyn_dbpy,gnsyn_dbpy;
	/* Get total number of different types of synapses */
	nsyn_pyba = py_ba->get_nonzero();
	MPI_Reduce(&nsyn_pyba,&gnsyn_pyba,1,MPI_INT,MPI_SUM,0,*sys->get_com());

	nsyn_bapy = ba_py->get_nonzero();
	MPI_Reduce(&nsyn_bapy,&gnsyn_bapy,1,MPI_INT,MPI_SUM,0,*sys->get_com());
	
	nsyn_pypyL = py_pyL->get_nonzero();
	MPI_Reduce(&nsyn_pypyL,&gnsyn_pypyL,1,MPI_INT,MPI_SUM,0,*sys->get_com());

	nsyn_pypyG = py_pyG->get_nonzero();
	MPI_Reduce(&nsyn_pypyG,&gnsyn_pypyG,1,MPI_INT,MPI_SUM,0,*sys->get_com());

	nsyn_pydb = py_db->get_nonzero();
	MPI_Reduce(&nsyn_pydb,&gnsyn_pydb,1,MPI_INT,MPI_SUM,0,*sys->get_com());

	nsyn_dbpy = db_py->get_nonzero();
	MPI_Reduce(&nsyn_dbpy,&gnsyn_dbpy,1,MPI_INT,MPI_SUM,0,*sys->get_com());
	
	if (sys->mpi_rank()==0) {
		std::cerr << "Execution time = " << MPI::Wtime() - start << " sec\n";

		std::cerr << "N:o py->ba weights = " << gnsyn_pyba << std::endl;
		
		std::cerr << "N:o ba->py weights = " << gnsyn_bapy << std::endl;
		
		std::cerr << "N:o py->py local weights = " << gnsyn_pypyL << std::endl;
		
		std::cerr << "N:o py->py long-range weights = " << gnsyn_pypyG << std::endl;
		
		std::cerr << "N:o py->db weights = " << gnsyn_pydb << std::endl;
		
		std::cerr << "N:o db->py weights = " << gnsyn_dbpy << std::endl;
		
	    std::cerr << "Maximum send buffer size: " << sys->get_max_send_buffer_size() << std::endl;

	}

	auryn_free();

	return errcode;
}
