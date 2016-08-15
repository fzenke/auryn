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
*
* If you are using Auryn or parts of it for your work please cite:
* Zenke, F. and Gerstner, W., 2014. Limits to high-speed simulations 
* of spiking neural networks using general-purpose computers. 
* Front Neuroinform 8, 76. doi: 10.3389/fninf.2014.00076
*/

#include "auryn_global.h"


namespace auryn {
	mpi::communicator * mpicommunicator;
	mpi::environment * mpienv;
	Logger * logger;
	System * sys;

	void auryn_init(int ac, char* av[], string dir, string simulation_name, string logfile_prefix )
	{
		// init MPI environment
		mpienv = new mpi::environment(ac, av); 
		mpicommunicator = new mpi::communicator(); 

		// Init logger environment
		try 
		{ 
			string log_prefix_ = boost::filesystem::basename(av[0]);
			log_prefix_.erase(std::remove(log_prefix_.begin(),log_prefix_.end(),' '),log_prefix_.end()); // remove spaces
			std::transform(log_prefix_.begin(), log_prefix_.end(), log_prefix_.begin(), ::tolower); // convert to lower case

			if ( !logfile_prefix.empty() ) log_prefix_ = logfile_prefix;

			char strbuf_tmp [255]; 
			sprintf(strbuf_tmp, "%s/%s.%d.log", dir.c_str(), log_prefix_.c_str(), mpicommunicator->rank()); 
			string auryn_simulation_logfile = strbuf_tmp; 
			logger = new Logger(auryn_simulation_logfile,mpicommunicator->rank(),PROGRESS,EVERYTHING); 
		} 
		catch ( AurynOpenFileException excpt ) 
		{ 
			std::cerr << "Cannot proceed without log file. Exiting all ranks ..." << std::endl; 
			mpienv->abort(1); 
		} 

		// Init Auryn Kernel
		auryn::sys = new System(mpicommunicator); 
		sys->set_output_dir(dir);
		sys->set_simulation_name(simulation_name);
	}

	void auryn_free()
	{
		delete sys;
		delete logger;
		delete mpicommunicator;
		delete mpienv;
	}

	void auryn_abort(int errcode) {
		delete sys;
		delete logger;
		mpienv->abort(errcode);
		// The above line should have killed this process by now, but just in case ..
		std::exit(errcode); 
	}
}
