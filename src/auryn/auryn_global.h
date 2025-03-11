/* 
* Copyright 2014-2025 Friedemann Zenke
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

#ifndef AURYN_GLOBAL_H_
#define AURYN_GLOBAL_H_

#include <boost/filesystem.hpp>
#include "auryn_definitions.h"
#include "Logger.h"
#include "System.h"

// Global variables that are exported here.

namespace auryn {

	/*! \brief Global pointer to instance of System which needs to be initialized in every simulation main program. */
	extern System * sys;

	/*! \brief Global pointer to instance of Logger which needs to be initialized in every simulation main program. */
	extern Logger * logger;

#ifdef AURYN_CODE_USE_MPI
	/*! \brief Global pointer to instance of mpi::environment which needs to be initialized in every simulation main program. */
	extern mpi::environment * mpienv;

	/*! \brief Global pointer to instance of mpi::mpicommunicator which needs to be initialized in every simulation main program. */
	extern mpi::communicator * mpicommunicator;
#endif // AURYN_CODE_USE_MPI


	/*! \brief Initalizes MPI and the Auryn simulation environment. 
	 * 
	 * This function has to be called only once and before any Auryn class is used.
	 * It is thus typically invoked in the main function of your simulation program.
	 * The function initalizes up the global objects mpienv, mpicommunicator, sys and 
	 * logger. 
	 * It takes the command line parameters from your main method (needed to initializes 
	 * the MPI environment) as well as some additional arguments.
	 * You can for instance pass a default output directory which can be used by Device 
	 * instances or any class which writes to disk.
	 * Moreover, you can pass a simulation_name to auryn_init which will appear in logging
	 * output.
	 * Finally, you can explicitly pass a logfile_prefix string which Auryn will decorate with 
	 * the mpi rank number and a "log" extension. If this option is omitted, Auryn will derive
	 * a log file name from the name of your simulation binary.
	 * At the end of your simulation, make sure to call auryn_free() to cleanly terminate
	 * Auryn and avoid data loss.
	 *
	 * \param ac Number of command line parameters argc passed to main
	 * \param av Command line parameters passed as argv to main
	 * \param dir The default output directory for files generated by your simulation
	 * \param simulation_name A name for your simulation which will appear in logfiles
	 * \param logfile_prefix A file prefix (without path) which Auryn will use to generate
	 * a log file name.
	 * */
	void auryn_init(int ac, char* av[], string dir=".", string simulation_name="default", string logfile_prefix="", LogMessageType filelog_level=NOTIFICATION, LogMessageType consolelog_level=PROGRESS);

	/*! \brief Initalizes Auryn base environment (used internally)
	 *
	 * This function is called as port of auryn_init and is typically called internally. 
	 * It instantiates the global Logger object and initializes the MPI environment.
	 */
	void auryn_env_init(int ac, char* av[], string dir=".", string logfile_prefix="", LogMessageType filelog_level=NOTIFICATION, LogMessageType consolelog_level=PROGRESS);

	/*! \brief Initalizes the Auryn kernel (used internally)
	 *
	 * This function is called as port of auryn_init and is typically called internally.
	 * However, it can be used if several simulations are to be run from a single simulation binary.
	 * Then manual calles of auryn_kernel_init and auryn_kernel_free can be used to reinitalize the kernel
	 * without closing down the MPI environment.
	 */
	void auryn_kernel_init(string dir=".", string simulation_name="default");

	/*! \brief Cleanly shuts down Auryn simulation environment. 
	 *
	 * Deletes global variables mpienv, mpicommunicator, sys and logger and ensures 
	 * that all data is written to disk.*/
	void auryn_free();

	/*! \brief Frees logger and MPI 
	 *
	 * This function frees the MPI modules and logger. It is usually called by auryn_free()
	 * \see auryn_kernel_init
	 */
	void auryn_env_free();

	/*! \brief Frees the current auryn kernel (used interally)
	 *
	 * This function frees the current auryn kernel. It is usually called by auryn_free(), 
	 * but in some cases it may make sense to use this independently.
	 * \see auryn_kernel_init
	 */
	void auryn_kernel_free();

	/*! \brief Terminates Auryn simulation abnormally 
	 *
	 * This issues a term signal to all MPI processes in case of error, but first closes
	 * the Auryn kernel and terminates the logger to ensure all information of the issuing
	 * rank are written to disk.
	 *
	 * \param errcode The errorcode to be returned by the MPI processes
	 * */
	void auryn_abort(int errcode=0); 
}

#endif /*AURYN_GLOBAL_H__*/
