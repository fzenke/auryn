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

#ifndef SYSTEM_H_
#define SYSTEM_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "Device.h"
#include "SpikingGroup.h"
#include "Connection.h"
#include "Checker.h"
#include "AurynVersion.h"


#ifdef AURYN_CODE_USE_MPI
#include <boost/mpi.hpp>
#include "SyncBuffer.h"
#endif //AURYN_CODE_USE_MPI

#include <ctime>
#include <vector>

#include <boost/timer/timer.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#define PROGRESSBAR_DEFAULT_UPDATE_INTERVAL 1000
#define LOGGER_MARK_INTERVAL (10000*1000) // 1000s

namespace auryn {

// class SpikingGroup; // forward declaration
// class Connection; // forward declaration
	
/*! \brief Class that implements system wide variables and methods to manage and run simulations.
 *
 * This Class contains methods to manage and run sets of classes that make up
 * the simulation. In particular it distinguishes between constituents of types
 * SpikingGroup, Connection, Monitor and Checker. A MPI implementation should
 * implement communicators and all that stuff in here. All the constituent
 * object of a single simulation are stored in STL vectors. The methods
 * evolve() and propagate() from each object in these vectors are called
 * alternatingly from within the run procedure.
 */
	class System
	{

	private:
		AurynTime clock;
#ifdef AURYN_CODE_USE_MPI
		mpi::environment * mpienv;
		mpi::communicator * mpicom;
		SyncBuffer * syncbuffer;
#endif // AURYN_CODE_USE_MPI

		unsigned int mpi_size_;
		unsigned int mpi_rank_;

		std::string simulation_name;

		std::vector<SpikingGroup *> spiking_groups;
		std::vector<Connection *> connections;
		std::vector<Device *> devices;
		std::vector<Checker *> checkers;

		string outputdir;

		boost::mt19937 seed_gen; 
		boost::random::uniform_int_distribution<> * seed_dist;
		boost::variate_generator<boost::mt19937&, boost::random::uniform_int_distribution<> > * seed_die;

		double simulation_time_realtime_ratio;

		/*! \brief Store elapsed time for last call of run */
		double last_elapsed_time;

		/*! \brief Store staring time of program */
		time_t t_sys_start;

		int online_rate_monitor_id;
		SpikingGroup * online_rate_monitor_target;
		double online_rate_monitor_tau;
		double online_rate_monitor_mul;
		double online_rate_monitor_state;

		/*! \brief Evolves the online rate monitor for the status bar. */
		void evolve_online_rate_monitor();

		/*! \brief Returns string with a human readable time. */
		std::string get_nice_time ( AurynTime clk );	

		/*! \brief Draws the progress bar */
		void progressbar ( double fraction, AurynTime clk );	

		/*! \brief Run simulation with given start and stop times and displays a progress bar. Mainly for internal use in System.
		 * \param starttime Start time used for display of progress bar.
		 * \param stoptime Stop time used for the display of the progress bar. The simulation stop when get_clock()>stoptime (this is the relevant one for the simulation)
		 * \param simulation_time again a time used for the display of the progress bar */
		bool run(AurynTime starttime, AurynTime stoptime, AurynFloat total_time, bool checking=true);

		/*! \brief Synchronizes SpikingGroups */
		void sync();

		/*! \brief Evolves all objects that need integration. */
		void evolve();

		/*! \brief Propagates the spikes and evolves connection objects. */
		void propagate();


		/*! \brief Performs integration of Connection objects. 
		 *
		 * Since this is independent of the SpikingGroup evolve we 
		 * can do this while we are waiting for synchronization. */
		void evolve_connections();

		/*! \brief Calls all monitors. */
		void execute_devices();

		/*! \brief Calls all checkers. */
		bool execute_checkers();

		/*! Implements integration and spike propagation of a single integration step. */
		void step();

		/*! Initialializes the recvs for all the MPI sync */
		void sync_prepare();

		void init();
		void free();

	public:
		/*! \brief Switch to turn output to quiet mode (no progress bar). */
		bool quiet;

		/*! \brief Version info */
		AurynVersion build;

		/*! \brief The progressbar update interval in timesteps of auryn_timestep. */
		unsigned int progressbar_update_interval;

#ifdef AURYN_CODE_USE_MPI
		/*! \brief Default constructor for MPI enabled. */
		System(mpi::environment * environment, mpi::communicator * communicator);
#else
		/*! \brief Default constructor for MPI disabled. */
		System();
#endif // AURYN_CODE_USE_MPI

		/*! \brief Sets the simulation name. */
		void set_simulation_name(std::string name);

		/*! \brief Returns the simulation name. */
		std::string get_simulation_name();

		/*! \brief Set output dir for fn function */
		void set_output_dir(std::string path);

		virtual ~System();


		/*! \brief Sets the target group for online rate estimate */
		void set_online_rate_monitor_target( SpikingGroup * group = NULL );

		/*! \brief Sets the SpikingGroup used to display the rate estimate in the
		 * progressbar 
		 *
		 * \deprecated This function should not be used any more. Use
		 * set_online_rate_monitor_target instead.
		 *
		 * This typically is reflected by the order in
		 * which you define the SpikingGroup and NeuronGroup classes. It starts
		 * numbering from 0. */
		void set_online_rate_monitor_id( unsigned int id=0 );

		/*! \brief Sets the timeconstant to compute the online rate average for the status bar. */
		void set_online_rate_monitor_tau( AurynDouble tau=100e-3 );

		/*! \brief Returns last elapsed time in seconds. */
		double get_last_elapsed_time();

		/*! \brief Returns total elapsed time in seconds. */
		double get_total_elapsed_time();

		/*! \brief Saves network state to a netstate file
		 *
		 * This function saves the network state to one serialized file. The network 
		 * state includes the internal state variables of all neurons and the synaptic 
		 * connections. It currently does not save the state of any random number
		 * generators (v0.5) but this is planned to change in the future. Note that
		 * netstate files do not contain any parameters either. This was done to
		 * allow to run a simulation with a certain parameter set for a given amount
		 * of time. Save the network state and then continue the simulation from
		 * that point with a changed parameter set (e.g. a new stimulus set or
		 * similar).
		 *
		 * \param basename (including directory path) of the netstate file without extension
		 */
		void save_network_state(std::string basename);

		/*! \brief Loads network state from a netstate file
		 *
		 * \param basename (directory and prefix of file) of the netstate file without extension
		 */
		void load_network_state(std::string basename);

		/*! \brief Saves the network state to human readable text files
		 *
		 * This deprecated method of saving the network state generates a large number of files 
		 * because each Connection object or SpikingGroup creates their own respective file. This 
		 * function might still be useful if you have code in which you analaze these files offline.
		 * In most cases you will want to use save_network_state and only dump a limited subset (e.g. 
		 * all the plastic connections) in human-readable text files for analysis.
		 *
		 * \param Basename (directory and prefix of file) of the netstate file without extension
		 */
		void save_network_state_text(std::string basename);


		/*! \brief Flush devices 
		 *
		 * Write monitor data buffers to file. */
		void flush_devices();


		/*! \brief Registers an instance of SpikingGroup to the spiking_groups vector. 
		 *
		 * Called internally by constructor of SpikingGroup. */
		void register_spiking_group(SpikingGroup * spiking_group);

		/*! \brief Registers an instance of Connection to the connections vector. 
		 *
		 * Called internally by constructor of Connection. */
		void register_connection(Connection * connection);

		/*! \brief Registers an instance of Device to the devices vector. 
		 *
		 * Called internally by constructor of Monitor. */
		void register_device(Device * device);

		/*! \brief Registers an instance of Checker to the checkers vector. 
		 * 
		 * Note: The first checker that is registered is by default used by System for the rate output in the progress bar. 
		 * Called internally by constructor of Monitor. */
		void register_checker(Checker * checker);


		/*! \brief Runs a simulation for a given amount of time.
		 *
		 * \param simulation_time time to run the simulation in seconds.
		 * \param checking true if checkers can break the run (e.g. if a
		 * frequency get's to high and the network explodes) 
		 * \return bool true on success
		 * */
		bool run(AurynFloat simulation_time, bool checking=true);

		/*! \brief Runs simulation for a given amount of time 
		 *
		 * Exposes the interface to the progressbar params. Can be used to run
		 * a single progress bar, but cut it in different chunks to turn on and
		 * off stuff in the simulation without perturbing the output.
		 * interval_start and end define the total duration of the simulation
		 * (this is used to build the progress bar, while chung_time is the
		 * actual time that is simulated for each call.*/
		bool run_chunk(AurynFloat chunk_time, AurynFloat interval_start, AurynFloat interval_end, bool checking=true);

		/*! \brief Gets the current system time in [s] */
		AurynDouble get_time();

		/*! \brief Gets the current clock value in AurynTime */
		AurynTime get_clock();

		/*! \brief Gets a pointer to the current clock. */
		AurynTime * get_clock_ptr();

		/*! \brief Get total number of registered neurons */
		AurynLong get_total_neurons();

		/*! \brief Get total effective load */
		AurynDouble get_total_effective_load();

		/*! \brief Get total number of registered synapses */
		AurynLong get_total_synapses();

		/*! \brief Format output file name 
		 *
		 * Formats output files according to the following convention:
		 * <outputdir>/<name>.<rank>.<extension> where <name> is taken 
		 * from System->get_name()
		 * and returns it as a c string;
		 * */
		string fn(std::string extension);


		/*! \brief Format output file name 
		 *
		 * Formats output files according to the following convention:
		 * <outputdir>/<name>.<rank>.<extension>
		 * and returns it as a c string;
		 * */
		string fn(std::string name, std::string extension);

		/*! \brief Format output file name 
		 *
		 * Formats output files according to the following convention:
		 * <outputdir>/<name><index>.<rank>.<extension>
		 * and returns it as a c string;
		 * */
		string fn(std::string name, NeuronID index, std::string extension);


		/*! \brief Returns global mpi communicator */
#ifdef AURYN_CODE_USE_MPI
		mpi::communicator * get_com();
#endif // AURYN_CODE_USE_MPI

		/*! \brief Returns number of ranks
		 *
		 * like mpicom->size(), but also defined when run
		 * without mpi. */
		unsigned int mpi_size();

		/*! \brief Returns current rank
		 *
		 * like mpicom->rank(), but also defined when run
		 * without mpi. */
		unsigned int mpi_rank();

		/*! \brief Set master seed
		 *
		 * Set the master seed from which other seeds are drawn. 
		 * When the master seed is set to 0 a master seed is generated
		 * from ctime and will be different at each run.
		 * */
		void  set_master_seed( unsigned int seed = 123);

		/*! \brief Returns a random seed which is different on each rank
		 *
		 * */
		unsigned int get_seed();

		/*! \brief Returns a random seed which is the same on each rank 
		 *
		 * Can be used to synchronize randomness across ranks StimulusGroup etc
		 * */
		unsigned int get_synced_seed();

#ifdef CODE_COLLECT_SYNC_TIMING_STATS
		AurynDouble deltaT;
		AurynDouble measurement_start;
		AurynDouble get_relative_sync_time();
		AurynDouble get_sync_time();
		AurynDouble get_elapsed_wall_time();
		void reset_sync_time();
#endif

	};
} // end of namespace brackets

#endif /*SYSTEM_H_*/
