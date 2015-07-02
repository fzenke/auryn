/* 
* Copyright 2014-2015 Friedemann Zenke
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
#include "SpikingGroup.h"
#include "Connection.h"
#include "Monitor.h"
#include "Checker.h"
#include "SyncBuffer.h"

#include <ctime>

#include <vector>
#include <boost/mpi.hpp>
#include <boost/progress.hpp>

#define PROGRESSBAR_UPDATE_INTERVAL 1000
#define LOGGER_MARK_INTERVAL (10000*1000) // 1000s

using namespace std;
namespace mpi = boost::mpi;


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
	mpi::communicator * mpicom;
	string simulation_name;

	SyncBuffer * syncbuffer;


	vector<SpikingGroup *> spiking_groups;
	vector<Connection *> connections;
	vector<Monitor *> monitors;
	vector<Checker *> checkers;

	double simulation_time_realtime_ratio;

	int online_rate_monitor_id;
	double online_rate_monitor_tau;
	double online_rate_monitor_mul;
	double online_rate_monitor_state;

	/*! Evolves the online rate monitor for the status bar. */
	void evolve_online_rate_monitor();

	/*! Returns string with a human readable time. */
	string get_nice_time ( AurynTime clk );	

	/*! Draws the progress bar */
	void progressbar ( double fraction, AurynTime clk );	

	/*! Run simulation with given start and stop times and displays a progress bar. Mainly for internal use in System.
	 * \param starttime Start time used for display of progress bar.
	 * \param stoptime Stop time used for the display of the progress bar. The simulation stop when get_clock()>stoptime (this is the relevant one for the simulation)
	 * \param simulation_time again a time used for the display of the progress bar */
	bool run(AurynTime starttime, AurynTime stoptime, AurynFloat total_time, bool checking=true);

	/*! Synchronizes SpikingGroups */
	void sync();

	/*! Evolves all objects that need integration. */
	void evolve();

	/*! Propagates the spikes and evolves connection objects. */
	void propagate();

	/*! Performs integration of Connection objects. 
	 * Since this is independent of the SpikingGroup evolve we 
	 * can do this while we are waiting for synchronization. */
	void evolve_independent();

	/*! Calls all monitors. */
	bool monitor(bool checking);

public:
	/*! Switch to turn output to quiet mode (no progress bar). */
	bool quiet;

	System();
	System(mpi::communicator * communicator);
	void init();
	void set_simulation_name(string name);
	virtual ~System();
	void free();

	/*! Implements integration and spike propagation of a single integration step. */
	void step();

	/*! Initialializes the recvs for all the MPI sync */
	void sync_prepare();

	/*! Sets the SpikingGroup ID used to display the rate estimate in the
	 * progressbar (this typically is reflected by the order in
	 * which you define the SpikingGroup and NeuronGroup classes. It starts
	 * numbering from 0.). */
	void set_online_rate_monitor_id( int id=0 );

	/*! Sets the timeconstant to compute the online rate average for the status bar. */
	void set_online_rate_monitor_tau( AurynDouble tau=100e-3 );

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
	 * \param Prefix (including directory path) of the netstate file without extension
	 */
	void save_network_state(string basename);

	/*! \brief Loads network state from a netstate file
	 *
	 * \param Basename (directory and prefix of file) of the netstate file without extension
	 */
	void load_network_state(string basename);

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
	void save_network_state_text(string basename);


	/*! Registers an instance of SpikingGroup to the spiking_groups vector. */
	void register_spiking_group(SpikingGroup * spiking_group);

	/*! Registers an instance of Connection to the connections vector. */
	void register_connection(Connection * connection);

	/*! Registers an instance of Monitor to the monitors vector. */
	void register_monitor(Monitor * monitor);

	/*! Registers an instance of Checker to the checkers vector. 
	 * 
	 * Note: The first checker that is registered is by default used by System for the rate output in the progress bar.*/
	void register_checker(Checker * checker);


	/*! Run simulation for a given time.
	 * \param simulation_time time to run the simulation 
	 * \param checking true if checkers can break the run (e.g. if a frequency get's to high and the network explodes) */
	bool run(AurynFloat simulation_time, bool checking=true);

	/*! This and interface to run a single progress bar, but cut it in different chunks to turn on
	 * and off stuff in the simulation without perturbing the output. interval_start and end define 
	 * the total duration of the simulation (this is used to build the progress bar, while chung_time 
	 * is the actual time that is simulated for each call.*/
	bool run_chunk(AurynFloat chunk_time, AurynFloat interval_start, AurynFloat interval_end, bool checking=true);

	/*! Get the current system time in [s] */
	AurynDouble get_time();

	/*! Get the current clock value in AurynTime */
	AurynTime get_clock();

	/*! Get a pointer to the current clock. */
	AurynTime * get_clock_ptr();

	/*! Get total number of registered neurons */
	AurynLong get_total_neurons();

	/*! Get total effective load */
	AurynDouble get_total_effective_load();

	/*! Get total number of registered synapses */
	AurynLong get_total_synapses();


	mpi::communicator * get_com();

#ifdef CODE_COLLECT_SYNC_TIMING_STATS
	AurynDouble deltaT;
	AurynDouble measurement_start;
	AurynDouble get_relative_sync_time();
	AurynDouble get_sync_time();
	AurynDouble get_elapsed_wall_time();
	void reset_sync_time();
#endif

};

#endif /*SYSTEM_H_*/
