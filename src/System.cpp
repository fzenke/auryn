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

#include "System.h"

using namespace auryn;

void System::init() {
	clock = 0;
	quiet = false;
	set_simulation_name("default");
	set_output_dir(".");

	set_online_rate_monitor_tau();
	// assumes that we have at least one spiking group in the sim
	online_rate_monitor_id = 0; 
	online_rate_monitor_state = 0.0;


	progressbar_update_interval = PROGRESSBAR_DEFAULT_UPDATE_INTERVAL;

	syncbuffer = new SyncBuffer(mpicom);

	std::stringstream oss;
	oss << "Auryn version "
		<< get_version_string();

	oss << " ( compiled " << __DATE__ << " " << __TIME__ << " )";
	auryn::logger->msg(oss.str(),NOTIFICATION);

	oss.str("");
	oss << "Current AurynTime good for simulations up to "
		<< std::numeric_limits<AurynTime>::max()*dt << "s  "
		<< "( " << std::numeric_limits<AurynTime>::max()*dt/3600 << "h )";
	auryn::logger->msg(oss.str(),VERBOSE);

	oss.str("");
	oss << "Current NeuronID and sync are good for simulations up to "
		<< std::numeric_limits<NeuronID>::max()/MINDELAY << " cells.";
	auryn::logger->msg(oss.str(),VERBOSE);

}

string System::get_version_string()
{
	std::stringstream oss;
	oss << AURYNVERSION
		<< "."
		<< AURYNSUBVERSION;

	if ( AURYNREVISION ) {
		oss << "."
		<< AURYNREVISION;
	}

#ifdef AURYNVERSIONSUFFIX
	oss << AURYNVERSIONSUFFIX;
#endif

	return oss.str();
}

System::System()
{
	init();
}

System::System(mpi::communicator * communicator)
{
	mpicom = communicator;
	init();

	std::stringstream oss;

	if ( mpicom->size() > 0 ) {
		oss << "MPI run rank "
			<<  mpicom->rank() << " out of "
			<<  mpicom->size() << " ranks total.";
		auryn::logger->msg(oss.str(),NOTIFICATION);
	}

	if ( mpicom->size() > 0 && (mpicom->size() & (mpicom->size()-1)) ) {
		oss.str("");
		oss << "WARNING! The number of processes is not a power of two. "
			<< "This causes impaired performance or even crashes "
			<< "in some MPI implementations.";
		auryn::logger->msg(oss.str(),WARNING,true);
	}
}

System::~System()
{
	free();
}

void System::free() 
{
	for ( unsigned int i = 0 ; i < spiking_groups.size() ; ++i )
		delete spiking_groups[i];
	for ( unsigned int i = 0 ; i < connections.size() ; ++i )
		delete connections[i];
	for ( unsigned int i = 0 ; i < monitors.size() ; ++i )
		delete monitors[i];
	for ( unsigned int i = 0 ; i < checkers.size() ; ++i )
		delete checkers[i];

	spiking_groups.clear();
	connections.clear();
	monitors.clear();
	checkers.clear();

	delete syncbuffer;
}

void System::step()
{
	clock++;
}

AurynDouble System::get_time()
{
	return dt * clock;
}

AurynTime System::get_clock()
{
	return clock;
}

AurynTime * System::get_clock_ptr()
{
	return &clock;
}

AurynLong System::get_total_neurons()
{
	AurynLong sum = 0;
	std::vector<SpikingGroup *>::const_iterator iter;
	for ( iter = spiking_groups.begin() ; iter != spiking_groups.end() ; ++iter ) {
		sum += (*iter)->get_rank_size(); 
	}
	return sum;
}

AurynDouble System::get_total_effective_load()
{
	AurynDouble sum = 0.;
	std::vector<SpikingGroup *>::const_iterator iter;
	for ( iter = spiking_groups.begin() ; iter != spiking_groups.end() ; ++iter ) {
		sum += (*iter)->get_effective_load(); 
	}
	return sum;
}

AurynLong System::get_total_synapses()
{
	AurynLong sum = 0;
	std::vector<Connection *>::const_iterator iter;
	for ( iter = connections.begin() ; iter != connections.end() ; ++iter ) {
		sum += (*iter)->get_nonzero(); 
	}
	return sum;
}

void System::register_spiking_group(SpikingGroup * spiking_group)
{
	spiking_groups.push_back(spiking_group);
	spiking_group->set_clock_ptr(get_clock_ptr());
}

void System::register_connection(Connection * connection)
{
	connections.push_back(connection);
}

void System::register_monitor(Monitor * monitor)
{
	monitors.push_back(monitor);
}

void System::register_checker(Checker * checker)
{
	checkers.push_back(checker);
}

void System::sync()
{

#ifdef CODE_COLLECT_SYNC_TIMING_STATS
	double T1, T2;              
    T1 = MPI_Wtime();     /* start time */
#endif

	std::vector<SpikingGroup *>::const_iterator iter;
	for ( iter = spiking_groups.begin() ; iter != spiking_groups.end() ; ++iter ) 
		syncbuffer->push((*iter)->delay,(*iter)->get_size()); 

	// // use this to artificially worsening the sync
	// struct timespec tim, tim2;
	// tim.tv_sec = 0;
	// tim.tv_nsec = 500000L;
	// nanosleep(&tim,&tim2);
	
	syncbuffer->sync();
	for ( iter = spiking_groups.begin() ; iter != spiking_groups.end() ; ++iter ) 
		syncbuffer->pop((*iter)->delay,(*iter)->get_size()); 

#ifdef CODE_COLLECT_SYNC_TIMING_STATS
    T2 = MPI_Wtime();     /* end time */
	deltaT += (T2-T1);
#endif
}

void System::evolve()
{
	std::vector<SpikingGroup *>::const_iterator iter;

	for ( iter = spiking_groups.begin() ; iter != spiking_groups.end() ; ++iter ) 
		(*iter)->conditional_evolve(); // evolve only if existing on rank

	// update the online rate estimate
	evolve_online_rate_monitor();
}

void System::evolve_independent()
{
	for ( std::vector<SpikingGroup *>::const_iterator iter = spiking_groups.begin() ; 
		  iter != spiking_groups.end() ; 
		  ++iter ) 
		(*iter)->evolve_traces(); // evolve only if existing on rank

	for ( std::vector<Connection *>::const_iterator iter = connections.begin() ; 
			iter != connections.end() ; 
			++iter )
		(*iter)->evolve(); 
}

void System::propagate()
{
	std::vector<Connection *>::const_iterator iter;
	for ( iter = connections.begin() ; iter != connections.end() ; ++iter )
		(*iter)->propagate(); 
}

bool System::monitor(bool checking)
{
	std::vector<Monitor *>::const_iterator iter;
	for ( iter = monitors.begin() ; iter != monitors.end() ; ++iter )
		(*iter)->propagate();

	for ( unsigned int i = 0 ; i < checkers.size() ; ++i )
		if (!checkers[i]->propagate() && checking) {
			std::stringstream oss;
			oss << "Checker " << i << " broke run!";
			auryn::logger->msg(oss.str(),WARNING);
			return false;
		}

	return true;
}


void System::progressbar ( double fraction, AurynTime clk ) {
	std::string bar;
	int percent = 100*fraction;
	const int division = 4;
	for(int i = 0; i < 100/division; i++) {
		if( i < (percent/division)){
			bar.replace(i,1,"=");
		}else if( i == (percent/division)){
		  bar.replace(i,1,">");
		}else{
		  bar.replace(i,1," ");
		}
	}

	std::cout << std::fixed << "\r" "[" << bar << "] ";
	std::cout.width( 3 );

	std::string time = get_nice_time(clk);

	std::cout<< percent << "%     "<< setiosflags(std::ios::fixed) << " t=" << time ;

	if ( online_rate_monitor_id >= 0 ) 
		std::cout  << std::setprecision(1) << "  f=" << online_rate_monitor_state << " Hz  ";

	std::cout << std::flush;

	if ( fraction >= 1. )
		std::cout << std::endl;
}

std::string System::get_nice_time(AurynTime clk)
{
	const AurynTime hour = 3600/dt;
	const AurynTime day  = 24*hour;
	std::stringstream oss;
	if ( clk > day ) {
		int d = clk/day;
		oss << d <<"d ";
		clk -= d*day;
	}
	if ( clk > hour ) {
		int h = clk/hour;
		oss << h <<"h ";
		clk -= h*hour;
	}
	oss << std::fixed << std::setprecision(1) << clk*dt << "s";
	return oss.str();
}

bool System::run(AurynTime starttime, AurynTime stoptime, AurynFloat total_time, bool checking)
{
	// issue a warning if there are no units on the rank specified
	if ( get_total_neurons() == 0 ) {
		auryn::logger->msg("There are no units assigned to this rank!",WARNING);
	}


	double runtime = (stoptime - get_clock())*dt;

	std::stringstream oss;
	oss << "Simulation triggered ( " 
		<< "runtime=" << runtime << "s )";
	auryn::logger->msg(oss.str(),NOTIFICATION);

	oss.str("");
	oss	<< "On this rank: neurons_total="<< get_total_neurons() 
		<< ", effective_load=" << get_total_effective_load()
		<< ", synapses_total=" << get_total_synapses();
	auryn::logger->msg(oss.str(),SETTINGS);


	if (mpicom->rank() == 0) {
		AurynLong all_ranks_total_neurons;
		reduce(*mpicom, get_total_neurons(), all_ranks_total_neurons, std::plus<AurynLong>(), 0);

		AurynLong all_ranks_total_synapses;
		reduce(*mpicom, get_total_synapses(), all_ranks_total_synapses, std::plus<AurynLong>(), 0);

		oss.str("");
		oss	<< "On all ranks: neurons_total="<< all_ranks_total_neurons 
			<< ", synapses_total=" << all_ranks_total_synapses;
		auryn::logger->msg(oss.str(),SETTINGS);
	} else {
		reduce(*mpicom, get_total_neurons(), std::plus<AurynLong>(), 0);
		reduce(*mpicom, get_total_synapses(), std::plus<AurynLong>(), 0);
	}

#ifdef CODE_COLLECT_SYNC_TIMING_STATS
	syncbuffer->reset_sync_time();
#endif

	time_t t_sim_start;
	time(&t_sim_start);
	time_t t_last_mark = t_sim_start;

	while ( get_clock() < stoptime ) {

	    if ( (mpicom->rank()==0) && (not quiet) && ( (get_clock()%progressbar_update_interval==0) || get_clock()==(stoptime-1) ) ) {
			double fraction = 1.0*(get_clock()-starttime+1)*dt/total_time;
			progressbar(fraction,get_clock()); // TODO find neat solution for the rate
		}

		if ( get_clock()%LOGGER_MARK_INTERVAL==0 && get_clock()>0 ) // set a mark 
		{
			time_t t_now; 
			time(&t_now);
			double td = difftime(t_now,t_last_mark);
			t_last_mark = t_now;

			oss.str("");
			oss << "Mark set ("
				<< get_time()
				<< "s). Ran for " << td
				<< "s with SpeedFactor=" << td/(LOGGER_MARK_INTERVAL*dt);

			AurynTime simtime_left = total_time-dt*(get_clock()-starttime+1);
			AurynDouble remaining = simtime_left*td/(LOGGER_MARK_INTERVAL*dt)/60; // in minutes
			if ( remaining > 5 ) { // only show when more than 5min
			oss	<< ", approximately "
				<< remaining
				<< "min of runtime remaining";
			}

			auryn::logger->msg(oss.str(),NOTIFICATION);
		}

		evolve();
		propagate();

		if (!monitor(checking))
			return false;

		evolve_independent(); // used to run in parallel to the sync (and could still in principle)
		// what is important for event based integration such as done in LinearTrace that this stays
		// on the same side of step() otherwise the results will be wrong (or evolve in LinearTrace has
		// to be adapted.
		
		step();	

		if ( mpicom->size()>1 && (get_clock())%(MINDELAY) == 0 ) {
			sync();
		} 

	}



#ifdef CODE_COLLECT_SYNC_TIMING_STATS
	double elapsed  = syncbuffer->get_elapsed_wall_time();

	oss.str("");
	oss << "Total synctime " 
		<< get_sync_time()
		<< " (relative "
		<< get_relative_sync_time()
		<< " )";
	auryn::logger->msg(oss.str(),NOTIFICATION);

#else
	time_t t_now ;
	time(&t_now);
	double elapsed  = difftime(t_now,t_sim_start);
#endif 

	oss.str("");
	oss << "Simulation finished. Ran for " 
		<< elapsed 
		<< "s with SpeedFactor=" 
		<< elapsed/runtime
		<< " (network clock=" << get_clock() << ")";
	auryn::logger->msg(oss.str(),NOTIFICATION);


#ifdef CODE_COLLECT_SYNC_TIMING_STATS
	if (mpicom->rank() == 0) {
		double minimum;
		reduce(*mpicom, elapsed, minimum, mpi::minimum<double>(), 0);

		double minsync;
		reduce(*mpicom, syncbuffer->get_sync_time(), minsync, mpi::minimum<double>(), 0);

		double maxsync;
		reduce(*mpicom, syncbuffer->get_sync_time(), maxsync, mpi::maximum<double>(), 0);

		oss.str("");
		oss << "Fastest rank. Ran for " 
			<< elapsed 
			<< " with minsynctime " 
			<< minsync
			<< " and maxsynctime "
			<< maxsync;
		auryn::logger->msg(oss.str(),NOTIFICATION);
	} else {
		reduce(*mpicom, elapsed, mpi::minimum<double>(), 0);
		reduce(*mpicom, syncbuffer->get_sync_time(), mpi::minimum<double>(), 0);
		reduce(*mpicom, syncbuffer->get_sync_time(), mpi::maximum<double>(), 0);
	}
#endif 


	return true;
}

bool System::run(AurynFloat simulation_time, bool checking)
{

	AurynTime starttime = get_clock();
	AurynTime stoptime = get_clock() + (AurynTime) (simulation_time/dt);

	// throw an exception if the stoptime is post the range of AurynTime
	if ( get_time() + simulation_time > std::numeric_limits<AurynTime>::max()*dt ) {
		auryn::logger->msg("The requested simulation time exceeds the number of possible timesteps limited by AurynTime datatype.",ERROR);
		throw AurynTimeOverFlowException();
	}

	return run(starttime, stoptime, simulation_time, checking);
}

bool System::run_chunk(AurynFloat chunk_time, AurynFloat interval_start, AurynFloat interval_end, bool checking)
{
	AurynTime stopclock = get_clock()+chunk_time/dt;

	// throw an exception if the stoptime is post the range of AurynTime
	if ( interval_end > std::numeric_limits<AurynTime>::max()*dt ) {
		auryn::logger->msg("The requested simulation time exceeds the number of possible timesteps limited by AurynTime datatype.",ERROR);
		throw AurynTimeOverFlowException();
	}

	return run(interval_start/dt, stopclock, (interval_end-interval_start), checking);
}


mpi::communicator * System::get_com()
{
	return mpicom;
}

void System::set_simulation_name(std::string name)
{
	simulation_name = name;
}

void System::set_output_dir(std::string path)
{
	outputdir = path;
}

string System::fn(std::string name, std::string extension)
{
	std::stringstream oss;
	oss << outputdir << "/" << name
	<< "." << mpicom->rank()
	<< "." << extension;
	return oss.str();
}

void System::save_network_state(std::string basename)
{
	auryn::logger->msg("Saving network state", NOTIFICATION);

	std::string netstate_filename;
	{
		std::stringstream oss;
		oss << outputdir 
			<< "/" << basename
			<< "." << mpicom->rank()
			<< ".netstate";
		netstate_filename = oss.str();
	} // oss goes out of focus

	auryn::logger->msg("Opening output stream ...",VERBOSE);
	std::ofstream ofs(netstate_filename.c_str());
	boost::archive::binary_oarchive oa(ofs);
	
	/* Translate version values to const values */
	const int auryn_version = AURYNVERSION;
	const int auryn_subversion = AURYNSUBVERSION;
	const int auryn_revision = AURYNREVISION;

	auryn::logger->msg("Saving version information ...",VERBOSE);
	// save simulator version information 
	oa << auryn_version;
	oa << auryn_subversion;
	oa << auryn_revision;

	auryn::logger->msg("Saving communicator information ...",VERBOSE);
	// save communicator information 
	int tmp_int = mpicom->size();
	oa << tmp_int;
	tmp_int = mpicom->rank();
	oa << tmp_int;


	auryn::logger->msg("Saving Connections ...",VERBOSE);
	for ( unsigned int i = 0 ; i < connections.size() ; ++i ) {

		std::stringstream oss;
		oss << "Saving connection "
			<<  i 
			<< " '"
			<< connections[i]->get_name()
			<< "' "
			<< " to stream";
		auryn::logger->msg(oss.str(),VERBOSE);

		oa << *(connections[i]);
	}

	auryn::logger->msg("Saving SpikingGroups ...",VERBOSE);
	for ( unsigned int i = 0 ; i < spiking_groups.size() ; ++i ) {

		std::stringstream oss;
		oss << "Saving SpikingGroup "
			<<  i 
			<< " ("
			<< spiking_groups[i]->get_name()
			<< ")"
			<< " to stream";
		auryn::logger->msg(oss.str(),VERBOSE);

		oa << *(spiking_groups[i]);
	}

	// Save Monitors
	auryn::logger->msg("Saving Monitors ...",VERBOSE);
	for ( unsigned int i = 0 ; i < monitors.size() ; ++i ) {

		std::stringstream oss;
		oss << "Saving Monitor "
			<<  i 
			<< " to stream";
		auryn::logger->msg(oss.str(),VERBOSE);

		oa << *(monitors[i]);
	}

	auryn::logger->msg("Saving Checkers ...",VERBOSE);
	for ( unsigned int i = 0 ; i < checkers.size() ; ++i ) {

		std::stringstream oss;
		oss << "Saving Checker "
			<<  i 
			<< " to stream";
		auryn::logger->msg(oss.str(),VERBOSE);

		oa << *(checkers[i]);
	}

	ofs.close();
}

void System::save_network_state_text(std::string basename)
{
	auryn::logger->msg("Saving network state to textfile", NOTIFICATION);

	char filename [255];
	for ( unsigned int i = 0 ; i < connections.size() ; ++i ) {
		sprintf(filename, "%s.%d.%d.wmat", basename.c_str(), i, mpicom->rank());

		std::stringstream oss;
		oss << "Saving connection "
			<<  filename ;
		auryn::logger->msg(oss.str(),VERBOSE);

		connections[i]->write_to_file(filename);
	}

	for ( unsigned int i = 0 ; i < spiking_groups.size() ; ++i ) {
		sprintf(filename, "%s.%d.%d.gstate", basename.c_str(), i, mpicom->rank());

		std::stringstream oss;
		oss << "Saving group "
			<<  filename ;
		auryn::logger->msg(oss.str(),VERBOSE);

		spiking_groups[i]->write_to_file(filename);
	}
}

void System::load_network_state(std::string basename)
{
	auryn::logger->msg("Loading network state", NOTIFICATION);


	std::string netstate_filename;
	{
		std::stringstream oss;
		oss << basename
			<< "." << mpicom->rank()
			<< ".netstate";
		netstate_filename = oss.str();
	} // oss goes out of focus

	std::ifstream ifs(netstate_filename.c_str());

	if ( !ifs.is_open() ) {
		std::stringstream oss;
		oss << "Error opening netstate file: "
			<< netstate_filename;
		auryn::logger->msg(oss.str(),ERROR);
		throw AurynOpenFileException();
	}

	boost::archive::binary_iarchive ia(ifs);


	// verify simulator version information 
	bool pass_version = true;
	int tmp_version;
	ia >> tmp_version;
	pass_version = pass_version && AURYNVERSION==tmp_version;
	ia >> tmp_version;
	pass_version = pass_version && AURYNSUBVERSION==tmp_version;
	ia >> tmp_version;
	pass_version = pass_version && AURYNREVISION==tmp_version;

	if ( !pass_version ) {
		auryn::logger->msg("WARNING: Version check failed! Current Auryn version " 
				"does not match the version which created the file. "
				"This could pose a problem. " 
				"Proceed with caution!" ,WARNING);
	}

	// verify communicator information 
	bool pass_comm = true;
	int tmp_int;
	ia >> tmp_int;
	pass_comm = pass_comm && (tmp_int == mpicom->size());
	ia >> tmp_int;
	pass_comm = pass_comm && (tmp_int == mpicom->rank());

	if ( !pass_comm ) {
		auryn::logger->msg("ERROR: Communicator size or rank do not match! "
				"Presumably you are trying to load the network "
				"state netstate from a simulation which was run "
				"on a different number of cores." ,ERROR);
	}  

	auryn::logger->msg("Loading connections ...",VERBOSE);
	for ( unsigned int i = 0 ; i < connections.size() ; ++i ) {

		std::stringstream oss;
		oss << "Loading connection "
			<<  i ;
		auryn::logger->msg(oss.str(),VERBOSE);

		ia >> *(connections[i]);
		connections[i]->finalize();
	}

	auryn::logger->msg("Loading SpikingGroups ...",VERBOSE);
	for ( unsigned int i = 0 ; i < spiking_groups.size() ; ++i ) {

		std::stringstream oss;
		oss << "Loading group "
			<<  i ;
		auryn::logger->msg(oss.str(),VERBOSE);

		ia >> *(spiking_groups[i]);
	}

	// Loading Monitors states
	auryn::logger->msg("Loading Monitors ...",VERBOSE);
	for ( unsigned int i = 0 ; i < monitors.size() ; ++i ) {

		std::stringstream oss;
		oss << "Loading Monitor "
			<<  i;
		auryn::logger->msg(oss.str(),VERBOSE);

		ia >> *(monitors[i]);
	}

	
	auryn::logger->msg("Loading Checkers ...",VERBOSE);
	for ( unsigned int i = 0 ; i < checkers.size() ; ++i ) {

		std::stringstream oss;
		oss << "Loading Checker "
			<<  i;
		auryn::logger->msg(oss.str(),VERBOSE);

		ia >> *(checkers[i]);
	}

	ifs.close();
}

void System::set_online_rate_monitor_tau(AurynDouble tau)
{
	online_rate_monitor_tau = tau;
	online_rate_monitor_mul = exp(-dt/tau);
}

void System::evolve_online_rate_monitor()
{
	if ( online_rate_monitor_id >= 0 ) {
		online_rate_monitor_state *= online_rate_monitor_mul;
		SpikingGroup * src = spiking_groups[online_rate_monitor_id];
		online_rate_monitor_state += 1.0*src->get_spikes()->size()/online_rate_monitor_tau/src->get_size();
	}
}

void System::set_online_rate_monitor_id( int id )
{
	online_rate_monitor_state = 0.0;
	if ( id < spiking_groups.size() ) 
		online_rate_monitor_id = id;
	else
		online_rate_monitor_id = -1;
}

#ifdef CODE_COLLECT_SYNC_TIMING_STATS
AurynDouble System::get_relative_sync_time()
{
	AurynDouble temp = MPI_Wtime();
	return (deltaT/(temp-measurement_start));
}

AurynDouble System::get_sync_time()
{
	return deltaT;
}

AurynDouble System::get_elapsed_wall_time()
{
	AurynDouble temp = MPI_Wtime();
	return temp-measurement_start;
}

void System::reset_sync_time()
{
	AurynDouble temp = MPI_Wtime();
    measurement_start = temp;     
	deltaT = 0.0;
}

#endif
