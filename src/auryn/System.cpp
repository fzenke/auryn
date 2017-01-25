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


	std::stringstream oss;
	auryn::logger->msg("Starting Auryn Kernel",NOTIFICATION);

	oss << "Auryn version "
		<< build.get_version_string()
	    << " ( compiled " << __DATE__ << " " << __TIME__ << " )";
	auryn::logger->msg(oss.str(),NOTIFICATION);

	oss.str("");
	oss << "Git repository revision "
	    << build.git_describe;
	auryn::logger->msg(oss.str(),NOTIFICATION);


	clock = 0;

	// auryn_timestep = 1e-4;

	quiet = false;
	set_simulation_name("default");
	set_output_dir(".");

	set_online_rate_monitor_tau();
	// assumes that we have at least one spiking group in the sim
	online_rate_monitor_id = 0; 
	online_rate_monitor_state = 0.0;

	last_elapsed_time = -1.0;
	// remember starting time
	time(&t_sys_start);

	progressbar_update_interval = PROGRESSBAR_DEFAULT_UPDATE_INTERVAL;

	mpi_size_ = 1;
	mpi_rank_ = 0;
#ifdef AURYN_CODE_USE_MPI
	syncbuffer = NULL;
	if ( mpicom ) {
		syncbuffer = new SyncBuffer(mpicom);
		mpi_size_ = mpicom->size();
		mpi_rank_ = mpicom->rank();
	} 
#endif // AURYN_CODE_USE_MPI


	oss.str("");
	oss << "NeuronID type has size of "
		<< sizeof(NeuronID) << " bytes.";
	auryn::logger->msg(oss.str(),VERBOSE);

	oss.str("");
	oss << "AurynLong type has size of "
		<< sizeof(AurynLong) << " bytes.";
	auryn::logger->msg(oss.str(),VERBOSE);

	oss.str("");
	oss << "Current NeuronID and sync are good for simulations up to "
		<< std::numeric_limits<NeuronID>::max()-1 << " cells.";
	auryn::logger->msg(oss.str(),INFO);

	if ( sizeof(NeuronID) != sizeof(AurynFloat) ) {
		oss.str("");
		oss << " NeuronID and AurynFloat have different byte sizes which is not supported by SyncBuffer.";
		auryn::logger->msg(oss.str(),ERROR);
	}

	oss.str("");
	oss << "Simulation timestep is set to "
		<< std::scientific << auryn_timestep << "s  ";
	auryn::logger->msg(oss.str(),SETTINGS);

	oss.str("");
	oss << "Current AurynTime good for simulations up to "
		<< std::numeric_limits<AurynTime>::max()*auryn_timestep << "s  "
		<< "( " << std::numeric_limits<AurynTime>::max()*auryn_timestep/3600 << "h )";
	auryn::logger->msg(oss.str(),INFO);


	// init random number generator
	// gen  = boost::mt19937();
	dist = new boost::random::uniform_int_distribution<> ();
	die  = new boost::variate_generator<boost::mt19937&, boost::random::uniform_int_distribution<> > ( gen, *dist );
	unsigned int hardcoded_seed = 3521;
	set_master_seed(hardcoded_seed);



#ifndef NDEBUG
	oss.str("");
	oss << "Warning Auryn was compiled with debugging features which will impair performance.";
	auryn::logger->warning(oss.str());
#endif 

#ifdef AURYN_CODE_USE_MPI
	auryn::logger->msg("Auryn was compiled with MPI support.",NOTIFICATION);
#else
	auryn::logger->msg("Auryn was compiled without MPI support.",NOTIFICATION);
#endif // AURYN_CODE_USE_MPI

}



#ifdef AURYN_CODE_USE_MPI
System::System(mpi::communicator * communicator)
{

	mpicom = communicator;
	init();


	std::stringstream oss;
	if ( mpi_size() > 1 ) {
		oss << "This is an MPI run. I am rank "
			<<  mpi_rank() << " of a total of "
			<<  mpi_size() << " ranks.";
		auryn::logger->msg(oss.str(),NOTIFICATION);
	} else {
		auryn::logger->msg("Not running a parallel simulation.",NOTIFICATION);
	}

	if ( mpi_size() > 0 && (mpi_size() & (mpi_size()-1)) ) {
		oss.str("");
		oss << "WARNING! The number of processes is not a power of two. "
			<< "This causes impaired performance or even crashes "
			<< "in some MPI implementations.";
		auryn::logger->msg(oss.str(),WARNING,true);
	}
}
#else
System::System()
{
	init();
}
#endif // AURYN_CODE_USE_MPI

System::~System()
{
	free();
}

void System::free() 
{
	for ( unsigned int i = 0 ; i < checkers.size() ; ++i )
		delete checkers[i];
	for ( unsigned int i = 0 ; i < devices.size() ; ++i )
		delete devices[i];
	for ( unsigned int i = 0 ; i < connections.size() ; ++i ) {
		std::stringstream oss;
		oss << "System:: Freeing Connection: " << connections[i]->get_name();
		logger->debug(oss.str());
		delete connections[i];
	}
	for ( unsigned int i = 0 ; i < spiking_groups.size() ; ++i ) {
		std::stringstream oss;
		oss << "System:: Freeing SpikingGroup: " << spiking_groups[i]->get_name();
		logger->debug(oss.str());
		delete spiking_groups[i];
	}

	spiking_groups.clear();
	connections.clear();
	devices.clear();
	checkers.clear();


#ifdef AURYN_CODE_USE_MPI
	if ( syncbuffer != NULL )
		delete syncbuffer;
#endif // AURYN_CODE_USE_MPI

    delete dist;
	delete die;
}

void System::step()
{
	clock++;
}

AurynDouble System::get_time()
{
	return auryn_timestep * clock;
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

void System::register_device(Device * device)
{
	devices.push_back(device);
}

void System::register_checker(Checker * checker)
{
	checkers.push_back(checker);
}

void System::sync()
{

#ifdef AURYN_CODE_USE_MPI

#ifdef CODE_COLLECT_SYNC_TIMING_STATS
	double T1, T2;              
    T1 = MPI_Wtime();     /* start time */
#endif

	std::vector<SpikingGroup *>::const_iterator iter;
	for ( iter = spiking_groups.begin() ; iter != spiking_groups.end() ; ++iter ) 
		syncbuffer->push((*iter)->delay,(*iter)->get_size()); 
	syncbuffer->null_terminate_send_buffer();

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

#endif // AURYN_CODE_USE_MPI
}

void System::evolve()
{
	{ // evolve spiking groups
		std::vector<SpikingGroup *>::const_iterator iter;
		for ( iter = spiking_groups.begin() ; iter != spiking_groups.end() ; ++iter ) 
			(*iter)->conditional_evolve(); // evolve only if existing on rank
	}

	evolve_connections(); // used to run in parallel to the sync (and could still in principle)
	// what is important for event based integration such as done in LinearTrace that this stays
	// on the same side of step() otherwise the results will be wrong (or evolve in LinearTrace has
	// to be adapted.
	
	{ 	// evolve devices
		std::vector<Device *>::const_iterator iter;
		for ( iter = devices.begin() ; iter != devices.end() ; ++iter ) 
			(*iter)->evolve(); 
	}
	
	// update the online rate estimate
	evolve_online_rate_monitor();
}

void System::evolve_connections()
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

void System::execute_devices()
{
	std::vector<Device *>::const_iterator iter;
	for ( iter = devices.begin() ; iter != devices.end() ; ++iter ) {
		(*iter)->execute();
	}
}


bool System::execute_checkers()
{
	for ( unsigned int i = 0 ; i < checkers.size() ; ++i ) {
		if (!checkers[i]->propagate() ) {
			std::stringstream oss;
			oss << "Checker " << i << " triggered abort of simulation!";
			auryn::logger->msg(oss.str(),WARNING);
			return false;
		}
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

	std::cout<< percent << "%     "<< std::setiosflags(std::ios::fixed) << " t=" << time ;

	if ( online_rate_monitor_id >= 0 ) {
		std::cout  << std::setprecision(1) << "  f=" << online_rate_monitor_state << " Hz"
			<< " in " << spiking_groups.at(online_rate_monitor_id)->get_name() << "   ";

	}

	std::cout << std::flush;

	if ( fraction >= 1. )
		std::cout << std::endl;
}

std::string System::get_nice_time(AurynTime clk)
{
	const AurynTime hour = 3600/auryn_timestep;
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
	oss << std::fixed << std::setprecision(1) << clk*auryn_timestep << "s";
	return oss.str();
}

bool System::run(AurynTime starttime, AurynTime stoptime, AurynFloat total_time, bool checking)
{
	// issue a warning if there are no units on the rank specified
	if ( get_total_neurons() == 0 ) {
		auryn::logger->msg("There are no units assigned to this rank!",WARNING);
	}


	double runtime = (stoptime - get_clock())*auryn_timestep;

	std::stringstream oss;
	oss << "Simulation triggered ( " 
		<< "runtime=" << runtime << "s ) ...";
	auryn::logger->msg(oss.str(),NOTIFICATION);

	if ( clock == 0 ) { // only show this once for clock==0
		oss.str("");
		oss	<< "On this rank: neurons_total="<< get_total_neurons() 
			<< ", synapses_total=" << get_total_synapses();
		auryn::logger->msg(oss.str(),SETTINGS);

#ifdef AURYN_CODE_USE_MPI
		if ( mpi_rank() == 0 ) { 
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
#endif // AURYN_CODE_USE_MPI
	}

#ifdef AURYN_CODE_USE_MPI
#ifdef CODE_COLLECT_SYNC_TIMING_STATS
	syncbuffer->reset_sync_time();
#endif
#endif // AURYN_CODE_USE_MPI

	time_t t_sim_start;
	time(&t_sim_start);
	time_t t_last_mark = t_sim_start;

	while ( get_clock() < stoptime ) {

	    if ( (mpi_rank()==0) && (not quiet) && ( (get_clock()%progressbar_update_interval==0) || get_clock()==(stoptime-1) ) ) {
			double fraction = 1.0*(get_clock()-starttime+1)*auryn_timestep/total_time;
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
				<< "s). ";

			if ( td > 50 ) {
				oss << "Ran for " << td << "s "
					<< "with SpeedFactor=" 
					<< std::scientific << td/(LOGGER_MARK_INTERVAL*auryn_timestep);
			}

			AurynTime simtime_left = total_time-auryn_timestep*(get_clock()-starttime+1);
			AurynDouble remaining_minutes = simtime_left*td/(LOGGER_MARK_INTERVAL*auryn_timestep)/60; // in minutes
			if ( remaining_minutes > 5 ) { // only show when more than 5min
			oss	<< ", approximately "
				<< std::setprecision(0) << remaining_minutes
				<< "min of runtime remaining";
			}

			auryn::logger->msg(oss.str(),NOTIFICATION);
		}

		// Auryn duty cycle
		evolve();
		propagate();
		execute_devices();

		if ( checking ) {
			if (!execute_checkers()) {
				return false;
			}
		}

		
		step();	

#ifdef AURYN_CODE_USE_MPI
		if ( mpi_size()>1 && (get_clock())%(MINDELAY) == 0 ) {
			sync();
		} 
#endif // AURYN_CODE_USE_MPI


	}



#ifdef CODE_COLLECT_SYNC_TIMING_STATS
#ifdef AURYN_CODE_USE_MPI
	double elapsed  = syncbuffer->get_elapsed_wall_time();

	oss.str("");
	oss << "Total synctime " 
		<< get_sync_time()
		<< " (relative "
		<< get_relative_sync_time()
		<< " )";
	auryn::logger->msg(oss.str(),NOTIFICATION);

#endif // AURYN_CODE_USE_MPI
#else
	time_t t_now ;
	time(&t_now);
	double elapsed  = difftime(t_now,t_sim_start);
#endif 

	last_elapsed_time = elapsed;

	oss.str("");
	oss << "Simulation finished. Elapsed wall time " 
		<< elapsed << "s. ";

	if ( elapsed > 50 ) { // only display if we have some stats
		oss << "with SpeedFactor=" 
			<< std::scientific << elapsed/runtime
			<< " (network clock=" << get_clock() << ")";
	}
	auryn::logger->msg(oss.str(),NOTIFICATION);


#ifdef AURYN_CODE_USE_MPI
#ifdef CODE_COLLECT_SYNC_TIMING_STATS
	if (mpi_rank() == 0) {
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
#endif // CODE_COLLECT_SYNC_TIMING_STATS
#endif // AURYN_CODE_USE_MPI


	return true;
}

bool System::run(AurynFloat simulation_time, bool checking)
{
	if ( simulation_time < 0.0 ) {
		logger->error("Negative run time not allowed.");
		return false;
	}

	// throw an exception if the stoptime is post the range of AurynTime
	if ( get_time() + simulation_time > std::numeric_limits<AurynTime>::max()*auryn_timestep ) {
		auryn::logger->msg("The requested simulation time exceeds the number of possible timesteps limited by AurynTime datatype.",ERROR);
		throw AurynTimeOverFlowException();
	}

	AurynTime starttime = get_clock();
	AurynTime stoptime = get_clock() + (AurynTime) (simulation_time/auryn_timestep);

	return run(starttime, stoptime, simulation_time, checking);
}

bool System::run_chunk(AurynFloat chunk_time, AurynFloat interval_start, AurynFloat interval_end, bool checking)
{
	AurynTime stopclock = get_clock()+chunk_time/auryn_timestep;

	// throw an exception if the stoptime is post the range of AurynTime
	if ( interval_end > std::numeric_limits<AurynTime>::max()*auryn_timestep ) {
		auryn::logger->msg("The requested simulation time exceeds the number of possible timesteps limited by AurynTime datatype.",ERROR);
		throw AurynTimeOverFlowException();
	}

	return run(interval_start/auryn_timestep, stopclock, (interval_end-interval_start), checking);
}

#ifdef AURYN_CODE_USE_MPI
mpi::communicator * System::get_com()
{
	return mpicom;
}
#endif // AURYN_CODE_USE_MPI

void System::set_simulation_name(std::string name)
{
	simulation_name = name;
}

std::string System::get_simulation_name()
{
	return simulation_name;
}

void System::set_output_dir(std::string path)
{
	outputdir = path;
}

string System::fn(std::string name, std::string extension)
{
	std::stringstream oss;
	oss << outputdir << "/" << name
	<< "." << mpi_rank()
	<< "." << extension;
	return oss.str();
}

string System::fn(std::string name, NeuronID index, std::string extension)
{
	std::stringstream oss;
	oss << outputdir << "/" 
		<< name
		<< index
		<< "." << mpi_rank()
		<< "." << extension;
	return oss.str();
}

string System::fn(std::string extension)
{
	return fn(get_simulation_name(), extension);
}

void System::save_network_state(std::string basename)
{
	auryn::logger->msg("Saving network state", NOTIFICATION);

	std::string netstate_filename;
	{
		std::stringstream oss;
		oss << outputdir 
			<< "/" << basename
			<< "." << mpi_rank()
			<< ".netstate";
		netstate_filename = oss.str();
	} // oss goes out of focus

	auryn::logger->msg("Opening output stream ...",VERBOSE);
	std::ofstream ofs(netstate_filename.c_str());
	boost::archive::binary_oarchive oa(ofs);
	
	auryn::logger->msg("Saving version information ...",VERBOSE);
	// save simulator version information 
	oa << build.version;
	oa << build.subversion;
	oa << build.revision_number;

	auryn::logger->msg("Saving communicator information ...",VERBOSE);
	// save communicator information 
	unsigned int tmp_int = mpi_size();
	oa << tmp_int;
	tmp_int = mpi_rank();
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

	// Save Devices
	auryn::logger->msg("Saving Devices ...",VERBOSE);
	for ( unsigned int i = 0 ; i < devices.size() ; ++i ) {

		std::stringstream oss;
		oss << "Saving Device "
			<<  i 
			<< " to stream";
		auryn::logger->msg(oss.str(),VERBOSE);

		oa << *(devices[i]);
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
		sprintf(filename, "%s.%d.%d.wmat", basename.c_str(), i, mpi_rank());

		std::stringstream oss;
		oss << "Saving connection "
			<<  filename ;
		auryn::logger->msg(oss.str(),VERBOSE);

		connections[i]->write_to_file(filename);
	}

	for ( unsigned int i = 0 ; i < spiking_groups.size() ; ++i ) {
		sprintf(filename, "%s.%d.%d.gstate", basename.c_str(), i, mpi_rank());

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
			<< "." << mpi_rank()
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
	pass_version = pass_version && build.version==tmp_version;
	ia >> tmp_version;
	pass_version = pass_version && build.subversion==tmp_version;
	ia >> tmp_version;
	pass_version = pass_version && build.revision_number==tmp_version;

	if ( !pass_version ) {
		auryn::logger->msg("WARNING: Version check failed! Current Auryn version " 
				"does not match the version which created the file. "
				"This could pose a problem. " 
				"Proceed with caution!" ,WARNING);
	}

	// verify communicator information 
	bool pass_comm = true;
	unsigned int tmp_int;
	ia >> tmp_int;
	pass_comm = pass_comm && (tmp_int == mpi_size());
	ia >> tmp_int;
	pass_comm = pass_comm && (tmp_int == mpi_rank());

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
			<<  i 
			<< ": " 
			<< connections[i]->get_name();
		auryn::logger->msg(oss.str(),VERBOSE);

		ia >> *(connections[i]);
		connections[i]->finalize();
	}

	auryn::logger->msg("Loading SpikingGroups ...",VERBOSE);
	for ( unsigned int i = 0 ; i < spiking_groups.size() ; ++i ) {

		std::stringstream oss;
		oss << "Loading group "
			<<  i 
			<< ": "
			<< spiking_groups[i]->get_name();
		auryn::logger->msg(oss.str(),VERBOSE);

		ia >> *(spiking_groups[i]);
	}

	// Loading Devices states
	auryn::logger->msg("Loading Devices ...",VERBOSE);
	for ( unsigned int i = 0 ; i < devices.size() ; ++i ) {

		std::stringstream oss;
		oss << "Loading Device "
			<<  i;
		auryn::logger->msg(oss.str(),VERBOSE);

		ia >> *(devices[i]);
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
	online_rate_monitor_mul = exp(-auryn_timestep/tau);
}

void System::evolve_online_rate_monitor()
{
	if ( online_rate_monitor_id >= 0 ) {
		online_rate_monitor_state *= online_rate_monitor_mul;
		SpikingGroup * src = spiking_groups[online_rate_monitor_id];
		online_rate_monitor_state += 1.0*src->get_spikes()->size()/online_rate_monitor_tau/src->get_size();
	}
}

void System::set_online_rate_monitor_id( unsigned int id )
{
	online_rate_monitor_state = 0.0;
	if ( id < spiking_groups.size() ) 
		online_rate_monitor_id = id;
	else
		online_rate_monitor_id = -1;
}

AurynDouble System::get_last_elapsed_time()
{
	return last_elapsed_time;
}

AurynDouble System::get_total_elapsed_time()
{
	time_t t_now ;
	time(&t_now);
	double elapsed  = difftime(t_now,t_sys_start);
	return elapsed;
}

void System::flush_devices()
{
	for ( unsigned int i = 0 ; i < devices.size() ; ++i )
		devices[i]->flush();
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


unsigned int System::mpi_size()
{
	return mpi_size_;
}

unsigned int System::mpi_rank()
{
	return mpi_rank_;
}

void System::set_master_seed( unsigned int seed )
{
	if ( seed == 0 ) {
		seed = static_cast<unsigned int>(std::time(0));
	}

	const unsigned int master_seed_multiplier = 257;
	const unsigned int rank_master_seed = seed*master_seed_multiplier*(mpi_rank()+1);

	std::stringstream oss;
	oss << "Seeding this rank with master seed " << rank_master_seed;
	auryn::logger->msg(oss.str(),INFO);

	gen.seed(rank_master_seed);
}

unsigned int System::get_seed()
{
	return (*die)();
}

unsigned int System::get_synced_seed()
{
	unsigned int value;
	if ( mpi_rank() == 0 ) value = get_seed();
#ifdef AURYN_CODE_USE_MPI
	broadcast(*mpicom, value, 0);
#endif // AURYN_CODE_USE_MPI
	// std::cout << mpi_rank() << " " << value << std::endl;
	return value;
}
