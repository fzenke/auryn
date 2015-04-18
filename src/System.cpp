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

#include "System.h"

void System::init() {
	clock = 0;
	quiet = false;
	set_simulation_name("default");

	syncbuffer = new SyncBuffer(mpicom);

	stringstream oss;
	oss.str("");
	oss << "Auryn version "
		<< AURYNVERSION;

	if ( AURYNSUBVERSION ) {
		oss << "."
		<< AURYNSUBVERSION;
	}

	oss << " ( compiled " << __DATE__ << " " << __TIME__ << " )";
	logger->msg(oss.str(),NOTIFICATION);

	oss.str("");
	oss << "Current AurynTime good for simulations up to "
		<< std::numeric_limits<AurynTime>::max()*dt << "s  "
		<< "( " << std::numeric_limits<AurynTime>::max()*dt/3600 << "h )";
	logger->msg(oss.str(),DEBUG);

	oss.str("");
	oss << "Current NeuronID and sync are good for simulations up to "
		<< std::numeric_limits<NeuronID>::max()/MINDELAY << " cells.";
	logger->msg(oss.str(),DEBUG);

}

System::System()
{
	init();
}

System::System(mpi::communicator * communicator)
{
	mpicom = communicator;
	init();

	stringstream oss;

	if ( mpicom->size() > 0 ) {
		oss << "MPI run rank "
			<<  mpicom->rank() << " of "
			<<  mpicom->size() << ".";
		logger->msg(oss.str(),NOTIFICATION);
	}

	if ( mpicom->size() > 0 && (mpicom->size() & (mpicom->size()-1)) ) {
		oss.str("");
		oss << "Warning! The number of processes is not a power of two. "
			<< "This could cause impaired performance or even crashes "
			<< "in some MPI implementations.";
		logger->msg(oss.str(),WARNING,true);
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
	vector<SpikingGroup *>::const_iterator iter;
	for ( iter = spiking_groups.begin() ; iter != spiking_groups.end() ; ++iter ) {
		sum += (*iter)->get_rank_size(); 
	}
	return sum;
}

AurynDouble System::get_total_effective_load()
{
	AurynDouble sum = 0.;
	vector<SpikingGroup *>::const_iterator iter;
	for ( iter = spiking_groups.begin() ; iter != spiking_groups.end() ; ++iter ) {
		sum += (*iter)->get_effective_load(); 
	}
	return sum;
}

AurynLong System::get_total_synapses()
{
	AurynLong sum = 0;
	vector<Connection *>::const_iterator iter;
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

	vector<SpikingGroup *>::const_iterator iter;
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
	vector<SpikingGroup *>::const_iterator iter;

	for ( iter = spiking_groups.begin() ; iter != spiking_groups.end() ; ++iter ) 
		(*iter)->conditional_evolve(); // evolve only if existing on rank
}

void System::evolve_independent()
{
	for ( vector<SpikingGroup *>::const_iterator iter = spiking_groups.begin() ; 
		  iter != spiking_groups.end() ; 
		  ++iter ) 
		(*iter)->evolve_traces(); // evolve only if existing on rank

	for ( vector<Connection *>::const_iterator iter = connections.begin() ; 
			iter != connections.end() ; 
			++iter )
		(*iter)->evolve(); 
}

void System::propagate()
{
	vector<Connection *>::const_iterator iter;
	for ( iter = connections.begin() ; iter != connections.end() ; ++iter )
		(*iter)->propagate(); 
}

bool System::monitor(bool checking)
{
	vector<Monitor *>::const_iterator iter;
	for ( iter = monitors.begin() ; iter != monitors.end() ; ++iter )
		(*iter)->propagate();

	for ( unsigned int i = 0 ; i < checkers.size() ; ++i )
		if (!checkers[i]->propagate() && checking) {
			stringstream oss;
			oss << "Checker " << i << " broke run!";
			logger->msg(oss.str(),WARNING);
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

	cout << fixed << "\r" "[" << bar << "] ";
	cout.width( 3 );

	string time = get_nice_time(clk);

	cout<< percent << "%     "<< setiosflags(ios::fixed) << " t=" << time ;

	if (checkers.size())
		cout  << setprecision(1) << "  f=" << checkers[0]->get_property() << " Hz  ";

	cout << std::flush;

	if ( fraction >= 1. )
		cout << endl;
}

string System::get_nice_time(AurynTime clk)
{
	const AurynTime hour = 3600/dt;
	const AurynTime day  = 24*hour;
	stringstream oss;
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
	oss << fixed << setprecision(1) << clk*dt << "s";
	return oss.str();
}

bool System::run(AurynTime starttime, AurynTime stoptime, AurynFloat total_time, bool checking)
{
	// issue a warning if there are no units on the rank specified
	if ( get_total_neurons() == 0 ) {
		logger->msg("There are no units assigned to this rank!",WARNING);
	}

	double runtime = (stoptime - get_clock())*dt;

	stringstream oss;
	oss << "Simulation triggered ( " 
		<< "runtime=" << runtime << "s )";
	logger->msg(oss.str(),NOTIFICATION);

	oss.str("");
	oss	<< "On this rank: neurons_total="<< get_total_neurons() 
		<< ", effective_load=" << get_total_effective_load()
		<< ", synapses_total=" << get_total_synapses();
	logger->msg(oss.str(),SETTINGS);

	if (mpicom->rank() == 0) {
		AurynLong all_ranks_total_neurons;
		reduce(*mpicom, get_total_neurons(), all_ranks_total_neurons, std::plus<AurynLong>(), 0);

		AurynLong all_ranks_total_synapses;
		reduce(*mpicom, get_total_synapses(), all_ranks_total_synapses, std::plus<AurynLong>(), 0);

		oss.str("");
		oss	<< "On all ranks: neurons_total="<< all_ranks_total_neurons 
			<< ", synapses_total=" << all_ranks_total_synapses;
		logger->msg(oss.str(),SETTINGS);
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

	    if ( (mpicom->rank()==0) && (not quiet) && ( (get_clock()%PROGRESSBAR_UPDATE_INTERVAL==0) || get_clock()==(stoptime-1) ) ) {
			double fraction = 1.0*(get_clock()-starttime+1)*dt/total_time;
			progressbar(fraction,get_clock()); // TODO find neat solution for the rate
		}

		if ( get_clock()%LOGGER_MARK_INTERVAL==0 ) // set a mark 
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

			logger->msg(oss.str(),NOTIFICATION);
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
	logger->msg(oss.str(),NOTIFICATION);

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
		<< "(clock=" << get_clock() << ")";
	logger->msg(oss.str(),NOTIFICATION);


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
		logger->msg(oss.str(),NOTIFICATION);
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
		logger->msg("The requested simulation time exceeds the number of possible timesteps limited by AurynTime datatype.",ERROR);
		throw AurynTimeOverFlowException();
	}

	return run(starttime, stoptime, simulation_time, checking);
}

bool System::run_chunk(AurynFloat chunk_time, AurynFloat interval_start, AurynFloat interval_end, bool checking)
{
	AurynTime stopclock = get_clock()+chunk_time/dt;

	// throw an exception if the stoptime is post the range of AurynTime
	if ( interval_end > std::numeric_limits<AurynTime>::max()*dt ) {
		logger->msg("The requested simulation time exceeds the number of possible timesteps limited by AurynTime datatype.",ERROR);
		throw AurynTimeOverFlowException();
	}

	return run(interval_start/dt, stopclock, (interval_end-interval_start), checking);
}


mpi::communicator * System::get_com()
{
	return mpicom;
}

void System::set_simulation_name(string name)
{
	simulation_name = name;
}

void System::save_network_state(string basename)
{
	string netstate_filename;
	{
		stringstream oss;
		oss << basename
			<< "." << mpicom->rank()
			<< ".netstate";
		netstate_filename = oss.str();
	} // oss goes out of focus

	std::ofstream ofs(netstate_filename.c_str());
	boost::archive::binary_oarchive oa(ofs);

	for ( unsigned int i = 0 ; i < connections.size() ; ++i ) {
		// sprintf(filename, "%s.%d.%d.wmat", basename.c_str(), i, mpicom->rank());

		stringstream oss;
		oss << "Saving connection "
			<<  i 
			<< " to stream";
		logger->msg(oss.str(),NOTIFICATION);

		oa << *(connections[i]);
	}

	for ( unsigned int i = 0 ; i < spiking_groups.size() ; ++i ) {
		// sprintf(filename, "%s.%d.%d.gstate", basename.c_str(), i, mpicom->rank());

		stringstream oss;
		oss << "Saving SpikingGroup "
			<<  i 
			<< " to stream";
		logger->msg(oss.str(),NOTIFICATION);

		oa << *(spiking_groups[i]);
	}

	ofs.close();
}

void System::save_network_state_text(string basename)
{
	char filename [255];
	for ( unsigned int i = 0 ; i < connections.size() ; ++i ) {
		sprintf(filename, "%s.%d.%d.wmat", basename.c_str(), i, mpicom->rank());

		stringstream oss;
		oss << "Saving connection "
			<<  filename ;
		logger->msg(oss.str(),NOTIFICATION);

		connections[i]->write_to_file(filename);
	}

	for ( unsigned int i = 0 ; i < spiking_groups.size() ; ++i ) {
		sprintf(filename, "%s.%d.%d.gstate", basename.c_str(), i, mpicom->rank());

		stringstream oss;
		oss << "Saving group "
			<<  filename ;
		logger->msg(oss.str(),NOTIFICATION);

		spiking_groups[i]->write_to_file(filename);
	}
}

void System::load_network_state(string basename)
{
	string netstate_filename;
	{
		stringstream oss;
		oss << basename
			<< "." << mpicom->rank()
			<< ".netstate";
		netstate_filename = oss.str();
	} // oss goes out of focus

	std::ifstream ifs(netstate_filename.c_str());
	boost::archive::binary_iarchive ia(ifs);

	for ( unsigned int i = 0 ; i < connections.size() ; ++i ) {

		stringstream oss;
		oss << "Loading connection "
			<<  i ;
		logger->msg(oss.str(),NOTIFICATION);

		ia >> *(connections[i]);
		connections[i]->finalize();
	}

	for ( unsigned int i = 0 ; i < spiking_groups.size() ; ++i ) {

		stringstream oss;
		oss << "Loading group "
			<<  i ;
		logger->msg(oss.str(),NOTIFICATION);

		ia >> *(spiking_groups[i]);
	}

	ifs.close();
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
