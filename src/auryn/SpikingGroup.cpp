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

#include "SpikingGroup.h"

using namespace auryn;

int SpikingGroup::last_locked_rank = 0;

NeuronID SpikingGroup::unique_id_count = 0;


AurynTime * SpikingGroup::clock_ptr = NULL;


SpikingGroup::SpikingGroup( NeuronID n, NodeDistributionMode mode ) 
{
	init( n, mode );
}

SpikingGroup::~SpikingGroup()
{
	free();
}

void SpikingGroup::init( NeuronID n, NodeDistributionMode mode )
{
	group_name = "SpikingGroup";
	unique_id  = unique_id_count++;
	size = n;
	active = true;

	// setting up default values
	evolve_locally_bool = true;

	// can't import System in this abstract base class,
	// so have to define these quantities localy
#ifdef AURYN_CODE_USE_MPI
	mpi_size = auryn::mpicommunicator->size();
	mpi_rank = auryn::mpicommunicator->rank();
#else
	mpi_size = 1;
	mpi_rank = 0;
#endif // AURYN_CODE_USE_MPI

	locked_rank = 0;
	locked_range = mpi_size;
	rank_size = calculate_rank_size(); // set the rank size

	// if this is a non-parallel sim we don't need to worry about node distribution
	// -> default mode
	if ( mpi_size == 1 ) { 
		mode = ROUNDROBIN; 
	}

	double fraction = (double)calculate_rank_size(0)/DEFAULT_MINDISTRIBUTEDSIZE;
	if ( mode == AUTO ) {
		if ( fraction >= 0 && fraction < 1. ) { 
			mode = BLOCKLOCK;
		} else { 
			mode = ROUNDROBIN;
		}
	}

	switch ( mode ) { // AUTO case should have already been changed into something definite at this point
		case BLOCKLOCK:
			lock_range( fraction );
			break;
		case RANKLOCK:
			lock_range( 0.0 );
			break;
		case ROUNDROBIN: // also serves as default
		default:
			locked_rank = 0;
			locked_range = mpi_size;

			std::stringstream oss;
			oss << get_log_name() << ":: Size " << get_rank_size() ;
			if ( mpi_size > 1 ) oss << " (ROUNDROBIN)";
			auryn::logger->msg(oss.str(),NOTIFICATION);
	}

	// register spike delay
	std::stringstream oss;
	oss << get_log_name() 
		<< ":: Registering SpikeDelay (MINDELAY=" 
		<< MINDELAY << ")";
	auryn::logger->msg(oss.str(),VERBOSE);

	delay = new SpikeDelay( );
	set_delay(MINDELAY+1); 

	evolve_locally_bool = evolve_locally_bool && ( get_rank_size() > 0 );

	// do some some safety checks
	
	// Issue a warning for large neuron groups to check SyncBuffer delta datatype
	if ( 1.0*size*MINDELAY > 0.8*std::numeric_limits<NeuronID>::max() ) {
		oss.str();
		oss << get_log_name() 
			<< ":: Auryn detected that you are using at least one large SpikingGroup. " 
			<< "Please ensure that SyncBuffer is compiled with a delta datatype of sufficient size. "
			<< "It currently uses SYNCBUFFER_DELTA_DATATYPE. "
			<< "Failure to do so might create uncought overflows in SyncBuffer which might lead to " 
			<< "undefined behavior in parallel simulations.";
		auryn::logger->warning(oss.str());
	}
}

void SpikingGroup::lock_range( double rank_fraction )
{
	locked_rank = last_locked_rank%mpi_size; // TODO might cause a bug with the block lock stuff

	if ( rank_fraction == 0.0 ) { // this is the classical rank lock to one single rank
		std::stringstream oss;
		oss << get_log_name() << ":: Group will run on single rank only (RANKLOCK).";
		auryn::logger->msg(oss.str(),NOTIFICATION);
		locked_range = 1;
	} else { // this is for multiple rank ranges
		unsigned int free_ranks = mpi_size-last_locked_rank;

		locked_range = rank_fraction*mpi_size+0.5;
		if ( locked_range == 0 ) { // needs at least one rank
			locked_range = 1; 
		}

		if ( locked_range > free_ranks ) {
			std::stringstream oss;
			// oss << "SpikingGroup:: Not enough free ranks to put SpikingGroup defaulting to ROUNDROBIN distribution.";
			oss << get_log_name() << ":: Not enough free ranks for RANGELOCK. Starting to fill at zero again.";
			auryn::logger->msg(oss.str(),NOTIFICATION);
			locked_rank = 0;
			free_ranks = mpi_size;
			// return;
		}
	}

	unsigned int rank = (unsigned int) mpi_rank;
	evolve_locally_bool = ( rank >= locked_rank && rank < (locked_rank+locked_range) );

	last_locked_rank = (locked_rank+locked_range)%mpi_size;
	rank_size = calculate_rank_size(); // recalculate the rank size

	// logging
	if ( evolve_locally_bool ) {
		std::stringstream oss;
		oss << get_log_name() << ":: Size "<< get_rank_size() <<" (BLOCKLOCK: ["<< locked_rank 
			<< ":" << locked_rank+locked_range-1 << "] )";
		auryn::logger->msg(oss.str(),NOTIFICATION);
	} else {
		std::stringstream oss;
		oss << get_log_name() << ":: Passive on this rank (BLOCKLOCK: ["<< locked_rank 
			<< ":" << locked_rank+locked_range-1 << "] )";
		auryn::logger->msg(oss.str(),VERBOSE);
	}
	
}


void SpikingGroup::free()
{
	std::stringstream oss;
	oss << "SpikingGroup:: " << 
		get_log_name()
		<< " freeing ...";
	auryn::logger->msg(oss.str(),VERBOSE);

	auryn::logger->msg("SpikingGroup:: Deleting delay",VERBOSE);
	delete delay;

	auryn::logger->msg("SpikingGroup:: Freeing pretraces",VERBOSE);
	for ( NeuronID i = 0 ; i < pretraces.size() ; ++i )
		delete pretraces[i];

	auryn::logger->msg("SpikingGroup:: Freeing posttraces",VERBOSE);
	for ( NeuronID i = 0 ; i < posttraces.size() ; ++i )
		delete posttraces[i];

	auryn::logger->msg("SpikingGroup:: Freeing state traces",VERBOSE);
	for ( NeuronID i = 0 ; i < post_state_traces.size() ; ++i ) {
		std::stringstream oss;
		oss << get_log_name() 
			<< ":: Freeing state trace "
			<< post_state_traces_states[i]
			<< " at "
			<< i;
		auryn::logger->msg(oss.str() ,VERBOSE);
		delete post_state_traces[i];
	}

	auryn::logger->msg("SpikingGroup:: Freeing state vectors",VERBOSE);
	for ( std::map<std::string,AurynStateVector *>::const_iterator iter = state_vectors.begin() ; 
			iter != state_vectors.end() ;
			++iter ) {
		std::stringstream oss;
		oss << "Freeing " << get_name()
			<< ": " << iter->first;
		auryn::logger->msg(oss.str(),VERBOSE);
		if ( iter->first[0] == '_' ) continue; // do not process volatile state_vector
		delete iter->second;
	}
	state_vectors.clear();

}

void SpikingGroup::set_clock_ptr(AurynTime * clock) {
	clock_ptr = clock;
	delay->set_clock_ptr(clock);
}


void SpikingGroup::conditional_evolve()
{
	spikes = get_spikes_immediate(); 
	spikes->clear();
	attribs = get_attributes_immediate(); 
	attribs->clear();
	if ( evolve_locally() ) {
		evolve();
	}
}

SpikeContainer * SpikingGroup::get_spikes()
{
	return delay->get_spikes();
}

SpikeContainer * SpikingGroup::get_spikes_immediate()
{
	return delay->get_spikes_immediate();
}

AttributeContainer * SpikingGroup::get_attributes()
{
	return delay->get_attributes();
}

AttributeContainer * SpikingGroup::get_attributes_immediate()
{
	return delay->get_attributes_immediate();
}

NeuronID SpikingGroup::get_uid()
{
	return unique_id;
}

void SpikingGroup::push_spike(NeuronID spike) 
{
	spikes->push_back(rank2global(spike));
}

void SpikingGroup::push_attribute(AurynFloat attrib) 
{
	attribs->push_back(attrib);
}


NeuronID SpikingGroup::calculate_rank_size(int rank)
{
	unsigned int comrank ;
	if ( rank >= 0 ) 
		comrank = rank;
	else
		comrank = (unsigned int) mpi_rank;

	if ( comrank >= locked_rank && comrank < (locked_rank+locked_range) ) {
		if (comrank-locked_rank >= size%locked_range)
			return size/locked_range; // abgerundete groesse
		else
			return size/locked_range+1;  // groesse plus rest
	}

	return 0;
}


NeuronID SpikingGroup::ranksize() {
	return get_rank_size();
}

NeuronID SpikingGroup::get_size()
{
	return size;
} 

NeuronID SpikingGroup::get_pre_size()
{
	return get_size();
} 

NeuronID SpikingGroup::get_post_size()
{
	return get_rank_size();
} 

NeuronID SpikingGroup::rank2global(NeuronID i) {
	return i*locked_range+(mpi_rank-locked_rank);
}

bool SpikingGroup::evolve_locally()
{
	return evolve_locally_bool;
}


unsigned int SpikingGroup::get_locked_rank()
{
	return locked_rank;
}

unsigned int SpikingGroup::get_locked_range()
{
	return locked_range;
}

void SpikingGroup::clear_spikes()
{
	delay->clear();
	get_spikes_immediate()->clear();
}

void SpikingGroup::set_delay( int d ) 
{
	if ( d < MINDELAY ) {
		throw AurynDelayTooSmallException();
	}

	axonaldelay = d;

	delay->set_delay(d); // the plus one here takes care of the current timestep

	// spikes  = delay->get_spikes_immediate();
	// attribs = delay->get_attributes_immediate();
}

Trace * SpikingGroup::get_pre_trace( AurynFloat x ) 
{
	for ( NeuronID i = 0 ; i < pretraces.size() ; i++ ) {
		if ( pretraces[i]->get_tau() == x ) {
			std::stringstream oss;
			oss << get_log_name() << ":: Sharing pre trace with " << x << "s timeconstant." ;
			auryn::logger->msg(oss.str(),VERBOSE);
			return pretraces[i];
		}
	}

	auryn::logger->msg("Initializing pre trace instance",VERBOSE);
	Trace * tmp = new EulerTrace(get_pre_size(),x);
	add_pre_trace(tmp);
	return tmp;
}

void SpikingGroup::add_pre_trace( Trace * tr ) 
{
	if ( tr->size != calculate_vector_size(get_pre_size()) ) {
		std::stringstream oss;
		oss << "Trying to add as pre trace, but its size does not match the SpikinGroup. "
			<< "Trace size: " << tr->size
			<< " Pre size: " << get_pre_size()
			<< " Expected trace size: " << calculate_vector_size(get_pre_size());
		logger->warning(oss.str());
		return;
	}
	pretraces.push_back(tr);
}

Trace * SpikingGroup::get_post_trace( AurynFloat x ) 
{
	for ( NeuronID i = 0 ; i < posttraces.size() ; i++ ) {
		if ( posttraces[i]->get_tau() == x ) {
			std::stringstream oss;
			oss << get_log_name() << ":: Sharing post trace with " << x << "s timeconstant." ;
			auryn::logger->msg(oss.str(),VERBOSE);
			return posttraces[i];
		}
	}


	auryn::logger->msg("Initializing post trace instance",VERBOSE);
	Trace * tmp = new EulerTrace(get_post_size(),x);
	add_post_trace(tmp);
	return tmp;
}

void SpikingGroup::add_post_trace( Trace * tr ) 
{
	if ( tr->size != get_vector_size() ) {
		std::stringstream oss;
		oss << "Trying to add as post trace, but its size does not match the SpikinGroup. "
			<< "Trace size: " << tr->size
			<< " Post size: " << get_post_size()
			<< " Expected trace size: " << get_vector_size();
		logger->warning(oss.str());
		return;
	}
	posttraces.push_back(tr);
}

Trace * SpikingGroup::get_post_state_trace( AurynStateVector * state, AurynFloat tau, AurynFloat b ) 
{
	// first let's check if a state with that name exists
	if ( state == NULL ) {
		auryn::logger->msg("A state vector was not found at this pointer reference.", ERROR);
		throw AurynStateVectorException();
	} // good to go

	for ( NeuronID i = 0 ; i < post_state_traces.size() ; i++ ) {
		if ( post_state_traces[i]->get_tau() == tau 
				&& post_state_traces_spike_biases[i] == b
				&& post_state_traces_states[i] == state ) {
			std::stringstream oss;
			oss << get_log_name() 
				<< ":: Sharing post state trace for ptr reference "
				<< state  // TODO replace by name reverse lookup
				<< " with " 
				<< tau 
				<< "s timeconstant." ;
			auryn::logger->msg(oss.str(),VERBOSE);
			return post_state_traces[i];
		}
	}

	// trace does not exist yet, so we are creating 
	// it and do the book keeping
	auryn::logger->msg("Initializing post trace instance",VERBOSE);
	Trace * tmp = new EulerTrace(get_post_size(),tau);
	tmp->set_target(state);
	post_state_traces.push_back(tmp);
	post_state_traces_spike_biases.push_back(b);
	post_state_traces_states.push_back(state);
	return tmp;
}

Trace * SpikingGroup::get_post_state_trace( std::string state_name, AurynFloat tau, AurynFloat b ) 
{
	AurynStateVector * state = find_state_vector( state_name );
	return get_post_state_trace(state, tau, b);
}

void SpikingGroup::evolve_traces()
{

	// evolve pre synaptic traces
	for ( NeuronID i = 0 ; i < pretraces.size() ; i++ ) { // loop over all traces 
		for (SpikeContainer::const_iterator spike = get_spikes()->begin() ; // spike = pre_spike
				spike != get_spikes()->end() ; 
				++spike ) {
			// std::cout << " bar " << *spike << std::endl;
			pretraces[i]->inc(*spike);
		}
		pretraces[i]->evolve();
	}

	// evolve post synaptic traces
	for ( NeuronID i = 0 ; i < posttraces.size() ; i++ ) {
		for (SpikeContainer::const_iterator spike = get_spikes_immediate()->begin() ; 
				spike != get_spikes_immediate()->end() ; 
				++spike ) {
			NeuronID translated_spike = global2rank(*spike); // only to be used for post traces
			// std::cout << " foo " << translated_spike << std::endl;
			posttraces[i]->inc(translated_spike);
		}
		posttraces[i]->evolve();
	}

	// evolve state traces
	for ( NeuronID i = 0 ; i < post_state_traces.size() ; i++ ) {

		// spike triggered component
		if ( post_state_traces_spike_biases[i] != 0 ) {
			for (SpikeContainer::const_iterator spike = get_spikes_immediate()->begin() ; 
					spike != get_spikes_immediate()->end() ; 
					++spike ) {
				NeuronID translated_spike = global2rank(*spike); // only to be used for post traces
				post_state_traces[i]->add_specific(translated_spike, post_state_traces_spike_biases[i]);
			}
		}

		// follow the target vector (instead of evolve)
		post_state_traces[i]->follow();
	}
}

void SpikingGroup::set_name( std::string s ) 
{
	group_name = s;
}

std::string SpikingGroup::get_name()
{
	return group_name;
}

std::string SpikingGroup::get_file_name()
{
	std::string filename (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__);
	return filename;
}

std::string SpikingGroup::get_log_name()
{
	std::stringstream oss;
	oss << get_name() << " ("
		<< get_file_name() << "): ";
	return oss.str();
}

bool SpikingGroup::localrank(NeuronID i) {

#ifdef DEBUG
	std::cout << ( (i%locked_range+locked_rank)==mpi_rank ) << " "
		<< ( (int) mpi_rank >= locked_rank) << " "
		<< ( (int) mpi_rank >= locked_rank) << " "
		<< ( (int) mpi_rank < (locked_rank+locked_range) ) << " "
		<< ( i/locked_range < get_rank_size() ) << std::endl; 
#endif //DEBUG

	bool t = ( (i%locked_range+locked_rank)==mpi_rank )
		 && (int) mpi_rank >= locked_rank
		 && (int) mpi_rank < (locked_rank+locked_range)
		 && i/locked_range < get_rank_size(); 
	return t; 
}


bool SpikingGroup::write_to_file(const char * filename)
{
	if ( !evolve_locally() ) return true;

	std::ofstream outfile;
	outfile.open(filename,std::ios::out);
	if (!outfile) {
		std::cerr << "Can't open output file " << filename << std::endl;
	  throw AurynOpenFileException();
	}

	outfile << "# Auryn SpikingGroup state file for n="<< get_rank_size() <<" neurons" << std::endl;
	outfile << "# Default field order (might be overwritten): ";
	for ( std::map<std::string,AurynStateVector *>::const_iterator iter = state_vectors.begin() ; 
			iter != state_vectors.end() ;
			++iter ) {
		if ( iter->first[0] == '_' ) continue; // do not process volatile state_vector
		outfile << std::scientific << iter->first << " ";
	}
	outfile << "(plus traces)";
	outfile << std::endl;


	boost::archive::text_oarchive oa(outfile);
	oa << *(delay); 
	outfile << std::endl;

	for ( NeuronID i = 0 ; i < get_rank_size() ; ++i ) 
	{
		outfile << get_output_line(i);
	}

	outfile.close();
	return true;
}

bool SpikingGroup::load_from_file(const char * filename)
{
	if ( !evolve_locally() ) return true;

	std::stringstream oss;
	oss << "Loading SpikingGroup from " << filename;
	auryn::logger->msg(oss.str(),NOTIFICATION);
	
	std::ifstream infile (filename);

	if (!infile) {
		std::stringstream oes;
		oes << "Can't open input file " << filename;
		auryn::logger->msg(oes.str(),ERROR);
		throw AurynOpenFileException();
	}

	NeuronID count = 0;
	char buffer[1024];

	infile.getline (buffer,1024); // skipping header TODO once could make this logic a bit smarter
	infile.getline (buffer,1024); // skipping header 

	boost::archive::text_iarchive ia(infile);
	ia >> *delay;

	infile.getline (buffer,1024); // jumpting to next line

	while ( infile.getline (buffer,1024) )
	{
		load_input_line(count,buffer);
		count++;
	}

	if ( get_rank_size() != count ) {
		// issue warning
		std::stringstream oes;
		oes << "SpikingGroup:: NeuronState file corrupted. Read " 
			<< count << " entries, but " 
			<< get_rank_size() << " expected in " << filename;
		auryn::logger->msg(oes.str(),WARNING);
	}

	infile.close();
	return true;
}

void SpikingGroup::virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version ) 
{
	std::stringstream oss;
	oss << get_log_name() << " serializing delay";
	auryn::logger->msg(oss.str(),VERBOSE);

	ar & size & axonaldelay;
	ar & *delay;

	oss.str("");
	oss << get_log_name() << " serializing " << state_vectors.size() << " state vectors";
	auryn::logger->msg(oss.str(),VERBOSE);
	for ( std::map<std::string,AurynStateVector *>::const_iterator iter = state_vectors.begin() ; 
			iter != state_vectors.end() ;
			++iter ) {
		if ( iter->first[0] == '_' ) continue; // do not process volatile state_vector
		ar & iter->first;
		ar & *(iter->second);
	}

	oss.str("");
	oss << get_log_name() << " serializing " << state_variables.size() << " state variables";
	ar & state_variables;

	oss.str("");
	oss << get_log_name() << " serializing " << pretraces.size() << " pre traces";
	auryn::logger->msg(oss.str(),VERBOSE);
	for ( NeuronID i = 0 ; i < pretraces.size() ; ++i )
		ar & *(pretraces[i]);

	oss.str("");
	oss << get_log_name() << " serializing " << posttraces.size() << " post traces";
	auryn::logger->msg(oss.str(),VERBOSE);
	for ( NeuronID i = 0 ; i < posttraces.size() ; ++i )
		ar & *(posttraces[i]);

	oss.str("");
	oss << get_log_name() << " serializing " << post_state_traces.size() << " post_state traces";
	auryn::logger->msg(oss.str(),VERBOSE);
	for ( NeuronID i = 0 ; i < post_state_traces.size() ; ++i )
		ar & *(post_state_traces[i]);

	oss.str("");
	oss << get_log_name() << " serialize finished for group ";
	auryn::logger->msg(oss.str(),VERBOSE);
}

void SpikingGroup::virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version ) 
{
	std::stringstream oss;
	oss << get_log_name() << " reading delay";
	auryn::logger->msg(oss.str(),VERBOSE);

	ar & size & axonaldelay ; 
	ar & *delay;

	oss.str("");
	oss << get_log_name() << " reading " << state_vectors.size() << " state vectors";
	auryn::logger->msg(oss.str(),VERBOSE);
	for ( std::map<std::string,AurynStateVector *>::const_iterator iter = state_vectors.begin() ; 
			iter != state_vectors.end() ;
			++iter ) {
		if ( iter->first[0] == '_' ) continue; // do not process volatile state_vector
		std::string key;
		ar & key;
		AurynStateVector * vect = find_state_vector(key);
		ar & *vect;
	}

	oss.str("");
	oss << get_log_name() << " reading " << state_variables.size() << " state variables";
	ar & state_variables; 

	oss.str("");
	oss << get_log_name() << " reading " << pretraces.size() << " pre traces";
	auryn::logger->msg(oss.str(),VERBOSE);
	for ( NeuronID i = 0 ; i < pretraces.size() ; ++i )
		ar & *(pretraces[i]);

	oss.str("");
	oss << get_log_name() << " reading " << posttraces.size() << " post traces";
	auryn::logger->msg(oss.str(),VERBOSE);
	for ( NeuronID i = 0 ; i < posttraces.size() ; ++i )
		ar & *(posttraces[i]);

	oss.str("");
	oss << get_log_name() << " reading " << post_state_traces.size() << " post_state_traces";
	auryn::logger->msg(oss.str(),VERBOSE);
	for ( NeuronID i = 0 ; i < post_state_traces.size() ; ++i )
		ar & *(post_state_traces[i]);

	oss.str("");
	oss << get_log_name() << " serialize finished for group ";
	auryn::logger->msg(oss.str(),VERBOSE);
}


void SpikingGroup::add_state_vector(std::string key, AurynStateVector * state_vector)
{

	if ( key[0] == '_' ) {
		std::stringstream oss;
		oss << "SpikingGroup:: State vector " 
			<< key 
			<< " marked as volatile alias vector."
			<< " It will neither be saved nor freed.";
		auryn::logger->msg(oss.str(), VERBOSE);
	}
	state_vectors[key] = state_vector;
}

void SpikingGroup::remove_state_vector( std::string key )
{
	state_vectors.erase(key);
}

AurynStateVector * SpikingGroup::get_state_vector(std::string key)
{
	if ( state_vectors.find(key) == state_vectors.end() ) {
		if ( get_vector_size() == 0 ) return NULL;
		AurynStateVector * vec = new AurynStateVector(get_vector_size()); 
		add_state_vector(key, vec);

		if ( auryn_AlignOffset( vec->size, vec->data, sizeof(float), 16) ) {
			throw AurynMemoryAlignmentException();
		}

		return vec;
	} else {
		return state_vectors.find(key)->second;
	}
}

AurynStateVector * SpikingGroup::get_new_state_vector(std::string key) {
	return get_state_vector(key);
}

AurynStateVector * SpikingGroup::find_state_vector(std::string key)
{
	std::stringstream oss;
	oss << "SpikingGroup:: Running find_state_vector for " << key;
	auryn::logger->msg(oss.str(),VERBOSE);
	if ( state_vectors.find(key) == state_vectors.end() ) {
		return NULL;
	} else {
		return state_vectors.find(key)->second;
	}
}

void SpikingGroup::randomize_state_vector_gauss(std::string state_vector_name, AurynState mean, AurynState sigma, int seed)
{
	boost::mt19937 ng_gen(seed+mpi_rank); // produces same series every time 
	boost::normal_distribution<> dist((double)mean, (double)sigma);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > die(ng_gen, dist);
	AurynState rv;

	AurynStateVector * vec = get_state_vector(state_vector_name); 


	for ( AurynLong i = 0 ; i<get_rank_size() ; ++i ) {
		rv = die();
		vec->set( i, rv );
	}

}

std::string SpikingGroup::get_output_line(NeuronID i)
{
	std::stringstream oss;

	for ( std::map<std::string,AurynStateVector *>::const_iterator iter = state_vectors.begin() ; 
			iter != state_vectors.end() ;
			++iter ) {
		if ( iter->first[0] == '_' ) continue; // do not process volatile state_vector
		oss << std::scientific << iter->second->get( i ) << " ";
	}

	for ( NeuronID k = 0 ; k < pretraces.size() ; k++ ) { 
		// TODO this is actually a bug and only a part of the pretrace gets saved this way
		for ( NeuronID l = 0 ; l < get_locked_range() ; ++l ) {
			NeuronID t = get_locked_range()*(i)+l;
			if ( t < get_size() ) 
				oss << pretraces[k]->get(t) << " ";
			else
				oss << 0.0 << " ";
		}
	}

	for ( NeuronID k = 0 ; k < posttraces.size() ; k++ ) {
		oss << posttraces[k]->get(i) << " ";
	}

	oss << "\n";

	return oss.str();
}

void SpikingGroup::load_input_line(NeuronID i, const char * buf)
{
		int nums_now, bytes_now;
		int bytes_consumed = 0, nums_read = 0;
		float temp;

		// read the state_vectors
		for ( std::map<std::string,AurynStateVector *>::const_iterator iter = state_vectors.begin() ; 
			iter != state_vectors.end() ;
			++iter ) {
			if ( iter->first[0] == '_' ) continue; // do not process volatile state_vector
			if ( ( nums_now = sscanf( buf + bytes_consumed, "%f%n", & temp, & bytes_now ) ) <= 0 )
			{
				// error handling
				auryn::logger->msg("Expected additional fields for single neuron parameters. Corrupted nstate file? Aborting.",ERROR);
				return;
			}
			bytes_consumed += bytes_now;
			nums_read += nums_now;
			iter->second->set( i, temp );
		}

		for ( int k = 0 ; k < pretraces.size() ; k++ ) {
			for ( int l = 0 ; l < get_locked_range() ; ++l ) {
				if ( ( nums_now = sscanf( buf + bytes_consumed, "%f%n", & temp, & bytes_now ) ) <= 0 )
				{
					// error handling
					auryn::logger->msg("Expected additional fields for pretrace values. Corrupted nstate file? Aborting.",ERROR);
					return;
				}
				bytes_consumed += bytes_now;
				nums_read += nums_now;
				NeuronID t = get_locked_range()*(i)+l;
				if ( t<get_size() )
					pretraces[k]->set(t,temp);

				// std::cout << temp << std::endl;
			}
		}

		for ( int k = 0 ; k < posttraces.size() ; k++ ) {
			if ( ( nums_now = sscanf( buf + bytes_consumed, "%f%n", & temp, & bytes_now ) ) <= 0 )
			{
				// error handling
				auryn::logger->msg("Expected additional fields for posttrace values. Corrupted nstate file? Aborting.",ERROR);
				return;
			}
			bytes_consumed += bytes_now;
			nums_read += nums_now;
			posttraces[k]->set(i,temp);
		}


		// check if we read all the values on that line
		if ( ( nums_now = sscanf( buf + bytes_consumed, "%f%n", & temp, & bytes_now ) ) > 0 )
		{
			// error handling
			auryn::logger->msg("There were unprocessed values in nstatefile.",WARNING);
		}
}

NeuronID SpikingGroup::get_vector_size()
{
	return calculate_vector_size(get_rank_size());
}

void SpikingGroup::inc_num_spike_attributes(int x)
{
	std::stringstream oss;
	oss << get_log_name() << ":: Registering " << x << " spike attributes." ;
	auryn::logger->msg(oss.str(),VERBOSE);

	delay->inc_num_attributes(x);
}

int SpikingGroup::get_num_spike_attributes()
{
	return delay->get_num_attributes();
}


AurynState * SpikingGroup::get_state_variable(std::string key)
{
	if ( state_variables.find(key) == state_variables.end() ) {
		state_variables[key] = 0.0;
	} 
	return &(state_variables.find(key)->second);
}

