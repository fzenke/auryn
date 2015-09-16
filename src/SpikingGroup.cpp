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

#include "SpikingGroup.h"

int SpikingGroup::last_locked_rank = 0;

NeuronID SpikingGroup::unique_id_count = 0;


AurynTime * SpikingGroup::clock_ptr = NULL;

vector<mpi::request> SpikingGroup::reqs;

NeuronID SpikingGroup::anticipated_total = 0;



SpikingGroup::SpikingGroup(NeuronID n, double loadmultiplier, NeuronID total ) 
{
	init( n, loadmultiplier, total );
}

SpikingGroup::~SpikingGroup()
{
	free();
}

void SpikingGroup::init(NeuronID n, double loadmultiplier, NeuronID total )
{
	group_name = "SpikingGroup";
	unique_id  = unique_id_count++;
	size = n;
	effective_load_multiplier = loadmultiplier;

	if ( total > 0 ) {
		anticipated_total = total;
		stringstream oss;
		oss << get_name() << ":: Anticipating " << anticipated_total << " units in total." ;
		logger->msg(oss.str(),NOTIFICATION);
	}

	// setting up default values
	evolve_locally_bool = true;
	locked_rank = 0;
	locked_range = communicator->size();
	rank_size = calculate_rank_size(); // set the rank size

	double fraction = (double)calculate_rank_size(0)*effective_load_multiplier/DEFAULT_MINDISTRIBUTEDSIZE;

	if ( anticipated_total > 0 )
		fraction = (1.*size*effective_load_multiplier)/anticipated_total;

	if ( fraction >= 0 && fraction < 1. ) { 
		lock_range( fraction );
	} else { // ROUNDROBIN which is default
		locked_rank = 0;
		locked_range = communicator->size();

		stringstream oss;
		oss << get_name() << ":: Size " << get_rank_size() << " (ROUNDROBIN)";
		logger->msg(oss.str(),NOTIFICATION);
	}

	stringstream oss;
	oss << get_name() 
		<< ":: Registering SpikeDelay (MINDELAY=" 
		<< MINDELAY << ")";
	logger->msg(oss.str(),VERBOSE);

	delay = new SpikeDelay( );
	set_delay(MINDELAY+1); 

	evolve_locally_bool = evolve_locally_bool && ( get_rank_size() > 0 );
}

void SpikingGroup::lock_range( double rank_fraction )
{
	locked_rank = last_locked_rank%communicator->size(); // TODO might cause a bug with the block lock stuff

	// TODO get the loads for the different ranks and try to minimize this

	if ( rank_fraction == 0 ) { // this is the classical rank lock to one single rank
		stringstream oss;
		oss << get_name() << ":: Groups demands to run on single rank only (RANKLOCK).";
		logger->msg(oss.str(),NOTIFICATION);
		locked_range = 1;
	} else { // this is for multiple rank ranges
		unsigned int free_ranks = communicator->size()-last_locked_rank;

		locked_range = rank_fraction*communicator->size()+0.5;
		if ( locked_range == 0 ) { // needs at least one rank
			locked_range = 1; 
		}

		if ( locked_range > free_ranks ) {
			stringstream oss;
			// oss << "SpikingGroup:: Not enough free ranks to put SpikingGroup defaulting to ROUNDROBIN distribution.";
			oss << get_name() << ":: Not enough free ranks for RANGELOCK. Starting to fill at zero again.";
			logger->msg(oss.str(),NOTIFICATION);
			locked_rank = 0;
			free_ranks = communicator->size();
			// return;
		}
	}

	unsigned int rank = (unsigned int) communicator->rank();
	evolve_locally_bool = ( rank >= locked_rank && rank < (locked_rank+locked_range) );

	last_locked_rank = (locked_rank+locked_range)%communicator->size();
	rank_size = calculate_rank_size(); // recalculate the rank size

	// logging
	if ( evolve_locally_bool ) {
		stringstream oss;
		oss << get_name() << ":: Size "<< get_rank_size() <<" (BLOCKLOCK: ["<< locked_rank 
			<< ":" << locked_rank+locked_range-1 << "] )";
		logger->msg(oss.str(),NOTIFICATION);
	} else {
		stringstream oss;
		oss << get_name() << ":: Passive on this rank (BLOCKLOCK: ["<< locked_rank 
			<< ":" << locked_rank+locked_range-1 << "] )";
		logger->msg(oss.str(),VERBOSE);
	}
	
}


void SpikingGroup::free()
{
	logger->msg("SpikingGroup:: Deleting delay",VERBOSE);
	delete delay;

	logger->msg("SpikingGroup:: Freeing pretraces",VERBOSE);
	for ( NeuronID i = 0 ; i < pretraces.size() ; ++i )
		delete pretraces[i];

	logger->msg("SpikingGroup:: Freeing posttraces",VERBOSE);
	for ( NeuronID i = 0 ; i < posttraces.size() ; ++i )
		delete posttraces[i];

	logger->msg("SpikingGroup:: Freeing state traces",VERBOSE);
	for ( NeuronID i = 0 ; i < post_state_traces.size() ; ++i ) {
		stringstream oss;
		oss << get_name() 
			<< ":: Freeing state trace "
			<< post_state_traces_state_names[i]
			<< " at "
			<< i;
		logger->msg(oss.str() ,VERBOSE);
		delete post_state_traces[i];
	}

	logger->msg("SpikingGroup:: Freeing state vectors",VERBOSE);
	for ( map<string,auryn_vector_float *>::const_iterator iter = state_vectors.begin() ; 
			iter != state_vectors.end() ;
			++iter ) {
		if ( iter->first[0] == '_' ) continue; // do not process volatile state_vector
		auryn_vector_float_free ( iter->second );
	}
	state_vectors.clear();

}

inline int SpikingGroup::msgtag(int x, int y) {
	mpi::communicator * mpicom = communicator;
	return x*mpicom->size()+y + (mpicom->size()*mpicom->size()) * get_uid(); // make messages unique for each SpikingGroup
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
		comrank = (unsigned int) communicator->rank();

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

AurynDouble SpikingGroup::get_effective_load()
{
	return get_rank_size()*effective_load_multiplier;
}


NeuronID SpikingGroup::rank2global(NeuronID i) {
	return i*locked_range+(communicator->rank()-locked_rank);
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

PRE_TRACE_MODEL * SpikingGroup::get_pre_trace( AurynFloat x ) 
{
	for ( int i = 0 ; i < pretraces.size() ; i++ ) {
		if ( pretraces[i]->get_tau() == x ) {
			stringstream oss;
			oss << get_name() << ":: Sharing pre trace with " << x << "s timeconstant." ;
			logger->msg(oss.str(),VERBOSE);
			return pretraces[i];
		}
	}

#ifndef PRE_TRACE_MODEL_LINTRACE
	DEFAULT_TRACE_MODEL * tmp = new DEFAULT_TRACE_MODEL(get_pre_size(),x);
#else
	PRE_TRACE_MODEL * tmp = new PRE_TRACE_MODEL(get_pre_size(),x,clock_ptr);
#endif
	pretraces.push_back(tmp);
	return tmp;
}

DEFAULT_TRACE_MODEL * SpikingGroup::get_post_trace( AurynFloat x ) 
{
	for ( NeuronID i = 0 ; i < posttraces.size() ; i++ ) {
		if ( posttraces[i]->get_tau() == x ) {
			stringstream oss;
			oss << get_name() << ":: Sharing post trace with " << x << "s timeconstant." ;
			logger->msg(oss.str(),VERBOSE);
			return posttraces[i];
		}
	}


	DEFAULT_TRACE_MODEL * tmp = new DEFAULT_TRACE_MODEL(get_post_size(),x);
	posttraces.push_back(tmp);
	return tmp;
}

EulerTrace * SpikingGroup::get_post_state_trace( AurynFloat tau, string state_name, AurynFloat b ) 
{
	// first let's check if a state with that name exists
	if ( find_state_vector( state_name ) == NULL ) {
		logger->msg("A state vector with this name "+state_name+" does not exist", ERROR);
		throw AurynStateVectorException();
	} // good to go

	for ( NeuronID i = 0 ; i < post_state_traces.size() ; i++ ) {
		if ( post_state_traces[i]->get_tau() == tau 
				&& post_state_traces_spike_biases[i] == b
				&& post_state_traces_state_names[i] == state_name ) {
			stringstream oss;
			oss << get_name() 
				<< ":: Sharing post state trace for "
				<< state_name 
				<< " with " 
				<< tau 
				<< "s timeconstant." ;
			logger->msg(oss.str(),VERBOSE);
			return post_state_traces[i];
		}
	}

	// trace does not exist yet, so we are creating 
	// it and do the book keeping
	logger->msg("Creating new post state trace",VERBOSE);
	EulerTrace * tmp = new EulerTrace(get_post_size(),tau);
	tmp->set_target(get_state_vector(state_name));
	post_state_traces.push_back(tmp);
	post_state_traces_spike_biases.push_back(b);
	post_state_traces_state_names.push_back(state_name);
	return tmp;
}

void SpikingGroup::evolve_traces()
{

	// evolve pre synaptic traces
	for ( NeuronID i = 0 ; i < pretraces.size() ; i++ ) { // loop over all traces 
		for (SpikeContainer::const_iterator spike = get_spikes()->begin() ; // spike = pre_spike
				spike != get_spikes()->end() ; 
				++spike ) {
			// cout << " bar " << *spike << endl;
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
			// cout << " foo " << translated_spike << endl;
			posttraces[i]->inc(translated_spike);
		}
		posttraces[i]->evolve();
	}

	// evolve state traces
	for ( NeuronID i = 0 ; i < post_state_traces.size() ; i++ ) {

		// spike triggered component
		for (SpikeContainer::const_iterator spike = get_spikes_immediate()->begin() ; 
				spike != get_spikes_immediate()->end() ; 
				++spike ) {
			NeuronID translated_spike = global2rank(*spike); // only to be used for post traces
			post_state_traces[i]->add(translated_spike, post_state_traces_spike_biases[i]);
		}

		// follow the target vector (instead of evolve)
		post_state_traces[i]->follow();
	}
}

void SpikingGroup::set_name( string s ) 
{
	group_name = s;
}

string SpikingGroup::get_name()
{
	return group_name;
}

bool SpikingGroup::localrank(NeuronID i) {
	bool t = ( (i+locked_rank)%locked_range==communicator->rank() )
		 && (int) communicator->rank() >= locked_rank
		 && (int) communicator->rank() < (locked_rank+locked_range)
		 && i/locked_range < get_rank_size(); 
	return t; 
}


bool SpikingGroup::write_to_file(const char * filename)
{
	if ( !evolve_locally() ) return true;

	ofstream outfile;
	outfile.open(filename,ios::out);
	if (!outfile) {
	  cerr << "Can't open output file " << filename << endl;
	  throw AurynOpenFileException();
	}

	outfile << "# Auryn SpikingGroup state file for n="<< get_rank_size() <<" neurons (ver. " << AURYNVERSION << ")" << endl;
	outfile << "# Default field order (might be overwritten): ";
	for ( map<string,auryn_vector_float *>::const_iterator iter = state_vectors.begin() ; 
			iter != state_vectors.end() ;
			++iter ) {
		if ( iter->first[0] == '_' ) continue; // do not process volatile state_vector
		outfile << scientific << iter->first << " ";
	}
	outfile << "(plus traces)";
	outfile << endl;


	boost::archive::text_oarchive oa(outfile);
	oa << *(delay); 
	outfile << endl;

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

	stringstream oss;
	oss << "Loading SpikingGroup from " << filename;
	logger->msg(oss.str(),NOTIFICATION);
	
	ifstream infile (filename);

	if (!infile) {
		stringstream oes;
		oes << "Can't open input file " << filename;
		logger->msg(oes.str(),ERROR);
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
		stringstream oes;
		oes << "SpikingGroup:: NeuronState file corrupted. Read " 
			<< count << " entries, but " 
			<< get_rank_size() << " expected in " << filename;
		logger->msg(oes.str(),WARNING);
	}

	infile.close();
	return true;
}

void SpikingGroup::virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version ) 
{
	ar & size & axonaldelay;
	ar & *delay;

	logger->msg("SpikingGroup:: serializing state vectors",VERBOSE);
	for ( map<string,auryn_vector_float *>::const_iterator iter = state_vectors.begin() ; 
			iter != state_vectors.end() ;
			++iter ) {
		if ( iter->first[0] == '_' ) continue; // do not process volatile state_vector
		ar & iter->first;
		ar & *(iter->second);
	}

	logger->msg("SpikingGroup:: serializing pre traces",VERBOSE);
	for ( NeuronID i = 0 ; i < pretraces.size() ; ++i )
		ar & *(pretraces[i]);

	logger->msg("SpikingGroup:: serializing post traces",VERBOSE);
	for ( NeuronID i = 0 ; i < posttraces.size() ; ++i )
		ar & *(posttraces[i]);

	logger->msg("NeuronGroup:: serializing state traces",VERBOSE);
	for ( NeuronID i = 0 ; i < post_state_traces.size() ; ++i )
		ar & *(post_state_traces[i]);
}

void SpikingGroup::virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version ) 
{
	ar & size & axonaldelay ; 
	ar & *delay;

	logger->msg("SpikingGroup:: reading state vectors",VERBOSE);
	for ( map<string,auryn_vector_float *>::const_iterator iter = state_vectors.begin() ; 
			iter != state_vectors.end() ;
			++iter ) {
		if ( iter->first[0] == '_' ) continue; // do not process volatile state_vector
		string key;
		ar & key;
		auryn_vector_float * vect = get_state_vector(key);
		ar & *vect;
	}

	logger->msg("SpikingGroup:: reading pre traces",VERBOSE);
	for ( NeuronID i = 0 ; i < pretraces.size() ; ++i )
		ar & *(pretraces[i]);

	logger->msg("SpikingGroup:: reading post traces",VERBOSE);
	for ( NeuronID i = 0 ; i < posttraces.size() ; ++i )
		ar & *(posttraces[i]);

	logger->msg("NeuronGroup:: loading post state traces",VERBOSE);
	for ( NeuronID i = 0 ; i < post_state_traces.size() ; ++i )
		ar & *(post_state_traces[i]);
}


void SpikingGroup::add_state_vector(string key, auryn_vector_float * state_vector)
{

	if ( key[0] == '_' ) {
		stringstream oss;
		oss << "SpikingGroup:: State vector " 
			<< key 
			<< " marked as volatile alias vector."
			<< " It will neither be saved nor freed.";
		logger->msg(oss.str(), VERBOSE);
	}
	state_vectors[key] = state_vector;
}

auryn_vector_float * SpikingGroup::get_state_vector(string key)
{
	if ( state_vectors.find(key) == state_vectors.end() ) {
		if ( get_vector_size() == 0 ) return NULL;
		auryn_vector_float * vec = auryn_vector_float_alloc (get_vector_size()); 
		auryn_vector_float_set_zero( vec );
		add_state_vector(key, vec);
		return vec;
	} else {
		return state_vectors.find(key)->second;
	}
}

auryn_vector_float * SpikingGroup::find_state_vector(string key)
{
	if ( state_vectors.find(key) == state_vectors.end() ) {
		return NULL;
	} else {
		return state_vectors.find(key)->second;
	}
}

void SpikingGroup::randomize_state_vector_gauss(string state_vector_name, AurynState mean, AurynState sigma, int seed)
{
	boost::mt19937 ng_gen(seed+communicator->rank()); // produces same series every time 
	boost::normal_distribution<> dist((double)mean, (double)sigma);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > die(ng_gen, dist);
	AurynState rv;

	auryn_vector_float * vec = get_state_vector(state_vector_name); 


	for ( AurynLong i = 0 ; i<get_rank_size() ; ++i ) {
		rv = die();
		auryn_vector_float_set( vec, i, rv );
	}

}

string SpikingGroup::get_output_line(NeuronID i)
{
	stringstream oss;

	for ( map<string,auryn_vector_float *>::const_iterator iter = state_vectors.begin() ; 
			iter != state_vectors.end() ;
			++iter ) {
		if ( iter->first[0] == '_' ) continue; // do not process volatile state_vector
		oss << scientific << auryn_vector_float_get( iter->second, i ) << " ";
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
		for ( map<string,auryn_vector_float *>::const_iterator iter = state_vectors.begin() ; 
			iter != state_vectors.end() ;
			++iter ) {
			if ( iter->first[0] == '_' ) continue; // do not process volatile state_vector
			if ( ( nums_now = sscanf( buf + bytes_consumed, "%f%n", & temp, & bytes_now ) ) <= 0 )
			{
				// error handling
				logger->msg("Expected additional fields for single neuron parameters. Corrupted nstate file? Aborting.",ERROR);
				return;
			}
			bytes_consumed += bytes_now;
			nums_read += nums_now;
			auryn_vector_float_set(iter->second, i, temp );
		}

		for ( int k = 0 ; k < pretraces.size() ; k++ ) {
			for ( int l = 0 ; l < get_locked_range() ; ++l ) {
				if ( ( nums_now = sscanf( buf + bytes_consumed, "%f%n", & temp, & bytes_now ) ) <= 0 )
				{
					// error handling
					logger->msg("Expected additional fields for pretrace values. Corrupted nstate file? Aborting.",ERROR);
					return;
				}
				bytes_consumed += bytes_now;
				nums_read += nums_now;
				NeuronID t = get_locked_range()*(i)+l;
				if ( t<get_size() )
					pretraces[k]->set(t,temp);

				// cout << temp << endl;
			}
		}

		for ( int k = 0 ; k < posttraces.size() ; k++ ) {
			if ( ( nums_now = sscanf( buf + bytes_consumed, "%f%n", & temp, & bytes_now ) ) <= 0 )
			{
				// error handling
				logger->msg("Expected additional fields for posttrace values. Corrupted nstate file? Aborting.",ERROR);
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
			logger->msg("There were unprocessed values in nstatefile.",WARNING);
		}
}

NeuronID SpikingGroup::get_vector_size()
{
	return calculate_vector_size(get_rank_size());
}

void SpikingGroup::set_num_spike_attributes(int x)
{
	delay->inc_num_attributes(x);
}

int SpikingGroup::get_num_spike_attributes()
{
	return delay->get_num_attributes();
}
