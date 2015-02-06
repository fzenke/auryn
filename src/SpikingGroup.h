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

#ifndef SPIKINGGROUP_H_
#define SPIKINGGROUP_H_

#include "auryn_definitions.h"
// #include "SpikeContainer.h"
#include "SpikeDelay.h"
#include "EulerTrace.h"
#include "LinearTrace.h"


#include <vector>
#include <map>


using namespace std;
namespace mpi = boost::mpi;

class System;

/*! \brief Abstract base class of all objects producing spikes
 *
 * This is the abstract/virtual base class from which all spiking objects should be derived. All classes derived from SpikingGroup have in common that they can emit spikes. Furthermore they should implement the method evolve() for carring out internal state changes such as integration of the e.g. Euler Step.
 * Other classes interact with inheritants of SpikingGroup by calling the functions get_spikes() and get_spikes_immediate().
 */
class SpikingGroup
{
private:
	/* Functions necesssary for serialization */
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		virtual_serialize(ar, version);
	}

	/*! Stores the groupID gid of this group */
	NeuronID unique_id;

	/*! Stores the current value of the gid count */
	static NeuronID unique_id_count;

	static vector<mpi::request> reqs;

	/*! Standard initialization of the object. */
	void init(NeuronID size, double loadmultiplier, NeuronID total );

	/*! Stores the number of anticipated units to optimize loadbalancing */
	static NeuronID anticipated_total;

	void lock_range( double rank_fraction );
	/*! If not distributed the first rank to lock it to. */
	unsigned int locked_rank;
	/*! If not distributed the number of ranks to lock it to. */
	unsigned int locked_range;
	/*! Keeps track on where rank-locking is */
	static int last_locked_rank;

	bool evolve_locally_bool;
	inline int msgtag(int x, int y);

	/*! Parameter that characterizes the computational load the SpikingGroup causes with respect to IFGroup. It's used vor load balancing*/
	double effective_load_multiplier;

	/*! Stores axonal delay value - by default MINDELAY */
	int axonaldelay;

protected:
	/*! Pretraces */
	vector<PRE_TRACE_MODEL *> pretraces;

	/*! Posttraces */
	vector<DEFAULT_TRACE_MODEL *> posttraces;

	/*! Identifying name for object */
	string group_name;

	/*! Stores the size of the group */
    NeuronID size;
	/*! Stores the size of the group on this rank */
	NeuronID rank_size;
	/*! SpikeContainers to store spikes produced during one step of evolve. */
	SpikeContainer * spikes;
	AttributeContainer * attribs;

	/*! Stores the length of output delay */
	static AurynTime * clock_ptr;

	/* Functions related to loading and storing the state from files */
	virtual void load_input_line(NeuronID i, const char * buf);
	virtual string get_output_line(NeuronID i);

	virtual void virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version );
	virtual void virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version );
	
public:
	SpikeDelay * delay;

	/*! Can hold single neuron vectors such as target rates or STP states etc  */
	map<string,auryn_vector_float *> state_vectors;

	/*! Returns existing state vector by name */
	auryn_vector_float * find_state_vector(string key);

	/*! Creates a new or returns an existing state vector */
	auryn_vector_float * get_state_vector(string key);

	/*! Randomizes the content of a state vector with Gaussian random numbers. Seeding is MPI save. */
	void randomize_state_vector_gauss(string state_vector_name, AurynState mean, AurynState sigma, int seed=12239);

	/*! Default constructor */
	SpikingGroup(NeuronID size, double loadmultiplier = 1., NeuronID total = 0 );
	/*! Default destructor */
	virtual ~SpikingGroup();

	/*! Give a name */
	void set_name(string s);

	/*! Retrieves the groups name */
	string get_name();

	void set_num_spike_attributes(int x);
	int get_num_spike_attributes();

	/*! Frees potentially allocated memory */
	void free();
	virtual void evolve() = 0;
	void conditional_evolve();
	unsigned int get_locked_rank();
	unsigned int get_locked_range();
	SpikeContainer * get_spikes();
	/*! Supplies pointer to SpikeContainer of spikes generated during the last evolve() step. */
	SpikeContainer * get_spikes_immediate();
	/*! Supplies pointer to Attributecontainer for usage in propagating Connection objects. Same as get_spikes_immediate(), however might be overwritten to contain Spikes that have been delayed. */
	AttributeContainer * get_attributes();
	/*! Supplies pointer to Attributecontainer of spikes generated during the last evolve() step. */
	AttributeContainer * get_attributes_immediate();
	/*! Returns the size of the group. */
	NeuronID get_size();
	NeuronID get_pre_size();
	/*! Determines rank size and stores it in local variable. */
	NeuronID calculate_rank_size(int rank = -1);
	/*! Returns the size of the rank. */
	NeuronID get_rank_size();
	NeuronID get_post_size();
	/*! Returns the effective load of the group. */
	AurynDouble get_effective_load();

	void set_clock_ptr(AurynTime * clock);
	/*! Returns true if this group is hosted at a single CPU. */
	bool evolve_locally();

	/*! Evolves traces */
	void evolve_traces();

	/*! Get the unique ID of the class */
	NeuronID get_uid();

	/*! Returns a pre trace with time constant x 
	 * Checks first if an instance of a trace exists and returns it
	 * otherwise creates a new instance first */
	PRE_TRACE_MODEL * get_pre_trace( AurynFloat x );

	/*! Returns a post trace with time constant x 
	 * Checks first if an instance of a trace exists and returns it
	 * otherwise creates a new instance first */
	DEFAULT_TRACE_MODEL * get_post_trace( AurynFloat x );

	void push_spike(NeuronID spike);

	void push_attribute(AurynFloat attrib);

	/*! Clear all spikes stored in the delays which is useful to reset a network during runtime */
	void clear_spikes();

	/*! Sets axonal delay for this SpikingGroup */
	void set_delay( int d );

	virtual bool write_to_file(const char * filename);
	virtual bool load_from_file(const char * filename);

	NeuronID ranksize();
	NeuronID global2rank(NeuronID i); 
	NeuronID rank2global(NeuronID i);
	bool localrank(NeuronID i);

	/*! Rank size but rounded up to multiples of 4 for SSE compatibility */
	NeuronID get_vector_size();

#ifdef CODE_GATHER_STATS_WAITALL_TIME
	/*! Variables to collect stats if enabled */
	boost::timer waitall_timer;
	static double waitall_time;
	static double waitall_time2;
#endif
};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(SpikingGroup)

extern System * sys;
extern Logger * logger;
extern mpi::communicator * communicator;


inline NeuronID SpikingGroup::global2rank(NeuronID i) {
	return i/locked_range;
}

inline NeuronID SpikingGroup::get_rank_size()
{
	return rank_size;
} 



#endif /*SPIKINGGROUP_H_*/
