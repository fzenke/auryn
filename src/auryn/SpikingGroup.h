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

#ifndef SPIKINGGROUP_H_
#define SPIKINGGROUP_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "SpikeDelay.h"
#include "Trace.h"
#include "EulerTrace.h"

#include <vector>
#include <map>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>

#include <boost/archive/text_oarchive.hpp> 
#include <boost/archive/text_iarchive.hpp> 
#include <boost/archive/binary_oarchive.hpp> 
#include <boost/archive/binary_iarchive.hpp> 


namespace auryn {

/*! \brief Specifies howto distribute different neurons across ranks when simulation is run in parallel. */
enum NodeDistributionMode { 
	AUTO, //!< Tries to make a smart choice
	ROUNDROBIN, //!< Default mode of distribution
	BLOCKLOCK, //!< Tries to implement block lock 
	RANKLOCK //!< Locks to single rank (this is a special case of BLOCKLOCK
};


/*! \brief Abstract base class of all objects producing spikes
 *
 * This is the abstract/virtual base class from which all spiking objects
 * should be derived. All classes derived from SpikingGroup have in common that
 * they can emit spikes. Furthermore they should implement the method evolve()
 * for carring out internal state changes such as integration of the e.g. Euler
 * Step.  Other classes interact with inheritants of SpikingGroup by calling
 * the functions get_spikes() and get_spikes_immediate().
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

	unsigned int mpi_rank;
	unsigned int mpi_size;

	/*! Stores the current value of the gid count */
	static NeuronID unique_id_count;


	void lock_range( double rank_fraction );
	/*! If not distributed the first rank to lock it to. */
	unsigned int locked_rank;
	/*! If not distributed the number of ranks to lock it to. */
	unsigned int locked_range;
	/*! Keeps track on where rank-locking is */
	static int last_locked_rank;

	/*! Standard initialization of the object. */
	void init( NeuronID size, NodeDistributionMode mode );

	/*! Stores axonal delay value - by default MINDELAY */
	int axonaldelay;

protected:
	bool evolve_locally_bool;

	/*! \brief Pretraces */
	std::vector<Trace *> pretraces;

	/*! \brief Posttraces */
	std::vector<Trace *> posttraces;

	/*! \brief Post state traces */
	std::vector<Trace *> post_state_traces;
	std::vector<AurynFloat> post_state_traces_spike_biases;
	std::vector<AurynStateVector *> post_state_traces_states;

	/*! \brief Identifying name for object */
	std::string group_name;

	/*! \brief Stores the size of the group */
    NeuronID size;
	/*! \brief Stores the size of the group on this rank */
	NeuronID rank_size;
	/*! \brief SpikeContainers to store spikes produced during one step of evolve. */
	SpikeContainer * spikes;
	AttributeContainer * attribs;

	/*! Stores the length of output delay */
	static AurynTime * clock_ptr;

	/* Functions related to loading and storing the state from files */
	virtual void load_input_line(NeuronID i, const char * buf);

	virtual std::string get_output_line(NeuronID i);

	/*! \brief Implementatinon of serialize function for writing. */
	virtual void virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version );

	/*! \brief Implementatinon of serialize function for reading. */
	virtual void virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version );


	/*! \brief Frees potentially allocated memory */
	void free();
	
public:
	SpikeDelay * delay;

	/*! \brief Toggles group active 
	 *
	 * Groups do not necessarily obey this toggle though. */
	bool active;


	/*! Can hold single neuron vectors such as target rates or STP states etc  */
	std::map<std::string,AurynStateVector *> state_vectors;

	/*! Holds group-wide state variables such as population target rates or global protein levels etc */
	std::map<std::string,AurynState> state_variables;


	/*! \brief Adds a state vector passed as an argument to the dictinary. */
	void add_state_vector( std::string key, AurynStateVector * state_vector );

	/*! \brief Removes a state vector passed as an argument to the dictinary.
	 *
	 * The state vector is not freed automatically! */
	void remove_state_vector( std::string key );

	/*! \brief Returns existing state vector by name or NULL if it does not exist. 
	 *
	 * \param key A string with the name of the state vector.
	 *
	 * \returns The state vector and NULL if the state_vector does not 
	 * exist. */
	AurynStateVector * find_state_vector(std::string key);

	/*! \brief Creates a new or returns an existing state vector by name. 
	 *
	 * This function returns a state vector for the given key. If a vector with the key does not exist
	 * the function creates one. If it does exist it returns the existing one.
	 * The function returns NULL if the statevector would have zero length on that rank.
	 *
	 * \param key The key name of the state vector to be return
	 *
	 * \returns A StateVector matching the key provided.
	 * */
	AurynStateVector * get_state_vector(std::string key);

	/*! \brief Returns an existing state with the supplied name. 
	 *
	 *
	 * \param key A string with the name of the state vector.
	 *
	 * \returns The state vector and NULL if the state_vector does not 
	 * exist. */
	AurynStateVector * get_existing_state_vector(std::string key);

	/*! \brief Creates a new state vector and throws an exception if a vector with the same name exists. 
	 *
	 * Returns a new state vector with the given key and returns an exception if a vector with the same
	 * key already exists. Moreover, NULL is returned if the state vector would have length 0 otherwise.
	 *
	 * \throws AurynDelayTooSmallException if a state vector with the same key already exists. 
	 * \param key The key of the state vector to be created.
	 * \returns An AurynStateVector with the provided key.
	 * */
	AurynStateVector * create_state_vector(std::string key);

	/*! \brief Creates a new group-wide state variable or returns an existing group-wide variable by name then returns a pointer to it. */
	AurynState * get_state_variable(std::string key);

	/*! \brief Randomizes the content of a state vector with Gaussian random numbers. Seeding is MPI save. */
	void randomize_state_vector_gauss(std::string state_vector_name, AurynState mean, AurynState sigma, int seed=12239);

	/*! \brief Default constructor */
	SpikingGroup( NeuronID size, NodeDistributionMode mode=AUTO );

	/*! \brief Default destructor */
	virtual ~SpikingGroup();

	/*! \brief Evolves traces */
	virtual void evolve_traces();

	/*! \brief Set connection name */
	void set_name(std::string s);

	/*! \brief Retrieves the groups name */
	std::string get_name();

	/*! \brief Extracts the class name of the connection from the file name */
	std::string get_file_name();

	/*! \brief Returns a string which is the combination of file and connection name for logging. */
	std::string get_log_name();

	/*! \brief Instructs SpikingGroup to increase the number of spike attributes by x.
	 *
	 * The reason we only increment the size is that multiple Connection objects such as
	 * STPConnection might want to add an attribute to a spike. These might be different
	 * to allow synaptic type dependent plasticity and hence will all have to be transmitted 
	 * without knowledge about the other synapses which might want to submit a different value.*/
	void inc_num_spike_attributes(int x);
	int get_num_spike_attributes();

	/*! \brief Virtual pure evolve function which needs to be implemented by derived classes
	 *
	 * The evolve function is called during simulations in every timestep by the System class. 
	 * It updates the internal state of the spiking group and pushes spikes which 
	 * are generated in this timestep to the axonal output delay (SpikeDelay).
	 */
	virtual void evolve() = 0;

	/*! \brief Conditional evolve functino which is called by System
	 *
	 * This function invoces evolve if the present group has work to be done on the current rank.
	 * Thus the call of evolve is made conditional on the fact if there is work to be done for this
	 * group on the present rank. This is only important for MPI parallel simulations.
	 */
	void conditional_evolve();

	/*! \brief Returns locked rank for SpikingGroups which are not distributed across all ranks */
	unsigned int get_locked_rank();

	/*! \brief Returns locked range of ranks for SpikingGroups which are not distributed across all ranks */
	unsigned int get_locked_range();

	/*! \brief Returns pointer to a spike container that contains spikes which arrive in this timestep from all neurons in this group. 
	 *
	 * In paralell simulations the SpikeContainer returned is at this point in time guaranteed to contain spikes from all ranks. */
	SpikeContainer * get_spikes();

	/*! \brief Returns pointer to SpikeContainer of spikes generated during the last evolve() step. */
	SpikeContainer * get_spikes_immediate();

	/*! \brief Returns pointer to Attributecontainer for usage in propagating Connection objects. Same as get_spikes_immediate(), however might be overwritten to contain Spikes that have been delayed. */
	AttributeContainer * get_attributes();

	/*! \brief Returns pointer to Attributecontainer of spikes generated during the last evolve() step. */
	AttributeContainer * get_attributes_immediate();

	/*! \brief Returns the size of the group. */
	NeuronID get_size();

	/*! \brief Returns the size of the group. 
	 *
	 * It's the size that should be used when a presynaptic trace is defined on this grou,  hence the name. */
	NeuronID get_pre_size();

	/*! \brief Determines rank size and stores it in local variable. */
	NeuronID calculate_rank_size(int rank = -1);

	/*! \brief Returns the size on this rank. */
	NeuronID get_rank_size();

	/*! \brief Returns the size on this rank. 
	 *
	 * It's the size that should be used when a postsynaptic trace is defined on this group, hence the name. */
	NeuronID get_post_size();

	void set_clock_ptr(AurynTime * clock);

	/*! \brief Returns true if the calling instance has units which are integrated on the current rank. */
	bool evolve_locally();


	/*! \brief Get the unique ID of the class 
	 *
	 * \returns a unique numerical identifier of the present SpikingGroup instance. */
	NeuronID get_uid();

	/*! \brief Returns a pre trace with time constant x 
	 * 
	 * Checks first if an instance of a trace exists and returns it
	 * otherwise creates a new instance first. 
	 *
	 * \param x The timeconstant for the presynaptic trace
	 * \returns a pointer to a pre trace instance */
	Trace * get_pre_trace( AurynFloat x );

	/*! \brief Adds trace to pretrace stack of a connection
	 * 
	 * Mostly for internal use by get_pre_trace()
	 *
	 * \param tr A pointer to a presynaptic trace instance to be added to the present SpikingGroup
	 * */
	void add_pre_trace( Trace * tr );

	/*! \brief Returns a post trace with time constant x 
	 *
	 * Checks first if an instance of a trace exists and returns it
	 * otherwise creates a new instance first. 
	 *
	 * \param x The timeconstant for the postsynaptic trace.
	 * \returns a pointer to a postsynaptic trace instance. */
	Trace * get_post_trace( AurynFloat x );

	/*! \brief Adds trace to posttrace stack of a connection
	 * 
	 * Mostly for internal use by get_post_trace()
	  */
	void add_post_trace( Trace * tr );

	/*! \brief Pushes a local NeuronID as spike into the axonal SpikeDelay buffer 
	 *
	 * This function expectes the local NeuronID and translates the argument to a global argument.
	 * \see rank2global, global2rank
	 *
	 * \param spike The NeuronID of the units which spiked in the current timestep 
	 * */
	void push_spike(NeuronID spike);

	/*! \brief Pushes a spike attribute into the axonal SpikeDelay buffer 
	 *
	 * This is for instance used to implement short-term plasticity in which each presynaptic spike is associated
	 * with a certain currently available amount of presynaptic neurotransmitter. Spike attributes are float values
	 * which can be attaced to a spike to convey this information. */
	void push_attribute(AurynFloat attrib);

	/*! \brief Clears all spikes stored in the delays which is useful to reset a network during runtime */
	void clear_spikes();


	/*! \brief Returns a post trace of a neuronal state variable e.g. the membrane 
	 * potential with time constant tau. 
	 *
	 * This trace is an cotinuously integrated Trace which uses the follow 
	 * function on the mem state vector. 
	 * @param state_name A string stating the neurons state name
	 * @param tau The time constant of the trace.
	 * @param b The optional parameter b allows to specify a spike triggered contribution
	 * which will be added instantaneously to the trace upon each 
	 * postsynaptic spike.
	 * */
	Trace * get_post_state_trace( std::string state_name="mem", AurynFloat tau=10e-3, AurynFloat b=0.0 );

	/*! \brief Returns a post trace of a neuronal state variable specified by pointer
	 *
	 * This trace is an cotinuously integrated Trace which uses the follow 
	 * function on the mem state vector. 
	 * @param state A pointer to the relevant state vector
	 * @param tau The time constant of the trace.
	 * @param b The optional parameter b allows to specify a spike triggered contribution
	 * which will be added instantaneously to the trace upon each 
	 * postsynaptic spike.
	 * */
	Trace * get_post_state_trace( AurynStateVector * state, AurynFloat tau=10e-3, AurynFloat b=0.0 );

	/*! \brief Sets axonal delay for this SpikingGroup 
	 *
	 * Note the delay needs to be larger er equal to the MINDELAY defined in auryn_definitions.h */
	void set_delay( int d );

	/*! \brief Writes current states of SpikingGroup to human-readible textfile if implemented in derived class */
	virtual bool write_to_file(const char * filename);

	/*! \brief Reads current states of SpikingGroup to human-readible textfile if implemented in derived class */
	virtual bool load_from_file(const char * filename);

	/*! \brief Returns size (num of neurons) on the current rank */
	NeuronID ranksize();

	/*! \brief Converts global NeuronID within the SpikingGroup to the local NeuronID on this rank. 
	 *
	 * The return value of global2rank is only defined if the SpikingGroup has units on the present rank.
	 * \see evolve_locally
	 *
	 * \param i The global NeuronID to convert 
	 * \returns The NeuronID of the referenced neuron on the current rankDoes not */
	NeuronID global2rank(NeuronID i); 

	/*! \brief Converts local NeuronID from the local rank to a global NeuronID. 
	 *
	 * \param i The local neuron id.
	 * \returns The global neuron id.
	 * \see global2rank
	 *
	 * This function performs id translation for parallel simulations from a local rank id to a global NeuronID.
	 *
	 * */
	NeuronID rank2global(NeuronID i);

	/*! \brief Checks if the global NeuronID i is integrated on this MPI rank. 
	 *
	 * \param i the NeuronID to check.
	 * \returns true if the neuron with global id i is integrated on the present rank. */
	bool localrank(NeuronID i);

	/*! \brief Rank size but rounded up to multiples of 4 (or potentially some other and larger number in future versions) for SSE compatibility */
	NeuronID get_vector_size();

#ifdef CODE_GATHER_STATS_WAITALL_TIME
	/*! \brief Variables to collect stats if enabled */
	boost::timer waitall_timer;
	static double waitall_time;
	static double waitall_time2;
#endif
};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(SpikingGroup)

// 	extern System * sys;
extern Logger * logger;
#ifdef AURYN_CODE_USE_MPI
 	extern mpi::communicator * mpicommunicator;
#endif // AURYN_CODE_USE_MPI


inline NeuronID SpikingGroup::global2rank(NeuronID i) {
	return i/locked_range;
}

inline NeuronID SpikingGroup::get_rank_size()
{
	return rank_size;
} 

} // closing namespace brackets

#endif /*SPIKINGGROUP_H_*/
