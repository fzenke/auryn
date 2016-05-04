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

#ifndef CONNECTION_H_
#define CONNECTION_H_


#include <string>

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "SpikingGroup.h"
#include "NeuronGroup.h"

namespace auryn {

class System;

/*! \brief The abstract base class for all Connection objects in Auryn
 *
 * Connections are designed to take up spikes from a source a group that
 * can spike in general (decendants of SpikingGroup) and to convey them
 * to a group that can integrate spikes such as the children of 
 * NeuronGroup.
 *
 * For that reason, when constructing any decendant of Connection you 
 * generally will have to specify a source (SpikingGroup) and a 
 * destination (NeuronGroup).
 */
class Connection 
{
private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		virtual_serialize(ar, version);
	}


	NeuronID m_rows,n_cols;
	std::string connection_name;

protected:
	/*! Serialization function for saving the Connection state. Implement in derived classes to save
	 * additional information. 
	 * */
	virtual void virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version ) 
	{
		ar & m_rows & n_cols & connection_name;
	}

	/*! Serialization function for loading the Connection state. Implement in derived classes to save
	 * additional information. 
	 * */
	virtual void virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version ) 
	{
		ar & m_rows & n_cols & connection_name;
	}

	SpikingGroup * src;
	NeuronGroup * dst;
	TransmitterType trans;
	AurynFloat * target;

	/*! \brief Number of spike attributes to expect with each spike transmitted through this connection.
	 *
	 * Should only be set through methods called during init.
	 */
	NeuronID number_of_spike_attributes;

	/*! \brief 	Stores spike attribute offset in attribute array */
	NeuronID spike_attribute_offset;

	void init(TransmitterType transmitter=GLUT);

public:
	Connection();
	Connection(NeuronID rows, NeuronID cols);
	Connection(SpikingGroup * source, NeuronGroup * destination, TransmitterType transmitter=GLUT, std::string name="Connection");
	virtual ~Connection();

	void set_size(NeuronID i, NeuronID j);

	/*! \brief Set name of connection
	 *
	 * The name will appear in error messages and save files */
	void set_name(std::string name);

	/*! \brief Returns name of connection */
	std::string get_name();

	/*! \brief Extracts the class name of the connection from the file name */
	std::string get_file_name();

	/*! \brief Returns a string which is the combination of file and connection name for logging. */
	std::string get_log_name();

	/*! \brief Get number of rows (presynaptic) in connection.
	 *
	 * Note that Matrices in Auryn have to be thought of as multiplied from the left. The number of rows thus
	 * corresponds to the maximum number of presynaptic cells. */
	NeuronID get_m_rows();

	/*! \brief Get number of columns (postsynaptic) in connection.
	 *
	 * Note that Matrices in Auryn have to be thought of as multiplied from the left. The number of columns thus
	 * corresponds to the maximum number of postsynaptic cells. */
	NeuronID get_n_cols();

	/*! \brief Returns transmitter type 
	 *
	 * This is one of Auryn default transmitter types. It essentially characterizes an array of a state vector
	 * to which this Connections weight will be added upon synaptic transmission. */
	TransmitterType get_transmitter();

	/*! \brief Sets target state of this connection directly via a pointer */
	void set_transmitter(AurynWeight * ptr);

	/*! \brief Sets target state of this connection directly via a StateVector */
	void set_transmitter(auryn_vector_float * ptr);

	/*! \brief Sets target state of this connection as one of Auryn's default transmitter types */
	void set_transmitter(TransmitterType transmitter);

	/*! \brief Sets target state of this connection directly the name of a state vector */
	void set_transmitter(string state_name);

	/*! \brief Sets source SpikingGroup of this connection. */
	void set_source(SpikingGroup * source);

	/*! \brief Returns pointer to the presynaptic group. */
	SpikingGroup * get_source();

	/*! \brief Sets destination SpikingGroup of this connection. */
	void set_destination(NeuronGroup * source);

	/*! \brief Returns pointer to the postsynaptic group. */
	NeuronGroup * get_destination();

	/* Purely virtual functions */
	/*! \brief Get weight value i,j if it exists. Otherwise the value is undefined. */
	virtual AurynWeight get(NeuronID i, NeuronID j) = 0;

	/*! \brief Return pointer to weight element i,j if it exists, otherwise return NULL. */
	virtual AurynWeight * get_ptr(NeuronID i, NeuronID j) = 0;

	/*! \brief Return weight element as index in data array */
	virtual AurynWeight get_data(NeuronID i) = 0;

	/*! \brief Set existing weight element i,j with value. */
	virtual void set(NeuronID i, NeuronID j, AurynWeight value) = 0;

	/*! \brief Return number of nonzero elements in this Connection. */
	virtual AurynLong get_nonzero() = 0;

	/*! \brief Finalize Connection after initialization to prepare for use in simulation. */
	virtual void finalize() = 0;

	/*! \brief Propagate method to propagate spikes. Called by System run method. */
	virtual void propagate() = 0;

	/*! \brief Evolve method to update internal connection state. Called by System run method. */
	virtual void evolve();

	/*! \brief DEPRECATED. (Such connections should not be registered in the first place) 
	 * Calls propagate only if the postsynaptic NeuronGroup exists on the local rank. */
	void conditional_propagate();

	/*! \brief Computes the sum of all weights in the connection. */
	virtual AurynDouble sum() = 0;

	/*! \brief Computes mean synaptic weight and std dev of all weights in this connection. */
	virtual void stats(AurynDouble &mean, AurynDouble &std) = 0;

	/*! \brief Implements save to file functionality. Also called in save_network_state from System class. */
	virtual bool write_to_file(std::string filename) = 0;

	/*! \brief Implements load from file functionality. Also called in save_network_state from System class. */
	virtual bool load_from_file(std::string filename) = 0;

	/*! \brief Transmits a spike to a postsynaptic partner
	 *
	 * The method adds a given amount to the respective element in the target/transmitter array of the postsynaptic
	 * neuron specifeid by id. This is a new approach which replaces tadd to old method, increments 
	 * transmitter specific state variables in neuron id. It turned out much faster that way, because the transmit
	 * function is one of the most often called function in the simulation and it can be efficiently inlined by the 
	 * compiler. */
	inline void transmit(NeuronID id, AurynWeight amount);

	/*! \brief Same as transmit but first checks if the target neuron exists and avoids segfaults that way (but it's also slower). */
	void safe_transmit(NeuronID id, AurynWeight amount);

	/*! Returns a vector of ConnectionsID of a block specified by the arguments. */
	virtual std::vector<neuron_pair>  get_block(NeuronID lo_row, NeuronID lo_col, NeuronID hi_row,  NeuronID hi_col) = 0;

	
	/*! \brief Supplies pointer to SpikeContainer of all presynaptic spikes.
	 *
	 * This includes spikes from this group from all other nodes. 
	 * This also means that these spikes have gone through the axonal dealy.
	 * Equivalent to calling src->get_spikes(). */
	SpikeContainer * get_pre_spikes();

	/*! \brief Returns pointer to SpikeContainer for postsynaptic spikes on this node. 
	 *
	 * This corresponds to calling get_spikes_immediate() from the SpikingGroup. 
	 * These spikes have been generated postsynaptically in the same timestep and have not 
	 * been delayed yet. 
	 * Equivalent to calling dst->get_spikes_immediate(). */
	SpikeContainer * get_post_spikes();

	/*! \brief Set up spike delay to accomodate x additional spike attributes.
	 *
	 * Spike attribute numbers can only be increased. 
	 * This function can only be run during initialization.
	 */
	void add_number_of_spike_attributes(int x);

	/*! \brief Returns spike attribute belonging to the spike at position i in the get_spikes() SpikeContainer. */
	AurynFloat get_spike_attribute(const NeuronID i, const int attribute_id=0);
};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(Connection)

inline void Connection::transmit(NeuronID id, AurynWeight amount) 
{
	NeuronID localid = dst->global2rank(id);
	target[localid]+=amount;
}
}

#endif /*CONNECTION_H_*/
